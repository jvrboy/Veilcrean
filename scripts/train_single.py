#!/usr/bin/env python3
"""Train a single instrument for N iterations, save state."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from training.signal_generator import generate_signal
from training.learning_engine import LearningEngine
from training.trade_simulator import simulate_trade

DATA = REPO / "data" / "historical_deriv"
OUT = REPO / "training" / "output"

def load_csv(p):
    cs=[]
    with open(p) as f:
        f.readline()
        for ln in f:
            q=ln.strip().split(",")
            if len(q)>=5: cs.append({"epoch":int(q[0]),"open":float(q[2]),"high":float(q[3]),"low":float(q[4]),"close":float(q[5]) if len(q)>5 else float(q[4]),"volume":0.0})
    cs.sort(key=lambda c:c["epoch"])
    return cs

def regim(cands):
    n=min(30,len(cands))
    if n<5: return "UNKNOWN"
    c=np.array([x["close"] for x in cands[-n:]])
    h=np.array([x["high"] for x in cands[-n:]])
    l=np.array([x["low"] for x in cands[-n:]])
    trs=np.maximum(h-l,np.abs(h-np.roll(c,1)));trs=np.maximum(trs,np.abs(l-np.roll(c,1)))
    atr=np.nanmean(trs[1:]);ap=atr/c[-1] if c[-1]>0 else 0
    d=np.diff(c);u,dn=np.where(d>0,d,0).mean(),np.where(d<0,-d,0).mean()
    dx=100*abs(u-dn)/max(u+dn,1e-10)
    pr=h.max()-l.min();st=np.nansum(trs[1:])
    chop=100*np.log10(st/max(pr,1e-10))/np.log10(n) if pr>0 else 50
    if dx>25 and ap>0.005: return "BREAKOUT"
    if dx>25: return "TRENDING"
    if ap>0.01: return "VOLATILE"
    if chop>61.8: return "CHOPPY"
    return "RANGING"

def atrp(cands,ps):
    n=min(15,len(cands))
    if n<3: return 50.0
    r=cands[-n:]
    trs=[max(r[i]["high"]-r[i]["low"],abs(r[i]["high"]-r[i-1]["close"]),abs(r[i]["low"]-r[i-1]["close"])) for i in range(1,n)]
    return np.mean(trs)/ps if ps>0 else np.mean(trs)

def train(sym, cs, cfg, le, msig=150, siv=3, mhold=20):
    ps=cfg["pip"];fd=cfg.get("fd");lb=100
    if len(cs)<lb+mhold+50: return {"skip":True}
    R={"sym":sym,"sigs":0,"w":0,"l":0,"be":0,"pnl":0.0,"tp":0.0,"sl":0.0}
    sc=0;bs=siv+1;act=None;t0=time.time()
    for i in range(lb,len(cs)-mhold-1):
        win=cs[max(0,i-lb):i+1];fut=cs[i+1:i+1+mhold]
        if not fut: continue
        if act is not None:
            ffs=[c for c in fut if c["epoch"]>act["ep"]][:mhold]
            if len(ffs)>=act["rem"] or i>=len(cs)-mhold-2:
                out=simulate_trade(act["dir"],act["epx"],act["tp"],act["sl"],ffs,ps,mhold,act["tr"],0.5,act["be"])
                le.learn_from_failure(act["ts"],act["reg"],out.failure_category or "",out.pnl_pips,act["ep"],act["tp"],act["sl"],sym,"1h",act["dir"])
                R["sigs"]+=1
                if out.outcome=="win": R["w"]+=1
                elif out.outcome=="loss": R["l"]+=1
                else: R["be"]+=1
                R["pnl"]+=out.pnl_pips
                if out.pnl_pips>0: R["tp"]+=out.pnl_pips
                else: R["sl"]+=abs(out.pnl_pips)
                act=None
        bs+=1
        if act is not None or bs<siv or sc>=msig: continue
        try:
            rg=regim(win);ap=atrp(win,ps)
            tp2,sl2,tr2,be2=le.get_optimal_tp_sl(rg,ap,sym,"1h")
            sig=generate_signal(candles=win,pip_size=ps,tp_override=tp2,sl_override=sl2)
            if sig.direction=="HOLD": continue
            if fd and sig.direction!=fd: sig.direction=fd;sig.confidence*=0.8
            ok,_=le.should_take_signal(sig.direction,sig.confidence,rg,sig.tool_scores,sym,"1h",win[-1]["epoch"])
            if not ok: continue
            act={"dir":sig.direction,"epx":win[-1]["close"],"tp":sig.recommended_tp,"sl":sig.recommended_sl,"ts":sig.tool_scores,"reg":rg,"conf":sig.confidence,"ep":win[-1]["epoch"],"rem":mhold,"tr":tr2,"be":be2}
            bs=0;sc+=1
        except: continue
    t=R["w"]+R["l"]+R["be"]
    R["wr"]=R["w"]/max(t,1)*100;R["pf"]=R["tp"]/max(R["sl"],1e-10);R["avg"]=R["pnl"]/max(t,1)
    R["time"]=round(time.time()-t0,1)
    return R

# All instrument configs
ALL_INSTS = {
    "1HZ50V":  {"n":"Volatility_50",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ50V_1h.csv","type":"volatility"},
    "1HZ75V":  {"n":"Volatility_75",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ75V_1h.csv","type":"volatility"},
    "1HZ100V": {"n":"Volatility_100", "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ100V_1h.csv","type":"volatility"},
    "BOOM500":  {"n":"Boom_500",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM500_1h.csv","type":"boom"},
    "BOOM900":  {"n":"Boom_900",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM900_8h.csv","type":"boom"},
    "BOOM1000": {"n":"Boom_1000", "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM1000_1h.csv","type":"boom"},
    "CRASH500": {"n":"Crash_500", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH500_8h.csv","type":"crash"},
    "CRASH900": {"n":"Crash_900", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH900_1h.csv","type":"crash"},
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sym = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    n_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    msig = int(sys.argv[3]) if len(sys.argv) > 3 else 150

    # Load shared learning state
    state_file = OUT / "v3_shared_state.json"
    if state_file.exists():
        with open(state_file) as f: le = LearningEngine.from_dict(json.load(f))
        print(f"Loaded shared state: {len(le.patterns)} patterns", flush=True)
    else:
        le = LearningEngine()
        print("Fresh learning engine", flush=True)

    insts = {sym: ALL_INSTS[sym]} if sym != "ALL" else ALL_INSTS
    all_results = {}
    bwr = 0

    for it in range(1, n_iters + 1):
        print(f"\n--- Iteration {it} ---", flush=True)
        # Between iterations: keep Q-tables and patterns (learned knowledge),
        # but reset signal stats, session stats, and indicator performance
        # to avoid over-filtering in fresh walk-forward passes
        if it > 1:
            saved_q = dict(le.q_table)
            saved_q_updates = dict(le.q_updates)
            saved_patterns = dict(le.patterns)
            saved_tp_adj = dict(le.tp_adjustments)
            saved_epsilon = le.q_epsilon
            from training.learning_engine import THRESHOLD_LEVELS
            le = LearningEngine()
            le.q_table = defaultdict(lambda: {t: 0.0 for t in THRESHOLD_LEVELS}, saved_q)
            le.q_updates = defaultdict(int, saved_q_updates)
            le.patterns = saved_patterns
            le.tp_adjustments = saved_tp_adj
            le.q_epsilon = saved_epsilon
        for s, cfg in insts.items():
            fp = DATA / cfg["csv"]
            if not fp.exists(): print(f"  {cfg['n']}: NO DATA", flush=True); continue
            cs = load_csv(str(fp))
            r = train(s, cs, cfg, le, msig=msig)
            if r.get("skip"): continue
            key = f"{s}_it{it}"
            all_results[key] = {k: v for k, v in r.items()}
            print(f"  {cfg['n']}: {r['sigs']}sigs WR:{r['wr']:.1f}% PF:{r['pf']:.2f} ({r['time']}s)", flush=True)
        le.q_epsilon = max(0.05, le.q_epsilon * 0.8)

    # Save shared state for next run
    with open(state_file, "w") as f:
        json.dump(le.to_dict(), f, indent=2, default=str)

    # Also save as best
    with open(OUT / "v3_learned_state_best.json", "w") as f:
        json.dump(le.to_dict(), f, indent=2, default=str)

    # Save results — maintain a valid JSON document (list of run snapshots).
    # Older versions appended raw JSON lines, so tolerate JSONL input too.
    results_path = OUT / "v3_all_results.json"
    history: list = []
    if results_path.exists():
        raw = results_path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                history = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                # Legacy JSON-lines file: one JSON object per line.
                history = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    history.append(all_results)
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    total_sigs = sum(r.get("sigs", 0) for r in all_results.values())
    total_w = sum(r.get("w", 0) for r in all_results.values())
    print(f"\nDone: {total_sigs} sigs, {total_w/max(total_sigs,1)*100:.1f}% WR, {len(le.patterns)} patterns", flush=True)

if __name__ == "__main__": main()