#!/usr/bin/env python3
"""Veilcrean Comprehensive Trainer v3 - Optimized"""
from __future__ import annotations
import sys, os, json, time, math, asyncio
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.signal_generator import generate_signal
from training.learning_engine import LearningEngine
from training.trade_simulator import simulate_trade

DATA_DIR = REPO_ROOT / "data" / "historical_deriv"
OUTPUT_DIR = REPO_ROOT / "training" / "output"
STRATEGIES_DIR = REPO_ROOT / "training" / "strategies"

# Only instruments with LOCAL CSV data - no API fetching that could hang
INSTRUMENTS = {
    "1HZ50V":  {"name":"Volatility 50",  "pip":0.01, "force":None, "csv":"01_SYNTHETICS_VOLATILITY/1HZ50V_1h.csv"},
    "1HZ75V":  {"name":"Volatility 75",  "pip":0.01, "force":None, "csv":"01_SYNTHETICS_VOLATILITY/1HZ75V_1h.csv"},
    "1HZ100V": {"name":"Volatility 100", "pip":0.01, "force":None, "csv":"01_SYNTHETICS_VOLATILITY/1HZ100V_1h.csv"},
    "BOOM500":  {"name":"Boom 500",  "pip":0.01, "force":"BUY",  "csv":"02_SYNTHETICS_BOOM/BOOM500_1h.csv"},
    "BOOM900":  {"name":"Boom 900",  "pip":0.01, "force":"BUY",  "csv":"02_SYNTHETICS_BOOM/BOOM900_8h.csv"},
    "BOOM1000": {"name":"Boom 1000", "pip":0.01, "force":"BUY",  "csv":"02_SYNTHETICS_BOOM/BOOM1000_1h.csv"},
    "CRASH500": {"name":"Crash 500", "pip":0.01, "force":"SELL", "csv":"03_SYNTHETICS_CRASH/CRASH500_8h.csv"},
    "CRASH900": {"name":"Crash 900", "pip":0.01, "force":"SELL", "csv":"03_SYNTHETICS_CRASH/CRASH900_1h.csv"},
}

def load_csv(path):
    cs = []
    with open(path) as f:
        f.readline()
        for ln in f:
            p = ln.strip().split(",")
            if len(p) >= 5:
                cs.append({"epoch":int(p[0]),"open":float(p[2]),"high":float(p[3]),"low":float(p[4]),"close":float(p[5]) if len(p)>5 else float(p[4]),"volume":0.0})
    cs.sort(key=lambda c:c["epoch"])
    return cs

def detect_regime(cands):
    n = min(30, len(cands))
    c = np.array([x["close"] for x in cands[-n:]])
    h = np.array([x["high"] for x in cands[-n:]])
    l = np.array([x["low"] for x in cands[-n:]])
    trs = np.maximum(h-l, np.abs(h-np.roll(c,1)))
    trs = np.maximum(trs, np.abs(l-np.roll(c,1)))
    atr = np.mean(trs[1:])
    ap = atr/c[-1] if c[-1]>0 else 0
    d = np.diff(c)
    u, dn = np.where(d>0,d,0).mean(), np.where(d<0,-d,0).mean()
    dx = 100*abs(u-dn)/max(u+dn,1e-10)
    pr = h.max()-l.min()
    st = np.sum(trs[1:])
    chop = 100*np.log10(st/max(pr,1e-10))/np.log10(n) if pr>0 else 50
    if dx>25 and ap>0.005: return "BREAKOUT"
    if dx>25: return "TRENDING"
    if ap>0.01: return "VOLATILE"
    if chop>61.8: return "CHOPPY"
    return "RANGING"

def atr_pips(cands, ps):
    n = min(15, len(cands))
    if n<3: return 50.0
    r = cands[-n:]
    trs = [max(r[i]["high"]-r[i]["low"], abs(r[i]["high"]-r[i-1]["close"]), abs(r[i]["low"]-r[i-1]["close"])) for i in range(1,n)]
    return np.mean(trs)/ps if ps>0 else np.mean(trs)

def train_inst(sym, cs, cfg, le, max_sig=300, sig_iv=3, max_hold=20):
    ps = cfg["pip"]; fd = cfg.get("force"); lb = 100
    if len(cs)<lb+max_hold+50: return {"skipped":True}
    R = {"sym":sym,"sigs":0,"w":0,"l":0,"be":0,"pnl":0.0,"tp":0.0,"sl":0.0,"trades":[]}
    sc=0; bss=sig_iv+1; act=None

    for i in range(lb, len(cs)-max_hold-1):
        win = cs[max(0,i-lb):i+1]
        fut = cs[i+1:i+1+max_hold]
        if not fut: continue

        # Resolve active trade
        if act is not None:
            ffs = [c for c in fut if c["epoch"]>act["ep"]][:max_hold]
            if len(ffs)>=act["rem"] or i>=len(cs)-max_hold-2:
                out = simulate_trade(act["dir"],act["epx"],act["tp"],act["sl"],ffs,ps,max_hold,act["tr"],0.5,act["be"])
                le.learn_from_failure(act["ts"],act["reg"],out.failure_category or "",out.pnl_pips,act["ep"],act["tp"],act["sl"],sym,act["tf"],act["dir"])
                R["sigs"]+=1
                if out.outcome=="win": R["w"]+=1
                elif out.outcome=="loss": R["l"]+=1
                else: R["be"]+=1
                R["pnl"]+=out.pnl_pips
                if out.pnl_pips>0: R["tp"]+=out.pnl_pips
                else: R["sl"]+=abs(out.pnl_pips)
                R["trades"].append({"d":act["dir"],"c":round(act["conf"],3),"r":act["reg"],"o":out.outcome,"p":round(out.pnl_pips,2)})
                act = None

        bss+=1
        if act is not None or bss<sig_iv or sc>=max_sig: continue

        try:
            reg = detect_regime(win)
            ap = atr_pips(win, ps)
            tp, sl, tr, be = le.get_optimal_tp_sl(reg, ap, sym, "1h")
            sig = generate_signal(candles=win, pip_size=ps, tp_override=tp, sl_override=sl)
            if sig.direction=="HOLD": continue
            if fd and sig.direction!=fd:
                sig.direction=fd; sig.confidence*=0.8
            ok, _ = le.should_take_signal(sig.direction, sig.confidence, reg, sig.tool_scores, sym, "1h", win[-1]["epoch"])
            if not ok: continue
            act = {"dir":sig.direction,"epx":win[-1]["close"],"tp":sig.recommended_tp,"sl":sig.recommended_sl,
                   "ts":sig.tool_scores,"reg":reg,"conf":sig.confidence,"ep":win[-1]["epoch"],
                   "tf":"1h","rem":max_hold,"tr":tr,"be":be}
            bss=0; sc+=1
        except Exception as e:
            if sc<3: print(f"    err {i}: {e}")
            continue

    t = R["w"]+R["l"]+R["be"]
    R["wr"]=R["w"]/max(t,1)*100; R["pf"]=R["tp"]/max(R["sl"],1e-10); R["avg"]=R["pnl"]/max(t,1)
    return R

def run_iter(insts, le, it, ms=300):
    print(f"\n{'='*50}\n  ITERATION {it}\n{'='*50}")
    ir={}; ts=tw=tl=0
    for sym,cfg in insts.items():
        print(f"\n  {cfg['name']} ({sym})...", end="", flush=True)
        fp = DATA_DIR/cfg["csv"]
        if not fp.exists(): print(" NO DATA"); continue
        cs = load_csv(str(fp))
        r = train_inst(sym, cs, cfg, le, max_sig=ms)
        if r.get("skipped"): print(" SKIP"); continue
        ir[sym]=r; ts+=r["sigs"]; tw+=r["w"]; tl+=r["l"]
        print(f" {r['sigs']}sigs WR:{r['wr']:.1f}% PF:{r['pf']:.2f} Avg:{r['avg']:.1f}p")
    owr = tw/max(ts,1)*100
    print(f"\n  >> ITER {it}: {ts} sigs, {owr:.1f}% WR, {len(le.patterns)} patterns")
    return {"iteration":it,"sigs":ts,"wins":tw,"losses":tl,"wr":owr,"instruments":ir}

def main():
    print("="*50+"\n  VEILCREAN TRAINER v3 - REAL DATA\n"+"="*50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    ex = OUTPUT_DIR/"v3_learned_state_best.json"
    if ex.exists():
        with open(ex) as f: le=LearningEngine.from_dict(json.load(f))
        print(f"Loaded: {len(le.patterns)} patterns")
    else: le=LearningEngine(); print("Fresh engine")

    best_wr=0; best_it=0; iters=[]
    for it in range(1,6):
        r=run_iter(INSTRUMENTS, le, it, ms=300)
        iters.append(r)
        with open(OUTPUT_DIR/f"v3_iter_{it}.json","w") as f: json.dump({k:v for k,v in r.items()},f,indent=2,default=str)
        if r["wr"]>best_wr:
            best_wr=r["wr"]; best_it=it
            with open(OUTPUT_DIR/"v3_learned_state_best.json","w") as f: json.dump(le.to_dict(),f,indent=2,default=str)
            print(f"  *** NEW BEST: {best_wr:.1f}% ***")
        else:
            with open(OUTPUT_DIR/f"v3_state_iter_{it}.json","w") as f: json.dump(le.to_dict(),f,indent=2,default=str)
        le.q_epsilon = max(0.05, le.q_epsilon*0.85)

    with open(OUTPUT_DIR/"v3_summary.json","w") as f:
        json.dump({"best_wr":best_wr,"best_it":best_it,"iters":len(iters),"wr_per_iter":[x["wr"] for x in iters],"patterns":len(le.patterns),"instruments":list(INSTRUMENTS.keys())},f,indent=2)
    print(f"\n{'='*50}\n  PHASE 1 DONE: WR={best_wr:.1f}% (iter{best_it}), {len(le.patterns)} patterns\n{'='*50}")

    # Reload best state for strategy creation
    with open(OUTPUT_DIR/"v3_learned_state_best.json") as f: le=LearningEngine.from_dict(json.load(f))
    return le, INSTRUMENTS, iters

if __name__=="__main__": main()