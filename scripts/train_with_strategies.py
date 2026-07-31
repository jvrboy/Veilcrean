#!/usr/bin/env python3
"""Train each instrument with its 10 strategies, save best results."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from training.signal_generator import generate_signal
from training.learning_engine import LearningEngine, THRESHOLD_LEVELS
from training.trade_simulator import simulate_trade

DATA = REPO / "data" / "historical_deriv"
OUT = REPO / "training" / "output"
STRAT_DIR = REPO / "training" / "strategies"

INSTS = {
    "1HZ50V":  {"n":"Volatility_50",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ50V_1h.csv"},
    "1HZ75V":  {"n":"Volatility_75",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ75V_1h.csv"},
    "1HZ100V": {"n":"Volatility_100", "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ100V_1h.csv"},
    "BOOM500":  {"n":"Boom_500",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM500_1h.csv"},
    "BOOM900":  {"n":"Boom_900",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM900_8h.csv"},
    "BOOM1000": {"n":"Boom_1000", "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM1000_1h.csv"},
    "CRASH500": {"n":"Crash_500", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH500_8h.csv"},
    "CRASH900": {"n":"Crash_900", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH900_1h.csv"},
}

def load_csv(p):
    cs=[]
    with open(p) as f:
        f.readline()
        for ln in f:
            q=ln.strip().split(",")
            if len(q)>=5: cs.append({"epoch":int(q[0]),"open":float(q[2]),"high":float(q[3]),"low":float(q[4]),"close":float(q[5]) if len(q)>5 else float(q[4]),"volume":0.0})
    cs.sort(key=lambda c:c["epoch"]); return cs

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

def train_with_strategy(sym, cs, cfg, le, strat, msig=80):
    ps=cfg["pip"];fd=strat.get("force_direction") or cfg.get("fd")
    regime_filter=strat.get("regime_filter",[])
    min_conf=strat.get("min_confidence",0.5)
    tp_m=strat.get("tp_mult",1.0);sl_m=strat.get("sl_mult",1.0)
    lb=100;mhold=20
    R={"strat_id":strat["id"],"sigs":0,"w":0,"l":0,"be":0,"pnl":0.0,"tp":0.0,"sl":0.0}
    sc=0;bs=4;act=None
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
        if act is not None or bs<3 or sc>=msig: continue
        try:
            rg=regim(win)
            if regime_filter and rg not in regime_filter: continue
            ap=atrp(win,ps)
            tp2,sl2,tr2,be2=le.get_optimal_tp_sl(rg,ap,sym,"1h")
            tp2*=tp_m;sl2*=sl_m
            sig=generate_signal(candles=win,pip_size=ps,tp_override=tp2,sl_override=sl2)
            if sig.direction=="HOLD": continue
            if fd and sig.direction!=fd: sig.direction=fd;sig.confidence*=0.8
            if sig.confidence<min_conf: continue
            ok,_=le.should_take_signal(sig.direction,sig.confidence,rg,sig.tool_scores,sym,"1h",win[-1]["epoch"])
            if not ok: continue
            act={"dir":sig.direction,"epx":win[-1]["close"],"tp":sig.recommended_tp,"sl":sig.recommended_sl,"ts":sig.tool_scores,"reg":rg,"conf":sig.confidence,"ep":win[-1]["epoch"],"rem":mhold,"tr":tr2,"be":be2}
            bs=0;sc+=1
        except: continue
    t=R["w"]+R["l"]+R["be"]
    R["wr"]=R["w"]/max(t,1)*100;R["pf"]=R["tp"]/max(R["sl"],1e-10);R["avg"]=R["pnl"]/max(t,1)
    return R

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sym = sys.argv[1] if len(sys.argv) > 1 else "1HZ75V"

    cfg = INSTS[sym]
    strat_file = STRAT_DIR / f"{sym}_strategies.json"
    if not strat_file.exists():
        print(f"No strategies for {sym}"); return

    with open(strat_file) as f: strategies = json.load(f)
    fp = DATA / cfg["csv"]
    if not fp.exists(): print(f"No data for {sym}"); return
    cs = load_csv(str(fp))
    print(f"Training {cfg['n']} ({sym}): {len(cs)} bars, {len(strategies)} strategies", flush=True)

    results = []
    for strat in strategies:
        le = LearningEngine()  # Fresh engine per strategy for clean evaluation
        r = train_with_strategy(sym, cs, cfg, le, strat, msig=80)
        results.append(r)
        wr_str = f'{r["wr"]:.1f}%' if r["sigs"]>0 else 'N/A'
        pf_str = f'{r["pf"]:.2f}' if r["sigs"]>0 else 'N/A'
        print(f'  {strat["id"][:35]:35s} {r["sigs"]:3d}sigs WR:{wr_str:>6s} PF:{pf_str:>6s}', flush=True)

    # Sort by WR and save
    results.sort(key=lambda x: x["wr"], reverse=True)
    with open(OUT / f"v3_strat_results_{sym}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    best = results[0] if results else {"wr":0}
    print(f'  Best: {best.get("strat_id","?")} WR={best.get("wr",0):.1f}%', flush=True)

if __name__ == "__main__": main()