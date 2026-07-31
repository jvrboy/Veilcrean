#!/usr/bin/env python3
"""Phase 1: Q-Learning Training - 5 iterations over 8 instruments."""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from training.signal_generator import generate_signal
from training.learning_engine import LearningEngine
from training.trade_simulator import simulate_trade

DATA = REPO / "data" / "historical_deriv"
OUT = REPO / "training" / "output"

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

def train(sym,cs,cfg,le,msig=200,siv=3,mhold=20):
    ps=cfg["pip"];fd=cfg.get("fd");lb=100
    if len(cs)<lb+mhold+50: return {"skip":True}
    R={"sym":sym,"sigs":0,"w":0,"l":0,"be":0,"pnl":0.0,"tp":0.0,"sl":0.0}
    sc=0;bs=siv+1;act=None;t0=time.time()
    for i in range(lb,len(cs)-mhold-1):
        if (i-lb)%2000==0 and (i-lb)>0: print(f"    bar {i-lb}/{len(cs)-lb-mhold} sigs={sc}", flush=True)
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

def run_iter(insts,le,it,ms=200):
    print(f"\n{'='*50}\nITERATION {it}\n{'='*50}", flush=True)
    ir={};ts=tw=tl=0
    for sym,cfg in insts.items():
        fp=DATA/cfg["csv"]
        if not fp.exists(): print(f"  {cfg['n']}: NO DATA",flush=True);continue
        cs=load_csv(str(fp))
        print(f"  {cfg['n']}: {len(cs)} bars...",flush=True)
        r=train(sym,cs,cfg,le,msig=ms)
        if r.get("skip"): print(f"    SKIP",flush=True);continue
        ir[sym]=r;ts+=r["sigs"];tw+=r["w"];tl+=r["l"]
        print(f"    {r['sigs']}sigs WR:{r['wr']:.1f}% PF:{r['pf']:.2f} Avg:{r['avg']:.1f}p ({r['time']}s)",flush=True)
    owr=tw/max(ts,1)*100
    print(f"  >> ITER {it}: {ts} sigs, {owr:.1f}% WR, {len(le.patterns)} patterns",flush=True)
    return {"it":it,"sigs":ts,"w":tw,"l":tl,"wr":owr,"inst":ir}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    le=LearningEngine()
    bwr=0;bit=0;iters=[]
    for it in range(1,6):
        r=run_iter(INSTS,le,it,ms=200)
        iters.append(r)
        with open(OUT/f"v3_iter_{it}.json","w") as f: json.dump({k:v for k,v in r.items()},f,indent=2,default=str)
        if r["wr"]>bwr:
            bwr=r["wr"];bit=it
            with open(OUT/"v3_learned_state_best.json","w") as f: json.dump(le.to_dict(),f,indent=2,default=str)
            print(f"  *** NEW BEST: {bwr:.1f}% ***",flush=True)
        le.q_epsilon=max(0.05,le.q_epsilon*0.8)
    with open(OUT/"v3_summary.json","w") as f:
        json.dump({"best_wr":bwr,"best_it":bit,"iters":len(iters),"wr_hist":[x["wr"] for x in iters],"patterns":len(le.patterns)},f,indent=2)
    print(f"\nPHASE 1 DONE: WR={bwr:.1f}% (iter{bit}), {len(le.patterns)} patterns",flush=True)

if __name__=="__main__": main()
