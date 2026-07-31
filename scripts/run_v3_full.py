#!/usr/bin/env python3
"""Veilcrean v3 Full Trainer - runs training, creates strategies, pushes."""
from __future__ import annotations
import sys, os, json, time, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from training.signal_generator import generate_signal
from training.learning_engine import LearningEngine
from training.trade_simulator import simulate_trade

DATA = REPO / "data" / "historical_deriv"
OUT = REPO / "training" / "output"
STRAT_DIR = REPO / "training" / "strategies"

INSTS = {
    "1HZ50V":  {"n":"Volatility_50",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ50V_1h.csv","type":"volatility"},
    "1HZ75V":  {"n":"Volatility_75",  "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ75V_1h.csv","type":"volatility"},
    "1HZ100V": {"n":"Volatility_100", "pip":0.01,"fd":None,"csv":"01_SYNTHETICS_VOLATILITY/1HZ100V_1h.csv","type":"volatility"},
    "BOOM500":  {"n":"Boom_500",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM500_1h.csv","type":"boom"},
    "BOOM900":  {"n":"Boom_900",  "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM900_8h.csv","type":"boom"},
    "BOOM1000": {"n":"Boom_1000", "pip":0.01,"fd":"BUY", "csv":"02_SYNTHETICS_BOOM/BOOM1000_1h.csv","type":"boom"},
    "CRASH500": {"n":"Crash_500", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH500_8h.csv","type":"crash"},
    "CRASH900": {"n":"Crash_900", "pip":0.01,"fd":"SELL","csv":"03_SYNTHETICS_CRASH/CRASH900_1h.csv","type":"crash"},
}

def P(msg): print(msg, flush=True)

def load_csv(p):
    cs=[]
    with open(p) as f:
        f.readline()
        for ln in f:
            q=ln.strip().split(",")
            if len(q)>=5:
                cs.append({"epoch":int(q[0]),"open":float(q[2]),"high":float(q[3]),"low":float(q[4]),"close":float(q[5]) if len(q)>5 else float(q[4]),"volume":0.0})
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
        if (i-lb)%2000==0 and (i-lb)>0:
            P(f"    bar {i-lb}/{len(cs)-lb-mhold} sigs={sc}")
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
    P(f"\n{'='*50}\nITERATION {it}\n{'='*50}")
    ir={};ts=tw=tl=0
    for sym,cfg in insts.items():
        fp=DATA/cfg["csv"]
        if not fp.exists(): P(f"  {cfg['n']}: NO DATA");continue
        cs=load_csv(str(fp))
        P(f"  {cfg['n']}: {len(cs)} bars...")
        r=train(sym,cs,cfg,le,msig=ms)
        if r.get("skip"): P(f"    SKIP");continue
        ir[sym]=r;ts+=r["sigs"];tw+=r["w"];tl+=r["l"]
        P(f"    {r['sigs']}sigs WR:{r['wr']:.1f}% PF:{r['pf']:.2f} Avg:{r['avg']:.1f}p ({r['time']}s)")
    owr=tw/max(ts,1)*100
    P(f"  >> ITER {it}: {ts} sigs, {owr:.1f}% WR, {len(le.patterns)} patterns")
    return {"it":it,"sigs":ts,"w":tw,"l":tl,"wr":owr,"inst":ir}

def phase1_training():
    P("\n"+"#"*60)
    P("  PHASE 1: Q-LEARNING TRAINING (5 iterations)")
    P("#"*60)
    OUT.mkdir(parents=True, exist_ok=True)

    # Fresh start
    le=LearningEngine()
    bwr=0;bit=0;iters=[]

    for it in range(1,6):
        r=run_iter(INSTS,le,it,ms=200)
        iters.append(r)
        sr={k:v for k,v in r.items()}
        with open(OUT/f"v3_iter_{it}.json","w") as f: json.dump(sr,f,indent=2,default=str)
        if r["wr"]>bwr:
            bwr=r["wr"];bit=it
            with open(OUT/"v3_learned_state_best.json","w") as f: json.dump(le.to_dict(),f,indent=2,default=str)
            P(f"  *** NEW BEST: {bwr:.1f}% ***")
        le.q_epsilon=max(0.05,le.q_epsilon*0.8)

    with open(OUT/"v3_summary.json","w") as f:
        json.dump({"best_wr":bwr,"best_it":bit,"iters":len(iters),"wr_hist":[x["wr"] for x in iters],"patterns":len(le.patterns)},f,indent=2)
    P(f"\n  PHASE 1 RESULT: WR={bwr:.1f}% (iter{bit}), {len(le.patterns)} patterns")
    return le


def phase2_strategies(le):
    P("\n"+"#"*60)
    P("  PHASE 2: CREATE 10 STRATEGIES PER INSTRUMENT")
    P("#"*60)
    STRAT_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy templates with instrument-specific parameters
    # Each strategy defines: indicator emphasis, regime filter, TP/SL ratios, signal filters
    BASE_STRATEGIES = [
        {"name":"rsi_reversal","desc":"RSI extreme reversal with Bollinger confirmation","indicators":{"rsi":1.5,"bollinger":1.3,"stochastic":0.8},"regime_filter":["RANGING","CHOPPY"],"tp_mult":1.0,"sl_mult":0.6,"min_confidence":0.55},
        {"name":"trend_follow_macd","desc":"MACD crossover trend following with ADX confirmation","indicators":{"macd":1.5,"dmi_direction":1.3,"hull_ma":1.2,"adx":0.8},"regime_filter":["TRENDING"],"tp_mult":2.5,"sl_mult":1.0,"min_confidence":0.45},
        {"name":"breakout_momentum","desc":"Breakout detection with momentum and volume surge","indicators":{"atr_channel":1.5,"momentum":1.4,"roc":1.2,"ttm_squeeze":1.0},"regime_filter":["BREAKOUT","VOLATILE"],"tp_mult":2.0,"sl_mult":0.8,"min_confidence":0.50},
        {"name":"mean_reversion_bb","desc":"Bollinger Band mean reversion with RSI filter","indicators":{"bollinger":1.8,"rsi":1.2,"williams_r":1.0,"keltner":0.5},"regime_filter":["RANGING"],"tp_mult":0.8,"sl_mult":0.5,"min_confidence":0.50},
        {"name":"ichimoku_cloud","desc":"Ichimoku Cloud trend with price action confirmation","indicators":{"ichimoku":1.8,"multi_tf_trend":1.3,"hull_ma":0.8},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":2.0,"sl_mult":0.8,"min_confidence":0.45},
        {"name":"stochastic_divergence","desc":"Stochastic RSI divergence with supply/demand zones","indicators":{"stochastic_rsi":1.8,"rsi_divergence":1.5,"supply_demand":1.0},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":1.2,"sl_mult":0.7,"min_confidence":0.55},
        {"name":"volatility_squeeze","desc":"TTM Squeeze expansion with ATR breakout","indicators":{"ttm_squeeze":1.8,"atr_channel":1.5,"bollinger_width":1.2,"chop_index":0.8},"regime_filter":["BREAKOUT","TRENDING"],"tp_mult":2.5,"sl_mult":0.8,"min_confidence":0.50},
        {"name":"multi_tf_alignment","desc":"Multi-timeframe trend alignment with momentum","indicators":{"multi_tf_trend":1.8,"hull_ma":1.3,"macd":1.0,"dmi_direction":0.8},"regime_filter":["TRENDING"],"tp_mult":2.0,"sl_mult":0.7,"min_confidence":0.50},
        {"name":"supply_demand_bounce","desc":"Supply/demand zone bounce with volume confirmation","indicators":{"supply_demand":1.8,"volume_conf":1.3,"bollinger":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":1.0,"sl_mult":0.6,"min_confidence":0.55},
        {"name":"adaptive_momentum","desc":"Adaptive momentum using ROC, CCI and awesome oscillator","indicators":{"roc":1.5,"cci":1.3,"awesome_osc":1.2,"momentum":1.0},"regime_filter":["TRENDING","VOLATILE","BREAKOUT"],"tp_mult":1.5,"sl_mult":0.8,"min_confidence":0.45},
    ]

    all_strategies = {}
    for sym, cfg in INSTS.items():
        inst_type = cfg.get("type", "volatility")
        P(f"\n  Creating strategies for {cfg['n']} ({sym})...")
        strategies = []
        for i, base in enumerate(BASE_STRATEGIES):
            # Tailor strategy to instrument type
            s = dict(base)
            if inst_type == "boom":
                # Boom: force BUY, wider TP for spikes
                s["force_direction"] = "BUY"
                s["tp_mult"] *= 1.5
                s["sl_mult"] *= 1.2
                s["name"] = f"{s['name']}_boom_adapted"
            elif inst_type == "crash":
                # Crash: force SELL, wider TP for drops
                s["force_direction"] = "SELL"
                s["tp_mult"] *= 1.5
                s["sl_mult"] *= 1.2
                s["name"] = f"{s['name']}_crash_adapted"
            elif inst_type == "volatility":
                # Volatility: moderate TP, tighter SL
                s["tp_mult"] *= 0.9
                s["sl_mult"] *= 0.9
                s["name"] = f"{s['name']}_vol_adapted"

            s["instrument"] = sym
            s["instrument_name"] = cfg["n"]
            s["instrument_type"] = inst_type
            s["pip_size"] = cfg["pip"]
            s["id"] = f"{sym}_{s['name']}"
            strategies.append(s)

        # Save per-instrument strategy file
        strat_path = STRAT_DIR / f"{sym}_strategies.json"
        with open(strat_path, "w") as f:
            json.dump(strategies, f, indent=2)
        all_strategies[sym] = strategies
        P(f"    Saved {len(strategies)} strategies to {strat_path.name}")

    # Save all strategies index
    index_path = STRAT_DIR / "all_strategies_index.json"
    index = {sym: [s["id"] for s in strats] for sym, strats in all_strategies.items()}
    index["total"] = sum(len(v) for v in index.values())
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    P(f"\n  TOTAL: {index['total']} strategies across {len(all_strategies)} instruments")
    return all_strategies


def phase3_train_with_strategies(le, all_strategies):
    P("\n"+"#"*60)
    P("  PHASE 3: RETRAIN WITH STRATEGIES")
    P("#"*60)

    # Reload best learning state
    best = OUT / "v3_learned_state_best.json"
    if best.exists():
        with open(best) as f: le = LearningEngine.from_dict(json.load(f))
        P(f"  Loaded best state: {len(le.patterns)} patterns")

    strategy_results = {}
    for sym, cfg in INSTS.items():
        fp = DATA / cfg["csv"]
        if not fp.exists(): continue
        cs = load_csv(str(fp))
        strats = all_strategies.get(sym, [])
        P(f"\n  {cfg['n']}: {len(strats)} strategies, {len(cs)} bars")

        inst_results = []
        for strat in strats:
            # Run training with this strategy's parameters
            ps = cfg["pip"]
            lb = 100; mhold = 20
            R = {"strat": strat["id"], "sigs": 0, "w": 0, "l": 0, "be": 0, "pnl": 0.0, "tp": 0.0, "sl": 0.0}
            sc = 0; bs = 4; act = None
            ind_weights = strat.get("indicators", {})
            regime_filter = strat.get("regime_filter", [])
            force_d = strat.get("force_direction", cfg.get("fd"))
            min_conf = strat.get("min_confidence", 0.5)

            for i in range(lb, len(cs) - mhold - 1):
                win = cs[max(0, i - lb):i + 1]
                fut = cs[i + 1:i + 1 + mhold]
                if not fut: continue

                if act is not None:
                    ffs = [c for c in fut if c["epoch"] > act["ep"]][:mhold]
                    if len(ffs) >= act["rem"] or i >= len(cs) - mhold - 2:
                        out = simulate_trade(act["dir"], act["epx"], act["tp"], act["sl"], ffs, ps, mhold, act["tr"], 0.5, act["be"])
                        le.learn_from_failure(act["ts"], act["reg"], out.failure_category or "", out.pnl_pips, act["ep"], act["tp"], act["sl"], sym, "1h", act["dir"])
                        R["sigs"] += 1
                        if out.outcome == "win": R["w"] += 1
                        elif out.outcome == "loss": R["l"] += 1
                        else: R["be"] += 1
                        R["pnl"] += out.pnl_pips
                        if out.pnl_pips > 0: R["tp"] += out.pnl_pips
                        else: R["sl"] += abs(out.pnl_pips)
                        act = None

                bs += 1
                if act is not None or bs < 3 or sc >= 100: continue

                try:
                    rg = regim(win)
                    if regime_filter and rg not in regime_filter: continue

                    ap = atrp(win, ps)
                    # Apply strategy TP/SL multipliers
                    base_tp, base_sl, tr, be = le.get_optimal_tp_sl(rg, ap, sym, "1h")
                    tp2 = base_tp * strat.get("tp_mult", 1.0)
                    sl2 = base_sl * strat.get("sl_mult", 1.0)

                    sig = generate_signal(candles=win, pip_size=ps, tp_override=tp2, sl_override=sl2)
                    if sig.direction == "HOLD": continue
                    if force_d and sig.direction != force_d:
                        sig.direction = force_d
                        sig.confidence *= 0.8
                    if sig.confidence < min_conf: continue

                    ok, _ = le.should_take_signal(sig.direction, sig.confidence, rg, sig.tool_scores, sym, "1h", win[-1]["epoch"])
                    if not ok: continue

                    act = {"dir": sig.direction, "epx": win[-1]["close"], "tp": sig.recommended_tp, "sl": sig.recommended_sl,
                           "ts": sig.tool_scores, "reg": rg, "conf": sig.confidence, "ep": win[-1]["epoch"], "rem": mhold, "tr": tr, "be": be}
                    bs = 0; sc += 1
                except: continue

            t = R["w"] + R["l"] + R["be"]
            R["wr"] = R["w"] / max(t, 1) * 100
            R["pf"] = R["tp"] / max(R["sl"], 1e-10)
            R["avg"] = R["pnl"] / max(t, 1)
            inst_results.append(R)

        # Find best strategy for this instrument
        inst_results.sort(key=lambda x: x["wr"], reverse=True)
        strategy_results[sym] = inst_results
        best_s = inst_results[0] if inst_results else {"wr": 0}
        P(f"    Best: {best_s.get('strat','?')} WR={best_s.get('wr',0):.1f}% PF={best_s.get('pf',0):.2f}")

    # Save strategy training results
    with open(OUT / "v3_strategy_results.json", "w") as f:
        json.dump(strategy_results, f, indent=2, default=str)

    # Save final learned state
    with open(OUT / "v3_learned_state_final.json", "w") as f:
        json.dump(le.to_dict(), f, indent=2, default=str)
    with open(OUT / "v3_learned_state_best.json", "w") as f:
        json.dump(le.to_dict(), f, indent=2, default=str)

    P(f"\n  FINAL: {len(le.patterns)} patterns learned")
    return le, strategy_results


def main():
    t_start = time.time()
    P("#" * 60)
    P("  VEILCREAN FULL TRAINING PIPELINE")
    P("  Phase 1: Q-Learning | Phase 2: 10 Strategies/Instrument | Phase 3: Retrain")
    P("#" * 60)

    # Phase 1: Train with Q-learning
    le = phase1_training()

    # Phase 2: Create strategies
    all_strats = phase2_strategies(le)

    # Phase 3: Retrain with strategies
    le, strat_results = phase3_train_with_strategies(le, all_strats)

    total_time = time.time() - t_start
    P(f"\n{'#'*60}")
    P(f"  ALL PHASES COMPLETE in {total_time:.0f}s")
    P(f"  Final patterns: {len(le.patterns)}")
    P(f"  Files saved to training/output/ and training/strategies/")
    P(f"{'#'*60}")

if __name__ == "__main__":
    main()