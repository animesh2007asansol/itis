#!/usr/bin/env python3
"""
presignal_analyzer.py - Pre-signal context analyzer
Analyzes 1-20 days before each signal for Short Hold and Monthly Picks stocks.
"""
import json, sys, warnings, math
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "equity"
OUT      = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

LOOKBACK_DAYS  = [1, 2, 3, 5, 7, 10, 15, 20]
RSI_PERIOD     = 14
ATR_PERIOD     = 14
VOL_MA_DAYS    = 20
WEEK_52_DAYS   = 252
MIN_SIGNALS    = 3
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None

def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing."); sys.exit(1)
    manifest = json.loads(MANIFEST.read_text())
    dates = sorted(manifest.keys())
    print(f"  Loading {len(dates)} trading dates...")
    frames = []
    for ds in dates:
        y, mo, _ = ds.split("-")
        p = DATA / y / mo / f"{ds}.csv"
        if not p.exists(): continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = df.columns.str.strip()
            cm = {}
            for c in df.columns:
                u = c.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):               cm[c]="sym"
                elif u in ("SERIES","SCTYSRS"):              cm[c]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):   cm[c]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):   cm[c]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):      cm[c]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"): cm[c]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):       cm[c]="v"
            df = df.rename(columns=cm)
            if not {"sym","series","o","h","l","c","v"}.issubset(df.columns): continue
            df = df[df["series"].str.strip().isin(["EQ","BE"])].copy()
            df["date"] = pd.to_datetime(ds)
            frames.append(df[["sym","o","h","l","c","v","date"]])
        except: continue
    if not frames: sys.exit(1)
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    return all_data.dropna(subset=["o","h","l","c","v"]).sort_values(["sym","date"]).reset_index(drop=True)

def add_indicators(df):
    c = df["c"]; h = df["h"]; l = df["l"]; v = df["v"]
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    al    = loss.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    rs    = ag / al.replace(0, np.nan)
    df["rsi"] = (100 - 100/(1+rs)).round(2)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(ATR_PERIOD, min_periods=5).mean()
    df["atr_pct"] = (df["atr"] / c * 100).round(3)
    df["vol_ma"]    = v.rolling(VOL_MA_DAYS, min_periods=5).mean()
    df["vol_ratio"] = (v / df["vol_ma"].replace(0, np.nan)).round(3)
    for d in LOOKBACK_DAYS:
        df[f"ret{d}d"] = (c.pct_change(d)*100).round(3)
    up = (c > c.shift()).astype(int)
    dn = (c < c.shift()).astype(int)
    df["c_up"] = up.groupby((up!=up.shift()).cumsum()).cumsum() * up
    df["c_dn"] = dn.groupby((dn!=dn.shift()).cumsum()).cumsum() * dn
    df["hi52"]          = c.rolling(WEEK_52_DAYS, min_periods=60).max()
    df["lo52"]          = c.rolling(WEEK_52_DAYS, min_periods=60).min()
    df["pct_from_hi52"] = ((c - df["hi52"]) / df["hi52"] * 100).round(2)
    df["pct_from_lo52"] = ((c - df["lo52"]) / df["lo52"] * 100).round(2)
    df["sma20"]         = c.rolling(20, min_periods=10).mean()
    return df

def context_at(df, idx):
    c = df["c"]
    ctx = {"sig_date": str(df["date"].iloc[idx].date())}
    ctx["price"]         = r2(c.iloc[idx])
    ctx["rsi"]           = r2(df["rsi"].iloc[idx])
    ctx["vol_ratio"]     = r2(df["vol_ratio"].iloc[idx])
    ctx["atr_pct"]       = r2(df["atr_pct"].iloc[idx])
    ctx["c_dn"]          = int(df["c_dn"].iloc[idx]) if not pd.isna(df["c_dn"].iloc[idx]) else 0
    ctx["c_up"]          = int(df["c_up"].iloc[idx]) if not pd.isna(df["c_up"].iloc[idx]) else 0
    ctx["pct_from_hi52"] = r2(df["pct_from_hi52"].iloc[idx])
    ctx["pct_from_lo52"] = r2(df["pct_from_lo52"].iloc[idx])
    sma = df["sma20"].iloc[idx]
    ctx["above_sma20"]   = bool(c.iloc[idx] > sma) if not pd.isna(sma) else None
    for d in LOOKBACK_DAYS:
        i = idx - d
        if i >= 0:
            ctx[f"ret_before_{d}d"] = r2((c.iloc[idx] - c.iloc[i]) / c.iloc[i] * 100)
            ctx[f"vol_before_{d}d"] = r2(df["vol_ratio"].iloc[i])
            ctx[f"rsi_before_{d}d"] = r2(df["rsi"].iloc[i])
        else:
            ctx[f"ret_before_{d}d"] = None
            ctx[f"vol_before_{d}d"] = None
            ctx[f"rsi_before_{d}d"] = None
    if idx >= 20:
        w = c.iloc[idx-20:idx+1].values
        ti = int(np.argmin(w[:-1]))
        tv = w[ti]; pk = w[0]; pn = c.iloc[idx]
        ctx["max_dip_20d"]          = r2((tv - pk) / pk * 100)
        ctx["recovery_from_trough"] = r2((pn - tv) / tv * 100)
        ctx["dip_days_ago"]         = 20 - ti
        ctx["loss_then_rise"]       = (ctx["max_dip_20d"] is not None and ctx["max_dip_20d"] < -5 and
                                       ctx["recovery_from_trough"] is not None and ctx["recovery_from_trough"] > 3)
    else:
        ctx["max_dip_20d"] = None; ctx["recovery_from_trough"] = None
        ctx["dip_days_ago"] = None; ctx["loss_then_rise"] = False
    if idx >= 5:
        vols = df["vol_ratio"].iloc[idx-4:idx+1].values
        valid = vols[~np.isnan(vols)]
        ctx["vol_trend_5d"] = ("rising" if len(valid)>1 and valid[-1]>valid[0]*1.2 else
                               "falling" if len(valid)>1 and valid[-1]<valid[0]*0.8 else "flat")
        ctx["vol_avg_5d"]   = r2(float(np.nanmean(vols)))
        ctx["vol_max_5d"]   = r2(float(np.nanmax(vols)))
    else:
        ctx["vol_trend_5d"] = None; ctx["vol_avg_5d"] = None; ctx["vol_max_5d"] = None
    ctx["near_52w_high"] = (ctx["pct_from_hi52"] is not None and ctx["pct_from_hi52"] >= -8)
    ctx["near_52w_low"]  = (ctx["pct_from_lo52"] is not None and ctx["pct_from_lo52"] <= 15)
    return ctx

def safe_mean(lst):
    vals = [x for x in lst if x is not None]
    return r2(float(np.mean(vals))) if vals else None

def safe_pct(lst, fn):
    vals = [x for x in lst if x is not None]
    return r2(100*sum(1 for v in vals if fn(v))/len(vals)) if vals else None

def classify(ctxs):
    if not ctxs: return "Unknown"
    avg_rsi = safe_mean([c["rsi"] for c in ctxs])
    avg_r5  = safe_mean([c.get("ret_before_5d") for c in ctxs])
    avg_r10 = safe_mean([c.get("ret_before_10d") for c in ctxs])
    avg_vol = safe_mean([c["vol_ratio"] for c in ctxs])
    avg_v5  = safe_mean([c.get("vol_avg_5d") for c in ctxs])
    avg_ph  = safe_mean([c["pct_from_hi52"] for c in ctxs])
    avg_pl  = safe_mean([c["pct_from_lo52"] for c in ctxs])
    pct_ltr = safe_pct([c.get("loss_then_rise") for c in ctxs], lambda x: x is True)
    pct_vs  = safe_pct([c["vol_ratio"] for c in ctxs], lambda x: x is not None and x >= 1.8)
    pct_rlo = safe_pct([c["rsi"] for c in ctxs], lambda x: x is not None and x < 40)
    pct_nhi = safe_pct([c["pct_from_hi52"] for c in ctxs], lambda x: x is not None and x >= -8)
    pct_nlo = safe_pct([c["pct_from_lo52"] for c in ctxs], lambda x: x is not None and x <= 15)
    scores = {"Oversold Bounce":0,"Loss Then Rise":0,"52W High Breakout":0,
              "52W Low Reversal":0,"Volume Spike Entry":0,"Volume Accumulation":0,
              "Momentum Continuation":0,"Heavyweight":0}
    if avg_rsi is not None and avg_rsi<40: scores["Oversold Bounce"]+=2
    if pct_rlo is not None and pct_rlo>60: scores["Oversold Bounce"]+=2
    if pct_ltr is not None and pct_ltr>60: scores["Loss Then Rise"]+=3
    if avg_r10 is not None and avg_r10<-8:  scores["Loss Then Rise"]+=2
    if pct_nhi is not None and pct_nhi>70:  scores["52W High Breakout"]+=3
    if avg_ph  is not None and avg_ph>=-5:  scores["52W High Breakout"]+=2
    if pct_nlo is not None and pct_nlo>60:  scores["52W Low Reversal"]+=3
    if avg_pl  is not None and avg_pl<=10:  scores["52W Low Reversal"]+=2
    if pct_vs  is not None and pct_vs>60:   scores["Volume Spike Entry"]+=3
    if avg_vol is not None and avg_vol>2.0: scores["Volume Spike Entry"]+=2
    if avg_v5  is not None and avg_v5>1.3:  scores["Volume Accumulation"]+=2
    if avg_r5  is not None and avg_r5>5:    scores["Momentum Continuation"]+=2
    if avg_rsi is not None and avg_rsi>60:  scores["Momentum Continuation"]+=1
    rsi_vals = [c["rsi"] for c in ctxs if c.get("rsi") is not None]
    rsi_std  = float(np.std(rsi_vals)) if len(rsi_vals)>2 else 99
    if rsi_std < 15 and len(ctxs)>=5: scores["Heavyweight"]+=2
    best = max(scores.values())
    if best <= 1: return "Mixed / Multiple Factors"
    top = [k for k,v in scores.items() if v==best]
    return top[0] if len(top)==1 else top[0]+" + "+top[1]

def build_fp(ctxs):
    if not ctxs: return {}
    def avg(f): return safe_mean([c.get(f) for c in ctxs])
    def pct(f,fn): return safe_pct([c.get(f) for c in ctxs], fn)
    pattern = classify(ctxs)
    daily = []
    for d in LOOKBACK_DAYS:
        rk=f"ret_before_{d}d"; vk=f"vol_before_{d}d"; sk=f"rsi_before_{d}d"
        rv=[c.get(rk) for c in ctxs if c.get(rk) is not None]
        vv=[c.get(vk) for c in ctxs if c.get(vk) is not None]
        sv=[c.get(sk) for c in ctxs if c.get(sk) is not None]
        if rv:
            daily.append({"days_before":d,"avg_ret":r2(float(np.mean(rv))),"min_ret":r2(float(np.min(rv))),
                          "max_ret":r2(float(np.max(rv))),"pct_down":r2(100*sum(1 for v in rv if v<0)/len(rv)),
                          "avg_vol":r2(float(np.mean(vv))) if vv else None,
                          "avg_rsi":r2(float(np.mean(sv))) if sv else None})
    ar=avg("rsi"); ar5=avg("ret_before_5d"); ar10=avg("ret_before_10d"); ar20=avg("ret_before_20d")
    av=avg("vol_ratio"); av5=avg("vol_avg_5d"); aph=avg("pct_from_hi52"); apl=avg("pct_from_lo52")
    altr=pct("loss_then_rise",lambda x:x is True); acd=avg("c_dn")
    pnhi=pct("pct_from_hi52",lambda x:x is not None and x>=-8)
    pnlo=pct("pct_from_lo52",lambda x:x is not None and x<=15)
    pn5=pct("ret_before_5d",lambda x:x is not None and x<0)
    pvs=pct("vol_ratio",lambda x:x is not None and x>=1.8)
    prlo=pct("rsi",lambda x:x is not None and x<40)
    avg_dip=avg("max_dip_20d"); avg_rec=avg("recovery_from_trough")
    conds=[]
    if ar is not None:
        if ar<30: conds.append(f"RSI deeply oversold at signal (avg {ar:.0f}) — stock very cheap historically")
        elif ar<40: conds.append(f"RSI in oversold zone at signal (avg {ar:.0f}) — buying opportunity")
        elif ar>65: conds.append(f"RSI strong at signal (avg {ar:.0f}) — momentum/breakout entry")
        else: conds.append(f"RSI neutral (avg {ar:.0f}) at signal")
    if ar5 is not None:
        if ar5<-10: conds.append(f"Stock fell avg {abs(ar5):.1f}% in 5 days before signal (deep dip then bounce). {pn5 or 0:.0f}% of signals had prior 5d loss.")
        elif ar5<-5: conds.append(f"Stock fell avg {abs(ar5):.1f}% in 5 days before signal. {pn5 or 0:.0f}% had prior loss.")
        elif ar5>8: conds.append(f"Stock was already rising {ar5:.1f}% in 5 days before signal — momentum continuation.")
        else: conds.append(f"Stock was flat ({ar5:+.1f}%) in 5 days before signal.")
    if ar10 and abs(ar10)>5: conds.append(f"10-day trend before signal: {ar10:+.1f}%")
    if ar20 and abs(ar20)>8: conds.append(f"20-day trend before signal: {ar20:+.1f}%")
    if av is not None:
        if av>=1.8: conds.append(f"Volume spike on signal day (avg {av:.1f}x normal). {pvs or 0:.0f}% of signals had vol ≥1.8x.")
        elif av5 and av5>=1.3: conds.append(f"Volume accumulating 5 days before signal (avg {av5:.1f}x normal) — smart money quietly entering.")
        else: conds.append(f"Volume normal at signal ({av:.1f}x average).")
    if aph is not None and aph>=-5: conds.append(f"Stock near 52-week HIGH at signal (avg {aph:.1f}% from hi52). {pnhi or 0:.0f}% of signals near 52w high — breakout pattern.")
    elif aph is not None and aph<=-25: conds.append(f"Stock far from 52w high (avg {aph:.1f}%) — in deep correction or recovery.")
    if apl is not None and apl<=10: conds.append(f"Stock near 52-week LOW at signal (avg {apl:.1f}% above lo52). {pnlo or 0:.0f}% of signals near 52w low — reversal entry.")
    if altr and altr>=60: conds.append(f"{altr:.0f}% of signals showed loss-then-recovery: fell avg {abs(avg_dip or 0):.1f}%, recovered {avg_rec or 0:.1f}% before signal.")
    if acd and acd>=2: conds.append(f"Average {acd:.0f} consecutive DOWN days before signal — sellers exhausted before reversal.")
    bw=[]
    if "52W High" in pattern: bw.append("Enter when stock approaches within 5-8% of 52-week high on rising volume.")
    if "52W Low"  in pattern: bw.append("Enter when stock stabilises near 52-week low and stops making new lows.")
    if "Oversold" in pattern: bw.append(f"Wait for RSI < {40 if (ar or 50)>35 else 35}. Enter when RSI turns up from oversold zone.")
    if "Loss Then Rise" in pattern: bw.append(f"Wait for a {abs(ar5 or 5):.0f}%+ decline over 3-5 days. Enter when price shows 2 consecutive up closes.")
    if "Volume Spike" in pattern: bw.append(f"Enter when volume exceeds {(av or 1.5)*0.8:.1f}x the 20-day average — do not wait.")
    if "Volume Accum" in pattern: bw.append("Enter after 3+ consecutive days of above-average volume.")
    if "Momentum"    in pattern: bw.append("Do not wait for a dip. Enter while stock is already rising.")
    if "Heavyweight" in pattern: bw.append("Enter on the defined signal date. Conditions do not matter for this stock.")
    if not bw: bw.append("Enter on the defined signal date (1st of month / specific week) regardless of current conditions.")
    return {"pattern_type":pattern,"conditions":[cond for cond in conds if cond],"buy_when":" ".join(bw),
            "daily_profile":daily,"avg_rsi_signal":ar,"avg_ret_5d_before":ar5,"avg_ret_10d_before":ar10,
            "avg_ret_20d_before":ar20,"avg_vol_signal":av,"avg_vol_5d_before":av5,
            "avg_pct_from_hi52":aph,"avg_pct_from_lo52":apl,
            "pct_near_hi52":pnhi,"pct_near_lo52":pnlo,"pct_loss_before_5d":pn5,
            "pct_vol_spike":pvs,"pct_rsi_below_40":prlo,"pct_loss_then_rise":altr,
            "avg_max_dip_20d":avg_dip,"avg_recovery":avg_rec,"n_contexts":len(ctxs)}

def main():
    print(f"\n{chr(61)*60}\nPre-Signal Context Analyzer\nStarted: {now}\n{chr(61)*60}")
    sym_sigs = defaultdict(list); sym_source = {}
    sh_path = OUT/"short_hold.json"
    if sh_path.exists():
        sh = json.loads(sh_path.read_text())
        for stk in (sh.get("stocks") or []):
            sym = stk["sym"]
            for occ in (stk.get("occurrences") or []):
                d = occ.get("sig_date") or occ.get("entry_date")
                if d: sym_sigs[sym].append(d)
            sym_source[sym] = "Short Hold"
        print(f"  Short Hold: {len(sym_sigs)} stocks")
    tm_path = OUT/"timing_all.json"
    if tm_path.exists():
        tm = json.loads(tm_path.read_text()); nb = len(sym_sigs)
        for stk in (tm.get("stocks") or []):
            sym = stk["sym"]
            for t in (stk.get("timing") or []):
                # Try occurrences first, then per_occurrence, then exit_curve dates
                occ_list = (t.get("occurrences") or
                            t.get("per_occurrence") or [])
                for occ in occ_list:
                    d = occ.get("sig_date") or occ.get("entry_date")
                    if d: sym_sigs[sym].append(d)
            if sym not in sym_source: sym_source[sym] = "Monthly Picks"
            elif sym_source[sym]=="Short Hold": sym_source[sym]="Short Hold + Monthly"
        print(f"  Monthly Picks: {len(sym_sigs)-nb} additional stocks")
    for sym in sym_sigs: sym_sigs[sym] = sorted(set(sym_sigs[sym]))
    if not sym_sigs: print("ERROR: No signal data. Run Short Hold and Seasonal Timing first."); sys.exit(1)
    all_data = load_all(); grouped = all_data.groupby("sym")
    syms = sorted(sym_sigs.keys()); print(f"\nAnalyzing {len(syms)} stocks...")
    results = []; skipped = 0
    for i, sym in enumerate(syms):
        if (i+1)%100==0: print(f"  {i+1}/{len(syms)} — done {len(results)}")
        if sym not in grouped.groups: skipped+=1; continue
        try:
            df = add_indicators(grouped.get_group(sym).sort_values("date").reset_index(drop=True))
            d2i = {str(df["date"].iloc[j].date()):j for j in range(len(df))}
            ctxs = []
            for d in sym_sigs[sym]:
                idx = d2i.get(d)
                if idx is None or idx<max(LOOKBACK_DAYS): continue
                try: ctxs.append(context_at(df,idx))
                except: pass
            if len(ctxs)<MIN_SIGNALS: skipped+=1; continue
            fp = build_fp(ctxs)
            fp["per_occurrence"] = sorted(ctxs, key=lambda x:x.get("sig_date",""), reverse=True)
            results.append({"sym":sym,"price":r2(float(df["c"].iloc[-1])),
                            "source":sym_source.get(sym,""),"n_signals":len(ctxs),**fp})
        except: skipped+=1
    PR={"52W High Breakout":1,"52W Low Reversal":2,"Volume Spike Entry":3,"Volume Accumulation":4,
        "Oversold Bounce":5,"Loss Then Rise":6,"Momentum Continuation":7,"Heavyweight":8,"Mixed / Multiple Factors":9}
    results.sort(key=lambda x:(PR.get(x.get("pattern_type","").split("+")[0].strip(),9),-x.get("n_signals",0)))
    output={"generated_at":now,"n_stocks":len(results),"n_skipped":skipped,
            "pattern_types":{"Oversold Bounce":"RSI<40 + stock fell before signal. Wait for RSI reversal.",
                "Loss Then Rise":"Consistent dip then recovery before signal.",
                "52W High Breakout":"Signal fires near 52-week high — breakout momentum.",
                "52W Low Reversal":"Signal fires near 52-week low — bottom reversal.",
                "Volume Spike Entry":"Sudden volume surge on signal day.",
                "Volume Accumulation":"Quiet volume buildup before signal — institutional accumulation.",
                "Momentum Continuation":"Stock already rising before signal.",
                "Heavyweight":"Works regardless of conditions — just enter on signal date.",
                "Mixed / Multiple Factors":"Multiple factors together trigger the pattern."},
            "stocks":results}
    path = OUT/"presignal_context.json"
    path.write_text(json.dumps(output,indent=2))
    print(f"\n Written: {path} ({len(results)} stocks)")
    by_pat = Counter(r.get("pattern_type","").split("+")[0].strip() for r in results)
    print("\nPattern breakdown:")
    for pt,n in sorted(by_pat.items(),key=lambda x:-x[1]): print(f"  {n:4d}  {pt}")

if __name__=="__main__": main()
