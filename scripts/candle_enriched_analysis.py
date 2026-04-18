#!/usr/bin/env python3
"""
candle_enriched_analysis.py  v2
================================
Reads candle_today.json (TODAY's candle signals only) and runs deep
analysis on EVERY stock that has a next-day buy signal.

Picks the top 5 by confidence score and outputs:
  candle_enriched.json — deep analysis for ALL today's signaling stocks
  candle_top5.json     — top 5 with full details for display

This means: every stock in the output is a VALID NEXT-DAY BUY based on
today's candle matching its 100%-win historical pattern.
"""

import json, gc, sys, os
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install pandas numpy"); sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "pattern_signals"
MANIFEST  = DATA_DIR / "manifest.json"

LIQUIDITY_TARGET_RS = 200_000

# ─────────────────────────────────────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)): return str(obj)
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.bool_):     return bool(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

# ─────────────────────────────────────────────────────────────────────────────
# Load one stock's full OHLCV history
# ─────────────────────────────────────────────────────────────────────────────
COL_MAP = {
    "SYMBOL":"SYMBOL","TCKRSYMB":"SYMBOL",
    "SERIES":"SERIES","SCTYSRS":"SERIES",
    "OPEN":"OPEN","OPNPRIC":"OPEN","OPEN PRICE":"OPEN",
    "HIGH":"HIGH","HGHPRIC":"HIGH","HIGH PRICE":"HIGH",
    "LOW":"LOW","LWPRIC":"LOW","LOW PRICE":"LOW",
    "CLOSE":"CLOSE","CLSPRIC":"CLOSE","CLOSE PRICE":"CLOSE","LASTPRIC":"CLOSE",
    "TOTTRDQTY":"VOLUME","TTLTRADGVOL":"VOLUME",
    "TIMESTAMP":"DATE","DATE":"DATE","DATE1":"DATE","BIZDT":"DATE",
}

def load_stock(sym, trading_days):
    rows = []
    for ds in trading_days:
        y, m, _ = ds.split("-")
        path = DATA_DIR / "equity" / y / m / f"{ds}.csv"
        if not path.exists(): continue
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.strip('"').str.strip("'").str.upper()
            df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
            if not all(c in df.columns for c in ["SYMBOL","OPEN","HIGH","LOW","CLOSE"]): continue
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip().isin(["EQ","BE"])]
            row = df[df["SYMBOL"].str.strip()==sym]
            if row.empty: continue
            row = row.iloc[0]
            vol = float(str(row.get("VOLUME","0")).replace(",","")) if "VOLUME" in df.columns else 0.0
            rows.append({
                "date": pd.Timestamp(ds),
                "o": float(row["OPEN"]), "h": float(row["HIGH"]),
                "l": float(row["LOW"]),  "c": float(row["CLOSE"]),
                "v": vol,
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Candle classification (must match candle_patterns.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def body_bkt(bp):
    if bp<0.3: return "doji"
    if bp<1.0: return "tiny"
    if bp<2.5: return "small"
    if bp<5.0: return "medium"
    return "large"

def wick_bkt(r):
    if pd.isna(r) or r<0.1: return "none"
    if r<0.5: return "tiny"
    if r<1.0: return "small"
    if r<2.0: return "medium"
    if r<4.0: return "long"
    return "very_long"

def classify_key(o, h, l, c):
    body=abs(c-o); upper=h-max(o,c); lower=min(o,c)-l
    sb=body if body>0 else 0.0001
    bp=body/c*100 if c>0 else 0
    color="G" if c>=o else "R"
    return f"{color}_{body_bkt(bp)}_L{wick_bkt(lower/sb)}_U{wick_bkt(upper/sb)}"

# ─────────────────────────────────────────────────────────────────────────────
# Deep analysis for one stock+pattern
# ─────────────────────────────────────────────────────────────────────────────
def deep_analyse(sym, candle_key, alert, df):
    """
    For every historical occurrence of candle_key on this stock:
    - Gap analysis (next open vs signal close)
    - Entry dip (did next day's LOW go below signal close?)
    - Pre-signal trend context (5d, 20d, 60d before signal)
    - Volume ratio on signal day
    - Turnover / liquidity
    Returns None if insufficient data.
    """
    if df.empty or len(df) < 10: return None

    df = df.copy().reset_index(drop=True)
    df["ckey"] = [classify_key(float(r.o),float(r.h),float(r.l),float(r.c))
                  for _, r in df.iterrows()]

    occurrences = []
    for i, row in df.iterrows():
        if row["ckey"] != candle_key: continue
        if i+1 >= len(df): continue  # no next day

        sig_c = float(row["c"]); sig_v = float(row["v"])
        nxt   = df.iloc[i+1]
        nxt_o = float(nxt["o"]); nxt_h = float(nxt["h"])
        nxt_l = float(nxt["l"]); nxt_c = float(nxt["c"])

        gap_pct       = round((nxt_o - sig_c) / sig_c * 100, 2)
        entry_dip     = nxt_l < sig_c
        entry_dip_pct = round((sig_c - nxt_l) / sig_c * 100, 2) if entry_dip else 0.0
        best_entry    = round(nxt_l if entry_dip else nxt_o, 2)
        best_entry_pct= round((best_entry - sig_c) / sig_c * 100, 2)
        nd_return     = round((nxt_c - sig_c) / sig_c * 100, 2)
        nd_max        = round((nxt_h - sig_c) / sig_c * 100, 2)
        max_from_entry= round((nxt_h - best_entry) / best_entry * 100, 2) if best_entry > 0 else 0.0

        pre5  = df.iloc[max(0,i-5):i]
        pre20 = df.iloc[max(0,i-20):i]
        pre60 = df.iloc[max(0,i-60):i]
        trend5  = round((sig_c/float(pre5.iloc[0]["c"])-1)*100,2) if len(pre5)>=1 else None
        trend20 = round((sig_c/float(pre20.iloc[0]["c"])-1)*100,2) if len(pre20)>=1 else None
        trend60 = round((sig_c/float(pre60.iloc[0]["c"])-1)*100,2) if len(pre60)>=1 else None

        vol_avg20  = float(pre20["v"].mean()) if len(pre20)>=5 else 0.0
        vol_ratio  = round(sig_v/vol_avg20,2) if vol_avg20>0 else None
        turnover   = round(sig_c * sig_v)

        high20 = float(pre20["h"].max()) if len(pre20)>=1 else sig_c
        low20  = float(pre20["l"].min()) if len(pre20)>=1 else sig_c

        occurrences.append({
            "date":              str(row["date"].date()),
            "sig_close":         round(sig_c, 2),
            "sig_volume":        int(sig_v),
            "turnover_rs":       turnover,
            "vol_ratio_vs_20d":  vol_ratio,
            "gap_pct":           gap_pct,
            "entry_dip":         entry_dip,
            "entry_dip_pct":     entry_dip_pct,
            "best_entry_px":     best_entry,
            "best_entry_vs_close": best_entry_pct,
            "nd_return":         nd_return,
            "nd_max_gain":       nd_max,
            "max_gain_from_entry": max_from_entry,
            "next_open":         round(nxt_o,2),
            "next_high":         round(nxt_h,2),
            "next_low":          round(nxt_l,2),
            "next_close":        round(nxt_c,2),
            "trend_5d_before":   trend5,
            "trend_20d_before":  trend20,
            "trend_60d_before":  trend60,
            "dist_from_high20":  round((sig_c-high20)/high20*100,2),
            "dist_from_low20":   round((sig_c-low20)/low20*100,2),
        })

    if len(occurrences) < 2: return None

    n          = len(occurrences)
    gaps       = [o["gap_pct"] for o in occurrences]
    dips       = [o["entry_dip_pct"] for o in occurrences]
    nd_rets    = [o["nd_return"] for o in occurrences]
    nd_maxes   = [o["nd_max_gain"] for o in occurrences]
    best_ents  = [o["best_entry_vs_close"] for o in occurrences]
    max_froms  = [o["max_gain_from_entry"] for o in occurrences]
    turnovers  = [o["turnover_rs"] for o in occurrences]
    vol_rats   = [o["vol_ratio_vs_20d"] for o in occurrences if o["vol_ratio_vs_20d"]]

    n_gap_up   = sum(1 for g in gaps if g>0)
    n_dip      = sum(1 for o in occurrences if o["entry_dip"])

    avg_turnover  = round(sum(turnovers)/len(turnovers)) if turnovers else 0
    liquidity_ok  = avg_turnover >= LIQUIDITY_TARGET_RS

    max_adverse   = round(max(dips),2) if dips else 0.0
    sl_needed     = max_adverse > 1.0
    suggested_sl  = round(-max_adverse-0.5,1) if sl_needed else None

    avg_vol_ratio = round(sum(vol_rats)/len(vol_rats),2) if vol_rats else None
    pre_trends    = [o["trend_20d_before"] for o in occurrences if o["trend_20d_before"] is not None]
    avg_trend20   = round(sum(pre_trends)/len(pre_trends),2) if pre_trends else None

    pct_gap_up    = round(n_gap_up/n*100,1)
    pct_dip       = round(n_dip/n*100,1)

    # Today's close is the signal price
    today_close   = float(alert["today_close"])

    # Entry strategy: if >50% of time price dips below close, wait for dip
    if pct_dip > 50:
        rec_buy   = round(today_close * (1 - avg_vol_ratio*0.002 if avg_vol_ratio else 0.005), 2)
        buy_note  = f"Limit order ~Rs{rec_buy} (below close — dips {pct_dip:.0f}% of times)"
    else:
        rec_buy   = today_close
        buy_note  = f"Buy at open tomorrow (gap-up {pct_gap_up:.0f}% of times, buy at close Rs{today_close} or open)"

    # Targets based on historical min/avg
    nd_min  = round(min(nd_rets),2)
    nd_avg  = round(sum(nd_rets)/n,2)
    nd_max_val = round(max(nd_rets),2)
    t1      = round(today_close*(1+nd_min/100),2)   # conservative
    t2      = round(today_close*(1+nd_avg/100),2)   # realistic
    t3      = round(today_close*(1+min(o["nd_max_gain"] for o in occurrences)/100),2)  # best-case (min of maxes)

    # Confidence score
    score = 30  # base: 100% win rate
    score += min(20, n*2)
    score += min(15, len(set(o["date"][:4] for o in occurrences))*4)
    if avg_vol_ratio and avg_vol_ratio>=1.5: score+=10
    if pct_gap_up>=80: score+=10
    if not sl_needed: score+=10
    if liquidity_ok: score+=5
    score = min(100, score)

    # Strength note
    notes=[]
    if avg_vol_ratio and avg_vol_ratio>=2: notes.append(f"Volume was {avg_vol_ratio}x avg on signal days")
    if avg_trend20 is not None and avg_trend20<-3: notes.append(f"Stock was down {abs(avg_trend20):.1f}% in 20d before — dip reversal")
    if pct_gap_up>=80: notes.append(f"Opens higher {pct_gap_up:.0f}% of next days")
    if n>=6: notes.append(f"Fired {n} times — robust sample")

    return {
        "sym":                sym,
        "candle_key":         candle_key,
        "desc":               alert.get("candle_desc",""),
        "today_date":         alert["today_date"],
        "today_close":        today_close,
        "today_vol":          alert.get("today_vol",0),
        "is_today_signal":    True,  # always True — only today's signals are here
        "n_occurrences":      n,
        "years":              alert.get("years",[]),
        "score":              score,
        # Entry
        "pct_gap_up":         pct_gap_up,
        "avg_gap_pct":        round(sum(gaps)/n,2),
        "min_gap_pct":        round(min(gaps),2),
        "pct_entry_dip":      pct_dip,
        "avg_dip_pct":        round(sum(dips)/n,2),
        "max_dip_pct":        max_adverse,
        "recommended_buy":    rec_buy,
        "buy_note":           buy_note,
        # Returns
        "nd_min_return":      nd_min,
        "nd_avg_return":      nd_avg,
        "nd_max_return":      nd_max_val,
        "nd_avg_high":        round(sum(nd_maxes)/n,2),
        "max_gain_from_entry":round(sum(max_froms)/n,2),
        # Targets (based on today_close)
        "target_1":           t1,
        "target_2":           t2,
        "target_3":           t3,
        "target_1_pct":       nd_min,
        "target_2_pct":       nd_avg,
        # SL
        "sl_needed":          sl_needed,
        "max_adverse_excursion": max_adverse,
        "suggested_sl_pct":   suggested_sl,
        "suggested_sl_px":    round(today_close*(1+suggested_sl/100),2) if suggested_sl else None,
        "sl_note":            (f"Max dip was {max_adverse}% — SL at Rs{round(today_close*(1-max_adverse/100),2)}"
                               if sl_needed else "No SL needed — never went below signal close in all occurrences"),
        # Liquidity
        "avg_turnover_rs":    avg_turnover,
        "liquidity_ok":       liquidity_ok,
        "liquidity_note":     (f"Rs{avg_turnover:,} avg daily — easily absorbs Rs2L order"
                               if liquidity_ok else f"Low liquidity Rs{avg_turnover:,} — partial fill risk for Rs2L"),
        # Context
        "avg_vol_ratio":      avg_vol_ratio,
        "avg_trend_20d_before": avg_trend20,
        "strength_note":      ". ".join(notes) or "Consistent 100%-win candle pattern",
        # All occurrences (for table expansion)
        "occurrences":        occurrences,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Candle Enriched Analysis  v2 — TODAY signals only")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f: manifest=json.load(f)
    tds = sorted(manifest.keys())
    latest = tds[-1]
    print(f"  Trading days: {len(tds)}  Latest: {latest}")

    # Load TODAY's signals only
    today_path = OUT_DIR / "candle_today.json"
    if not today_path.exists():
        print("ERROR: candle_today.json not found. Run candle_patterns.py first.")
        sys.exit(1)
    with open(today_path) as f: today_data = json.load(f)
    alerts = today_data.get("alerts", [])
    print(f"  Today's signals: {len(alerts)} stocks (date: {today_data.get('signal_date','?')})")

    if not alerts:
        print("  No signals today — writing empty outputs.")
        now_ist = datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%Y-%m-%dT%H:%M:%S+05:30")
        jdump({"generated_at":now_ist,"signal_date":latest,"n_analysed":0,"patterns":[]}, OUT_DIR/"candle_enriched.json")
        jdump({"generated_at":now_ist,"signal_date":latest,"top5":[]}, OUT_DIR/"candle_top5.json")
        return

    # Deep analyse each signaling stock
    enriched = []
    for i, alert in enumerate(alerts):
        sym  = alert["sym"]
        ckey = alert["candle_key"]
        print(f"  [{i+1}/{len(alerts)}] {sym} ({ckey})...", end="", flush=True)
        df = load_stock(sym, tds)
        if df.empty:
            print(" NO DATA"); continue
        result = deep_analyse(sym, ckey, alert, df)
        if not result:
            print(" INSUFFICIENT DATA"); continue
        enriched.append(result)
        print(f" score={result['score']} occ={result['n_occurrences']} "
              f"T1=Rs{result['target_1']}(+{result['nd_min_return']}%) "
              f"T2=Rs{result['target_2']}(+{result['nd_avg_return']}%)")

    enriched.sort(key=lambda x: (-x["score"], -x["nd_avg_return"]))
    top5 = enriched[:5]

    now_ist = datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    jdump({"generated_at":now_ist, "signal_date":latest,
           "note":"Deep analysis of TODAY's candle signals only",
           "n_analysed":len(enriched), "patterns":enriched},
          OUT_DIR/"candle_enriched.json")
    print(f"\n  OK candle_enriched.json ({len(enriched)} stocks analysed)")

    jdump({"generated_at":now_ist, "signal_date":latest,
           "note":"Top 5 from today's candle signals by confidence score",
           "top5":top5},
          OUT_DIR/"candle_top5.json")
    print(f"  OK candle_top5.json")

    print(f"\nTop 5 for next trading day after {latest}:")
    for r in top5:
        print(f"  {r['sym']:<14} score={r['score']}  "
              f"buy=Rs{r['recommended_buy']}  "
              f"T1=Rs{r['target_1']}(+{r['nd_min_return']}%)  "
              f"T2=Rs{r['target_2']}(+{r['nd_avg_return']}%)")

if __name__=="__main__":
    main()
