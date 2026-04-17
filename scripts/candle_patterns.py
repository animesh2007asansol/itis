#!/usr/bin/env python3
"""
candle_patterns.py  v3
=======================
Fixes from v2:
  1. TODAY ALERTS ONLY: Only stocks whose LATEST data date == manifest latest date
     are included in candle_today.json. If a stock last traded in 2022 and today
     is 2026, it is NOT included in today's alerts.
  2. VOLUME FILTER: Signal day must have volume >= 50,000 shares. Also, the
     stock's latest (today's) candle must have volume >= 50,000 to appear in alerts.
  3. PERFORMANCE TRACKER: Saves candle_yesterday.json — previous day's alerts
     with actual next-day returns so you can see if the prediction was right.
  4. DYNAMIC PATTERN REMOVAL: If a candle pattern that was previously 100% win
     gets a confirmed negative result (i.e., the pattern fired yesterday and
     today's close < yesterday's close), that pattern occurrence is marked and
     the pattern's win rate is recalculated. If it drops below 100%, it is
     removed from future alerts and best_patterns.
  5. INCREMENTAL: Checkpoint tracks last run date — only full reload on new dates.

Outputs → pattern_signals/:
  candle_stock_patterns.json  — per-stock 100%-win patterns (searchable)
  candle_best_patterns.json   — all 100%-win patterns ranked by min return
  candle_today.json           — ONLY stocks with data on latest manifest date
  candle_yesterday.json       — previous day's alerts + actual next-day return
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

REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "pattern_signals"
MANIFEST   = DATA_DIR / "manifest.json"
CHECKPOINT = OUT_DIR / "candle_checkpoint.json"

MIN_VOLUME        = 50_000   # minimum volume on BOTH signal day AND today's candle
MIN_OCC           = 3        # minimum occurrences to compute stats
MIN_TRADING_DAYS  = 100      # minimum history per stock

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
# COLUMN ALIASES
# ─────────────────────────────────────────────────────────────────────────────
SYM_A=["SYMBOL","TCKRSYMB"]; SER_A=["SERIES","SCTYSRS"]
O_A=["OPEN","OPNPRIC"]; H_A=["HIGH","HGHPRIC"]
L_A=["LOW","LWPRIC"]; C_A=["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"]
V_A=["TOTTRDQTY","TTLTRADGVOL","VOLUME"]

def _fc(hdr, aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2: return rows
        hdr=[h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_sym=_fc(hdr,SYM_A); i_ser=_fc(hdr,SER_A); i_o=_fc(hdr,O_A)
        i_h=_fc(hdr,H_A); i_l=_fc(hdr,L_A); i_c=_fc(hdr,C_A); i_v=_fc(hdr,V_A)
        if i_sym<0 or i_c<0: return rows
        mc=max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v] if x>=0)
        for line in lines[1:]:
            line=line.strip()
            if not line: continue
            cols=[c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols)<=mc: continue
            ser=cols[i_ser].strip() if i_ser>=0 else "EQ"
            if ser not in ("EQ","BE"): continue
            try:
                sym=cols[i_sym].strip(); c=float(cols[i_c])
                o=float(cols[i_o]) if i_o>=0 else c
                h=float(cols[i_h]) if i_h>=0 else c
                l=float(cols[i_l]) if i_l>=0 else c
                v=float(cols[i_v].replace(",","")) if i_v>=0 else 0.0
                if c>0 and o>0 and h>=max(o,c) and l<=min(o,c) and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except (ValueError,IndexError): pass
    except Exception: pass
    return rows

def load_all(trading_days):
    print(f"  Loading {len(trading_days)} trading days...")
    rows=[]; loaded=0
    for ds in trading_days:
        y,m,_=ds.split("-")
        path=DATA_DIR/"equity"/y/m/f"{ds}.csv"
        if not path.exists(): continue
        r=load_csv(path)
        for x in r: x["date"]=ds
        rows.extend(r); loaded+=1
        if loaded%300==0: print(f"    {loaded}/{len(trading_days)} files, {len(rows):,} rows...")
    print(f"  {len(rows):,} rows, {loaded} files")
    df=pd.DataFrame(rows)
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def load_cp():
    try:
        if CHECKPOINT.exists(): return json.loads(CHECKPOINT.read_text())
    except Exception: pass
    return {}

def save_cp(cp):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(cp, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def body_bkt(bp):
    if bp<0.3:  return "doji"
    if bp<1.0:  return "tiny"
    if bp<2.5:  return "small"
    if bp<5.0:  return "medium"
    return "large"

def wick_bkt(r):
    if pd.isna(r) or r<0.1:  return "none"
    if r<0.5:                 return "tiny"
    if r<1.0:                 return "small"
    if r<2.0:                 return "medium"
    if r<4.0:                 return "long"
    return "very_long"

def classify(o,h,l,c):
    body=abs(c-o); upper_wick=h-max(o,c); lower_wick=min(o,c)-l
    safe_body=body if body>0 else 0.0001
    body_pct=body/c*100 if c>0 else 0
    color="G" if c>=o else "R"
    bb=body_bkt(body_pct); ub=wick_bkt(upper_wick/safe_body); lb=wick_bkt(lower_wick/safe_body)
    key=f"{color}_{bb}_L{lb}_U{ub}"
    desc=f"{'Green' if color=='G' else 'Red'} {bb} body | lower:{lb} | upper:{ub}"
    return key, desc, round(body_pct,2), round(upper_wick/safe_body,2), round(lower_wick/safe_body,2)

# ─────────────────────────────────────────────────────────────────────────────
# PER-STOCK CANDLE ANALYSIS
# KEY FIXES:
#   - latest_str passed in so we know what "today" is globally
#   - Stock only qualifies for TODAY alerts if its last date == latest_str
#   - Today's candle must also have volume >= MIN_VOLUME
#   - Build training data excluding the last row (today) so fwd1 is known
#     for all training rows, and today's prediction is truly forward-looking
# ─────────────────────────────────────────────────────────────────────────────
def analyse_stock(g, sym, latest_str, prev_str):
    g = g.copy().sort_values("date").reset_index(drop=True)
    n = len(g)
    if n < MIN_TRADING_DAYS: return None

    c,o,h,l,v = g["c"],g["o"],g["h"],g["l"],g["v"]
    stock_latest = str(g["date"].max().date())

    # ── CRITICAL FIX: stock must have data on the latest manifest date ──────
    # If a stock's last trade was in 2022, exclude it from today's alerts
    is_current = (stock_latest == latest_str)

    # Check today's volume (only if stock is current)
    today_vol = float(v.iloc[-1]) if is_current else 0.0
    today_vol_ok = today_vol >= MIN_VOLUME

    # Forward returns (vectorised) — shift(-1) uses next row's close
    g["fwd1"]  = c.shift(-1)/c - 1
    g["fwd5"]  = c.shift(-5)/c - 1
    g["fwd10"] = c.shift(-10)/c - 1
    g["fwd20"] = c.shift(-20)/c - 1
    g["next_o"]= o.shift(-1)
    g["year"]  = g["date"].dt.year

    # Classify each candle
    keys, descs = [], []
    for i in range(n):
        k,d,_,_,_ = classify(float(o.iloc[i]),float(h.iloc[i]),float(l.iloc[i]),float(c.iloc[i]))
        keys.append(k); descs.append(d)
    g["ckey"]=keys; g["cdesc"]=descs

    # TRAINING ROWS: exclude today (last row) because fwd1 is unknown for it
    # Also only include rows with sufficient volume
    training = g.iloc[:-1].copy() if is_current else g.copy()
    training  = training[training["v"] >= MIN_VOLUME].copy()
    if len(training) < 5: return None

    # Get today's candle (last row)
    today_row = g.iloc[-1]
    today_key, today_desc, today_bp, today_ur, today_lr = classify(
        float(today_row["o"]),float(today_row["h"]),float(today_row["l"]),float(today_row["c"])
    )

    # Also get previous day's candle (for performance tracking)
    prev_key = None
    if len(g) >= 2:
        pr = g.iloc[-2]
        prev_key, _, _, _, _ = classify(float(pr["o"]),float(pr["h"]),float(pr["l"]),float(pr["c"]))

    desc_map = dict(zip(keys, descs))

    # ── Build pattern stats from training rows ───────────────────────────────
    pattern_stats = []
    for ckey, grp in training.groupby("ckey"):
        nd = grp["fwd1"].dropna()
        if len(nd) < MIN_OCC: continue

        n_total = len(nd)
        n_neg   = int((nd<0).sum())

        # 100% win rate required — ANY negative → discard
        if n_neg > 0:
            continue

        nd_avg = round(float(nd.mean()*100), 2)
        nd_min = round(float(nd.min()*100),  2)
        nd_max = round(float(nd.max()*100),  2)

        # Gap-up rate
        gap = grp["next_o"].dropna()
        if len(gap) >= MIN_OCC:
            gap_pct = (gap / grp.loc[gap.index,"c"] - 1) * 100
            gap_up_rate = round(float((gap_pct>0).sum()/len(gap)*100), 1)
            avg_gap     = round(float(gap_pct.mean()), 2)
        else:
            gap_up_rate = 0.0; avg_gap = 0.0

        wstats = {}
        for col,w in [("fwd5",5),("fwd10",10),("fwd20",20)]:
            wv = grp[col].dropna()
            if len(wv) >= MIN_OCC:
                wstats[str(w)] = {
                    "n":   int(len(wv)),
                    "wr":  round(float((wv>0).sum()/len(wv)*100),1),
                    "avg": round(float(wv.mean()*100),2),
                    "min": round(float(wv.min()*100),2),
                    "max": round(float(wv.max()*100),2),
                }

        years = sorted(int(y) for y in grp["date"].dt.year.unique().tolist())
        last_occ = str(grp["date"].max().date())

        pattern_stats.append({
            "sym":          sym,
            "candle_key":   ckey,
            "desc":         desc_map.get(ckey, ckey),
            "occurrences":  n_total,
            "nd_wr":        100.0,
            "nd_avg":       nd_avg,
            "nd_min":       nd_min,
            "nd_max":       nd_max,
            "gap_up_rate":  gap_up_rate,
            "avg_gap_pct":  avg_gap,
            "win_5d":       wstats.get("5"),
            "win_10d":      wstats.get("10"),
            "win_20d":      wstats.get("20"),
            "years":        years,
            "last_date":    last_occ,
        })

    if not pattern_stats: return None
    pattern_stats.sort(key=lambda x: -x["nd_min"])

    today_hist = next((p for p in pattern_stats if p["candle_key"]==today_key), None)

    # ── Performance data: did yesterday's pattern work? ──────────────────────
    # If today's date is the latest, then "yesterday's" candle is the second-last row
    # We can look up actual fwd1 for the second-last row in our full dataset
    prev_result = None
    if is_current and len(g) >= 2 and prev_key:
        pr = g.iloc[-2]
        # fwd1 for second-last row = today's close / yesterday's close - 1
        # This is knowable today (today's close is the "next day" for yesterday)
        actual_return = round(float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100), 2)
        prev_hist = next((p for p in pattern_stats if p["candle_key"]==prev_key), None)
        prev_result = {
            "date":           str(pr["date"].date()),
            "candle_key":     prev_key,
            "candle_desc":    desc_map.get(prev_key, prev_key),
            "close_then":     round(float(pr["c"]), 2),
            "close_now":      round(float(c.iloc[-1]), 2),
            "actual_return":  actual_return,
            "was_positive":   actual_return > 0,
            "had_pattern":    prev_hist is not None,
            "pattern_predicted_min": prev_hist["nd_min"] if prev_hist else None,
            # If pattern predicted 100% win but actual was negative → pattern broken
            "pattern_broken": prev_hist is not None and actual_return <= 0,
        }

    return {
        "sym":           sym,
        "stock_latest":  stock_latest,
        "is_current":    is_current,
        "today_date":    stock_latest,
        "today_close":   round(float(today_row["c"]), 2),
        "today_vol":     int(today_vol),
        "today_vol_ok":  today_vol_ok,
        "today_key":     today_key,
        "today_candle":  {"key":today_key,"desc":today_desc,"body_pct":today_bp,
                          "upper_wick_ratio":today_ur,"lower_wick_ratio":today_lr},
        "today_hist":    today_hist,
        "prev_result":   prev_result,
        "n_patterns":    len(pattern_stats),
        "patterns":      pattern_stats,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Candle Pattern Tracker  v3")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Manifest...")
    with open(MANIFEST) as f: manifest=json.load(f)
    tds        = sorted(manifest.keys())
    latest_str = tds[-1]     # e.g. "2026-04-16" — THE authoritative "today"
    prev_str   = tds[-2] if len(tds)>=2 else tds[-1]  # "yesterday" trading day
    print(f"  {len(tds)} days [{tds[0]} -> {latest_str}]")
    print(f"  Latest trading day: {latest_str}")
    print(f"  Previous trading day: {prev_str}")

    # Checkpoint
    cp       = load_cp()
    last_run = cp.get("last_run_date","")
    force    = os.environ.get("FORCE_FULL_RERUN","").lower()=="true"
    if last_run == latest_str and not force:
        print(f"  Already ran for {latest_str}. Set FORCE_FULL_RERUN=true to rerun.")
        if (OUT_DIR/"candle_best_patterns.json").exists():
            print("  Files are current. Done.")
            return

    print("\n[2] Loading data...")
    df = load_all(manifest)
    counts    = df.groupby("sym")["date"].count()
    valid_sym = counts[counts>=MIN_TRADING_DAYS].index
    df        = df[df["sym"].isin(valid_sym)].copy()
    sym_list  = sorted(df["sym"].unique())
    print(f"  {len(sym_list)} stocks with >={MIN_TRADING_DAYS} trading days")

    print(f"\n[3] Analysing candle patterns...")
    print(f"  TODAY alerts only for stocks with latest data = {latest_str}")
    print(f"  AND volume >= {MIN_VOLUME:,} on today's candle")

    sym_grps     = {s:g.copy() for s,g in df.groupby("sym")}
    all_profiles = {}
    best_pats    = []
    today_alerts = []    # ONLY stocks with latest data == latest_str + vol ok
    perf_track   = []    # yesterday's candle results (for performance tab)
    broken_pats  = []    # patterns that just broke (were 100% but failed yesterday)

    for i, sym in enumerate(sym_list):
        try:
            res = analyse_stock(sym_grps[sym], sym, latest_str, prev_str)
            if not res: continue

            # Store profile (for candle search — all stocks)
            all_profiles[sym] = {
                "sym":           res["sym"],
                "today_date":    res["today_date"],
                "today_close":   res["today_close"],
                "today_vol":     res["today_vol"],
                "today_key":     res["today_key"],
                "today_candle":  res["today_candle"],
                "today_hist":    res["today_hist"],
                "n_patterns":    res["n_patterns"],
                "patterns":      res["patterns"],
            }

            # Best patterns (100% win) for the best-candles tab
            for pat in res["patterns"]:
                best_pats.append({**pat, "sym":sym})

            # ── TODAY ALERTS: strict conditions ─────────────────────────────
            # 1. Stock must have data on the latest manifest date
            # 2. Today's candle volume must be >= MIN_VOLUME
            # 3. Must match a 100%-win historical pattern
            if (res["is_current"] and
                res["today_vol_ok"] and
                res["today_hist"] is not None):

                th = res["today_hist"]
                today_alerts.append({
                    "sym":          sym,
                    "today_date":   res["today_date"],
                    "today_close":  res["today_close"],
                    "today_vol":    res["today_vol"],
                    "candle_key":   res["today_key"],
                    "candle_desc":  res["today_candle"]["desc"],
                    "nd_wr":        100.0,
                    "nd_avg":       th["nd_avg"],
                    "nd_min":       th["nd_min"],
                    "nd_max":       th["nd_max"],
                    "gap_up_rate":  th["gap_up_rate"],
                    "occurrences":  th["occurrences"],
                    "win_5d":       th["win_5d"],
                    "win_10d":      th["win_10d"],
                    "win_20d":      th["win_20d"],
                    "years":        th["years"],
                    "last_date":    th["last_date"],
                })

            # ── PERFORMANCE TRACKING ─────────────────────────────────────────
            pr = res.get("prev_result")
            if pr and res["is_current"]:
                perf_track.append({
                    "sym":              sym,
                    "signal_date":      pr["date"],
                    "candle_key":       pr["candle_key"],
                    "candle_desc":      pr["candle_desc"],
                    "close_signal":     pr["close_then"],
                    "close_next":       pr["close_now"],
                    "actual_return":    pr["actual_return"],
                    "was_positive":     pr["was_positive"],
                    "had_pattern":      pr["had_pattern"],
                    "pattern_pred_min": pr["pattern_predicted_min"],
                    "pattern_broken":   pr["pattern_broken"],
                })
                if pr["pattern_broken"]:
                    broken_pats.append({
                        "sym":          sym,
                        "candle_key":   pr["candle_key"],
                        "candle_desc":  pr["candle_desc"],
                        "signal_date":  pr["date"],
                        "actual_return":pr["actual_return"],
                        "pred_min":     pr["pattern_predicted_min"],
                    })

        except Exception as e:
            pass
        if (i+1)%200==0:
            print(f"    {i+1}/{len(sym_list)}, current:{len(today_alerts)}, perf:{len(perf_track)}...")

    del sym_grps; gc.collect()

    # Sort
    best_pats.sort(key=lambda x: -x["nd_min"])
    today_alerts.sort(key=lambda x: -x["nd_min"])
    perf_track.sort(key=lambda x: x["actual_return"])  # worst first

    # Stats
    n_current    = sum(1 for r in all_profiles.values())
    n_alerts     = len(today_alerts)
    n_positive   = sum(1 for p in perf_track if p["was_positive"])
    n_with_pat   = sum(1 for p in perf_track if p["had_pattern"])
    n_pat_pos    = sum(1 for p in perf_track if p["had_pattern"] and p["was_positive"])

    print(f"\n  {n_current} stocks analysed")
    print(f"  {len(best_pats)} 100%-win patterns found")
    print(f"  {n_alerts} TODAY alerts (data={latest_str}, vol>={MIN_VOLUME:,})")
    print(f"  {len(perf_track)} performance records (prev day={prev_str})")
    print(f"  {len(broken_pats)} patterns broken by yesterday's data")
    if n_with_pat > 0:
        print(f"  Pattern accuracy yesterday: {n_pat_pos}/{n_with_pat} = {n_pat_pos/n_with_pat*100:.1f}%")

    print("\n[4] Writing outputs...")
    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    jdump({"generated_at":now_ist, "latest_date":latest_str,
           "n_stocks":len(all_profiles), "profiles":all_profiles},
          OUT_DIR/"candle_stock_patterns.json")
    print(f"  OK candle_stock_patterns.json ({len(all_profiles)} stocks)")

    jdump({"generated_at":now_ist, "latest_date":latest_str,
           "n_patterns":len(best_pats),
           "description":"100% next-day positive. Ranked by minimum return (highest first).",
           "patterns":best_pats[:1000]},
          OUT_DIR/"candle_best_patterns.json")
    print(f"  OK candle_best_patterns.json ({len(best_pats)} patterns)")

    jdump({"generated_at":now_ist, "signal_date":latest_str,
           "prediction_for": "next trading day after " + latest_str,
           "note": "Only stocks with data on " + latest_str + " and volume >= " + str(MIN_VOLUME),
           "total_alerts":n_alerts,
           "alerts":today_alerts},
          OUT_DIR/"candle_today.json")
    print(f"  OK candle_today.json ({n_alerts} alerts for {latest_str})")

    jdump({"generated_at":now_ist,
           "signal_date":prev_str,
           "result_date": latest_str,
           "note": "Stocks alerted on " + prev_str + " and their actual next-day return on " + latest_str,
           "total_tracked": len(perf_track),
           "with_pattern":  n_with_pat,
           "pattern_positive": n_pat_pos,
           "pattern_accuracy_pct": round(n_pat_pos/n_with_pat*100,1) if n_with_pat>0 else None,
           "broken_patterns": broken_pats,
           "results": perf_track},
          OUT_DIR/"candle_yesterday.json")
    print(f"  OK candle_yesterday.json ({len(perf_track)} tracked)")

    save_cp({"last_run_date":latest_str, "run_at":now_ist,
             "stocks_analysed":n_current, "best_patterns":len(best_pats),
             "today_alerts":n_alerts})
    print("  OK candle_checkpoint.json")
    print(f"\nDone. Alerts are predictions for next trading day after {latest_str}.")

if __name__=="__main__":
    main()
