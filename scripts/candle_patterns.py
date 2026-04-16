#!/usr/bin/env python3
"""
candle_patterns.py
===================
For every stock, for every trading day:
  1. Classify the candle into a shape bucket based on:
       - Body size as % of price  (tiny/small/medium/large/huge)
       - Lower wick ratio (wick/body)
       - Upper wick ratio (wick/body)
       - Color (green/red)
  2. Track next-day return and 5d/10d/20d returns
  3. Find which candle shapes gave 100% positive next-day return
     AND minimum +15% in at least one of next 5/10/20 trading days
  4. For each stock, record today's candle pattern and match it
     against its own historical pattern performance

Outputs (pattern_signals/):
  candle_stock_patterns.json  — per-stock pattern stats (searchable)
  candle_best_patterns.json   — patterns with 100% next-day positive rate
  candle_today.json           — today's candle for every stock + historical match
"""

import json, gc, sys, math
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

MIN_VOLUME        = 50_000    # minimum tradeable volume
MIN_OCC_FOR_STAT  = 3         # minimum occurrences to compute stats
MIN_WIN_NEXT_DAY  = 60.0      # show patterns where next-day positive >= 60%
MIN_AVG_15_ANY    = 15.0      # minimum avg return in at least ONE of 5/10/20d
PERFECT_WIN       = 100.0     # 100% next-day positive

# ─────────────────────────────────────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)): return str(obj)
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, np.bool_):      return bool(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADING
# ─────────────────────────────────────────────────────────────────────────────
SYM_A = ["SYMBOL","TCKRSYMB"]
SER_A = ["SERIES","SCTYSRS"]
O_A   = ["OPEN","OPNPRIC"]
H_A   = ["HIGH","HGHPRIC"]
L_A   = ["LOW","LWPRIC"]
C_A   = ["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"]
V_A   = ["TOTTRDQTY","TTLTRADGVOL","VOLUME"]

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
        hdr   = [h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_sym = _fc(hdr, SYM_A); i_ser = _fc(hdr, SER_A)
        i_o   = _fc(hdr, O_A);   i_h   = _fc(hdr, H_A)
        i_l   = _fc(hdr, L_A);   i_c   = _fc(hdr, C_A)
        i_v   = _fc(hdr, V_A)
        if i_sym < 0 or i_c < 0: return rows
        mc = max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v] if x >= 0)
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= mc: continue
            ser = cols[i_ser].strip() if i_ser >= 0 else "EQ"
            if ser not in ("EQ","BE"): continue
            try:
                sym = cols[i_sym].strip()
                c   = float(cols[i_c])
                o   = float(cols[i_o]) if i_o >= 0 else c
                h   = float(cols[i_h]) if i_h >= 0 else c
                l   = float(cols[i_l]) if i_l >= 0 else c
                v   = float(cols[i_v].replace(",","")) if i_v >= 0 else 0.0
                if c>0 and o>0 and h>=max(o,c) and l<=min(o,c) and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except (ValueError,IndexError): pass
    except Exception: pass
    return rows

def load_all(trading_days):
    print(f"  Loading {len(trading_days)} days...")
    rows = []; loaded = 0
    for ds in trading_days:
        y,m,_ = ds.split("-")
        path = DATA_DIR/"equity"/y/m/f"{ds}.csv"
        if not path.exists(): continue
        r = load_csv(path)
        for x in r: x["date"] = ds
        rows.extend(r); loaded += 1
        if loaded % 300 == 0:
            print(f"    {loaded}/{len(trading_days)} files, {len(rows):,} rows...")
    print(f"  {len(rows):,} rows, {loaded} files")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def body_bucket(body_pct):
    """Body size as % of close price."""
    if body_pct < 0.3:  return "doji"
    if body_pct < 1.0:  return "tiny"
    if body_pct < 2.0:  return "small"
    if body_pct < 4.0:  return "medium"
    if body_pct < 7.0:  return "large"
    return "huge"

def wick_bucket(ratio):
    """Wick size as multiple of body."""
    if pd.isna(ratio) or ratio < 0.1: return "none"
    if ratio < 0.5:   return "tiny"
    if ratio < 1.0:   return "small"
    if ratio < 2.0:   return "medium"
    if ratio < 3.5:   return "long"
    return "very_long"

def classify_candle(o, h, l, c):
    """Return a candle shape key and human-readable description."""
    body       = abs(c - o)
    rng        = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    color      = "green" if c >= o else "red"
    safe_body  = body if body > 0 else 0.0001
    body_pct   = body / c * 100 if c > 0 else 0

    bb  = body_bucket(body_pct)
    ubk = wick_bucket(upper_wick / safe_body)
    lbk = wick_bucket(lower_wick / safe_body)

    key  = f"{color}_{bb}_U{ubk}_L{lbk}"
    desc = f"{color.capitalize()} {bb} body | lower:{lbk} wick | upper:{ubk} wick"

    return {
        "key":        key,
        "desc":       desc,
        "color":      color,
        "body_pct":   round(body_pct, 2),
        "body_bucket": bb,
        "upper_wick_ratio": round(upper_wick / safe_body, 2),
        "lower_wick_ratio": round(lower_wick / safe_body, 2),
        "upper_bkt":  ubk,
        "lower_bkt":  lbk,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PER-STOCK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyse_stock_candles(g, sym):
    """
    For one stock, classify every candle and compute:
      - next-day return
      - 5/10/20 day returns
    Then group by candle shape and compute win rates.
    """
    g = g.copy().sort_values("date").reset_index(drop=True)
    c, o, h, l, v = g["c"], g["o"], g["h"], g["l"], g["v"]
    n = len(g)
    if n < 50: return None

    # Forward returns
    fwd1  = c.shift(-1) / c - 1   # next day
    fwd5  = c.shift(-5) / c - 1
    fwd10 = c.shift(-10) / c - 1
    fwd20 = c.shift(-20) / c - 1
    # Next open for gap analysis
    next_open = o.shift(-1)

    # Classify each candle
    candle_keys  = []
    candle_descs = {}
    for i in range(n):
        cand = classify_candle(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i])
        candle_keys.append(cand["key"])
        candle_descs[cand["key"]] = cand["desc"]

    g["ckey"]   = candle_keys
    g["fwd1"]   = fwd1
    g["fwd5"]   = fwd5
    g["fwd10"]  = fwd10
    g["fwd20"]  = fwd20
    g["next_o"] = next_open

    # Only tradeable volume rows
    tradeable = g[v >= MIN_VOLUME].copy()
    if len(tradeable) < 10: return None

    # Group by candle key
    pattern_stats = []
    for ckey, grp in tradeable.groupby("ckey"):
        # Next day stats
        nd_valid = grp["fwd1"].dropna()
        if len(nd_valid) < MIN_OCC_FOR_STAT: continue

        nd_wr   = round((nd_valid > 0).sum() / len(nd_valid) * 100, 1)
        nd_avg  = round(nd_valid.mean() * 100, 2)
        nd_min  = round(nd_valid.min()  * 100, 2)
        nd_max  = round(nd_valid.max()  * 100, 2)

        # Gap-up: next open vs today close
        gap_valid = grp["next_o"].dropna()
        gap_up_rate = 0.0
        avg_gap     = 0.0
        if len(gap_valid) >= MIN_OCC_FOR_STAT:
            gaps = (gap_valid / grp.loc[gap_valid.index, "c"] - 1) * 100
            gap_up_rate = round((gaps > 0).sum() / len(gaps) * 100, 1)
            avg_gap     = round(gaps.mean(), 2)

        # Multi-day stats
        w_stats = {}
        for col, w in [("fwd5",5),("fwd10",10),("fwd20",20)]:
            valid = grp[col].dropna()
            if len(valid) >= MIN_OCC_FOR_STAT:
                w_stats[str(w)] = {
                    "n":   int(len(valid)),
                    "wr":  round((valid>0).sum()/len(valid)*100,1),
                    "avg": round(valid.mean()*100,2),
                    "min": round(valid.min()*100,2),
                    "max": round(valid.max()*100,2),
                }

        # Check if this meets the 15% in any one window criterion
        max_avg_any = max(
            (w_stats.get(str(w),{}).get("avg",0) for w in [5,10,20]),
            default=0
        )
        meets_15pct = max_avg_any >= MIN_AVG_15_ANY

        # Date range this pattern appeared
        years = sorted(grp["date"].dt.year.unique().tolist())

        # Last occurrence
        last_date = str(grp["date"].max().date())

        pattern_stats.append({
            "sym":          sym,
            "candle_key":   ckey,
            "desc":         candle_descs.get(ckey, ckey),
            "occurrences":  int(len(grp)),
            "nd_win_rate":  nd_wr,       # % of times next day was positive
            "nd_avg_ret":   nd_avg,      # avg next-day return %
            "nd_min_ret":   nd_min,      # worst next day
            "nd_max_ret":   nd_max,      # best next day
            "gap_up_rate":  gap_up_rate, # % of times next day opened higher
            "avg_gap_pct":  avg_gap,
            "is_100pct_nd": nd_wr == 100.0,
            "meets_15pct":  meets_15pct,
            "max_avg_any_window": round(max_avg_any, 2),
            "win_5d":       w_stats.get("5",{}),
            "win_10d":      w_stats.get("10",{}),
            "win_20d":      w_stats.get("20",{}),
            "years":        years,
            "last_date":    last_date,
        })

    if not pattern_stats: return None
    pattern_stats.sort(key=lambda x: (-x["nd_win_rate"], -x["nd_avg_ret"]))

    # Today's candle
    latest = g.iloc[-1]
    today_cand = classify_candle(
        float(latest["o"]), float(latest["h"]),
        float(latest["l"]), float(latest["c"])
    )
    today_key  = today_cand["key"]
    today_hist = next((p for p in pattern_stats if p["candle_key"]==today_key), None)

    return {
        "sym":           sym,
        "patterns":      pattern_stats,
        "today_candle":  today_cand,
        "today_key":     today_key,
        "today_date":    str(latest["date"].date()),
        "today_close":   round(float(latest["c"]), 2),
        "today_hist":    today_hist,   # historical stats for today's pattern
        "n_patterns":    len(pattern_stats),
        "n_100pct":      sum(1 for p in pattern_stats if p["is_100pct_nd"]),
        "n_meets_15pct": sum(1 for p in pattern_stats if p["meets_15pct"]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Candle Pattern Tracker")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Manifest
    print("\n[1] Manifest...")
    with open(MANIFEST) as f: manifest = json.load(f)
    tds        = sorted(manifest.keys())
    latest_str = tds[-1]
    print(f"  {len(tds)} days [{tds[0]} -> {latest_str}]")

    # Load
    print("\n[2] Loading data...")
    df = load_all(manifest)
    print(f"  {df['sym'].nunique():,} symbols")

    # Filter stocks with enough data
    counts     = df.groupby("sym")["date"].count()
    valid_syms = counts[counts >= 100].index
    df         = df[df["sym"].isin(valid_syms)].copy()
    sym_list   = sorted(df["sym"].unique())
    print(f"  {len(sym_list)} stocks with >= 100 trading days")

    # Analyse each stock
    print("\n[3] Analysing candle patterns per stock...")
    sym_grps     = {s:g.copy() for s,g in df.groupby("sym")}
    all_profiles = {}
    best_patterns= []   # 100% next-day win + meets 15%
    today_alerts = []   # stocks where today's candle matches a strong historical pattern

    for i, sym in enumerate(sym_list):
        try:
            result = analyse_stock_candles(sym_grps[sym], sym)
            if not result: continue
            all_profiles[sym] = {
                "sym":           result["sym"],
                "n_patterns":    result["n_patterns"],
                "n_100pct":      result["n_100pct"],
                "n_meets_15pct": result["n_meets_15pct"],
                "today_candle":  result["today_candle"],
                "today_date":    result["today_date"],
                "today_close":   result["today_close"],
                "today_hist":    result["today_hist"],
                "patterns":      result["patterns"],
            }

            # Collect best patterns (100% next-day + ≥15% in any window)
            for pat in result["patterns"]:
                if pat["is_100pct_nd"] and pat["meets_15pct"]:
                    best_patterns.append({**pat, "sym": sym})

            # Today alert: today's candle matches a historically strong pattern
            th = result["today_hist"]
            if th and th["nd_win_rate"] >= MIN_WIN_NEXT_DAY:
                today_alerts.append({
                    "sym":         sym,
                    "today_date":  result["today_date"],
                    "today_close": result["today_close"],
                    "candle_key":  result["today_key"],
                    "candle_desc": result["today_candle"]["desc"],
                    "nd_win_rate": th["nd_win_rate"],
                    "nd_avg_ret":  th["nd_avg_ret"],
                    "nd_max_ret":  th["nd_max_ret"],
                    "gap_up_rate": th["gap_up_rate"],
                    "occurrences": th["occurrences"],
                    "is_100pct":   th["is_100pct_nd"],
                    "meets_15pct": th["meets_15pct"],
                    "max_avg_any": th["max_avg_any_window"],
                    "win_5d":      th["win_5d"],
                    "win_10d":     th["win_10d"],
                    "win_20d":     th["win_20d"],
                    "years":       th["years"],
                })
        except Exception as e:
            pass
        if (i+1) % 200 == 0:
            print(f"    {i+1}/{len(sym_list)}, best:{len(best_patterns)}, alerts:{len(today_alerts)}...")

    del sym_grps; gc.collect()

    # Sort
    best_patterns.sort(key=lambda x:(-x["nd_win_rate"],-x["max_avg_any_window"],-x["occurrences"]))
    today_alerts.sort(key=lambda x:(-x["nd_win_rate"],-x["nd_avg_ret"]))

    n_100 = sum(1 for s in all_profiles.values() if s["n_100pct"] > 0)
    print(f"\n  {len(all_profiles)} stocks analysed")
    print(f"  {n_100} stocks have at least one 100%-next-day pattern")
    print(f"  {len(best_patterns)} best patterns (100% next-day + ≥15% any window)")
    print(f"  {len(today_alerts)} today alerts")

    # Write outputs
    print("\n[4] Writing outputs...")
    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    # Per-stock profiles (searchable by symbol in the UI)
    jdump({
        "generated_at": now_ist,
        "latest_date":  latest_str,
        "n_stocks":     len(all_profiles),
        "profiles":     all_profiles,
    }, OUT_DIR/"candle_stock_patterns.json")
    print(f"  OK candle_stock_patterns.json ({len(all_profiles)} stocks)")

    # Best patterns (100% next-day + ≥15% in any window)
    jdump({
        "generated_at":   now_ist,
        "latest_date":    latest_str,
        "n_patterns":     len(best_patterns),
        "description":    "Candle patterns with 100% next-day positive rate AND avg ≥15% in at least one of 5/10/20 days",
        "patterns":       best_patterns[:500],
    }, OUT_DIR/"candle_best_patterns.json")
    print(f"  OK candle_best_patterns.json ({len(best_patterns)} patterns)")

    # Today's alerts
    jdump({
        "generated_at":  now_ist,
        "signal_date":   latest_str,
        "total_alerts":  len(today_alerts),
        "perfect_alerts":sum(1 for a in today_alerts if a["is_100pct"]),
        "alerts":        today_alerts,
    }, OUT_DIR/"candle_today.json")
    print(f"  OK candle_today.json ({len(today_alerts)} alerts)")
    print("\nDone.")

if __name__ == "__main__":
    main()
