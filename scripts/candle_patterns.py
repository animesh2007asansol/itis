#!/usr/bin/env python3
"""
candle_patterns.py  v2
=======================
For every stock over 5 years of daily data:
  1. Classify each candle into a shape bucket:
       color (green/red) x body_size x lower_wick_ratio x upper_wick_ratio
  2. For each candle shape, track next-day return across ALL historical occurrences
  3. Keep ONLY patterns where EVERY single occurrence gave positive next-day return
     (100% win rate — if even one occurrence was negative, pattern is excluded)
  4. Rank all such patterns by minimum next-day return (highest minimum first)
     so the top pattern is the one with the best "guaranteed floor" return
  5. For live alerts: match today's candle against each stock's 100%-win patterns
     and alert with min/avg/max historical next-day returns

No minimum return threshold — just 100% positive every time.
Volume filter: >= 50,000 shares on signal day.

Outputs → pattern_signals/:
  candle_stock_patterns.json   — per-stock all pattern stats (searchable)
  candle_best_patterns.json    — only 100%-win patterns, ranked by min return
  candle_today.json            — today's candle for each stock + historical match
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
CHECKPOINT= OUT_DIR  / "candle_checkpoint.json"

MIN_VOLUME       = 50_000    # minimum tradeable volume on signal day
MIN_OCC          = 3         # minimum occurrences of a candle pattern to compute stats
MIN_TRADING_DAYS = 100       # minimum history per stock

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
# COLUMN ALIASES (handles both old and new NSE CSV formats)
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
        hdr = [h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_sym=_fc(hdr,SYM_A); i_ser=_fc(hdr,SER_A); i_o=_fc(hdr,O_A)
        i_h=_fc(hdr,H_A);     i_l=_fc(hdr,L_A);    i_c=_fc(hdr,C_A); i_v=_fc(hdr,V_A)
        if i_sym<0 or i_c<0: return rows
        mc = max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v] if x>=0)
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

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT — incremental: only reload if new dates exist
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
# LOAD DATA (all trading days)
# ─────────────────────────────────────────────────────────────────────────────
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
# CANDLE CLASSIFICATION
# Body size: tiny(<0.3%) / small(0.3-1%) / medium(1-2.5%) / large(2.5-5%) / huge(>5%)
# Wick ratio vs body: none(<0.1x) / tiny(0.1-0.5x) / small(0.5-1x) / medium(1-2x) / long(2-4x) / very_long(>4x)
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
    body       = abs(c-o)
    upper_wick = h - max(o,c)
    lower_wick = min(o,c) - l
    safe_body  = body if body>0 else 0.0001
    body_pct   = body/c*100 if c>0 else 0
    color      = "G" if c>=o else "R"
    bb         = body_bkt(body_pct)
    ub         = wick_bkt(upper_wick/safe_body)
    lb         = wick_bkt(lower_wick/safe_body)
    key        = f"{color}_{bb}_L{lb}_U{ub}"
    desc       = f"{'Green' if color=='G' else 'Red'} {bb} body | lower:{lb} | upper:{ub}"
    return key, desc, round(body_pct,2), round(upper_wick/safe_body,2), round(lower_wick/safe_body,2)

# ─────────────────────────────────────────────────────────────────────────────
# PER-STOCK CANDLE ANALYSIS — vectorised
# ─────────────────────────────────────────────────────────────────────────────
def analyse_stock(g, sym):
    g = g.copy().sort_values("date").reset_index(drop=True)
    n = len(g)
    if n < MIN_TRADING_DAYS: return None

    c,o,h,l,v = g["c"],g["o"],g["h"],g["l"],g["v"]

    # Forward returns (vectorised)
    g["fwd1"]  = c.shift(-1)/c - 1   # next-day return
    g["fwd5"]  = c.shift(-5)/c - 1
    g["fwd10"] = c.shift(-10)/c - 1
    g["fwd20"] = c.shift(-20)/c - 1
    g["next_o"]= o.shift(-1)         # next-day open (for gap-up rate)
    g["year"]  = g["date"].dt.year

    # Classify each candle
    keys,descs,bpcts,urats,lrats = [],[],[],[],[]
    for i in range(n):
        k,d,bp,ur,lr = classify(float(o.iloc[i]),float(h.iloc[i]),float(l.iloc[i]),float(c.iloc[i]))
        keys.append(k); descs.append(d); bpcts.append(bp); urats.append(ur); lrats.append(lr)
    g["ckey"]=keys; g["cdesc"]=descs

    # Only tradeable rows
    tradeable = g[g["v"]>=MIN_VOLUME].copy()
    if len(tradeable)<5: return None

    latest_date  = g["date"].max()
    latest_idx   = g.index[-1]
    today_row    = g.iloc[-1]
    today_key, today_desc, today_bp, today_ur, today_lr = classify(
        float(today_row["o"]),float(today_row["h"]),float(today_row["l"]),float(today_row["c"])
    )

    # Build per-candle-key stats
    pattern_stats = []
    desc_map = dict(zip(keys, descs))

    for ckey, grp in tradeable.groupby("ckey"):
        # Next-day stats (requires future data — dropna removes last row)
        nd = grp["fwd1"].dropna()
        if len(nd) < MIN_OCC: continue

        n_total = len(nd)
        n_pos   = int((nd>0).sum())
        n_neg   = int((nd<0).sum())
        nd_wr   = round(n_pos/n_total*100, 1)

        # STRICT RULE: if ANY occurrence was negative → discard pattern
        if n_neg > 0:
            continue  # not 100% win rate

        nd_avg  = round(float(nd.mean()*100), 2)
        nd_min  = round(float(nd.min()*100),  2)   # worst (but still positive) occurrence
        nd_max  = round(float(nd.max()*100),  2)   # best occurrence

        # Gap-up rate (next day opens higher than today close)
        gap = grp["next_o"].dropna()
        if len(gap) >= MIN_OCC:
            gap_pct = (gap/grp.loc[gap.index,"c"]-1)*100
            gap_up_rate = round(float((gap_pct>0).sum()/len(gap)*100), 1)
            avg_gap     = round(float(gap_pct.mean()), 2)
        else:
            gap_up_rate = 0.0; avg_gap = 0.0

        # Multi-day stats
        wstats = {}
        for col,w in [("fwd5",5),("fwd10",10),("fwd20",20)]:
            wv = grp[col].dropna()
            if len(wv)>=MIN_OCC:
                wstats[str(w)] = {
                    "n":   int(len(wv)),
                    "wr":  round(float((wv>0).sum()/len(wv)*100),1),
                    "avg": round(float(wv.mean()*100),2),
                    "min": round(float(wv.min()*100),2),
                    "max": round(float(wv.max()*100),2),
                }

        # Years this pattern appeared
        years = sorted(int(y) for y in grp["date"].dt.year.unique().tolist())

        pattern_stats.append({
            "sym":          sym,
            "candle_key":   ckey,
            "desc":         desc_map.get(ckey, ckey),
            "occurrences":  n_total,
            "nd_wr":        100.0,       # always 100% (filtered above)
            "nd_avg":       nd_avg,
            "nd_min":       nd_min,      # LOWEST positive return — used for ranking
            "nd_max":       nd_max,
            "gap_up_rate":  gap_up_rate,
            "avg_gap_pct":  avg_gap,
            "win_5d":       wstats.get("5"),
            "win_10d":      wstats.get("10"),
            "win_20d":      wstats.get("20"),
            "years":        years,
            "last_date":    str(grp["date"].max().date()),
        })

    if not pattern_stats: return None

    # Sort by nd_min DESCENDING — highest guaranteed floor return first
    pattern_stats.sort(key=lambda x: -x["nd_min"])

    # Today's candle match
    today_hist = next((p for p in pattern_stats if p["candle_key"]==today_key), None)

    return {
        "sym":         sym,
        "today_date":  str(latest_date.date()),
        "today_close": round(float(today_row["c"]),2),
        "today_key":   today_key,
        "today_candle":{"key":today_key,"desc":today_desc,"body_pct":today_bp,
                        "upper_wick_ratio":today_ur,"lower_wick_ratio":today_lr},
        "today_hist":  today_hist,
        "n_patterns":  len(pattern_stats),
        "patterns":    pattern_stats,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Candle Pattern Tracker  v2")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Manifest...")
    with open(MANIFEST) as f: manifest=json.load(f)
    tds        = sorted(manifest.keys())
    latest_str = tds[-1]
    print(f"  {len(tds)} days [{tds[0]} -> {latest_str}]")

    # Checkpoint
    cp        = load_cp()
    last_run  = cp.get("last_run_date","")
    force     = (os.environ.get("FORCE_FULL_RERUN","").lower()=="true"
                 if __import__("os").environ.get("FORCE_FULL_RERUN") else False)
    if last_run == latest_str and not force:
        print(f"  Already ran for {latest_str}. Set FORCE_FULL_RERUN=true to rerun.")
        # Still write today's signals since window may have shifted
        # Just skip the expensive full load — reuse saved data
        # If files exist, just exit
        if (OUT_DIR/"candle_best_patterns.json").exists():
            print("  Existing files are current. Done.")
            return

    print("\n[2] Loading data...")
    import os
    df = load_all(manifest)
    counts    = df.groupby("sym")["date"].count()
    valid_sym = counts[counts>=MIN_TRADING_DAYS].index
    df        = df[df["sym"].isin(valid_sym)].copy()
    sym_list  = sorted(df["sym"].unique())
    print(f"  {len(sym_list)} stocks with >={MIN_TRADING_DAYS} trading days")

    print("\n[3] Analysing candle patterns per stock...")
    sym_grps     = {s:g.copy() for s,g in df.groupby("sym")}
    all_profiles = {}
    best_pats    = []   # 100% next-day win across ALL occurrences
    today_alerts = []   # today's candle matches a 100%-win historical pattern

    for i, sym in enumerate(sym_list):
        try:
            res = analyse_stock(sym_grps[sym], sym)
            if not res: continue
            all_profiles[sym] = res

            # Collect best patterns (100% win, all occurrences positive)
            for pat in res["patterns"]:
                best_pats.append({**pat,"sym":sym})

            # Today alert
            th = res["today_hist"]
            if th:
                today_alerts.append({
                    "sym":          sym,
                    "today_date":   res["today_date"],
                    "today_close":  res["today_close"],
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
        except Exception as e:
            pass
        if (i+1)%200==0:
            print(f"    {i+1}/{len(sym_list)}, best:{len(best_pats)}, alerts:{len(today_alerts)}...")

    del sym_grps; gc.collect()

    # Sort best patterns by nd_min DESCENDING — highest guaranteed floor first
    best_pats.sort(key=lambda x: -x["nd_min"])
    # Sort today alerts by nd_min DESCENDING
    today_alerts.sort(key=lambda x: -x["nd_min"])

    print(f"\n  {len(all_profiles)} stocks analysed")
    print(f"  {len(best_pats)} 100%-win candle patterns found")
    print(f"  {len(today_alerts)} today alerts")

    print("\n[4] Writing outputs...")
    import os as _os
    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    # Per-stock profiles (full data, searchable)
    jdump({"generated_at":now_ist,"latest_date":latest_str,
           "n_stocks":len(all_profiles),"profiles":all_profiles},
          OUT_DIR/"candle_stock_patterns.json")
    print(f"  OK candle_stock_patterns.json ({len(all_profiles)} stocks)")

    # Best patterns — 100% next-day win, ranked by nd_min
    jdump({"generated_at":now_ist,"latest_date":latest_str,
           "n_patterns":len(best_pats),
           "description":"100% next-day positive in ALL occurrences. Ranked by minimum return (highest first).",
           "patterns":best_pats[:1000]},
          OUT_DIR/"candle_best_patterns.json")
    print(f"  OK candle_best_patterns.json ({len(best_pats)} patterns)")

    # Today's signals
    jdump({"generated_at":now_ist,"signal_date":latest_str,
           "total_alerts":len(today_alerts),
           "alerts":today_alerts},
          OUT_DIR/"candle_today.json")
    print(f"  OK candle_today.json ({len(today_alerts)} alerts)")

    save_cp({"last_run_date":latest_str,"run_at":now_ist,
             "stocks_analysed":len(all_profiles),"best_patterns":len(best_pats)})
    print("  OK candle_checkpoint.json")
    print(f"\nDone.")

if __name__=="__main__":
    main()
