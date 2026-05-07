#!/usr/bin/env python3
"""
bounce_analyzer.py
==================
Finds stocks where, after reaching a historical support level (price low) 
within ±5%, the stock consistently bounced up by a minimum % over the next
1 week (5td), 2 weeks (10td), 1 month (20td), 2 months (40td), 3 months (60td).

Also detects candle patterns at the support touch:
  Hammer, Bullish Engulfing, Doji, Morning Star, Piercing Line

Daily alert: if today's price is within ±5% of a proven support level.

Output: stock_analysis/bounce_signals.json
"""
import json, sys, warnings, math
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "equity"
OUT      = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

MIN_PRICE      = 10.0
MIN_TURNOVER   = 3_000_000   # 3 Cr last day
MIN_OCC        = 3
MIN_BOUNCE     = 8.0         # minimum 8% bounce to qualify
SUPPORT_WINDOW = 60          # look back 60td to find local lows
PRICE_BAND     = 5.0         # ±5% from support level
HOLD_PERIODS   = [5, 10, 20, 40, 60]  # trading days
HOLD_LABELS    = {5:"1 Week", 10:"2 Weeks", 20:"1 Month", 40:"2 Months", 60:"3 Months"}

EXCLUDED_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
cur_yr = datetime.now().year

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v,2)
    except: return None

def load_all():
    if not MANIFEST.exists(): print("ERROR: manifest.json missing"); sys.exit(1)
    manifest = json.loads(MANIFEST.read_text())
    dates = sorted(manifest.keys())
    print(f"  Loading {len(dates)} dates...")
    frames = []
    for ds in dates:
        y,mo,_ = ds.split("-")
        p = DATA/y/mo/f"{ds}.csv"
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

# ── CANDLE PATTERN DETECTION ──────────────────────────────────────────────────
def detect_candle(o_arr, h_arr, l_arr, c_arr, idx):
    """Detect bullish candle patterns at index idx."""
    patterns = []
    if idx < 2: return patterns
    o0,h0,l0,c0 = o_arr[idx],  h_arr[idx],  l_arr[idx],  c_arr[idx]
    o1,h1,l1,c1 = o_arr[idx-1],h_arr[idx-1],l_arr[idx-1],c_arr[idx-1]

    body0 = abs(c0-o0); rng0 = h0-l0 if h0!=l0 else 0.001
    body1 = abs(c1-o1)

    lo_wick0 = min(o0,c0)-l0
    up_wick0 = h0-max(o0,c0)
    lo_wick1 = min(o1,c1)-l1

    # Hammer: small body, long lower wick >= 2x body, upper wick < 0.3x body
    if body0>0 and lo_wick0 >= 2*body0 and up_wick0 <= 0.3*body0:
        patterns.append("Hammer")

    # Bullish Engulfing: prev candle red, today green, today body engulfs prev body
    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        patterns.append("Bullish Engulfing")

    # Doji: very small body relative to range
    if rng0 > 0 and body0/rng0 < 0.1:
        patterns.append("Doji")

    # Piercing Line: prev red, today opens below prev low, closes above midpoint of prev body
    if c1 < o1 and c0 > o0 and o0 < l1 and c0 > (o1+c1)/2:
        patterns.append("Piercing Line")

    # Morning Star: 3-candle — prev2 big red, prev1 small body (gap), today big green
    if idx >= 2:
        o2,h2,l2,c2 = o_arr[idx-2],h_arr[idx-2],l_arr[idx-2],c_arr[idx-2]
        body2 = abs(c2-o2)
        if c2<o2 and body2>0 and body1<body2*0.5 and c0>o0 and c0>(o2+c2)/2:
            patterns.append("Morning Star")

    return patterns if patterns else ["No pattern"]

# ── FIND SUPPORT LEVELS ───────────────────────────────────────────────────────
def find_support_levels(c, l, n):
    """
    Find significant local lows (support levels) across the price history.
    A local low is a price where the stock touched a level at least 3 times.
    Returns list of support price levels.
    """
    supports = []
    # Rolling min over SUPPORT_WINDOW — find local minima
    for i in range(SUPPORT_WINDOW, n-SUPPORT_WINDOW):
        local_min = float(l.iloc[max(0,i-SUPPORT_WINDOW):i].min())
        # Check if this is a fresh local low (not too close to previous ones)
        if not supports or min(abs(local_min - s) / s * 100 for s in supports) > PRICE_BAND:
            # Count how many times price came within PRICE_BAND of this level
            touches = sum(1 for j in range(n) if abs(float(l.iloc[j]) - local_min) / local_min * 100 <= PRICE_BAND)
            if touches >= MIN_OCC:
                supports.append(local_min)
    return supports

# ── ANALYZE ONE STOCK ─────────────────────────────────────────────────────────
def analyze_stock(sym, df):
    n = len(df)
    o = df["o"].values; h = df["h"].values
    l = df["l"].values; c = df["c"].values
    v = df["v"].values
    dates = pd.to_datetime(df["date"].values)

    if float(c[-1]) < MIN_PRICE: return None
    if float(c[-1]) * float(v[-1]) < MIN_TURNOVER: return None

    supports = find_support_levels(df["c"], df["l"], n)
    if not supports: return None

    # For each support level, find all touches and subsequent bounces
    level_results = []
    for support in supports:
        touches = []
        i = 0
        while i < n - max(HOLD_PERIODS) - 1:
            # Is today's low within ±5% of support?
            pct_from_sup = (float(l[i]) - support) / support * 100
            if abs(pct_from_sup) <= PRICE_BAND:
                candles = detect_candle(o, h, l, c, i)
                touch = {
                    "date":         str(dates[i].date()),
                    "price":        r2(float(c[i])),
                    "low":          r2(float(l[i])),
                    "support_level":r2(support),
                    "pct_from_sup": r2(pct_from_sup),
                    "candle":       candles,
                    "year":         int(dates[i].year),
                    "bounces":      {}
                }
                # Measure bounce for each hold period
                for hold in HOLD_PERIODS:
                    xi = i + hold
                    if xi < n:
                        ep = float(o[i+1]) if i+1<n else float(c[i])
                        if ep > 0:
                            # Return from entry (next day open) to exit close
                            ret = (float(c[xi]) - ep) / ep * 100
                            # Max gain in window
                            max_h = float(max(h[i+1:xi+1])) if xi>i+1 else float(h[i])
                            touch["bounces"][hold] = {
                                "ret":      r2(ret),
                                "max_gain": r2((max_h - ep)/ep*100),
                                "exit_px":  r2(float(c[xi])),
                                "entry_px": r2(ep),
                            }
                touches.append(touch)
                # Skip forward to avoid overlapping
                i += max(5, int(SUPPORT_WINDOW/4))
            else:
                i += 1

        if len(touches) < MIN_OCC: continue

        # For each hold period, check if bounces are consistent
        for hold in HOLD_PERIODS:
            rets = [t["bounces"].get(hold,{}).get("ret") for t in touches]
            rets = [r for r in rets if r is not None]
            if len(rets) < MIN_OCC: continue
            if not all(r >= MIN_BOUNCE for r in rets): continue

            avg_ret  = r2(sum(rets)/len(rets))
            min_ret  = r2(min(rets))
            max_rets = [t["bounces"].get(hold,{}).get("max_gain") for t in touches]
            max_rets = [r for r in max_rets if r is not None]
            avg_max  = r2(sum(max_rets)/len(max_rets)) if max_rets else None

            # Find most common candle pattern
            all_candles = []
            for t in touches:
                all_candles.extend(t.get("candle",[]))
            from collections import Counter
            candle_counts = Counter(c for c in all_candles if c != "No pattern")
            dominant_candle = candle_counts.most_common(1)[0][0] if candle_counts else "None"

            level_results.append({
                "support_level":   r2(support),
                "hold_days":       hold,
                "hold_label":      HOLD_LABELS[hold],
                "n_touches":       len(touches),
                "avg_bounce":      avg_ret,
                "min_bounce":      min_ret,
                "avg_max_gain":    avg_max,
                "ret_rate":        r2(avg_ret/hold) if avg_ret else None,
                "dominant_candle": dominant_candle,
                "candle_counts":   dict(candle_counts.most_common(5)),
                "pct_above_support": r2(sum(1 for t in touches if t["pct_from_sup"]>0)/len(touches)*100),
                "touches":         sorted(touches, key=lambda x: x["date"], reverse=True),
                "alert_today":     abs(float(c[-1]) - support)/support*100 <= PRICE_BAND,
                "current_pct_from_sup": r2((float(c[-1]) - support)/support*100),
            })

    if not level_results: return None

    # Sort by avg_bounce descending
    level_results.sort(key=lambda x: -(x.get("avg_bounce") or 0))
    return level_results

def main():
    print(f"\n{'='*60}\nBounce/Support Analyzer\nStarted: {now}\n{'='*60}")
    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results  = []; skipped = 0; excluded = 0
    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} stocks")
        if any(sym.upper().endswith(s) for s in EXCLUDED_SFX):
            excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 200: skipped += 1; continue
            levels = analyze_stock(sym, grp)
            if levels:
                results.append({
                    "sym":   sym,
                    "price": r2(float(grp["c"].iloc[-1])),
                    "levels": levels,
                    "n_levels": len(levels),
                    "best_avg_bounce": levels[0]["avg_bounce"] if levels else None,
                    "alert_today": any(lv["alert_today"] for lv in levels),
                    "alert_levels": [lv for lv in levels if lv["alert_today"]],
                })
        except: skipped += 1

    # Sort by best avg bounce
    results.sort(key=lambda x: -(x.get("best_avg_bounce") or 0))

    # Build alert list (today's price near a support level)
    alerts_today = [r for r in results if r.get("alert_today")]
    alerts_today.sort(key=lambda x: -(x.get("best_avg_bounce") or 0))

    output = {
        "generated_at":    now,
        "today_date":      datetime.now().strftime("%Y-%m-%d"),
        "n_stocks":        len(results),
        "n_alerts_today":  len(alerts_today),
        "alerts_today":    alerts_today,
        "description":     f"Stocks where touching a support level (±{PRICE_BAND}%) led to ≥{MIN_BOUNCE}% bounce in 100% of cases, across hold periods: {HOLD_LABELS}",
        "hold_periods":    HOLD_LABELS,
        "stocks":          results,
    }

    path = OUT / "bounce_signals.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written {len(results)} stocks, {len(alerts_today)} alerts today")
    if alerts_today:
        print(f"\n*** {len(alerts_today)} STOCKS NEAR SUPPORT TODAY ***")
        for r in alerts_today[:10]:
            al = r["alert_levels"][0]
            print(f"  {r['sym']:<14} price={r['price']} support={al['support_level']} ({al['current_pct_from_sup']:+.1f}%) avg_bounce={al['avg_bounce']}%")

if __name__ == "__main__":
    main()
