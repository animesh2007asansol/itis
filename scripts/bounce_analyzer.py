#!/usr/bin/env python3
"""
bounce_analyzer.py
==================
Finds stocks where, after reaching a historical support level (pivot low) 
within ±5%, the stock consistently bounced up by a minimum % over the next
1 week (5td), 2 weeks (10td), 1 month (20td), 2 months (40td), 3 months (60td).

Support levels are found using PIVOT LOWS — actual price levels where the stock 
touched a low point surrounded by higher prices on both sides, and returned to 
that same level multiple times. This ensures the support level is always a real 
traded price.

Also detects candle patterns: Hammer, Bullish Engulfing, Doji, Morning Star, Piercing Line.
Daily alert: if today's price is within ±5% of a proven support level.

Output: stock_analysis/bounce_signals.json
"""
import json, sys, warnings, math
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "equity"
OUT      = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

MIN_PRICE      = 10.0
MIN_TURNOVER   = 3_000_000    # Rs 3 Cr last day turnover
MIN_RETURN     = 8.0          # minimum bounce %
MIN_TOUCHES    = 3            # minimum times price returned to support
PIVOT_WINDOW   = 10           # N days on each side for pivot detection
PRICE_BAND     = 5.0          # ±5% counts as "near support"
HOLD_PERIODS   = [5, 10, 20, 40, 60]
HOLD_LABELS    = {5:"1 Week", 10:"2 Weeks", 20:"1 Month", 40:"2 Months", 60:"3 Months"}
# Only consider support levels within this range of current price
# (avoids showing support from years ago when stock was at completely different price)
MAX_DIST_FROM_CUR = 60.0      # support must be within 60% of current price

EXCLUDED_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing"); sys.exit(1)
    manifest = json.loads(MANIFEST.read_text())
    dates = sorted(manifest.keys())
    print(f"  Loading {len(dates)} dates...")
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
                if u in ("SYMBOL","TCKRSYMB"):               cm[c] = "sym"
                elif u in ("SERIES","SCTYSRS"):              cm[c] = "series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):   cm[c] = "o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):   cm[c] = "h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):      cm[c] = "l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"): cm[c] = "c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):       cm[c] = "v"
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


def detect_candle(o_arr, h_arr, l_arr, c_arr, idx):
    """Detect bullish candle patterns at index idx."""
    if idx < 2: return ["No pattern"]
    o0,h0,l0,c0 = o_arr[idx],   h_arr[idx],   l_arr[idx],   c_arr[idx]
    o1,h1,l1,c1 = o_arr[idx-1], h_arr[idx-1], l_arr[idx-1], c_arr[idx-1]
    body0 = abs(c0-o0)
    rng0  = h0-l0 if h0!=l0 else 0.001
    lo_w0 = min(o0,c0)-l0
    up_w0 = h0-max(o0,c0)

    patterns = []

    # Hammer: small body top, long lower wick ≥ 2× body, tiny upper wick
    if body0 > 0 and lo_w0 >= 2*body0 and up_w0 <= 0.3*body0:
        patterns.append("Hammer")

    # Bullish Engulfing: prev red, today green, today body fully engulfs prev
    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        patterns.append("Bullish Engulfing")

    # Doji: tiny body relative to range
    if rng0 > 0 and body0/rng0 < 0.1:
        patterns.append("Doji")

    # Piercing Line: prev red, today opens below prev low, closes above prev midpoint
    if c1 < o1 and c0 > o0 and o0 < l1 and c0 > (o1+c1)/2:
        patterns.append("Piercing Line")

    # Morning Star (3-candle)
    if idx >= 2:
        o2,c2 = o_arr[idx-2], c_arr[idx-2]
        body2 = abs(c2-o2); body1 = abs(c1-o1)
        if c2 < o2 and body2 > 0 and body1 < body2*0.5 and c0 > o0 and c0 > (o2+c2)/2:
            patterns.append("Morning Star")

    return patterns if patterns else ["No pattern"]


def find_pivot_lows(l_arr, n, pw=PIVOT_WINDOW):
    """
    Find genuine pivot lows: index i where l[i] is the minimum of
    the window [i-pw : i+pw+1] — i.e., lower than pw days on both sides.
    These are REAL traded price levels, not rolling computations.
    """
    pivots = []
    for i in range(pw, n - pw):
        window = l_arr[i-pw:i+pw+1]
        if len(window) < 2*pw+1: continue
        if float(l_arr[i]) <= float(np.min(window)) + 0.001:
            pivots.append((i, float(l_arr[i])))
    return pivots


def cluster_supports(pivots, price_band=PRICE_BAND):
    """
    Cluster nearby pivot lows into support zones.
    Two pivots belong to the same zone if within price_band% of each other.
    Returns list of (representative_price, [pivot_indices]).
    """
    if not pivots: return []
    # Sort by price
    sorted_p = sorted(pivots, key=lambda x: x[1])
    clusters = []
    cur_cluster = [sorted_p[0]]

    for i in range(1, len(sorted_p)):
        ref_price = cur_cluster[0][1]  # anchor on first item, not last
        this_price = sorted_p[i][1]
        if abs(this_price - ref_price) / ref_price * 100 <= price_band:
            cur_cluster.append(sorted_p[i])
        else:
            clusters.append(cur_cluster)
            cur_cluster = [sorted_p[i]]
    clusters.append(cur_cluster)

    result = []
    for cl in clusters:
        prices = [p[1] for p in cl]
        # Representative price = median of cluster
        rep_price = float(np.median(prices))
        indices   = [p[0] for p in cl]
        result.append((rep_price, indices))

    return result


def analyze_stock(sym, df):
    n = len(df)
    o = df["o"].values; h = df["h"].values
    l = df["l"].values; c = df["c"].values
    v = df["v"].values
    dates = pd.to_datetime(df["date"].values)

    cur_price = float(c[-1])
    if cur_price < MIN_PRICE: return None
    # Last-day turnover check
    if cur_price * float(v[-1]) < MIN_TURNOVER: return None

    # Step 1: Find actual pivot lows
    pivots = find_pivot_lows(l, n)
    if not pivots: return None

    # Step 2: Cluster into support zones
    clusters = cluster_supports(pivots)

    level_results = []

    for (support, pivot_idxs) in clusters:
        # CRITICAL: support must be within MAX_DIST_FROM_CUR of current price
        # This prevents showing Rs 2260 support for a stock now at Rs 400
        pct_dist = abs(cur_price - support) / support * 100
        if pct_dist > MAX_DIST_FROM_CUR: continue

        # Also verify: the support price must be a real price the stock traded at
        # (i.e., at least one actual low price within 5% of the support level)
        real_lows_near = [float(l[i]) for i in range(n)
                          if abs(float(l[i]) - support) / support * 100 <= PRICE_BAND]
        if not real_lows_near: continue

        # Find all times price came within ±5% of support (not just pivot days)
        touches_all = []
        i = 0
        while i < n - max(HOLD_PERIODS) - 1:
            pct_from_sup = (float(l[i]) - support) / support * 100
            if abs(pct_from_sup) <= PRICE_BAND:
                candles = detect_candle(o, h, l, c, i)
                touch = {
                    "date":          str(dates[i].date()),
                    "price":         r2(float(c[i])),
                    "low":           r2(float(l[i])),
                    "support_level": r2(support),
                    "pct_from_sup":  r2(pct_from_sup),
                    "candle":        candles,
                    "year":          int(dates[i].year),
                    "bounces":       {}
                }
                # Measure bounce for each hold period
                for hold in HOLD_PERIODS:
                    xi = i + hold
                    if xi < n:
                        ep = float(o[i+1]) if i+1 < n else float(c[i])
                        if ep > 0:
                            ret      = (float(c[xi]) - ep) / ep * 100
                            max_h    = float(max(h[i+1:xi+1])) if xi > i+1 else float(h[i])
                            max_gain = (max_h - ep) / ep * 100
                            touch["bounces"][hold] = {
                                "ret":      r2(ret),
                                "max_gain": r2(max_gain),
                                "exit_px":  r2(float(c[xi])),
                                "entry_px": r2(ep),
                            }
                touches_all.append(touch)
                # Skip forward by min hold period to avoid overlapping touches
                i += min(HOLD_PERIODS)
            else:
                i += 1

        if len(touches_all) < MIN_TOUCHES: continue

        # For each hold period: require 100% of touches gave >= MIN_RETURN
        for hold in HOLD_PERIODS:
            rets = [t["bounces"].get(hold, {}).get("ret") for t in touches_all]
            rets = [r for r in rets if r is not None]
            if len(rets) < MIN_TOUCHES: continue
            if not all(r >= MIN_RETURN for r in rets): continue

            avg_ret  = r2(sum(rets) / len(rets))
            min_ret  = r2(min(rets))
            max_gs   = [t["bounces"].get(hold, {}).get("max_gain") for t in touches_all]
            max_gs   = [g for g in max_gs if g is not None]
            avg_max  = r2(sum(max_gs) / len(max_gs)) if max_gs else None

            # Dominant candle pattern
            all_candles = []
            for t in touches_all:
                all_candles.extend(pat for pat in t.get("candle", []) if pat != "No pattern")
            cc = Counter(all_candles)
            dominant_candle = cc.most_common(1)[0][0] if cc else "None"

            # Alert: is current price near this support?
            cur_pct = r2((cur_price - support) / support * 100)
            alert   = abs(cur_price - support) / support * 100 <= PRICE_BAND

            level_results.append({
                "support_level":        r2(support),
                "hold_days":            hold,
                "hold_label":           HOLD_LABELS[hold],
                "n_touches":            len(touches_all),
                "avg_bounce":           avg_ret,
                "min_bounce":           min_ret,
                "avg_max_gain":         avg_max,
                "ret_rate":             r2(avg_ret / hold) if avg_ret else None,
                "dominant_candle":      dominant_candle,
                "candle_counts":        dict(cc.most_common(5)),
                "pct_above_support":    r2(sum(1 for t in touches_all if t["pct_from_sup"] > 0) / len(touches_all) * 100),
                "touches":              sorted(touches_all, key=lambda x: x["date"], reverse=True),
                "alert_today":          alert,
                "current_pct_from_sup": cur_pct,
            })

    if not level_results: return None
    level_results.sort(key=lambda x: -(x.get("avg_bounce") or 0))
    return level_results


def main():
    print(f"\n{'='*60}\nBounce/Support Analyzer\nStarted: {now}\n{'='*60}")
    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results = []; skipped = 0; excluded = 0
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
                    "sym":             sym,
                    "price":           r2(float(grp["c"].iloc[-1])),
                    "levels":          levels,
                    "n_levels":        len(levels),
                    "best_avg_bounce": levels[0]["avg_bounce"] if levels else None,
                    "alert_today":     any(lv["alert_today"] for lv in levels),
                    "alert_levels":    [lv for lv in levels if lv["alert_today"]],
                })
        except: skipped += 1

    results.sort(key=lambda x: -(x.get("best_avg_bounce") or 0))
    alerts_today = sorted([r for r in results if r.get("alert_today")],
                          key=lambda x: -(x.get("best_avg_bounce") or 0))

    output = {
        "generated_at":   now,
        "today_date":     datetime.now().strftime("%Y-%m-%d"),
        "n_stocks":       len(results),
        "n_alerts_today": len(alerts_today),
        "alerts_today":   alerts_today,
        "description":    (f"Stocks where touching a pivot support level (±{PRICE_BAND}%) "
                           f"led to ≥{MIN_RETURN}% bounce every time. Support levels are "
                           f"genuine pivot lows from actual price history, within {MAX_DIST_FROM_CUR}% of current price."),
        "hold_periods":   HOLD_LABELS,
        "stocks":         results,
    }

    path = OUT / "bounce_signals.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written {len(results)} stocks, {len(alerts_today)} alerts today")
    if alerts_today:
        print(f"\n*** {len(alerts_today)} STOCKS NEAR SUPPORT TODAY ***")
        for r in alerts_today[:10]:
            al = r["alert_levels"][0]
            print(f"  {r['sym']:<14} price={r['price']} support={al['support_level']} "
                  f"({al['current_pct_from_sup']:+.1f}%) avg_bounce={al['avg_bounce']}%")


if __name__ == "__main__":
    main()
