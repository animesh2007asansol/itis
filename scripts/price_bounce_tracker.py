#!/usr/bin/env python3
"""
price_bounce_tracker.py
========================
For every qualifying stock, finds historical support levels (pivot lows)
where the price bounced immediately and consistently.

Tracks returns after touching support in:
  1 week (5td), 2 weeks (10td), 3 weeks (15td), 4 weeks (20td), 1 month (22td)

Requirements:
  - Daily turnover (price × volume) >= Rs 5 Cr
  - Must have traded in ALL of the last 6 calendar years
  - Currently active (traded within last 5 trading dates)
  - Support touch = price within ±5% or ±2% of pivot low level
  - Every historical touch must have bounced (100% win rate enforced per hold period)

Outputs daily alerts + weekly alerts + chronological alert history.
Output: stock_analysis/price_bounce.json
"""

import json, sys, warnings, math
from datetime import datetime, timedelta, timezone
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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MIN_PRICE        = 10.0
MIN_TURNOVER     = 5_000_000      # Rs 5 Cr
REQUIRED_YEARS   = 6              # must have data in all last 6 calendar years
RECENT_DAYS      = 5              # must have traded within last 5 trading dates
PRICE_BAND_TIGHT = 2.0            # ±2% tight band
PRICE_BAND_LOOSE = 5.0            # ±5% loose band
PIVOT_WINDOW     = 8              # days each side to confirm pivot low
MIN_BOUNCES      = 3              # minimum touch-and-bounce occurrences
MIN_BOUNCE_PCT   = 5.0            # minimum 5% bounce to count
HOLD_PERIODS     = {
    "1W":  5,    # 1 week  = 5 trading days
    "2W":  10,   # 2 weeks = 10 trading days
    "3W":  15,   # 3 weeks = 15 trading days
    "4W":  20,   # 4 weeks = 20 trading days
    "1M":  22,   # 1 month = 22 trading days
}

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")

IST       = timezone(timedelta(hours=5, minutes=30))
now_ist   = datetime.now(IST)
today_str = now_ist.strftime("%Y-%m-%d")
now_str   = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
cur_yr    = now_ist.year

# Last 6 completed calendar years
req_yrs   = set(range(cur_yr - REQUIRED_YEARS, cur_yr))  # e.g. 2020-2025

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing"); sys.exit(1)
    manifest = json.loads(MANIFEST.read_text())
    dates    = sorted(manifest.keys())
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
    if not frames: print("ERROR: No data."); sys.exit(1)
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


# ── PIVOT LOWS ─────────────────────────────────────────────────────────────────
def find_pivot_lows(l_arr, n):
    """Find indices where l[i] is the minimum within PIVOT_WINDOW days on each side."""
    pivots = []
    pw = PIVOT_WINDOW
    for i in range(pw, n - pw):
        window = l_arr[i - pw: i + pw + 1]
        if float(l_arr[i]) <= float(np.min(window)) + 0.001:
            pivots.append((i, float(l_arr[i])))
    return pivots


def cluster_pivots(pivots, band=PRICE_BAND_LOOSE):
    """Cluster nearby pivot lows. Returns list of (median_price, [indices])."""
    if not pivots: return []
    sp = sorted(pivots, key=lambda x: x[1])
    clusters = [[sp[0]]]
    for i in range(1, len(sp)):
        anchor = clusters[-1][0][1]
        if abs(sp[i][1] - anchor) / anchor * 100 <= band:
            clusters[-1].append(sp[i])
        else:
            clusters.append([sp[i]])
    return [(float(np.median([p[1] for p in cl])), [p[0] for p in cl])
            for cl in clusters]


# ── CANDLE PATTERN ─────────────────────────────────────────────────────────────
def detect_candle(o_arr, h_arr, l_arr, c_arr, idx):
    if idx < 1: return "No pattern"
    try:
        o0,h0,l0,c0 = float(o_arr[idx]),  float(h_arr[idx]),  float(l_arr[idx]),  float(c_arr[idx])
        o1,h1,l1,c1 = float(o_arr[idx-1]),float(h_arr[idx-1]),float(l_arr[idx-1]),float(c_arr[idx-1])
    except: return "No pattern"
    body0  = abs(c0 - o0)
    rng0   = h0 - l0 if h0 != l0 else 0.001
    lo_w0  = min(o0, c0) - l0
    up_w0  = h0 - max(o0, c0)
    if body0 > 0 and lo_w0 >= 2*body0 and up_w0 <= 0.3*body0:
        return "Hammer"
    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        return "Bullish Engulfing"
    if c1 < o1 and c0 > o0 and o0 < l1 and c0 > (o1+c1)/2:
        return "Piercing Line"
    if rng0 > 0 and body0/rng0 < 0.08:
        return "Doji"
    return "No pattern"


# ── ANALYZE ONE STOCK ──────────────────────────────────────────────────────────
def analyze_stock(sym, df, latest_set):
    n       = len(df)
    o_arr   = df["o"].values
    h_arr   = df["h"].values
    l_arr   = df["l"].values
    c_arr   = df["c"].values
    v_arr   = df["v"].values
    dates   = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    if cur_price < MIN_PRICE: return None
    # Must be recently active
    if last_date not in latest_set: return None
    # Turnover: avg last 5 days
    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER: return None
    # Must have data in ALL required years
    stock_years = set(int(dates[i].year) for i in range(n))
    if not req_yrs.issubset(stock_years): return None

    # Find pivot lows and cluster into support zones
    pivots   = find_pivot_lows(l_arr, n)
    if not pivots: return None
    clusters = cluster_pivots(pivots)

    support_levels = []
    max_hold = max(HOLD_PERIODS.values())

    for (support, _pivot_idxs) in clusters:
        # Support must be within 60% of current price (relevance filter)
        if abs(cur_price - support) / support * 100 > 60: continue

        for band_name, band_pct in [("tight_2pct", PRICE_BAND_TIGHT),
                                     ("loose_5pct", PRICE_BAND_LOOSE)]:
            touches = []
            i = 0
            while i < n - max_hold - 1:
                low_i = float(l_arr[i])
                pct_from_sup = (low_i - support) / support * 100
                if abs(pct_from_sup) <= band_pct:
                    # Candle pattern
                    candle = detect_candle(o_arr, h_arr, l_arr, c_arr, i)
                    # Entry = next day open
                    ai = i + 1
                    if ai >= n: break
                    ep = float(o_arr[ai])
                    if ep <= 0: i += 1; continue

                    hold_rets = {}
                    for label, hold_td in HOLD_PERIODS.items():
                        xi = ai + hold_td
                        if xi < n:
                            xp  = float(c_arr[xi])
                            ret = (xp - ep) / ep * 100
                            # Max gain in window
                            wh  = float(max(h_arr[ai:xi+1]))
                            mx  = (wh - ep) / ep * 100
                            hold_rets[label] = {"ret": r2(ret), "max": r2(mx),
                                                "exit_px": r2(xp), "entry_px": r2(ep)}

                    touches.append({
                        "date":          str(dates[i].date()),
                        "year":          int(dates[i].year),
                        "price":         r2(float(c_arr[i])),
                        "low":           r2(low_i),
                        "pct_from_sup":  r2(pct_from_sup),
                        "candle":        candle,
                        "hold_rets":     hold_rets,
                    })
                    i += PIVOT_WINDOW  # skip forward to avoid overlaps
                else:
                    i += 1

            if len(touches) < MIN_BOUNCES: continue

            # For each hold period: check 100% bounced at least MIN_BOUNCE_PCT
            hold_stats = {}
            for label, hold_td in HOLD_PERIODS.items():
                rets = [t["hold_rets"].get(label,{}).get("ret")
                        for t in touches if t["hold_rets"].get(label,{}).get("ret") is not None]
                if len(rets) < MIN_BOUNCES: continue
                if not all(r >= MIN_BOUNCE_PCT for r in rets): continue
                maxs = [t["hold_rets"][label]["max"] for t in touches
                        if t["hold_rets"].get(label,{}).get("max") is not None]
                hold_stats[label] = {
                    "hold_td":   hold_td,
                    "n":         len(rets),
                    "avg_ret":   r2(sum(rets)/len(rets)),
                    "min_ret":   r2(min(rets)),
                    "max_ret":   r2(max(rets)),
                    "avg_max":   r2(sum(maxs)/len(maxs)) if maxs else None,
                    "win_rate":  100.0,
                }

            if not hold_stats: continue

            # Best hold = highest avg_ret
            best_label = max(hold_stats, key=lambda k: hold_stats[k]["avg_ret"] or 0)
            best       = hold_stats[best_label]

            # Alert: is today's price within this band of support?
            cur_pct    = r2((cur_price - support) / support * 100)
            alert_now  = abs(cur_price - support) / support * 100 <= band_pct

            support_levels.append({
                "support":      r2(support),
                "band":         band_name,
                "band_pct":     band_pct,
                "n_touches":    len(touches),
                "best_hold":    best_label,
                "best_avg_ret": best["avg_ret"],
                "best_min_ret": best["min_ret"],
                "hold_stats":   hold_stats,
                "cur_pct_from_sup": cur_pct,
                "alert_now":    alert_now,
                # Target prices from today's price
                "targets": {
                    lbl: {
                        "avg_tgt": r2(cur_price * (1 + s["avg_ret"]/100)),
                        "min_tgt": r2(cur_price * (1 + s["min_ret"]/100)),
                    }
                    for lbl, s in hold_stats.items()
                },
                "touches": sorted(touches, key=lambda x: x["date"], reverse=True),
            })

    if not support_levels: return None
    support_levels.sort(key=lambda x: -(x.get("best_avg_ret") or 0))

    # Dominant candle at alert levels
    all_candles = []
    for sl in support_levels:
        all_candles.extend(t["candle"] for t in sl["touches"] if t["candle"] != "No pattern")
    dom_candle = Counter(all_candles).most_common(1)[0][0] if all_candles else "None"

    return {
        "sym":            sym,
        "price":          r2(cur_price),
        "last_date":      last_date,
        "years_active":   sorted(stock_years & req_yrs),
        "n_levels":       len(support_levels),
        "alert_now":      any(sl["alert_now"] for sl in support_levels),
        "alert_levels":   [sl for sl in support_levels if sl["alert_now"]],
        "best_avg_ret":   max((sl["best_avg_ret"] or 0) for sl in support_levels),
        "dominant_candle":dom_candle,
        "levels":         support_levels,
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"Price Bounce Tracker")
    print(f"IST: {now_str}")
    print(f"Required years: {sorted(req_yrs)}")
    print(f"{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())

    all_dates  = sorted(set(str(dt.date()) for dt in pd.to_datetime(all_data["date"].unique())))
    latest_set = set(all_dates[-RECENT_DAYS:])
    print(f"\nLatest trading date: {max(all_dates)}")
    print(f"Analyzing {len(syms):,} symbols...")

    results  = []
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} stocks")
        if any(sym.upper().endswith(s) for s in EXCL_SFX):
            excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 300: skipped += 1; continue
            res = analyze_stock(sym, grp, latest_set)
            if res: results.append(res)
            else:   skipped += 1
        except: skipped += 1

    results.sort(key=lambda x: -(x.get("best_avg_ret") or 0))

    # Alerts
    alerts_now  = sorted([r for r in results if r["alert_now"]],
                         key=lambda x: -(x.get("best_avg_ret") or 0))

    # Weekly alerts: touches within last 5 trading days
    week_ago = all_dates[-6] if len(all_dates) >= 6 else all_dates[0]
    alerts_week = []
    for r in results:
        week_touches = []
        for sl in r["levels"]:
            for t in sl["touches"]:
                if t["date"] >= week_ago:
                    week_touches.append({**t, "support": sl["support"],
                                         "band": sl["band"],
                                         "hold_stats": sl["hold_stats"],
                                         "sym": r["sym"], "price": r["price"]})
        if week_touches:
            alerts_week.append({"sym": r["sym"], "price": r["price"],
                                 "touches": sorted(week_touches,
                                                   key=lambda x: x["date"], reverse=True)})
    alerts_week.sort(key=lambda x: -(x["touches"][0].get("hold_rets",{})
                                      .get("1W",{}).get("ret") or 0))

    # Chronological alert history: all touches within last 30 days
    month_ago = all_dates[-22] if len(all_dates) >= 22 else all_dates[0]
    chron_alerts = []
    for r in results:
        for sl in r["levels"]:
            for t in sl["touches"]:
                if t["date"] >= month_ago:
                    chron_alerts.append({
                        "date":     t["date"],
                        "sym":      r["sym"],
                        "price":    r["price"],
                        "low":      t["low"],
                        "support":  sl["support"],
                        "band":     sl["band"],
                        "candle":   t["candle"],
                        "pct_from_sup": t["pct_from_sup"],
                        "1W_ret":   t["hold_rets"].get("1W",{}).get("ret"),
                        "2W_ret":   t["hold_rets"].get("2W",{}).get("ret"),
                        "4W_ret":   t["hold_rets"].get("4W",{}).get("ret"),
                        "1M_ret":   t["hold_rets"].get("1M",{}).get("ret"),
                    })
    chron_alerts.sort(key=lambda x: x["date"], reverse=True)

    output = {
        "generated_at":  now_str,
        "today_ist":     today_str,
        "required_years":sorted(req_yrs),
        "n_stocks":      len(results),
        "n_alerts_now":  len(alerts_now),
        "n_alerts_week": len(alerts_week),
        "n_chron":       len(chron_alerts),
        "alerts_now":    alerts_now,
        "alerts_week":   alerts_week,
        "chron_alerts":  chron_alerts,
        "hold_labels":   {k:f"{v} trading days" for k,v in HOLD_PERIODS.items()},
        "description":   (
            f"Stocks touching historical support (±2% or ±5%) that always bounce "
            f"upward. Min 5 Cr daily turnover. Active in all last {REQUIRED_YEARS} years."
        ),
        "stocks": results,
    }

    path = OUT / "price_bounce.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {path}")
    print(f"  Qualifying stocks   : {len(results)}")
    print(f"  Alerts now (today)  : {len(alerts_now)}")
    print(f"  Alerts this week    : {len(alerts_week)}")
    print(f"  Excluded (ETF etc)  : {excluded}")
    print(f"  Skipped             : {skipped}")
    if alerts_now:
        print(f"\n*** TODAY — near support ***")
        for r in alerts_now[:10]:
            al = r["alert_levels"][0]
            print(f"  {r['sym']:<14} price={r['price']} support={al['support']} "
                  f"({al['cur_pct_from_sup']:+.1f}%) best={al['best_avg_ret']}%")


if __name__ == "__main__":
    main()
