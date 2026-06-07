#!/usr/bin/env python3
"""
candle_pattern_analyzer.py
===========================
Finds stocks that, whenever the price drops to a historically significant
support zone (once every few months to years), ALWAYS bounce up reliably.

Logic:
1. For each stock, find its rolling 1-year low at every point in history
2. When current price comes within 5% ABOVE that 1-year low = "support touch"
3. Minimum 30 trading days gap between consecutive touches (no duplicate counting)
4. After each touch, measure returns: 3d, 5d, 10d, 20d, 1M (22td)
5. ONLY keep stocks where EVERY single touch led to a positive bounce
6. Sort by best average return
7. TODAY ALERT: if price is currently within 5% of its 1-year low

Requirements:
  - Rs 5 Cr+ daily turnover
  - 5+ years of trading history, currently active
  - At least 2 historical support touches
  - 100% of touches gave positive return in the BEST hold period

Output: stock_analysis/candle_patterns.json
"""

import json, sys, warnings, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "equity"
OUT      = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

# CONFIG
MIN_TURNOVER  = 5_000_000    # Rs 5 Cr daily
MIN_YEARS     = 5
RECENT_DAYS   = 5
MIN_TOUCHES   = 2            # minimum 2 support touches
MIN_GAP_TD    = 30           # minimum 30 trading days between touches
SUPPORT_BAND  = 5.0          # within 5% ABOVE the 1-year low = support zone
HOLD_DAYS     = [3, 5, 10, 20, 22]
HOLD_LABELS   = {3:"3 Days", 5:"5 Days", 10:"10 Days", 20:"20 Days", 22:"1 Month"}
YEAR_WINDOW   = 252          # 1 year in trading days

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID","NIFTY","SENSEX")

IST     = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
now_str = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today   = now_ist.strftime("%Y-%m-%d")
cur_yr  = now_ist.year

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


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
            for col in df.columns:
                u = col.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):               cm[col] = "sym"
                elif u in ("SERIES","SCTYSRS"):              cm[col] = "series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):   cm[col] = "o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):   cm[col] = "h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):      cm[col] = "l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"): cm[col] = "c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):       cm[col] = "v"
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
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


def analyze(sym, df, latest_set):
    n       = len(df)
    c_arr   = df["c"].values
    l_arr   = df["l"].values
    v_arr   = df["v"].values
    dates   = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    # Basic filters
    if cur_price < 5: return None
    if last_date not in latest_set: return None
    yrs = set(int(d.year) for d in dates)
    if max(yrs) - min(yrs) < MIN_YEARS - 1: return None
    if min(yrs) > cur_yr - MIN_YEARS: return None
    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER: return None

    # Compute rolling 1-year low (minimum LOW over past YEAR_WINDOW days)
    low_series  = pd.Series(l_arr.astype(float))
    yr_low      = low_series.rolling(YEAR_WINDOW, min_periods=60).min().values

    # Find support touches: price within SUPPORT_BAND% above 1-year low
    # i.e.  low_price <= c_arr[i] <= yr_low[i] * (1 + SUPPORT_BAND/100)
    # AND   the 1-year low itself is not brand-new (stock is revisiting old support)
    touches = []
    last_touch_idx = -MIN_GAP_TD - 1

    max_hold = max(HOLD_DAYS)
    for i in range(YEAR_WINDOW, n - max_hold - 1):
        yl = float(yr_low[i])
        if yl <= 0 or math.isnan(yl): continue
        cp = float(c_arr[i])
        # Price must be within SUPPORT_BAND% above the 1-year low
        pct_above = (cp - yl) / yl * 100
        if pct_above < 0 or pct_above > SUPPORT_BAND:
            continue
        # Minimum gap between touches
        if i - last_touch_idx < MIN_GAP_TD:
            continue

        # Measure forward returns from entry (this day's close)
        hold_rets = {}
        for hd in HOLD_DAYS:
            xi = i + hd
            if xi < n:
                hold_rets[hd] = r2((float(c_arr[xi]) - cp) / cp * 100)

        touches.append({
            "date":       str(dates[i].date()),
            "year":       int(dates[i].year),
            "price":      r2(cp),
            "yr_low":     r2(yl),
            "pct_above_low": r2(pct_above),
            "hold_rets":  hold_rets,
        })
        last_touch_idx = i

    if len(touches) < MIN_TOUCHES: return None

    # For each hold period, check if 100% of touches gave positive return
    # Find the best hold period (highest % of touches positive, then highest avg)
    best_hold   = None
    best_avg    = -999
    hold_stats  = {}

    for hd in HOLD_DAYS:
        rets = [t["hold_rets"].get(hd) for t in touches
                if t["hold_rets"].get(hd) is not None]
        if len(rets) < MIN_TOUCHES: continue
        pct_pos = sum(1 for r in rets if r > 0) / len(rets) * 100
        if pct_pos < 100: continue  # 100% win rate required
        avg_ret = sum(rets) / len(rets)
        min_ret = min(rets)
        max_ret = max(rets)
        hold_stats[hd] = {
            "label":   HOLD_LABELS[hd],
            "n":       len(rets),
            "avg_ret": r2(avg_ret),
            "min_ret": r2(min_ret),
            "max_ret": r2(max_ret),
            "win_rate":100.0,
        }
        if avg_ret > best_avg:
            best_avg  = avg_ret
            best_hold = hd

    if not hold_stats: return None
    bs = hold_stats[best_hold]

    # Alert: is today's price in the support zone?
    yl_now = float(yr_low[-1]) if not math.isnan(float(yr_low[-1])) else 0
    alert_now = False
    pct_above_now = None
    tgt_prices = {}
    if yl_now > 0:
        pct_above_now = r2((cur_price - yl_now) / yl_now * 100)
        if 0 <= (cur_price - yl_now) / yl_now * 100 <= SUPPORT_BAND:
            alert_now = True
            for hd, hs in hold_stats.items():
                tgt_prices[hd] = {
                    "min_tgt": r2(cur_price * (1 + hs["min_ret"]/100)),
                    "avg_tgt": r2(cur_price * (1 + hs["avg_ret"]/100)),
                    "label":   hs["label"],
                }

    # Active alerts: touched support in last 22 trading days, track live return
    active = []
    for t in reversed(touches):
        try:
            days_ago = (now_ist.replace(tzinfo=None) -
                        datetime.strptime(t["date"], "%Y-%m-%d")).days
        except: continue
        if days_ago <= 0 or days_ago > 35: continue
        ep = t["price"] or cur_price
        lr = r2((cur_price - ep) / ep * 100) if ep > 0 else None
        active.append({
            "date":        t["date"],
            "entry_px":    t["price"],
            "cur_price":   r2(cur_price),
            "live_ret":    lr,
            "yr_low":      t["yr_low"],
            "pct_above":   t["pct_above_low"],
            "days_elapsed":days_ago,
            "best_hold":   best_hold,
            "avg_tgt":     r2(ep*(1+bs["avg_ret"]/100)) if ep else None,
            "min_tgt":     r2(ep*(1+bs["min_ret"]/100)) if ep else None,
        })
        break

    # Recent context
    ret5  = r2((cur_price-float(c_arr[max(0,n-6)]))/float(c_arr[max(0,n-6)])*100) if n>5 else None
    ret10 = r2((cur_price-float(c_arr[max(0,n-11)]))/float(c_arr[max(0,n-11)])*100) if n>10 else None
    ret15 = r2((cur_price-float(c_arr[max(0,n-16)]))/float(c_arr[max(0,n-16)])*100) if n>15 else None

    return {
        "sym":          sym,
        "price":        r2(cur_price),
        "last_date":    last_date,
        "turnover_cr":  r2(sum(tv5)/len(tv5)/1e7),
        "yr_low_now":   r2(yl_now),
        "pct_above_low":pct_above_now,
        "n_signals":    len(touches),
        "years":        sorted(set(t["year"] for t in touches)),
        "best_hold":    best_hold,
        "best_hold_label": HOLD_LABELS.get(best_hold,""),
        "avg_pk":       bs["avg_ret"],
        "min_pk":       bs["min_ret"],
        "max_pk":       bs["max_ret"],
        "hold_stats":   hold_stats,
        "ret5":         ret5,
        "ret10":        ret10,
        "ret15":        ret15,
        "alert_now":    alert_now,
        "active":       active,
        "has_today":    alert_now,
        "has_active":   len(active) > 0,
        "tgt_prices":   tgt_prices,
        "signals":      sorted(touches, key=lambda x: x["date"], reverse=True),
    }


def main():
    print(f"\n{'='*60}")
    print(f"Support Bounce Signals  IST:{now_str}")
    print(f"Logic: price within {SUPPORT_BAND}% of 1-year low = support touch")
    print(f"{'='*60}")

    all_data  = load_all()
    grouped   = all_data.groupby("sym")
    syms      = sorted(grouped.groups.keys())
    all_dates = sorted(set(str(pd.to_datetime(d).date())
                           for d in all_data["date"].unique()))
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]
    print(f"Last fetch: {last_fetch}  Symbols: {len(syms):,}")

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
            res = analyze(sym, grp, latest_set)
            if res: results.append(res)
            else:   skipped += 1
        except: skipped += 1

    results.sort(key=lambda x: -(x.get("avg_pk") or 0))

    today_a  = [r for r in results if r["has_today"]]
    active_a = [r for r in results if r["has_active"]]
    all_open = [r for r in results if r["has_today"] or r["has_active"]]

    # Flat history of all signal touches
    all_hist = []
    for r in results:
        for s in (r.get("signals") or []):
            hr = s.get("hold_rets", {})
            all_hist.append({
                "date":  s["date"], "year": s["year"],
                "sym":   r["sym"],  "price_now": r["price"],
                "price_at_signal": s["price"],
                "yr_low": s["yr_low"],
                "pct_above": s["pct_above_low"],
                "r3":  hr.get(3),  "r5":  hr.get(5),
                "r10": hr.get(10), "r20": hr.get(20),
                "r1m": hr.get(22),
            })
    all_hist.sort(key=lambda x: x["date"], reverse=True)

    output = {
        "generated_at": now_str,
        "today_ist":    today,
        "last_fetch":   last_fetch,
        "n_stocks":     len(results),
        "n_today":      len(today_a),
        "n_active":     len(active_a),
        "n_all_hist":   len(all_hist),
        "today_alerts": today_a,
        "active_alerts":active_a,
        "all_open":     all_open,
        "all_hist":     all_hist,
        "hold_labels":  HOLD_LABELS,
        "description":  (
            f"Stocks where price within {SUPPORT_BAND}% of 1-year low always bounced. "
            f"100% win rate. Min {MIN_TOUCHES} touches, min {MIN_GAP_TD}td gap. "
            f"Rs 5Cr+ turnover, 5yr+ history."
        ),
        "stocks": results,
    }

    path = OUT / "candle_patterns.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {path}")
    print(f"  Stocks: {len(results)}  Today: {len(today_a)}  History: {len(all_hist)}")
    if today_a:
        print(f"\n*** NEAR 1-YEAR LOW TODAY ***")
        for r in today_a[:10]:
            print(f"  {r['sym']:<14} price={r['price']} "
                  f"yr_low={r['yr_low_now']} pct_above={r['pct_above_low']:.1f}% "
                  f"best_avg=+{r['avg_pk']}% in {r['best_hold_label']}")
    if results:
        print(f"\nTop 10 by avg return:")
        for r in results[:10]:
            print(f"  {r['sym']:<14} avg=+{r['avg_pk']}% "
                  f"min=+{r['min_pk']}% hold={r['best_hold_label']} n={r['n_signals']}")


if __name__ == "__main__":
    main()
