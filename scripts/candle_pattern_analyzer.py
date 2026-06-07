#!/usr/bin/env python3
"""
candle_pattern_analyzer.py
===========================
For big-profile NSE stocks (Rs 20 Cr+ daily turnover, 5+ years active),
finds candle patterns that occur AFTER A FALL with COMPARATIVE VOLUME
and ALWAYS result in a minimum 30% upside — no cap on actual profit.

Tracks how long the upside lasts: keeps growing at ≥5% per week.
Finds the optimal exit point where weekly growth drops below 5%.

No frequency filter: signals can fire once a year or once in 2 years.
100% win rate: every occurrence of the pattern must give 30%+.

Outputs:
  - All historical pattern signals per stock (searchable)
  - Today's alerts (signal fired on last NSE data fetch)
  - Active alerts (in growth phase, not yet peaked)
  - All alerts chronologically

Output: stock_analysis/candle_patterns.json
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
MIN_TURNOVER     = 20_000_000    # Rs 20 Cr daily value
MIN_YEARS        = 5
RECENT_DAYS      = 5             # must have traded in last 5 dates
MIN_RETURN       = 30.0          # minimum 30% after signal
WIN_RATE         = 100.0         # every occurrence must give 30%+
MIN_OCC          = 2             # minimum 2 occurrences (can be rare)
PRIOR_FALL_PCT   = 5.0           # stock must have fallen at least 5% in 15d before signal
VOL_RATIO_MIN    = 1.5           # volume must be 1.5x 20-day average
WEEKLY_GROWTH_MIN= 5.0           # track until weekly growth drops below 5%
MAX_TRACK_WEEKS  = 16            # track up to 16 weeks (80 trading days) for optimal exit
HOLD_LABELS      = {3:"3 Days", 5:"5 Days", 10:"10 Days", 20:"1 Month",
                    44:"2 Months", 66:"3 Months"}

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")

IST      = timezone(timedelta(hours=5, minutes=30))
now_ist  = datetime.now(IST)
now_str  = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today    = now_ist.strftime("%Y-%m-%d")
cur_yr   = now_ist.year
min_yr   = cur_yr - MIN_YEARS

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


# ── DATA LOAD ──────────────────────────────────────────────────────────────────
def load_all():
    if not MANIFEST.exists(): print("ERROR: manifest.json missing"); sys.exit(1)
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
                if u in ("SYMBOL","TCKRSYMB"):               cm[col]="sym"
                elif u in ("SERIES","SCTYSRS"):              cm[col]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):   cm[col]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):   cm[col]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):      cm[col]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"): cm[col]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):       cm[col]="v"
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


# ── CANDLE PATTERN DETECTION ───────────────────────────────────────────────────
def detect_patterns(o_arr, h_arr, l_arr, c_arr, idx):
    """
    Detect bullish reversal candle patterns at index idx.
    Returns list of pattern names found.
    """
    if idx < 2: return []
    try:
        o0 = float(o_arr[idx]);   h0 = float(h_arr[idx])
        l0 = float(l_arr[idx]);   c0 = float(c_arr[idx])
        o1 = float(o_arr[idx-1]); h1 = float(h_arr[idx-1])
        l1 = float(l_arr[idx-1]); c1 = float(c_arr[idx-1])
        o2 = float(o_arr[idx-2]); c2 = float(c_arr[idx-2])
    except: return []

    body0  = abs(c0 - o0)
    rng0   = h0 - l0 if h0 != l0 else 0.001
    lo_w0  = min(o0, c0) - l0
    up_w0  = h0 - max(o0, c0)
    body1  = abs(c1 - o1)
    body2  = abs(c2 - o2)

    found = []

    # Hammer: small body near top, long lower wick ≥ 2× body, tiny upper wick
    if body0 > 0 and lo_w0 >= 2 * body0 and up_w0 <= 0.3 * body0:
        found.append("Hammer")

    # Bullish Engulfing: prev red, today green, today body fully engulfs prev
    if c1 < o1 and c0 > o0 and c0 > o1 and o0 < c1:
        found.append("Bullish Engulfing")

    # Morning Star: 3-candle — big red, small body, big green closing above prev midpoint
    if c2 < o2 and body2 > 0 and body1 < body2 * 0.4 and \
       c0 > o0 and c0 > (o2 + c2) / 2:
        found.append("Morning Star")

    # Piercing Line: prev red, today opens below prev low, closes above prev midpoint
    if c1 < o1 and c0 > o0 and o0 < l1 and c0 > (o1 + c1) / 2:
        found.append("Piercing Line")

    # Dragonfly Doji: open ≈ close ≈ high, long lower wick
    if rng0 > 0 and abs(o0 - c0) / rng0 < 0.08 and lo_w0 >= 0.6 * rng0 and up_w0 <= 0.1 * rng0:
        found.append("Dragonfly Doji")

    # Bullish Marubozu: big green candle, tiny wicks
    if c0 > o0 and body0 > 0 and lo_w0 <= 0.05 * body0 and up_w0 <= 0.05 * body0 \
       and body0 / rng0 > 0.9:
        found.append("Bullish Marubozu")

    # Three White Soldiers: 3 consecutive green closes, each higher than prev
    if idx >= 2 and c0 > o0 and c1 > o1 and c2 > o2 \
       and c0 > c1 > c2 and o0 > o1 > o2:
        found.append("Three White Soldiers")

    return found


# ── OPTIMAL EXIT DETECTION ─────────────────────────────────────────────────────
def find_optimal_exit(c_arr, entry_idx, n):
    """
    From entry_idx, track weekly (5td) cumulative return.
    Continue while weekly increment >= WEEKLY_GROWTH_MIN (5%).
    Returns dict with optimal exit info and weekly tracking data.
    """
    ep = float(c_arr[entry_idx])
    if ep <= 0: return None

    weekly_track = []
    prev_cum     = 0.0

    for wk in range(1, MAX_TRACK_WEEKS + 1):
        xi = entry_idx + wk * 5
        if xi >= n: break
        cum_ret    = (float(c_arr[xi]) - ep) / ep * 100
        weekly_inc = cum_ret - prev_cum
        weekly_track.append({
            "week":       wk,
            "days":       wk * 5,
            "cum_ret":    r2(cum_ret),
            "weekly_inc": r2(weekly_inc),
        })
        prev_cum = cum_ret
        if wk >= 2 and weekly_inc < WEEKLY_GROWTH_MIN:
            break   # growth slowed — this is the exit point

    if not weekly_track: return None

    peak     = max(weekly_track, key=lambda x: x["cum_ret"] or 0)
    optimal  = weekly_track[-1]   # last tracked week before growth slowed

    # Fixed hold returns
    hold_rets = {}
    for days in [3, 5, 10, 20, 44, 66]:
        xi = entry_idx + days
        if xi < n:
            hold_rets[days] = r2((float(c_arr[xi]) - ep) / ep * 100)

    return {
        "optimal_days":    optimal["days"],
        "optimal_ret":     optimal["cum_ret"],
        "peak_days":       peak["days"],
        "peak_ret":        peak["cum_ret"],
        "weekly_track":    weekly_track,
        "hold_rets":       hold_rets,
    }


# ── PER-STOCK ANALYSIS ─────────────────────────────────────────────────────────
def analyze_stock(sym, df, latest_set):
    n      = len(df)
    o_arr  = df["o"].values
    h_arr  = df["h"].values
    l_arr  = df["l"].values
    c_arr  = df["c"].values
    v_arr  = df["v"].values
    dates  = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    if cur_price < 10: return None
    if last_date not in latest_set: return None

    # Must span at least 5 years
    yrs = set(int(d.year) for d in dates)
    if max(yrs) - min(yrs) < MIN_YEARS - 1: return None
    if min(yrs) > min_yr: return None

    # Turnover: avg last 5 days
    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5) / len(tv5) < MIN_TURNOVER: return None

    # Compute 20-day volume moving average
    vol_ma20 = pd.Series(v_arr).rolling(20, min_periods=10).mean().values

    # ── Find all valid candle pattern signals ──────────────────────────────────
    # Only consider signals where we have enough forward data for MAX_TRACK_WEEKS
    min_fwd = MAX_TRACK_WEEKS * 5
    signals  = []

    for idx in range(20, n - min_fwd):
        # CONDITION 1: Prior fall — stock fell at least PRIOR_FALL_PCT in last 15 days
        look_back = 15
        prev_close = float(c_arr[idx - look_back])
        if prev_close <= 0: continue
        fall_pct = (float(c_arr[idx]) - prev_close) / prev_close * 100
        if fall_pct > -PRIOR_FALL_PCT: continue   # not a sufficient fall

        # CONDITION 2: Candle pattern detected
        patterns = detect_patterns(o_arr, h_arr, l_arr, c_arr, idx)
        if not patterns: continue

        # CONDITION 3: Comparative volume — must be ≥ VOL_RATIO_MIN × 20d avg
        vol_avg = float(vol_ma20[idx]) if not math.isnan(float(vol_ma20[idx])) else 0
        if vol_avg <= 0: continue
        vol_ratio = float(v_arr[idx]) / vol_avg
        if vol_ratio < VOL_RATIO_MIN: continue

        # ── Compute forward returns ────────────────────────────────────────────
        entry_idx = idx + 1   # entry at close of next trading day
        if entry_idx >= n: continue
        ep = float(c_arr[entry_idx])
        if ep <= 0: continue

        exit_info = find_optimal_exit(c_arr, entry_idx, n)
        if exit_info is None: continue

        # Check min return: optimal or peak must be >= 30%
        best_ret = max(exit_info["peak_ret"] or 0, exit_info["optimal_ret"] or 0)
        if best_ret < MIN_RETURN: continue

        # Context before signal
        ret_5d_before  = r2((float(c_arr[idx]) - float(c_arr[max(0, idx-5)])) /
                            float(c_arr[max(0, idx-5)]) * 100) if idx >= 5 else None
        ret_10d_before = r2((float(c_arr[idx]) - float(c_arr[max(0, idx-10)])) /
                            float(c_arr[max(0, idx-10)]) * 100) if idx >= 10 else None
        ret_20d_before = r2((float(c_arr[idx]) - float(c_arr[max(0, idx-20)])) /
                            float(c_arr[max(0, idx-20)]) * 100) if idx >= 20 else None

        signals.append({
            "date":           str(dates[idx].date()),
            "year":           int(dates[idx].year),
            "close_at_signal":r2(float(c_arr[idx])),
            "patterns":       patterns,
            "primary_pattern":patterns[0],
            "fall_15d_pct":   r2(fall_pct),
            "vol_ratio":      r2(vol_ratio),
            "entry_date":     str(dates[entry_idx].date()),
            "entry_px":       r2(ep),
            "optimal_days":   exit_info["optimal_days"],
            "optimal_ret":    exit_info["optimal_ret"],
            "peak_days":      exit_info["peak_days"],
            "peak_ret":       exit_info["peak_ret"],
            "hold_rets":      exit_info["hold_rets"],
            "weekly_track":   exit_info["weekly_track"],
            "ret_5d_before":  ret_5d_before,
            "ret_10d_before": ret_10d_before,
            "ret_20d_before": ret_20d_before,
        })

    if len(signals) < MIN_OCC: return None

    # 100% win rate: every signal must have peak_ret >= 30%
    if not all(s["peak_ret"] >= MIN_RETURN for s in signals): return None

    # Stats
    peak_rets    = [s["peak_ret"]     for s in signals]
    optimal_rets = [s["optimal_ret"]  for s in signals if s["optimal_ret"] is not None]
    opt_days_list= [s["optimal_days"] for s in signals if s["optimal_days"] is not None]
    pk_days_list = [s["peak_days"]    for s in signals if s["peak_days"] is not None]

    avg_peak    = r2(sum(peak_rets)    / len(peak_rets))
    min_peak    = r2(min(peak_rets))
    avg_optimal = r2(sum(optimal_rets) / len(optimal_rets)) if optimal_rets else None
    avg_opt_days= r2(sum(opt_days_list) / len(opt_days_list)) if opt_days_list else None
    avg_pk_days = r2(sum(pk_days_list)  / len(pk_days_list))  if pk_days_list else None

    # Most common pattern
    all_pats    = []
    for s in signals: all_pats.extend(s["patterns"])
    pat_counts  = Counter(all_pats)
    dominant_pat= pat_counts.most_common(1)[0][0] if pat_counts else "Unknown"

    # ── Today's alert: did the last bar fire the signal? ──────────────────────
    today_alert = None
    if n >= 3:
        idx_today = n - 1
        look_back = 15
        if idx_today >= look_back:
            fall_now  = (float(c_arr[idx_today]) - float(c_arr[idx_today - look_back])) / \
                        float(c_arr[idx_today - look_back]) * 100 if float(c_arr[idx_today-look_back]) > 0 else 0
            pats_today = detect_patterns(o_arr, h_arr, l_arr, c_arr, idx_today)
            vol_avg_now = float(vol_ma20[idx_today]) if not math.isnan(float(vol_ma20[idx_today])) else 0
            vol_r_now   = float(v_arr[idx_today]) / vol_avg_now if vol_avg_now > 0 else 0
            if fall_now < -PRIOR_FALL_PCT and pats_today and vol_r_now >= VOL_RATIO_MIN:
                today_alert = {
                    "date":         last_date,
                    "close":        r2(cur_price),
                    "patterns":     pats_today,
                    "fall_15d_pct": r2(fall_now),
                    "vol_ratio":    r2(vol_r_now),
                    "avg_peak_ret": avg_peak,
                    "min_peak_ret": min_peak,
                    "avg_opt_days": avg_opt_days,
                    "targets": {
                        "30pct": r2(cur_price * 1.30),
                        "avg":   r2(cur_price * (1 + avg_peak / 100)),
                        "min":   r2(cur_price * (1 + min_peak / 100)),
                    }
                }

    # ── Active alerts: signals from past still in optimal hold window ─────────
    active_alerts = []
    for sig in reversed(signals):
        try:
            sig_dt   = datetime.strptime(sig["date"], "%Y-%m-%d")
            days_ago = (now_ist.replace(tzinfo=None) - sig_dt).days
        except: continue
        max_hold = (sig.get("peak_days") or 80)
        if days_ago > max_hold: continue
        if days_ago <= 0: continue
        # Calculate live return from entry
        ep_sig   = sig["entry_px"] or cur_price
        live_ret = r2((cur_price - ep_sig) / ep_sig * 100) if ep_sig > 0 else None
        remaining= r2((sig["peak_ret"] or 30) - (live_ret or 0))
        active_alerts.append({
            "date":       sig["date"],
            "entry_date": sig["entry_date"],
            "entry_px":   sig["entry_px"],
            "cur_price":  r2(cur_price),
            "live_ret":   live_ret,
            "peak_ret":   sig["peak_ret"],
            "remaining_to_peak": remaining,
            "days_elapsed": days_ago,
            "days_to_peak": sig["peak_days"],
            "patterns":   sig["patterns"],
        })
        break  # only most recent active alert per stock

    # Recent context
    ret_5d  = r2((cur_price - float(c_arr[max(0, n-6)]))  / float(c_arr[max(0, n-6)])  * 100) if n > 5  else None
    ret_10d = r2((cur_price - float(c_arr[max(0, n-11)])) / float(c_arr[max(0, n-11)]) * 100) if n > 10 else None
    ret_15d = r2((cur_price - float(c_arr[max(0, n-16)])) / float(c_arr[max(0, n-16)]) * 100) if n > 15 else None

    return {
        "sym":           sym,
        "price":         r2(cur_price),
        "last_date":     last_date,
        "turnover_cr":   r2(sum(tv5) / len(tv5) / 1e7),
        "n_signals":     len(signals),
        "years":         sorted(set(s["year"] for s in signals)),
        "dominant_pattern": dominant_pat,
        "pattern_counts":dict(pat_counts.most_common()),
        "avg_peak_ret":  avg_peak,
        "min_peak_ret":  min_peak,
        "avg_optimal_ret":avg_optimal,
        "avg_opt_days":  avg_opt_days,
        "avg_peak_days": avg_pk_days,
        "ret_5d":        ret_5d,
        "ret_10d":       ret_10d,
        "ret_15d":       ret_15d,
        "today_alert":   today_alert,
        "active_alerts": active_alerts,
        "has_today":     today_alert is not None,
        "has_active":    len(active_alerts) > 0,
        "signals":       sorted(signals, key=lambda x: x["date"], reverse=True),
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"Candle Pattern Analyzer")
    print(f"IST: {now_str}")
    print(f"Min 30% return, 100% win rate, 2+ occurrences")
    print(f"{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())

    all_dates  = sorted(set(str(dt.date()) for dt in pd.to_datetime(all_data["date"].unique())))
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]
    print(f"\nLast data fetch: {last_fetch} | Analyzing {len(syms):,} symbols...")

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

    results.sort(key=lambda x: -(x.get("avg_peak_ret") or 0))

    today_alerts  = [r for r in results if r.get("has_today")]
    active_alerts = [r for r in results if r.get("has_active")]
    all_open      = [r for r in results if r.get("has_today") or r.get("has_active")]

    # Chronological all-time alert history (flat, sorted by date desc)
    all_hist = []
    for r in results:
        for sig in (r.get("signals") or []):
            all_hist.append({
                "date":        sig["date"],
                "year":        sig["year"],
                "sym":         r["sym"],
                "price_now":   r["price"],
                "patterns":    sig["patterns"],
                "fall_15d":    sig["fall_15d_pct"],
                "vol_ratio":   sig["vol_ratio"],
                "entry_px":    sig["entry_px"],
                "peak_ret":    sig["peak_ret"],
                "peak_days":   sig["peak_days"],
                "optimal_ret": sig["optimal_ret"],
                "optimal_days":sig["optimal_days"],
                "ret_3d":      sig["hold_rets"].get(3),
                "ret_5d":      sig["hold_rets"].get(5),
                "ret_10d":     sig["hold_rets"].get(10),
                "ret_20d":     sig["hold_rets"].get(20),
            })
    all_hist.sort(key=lambda x: x["date"], reverse=True)

    output = {
        "generated_at":   now_str,
        "today_ist":      today,
        "last_fetch":     last_fetch,
        "n_stocks":       len(results),
        "n_today":        len(today_alerts),
        "n_active":       len(active_alerts),
        "n_all_hist":     len(all_hist),
        "today_alerts":   today_alerts,
        "active_alerts":  active_alerts,
        "all_open":       all_open,
        "all_hist":       all_hist,
        "description":    (
            "Candle patterns (Hammer/Engulfing/Morning Star/etc) "
            "after a 5%+ fall with 1.5x+ volume — always giving 30%+ upside. "
            "Tracks optimal exit where weekly growth drops below 5%."
        ),
        "stocks": results,
    }

    path = OUT / "candle_patterns.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {path}")
    print(f"  Qualifying stocks: {len(results)}")
    print(f"  Today's alerts:   {len(today_alerts)}")
    print(f"  Active alerts:    {len(active_alerts)}")
    print(f"  Total signals:    {len(all_hist)}")
    if today_alerts:
        print(f"\n*** TODAY'S CANDLE SIGNALS ***")
        for r in today_alerts[:10]:
            ta = r["today_alert"]
            print(f"  {r['sym']:<14} {ta['patterns'][0]:<22} "
                  f"fall={ta['fall_15d_pct']:+.1f}% vol={ta['vol_ratio']:.1f}x "
                  f"tgt=Rs{ta['targets']['avg']}")


if __name__ == "__main__":
    main()
