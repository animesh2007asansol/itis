#!/usr/bin/env python3
"""
uc2_timing_optimizer.py
=======================
For every UC2 seasonal stock (qualifying with ≥95% win rate on 1st-of-month entry),
this script runs a detailed day-by-day timing analysis:

1. ENTRY WINDOW: Tests entry on day -5 to +5 relative to the 1st trading day of
   the seasonal month. Finds which entry day gives the best average return.

2. EXIT WINDOW: For each entry day, tests every possible exit from day +1 to day +90.
   Builds a return curve and finds the peak (optimal hold length) before the stock
   starts giving back gains.

3. DRAWDOWN GUARD: Tracks the maximum gain seen during the hold and the subsequent
   pullback. Outputs the "exit before drawdown" recommendation.

Output: stock_analysis/uc2_timing.json
"""

import json, math, os, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data" / "equity"
OUT     = ROOT / "stock_analysis"
MANIFEST= ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
MIN_PRICE       = 10.0
MIN_TURNOVER    = 5_000_000      # Rs 5 Cr avg daily turnover
MIN_OCC         = 3              # minimum 3 historical years with signal
WIN_RATE        = 95.0           # % of years must be profitable
MIN_RETURN      = 10.0           # minimum 10% profit
ENTRY_WINDOW    = range(-5, 6)   # test entry from -5 days before to +5 days after 1st
EXIT_WINDOW     = range(1, 91)   # test exit from day 1 to day 90 after entry
MAX_DRAWDOWN_WARN = -5.0         # warn if peak-to-current drawdown exceeds -5%

# Month names
MON = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTHS = list(range(1, 13))

# ── HELPERS ──────────────────────────────────────────────────────────────────
r2  = lambda x: round(float(x), 2) if x is not None else None
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_stock_data():
    """Load all equity data from manifest dates into per-symbol DataFrames."""
    if not MANIFEST.exists():
        print("ERROR: manifest.json not found. Run main workflow first.")
        sys.exit(1)

    manifest = json.loads(MANIFEST.read_text())
    dates_sorted = sorted(manifest.keys())

    print(f"  Loading {len(dates_sorted)} trading dates...")

    frames = []
    for ds in dates_sorted:
        y, m_str, _ = ds.split("-")
        p = DATA / y / m_str / f"{ds}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = df.columns.str.strip()

            # Handle both old and new NSE column formats
            col_map = {}
            for col in df.columns:
                cl = col.strip().upper()
                if cl in ("SYMBOL","TCKRSYMB"):           col_map[col] = "sym"
                elif cl in ("SERIES","SCTYSRS"):          col_map[col] = "series"
                elif cl in ("OPEN","OPNPRIC","OPEN PRICE"): col_map[col] = "o"
                elif cl in ("HIGH","HGHPRIC","HIGH PRICE"): col_map[col] = "h"
                elif cl in ("LOW","LWPRIC","LOW PRICE"):   col_map[col] = "l"
                elif cl in ("CLOSE","CLSPRIC","CLOSE PRICE"): col_map[col] = "c"
                elif cl in ("TOTTRDQTY","TTLTRADGVOL"):    col_map[col] = "v"
            df = df.rename(columns=col_map)

            needed = {"sym","series","o","h","l","c","v"}
            if not needed.issubset(df.columns):
                continue

            df = df[df["series"].str.strip().isin(["EQ","BE"])].copy()
            df["date"] = pd.to_datetime(ds)
            frames.append(df[["sym","o","h","l","c","v","date"]])
        except Exception:
            continue

    if not frames:
        print("ERROR: No data loaded.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    all_data = all_data.sort_values(["sym","date"]).reset_index(drop=True)

    print(f"  Loaded {len(all_data):,} rows, {all_data['sym'].nunique():,} symbols")
    return all_data


def get_stock_df(group):
    """Convert a stock's group to a clean sorted DataFrame."""
    df = group.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def find_seasonal_months(df):
    """
    Find all months where this stock qualifies as UC2:
    ≥95% of 1st-of-month entries returned ≥10% within any hold period.
    Returns list of (month, hold_days, avg_ret, min_ret, years) tuples.
    """
    results = []
    c = df["c"].values
    dates = pd.to_datetime(df["date"].values)
    months = pd.DatetimeIndex(dates).month
    years  = pd.DatetimeIndex(dates).year
    n = len(df)

    for mo in MONTHS:
        # Find first trading day of each occurrence of this month
        first_days = []
        for i in range(n):
            if months[i] == mo and (i == 0 or months[i-1] != mo):
                first_days.append(i)

        if len(first_days) < MIN_OCC:
            continue

        for hold in [10, 15, 20, 30, 45, 60]:
            returns = []
            yr_list = []
            for fi in first_days:
                ei = fi + 1  # entry = next day open
                xi = min(ei + hold, n - 1)
                if xi == ei:
                    continue
                # Only count complete trades
                if xi < ei + hold:
                    continue
                ep = df["o"].iloc[ei]
                xp = df["c"].iloc[xi]
                if ep <= 0:
                    continue
                ret = (xp - ep) / ep * 100
                returns.append(ret)
                yr_list.append(int(years[fi]))

            if len(returns) < MIN_OCC:
                continue

            wins = sum(1 for r in returns if r >= MIN_RETURN)
            wr   = wins / len(returns) * 100

            if wr < WIN_RATE:
                continue
            if any(r <= 0 for r in returns):
                continue

            avg_ret = sum(returns) / len(returns)
            min_ret = min(returns)

            results.append({
                "month":    mo,
                "hold":     hold,
                "avg_ret":  r2(avg_ret),
                "min_ret":  r2(min_ret),
                "win_rate": r2(wr),
                "n":        len(returns),
                "years":    sorted(set(yr_list)),
            })

    # Keep best hold per month (highest avg_ret)
    best_per_month = {}
    for r in results:
        mo = r["month"]
        if mo not in best_per_month or r["avg_ret"] > best_per_month[mo]["avg_ret"]:
            best_per_month[mo] = r

    return list(best_per_month.values())


def analyze_entry_exit(df, month, base_hold):
    """
    Core analysis: for this stock and this seasonal month,
    test all entry offsets (-5 to +5 days from 1st of month)
    and all exit days (1 to 90) to find the optimal timing.

    Returns a dict with:
    - entry_analysis: for each offset, average/min/max returns
    - exit_curve: for best entry offset, return on each exit day
    - best_entry_offset: offset with highest average return
    - best_exit_day: day number with highest average return
    - peak_then_decline_day: first day after peak where avg return drops ≥2%
    - recommendation: plain English
    """
    c = df["c"].values
    o = df["o"].values
    h = df["h"].values
    l = df["l"].values
    dates = pd.to_datetime(df["date"].values)
    mons  = pd.DatetimeIndex(dates).month
    n     = len(df)

    # Find all 1st-of-month indices for this month
    first_days = [i for i in range(n) if mons[i] == month and (i == 0 or mons[i-1] != month)]

    if len(first_days) < MIN_OCC:
        return None

    # ── 1. ENTRY WINDOW ANALYSIS ─────────────────────────────────────────────
    entry_analysis = {}
    for offset in ENTRY_WINDOW:
        returns = []
        for fi in first_days:
            ei = fi + offset      # entry signal day (with offset)
            actual_ei = ei + 1    # actual entry = next day open
            xi = min(actual_ei + base_hold, n - 1)
            if actual_ei >= n or xi >= n or actual_ei < 0:
                continue
            if xi < actual_ei + base_hold:  # incomplete trade
                continue
            ep = o[actual_ei]
            xp = c[xi]
            if ep <= 0:
                continue
            returns.append((xp - ep) / ep * 100)

        if len(returns) < max(2, MIN_OCC - 1):
            continue

        entry_analysis[offset] = {
            "offset":    offset,
            "label":     f"{'1st' if offset==0 else (str(abs(offset))+'d '+('before' if offset<0 else 'after'))+' 1st'} of {MON[month]}",
            "avg_ret":   r2(sum(returns) / len(returns)),
            "min_ret":   r2(min(returns)),
            "max_ret":   r2(max(returns)),
            "n":         len(returns),
            "win_rate":  r2(sum(1 for r in returns if r >= MIN_RETURN) / len(returns) * 100),
            "all_neg":   all(r < 0 for r in returns),
        }

    if not entry_analysis:
        return None

    # Best entry offset = highest avg_ret with ≥WIN_RATE win rate
    valid_entries = {k: v for k, v in entry_analysis.items()
                     if v["win_rate"] >= WIN_RATE and not v["all_neg"]}
    if not valid_entries:
        valid_entries = entry_analysis  # fallback

    best_offset = max(valid_entries, key=lambda k: valid_entries[k]["avg_ret"])

    # ── 2. EXIT CURVE ANALYSIS ───────────────────────────────────────────────
    # For best entry offset: build day-by-day return curve
    exit_curve    = []   # [{day, avg_ret, min_ret, max_ret, win_rate, n}]
    max_gain_curve = []  # track max intraday gain curve too

    for exit_day in EXIT_WINDOW:
        daily_returns = []
        daily_max_gains = []

        for fi in first_days:
            ei       = fi + best_offset
            actual_ei = ei + 1
            exit_idx  = actual_ei + exit_day
            if actual_ei >= n or exit_idx >= n or actual_ei < 0:
                continue
            ep = o[actual_ei]
            if ep <= 0:
                continue

            # Return at this exit day (close)
            ret = (c[exit_idx] - ep) / ep * 100
            daily_returns.append(ret)

            # Max gain seen from entry to this exit day (highest high in window)
            window_highs = h[actual_ei:exit_idx+1]
            if len(window_highs) > 0:
                max_gain = (max(window_highs) - ep) / ep * 100
                daily_max_gains.append(max_gain)

        if len(daily_returns) < 2:
            continue

        avg = sum(daily_returns) / len(daily_returns)
        avg_maxg = sum(daily_max_gains) / len(daily_max_gains) if daily_max_gains else avg

        exit_curve.append({
            "day":           exit_day,
            "avg_ret":       r2(avg),
            "min_ret":       r2(min(daily_returns)),
            "max_ret":       r2(max(daily_returns)),
            "avg_max_gain":  r2(avg_maxg),
            "win_rate":      r2(sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100),
            "n":             len(daily_returns),
        })

    if not exit_curve:
        return None

    # ── 3. FIND OPTIMAL EXIT ─────────────────────────────────────────────────
    avg_rets = [p["avg_ret"] for p in exit_curve]
    days     = [p["day"]     for p in exit_curve]

    # Peak: day with highest average return
    peak_idx  = int(np.argmax(avg_rets))
    peak_day  = days[peak_idx]
    peak_ret  = avg_rets[peak_idx]

    # Find the day where return drops ≥2% from peak (start of consistent decline)
    decline_day = None
    for i in range(peak_idx + 1, len(avg_rets)):
        if avg_rets[i] <= peak_ret - 2.0:
            # Confirm it keeps declining (next 3 days also below peak-2%)
            confirm = [avg_rets[j] <= peak_ret - 1.5
                       for j in range(i, min(i+3, len(avg_rets)))]
            if sum(confirm) >= 2:
                decline_day = days[i]
                break

    # "Sweet spot" window: days where avg_ret >= 80% of peak
    sweet_start = next((days[j] for j in range(len(days)) if avg_rets[j] >= peak_ret * 0.80), peak_day)
    sweet_end   = peak_day
    for j in range(peak_idx, len(days)):
        if avg_rets[j] >= peak_ret * 0.80:
            sweet_end = days[j]
        else:
            break

    # ── 4. ENTRY RECOMMENDATION ──────────────────────────────────────────────
    best_e = entry_analysis.get(best_offset, {})
    entry_label = best_e.get("label", f"Day {best_offset:+d} from 1st of {MON[month]}")

    if best_offset < 0:
        entry_rec = f"Enter {abs(best_offset)} trading day{'s' if abs(best_offset)>1 else ''} BEFORE 1st of {MON[month]} — stock already rising into month start"
    elif best_offset == 0:
        entry_rec = f"Enter on 1st trading day of {MON[month]} at open"
    else:
        entry_rec = f"Enter {best_offset} trading day{'s' if best_offset>1 else ''} AFTER 1st of {MON[month]} — let initial volatility settle"

    exit_rec = (f"Exit around day {peak_day} after entry (avg +{peak_ret:.1f}%). "
                f"Sweet spot: day {sweet_start} to {sweet_end}. "
                + (f"Start declining after day {decline_day}." if decline_day else ""))

    return {
        "month":              month,
        "month_name":         MON[month],
        "base_hold_days":     base_hold,
        "best_entry_offset":  best_offset,
        "entry_label":        entry_label,
        "entry_recommendation": entry_rec,
        "peak_return_day":    peak_day,
        "peak_avg_return_pct": r2(peak_ret),
        "sweet_spot_start_day": sweet_start,
        "sweet_spot_end_day":   sweet_end,
        "decline_after_day":    decline_day,
        "exit_recommendation":  exit_rec,
        "entry_analysis":     list(entry_analysis.values()),
        "exit_curve":         exit_curve,
        "best_entry_stats":   best_e,
    }


def analyze_one_stock(sym, group):
    """Full analysis for one stock. Returns result dict or None."""
    df = get_stock_df(group)

    if len(df) < 250:
        return None
    c_last = float(df["c"].iloc[-1])
    if c_last < MIN_PRICE:
        return None
    tv = df["c"].iloc[-60:] * df["v"].iloc[-60:]
    if float(tv.mean()) < MIN_TURNOVER:
        return None

    # Find qualifying seasonal months
    seasonal = find_seasonal_months(df)
    if not seasonal:
        return None

    # For each seasonal month, do timing analysis
    timing_results = []
    for s in seasonal:
        timing = analyze_entry_exit(df, s["month"], s["hold"])
        if timing is None:
            continue
        timing.update({
            "season_win_rate":  s["win_rate"],
            "season_avg_ret":   s["avg_ret"],
            "season_min_ret":   s["min_ret"],
            "season_n_years":   s["n"],
            "season_years":     s["years"],
        })
        timing_results.append(timing)

    if not timing_results:
        return None

    return {
        "sym":          sym,
        "price":        r2(c_last),
        "n_months":     len(timing_results),
        "timing":       timing_results,
    }


def main():
    print("=" * 60)
    print("UC2 Seasonal Timing Optimizer")
    print(f"Started: {now}")
    print("=" * 60)

    all_data = load_stock_data()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results = []
    skipped = 0
    for i, sym in enumerate(syms):
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} seasonal stocks so far")
        try:
            res = analyze_one_stock(sym, grouped.get_group(sym))
            if res:
                results.append(res)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1

    # Sort by number of qualifying months then by first month's peak return
    results.sort(key=lambda x: (
        -x["n_months"],
        -max((t["peak_avg_return_pct"] or 0) for t in x["timing"])
    ))

    output = {
        "generated_at":   now,
        "n_stocks":       len(results),
        "n_skipped":      skipped,
        "description":    "Optimal entry/exit timing for UC2 seasonal stocks. entry_offset=0 means 1st trading day of month.",
        "stocks":         results,
    }

    out_path = OUT / "uc2_timing.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {out_path}")
    print(f"  Seasonal stocks: {len(results)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total months analyzed: {sum(s['n_months'] for s in results)}")
    print("\nTop 10 by peak return:")
    all_timing = [(s["sym"], t["month_name"], t["peak_avg_return_pct"], t["best_entry_offset"], t["peak_return_day"])
                  for s in results for t in s["timing"]]
    for sym, mo, pk, eo, pd_ in sorted(all_timing, key=lambda x: -(x[2] or 0))[:10]:
        print(f"  {sym:<14} {mo:>3}: peak +{pk:.1f}% on day {pd_:2d}, entry offset {eo:+d}d")


if __name__ == "__main__":
    main()
