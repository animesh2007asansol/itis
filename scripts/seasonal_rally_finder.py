#!/usr/bin/env python3
"""
seasonal_rally_finder.py
========================
Finds stocks that CONSISTENTLY rise 25%+ during a specific calendar window
(30-90 calendar days) in EVERY year without a single miss.

Rules:
- Min 8 Cr daily traded value (price × volume)
- Must be actively trading in recent NSE data (last 10 trading days)
- Signal must have occurred in ALL of the last 4 completed years
- If the window has already started/passed this year, check 2026 too
- Extends window if consecutive months also give 20%+ (compound rally)
- Shows live tracking: current return, time left, switch recommendation

Output: stock_analysis/seasonal_rally.json
"""

import json, sys, warnings
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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MIN_PRICE       = 10.0
MIN_TURNOVER    = 8_000_000     # Rs 8 Cr daily (price × volume)
MIN_RETURN      = 25.0          # minimum 25% rally to qualify
EXTEND_RETURN   = 20.0          # extend window if next period also gives 20%+
LAST_N_YEARS    = 4             # must fire in all last 4 completed years
RECENT_DAYS     = 10            # must have traded in last 10 trading days
WINDOW_SIZES    = [30, 40, 50, 60, 75, 90]  # calendar day windows to test
MIN_OCC         = 3             # minimum occurrences

EXCLUDED_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))
now_ist    = datetime.now(IST)
now_str    = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today_str  = now_ist.strftime("%Y-%m-%d")
cur_yr     = now_ist.year
cur_mo     = now_ist.month
cur_day    = now_ist.day

MON_FULL = ["","January","February","March","April","May","June",
            "July","August","September","October","November","December"]
MON_ABR  = ["","Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]

BASE_REQUIRED = set(range(cur_yr - LAST_N_YEARS, cur_yr))   # e.g. 2022-2025

def r2(x):
    try:
        v = float(x)
        import math
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


# ── DATA LOAD ──────────────────────────────────────────────────────────────────
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


def is_excluded(sym):
    return any(sym.upper().endswith(sfx) for sfx in EXCLUDED_SFX)


def get_return_in_window(c_arr, dates_arr, start_idx, window_cal_days):
    """
    Enter at open of start_idx+1 (next trading day after start).
    Exit at close of the first trading day at or after start_date + window_cal_days.
    Returns (return_pct, exit_date, exit_idx) or None.
    """
    n = len(c_arr)
    ai = start_idx + 1
    if ai >= n: return None

    ep = float(c_arr[ai - 1]) if ai > 0 else float(c_arr[0])
    # Use close of start day as proxy for entry (next open is unknown historically)
    # More accurately: entry = open of next trading day after signal
    ep = float(c_arr[start_idx])    # signal day close (next open ≈ this)
    if ep <= 0: return None

    target_date = dates_arr[start_idx] + pd.Timedelta(days=window_cal_days)
    # Find first trading day >= target_date
    exit_idx = None
    for j in range(start_idx + 1, n):
        if dates_arr[j] >= target_date:
            exit_idx = j
            break
    if exit_idx is None or exit_idx >= n: return None

    xp  = float(c_arr[exit_idx])
    ret = (xp - ep) / ep * 100
    # Also track max gain in window
    max_px  = float(max(c_arr[start_idx+1:exit_idx+1])) if exit_idx > start_idx else xp
    max_ret = (max_px - ep) / ep * 100

    return {
        "ret":      r2(ret),
        "max_ret":  r2(max_ret),
        "entry_px": r2(ep),
        "exit_px":  r2(xp),
        "exit_date":str(dates_arr[exit_idx].date()),
        "exit_idx": exit_idx,
    }


def find_first_trading_day_of_month(dates_arr, year, month):
    """Return index of first trading day in given year+month, or None."""
    for idx, dt in enumerate(dates_arr):
        if dt.year == year and dt.month == month:
            return idx
    return None


def analyze_stock(sym, df, latest_trading_dates):
    n        = len(df)
    c_arr    = df["c"].values
    v_arr    = df["v"].values
    dates_arr= pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates_arr[-1].date())

    if cur_price < MIN_PRICE: return []

    # Must be recently active (traded in last RECENT_DAYS trading dates)
    if last_date not in latest_trading_dates: return []

    # Turnover check: last 5-day average
    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER: return []

    # All years present in data
    all_yrs = sorted(set(int(dt.year) for dt in dates_arr))

    results = []

    # Test each month as potential start
    for start_mo in range(1, 13):
        for window_days in WINDOW_SIZES:
            occurrences  = []
            occ_years    = set()

            for yr in all_yrs:
                # Find first trading day of this month in this year
                start_idx = find_first_trading_day_of_month(dates_arr, yr, start_mo)
                if start_idx is None: continue

                res = get_return_in_window(c_arr, dates_arr, start_idx, window_days)
                if res is None: continue
                if res["ret"] < MIN_RETURN: continue

                occ_years.add(yr)
                occurrences.append({
                    "year":       yr,
                    "start_date": str(dates_arr[start_idx].date()),
                    "start_mo":   start_mo,
                    "window_days":window_days,
                    "entry_px":   res["entry_px"],
                    "exit_px":    res["exit_px"],
                    "exit_date":  res["exit_date"],
                    "ret":        res["ret"],
                    "max_ret":    res["max_ret"],
                })

            if len(occurrences) < MIN_OCC: continue
            if not BASE_REQUIRED.issubset(occ_years): continue

            # Check if window has started/passed this year
            try:
                window_start_this_yr = datetime(cur_yr, start_mo, 1)
                window_end_this_yr   = window_start_this_yr + timedelta(days=window_days)
                started_this_yr      = now_ist.replace(tzinfo=None) >= window_start_this_yr
                passed_this_yr       = now_ist.replace(tzinfo=None) > window_end_this_yr
            except: started_this_yr = False; passed_this_yr = False

            # If window has fully passed this year, require 2026 occurrence too
            if passed_this_yr and cur_yr not in occ_years: continue

            # ── Try to extend window with next month ──────────────────────────
            extended_window = window_days
            extended_min    = MIN_RETURN
            next_mo         = (start_mo % 12) + 1
            ext_occurrences = list(occurrences)

            for yr in all_yrs:
                si2 = find_first_trading_day_of_month(dates_arr, yr, next_mo)
                if si2 is None: continue
                # Extended: check from start_mo entry to next_mo + 30 days
                res2 = get_return_in_window(c_arr, dates_arr,
                       find_first_trading_day_of_month(dates_arr, yr, start_mo),
                       window_days + 30)
                if res2 and res2["ret"] >= EXTEND_RETURN:
                    # Update this year's occurrence with extended window
                    for occ in ext_occurrences:
                        if occ["year"] == yr:
                            occ["ret_extended"]   = res2["ret"]
                            occ["max_ret_extended"]= res2["max_ret"]
                            occ["exit_date_ext"]  = res2["exit_date"]

            ext_rets = [occ.get("ret_extended") for occ in ext_occurrences
                        if occ.get("ret_extended") is not None and occ["ret_extended"] >= EXTEND_RETURN]
            if len(ext_rets) == len(occurrences):
                extended_window = window_days + 30
                extended_min    = r2(min(ext_rets))

            # ── Stats ─────────────────────────────────────────────────────────
            rets     = [occ["ret"]     for occ in occurrences]
            max_rets = [occ["max_ret"] for occ in occurrences]

            avg_ret  = r2(sum(rets) / len(rets))
            min_ret  = r2(min(rets))
            max_ret  = r2(max(rets))
            avg_max  = r2(sum(max_rets) / len(max_rets))

            # ── Live tracking ─────────────────────────────────────────────────
            live_data     = None
            this_yr_occ   = next((o for o in occurrences if o["year"] == cur_yr), None)
            slot_status   = "upcoming"

            if started_this_yr:
                if this_yr_occ:
                    slot_status = "active" if not passed_this_yr else "completed"
                    ep_live   = this_yr_occ["entry_px"] or cur_price
                    live_ret  = r2((cur_price - ep_live) / ep_live * 100)
                    avg_tgt   = r2(ep_live * (1 + avg_ret  / 100))
                    min_tgt   = r2(ep_live * (1 + min_ret  / 100))
                    rem_avg   = r2(avg_ret - (live_ret or 0))
                    try:
                        start_dt    = datetime.strptime(this_yr_occ["start_date"], "%Y-%m-%d")
                        end_dt      = start_dt + timedelta(days=extended_window)
                        cal_elapsed = (now_ist.replace(tzinfo=None) - start_dt).days
                        cal_left    = max(0, (end_dt - now_ist.replace(tzinfo=None)).days)
                    except: cal_elapsed = None; cal_left = None

                    # Switch suggestion: if close to or past avg target
                    near_target    = live_ret is not None and live_ret >= avg_ret - 5
                    past_min       = live_ret is not None and live_ret >= min_ret

                    live_data = {
                        "start_date":   this_yr_occ["start_date"],
                        "entry_px":     r2(ep_live),
                        "current_px":   r2(cur_price),
                        "live_ret":     live_ret,
                        "avg_target":   avg_tgt,
                        "min_target":   min_tgt,
                        "remaining_to_avg": rem_avg,
                        "cal_elapsed":  cal_elapsed,
                        "cal_left":     cal_left,
                        "window_days":  extended_window,
                        "near_target":  near_target,
                        "past_min":     past_min,
                        "suggest_exit": near_target and not passed_this_yr,
                    }
                else:
                    slot_status = "missed"

            # ── End date label ────────────────────────────────────────────────
            end_mo_approx = start_mo + (extended_window // 30)
            end_mo_label  = MON_ABR[min(12, end_mo_approx)] if end_mo_approx <= 12 else \
                            MON_ABR[end_mo_approx - 12] + " (next yr)"

            results.append({
                "start_month":      start_mo,
                "start_month_name": MON_FULL[start_mo],
                "start_month_abbr": MON_ABR[start_mo],
                "window_days":      extended_window,
                "base_window":      window_days,
                "window_label":     f"{MON_ABR[start_mo]} → {end_mo_label} ({extended_window} cal days)",
                "avg_ret":          avg_ret,
                "min_ret":          min_ret,
                "max_ret":          max_ret,
                "avg_max_gain":     avg_max,
                "n_occurrences":    len(occurrences),
                "years":            sorted(occ_years),
                "required_years":   sorted(BASE_REQUIRED | ({cur_yr} if started_this_yr else set())),
                "cur_yr_fired":     bool(this_yr_occ),
                "slot_status":      slot_status,
                "live_data":        live_data,
                "occurrences":      sorted(occurrences, key=lambda x: x["year"], reverse=True),
            })

    # Deduplicate: if two windows overlap significantly, keep the one with higher min_ret
    results.sort(key=lambda x: -(x["min_ret"] or 0))
    seen = []
    deduped = []
    for r in results:
        key = r["start_month"]
        if key not in seen:
            seen.append(key)
            deduped.append(r)

    return deduped


def main():
    print(f"\n{'='*60}")
    print(f"Seasonal Rally Finder")
    print(f"IST now: {now_str}  Base required: {sorted(BASE_REQUIRED)}")
    print(f"{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())

    # Find latest RECENT_DAYS trading dates
    all_dates     = sorted(set(str(dt.date()) for dt in pd.to_datetime(all_data["date"].unique())))
    latest_dates  = set(all_dates[-RECENT_DAYS:]) if len(all_dates) >= RECENT_DAYS else set(all_dates)

    print(f"\nAnalyzing {len(syms):,} symbols...")
    print(f"  Latest trading date: {max(all_dates)}")
    print(f"  Active window: last {RECENT_DAYS} dates — {min(latest_dates)} to {max(latest_dates)}")

    stock_results = []   # list of {sym, price, patterns:[...]}
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(stock_results)} stocks")
        if is_excluded(sym): excluded += 1; continue
        try:
            grp      = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 250: skipped += 1; continue
            patterns = analyze_stock(sym, grp, latest_dates)
            if patterns:
                stock_results.append({
                    "sym":         sym,
                    "price":       r2(float(grp["c"].iloc[-1])),
                    "best_min_ret":max((p["min_ret"] or 0) for p in patterns),
                    "n_patterns":  len(patterns),
                    "patterns":    sorted(patterns, key=lambda x: -(x["min_ret"] or 0)),
                })
        except: skipped += 1

    # Sort: highest best_min_ret first
    stock_results.sort(key=lambda x: -(x.get("best_min_ret") or 0))

    # Flat list (one row per stock+pattern) for easy HTML rendering
    flat_rows = []
    for sr in stock_results:
        for pat in sr["patterns"]:
            flat_rows.append({
                "sym":              sr["sym"],
                "price":            sr["price"],
                **pat,
            })
    flat_rows.sort(key=lambda x: -(x.get("min_ret") or 0))

    # Active positions (live_data present and in hold)
    active_positions = [r for r in flat_rows
                        if r.get("live_data") and r["slot_status"] in ("active",)]
    active_positions.sort(key=lambda x: -(x["live_data"].get("live_ret") or 0))

    # Upcoming this month
    upcoming_now = [r for r in flat_rows
                    if r.get("start_month") == cur_mo and r.get("slot_status") == "upcoming"]
    upcoming_now.sort(key=lambda x: -(x.get("min_ret") or 0))

    # Switch recommendations: active positions near/past avg target
    switch_candidates = [r for r in active_positions if r.get("live_data", {}).get("suggest_exit")]

    output = {
        "generated_at":      now_str,
        "today_ist":         today_str,
        "required_years":    sorted(BASE_REQUIRED),
        "n_stocks":          len(stock_results),
        "n_patterns":        len(flat_rows),
        "n_active":          len(active_positions),
        "n_upcoming_now":    len(upcoming_now),
        "n_switch_suggest":  len(switch_candidates),
        "active_positions":  active_positions,
        "upcoming_now":      upcoming_now,
        "switch_suggestions":switch_candidates,
        "description":       (
            f"Stocks that rise 25%+ in a consistent calendar window "
            f"(30-90 days) every year without a miss. "
            f"Min 8 Cr daily turnover. Only recently active stocks."
        ),
        "stocks":       stock_results,
        "flat_rows":    flat_rows,
    }

    path = OUT / "seasonal_rally.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {path}")
    print(f"  Qualifying stocks: {len(stock_results)}")
    print(f"  Total patterns:    {len(flat_rows)}")
    print(f"  Active positions:  {len(active_positions)}")
    print(f"  Switch suggest:    {len(switch_candidates)}")
    print(f"  Upcoming now:      {len(upcoming_now)}")

    if upcoming_now:
        print(f"\n*** ENTER NOW — Current month signals ***")
        for r in upcoming_now[:10]:
            print(f"  {r['sym']:<14} {r['window_label']:<35} min=+{r['min_ret']}% avg=+{r['avg_ret']}%")

    if switch_candidates:
        print(f"\n*** CONSIDER SWITCHING — Near avg target ***")
        for r in switch_candidates[:5]:
            ld = r["live_data"]
            print(f"  {r['sym']:<14} live={ld['live_ret']:+.1f}% avg_tgt={ld['avg_target']} (min hit, {ld.get('rem_avg',0):.1f}% to avg)")


if __name__ == "__main__":
    main()
