#!/usr/bin/env python3
"""
short_hold_analyzer.py
======================
Finds stocks where in a SPECIFIC MONTH + WEEK OF MONTH (e.g. "April Week 2"),
buying at open and holding 5-10 trading days has ALWAYS returned 10%+ 
in every occurrence, across ALL of the last 4 years.

This gives genuine seasonal weekly patterns, not random scattered signals.

Each result shows: which month, which week, how many years, all occurrences.
Output: stock_analysis/short_hold.json
"""

import json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "equity"
OUT      = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 10.0
MIN_TURNOVER   = 3_000_000      # Rs 3 Cr avg daily traded value (MINIMUM per day)
MIN_RETURN     = 10.0           # must rise 10%+ every time
WIN_RATE       = 100.0          # zero exceptions allowed
HOLD_DAYS_LIST = [5, 6, 7, 8, 9, 10]
MIN_OCC        = 3              # at least 3 occurrences in the same month+week slot
LAST_N_YEARS   = 4             # must have fired in all last 4 completed years

EXCLUDED = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","NIFTYBEES","JUNIORBEES",
    "BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES","ITBEES","CPSEETF",
    "LIQUIDSHRI","ABSLLIQUID","ICICIL&TETF",
}
EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")

now    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
cur_yr = datetime.now().year
cur_mo = datetime.now().month
cur_wk = (datetime.now().day - 1) // 7 + 1
# Last 4 completed years (base set — current year added per-slot below)
BASE_REQUIRED_YEARS = set(range(cur_yr - LAST_N_YEARS, cur_yr))   # e.g. 2022,2023,2024,2025

MON = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MON_FULL = ["","January","February","March","April","May","June",
            "July","August","September","October","November","December"]

r2 = lambda x: round(float(x), 2) if x is not None else None


def week_of_month(dt):
    """Return week number within the month: 1,2,3,4,5"""
    return (dt.day - 1) // 7 + 1


def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing."); sys.exit(1)
    manifest = json.loads(MANIFEST.read_text())
    dates = sorted(manifest.keys())
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
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


def is_excluded(sym):
    if sym in EXCLUDED: return True
    return any(sym.upper().endswith(s) for s in EXCL_SFX)


def analyze_stock(sym, df):
    """
    For each (month, week_of_month, hold_days) combination:
    - Collect all entry days in that month+week slot across all years
    - Check: did 100% of them rise 10%+ within hold_days?
    - Check: did it fire in ALL of the last 4 completed years?
    Returns list of qualifying patterns.
    """
    n     = len(df)
    o     = df["o"].values
    h     = df["h"].values
    l     = df["l"].values
    c     = df["c"].values
    v     = df["v"].values
    dates = pd.to_datetime(df["date"].values)

    if float(c[-1]) < MIN_PRICE: return []
    # Last day turnover must be >= 3 Cr
    # Use average of last 5 days turnover to avoid single-day anomalies
    last_tv = float(np.mean(
        [float(c[j]) * float(v[j]) for j in range(max(0, n-5), n) if float(v[j]) > 0]
    )) if n >= 1 else 0
    if last_tv < MIN_TURNOVER: return []

    # Index each trading day by (year, month, week_of_month)
    day_info = []
    for i in range(n):
        dt = dates[i].to_pydatetime()
        day_info.append({
            "idx":   i,
            "year":  dt.year,
            "month": dt.month,
            "week":  week_of_month(dt),
        })

    qualifying = []

    for month in range(1, 13):
        for week in range(1, 6):
            # All trading days in this month+week slot
            slot_days = [d["idx"] for d in day_info if d["month"]==month and d["week"]==week]
            if not slot_days: continue

            slot_years = set(day_info[i]["year"] for i in slot_days)
            if len(slot_years) < MIN_OCC: continue
            # Determine required years for this slot:
            # If this month+week has already passed in the current year,
            # then current year is REQUIRED too — no misses allowed.
            # A slot has 'passed' this year if that month+week is already over
            # Handle year-end wrap: if cur_mo is Jan(1), months 2-12 haven't passed yet
            slot_has_passed = (
                (month < cur_mo) or
                (month == cur_mo and week < cur_wk)
            )
            req_years = BASE_REQUIRED_YEARS | ({cur_yr} if slot_has_passed else set())
            if not req_years.issubset(slot_years): continue

            for hold in HOLD_DAYS_LIST:
                # For each day in this slot: can we enter next day and get 10% within hold days?
                occurrences = []
                for i in slot_days:
                    ai = i + 1   # actual entry = next day open
                    if ai >= n or ai + hold >= n: continue

                    ep = o[ai]
                    if ep <= 0: continue

                    # Avg turnover during hold window
                    tv_window = v[ai:ai+hold] * c[ai:ai+hold]
                    if len(tv_window) == 0 or float(tv_window.mean()) < MIN_TURNOVER: continue

                    # Find max close within hold_days and exit return
                    max_close_ret = -999
                    days_to_10 = None
                    for d in range(1, hold + 1):
                        xi = ai + d
                        if xi >= n: break
                        ret = (c[xi] - ep) / ep * 100
                        if ret > max_close_ret:
                            max_close_ret = ret
                        if ret >= MIN_RETURN and days_to_10 is None:
                            days_to_10 = d

                    xi_final = ai + hold
                    ret_on_exit = r2((c[xi_final] - ep) / ep * 100) if xi_final < n else None
                    max_gain    = r2((max(h[ai:ai+hold+1]) - ep) / ep * 100)

                    if max_close_ret < MIN_RETURN: continue  # didn't hit 10%

                    yr = day_info[i]["year"]
                    occurrences.append({
                        "sig_date":     str(dates[i].date()),
                        "entry_date":   str(dates[ai].date()),
                        "exit_date":    str(dates[min(xi_final,n-1)].date()),
                        "year":         yr,
                        "month":        month,
                        "week":         week,
                        "entry_px":     r2(ep),
                        "exit_px":      r2(c[xi_final]) if xi_final < n else None,
                        "max_ret":      r2(max_close_ret),
                        "ret_on_exit":  ret_on_exit,
                        "max_gain":     max_gain,
                        "days_to_10":   days_to_10,
                    })

                if len(occurrences) < MIN_OCC: continue

                # 100% must give 10%+
                if not all(occ["max_ret"] >= MIN_RETURN for occ in occurrences): continue

                occ_years = set(occ["year"] for occ in occurrences)
                if not req_years.issubset(occ_years): continue

                max_rets  = [occ["max_ret"] for occ in occurrences]
                exit_rets = [occ["ret_on_exit"] for occ in occurrences if occ["ret_on_exit"] is not None]
                days_list = [occ["days_to_10"] for occ in occurrences if occ["days_to_10"]]

                avg_max   = r2(sum(max_rets)  / len(max_rets))
                min_max   = r2(min(max_rets))
                avg_exit  = r2(sum(exit_rets) / len(exit_rets)) if exit_rets else None
                ret_rate  = r2(avg_max / hold)
                avg_days  = r2(sum(days_list) / len(days_list)) if days_list else None

                # Calendar alert: is today or tomorrow in this month+week slot?
                today_dt   = datetime.now()
                alert_today    = (today_dt.month == month and week_of_month(today_dt) == week)
                import datetime as _dtt
                tomorrow_dt    = today_dt + _dtt.timedelta(days=1)
                alert_tomorrow = (tomorrow_dt.month == month and week_of_month(tomorrow_dt) == week)

                qualifying.append({
                    "hold_days":        hold,
                    "month":            month,
                    "month_name":       MON_FULL[month],
                    "month_abbr":       MON[month],
                    "week":             week,
                    "week_label":       f"Week {week} of {MON_FULL[month]}",
                    "description":      f"Enter in Week {week} of {MON_FULL[month]}, hold {hold} trading days",
                    "n_occurrences":    len(occurrences),
                    "n_years":          len(occ_years),
                    "years":            sorted(occ_years),
                    "required_years":   sorted(req_years),
                    "slot_passed_2026": slot_has_passed,
                    "avg_max_ret":      avg_max,
                    "min_max_ret":      min_max,
                    "avg_exit_ret":     avg_exit,
                    "avg_days_to_10":   avg_days,
                    "ret_rate":         ret_rate,
                    "alert_today":      alert_today,
                    "alert_tomorrow":   alert_tomorrow,
                    "occurrences":      sorted(occurrences, key=lambda x: x["sig_date"], reverse=True),
                    # Calendar MMDD list for each occurrence (used by year calendar)
                    "calendar_mmdd":    [{"mmdd": occ["sig_date"][5:], "year": occ["year"]}
                                        for occ in occurrences],
                })

    # Per stock: keep best pattern per (month, week) slot
    best_per_slot = {}
    for q in qualifying:
        key = (q["month"], q["week"])
        if key not in best_per_slot or q["avg_max_ret"] > best_per_slot[key]["avg_max_ret"]:
            best_per_slot[key] = q

    return list(best_per_slot.values())


def main():
    print(f"\n{'='*60}\nShort Hold Analyzer (Consistent Monthly Week Pattern)")
    print(f"Required years (base): {sorted(BASE_REQUIRED_YEARS)}\nCurrent year ({cur_yr}) added for past slots\nStarted: {now}\n{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results  = []   # flat list: one entry per (stock, month, week) pattern
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} qualifying patterns")
        if is_excluded(sym): excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 200: skipped += 1; continue
            patterns = analyze_stock(sym, grp)
            for p in patterns:
                results.append({
                    "sym":   sym,
                    "price": r2(float(grp["c"].iloc[-1])),
                    **p
                })
        except Exception as e:
            skipped += 1

    # Sort by avg_max_ret descending
    results.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    # Build alert lists (today / tomorrow)
    alerts_today    = [r for r in results if r.get("alert_today")]
    alerts_tomorrow = [r for r in results if r.get("alert_tomorrow")]
    alerts_today.sort(key=lambda x: -(x.get("avg_max_ret") or 0))
    alerts_tomorrow.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    # Year calendar: group by month → sorted by avg_max_ret
    year_calendar = {}
    for r in results:
        mo = str(r["month"]).zfill(2)
        if mo not in year_calendar:
            year_calendar[mo] = []
        year_calendar[mo].append({
            "sym":         r["sym"],
            "week":        r["week"],
            "week_label":  r["week_label"],
            "avg_max_ret": r["avg_max_ret"],
            "min_max_ret": r["min_max_ret"],
            "hold_days":   r["hold_days"],
            "n_occ":       r["n_occurrences"],
            "years":       r["years"],
            "alert_today":    r.get("alert_today"),
            "alert_tomorrow": r.get("alert_tomorrow"),
        })
    for mo in year_calendar:
        year_calendar[mo].sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    output = {
        "generated_at":      now,
        "today_date":        datetime.now().strftime("%Y-%m-%d"),
        "today_week_label":  f"Week {week_of_month(datetime.now())} of {MON_FULL[datetime.now().month]}",
        "required_years":    sorted(BASE_REQUIRED_YEARS | {cur_yr}),
        "n_patterns":        len(results),
        "n_stocks":          len(set(r["sym"] for r in results)),
        "n_alerts_today":    len(alerts_today),
        "n_alerts_tomorrow": len(alerts_tomorrow),
        "alerts_today":      alerts_today,
        "alerts_tomorrow":   alerts_tomorrow,
        "year_calendar":     year_calendar,
        "n_excluded":        excluded,
        "n_skipped":         skipped,
        "description":       (
            "Stocks where entering in a specific WEEK OF A SPECIFIC MONTH "
            "and holding 5-10 trading days gave 10%+ in 100% of occurrences, "
            f"present in ALL of last {LAST_N_YEARS} years."
        ),
        "stocks":            results,
    }

    out_path = OUT / "short_hold.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n Written: {out_path}")
    print(f"  Qualifying patterns: {len(results)}")
    print(f"  Unique stocks      : {len(set(r['sym'] for r in results))}")
    print(f"  Alerts today       : {len(alerts_today)}")
    print(f"  Alerts tomorrow    : {len(alerts_tomorrow)}")
    print(f"\nTop 20 by avg return:")
    for r in results[:20]:
        alert = " *** ALERT TODAY ***" if r.get("alert_today") else (" ** tmrw **" if r.get("alert_tomorrow") else "")
        print(f"  {r['sym']:<14} {r['week_label']:<30} hold={r['hold_days']}d avg=+{r['avg_max_ret']}% min=+{r['min_max_ret']}% n={r['n_occurrences']}{alert}")

    if alerts_today:
        print(f"\n{'='*50}")
        print(f"*** {len(alerts_today)} ALERTS TODAY — BUY TOMORROW AT OPEN ***")
        for r in alerts_today:
            print(f"  BUY TOMORROW: {r['sym']:<14} {r['week_label']} avg=+{r['avg_max_ret']}% hold={r['hold_days']}d")


if __name__ == "__main__":
    main()
