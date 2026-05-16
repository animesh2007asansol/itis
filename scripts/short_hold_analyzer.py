#!/usr/bin/env python3
"""
short_hold_analyzer.py
======================
Finds stocks where entering in a specific WEEK OF A SPECIFIC MONTH
and holding 5-10 trading days gave 10%+ in 100% of occurrences,
present in ALL of the last 4 completed years.

Also tracks live return if a signal fired this year.
Output: stock_analysis/short_hold.json
"""

import json, sys, warnings, datetime
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 10.0
MIN_TURNOVER   = 3_000_000   # Rs 3 Cr avg daily turnover
MIN_RETURN     = 10.0
WIN_RATE       = 100.0       # every occurrence must give 10%+
HOLD_DAYS_LIST = [5, 6, 7, 8, 9, 10]
MIN_OCC        = 3
LAST_N_YEARS   = 4

EXCLUDED_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")
MON = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MON_FULL = ["","January","February","March","April","May","June",
            "July","August","September","October","November","December"]

now     = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
cur_yr  = datetime.datetime.now().year
cur_mo  = datetime.datetime.now().month
cur_wk  = (datetime.datetime.now().day - 1) // 7 + 1

# Last 4 completed calendar years
BASE_REQUIRED_YEARS = set(range(cur_yr - LAST_N_YEARS, cur_yr))

r2 = lambda x: round(float(x), 2) if x is not None else None


def week_of_month(dt):
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
    if not frames: print("ERROR: No data loaded."); sys.exit(1)
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


def analyze_stock(sym, df):
    n       = len(df)
    o_arr   = df["o"].values
    h_arr   = df["h"].values
    l_arr   = df["l"].values
    c_arr   = df["c"].values
    v_arr   = df["v"].values
    dates   = pd.to_datetime(df["date"].values)

    # Price check
    if float(c_arr[-1]) < MIN_PRICE:
        return []

    # Turnover check: average last 5 days
    last5_tv = [float(c_arr[j]) * float(v_arr[j])
                for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not last5_tv or (sum(last5_tv) / len(last5_tv)) < MIN_TURNOVER:
        return []

    # Build day info
    day_info = []
    for i in range(n):
        dt = dates[i].to_pydatetime()
        day_info.append({
            "idx":   i,
            "year":  dt.year,
            "month": dt.month,
            "week":  (dt.day - 1) // 7 + 1,
        })

    qualifying = []

    for month in range(1, 13):
        for week in range(1, 6):
            # All trading days in this month+week slot
            slot_days = [d["idx"] for d in day_info
                         if d["month"] == month and d["week"] == week]
            if not slot_days: continue

            slot_years = set(day_info[i]["year"] for i in slot_days)
            if len(slot_years) < MIN_OCC: continue

            # Must have fired in all 4 completed years
            if not BASE_REQUIRED_YEARS.issubset(slot_years): continue

            for hold in HOLD_DAYS_LIST:
                occurrences = []
                for i in slot_days:
                    ai = i + 1         # entry = next day open
                    xi = ai + hold     # exit  = hold days later
                    if ai >= n or xi >= n: continue

                    ep = float(o_arr[ai])
                    if ep <= 0: continue

                    # Turnover during hold window must be ≥ MIN_TURNOVER
                    tv_window = [float(c_arr[j]) * float(v_arr[j])
                                 for j in range(ai, min(xi+1, n)) if float(v_arr[j]) > 0]
                    if not tv_window or (sum(tv_window)/len(tv_window)) < MIN_TURNOVER:
                        continue

                    # Max close return within hold days
                    max_ret = -999.0
                    days_to_10 = None
                    for d in range(1, hold + 1):
                        if ai + d >= n: break
                        ret = (float(c_arr[ai+d]) - ep) / ep * 100
                        if ret > max_ret:
                            max_ret = ret
                        if ret >= MIN_RETURN and days_to_10 is None:
                            days_to_10 = d

                    if max_ret < MIN_RETURN: continue

                    ret_on_exit = r2((float(c_arr[xi]) - ep) / ep * 100)
                    max_gain    = r2((float(max(h_arr[ai:xi+1])) - ep) / ep * 100)

                    occurrences.append({
                        "sig_date":    str(dates[i].date()),
                        "entry_date":  str(dates[ai].date()),
                        "exit_date":   str(dates[xi].date()),
                        "year":        int(day_info[i]["year"]),
                        "month":       month,
                        "week":        week,
                        "entry_px":    r2(ep),
                        "exit_px":     r2(float(c_arr[xi])),
                        "max_ret":     r2(max_ret),
                        "ret_on_exit": ret_on_exit,
                        "max_gain":    max_gain,
                        "days_to_10":  days_to_10,
                    })

                if len(occurrences) < MIN_OCC: continue

                # 100% must give MIN_RETURN
                if not all(occ["max_ret"] >= MIN_RETURN for occ in occurrences):
                    continue

                occ_years = set(occ["year"] for occ in occurrences)

                # All 4 completed years must be present in actual occurrences too
                if not BASE_REQUIRED_YEARS.issubset(occ_years): continue

                # Compute stats
                max_rets  = [occ["max_ret"]     for occ in occurrences]
                exit_rets = [occ["ret_on_exit"]  for occ in occurrences
                             if occ["ret_on_exit"] is not None]
                days_list = [occ["days_to_10"]   for occ in occurrences
                             if occ["days_to_10"] is not None]

                avg_max  = r2(sum(max_rets) / len(max_rets))
                min_max  = r2(min(max_rets))
                avg_exit = r2(sum(exit_rets) / len(exit_rets)) if exit_rets else None
                avg_days = r2(sum(days_list) / len(days_list)) if days_list else None
                ret_rate = r2(avg_max / hold)

                # Slot status
                slot_has_passed = (month < cur_mo) or (month == cur_mo and week < cur_wk)
                cur_yr_in_data  = cur_yr in occ_years

                if slot_has_passed and cur_yr_in_data:
                    slot_status = "active"
                elif slot_has_passed and not cur_yr_in_data:
                    slot_status = "missed"
                else:
                    slot_status = "upcoming"

                # Alert flags
                alert_today    = (month == cur_mo and week == cur_wk)
                tomorrow       = datetime.datetime.now() + datetime.timedelta(days=1)
                alert_tomorrow = (month == tomorrow.month and week == (tomorrow.day-1)//7+1)

                # Live tracking: find this year's occurrence
                live_data = None
                cur_yr_occ = next((occ for occ in occurrences if occ["year"] == cur_yr), None)
                if cur_yr_occ:
                    ep_live = cur_yr_occ.get("entry_px") or 0
                    if ep_live > 0:
                        cur_px   = float(c_arr[-1])
                        live_ret = r2((cur_px - ep_live) / ep_live * 100)
                        hold_cal = round(hold * 1.5)
                        try:
                            entry_dt    = datetime.datetime.strptime(cur_yr_occ["entry_date"], "%Y-%m-%d")
                            cal_elapsed = (datetime.datetime.now() - entry_dt).days
                            cal_remain  = max(0, hold_cal - cal_elapsed)
                        except:
                            cal_elapsed = None
                            cal_remain  = None

                        avg_tgt = r2(ep_live * (1 + avg_max / 100))
                        min_tgt = r2(ep_live * (1 + min_max / 100))
                        rem_to_avg = r2(avg_max - (live_ret or 0))

                        live_data = {
                            "entry_date":    cur_yr_occ["entry_date"],
                            "entry_px":      r2(ep_live),
                            "current_px":    r2(cur_px),
                            "live_ret":      live_ret,
                            "avg_target_px": avg_tgt,
                            "min_target_px": min_tgt,
                            "remaining_to_avg": rem_to_avg,
                            "cal_elapsed":   cal_elapsed,
                            "cal_remaining": cal_remain,
                            "hold_cal":      hold_cal,
                            "is_in_hold":    (cal_remain or 0) > 0,
                        }

                req_yrs_out = sorted(BASE_REQUIRED_YEARS | ({cur_yr} if slot_has_passed else set()))

                qualifying.append({
                    "hold_days":       hold,
                    "month":           month,
                    "month_name":      MON_FULL[month],
                    "month_abbr":      MON[month],
                    "week":            week,
                    "week_label":      f"Week {week} of {MON_FULL[month]}",
                    "description":     f"Enter Week {week} of {MON_FULL[month]}, hold {hold} trading days",
                    "n_occurrences":   len(occurrences),
                    "n_years":         len(occ_years),
                    "years":           sorted(occ_years),
                    "required_years":  req_yrs_out,
                    "avg_max_ret":     avg_max,
                    "min_max_ret":     min_max,
                    "avg_exit_ret":    avg_exit,
                    "avg_days_to_10":  avg_days,
                    "ret_rate":        ret_rate,
                    "slot_status":     slot_status,
                    "cur_yr_fired":    cur_yr_in_data,
                    "live_data":       live_data,
                    "alert_today":     alert_today,
                    "alert_tomorrow":  alert_tomorrow,
                    "calendar_mmdd":   [{"mmdd": occ["sig_date"][5:], "year": occ["year"]}
                                        for occ in occurrences],
                    "occurrences":     sorted(occurrences, key=lambda x: x["sig_date"], reverse=True),
                })

    # Best per (month, week) slot
    best_per_slot = {}
    for q in qualifying:
        key = (q["month"], q["week"])
        if key not in best_per_slot or q["avg_max_ret"] > best_per_slot[key]["avg_max_ret"]:
            best_per_slot[key] = q

    return list(best_per_slot.values())


def main():
    print(f"\n{'='*60}")
    print(f"Short Hold Analyzer — Consistent Monthly Week Pattern")
    print(f"Base required years: {sorted(BASE_REQUIRED_YEARS)}")
    print(f"Today: {datetime.datetime.now().strftime('%Y-%m-%d')} — Week {cur_wk} of {MON_FULL[cur_mo]}")
    print(f"Started: {now}")
    print(f"{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results  = []
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} qualifying patterns")
        if any(sym.upper().endswith(sfx) for sfx in EXCLUDED_SFX):
            excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 200: skipped += 1; continue
            patterns = analyze_stock(sym, grp)
            for p in patterns:
                results.append({"sym": sym,
                                 "price": r2(float(grp["c"].iloc[-1])),
                                 **p})
        except Exception as e:
            skipped += 1

    results.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    alerts_today    = [r for r in results if r.get("alert_today")]
    alerts_tomorrow = [r for r in results if r.get("alert_tomorrow")]
    alerts_active   = [r for r in results if r.get("slot_status") == "active"]
    alerts_today.sort(key=lambda x: -(x.get("avg_max_ret") or 0))
    alerts_tomorrow.sort(key=lambda x: -(x.get("avg_max_ret") or 0))
    alerts_active.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    # Year calendar
    year_calendar = {}
    for r in results:
        mo = str(r["month"]).zfill(2)
        if mo not in year_calendar:
            year_calendar[mo] = []
        year_calendar[mo].append({
            "sym":          r["sym"],
            "week":         r["week"],
            "week_label":   r["week_label"],
            "avg_max_ret":  r["avg_max_ret"],
            "min_max_ret":  r["min_max_ret"],
            "hold_days":    r["hold_days"],
            "n_occ":        r["n_occurrences"],
            "years":        r["years"],
            "slot_status":  r["slot_status"],
            "alert_today":  r["alert_today"],
            "alert_tomorrow": r["alert_tomorrow"],
        })
    for mo in year_calendar:
        year_calendar[mo].sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    n_patterns = len(results)
    n_stocks   = len(set(r["sym"] for r in results))

    output = {
        "generated_at":      now,
        "today_date":        datetime.datetime.now().strftime("%Y-%m-%d"),
        "today_week_label":  f"Week {cur_wk} of {MON_FULL[cur_mo]}",
        "required_years":    sorted(BASE_REQUIRED_YEARS),
        "n_patterns":        n_patterns,
        "n_stocks":          n_stocks,
        "n_alerts_today":    len(alerts_today),
        "n_alerts_tomorrow": len(alerts_tomorrow),
        "n_active":          len(alerts_active),
        "alerts_today":      alerts_today,
        "alerts_tomorrow":   alerts_tomorrow,
        "alerts_active":     alerts_active,
        "year_calendar":     year_calendar,
        "n_excluded":        excluded,
        "n_skipped":         skipped,
        "description":       (
            "Stocks where entering in a specific week of a month and holding "
            "5-10 trading days gave 10%+ in 100% of occurrences, "
            f"present in ALL last {LAST_N_YEARS} completed years."
        ),
        "stocks": results,
    }

    out_path = OUT / "short_hold.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n✓ Written: {out_path}")
    print(f"  Qualifying patterns : {n_patterns}")
    print(f"  Unique stocks       : {n_stocks}")
    print(f"  Active this period  : {len(alerts_active)}")
    print(f"  Alerts today        : {len(alerts_today)}")
    print(f"  Alerts tomorrow     : {len(alerts_tomorrow)}")
    print(f"  Excluded            : {excluded}")
    print(f"  Skipped             : {skipped}")

    if alerts_today:
        print(f"\n{'='*50}")
        print(f"*** THIS WEEK — ENTER NOW ***")
        for r in alerts_today[:10]:
            print(f"  {r['sym']:<14} {r['week_label']:<30} avg=+{r['avg_max_ret']}% hold={r['hold_days']}d")

    if alerts_active:
        print(f"\n*** ACTIVE SIGNALS — IN HOLD PERIOD ***")
        for r in alerts_active[:10]:
            ld = r.get("live_data") or {}
            ret_str = f"live={ld.get('live_ret',0):+.1f}%" if ld else ""
            print(f"  {r['sym']:<14} {r['week_label']:<28} {ret_str}")

    print(f"\nTop 20 by avg return:")
    for r in results[:20]:
        status = r.get("slot_status","")
        st_tag = " [ACTIVE]" if status=="active" else " [THIS WEEK]" if r.get("alert_today") else ""
        print(f"  {r['sym']:<14} {r['week_label']:<30} hold={r['hold_days']}d avg=+{r['avg_max_ret']}% min=+{r['min_max_ret']}%{st_tag}")


if __name__ == "__main__":
    main()
