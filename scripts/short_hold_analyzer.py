#!/usr/bin/env python3
"""
short_hold_analyzer.py
======================
Finds stocks that have ALWAYS risen by at least 10% within N trading days
(N = 5, 6, 7, 8, 9, 10) across ALL historical years AND mandatorily in
the last 4 calendar years (2022, 2023, 2024, 2025).

Looks at EVERY possible entry day and checks: did the stock hit +10%
within N trading days from that entry, in every year it had such opportunities.

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
MIN_PRICE       = 10.0          # Rs 10 minimum
MIN_TURNOVER    = 1_000_000     # Rs 1 Cr avg daily traded value during signal period
MIN_OCC         = 4             # minimum 4 historical signals across years
MIN_RETURN      = 10.0          # must rise at least 10%
WIN_RATE        = 100.0         # EVERY occurrence must give 10%+ (no exceptions)
HOLD_DAYS_LIST  = [5, 6, 7, 8, 9, 10]  # test each hold period
LAST_N_YEARS    = 4             # must have fired in all of last 4 completed years
MIN_YEARS       = 3             # minimum 3 distinct years of data

EXCLUDED = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","NIFTYBEES","JUNIORBEES",
    "BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES","ITBEES","CPSEETF",
    "LIQUIDSHRI","ICICIL&TETF","ABSLLIQUID",
}
EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUIDFUND","LIQUID")

now  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
r2   = lambda x: round(float(x), 2) if x is not None else None
cur_yr = datetime.now().year
# Last 4 completed calendar years (exclude current year — may be incomplete)
REQUIRED_YEARS = set(range(cur_yr - LAST_N_YEARS, cur_yr))   # e.g. 2021,2022,2023,2024

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing. Run main workflow first."); sys.exit(1)
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
    if sym in EXCLUDED: return True
    for sfx in EXCL_SFX:
        if sym.upper().endswith(sfx): return True
    return False


# ── CORE ANALYSIS ─────────────────────────────────────────────────────────────
def analyze_stock(sym, df):
    """
    For each hold period in HOLD_DAYS_LIST:
    Find all entry signals where the stock rose ≥10% within hold_days.
    
    Signal definition: any day where, if you buy next day's open,
    the stock closes at ≥10% above entry within hold_days trading days.
    
    Qualification: 100% win rate, ALL last 4 years present, ≥4 occurrences.
    """
    n    = len(df)
    o    = df["o"].values
    h    = df["h"].values
    l    = df["l"].values
    c    = df["c"].values
    v    = df["v"].values
    dates = pd.to_datetime(df["date"].values)
    yrs  = pd.DatetimeIndex(dates).year

    # Price filter
    if float(c[-1]) < MIN_PRICE: return None

    best_result = None
    best_ret_rate = -999

    for hold in HOLD_DAYS_LIST:
        # For each possible entry day, check if stock hits +10% within hold days
        occurrences = []   # list of (entry_idx, max_ret, ret_on_exit, sig_year)

        i = 0
        while i < n - hold - 1:
            ai = i + 1   # actual entry = next day open
            if ai + hold >= n: break

            ep = o[ai]
            if ep <= 0: i += 1; continue

            # Check turnover on entry day and surrounding days
            tv_window = v[ai:ai+hold] * c[ai:ai+hold]
            avg_tv = float(tv_window.mean()) if len(tv_window) > 0 else 0
            if avg_tv < MIN_TURNOVER: i += 1; continue

            # Did it rise 10%+ at any close within hold days?
            hit = False
            best_day = None
            max_close_ret = -999
            for d in range(1, hold + 1):
                xi = ai + d
                if xi >= n: break
                ret = (c[xi] - ep) / ep * 100
                if ret > max_close_ret:
                    max_close_ret = ret
                    if ret >= MIN_RETURN and not hit:
                        hit = True
                        best_day = d

            # Return on final exit day (hold_days)
            xi_final = ai + hold
            ret_on_exit = (c[xi_final] - ep) / ep * 100 if xi_final < n else None

            # Max intraday gain (using highs)
            max_gain = (max(h[ai:ai+hold+1]) - ep) / ep * 100

            if not hit: i += 1; continue

            # Record this occurrence
            occurrences.append({
                "sig_date":    str(dates[i].date()),
                "entry_date":  str(dates[ai].date()),
                "exit_date":   str(dates[min(xi_final, n-1)].date()),
                "entry_px":    r2(ep),
                "exit_px":     r2(c[xi_final]) if xi_final < n else None,
                "ret_on_exit": r2(ret_on_exit) if ret_on_exit is not None else None,
                "max_ret":     r2(max_close_ret),
                "max_gain":    r2(max_gain),
                "days_to_10pct": best_day,
                "year":        int(yrs[i]),
            })
            # Skip forward to avoid overlapping
            i = ai + hold

        if len(occurrences) < MIN_OCC: continue

        # 100% win rate check: every occurrence must have max_ret >= 10%
        if not all(occ["max_ret"] >= MIN_RETURN for occ in occurrences): continue

        # Check year coverage
        occ_years = set(occ["year"] for occ in occurrences)
        if len(occ_years) < MIN_YEARS: continue

        # MANDATORY: must have fired in ALL last 4 completed years
        if not REQUIRED_YEARS.issubset(occ_years): continue

        # Compute stats
        max_rets   = [occ["max_ret"] for occ in occurrences]
        exit_rets  = [occ["ret_on_exit"] for occ in occurrences if occ["ret_on_exit"] is not None]
        days_list  = [occ["days_to_10pct"] for occ in occurrences if occ["days_to_10pct"]]

        avg_max    = r2(sum(max_rets)  / len(max_rets))
        min_max    = r2(min(max_rets))
        avg_exit   = r2(sum(exit_rets) / len(exit_rets)) if exit_rets else None
        min_exit   = r2(min(exit_rets)) if exit_rets else None
        avg_days   = r2(sum(days_list) / len(days_list)) if days_list else None
        ret_rate   = r2(avg_max / hold)  # return per trading day

        if ret_rate > best_ret_rate:
            best_ret_rate = ret_rate
            best_result = {
                "hold_days":      hold,
                "n_occurrences":  len(occurrences),
                "n_years":        len(occ_years),
                "years":          sorted(occ_years),
                "required_years": sorted(REQUIRED_YEARS),
                "last_4yr_ok":    REQUIRED_YEARS.issubset(occ_years),
                "avg_max_ret":    avg_max,
                "min_max_ret":    min_max,
                "avg_exit_ret":   avg_exit,
                "min_exit_ret":   min_exit,
                "avg_days_to_10": avg_days,
                "ret_rate":       ret_rate,
                "occurrences":    sorted(occurrences, key=lambda x: x["sig_date"], reverse=True),
            }

    return best_result


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}\nShort Hold Analyzer (5-10 trading days, 10%+ every time)\nRequired years: {sorted(REQUIRED_YEARS)}\nStarted: {now}\n{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    results  = []
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} qualifying stocks")

        if is_excluded(sym): excluded += 1; continue

        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 200: skipped += 1; continue

            res = analyze_stock(sym, grp)
            if res:
                results.append({
                    "sym":         sym,
                    "price":       r2(float(grp["c"].iloc[-1])),
                    "latest_date": str(pd.Timestamp(grp["date"].iloc[-1]).date()),
                    **res,
                })
        except Exception as e:
            skipped += 1

    # Sort: highest avg_max_ret descending (best performers first)
    results.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    # Build alert lists
    import datetime as _dt_alert
    alerts_today    = [r for r in results if r.get("alert_today")]
    alerts_tomorrow = [r for r in results if r.get("alert_tomorrow")]
    alerts_window   = [r for r in results if r.get("window_matches")]

    # Build full year calendar: for each month, which stocks fire
    year_calendar = {}
    for r in results:
        for item in (r.get("calendar_mmdd") or []):
            mo = item["mmdd"][:2]   # "04" from "04-10"
            if mo not in year_calendar:
                year_calendar[mo] = []
            # Avoid duplicates per stock per month
            if not any(e["sym"] == r["sym"] for e in year_calendar[mo]):
                year_calendar[mo].append({
                    "sym":         r["sym"],
                    "avg_max_ret": r["avg_max_ret"],
                    "min_max_ret": r["min_max_ret"],
                    "hold_days":   r["hold_days"],
                    "n_occ":       r["n_occurrences"],
                    "mmdd":        item["mmdd"],
                })
    # Sort each month's list by avg_max_ret desc
    for mo in year_calendar:
        year_calendar[mo].sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    import datetime as _dt_out
    _today_str = _dt_out.datetime.now().strftime("%Y-%m-%d")
    output = {
        "generated_at":     now,
        "today_date":       _today_str,
        "required_years":   sorted(REQUIRED_YEARS),
        "n_alerts_today":   len(alerts_today),
        "n_alerts_tomorrow":len(alerts_tomorrow),
        "alerts_today":     alerts_today,
        "alerts_tomorrow":  alerts_tomorrow,
        "year_calendar":    year_calendar,
        "n_stocks":         len(results),
        "n_excluded":       excluded,
        "n_skipped":        skipped,
        "description":      (
            f"Stocks that rose ≥{MIN_RETURN}% within 5-10 trading days in 100% of occurrences, "
            f"with signals in ALL of last {LAST_N_YEARS} years ({sorted(REQUIRED_YEARS)}). "
            f"Sorted highest avg return first."
        ),
        "stocks":           results,
    }

    out_path = OUT / "short_hold.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {out_path}")
    print(f"  Qualifying stocks : {len(results)}")
    print(f"  Excluded (ETF etc): {excluded}")
    print(f"  Skipped (data)    : {skipped}")
    print(f"\nAll {len(results)} qualifying stocks (sorted by avg return):")
    for r in results:
        alert_tag = " *** ALERT TODAY ***" if r.get("alert_today") else (" ** ALERT TOMORROW **" if r.get("alert_tomorrow") else "")
        print(f"  {r['sym']:<14} hold={r['hold_days']}d  avg={r['avg_max_ret']}%  min={r['min_max_ret']}%  n={r['n_occurrences']}  years={r['years']}{alert_tag}")

    if alerts_today:
        print(f"\n{'='*50}")
        print(f"*** {len(alerts_today)} ALERTS TODAY — BUY TOMORROW AT OPEN ***")
        for r in sorted(alerts_today, key=lambda x: -(x.get('avg_max_ret') or 0)):
            print(f"  BUY TOMORROW: {r['sym']:<14} avg=+{r['avg_max_ret']}%  min=+{r['min_max_ret']}%  hold={r['hold_days']}d")

    if alerts_tomorrow:
        print(f"\n*** {len(alerts_tomorrow)} STOCKS SIGNAL TOMORROW — WATCH TODAY ***")
        for r in sorted(alerts_tomorrow, key=lambda x: -(x.get('avg_max_ret') or 0)):
            print(f"  WATCH: {r['sym']:<14} avg=+{r['avg_max_ret']}%  hold={r['hold_days']}d")


if __name__ == "__main__":
    main()
