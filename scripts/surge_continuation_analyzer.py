#!/usr/bin/env python3
"""
surge_continuation_analyzer.py
================================
For qualifying stocks, finds: when the stock rises X% in a single day above
the previous close, does it then gain another 15%+ from the next day's close
within 20 trading days — EVERY SINGLE TIME without a miss?

Rules:
  - Min Rs 20 Cr daily traded value (price × volume)
  - Trading since at least 5 years AND active today
  - Consecutive trigger dedup: if Day N-1 also triggered, Day N is ignored
  - 100% win rate: every trigger must lead to 15%+ within 20 trading days
  - Shows 5d/10d/15d context returns
  - Live alerts: triggered on last data fetch day

Output: stock_analysis/surge_signals.json
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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MIN_PRICE        = 10.0
MIN_TURNOVER     = 20_000_000       # Rs 20 Cr daily value
MIN_YEARS        = 5                # at least 5 years of history
RECENT_DAYS      = 5                # must have traded in last 5 dates
TARGET_PCT       = 15.0             # must gain 15%+ after trigger
MAX_HOLD_DAYS    = 20               # within 20 trading days
SURGE_THRESHOLDS = [2, 3, 4, 5, 6, 7, 8, 10]  # test these trigger %
MIN_OCCURRENCES  = 3                # at least 3 historical triggers

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID")

IST        = timezone(timedelta(hours=5, minutes=30))
now_ist    = datetime.now(IST)
today_str  = now_ist.strftime("%Y-%m-%d")
now_str    = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
cur_yr     = now_ist.year
min_yr     = cur_yr - MIN_YEARS

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


# ── DATA LOAD ──────────────────────────────────────────────────────────────────
def load_all():
    if not MANIFEST.exists():
        print("ERROR: manifest.json missing"); sys.exit(1)
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
    if not frames: print("ERROR: No data."); sys.exit(1)
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    all_data = all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


def is_excluded(sym):
    return any(sym.upper().endswith(s) for s in EXCL_SFX)


# ── CORE ANALYSIS ──────────────────────────────────────────────────────────────
def find_triggers(c_arr, v_arr, dates, surge_pct):
    """
    Find all days where close rose >= surge_pct% vs previous close.
    Apply consecutive dedup: if day i-1 was also a trigger, skip day i.
    Returns list of trigger indices.
    """
    n = len(c_arr)
    raw_triggers = []
    for i in range(1, n):
        prev = float(c_arr[i-1])
        curr = float(c_arr[i])
        if prev <= 0: continue
        pct = (curr - prev) / prev * 100
        if pct >= surge_pct:
            raw_triggers.append(i)

    # Consecutive dedup: keep only first day of consecutive triggers
    deduped = []
    for idx in raw_triggers:
        if deduped and idx == deduped[-1] + 1:
            continue   # consecutive — skip, only 1st day counts
        deduped.append(idx)
    return deduped


def analyze_surge(sym, df, last_date, surge_pct):
    """
    For a given surge threshold, find all triggers and check if 100%
    of them led to >= TARGET_PCT gain from next-day close within MAX_HOLD_DAYS.
    Returns qualifying result or None.
    """
    n       = len(df)
    c_arr   = df["c"].values
    v_arr   = df["v"].values
    h_arr   = df["h"].values
    dates   = pd.to_datetime(df["date"].values)

    triggers = find_triggers(c_arr, v_arr, dates, surge_pct)
    # Only count triggers where we have enough forward data to evaluate
    eval_triggers = [t for t in triggers if t + MAX_HOLD_DAYS < n]

    if len(eval_triggers) < MIN_OCCURRENCES: return None

    occurrences = []
    for trig_idx in eval_triggers:
        trig_date  = str(dates[trig_idx].date())
        trig_close = float(c_arr[trig_idx])
        # Surge % on trigger day
        prev_close = float(c_arr[trig_idx - 1])
        surge_day_pct = r2((trig_close - prev_close) / prev_close * 100)

        # Entry: NEXT DAY's close (day after trigger)
        entry_idx   = trig_idx + 1
        entry_px    = float(c_arr[entry_idx])
        if entry_px <= 0: continue

        # Track: day-by-day from entry until TARGET_PCT hit or MAX_HOLD_DAYS
        hit_day      = None
        hit_px       = None
        rets_by_day  = {}
        for d in range(1, MAX_HOLD_DAYS + 1):
            xi = entry_idx + d
            if xi >= n: break
            ret = (float(c_arr[xi]) - entry_px) / entry_px * 100
            rets_by_day[d] = r2(ret)
            if ret >= TARGET_PCT and hit_day is None:
                hit_day = d
                hit_px  = r2(float(c_arr[xi]))

        max_ret_20d = r2(max(rets_by_day.values())) if rets_by_day else None

        # Context: returns in 5d, 10d, 15d BEFORE trigger
        ret_5d_before  = r2((trig_close - float(c_arr[max(0, trig_idx-5)])) /
                            float(c_arr[max(0, trig_idx-5)]) * 100) if trig_idx >= 5 else None
        ret_10d_before = r2((trig_close - float(c_arr[max(0, trig_idx-10)])) /
                            float(c_arr[max(0, trig_idx-10)]) * 100) if trig_idx >= 10 else None
        ret_15d_before = r2((trig_close - float(c_arr[max(0, trig_idx-15)])) /
                            float(c_arr[max(0, trig_idx-15)]) * 100) if trig_idx >= 15 else None

        # Turnover on trigger day
        turnover = r2(float(c_arr[trig_idx]) * float(v_arr[trig_idx]))

        occurrences.append({
            "trig_date":      trig_date,
            "trig_year":      int(dates[trig_idx].year),
            "trig_close":     r2(trig_close),
            "surge_pct":      surge_day_pct,
            "entry_date":     str(dates[entry_idx].date()),
            "entry_px":       r2(entry_px),
            "hit_day":        hit_day,
            "hit_px":         hit_px,
            "max_ret_20d":    max_ret_20d,
            "rets_by_day":    rets_by_day,
            "ret_5d_before":  ret_5d_before,
            "ret_10d_before": ret_10d_before,
            "ret_15d_before": ret_15d_before,
            "turnover_cr":    r2(float(c_arr[trig_idx]) * float(v_arr[trig_idx]) / 1e7),
        })

    if not occurrences: return None

    # 100% win rate: every occurrence must hit TARGET_PCT within MAX_HOLD_DAYS
    if not all(occ["hit_day"] is not None for occ in occurrences): return None

    hit_days   = [occ["hit_day"] for occ in occurrences]
    max_rets   = [occ["max_ret_20d"] for occ in occurrences if occ["max_ret_20d"] is not None]
    entry_rets = [occ["rets_by_day"].get(MAX_HOLD_DAYS) for occ in occurrences
                  if occ["rets_by_day"].get(MAX_HOLD_DAYS) is not None]

    return {
        "surge_pct":        surge_pct,
        "n_occurrences":    len(occurrences),
        "avg_days_to_15":   r2(sum(hit_days) / len(hit_days)),
        "min_days_to_15":   min(hit_days),
        "max_days_to_15":   max(hit_days),
        "avg_max_ret":      r2(sum(max_rets) / len(max_rets)) if max_rets else None,
        "min_max_ret":      r2(min(max_rets)) if max_rets else None,
        "avg_exit_ret":     r2(sum(entry_rets) / len(entry_rets)) if entry_rets else None,
        "years":            sorted(set(occ["trig_year"] for occ in occurrences)),
        "occurrences":      sorted(occurrences, key=lambda x: x["trig_date"], reverse=True),
    }


def analyze_stock(sym, df, latest_set):
    n        = len(df)
    c_arr    = df["c"].values
    v_arr    = df["v"].values
    dates    = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    if cur_price < MIN_PRICE: return None
    if last_date not in latest_set: return None

    # 5-year data check
    stock_yrs = set(int(d.year) for d in dates)
    if max(stock_yrs) - min(stock_yrs) < MIN_YEARS - 1: return None
    if min(stock_yrs) > min_yr: return None

    # Turnover: avg last 5 days must be >= 20 Cr
    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5) / len(tv5) < MIN_TURNOVER: return None

    # Test each surge threshold
    best_result = None
    for surge_pct in SURGE_THRESHOLDS:
        result = analyze_surge(sym, df, last_date, surge_pct)
        if result is None: continue
        # Prefer the threshold with most occurrences; tie-break: lowest avg days to target
        if best_result is None:
            best_result = result
        elif result["n_occurrences"] > best_result["n_occurrences"]:
            best_result = result
        elif (result["n_occurrences"] == best_result["n_occurrences"] and
              (result["avg_days_to_15"] or 99) < (best_result["avg_days_to_15"] or 99)):
            best_result = result

    if best_result is None: return None

    # ── Live alert: did today trigger? ────────────────────────────────────────
    live_alert = None
    if n >= 2:
        prev_c = float(c_arr[-2])
        curr_c = float(c_arr[-1])
        if prev_c > 0:
            today_surge = (curr_c - prev_c) / prev_c * 100
            if today_surge >= best_result["surge_pct"]:
                # Check: was yesterday also a trigger? (consecutive dedup)
                yesterday_also = False
                if n >= 3:
                    prev2_c = float(c_arr[-3])
                    if prev2_c > 0 and (float(c_arr[-2]) - prev2_c) / prev2_c * 100 >= best_result["surge_pct"]:
                        yesterday_also = True
                if not yesterday_also:
                    target_px = r2(curr_c * (1 + TARGET_PCT / 100))
                    avg_tgt   = r2(curr_c * (1 + (best_result["avg_max_ret"] or TARGET_PCT) / 100))
                    live_alert = {
                        "trig_date":    last_date,
                        "trig_close":   r2(curr_c),
                        "surge_pct":    r2(today_surge),
                        "entry_date":   "tomorrow",
                        "entry_px":     None,    # next day's open (unknown yet)
                        "target_px":    target_px,
                        "avg_tgt":      avg_tgt,
                        "avg_days":     best_result["avg_days_to_15"],
                        "is_new":       True,
                    }

    # ── Active alerts: triggered in last MAX_HOLD_DAYS, not yet at 15% ───────
    active_alerts = []
    surge_pct = best_result["surge_pct"]
    all_triggers = find_triggers(c_arr, v_arr, dates, surge_pct)
    for trig_idx in reversed(all_triggers):
        trig_date = str(dates[trig_idx].date())
        if trig_date > last_date: continue
        entry_idx = trig_idx + 1
        if entry_idx >= n: continue
        entry_px  = float(c_arr[entry_idx])
        if entry_px <= 0: continue

        # Days since entry
        days_since = n - 1 - entry_idx
        if days_since < 0 or days_since > MAX_HOLD_DAYS: continue
        # Still within hold window
        cur_ret  = (cur_price - entry_px) / entry_px * 100
        if cur_ret >= TARGET_PCT: continue  # already hit target — not "active"

        days_left = MAX_HOLD_DAYS - days_since
        target_px = r2(entry_px * (1 + TARGET_PCT / 100))

        # Context returns
        ret_5d  = r2((float(c_arr[trig_idx]) - float(c_arr[max(0, trig_idx-5)])) /
                     float(c_arr[max(0, trig_idx-5)]) * 100) if trig_idx >= 5 else None
        ret_10d = r2((float(c_arr[trig_idx]) - float(c_arr[max(0, trig_idx-10)])) /
                     float(c_arr[max(0, trig_idx-10)]) * 100) if trig_idx >= 10 else None
        ret_15d = r2((float(c_arr[trig_idx]) - float(c_arr[max(0, trig_idx-15)])) /
                     float(c_arr[max(0, trig_idx-15)]) * 100) if trig_idx >= 15 else None

        active_alerts.append({
            "trig_date":   trig_date,
            "trig_close":  r2(float(c_arr[trig_idx])),
            "entry_date":  str(dates[entry_idx].date()),
            "entry_px":    r2(entry_px),
            "cur_price":   r2(cur_price),
            "cur_ret":     r2(cur_ret),
            "target_px":   target_px,
            "days_elapsed":days_since,
            "days_left":   days_left,
            "remaining_pct": r2(TARGET_PCT - cur_ret),
            "avg_days":    best_result["avg_days_to_15"],
            "ret_5d":      ret_5d,
            "ret_10d":     ret_10d,
            "ret_15d":     ret_15d,
        })
        break  # only most recent active alert per stock

    # Recent context for the stock
    ret_5d_cur  = r2((cur_price - float(c_arr[max(0, n-6)]))  / float(c_arr[max(0, n-6)])  * 100) if n > 5  else None
    ret_10d_cur = r2((cur_price - float(c_arr[max(0, n-11)])) / float(c_arr[max(0, n-11)]) * 100) if n > 10 else None
    ret_15d_cur = r2((cur_price - float(c_arr[max(0, n-16)])) / float(c_arr[max(0, n-16)]) * 100) if n > 15 else None

    turnover_avg = r2(sum(tv5) / len(tv5) / 1e7)  # in Crores

    return {
        "sym":          sym,
        "price":        r2(cur_price),
        "last_date":    last_date,
        "turnover_cr":  turnover_avg,
        "ret_5d":       ret_5d_cur,
        "ret_10d":      ret_10d_cur,
        "ret_15d":      ret_15d_cur,
        "live_alert":   live_alert,
        "active_alerts":active_alerts,
        "has_live":     live_alert is not None,
        "has_active":   len(active_alerts) > 0,
        **best_result,
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"Surge Continuation Analyzer")
    print(f"IST: {now_str}")
    print(f"Target: {TARGET_PCT}%+ after trigger within {MAX_HOLD_DAYS} trading days")
    print(f"{'='*60}")

    all_data = load_all()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())

    all_dates  = sorted(set(str(dt.date())
                            for dt in pd.to_datetime(all_data["date"].unique())))
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]

    print(f"\nLast data fetch: {last_fetch}")
    print(f"Analyzing {len(syms):,} symbols...")

    results  = []
    skipped  = 0
    excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)} qualifying stocks")
        if is_excluded(sym): excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 300: skipped += 1; continue
            res = analyze_stock(sym, grp, latest_set)
            if res: results.append(res)
            else:   skipped += 1
        except: skipped += 1

    # Sort by avg_max_ret descending
    results.sort(key=lambda x: -(x.get("avg_max_ret") or 0))

    # Re-rank: give a score = (avg_max_ret) / avg_days_to_15
    for r in results:
        avg_d = r.get("avg_days_to_15") or MAX_HOLD_DAYS
        r["score"] = r2((r.get("avg_max_ret") or 0) / avg_d)
    results.sort(key=lambda x: -(x.get("score") or 0))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    live_alerts   = [r for r in results if r.get("has_live")]
    active_alerts = [r for r in results if r.get("has_active")]
    all_open      = [r for r in results if r.get("has_live") or r.get("has_active")]

    output = {
        "generated_at":   now_str,
        "today_ist":      today_str,
        "last_fetch_date":last_fetch,
        "n_stocks":       len(results),
        "n_live_today":   len(live_alerts),
        "n_active":       len(active_alerts),
        "target_pct":     TARGET_PCT,
        "max_hold_days":  MAX_HOLD_DAYS,
        "live_today":     live_alerts,
        "active_alerts":  active_alerts,
        "all_open":       all_open,
        "description":    (
            f"Stocks where a surge of X% triggers a {TARGET_PCT}%+ gain "
            f"within {MAX_HOLD_DAYS} trading days, 100% win rate. "
            f"Min Rs 20 Cr daily turnover, 5+ years history."
        ),
        "stocks": results,
    }

    path = OUT / "surge_signals.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written: {path}")
    print(f"  Qualifying stocks : {len(results)}")
    print(f"  Live today        : {len(live_alerts)}")
    print(f"  Active (in hold)  : {len(active_alerts)}")
    print(f"  Excluded          : {excluded}")
    print(f"  Skipped           : {skipped}")

    if live_alerts:
        print(f"\n*** TRIGGERED TODAY — ENTER TOMORROW ***")
        for r in live_alerts[:10]:
            la = r["live_alert"]
            print(f"  {r['sym']:<14} surge=+{la['surge_pct']:.1f}%"
                  f"  tgt=Rs{la['target_px']}  avg={r['avg_max_ret']}%"
                  f"  in~{r['avg_days_to_15']:.0f}d")


if __name__ == "__main__":
    main()
