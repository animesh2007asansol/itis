#!/usr/bin/env python3
"""
breakout_analyzer.py  —  Consolidation → Breakout detector
===========================================================
Finds stocks that consolidate in a tight price range for 1-6 months,
then break out upward with a confirmed close above resistance.

KEY FIXES vs previous version
──────────────────────────────
1. find_consolidation now uses max(CLOSE) not max(HIGH) for the ceiling.
   HIGH prices include intraday wicks/spikes that don't represent real resistance.
   Using CLOSE gives the actual range where the market consistently closed.

2. ROOT CAUSE OF EMPTY ALERTS — while loop boundary fixed.
   Old boundary: `while start < n - cw - POST_DAYS - 1`
   This stopped the loop BEFORE reaching any breakout in the last POST_DAYS+1
   trading days. With ALERT_WINDOW=5, the code could NEVER detect any recent
   breakout — the scan terminated 5 days too early. Zero alerts every single run.
   Fix: `while start < n - cw` — the breakout day can now reach today's data.
   The avail_days check inside already handles 0-3 incomplete post-data days.

3. Guaranteed peak filter is 75% of HISTORICAL breakouts must peak >= 5%
   (not 100%). One bad breakout during a market crash should not permanently
   exclude a high-quality stock.

4. Peak filter skips recent breakouts with incomplete post-data (< POST_DAYS).
   A 2-day-old breakout only has T+1, T+2. Judging it against the 5% peak
   threshold was suppressing exactly the alerts you want to see.

5. ALERT_WINDOW increased from 3 → 5 trading days.

6. MIN_YEARS relaxed from 3 → 2 (captures stocks listed from 2022+).

7. load_all() scans data directory directly — no manifest dependency.

Output: stock_analysis/breakout_signals.json
"""

import json, sys, warnings, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "equity"
OUT  = ROOT / "stock_analysis"
OUT.mkdir(exist_ok=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MIN_PRICE           = 15.0
MIN_TURNOVER        = 5_000_000       # Rs 5 Cr daily
MIN_YEARS           = 2               # relaxed from 3 to catch stocks listed 2022+
RECENT_DAYS         = 7               # must have traded in last 7 dates in dataset
CONSOL_WINDOWS      = [15, 20, 30, 40, 60, 80, 120]  # added 15d for shorter patterns
CONSOL_BAND_MAX     = 12.0            # max close-to-close range % in consolidation
BREAKOUT_MIN        = 1.5             # close must exceed consol ceiling by this %
MIN_POST_POSITIVE   = 3               # need >= 3 of 4 post-breakout days positive
POST_DAYS           = 4               # days after breakout to track
MIN_OCCURRENCES     = 2               # need >= 2 historical confirmed breakouts
ALERT_WINDOW        = 5              # show last 5 trading days of alerts (was 3)
MIN_GUARANTEED_PEAK = 5.0             # peak threshold % for quality filter
PEAK_QUALIFY_RATE   = 0.75            # 75% of historical breakouts must peak >= threshold

# Substrings that identify ETF/index/commodity symbols to exclude
EXCL_CONTAINS = (
    "NIFTY","SENSEX","LIQUID","GILT","SETF","BEES","CPSE",
    "MAFANG","DJML","MONQ","MOSMALL","MASPTOP","HDFCMID",
    "HDFCSML","SMALLCAP","MIDSMALL","BANKPSU","PSUBANK",
    "MOM100","MON100","HDFCNIFBAN","BANKNIFTY",
)

IST     = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
now_str = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today   = now_ist.strftime("%Y-%m-%d")


def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except:
        return None


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_all():
    """Scan DATA directory directly — never reads manifest.json."""
    if not DATA.exists():
        sys.exit(f"ERROR: data directory not found: {DATA}")

    csv_files = sorted(DATA.glob("*/*/*.csv"))
    if not csv_files:
        sys.exit("ERROR: no CSV files found under data/equity/")

    print(f"  Found {len(csv_files)} CSV files (direct directory scan)")

    frames  = []
    n_skip  = 0
    for p in csv_files:
        ds = p.stem
        if len(ds) != 10 or ds[4] != '-' or ds[7] != '-':
            n_skip += 1
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = df.columns.str.strip()
            cm = {}
            for col in df.columns:
                u = col.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):                cm[col]="sym"
                elif u in ("SERIES","SCTYSRS"):               cm[col]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):    cm[col]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):    cm[col]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):       cm[col]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"):  cm[col]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):        cm[col]="v"
            df = df.rename(columns=cm)
            if not {"sym","series","o","h","l","c","v"}.issubset(df.columns):
                n_skip += 1
                continue
            df = df[df["series"].str.strip().isin(["EQ","BE"])].copy()
            df["date"] = pd.to_datetime(ds)
            frames.append(df[["sym","o","h","l","c","v","date"]])
        except:
            n_skip += 1

    if not frames:
        sys.exit("ERROR: no usable data found")

    if n_skip:
        print(f"  Skipped {n_skip} unreadable files")

    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    result = (all_data
              .dropna(subset=["o","h","l","c","v"])
              .sort_values(["sym","date"])
              .reset_index(drop=True))

    dates_found = sorted(result["date"].dt.strftime("%Y-%m-%d").unique())
    print(f"  Date range: {dates_found[0]}  to  {dates_found[-1]}  ({len(dates_found)} trading days)")
    print(f"  Latest 5 dates on disk: {dates_found[-5:]}")
    return result


# ── CONSOLIDATION DETECTION ────────────────────────────────────────────────────

def find_consolidation(c_arr, start, length):
    """
    BUG FIX: Use CLOSE for both hi and lo.

    Old code: lo=min(close), hi=max(HIGH)  — mixed, always too wide
    New code: lo=min(close), hi=max(close) — consistent, real close range

    Returns (lo, hi, range_pct) or None.
    hi becomes the resistance level: stock must CLOSE above this to break out.
    """
    window = c_arr[start: start + length]
    if len(window) < length:
        return None
    lo = float(np.min(window))
    hi = float(np.max(window))
    if lo <= 0 or hi <= 0:
        return None
    rng = (hi - lo) / lo * 100
    return r2(lo), r2(hi), r2(rng)


# ── STOCK ANALYSIS ─────────────────────────────────────────────────────────────

def analyze_stock(sym, df, latest_set, all_dates):
    n     = len(df)
    o_arr = df["o"].values
    h_arr = df["h"].values
    l_arr = df["l"].values
    c_arr = df["c"].values
    v_arr = df["v"].values
    dates = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    if cur_price < MIN_PRICE:                          return None
    if last_date not in latest_set:                    return None
    if len(set(int(d.year) for d in dates)) < MIN_YEARS: return None

    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER:   return None
    turnover_cr = r2(sum(tv5)/len(tv5)/1e7)

    # ── Breakout scan ──────────────────────────────────────────────────────────
    breakouts = []

    for cw in CONSOL_WINDOWS:
        # CRITICAL BUG FIX: old boundary was `n - cw - POST_DAYS - 1`
        # which stopped the loop BEFORE reaching any breakout in the last
        # POST_DAYS+1 = 5 trading days.  With ALERT_WINDOW = 5, this gave
        # zero alerts — the loop literally never scanned recent data.
        #
        # Fix: use `n - cw` so brk = start + cw can reach the last row (today).
        # avail_days inside the loop already handles 0..3 incomplete post-days.
        start = 0
        while start < n - cw:
            brk = start + cw     # day immediately after consolidation window

            res = find_consolidation(c_arr, start, cw)
            if res is None:
                start += 1
                continue

            consol_lo, consol_hi, consol_rng = res

            if consol_rng > CONSOL_BAND_MAX or consol_rng <= 0:
                start += 1
                continue

            if brk >= n:
                start += 1
                continue

            brk_close = float(c_arr[brk])
            brk_pct   = (brk_close - consol_hi) / consol_hi * 100

            # Must close clearly above the resistance ceiling
            if brk_pct < BREAKOUT_MIN:
                start += 1
                continue

            # Volume spike check (informational, not a filter)
            vol_window = v_arr[max(0, brk-20): brk]
            vol_ma     = float(np.mean(vol_window)) if len(vol_window) > 0 else float(v_arr[brk])
            vol_ratio  = float(v_arr[brk]) / vol_ma if vol_ma > 0 else 1.0

            # Post-breakout returns T+1..T+4 (all cumulative from brk_close)
            post_rets  = {}
            for d in range(1, POST_DAYS + 1):
                xi = brk + d
                if xi >= n:
                    break
                post_rets[d] = r2((float(c_arr[xi]) - brk_close) / brk_close * 100)

            # T+1 entry analysis
            t1 = brk + 1
            if t1 < n:
                t1_open_gap = r2((float(o_arr[t1]) - brk_close) / brk_close * 100)
                t1_dip_pct  = r2((float(l_arr[t1]) - brk_close) / brk_close * 100)
            else:
                t1_open_gap = None
                t1_dip_pct  = None

            avail_days = min(POST_DAYS, n - brk - 1)
            pos_days   = sum(1 for d in range(1, avail_days+1)
                            if (post_rets.get(d) or -999) > 0)

            # Only filter on positive-day count when enough data is available
            if avail_days >= MIN_POST_POSITIVE and pos_days < MIN_POST_POSITIVE:
                start += 1
                continue

            breakouts.append({
                "date":        str(dates[brk].date()),
                "year":        int(dates[brk].year),
                "consol_days": cw,
                "consol_lo":   consol_lo,
                "consol_hi":   consol_hi,
                "consol_rng":  consol_rng,
                "brk_close":   r2(brk_close),
                "brk_pct":     r2(brk_pct),
                "vol_ratio":   r2(vol_ratio),
                "t1_open_gap": t1_open_gap,
                "t1_dip_pct":  t1_dip_pct,
                "post_rets":   post_rets,
                "pos_days":    pos_days,
                "avail_days":  avail_days,
            })

            # Skip forward after a confirmed breakout to avoid re-scanning the same region
            start += max(cw // 2, 5)

    if len(breakouts) < MIN_OCCURRENCES:
        return None

    # ── Deduplication (same breakout found by multiple window sizes) ───────────
    breakouts.sort(key=lambda x: x["date"])
    deduped = [breakouts[0]]
    for b in breakouts[1:]:
        try:
            prev_d = datetime.strptime(deduped[-1]["date"], "%Y-%m-%d")
            this_d = datetime.strptime(b["date"], "%Y-%m-%d")
            if (this_d - prev_d).days < 5:
                if b["brk_pct"] > deduped[-1]["brk_pct"]:
                    deduped[-1] = b
            else:
                deduped.append(b)
        except:
            deduped.append(b)

    if len(deduped) < MIN_OCCURRENCES:
        return None

    # ── Guaranteed peak quality filter ────────────────────────────────────────
    #
    # BUG FIX 1: Skip breakouts with incomplete post-data (recent signals).
    #   A breakout from yesterday only has T+1 data. Its peak of +2% so far
    #   is not the final answer — it might hit +8% by T+4. Applying the filter
    #   to incomplete data was suppressing exactly the alerts you want to see.
    #
    # BUG FIX 2: 75% threshold instead of 100%.
    #   One bad breakout during a market crash should not permanently exclude
    #   an otherwise reliable stock pattern.

    def peak_of(b):
        pr   = b.get("post_rets", {})
        vals = [pr[d] for d in range(1, POST_DAYS+1) if pr.get(d) is not None]
        return max(vals) if vals else None

    historical = [b for b in deduped if b.get("avail_days", 0) >= POST_DAYS]

    if historical:
        peaks      = [peak_of(b) for b in historical]
        good_peaks = [p for p in peaks if p is not None]
        if good_peaks:
            qualifying   = sum(1 for p in good_peaks if p >= MIN_GUARANTEED_PEAK)
            qualify_rate = qualifying / len(good_peaks)
            if qualify_rate < PEAK_QUALIFY_RATE:
                return None
            min_peak_ret = r2(min(good_peaks))
            avg_peak_ret = r2(sum(good_peaks)/len(good_peaks))
        else:
            min_peak_ret = avg_peak_ret = None
    else:
        # All breakouts are very recent — no historical filter applicable
        min_peak_ret = avg_peak_ret = None

    for b in deduped:
        b["peak_ret"] = r2(peak_of(b))

    # ── Aggregate stats ────────────────────────────────────────────────────────
    def safe_avg(vals):
        v = [x for x in vals if x is not None]
        return r2(sum(v)/len(v)) if v else None

    avg_brk_pct    = safe_avg([b["brk_pct"]        for b in deduped])
    avg_t1_dip     = safe_avg([b["t1_dip_pct"]     for b in deduped])
    avg_t1_gap     = safe_avg([b["t1_open_gap"]    for b in deduped])
    avg_consol_rng = safe_avg([b["consol_rng"]     for b in deduped])
    avg_consol_days= safe_avg([b["consol_days"]    for b in deduped])
    avg_t1_ret     = safe_avg([b["post_rets"].get(1) for b in deduped])
    avg_t4_ret     = safe_avg([b["post_rets"].get(4) for b in deduped])

    # ── Entry recommendation ───────────────────────────────────────────────────
    if avg_t1_dip is not None and avg_t1_dip < -1.0:
        entry_rec  = (f"Wait for T+1 dip to ~{avg_t1_dip:.1f}% below signal close"
                      + (f" (avg open: {avg_t1_gap:+.1f}%)" if avg_t1_gap else ""))
        entry_type = "wait_for_dip"
    elif avg_t1_gap is not None and avg_t1_gap < 0:
        entry_rec  = f"Enter at T+1 open (typically opens {avg_t1_gap:+.1f}% below signal close)"
        entry_type = "t1_open"
    else:
        entry_rec  = "Enter at T+1 open (opens near or above signal close)"
        entry_type = "t1_open"

    # ── Alert detection ────────────────────────────────────────────────────────
    recent_set = set(all_dates[-ALERT_WINDOW:]) if len(all_dates) >= ALERT_WINDOW else set(all_dates)
    alerts = []

    for b in reversed(deduped):
        if b["date"] not in recent_set:
            continue
        try:
            days_since = (now_ist.replace(tzinfo=None) -
                          datetime.strptime(b["date"], "%Y-%m-%d")).days
        except:
            days_since = 0

        sig_idx = next((j for j, d in enumerate(dates)
                        if str(d.date()) == b["date"]), None)
        t1_date = str(dates[sig_idx+1].date()) if sig_idx is not None and sig_idx+1 < n else "tomorrow"
        t2_date = str(dates[sig_idx+2].date()) if sig_idx is not None and sig_idx+2 < n else "day after"

        if days_since == 0:
            action = "TODAY — enter tomorrow (T+1) at open or wait for dip"
            buy_on = "T+1"
        elif days_since <= 3:
            action = f"Signal {days_since}d ago — enter TODAY at open or wait for dip"
            buy_on = "TODAY"
        else:
            action = f"Signal {days_since}d ago — still valid, entering late"
            buy_on = "LATE"

        alerts.append({
            "sig_date":    b["date"],
            "days_since":  days_since,
            "brk_close":   b["brk_close"],
            "brk_pct":     b["brk_pct"],
            "consol_days": b["consol_days"],
            "consol_rng":  b["consol_rng"],
            "vol_ratio":   b["vol_ratio"],
            "t1_date":     t1_date,
            "t2_date":     t2_date,
            "t1_open_gap": b["t1_open_gap"],
            "t1_dip_pct":  b["t1_dip_pct"],
            "post_rets":   b["post_rets"],
            "peak_ret":    b.get("peak_ret"),
            "action":      action,
            "buy_on":      buy_on,
            "entry_type":  entry_type,
            "entry_rec":   entry_rec,
            "avg_t4_ret":  avg_t4_ret,
            "min_peak_ret":min_peak_ret,
        })

    return {
        "sym":             sym,
        "price":           r2(cur_price),
        "last_date":       last_date,
        "turnover_cr":     turnover_cr,
        "n_breakouts":     len(deduped),
        "years":           sorted(set(b["year"] for b in deduped)),
        "avg_brk_pct":     avg_brk_pct,
        "avg_consol_rng":  avg_consol_rng,
        "avg_consol_days": avg_consol_days,
        "avg_t1_dip":      avg_t1_dip,
        "avg_t1_gap":      avg_t1_gap,
        "avg_t1_ret":      avg_t1_ret,
        "avg_t4_ret":      avg_t4_ret,
        "min_peak_ret":    min_peak_ret,
        "avg_peak_ret":    avg_peak_ret,
        "entry_type":      entry_type,
        "entry_rec":       entry_rec,
        "has_alert":       len(alerts) > 0,
        "alerts":          alerts,
        "breakouts":       sorted(deduped, key=lambda x: x["date"], reverse=True),
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"NSE Breakout Analyzer  |  {now_str} IST")
    print(f"Quality filter : {PEAK_QUALIFY_RATE*100:.0f}% of historical breakouts must peak >= {MIN_GUARANTEED_PEAK}%")
    print(f"Alert window   : last {ALERT_WINDOW} trading days")
    print(f"{'='*60}")

    all_data   = load_all()
    grouped    = all_data.groupby("sym")
    syms       = sorted(grouped.groups.keys())
    all_dates  = sorted(all_data["date"].dt.strftime("%Y-%m-%d").unique())
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]

    print(f"Symbols: {len(syms):,}  |  Latest date: {last_fetch}")
    print(f"Alert window: {all_dates[-ALERT_WINDOW]}  →  {last_fetch}\n")

    results = []
    skipped = excluded = 0

    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)}  —  {len(results)} qualified")

        sym_up = sym.upper()
        if any(ex in sym_up for ex in EXCL_CONTAINS):
            excluded += 1
            continue

        try:
            grp = (grouped.get_group(sym)
                   .sort_values("date")
                   .reset_index(drop=True))
            if len(grp) < 150:
                skipped += 1
                continue
            res = analyze_stock(sym, grp, latest_set, all_dates)
            if res:
                results.append(res)
            else:
                skipped += 1
        except:
            skipped += 1

    alert_stocks = sorted(
        [r for r in results if r["has_alert"]],
        key=lambda x: -(x.get("turnover_cr") or 0)
    )
    all_stocks = sorted(
        results,
        key=lambda x: -(x.get("min_peak_ret") or x.get("avg_t4_ret") or 0)
    )

    all_alerts = []
    for r in results:
        for a in (r.get("alerts") or []):
            all_alerts.append({
                **a,
                "sym":          r["sym"],
                "price":        r["price"],
                "turnover_cr":  r["turnover_cr"],
                "avg_t1_dip":   r["avg_t1_dip"],
                "avg_peak_ret": r["avg_peak_ret"],
                "entry_rec":    r["entry_rec"],
                "entry_type":   r["entry_type"],
            })
    all_alerts.sort(key=lambda x: (x.get("days_since") or 99,
                                    -(x.get("turnover_cr") or 0)))

    output = {
        "generated_at": now_str,
        "today_ist":    today,
        "last_fetch":   last_fetch,
        "alert_window": f"{all_dates[-ALERT_WINDOW]} to {last_fetch}",
        "n_stocks":     len(results),
        "n_alerts":     len(alert_stocks),
        "n_all_alerts": len(all_alerts),
        "alert_stocks": alert_stocks,
        "all_alerts":   all_alerts,
        "stocks":       all_stocks,
        "description": (
            f"NSE consolidation breakout signals. "
            f"Quality filter: {PEAK_QUALIFY_RATE*100:.0f}% of historical breakouts "
            f"must peak >= {MIN_GUARANTEED_PEAK}% within T+1..T+4. "
            f"Alert window: last {ALERT_WINDOW} trading days."
        ),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "breakout_signals.json").write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print(f"  Qualified stocks     : {len(results)}")
    print(f"  With active alerts   : {len(alert_stocks)}")
    print(f"  Total alert signals  : {len(all_alerts)}")
    print(f"  Excluded (ETF/index) : {excluded}")
    print(f"  Skipped (filters)    : {skipped}")

    if alert_stocks:
        print(f"\n{'='*60}")
        print(f"TOP ALERTS — last {ALERT_WINDOW} trading days")
        print(f"{'='*60}")
        for r in alert_stocks[:20]:
            a  = r["alerts"][0]
            pk = f"+{r['min_peak_ret']}%" if r.get("min_peak_ret") else "?"
            print(f"  {r['sym']:<14}  brk=+{a['brk_pct']}%  "
                  f"consol={a['consol_days']}d  minPeak={pk}  "
                  f"{a['buy_on']}  {a['sig_date']}")


if __name__ == "__main__":
    main()
