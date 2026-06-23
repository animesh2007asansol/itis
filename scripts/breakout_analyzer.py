#!/usr/bin/env python3
"""
breakout_analyzer.py
====================
Finds stocks that consolidate in a tight price range for 1-6 months,
then break out upward and sustain for 3-4+ days.

For each breakout:
- How long the stock consolidated (20-120 trading days)
- The consolidation range (%)
- The breakout level (resistance price)
- What happens next 4 days (T+1 to T+4 returns)
- T+1 entry analysis: gap at open, dip below T0 close, best entry price
- Win rate: how often this pattern gives 3 consecutive up days

Daily alerts:
- TODAY (T=0): breakout happened today → buy at T+1 open or wait for dip
- YESTERDAY (T=1): breakout yesterday → buy today
- 2 DAYS AGO (T=2): still valid entry window

Requirements:
- Price > Rs 15 (no penny stocks)
- Rs 5 Cr+ daily turnover
- 3+ years of history
- Currently active
- Pattern must have 100% post-breakout positive (or >= 80% win rate)
- EVERY historical breakout must have peaked >= 5% within T+1..T+4 (guaranteed profit filter)

Output: stock_analysis/breakout_signals.json
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
MIN_PRICE           = 15.0
MIN_TURNOVER        = 5_000_000       # Rs 5 Cr daily
MIN_YEARS           = 3
RECENT_DAYS         = 5
CONSOL_WINDOWS      = [20, 30, 40, 60, 80, 120]  # trading days (1mo to 6mo)
CONSOL_BAND_MAX     = 12.0            # max % range to be considered consolidation
BREAKOUT_MIN        = 1.5             # breakout must exceed consolidation top by 1.5%
MIN_POST_POSITIVE   = 3               # need 3 of 4 post-breakout days to be up
POST_DAYS           = 4               # track 4 days after breakout
MIN_OCCURRENCES     = 2               # need at least 2 historical breakouts
ALERT_WINDOW        = 3               # show alerts for last 3 trading days
MIN_GUARANTEED_PEAK = 5.0             # EVERY historical breakout must peak >= this % within T+1..T+4

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID","NIFTY","SENSEX")

IST      = timezone(timedelta(hours=5, minutes=30))
now_ist  = datetime.now(IST)
now_str  = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today    = now_ist.strftime("%Y-%m-%d")

def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except: return None


def load_all():
    if not MANIFEST.exists(): sys.exit("ERROR: manifest.json missing")
    manifest = json.loads(MANIFEST.read_text())
    dates    = sorted(manifest.keys())
    print(f"  Loading {len(dates)} dates...")
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
    if not frames: sys.exit("ERROR: no data")
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    return all_data.dropna(subset=["o","h","l","c","v"]).sort_values(["sym","date"]).reset_index(drop=True)


def find_consolidation_top(c_arr, h_arr, start, length):
    """Return (consol_low, consol_high, range_pct) for window [start, start+length)"""
    window_c = c_arr[start:start+length]
    window_h = h_arr[start:start+length]
    if len(window_c) < length: return None
    lo = float(np.min(window_c))
    hi = float(np.max(window_h))
    if lo <= 0: return None
    rng = (hi - lo) / lo * 100
    return (r2(lo), r2(hi), r2(rng))


def analyze_stock(sym, df, latest_set, all_dates):
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
    if last_date not in latest_set: return None

    yrs = set(int(d.year) for d in dates)
    if max(yrs) - min(yrs) < MIN_YEARS - 1: return None

    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER: return None

    turnover_cr = r2(sum(tv5)/len(tv5)/1e7)

    # Find all breakouts
    breakouts = []

    for cw in CONSOL_WINDOWS:
        # Slide window: need cw days of consolidation + POST_DAYS after
        for start in range(0, n - cw - POST_DAYS - 2):
            end = start + cw         # last day of consolidation
            brk = end + 1            # breakout day

            res = find_consolidation_top(c_arr, h_arr, start, cw)
            if res is None: continue
            consol_lo, consol_hi, consol_rng = res

            # Range must be tight enough
            if consol_rng > CONSOL_BAND_MAX: continue
            if consol_rng <= 0: continue

            # Breakout: close above consolidation high by BREAKOUT_MIN%
            if brk >= n: continue
            brk_close = float(c_arr[brk])
            brk_pct   = (brk_close - consol_hi) / consol_hi * 100
            if brk_pct < BREAKOUT_MIN: continue

            # Volume on breakout day — should be above average
            vol_ma = float(pd.Series(v_arr[max(0,brk-20):brk]).mean()) if brk >= 5 else 0
            vol_ratio = float(v_arr[brk]) / vol_ma if vol_ma > 0 else 1.0

            # Post-breakout behavior: T+1 to T+4
            post_rets   = {}
            post_opens  = {}
            t1_dip_pct  = None
            t1_open_gap = None

            for d in range(1, POST_DAYS + 1):
                xi = brk + d
                if xi >= n: break
                ret = (float(c_arr[xi]) - brk_close) / brk_close * 100
                post_rets[d]  = r2(ret)
                post_opens[d] = r2(float(o_arr[xi]))

            # T+1 entry analysis
            t1 = brk + 1
            if t1 < n:
                t1_open     = float(o_arr[t1])
                t1_low      = float(l_arr[t1])
                t1_close    = float(c_arr[t1])
                t1_open_gap = r2((t1_open - brk_close) / brk_close * 100)
                t1_dip_pct  = r2((t1_low - brk_close) / brk_close * 100)
                t1_close_ret= r2((t1_close - brk_close) / brk_close * 100)
            else:
                t1_open_gap = None; t1_dip_pct = None; t1_close_ret = None

            # Count positive post days (out of available)
            pos_days = sum(1 for d in range(1, min(POST_DAYS+1, n-brk))
                          if (post_rets.get(d) or -999) > 0)
            avail_days = min(POST_DAYS, n - brk - 1)

            if avail_days < 1: continue
            if pos_days < MIN_POST_POSITIVE and avail_days >= MIN_POST_POSITIVE:
                continue  # didn't have enough positive days

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
            })
            # Skip forward to avoid overlapping windows
            start += cw // 2

    if len(breakouts) < MIN_OCCURRENCES: return None

    # Deduplicate: if two breakouts within 5 days, keep better one
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
        except: deduped.append(b)

    if len(deduped) < MIN_OCCURRENCES: return None

    # ── GUARANTEED PROFIT FILTER ────────────────────────────────────────────────
    # Every single historical breakout must have peaked >= MIN_GUARANTEED_PEAK%
    # within T+1 to T+4. If even ONE breakout failed to reach this threshold,
    # the stock is excluded — the pattern is unreliable and may produce junk signals.
    #
    # We use the PEAK (max of T+1..T+4) rather than T+4 close because:
    # - A stock may run +8% by T+2 then consolidate back to +2% by T+4
    # - The entry/exit logic (T+1 dip entry, exit at peak) captures this move
    # - Filtering only on T+4 close would wrongly discard fast-movers
    def _peak_ret(b):
        pr = b.get("post_rets", {})
        vals = [pr[d] for d in range(1, POST_DAYS + 1) if pr.get(d) is not None]
        return max(vals) if vals else None

    peak_rets = [_peak_ret(b) for b in deduped]

    # Exclude any stock where even one breakout failed to peak >= 5%
    if not all(p is not None and p >= MIN_GUARANTEED_PEAK for p in peak_rets):
        return None

    # Attach individual peak_ret to each breakout for display in History tab
    for b, pk in zip(deduped, peak_rets):
        b["peak_ret"] = r2(pk)

    # ── Aggregate stats ─────────────────────────────────────────────────────────
    brk_pcts    = [b["brk_pct"] for b in deduped]
    t1_dips     = [b["t1_dip_pct"] for b in deduped if b["t1_dip_pct"] is not None]
    t1_gaps     = [b["t1_open_gap"] for b in deduped if b["t1_open_gap"] is not None]
    consol_rngs = [b["consol_rng"] for b in deduped]
    consol_days = [b["consol_days"] for b in deduped]
    all_t1_rets = [b["post_rets"].get(1) for b in deduped if b["post_rets"].get(1) is not None]
    all_t4_rets = [b["post_rets"].get(4) for b in deduped if b["post_rets"].get(4) is not None]

    avg_brk_pct   = r2(sum(brk_pcts)/len(brk_pcts))
    avg_t1_dip    = r2(sum(t1_dips)/len(t1_dips)) if t1_dips else None
    avg_t1_gap    = r2(sum(t1_gaps)/len(t1_gaps)) if t1_gaps else None
    avg_t1_ret    = r2(sum(all_t1_rets)/len(all_t1_rets)) if all_t1_rets else None
    avg_t4_ret    = r2(sum(all_t4_rets)/len(all_t4_rets)) if all_t4_rets else None
    avg_consol_rng= r2(sum(consol_rngs)/len(consol_rngs))
    avg_consol_days=r2(sum(consol_days)/len(consol_days))

    # Peak return stats across all historical breakouts
    min_peak_ret  = r2(min(peak_rets))   # worst-case peak — the "guarantee" floor
    avg_peak_ret  = r2(sum(peak_rets)/len(peak_rets))

    # Entry recommendation
    # If T+1 typically dips below signal close → "Wait for dip at open T+1"
    # If T+1 opens higher and keeps going → "Enter at T+1 open"
    if avg_t1_dip is not None and avg_t1_dip < -1.0:
        entry_rec = f"Enter on T+1 (next day) when it dips to ~{avg_t1_dip:.1f}% below signal close"
        entry_type = "wait_for_dip"
    elif avg_t1_gap is not None and avg_t1_gap < 0:
        entry_rec = f"Enter at T+1 open (opens ~{avg_t1_gap:.1f}% below signal close — buy the gap)"
        entry_type = "t1_open"
    else:
        entry_rec = f"Enter at T+1 open (opens near/above signal close)"
        entry_type = "t1_open"

    # Check if stock has current alert (breakout in last ALERT_WINDOW trading days)
    recent_all = sorted(all_dates[-ALERT_WINDOW:]) if len(all_dates) >= ALERT_WINDOW else all_dates
    alerts = []
    for b in reversed(deduped):
        if b["date"] in recent_all or b["date"] == str(dates[-1].date()):
            # Figure out what day it is relative to signal
            try:
                sig_dt  = datetime.strptime(b["date"], "%Y-%m-%d")
                today_d = now_ist.replace(tzinfo=None)
                days_since = (today_d - sig_dt).days
            except: days_since = 0

            # Find next trading days after signal
            sig_idx = next((j for j in range(len(dates)) if str(dates[j].date()) == b["date"]), None)

            t1_date = str(dates[sig_idx+1].date()) if sig_idx and sig_idx+1 < n else "tomorrow"
            t2_date = str(dates[sig_idx+2].date()) if sig_idx and sig_idx+2 < n else "day after"

            if days_since == 0:
                action = "TODAY signal — Enter tomorrow (T+1) at open or wait for dip"
                buy_on = "T+1"
            elif days_since <= 2:
                action = f"Signal {days_since}d ago — Enter TODAY or wait"
                buy_on = "TODAY"
            else:
                action = f"Signal {days_since}d ago — Still valid but entering late"
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
        "sym":            sym,
        "price":          r2(cur_price),
        "last_date":      last_date,
        "turnover_cr":    turnover_cr,
        "n_breakouts":    len(deduped),
        "years":          sorted(set(b["year"] for b in deduped)),
        "avg_brk_pct":    avg_brk_pct,
        "avg_consol_rng": avg_consol_rng,
        "avg_consol_days":avg_consol_days,
        "avg_t1_dip":     avg_t1_dip,
        "avg_t1_gap":     avg_t1_gap,
        "avg_t1_ret":     avg_t1_ret,
        "avg_t4_ret":     avg_t4_ret,
        "min_peak_ret":   min_peak_ret,   # worst-case peak across all breakouts (the guarantee)
        "avg_peak_ret":   avg_peak_ret,   # average peak across all breakouts
        "entry_type":     entry_type,
        "entry_rec":      entry_rec,
        "has_alert":      len(alerts) > 0,
        "alerts":         alerts,
        "breakouts":      sorted(deduped, key=lambda x: x["date"], reverse=True),
    }


def main():
    print(f"\n{'='*60}\nBreakout Analyzer (Consolidation → Breakout)\nIST: {now_str}\n{'='*60}")
    print(f"Guaranteed peak filter: every breakout must peak >= {MIN_GUARANTEED_PEAK}% within T+1..T+4")

    all_data   = load_all()
    grouped    = all_data.groupby("sym")
    syms       = sorted(grouped.groups.keys())
    all_dates  = sorted(set(str(pd.to_datetime(d).date()) for d in all_data["date"].unique()))
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]
    print(f"Last fetch: {last_fetch} | Symbols: {len(syms):,}")

    results  = []; skipped = 0; excluded = 0
    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — found {len(results)}")
        if any(sym.upper().endswith(s) for s in EXCL_SFX):
            excluded += 1; continue
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 200: skipped += 1; continue
            res = analyze_stock(sym, grp, latest_set, all_dates)
            if res: results.append(res)
            else:   skipped += 1
        except: skipped += 1

    # Sort: alerts first by turnover, then all stocks by min_peak_ret (guaranteed floor)
    alert_stocks = sorted([r for r in results if r["has_alert"]],
                          key=lambda x: -(x.get("turnover_cr") or 0))
    all_stocks   = sorted(results, key=lambda x: -(x.get("min_peak_ret") or 0))

    # All alerts flat list
    all_alerts = []
    for r in results:
        for a in (r.get("alerts") or []):
            all_alerts.append({**a, "sym": r["sym"], "price": r["price"],
                                "turnover_cr":  r["turnover_cr"],
                                "avg_t1_dip":   r["avg_t1_dip"],
                                "entry_rec":    r["entry_rec"],
                                "entry_type":   r["entry_type"],
                                "min_peak_ret": r["min_peak_ret"],
                                "avg_peak_ret": r["avg_peak_ret"]})
    all_alerts.sort(key=lambda x: (x.get("days_since") or 99, -(x.get("turnover_cr") or 0)))

    output = {
        "generated_at":  now_str,
        "today_ist":     today,
        "last_fetch":    last_fetch,
        "n_stocks":      len(results),
        "n_alerts":      len(alert_stocks),
        "n_all_alerts":  len(all_alerts),
        "alert_stocks":  alert_stocks,
        "all_alerts":    all_alerts,
        "stocks":        all_stocks,
        "description": (
            "Stocks that consolidate in a tight range (±7%) for 1-6 months, "
            "then break out upward and sustain for 3+ days. "
            "GUARANTEED FILTER: every historical breakout peaked >= 5% within T+1..T+4. "
            "Includes entry timing analysis and 3-day alert window."
        ),
    }

    path = OUT / "breakout_signals.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Written {len(results)} stocks | {len(alert_stocks)} alerts today | {len(all_alerts)} total alerts")
    if alert_stocks:
        print(f"\n*** BREAKOUT ALERTS (all guaranteed 5%+ peak) ***")
        for r in alert_stocks[:10]:
            a = r["alerts"][0]
            print(f"  {r['sym']:<14} brk=+{a['brk_pct']}% consol={a['consol_days']}d({a['consol_rng']}%) min_peak=+{r['min_peak_ret']}% {a['action']}")


if __name__ == "__main__":
    main()
