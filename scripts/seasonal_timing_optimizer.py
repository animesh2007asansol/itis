#!/usr/bin/env python3
"""
seasonal_timing_optimizer.py
=============================
Analyzes ALL qualifying stocks (UC1 Surge, UC2 Seasonal, UC4 Technical) to find
the optimal entry day and exit day within 30 trading days that maximizes profit.

KEY INSIGHT: If stock dips AFTER the best entry day, entering on the dip
gives MORE remaining upside to the avg return target. The output includes
a day-by-day return table so the dashboard can:
  - Show any-day entry: from today's price, how much remains to avg/min target
  - Track live profit vs target as price moves daily

Outputs:
  stock_analysis/timing_uc2.json  — UC2 seasonal monthly timing
  stock_analysis/timing_uc1.json  — UC1 surge timing
  stock_analysis/timing_all.json  — all combined, sorted by month + peak return
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

# ── CONFIG ───────────────────────────────────────────────────────────────────
MIN_PRICE     = 10.0
MIN_TURNOVER  = 5_000_000   # Rs 5 Cr
MIN_OCC       = 3
WIN_RATE      = 95.0
MIN_RETURN    = 10.0
ENTRY_WINDOW  = range(-5, 6)   # -5 to +5 days from signal
EXIT_WINDOW   = range(1, 31)   # 1 to 30 trading days after entry

MON = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
r2  = lambda x: round(float(x), 2) if x is not None else None


# ── DATA LOAD ─────────────────────────────────────────────────────────────────
def load_all_data():
    manifest = json.loads(MANIFEST.read_text())
    dates = sorted(manifest.keys())
    print(f"  Loading {len(dates)} dates...")
    frames = []
    for ds in dates:
        y, m, _ = ds.split("-")
        p = DATA / y / m / f"{ds}.csv"
        if not p.exists(): continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = df.columns.str.strip()
            cm = {}
            for c in df.columns:
                u = c.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):            cm[c]="sym"
                elif u in ("SERIES","SCTYSRS"):           cm[c]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):cm[c]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):cm[c]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):   cm[c]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"):cm[c]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):    cm[c]="v"
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


# ── CORE: EXIT CURVE ──────────────────────────────────────────────────────────
def build_exit_curve(df, signal_indices, entry_offset=0):
    """
    For a given stock df and list of signal day indices,
    build a day-by-day return curve (day 1 to 30 after entry).

    Returns list of {day, avg_ret, min_ret, max_ret, win_rate, n,
                     avg_price_vs_entry, drawdown_from_peak}
    Also returns: entry_price_avg (avg open on actual entry days)
    """
    o = df["o"].values
    c = df["c"].values
    h = df["h"].values
    l = df["l"].values
    n = len(df)

    curve = []
    entry_prices = []

    for exit_day in EXIT_WINDOW:
        rets = []
        max_gains = []
        ep_list   = []

        for si in signal_indices:
            ei = si + entry_offset     # entry signal day
            ai = ei + 1                # actual entry = next open
            xi = ai + exit_day         # exit close

            if ai < 0 or ai >= n or xi >= n: continue

            ep = o[ai]
            if ep <= 0: continue

            # Return at exit day
            ret = (c[xi] - ep) / ep * 100
            rets.append(ret)
            ep_list.append(ep)

            # Max gain in window (best close in period)
            best = max(c[ai:xi+1]) if xi >= ai else c[ai]
            max_gains.append((best - ep) / ep * 100)

        if len(rets) < 2: continue

        avg = sum(rets) / len(rets)
        avg_mg = sum(max_gains) / len(max_gains) if max_gains else avg

        curve.append({
            "day":           exit_day,
            "avg_ret":       r2(avg),
            "min_ret":       r2(min(rets)),
            "max_ret":       r2(max(rets)),
            "avg_max_gain":  r2(avg_mg),
            "win_rate":      r2(sum(1 for r in rets if r > 0) / len(rets) * 100),
            "n":             len(rets),
        })
        if exit_day == 1:
            entry_prices = ep_list

    avg_entry_price = r2(sum(entry_prices) / len(entry_prices)) if entry_prices else None
    return curve, avg_entry_price


def find_optimal(curve):
    """From a curve list, find peak day, sweet spot, decline day."""
    if not curve: return {}
    vals = [p["avg_ret"] for p in curve]
    days = [p["day"]     for p in curve]

    peak_idx = int(np.argmax(vals))
    peak_day = days[peak_idx]
    peak_ret = vals[peak_idx]

    # Sweet spot: ≥80% of peak return
    sweet_start = peak_day; sweet_end = peak_day
    for j, d in enumerate(days):
        if vals[j] >= peak_ret * 0.80:
            if d < sweet_start: sweet_start = d
            if d > sweet_end:   sweet_end   = d

    # Decline: returns consistently drop ≥1.5% from peak after it
    decline_day = None
    for i in range(peak_idx + 1, len(vals)):
        if vals[i] <= peak_ret - 1.5:
            confirm = [vals[j] <= peak_ret - 1.0
                       for j in range(i, min(i+3, len(vals)))]
            if sum(confirm) >= 2:
                decline_day = days[i]
                break

    return {
        "peak_day":       peak_day,
        "peak_avg_ret":   r2(peak_ret),
        "sweet_start":    sweet_start,
        "sweet_end":      sweet_end,
        "decline_after":  decline_day,
    }


def build_daily_profile(df, signal_indices, best_entry_offset, avg_entry_price):
    """
    For each day 0-30 after the best-entry signal, compute:
    - avg price that day (open, high, low, close)
    - cumulative return from avg entry price
    - This lets the UI show: if current price = X, remaining to avg target = Y%

    Returns list of {day, avg_open, avg_close, avg_ret_from_entry, avg_high, avg_low}
    """
    o = df["o"].values
    c = df["c"].values
    h = df["h"].values
    l = df["l"].values
    n = len(df)
    profile = []

    for day in range(0, 31):
        opens = []; closes = []; highs = []; lows = []
        for si in signal_indices:
            ei = si + best_entry_offset
            ai = ei + 1  # actual entry
            di = ai + day
            if ai < 0 or ai >= n or di >= n: continue
            ep = o[ai]
            if ep <= 0: continue
            opens.append(o[di])
            closes.append(c[di])
            highs.append(h[di])
            lows.append(l[di])

        if not closes: continue
        avg_c = sum(closes) / len(closes)
        ep    = avg_entry_price or 1

        profile.append({
            "day":      day,
            "avg_open":  r2(sum(opens)  / len(opens)),
            "avg_high":  r2(sum(highs)  / len(highs)),
            "avg_low":   r2(sum(lows)   / len(lows)),
            "avg_close": r2(avg_c),
            "avg_ret_from_entry": r2((avg_c - ep) / ep * 100),
        })

    return profile


# ── UC2 SEASONAL ──────────────────────────────────────────────────────────────
def analyze_uc2(df, sym):
    results = []
    c  = df["c"].values
    dates = pd.to_datetime(df["date"].values)
    mons  = pd.DatetimeIndex(dates).month
    yrs   = pd.DatetimeIndex(dates).year
    n     = len(df)

    for mo in range(1, 13):
        # First trading day of each occurrence
        first_days = [i for i in range(n) if mons[i]==mo and (i==0 or mons[i-1]!=mo)]
        if len(first_days) < MIN_OCC: continue

        # Qualify: 1st-day entry with best hold up to 30d
        best_hold = None; best_ret = -999
        for hold in [10, 15, 20, 25, 30]:
            rets = []
            for fi in first_days:
                ai = fi + 1; xi = ai + hold
                if xi >= n or ai >= n: continue
                if xi < ai + hold: continue   # incomplete
                ep = df["o"].iloc[ai]
                if ep <= 0: continue
                ret = (df["c"].iloc[xi] - ep) / ep * 100
                rets.append(ret)
            if len(rets) < MIN_OCC: continue
            if any(r <= 0 for r in rets): continue
            wr = sum(1 for r in rets if r >= MIN_RETURN) / len(rets) * 100
            if wr < WIN_RATE: continue
            avg = sum(rets) / len(rets)
            if avg > best_ret:
                best_ret  = avg
                best_hold = hold
                base_rets = rets
                base_yrs  = [int(yrs[fi]) for fi in first_days if fi+1+hold < n]

        if best_hold is None: continue

        # Entry window: find best offset
        best_offset = 0; best_offset_ret = best_ret
        for offset in ENTRY_WINDOW:
            rets = []
            for fi in first_days:
                ai = fi + offset + 1; xi = ai + best_hold
                if ai < 0 or xi >= n or ai >= n: continue
                if xi < ai + best_hold: continue
                ep = df["o"].iloc[ai]
                if ep <= 0: continue
                ret = (df["c"].iloc[xi] - ep) / ep * 100
                rets.append(ret)
            if len(rets) < MIN_OCC: continue
            if any(r <= 0 for r in rets): continue
            wr = sum(1 for r in rets if r >= MIN_RETURN) / len(rets) * 100
            if wr < WIN_RATE: continue
            avg = sum(rets) / len(rets)
            if avg > best_offset_ret:
                best_offset_ret = avg
                best_offset     = offset

        # Build exit curve
        curve, avg_ep = build_exit_curve(df, first_days, best_offset)
        if not curve: continue

        opt = find_optimal(curve)
        if not opt: continue

        # Entry analysis (all offsets)
        ea = []
        for off in ENTRY_WINDOW:
            rets = []
            for fi in first_days:
                ai = fi + off + 1; xi = ai + best_hold
                if ai < 0 or xi >= n or ai >= n: continue
                if xi < ai + best_hold: continue
                ep = df["o"].iloc[ai]
                if ep <= 0: continue
                ret = (df["c"].iloc[xi] - ep) / ep * 100
                rets.append(ret)
            if len(rets) < 2: continue
            label = ("1st of "+MON[mo]) if off==0 else (str(abs(off))+"d "+("before" if off<0 else "after")+" 1st of "+MON[mo])
            ea.append({
                "offset":   off,
                "label":    label,
                "avg_ret":  r2(sum(rets)/len(rets)),
                "min_ret":  r2(min(rets)),
                "max_ret":  r2(max(rets)),
                "win_rate": r2(sum(1 for r in rets if r>=MIN_RETURN)/len(rets)*100),
                "n":        len(rets),
            })

        # Daily price profile for live tracking
        dp = build_daily_profile(df, first_days, best_offset, avg_ep)

        # Entry recommendation
        if best_offset < 0:
            entry_rec = f"Enter {abs(best_offset)} trading day{'s' if abs(best_offset)>1 else ''} BEFORE 1st of {MON[mo]} — stock builds momentum early"
        elif best_offset == 0:
            entry_rec = f"Enter ON 1st trading day of {MON[mo]}"
        else:
            entry_rec = f"Enter {best_offset} trading day{'s' if best_offset>1 else ''} AFTER 1st of {MON[mo]} — wait for initial volatility to settle"

        peak_day = opt["peak_day"]
        pk_ret   = opt["peak_avg_ret"]
        target_price = r2(avg_ep * (1 + pk_ret/100)) if avg_ep else None
        min_target   = r2(avg_ep * (1 + min(base_rets)/100)) if avg_ep and base_rets else None

        exit_rec = f"Exit on day {peak_day} after entry (avg +{pk_ret:.1f}%). Sweet spot: day {opt['sweet_start']}–{opt['sweet_end']}."
        if opt.get("decline_after"):
            exit_rec += f" Returns decline after day {opt['decline_after']}."

        results.append({
            "uc":              "UC2",
            "sym":             sym,
            "price":           r2(float(df["c"].iloc[-1])),
            "month":           mo,
            "month_name":      MON[mo],
            "season_win_rate": r2(sum(1 for r in base_rets if r>=MIN_RETURN)/len(base_rets)*100),
            "season_avg_ret":  r2(sum(base_rets)/len(base_rets)),
            "season_min_ret":  r2(min(base_rets)),
            "season_n_years":  len(base_rets),
            "season_years":    sorted(set(base_yrs)),
            "best_entry_offset": best_offset,
            "avg_entry_price": avg_ep,
            "target_price_avg": target_price,
            "min_target_price": min_target,
            "entry_recommendation": entry_rec,
            "exit_recommendation":  exit_rec,
            "entry_analysis":  ea,
            "exit_curve":      curve,
            "daily_profile":   dp,
            **opt,
        })

    return results


# ── UC1 SURGE ─────────────────────────────────────────────────────────────────
UC1_SURGE_PCT  = [3, 5, 7, 10, 15, 20]
UC1_SURGE_DAYS = [1, 2, 3, 5]
UC1_FWD_DAYS   = [5, 10, 15, 20, 25, 30]

def analyze_uc1(df, sym):
    """
    For UC1: find the best surge threshold + surge period that gives
    ≥95% win rate with ≥10% return. Then analyze the 30-day exit curve.
    Groups results by calendar month the signal tends to fire in.
    """
    c = df["c"].values
    o = df["o"].values
    n = len(df)
    dates = pd.to_datetime(df["date"].values)
    mons  = pd.DatetimeIndex(dates).month

    best_strategy = None; best_rate = -999

    for surge_d in UC1_SURGE_DAYS:
        roll = pd.Series(c).pct_change(surge_d).fillna(0).values * 100
        for surge_pct in UC1_SURGE_PCT:
            for fwd in UC1_FWD_DAYS:
                signal_days = [i for i in range(surge_d, n-fwd-1) if roll[i] >= surge_pct]
                if len(signal_days) < MIN_OCC: continue

                rets = []
                i = 0
                while i < len(signal_days):
                    si = signal_days[i]
                    ai = si + 1; xi = ai + fwd
                    if ai >= n or xi >= n: i+=1; continue
                    ep = o[ai]
                    if ep <= 0: i+=1; continue
                    ret = (c[xi] - ep) / ep * 100
                    rets.append(ret)
                    # skip overlapping
                    while i < len(signal_days) and signal_days[i] < xi: i+=1

                if len(rets) < MIN_OCC: continue
                if any(r <= 0 for r in rets): continue
                wr = sum(1 for r in rets if r >= MIN_RETURN) / len(rets) * 100
                if wr < WIN_RATE: continue
                avg = sum(rets) / len(rets)
                rate = avg / fwd
                if rate > best_rate:
                    best_rate     = rate
                    best_strategy = {
                        "surge_pct": surge_pct, "surge_days": surge_d,
                        "fwd_days": fwd, "avg_ret": r2(avg),
                        "min_ret": r2(min(rets)), "win_rate": r2(wr),
                        "n_trades": len(rets),
                    }

    if not best_strategy: return []

    # Rebuild signal days for best strategy
    surge_d   = best_strategy["surge_days"]
    surge_pct = best_strategy["surge_pct"]
    roll      = pd.Series(c).pct_change(surge_d).fillna(0).values * 100
    signal_days = [i for i in range(surge_d, n-31) if roll[i] >= surge_pct]
    if len(signal_days) < MIN_OCC: return []

    # Group by month to find which months this tends to fire
    month_signals = {}
    for si in signal_days:
        mo = int(mons[si])
        month_signals.setdefault(mo, []).append(si)

    results = []
    for mo, sigs in month_signals.items():
        if len(sigs) < 2: continue

        # Build exit curve (entry offset = 0 for UC1, signal = buy next day)
        curve, avg_ep = build_exit_curve(df, sigs, 0)
        if not curve: continue

        opt = find_optimal(curve)
        if not opt: continue

        dp = build_daily_profile(df, sigs, 0, avg_ep)
        pk_ret = opt["peak_avg_ret"]
        target_price = r2(avg_ep * (1 + pk_ret/100)) if avg_ep else None
        min_tgt      = r2(avg_ep * (1 + best_strategy["min_ret"]/100)) if avg_ep else None

        entry_rec = f"Buy next day open when stock rises {surge_pct}%+ in {surge_d} day(s)"
        exit_rec  = f"Exit on day {opt['peak_day']} (avg +{pk_ret:.1f}%). Sweet spot: day {opt['sweet_start']}–{opt['sweet_end']}."
        if opt.get("decline_after"):
            exit_rec += f" Returns decline after day {opt['decline_after']}."

        results.append({
            "uc":             "UC1",
            "sym":            sym,
            "price":          r2(float(df["c"].iloc[-1])),
            "month":          mo,
            "month_name":     MON[mo],
            "season_win_rate": best_strategy["win_rate"],
            "season_avg_ret":  best_strategy["avg_ret"],
            "season_min_ret":  best_strategy["min_ret"],
            "season_n_years":  len(sigs),
            "season_years":    sorted(set(int(mons[si]) for si in sigs)),
            "surge_pct":       surge_pct,
            "surge_days":      surge_d,
            "best_entry_offset": 0,
            "avg_entry_price": avg_ep,
            "target_price_avg": target_price,
            "min_target_price": min_tgt,
            "entry_recommendation": entry_rec,
            "exit_recommendation":  exit_rec,
            "entry_analysis":  [],
            "exit_curve":      curve,
            "daily_profile":   dp,
            **opt,
        })

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not MANIFEST.exists():
        print("ERROR: No manifest.json. Run main workflow first."); sys.exit(1)

    print(f"\n{'='*60}\nSeasonal Timing Optimizer (UC1 + UC2)\nStarted: {now}\n{'='*60}")
    all_data = load_all_data()
    grouped  = all_data.groupby("sym")
    syms     = sorted(grouped.groups.keys())
    print(f"\nAnalyzing {len(syms):,} symbols...")

    uc1_results = []; uc2_results = []; skipped = 0

    for i, sym in enumerate(syms):
        if (i+1) % 300 == 0:
            print(f"  {i+1}/{len(syms)} — UC1:{len(uc1_results)} UC2:{len(uc2_results)}")
        try:
            grp = grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp) < 250: skipped+=1; continue
            if float(grp["c"].iloc[-1]) < MIN_PRICE: skipped+=1; continue
            tv = grp["c"].iloc[-60:] * grp["v"].iloc[-60:]
            if float(tv.mean()) < MIN_TURNOVER: skipped+=1; continue

            uc2_results.extend(analyze_uc2(grp, sym))
            uc1_results.extend(analyze_uc1(grp, sym))
        except Exception as e:
            skipped += 1

    # Sort by month then peak return desc
    def sort_key(r): return (r["month"], -(r.get("peak_avg_ret") or 0))
    uc1_results.sort(key=sort_key)
    uc2_results.sort(key=sort_key)

    all_results = sorted(uc1_results + uc2_results, key=sort_key)

    # Write outputs
    for fname, data, label in [
        ("timing_uc2.json", {"generated_at":now,"n":len(uc2_results),"stocks":uc2_results}, "UC2"),
        ("timing_uc1.json", {"generated_at":now,"n":len(uc1_results),"stocks":uc1_results}, "UC1"),
        ("timing_all.json", {"generated_at":now,"n":len(all_results),"stocks":all_results}, "ALL"),
    ]:
        path = OUT / fname
        path.write_text(json.dumps(data, indent=2))
        print(f"  Written: {fname} ({data['n']} entries)")

    print(f"\nDone. UC2:{len(uc2_results)} UC1:{len(uc1_results)} Skipped:{skipped}")
    print("\nTop 10 by peak return:")
    top = sorted(all_results, key=lambda x: -(x.get("peak_avg_ret") or 0))[:10]
    for r in top:
        print(f"  {r['sym']:<12} {r['uc']} {r['month_name']:>3}: peak +{r['peak_avg_ret']:.1f}% on day {r['peak_day']}, entry offset {r['best_entry_offset']:+d}d")


if __name__ == "__main__":
    main()
