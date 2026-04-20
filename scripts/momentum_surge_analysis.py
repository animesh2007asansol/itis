#!/usr/bin/env python3
"""
momentum_surge_analysis.py
===========================
Finds stocks where a single-day surge of >6% triggers a momentum streak
of 3-4 consecutive positive days with meaningful returns.

Strategy logic:
  DAY 0  — Trigger: stock closes >MIN_SURGE% above previous close
             AND volume × price ≥ MIN_VALUE_CR crore (institutional confirmation)
             AND avg daily turnover (60d) ≥ MIN_AVG_TURNOVER (1000Cr+ proxy)
             AND stock price > MIN_PRICE (Rs 10)

  DAY 1  — Next trading day: does it open gap-up? Does it close higher?
             Track: open vs D0 close, close vs D0 close, intraday low (safe buy?)

  DAY 2  — Second day: entry signal — buy at D2 OPEN (or D1 close if held overnight)
             Track: open, close, how much it moved from D1 close

  DAY 3+ — Hold: at what point does the streak end?
             Track returns from D2 close entry for D3 and D4

Filter: pattern must have occurred ≥8 times in last 3 years with 100% consistency
        (every occurrence must have given ≥MIN_CONSEC_DAYS consecutive positive closes)

Output → pattern_signals/momentum_surge.json
"""

import json, sys, os
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install pandas numpy"); sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "pattern_signals"
MANIFEST  = DATA_DIR / "manifest.json"

# ── Thresholds ──────────────────────────────────────────────────────────────
MIN_SURGE           = 6.0       # trigger: day rises >6% vs prev close
MIN_CONSEC_DAYS     = 3         # must have at least 3 consecutive up days after trigger
MIN_OCC_3YR         = 8         # minimum 8 occurrences in last 3 years
MIN_STREAK_PCT      = 5.0       # cumulative gain over the streak (D1+D2+D3) must be >5%
MIN_PRICE           = 10.0      # stock price > Rs 10
MIN_VALUE_CR        = 2.0       # signal day: volume × price ≥ Rs 2 Cr
MIN_AVG_TURNOVER    = 10_000_000  # 60d avg daily turnover ≥ Rs 1 Cr (proxy for 1000Cr+ mcap)
LOOKBACK_YEARS_FULL = 5         # full history for 100% check
LOOKBACK_YEARS_OCC  = 3         # occurrences count window

# ── Column aliases ───────────────────────────────────────────────────────────
COL_MAP = {
    "SYMBOL":"SYMBOL","TCKRSYMB":"SYMBOL",
    "SERIES":"SERIES","SCTYSRS":"SERIES",
    "OPEN":"OPEN","OPNPRIC":"OPEN","OPEN PRICE":"OPEN",
    "HIGH":"HIGH","HGHPRIC":"HIGH","HIGH PRICE":"HIGH",
    "LOW":"LOW","LWPRIC":"LOW","LOW PRICE":"LOW",
    "CLOSE":"CLOSE","CLSPRIC":"CLOSE","CLOSE PRICE":"CLOSE","LASTPRIC":"CLOSE",
    "TOTTRDQTY":"VOLUME","TTLTRADGVOL":"VOLUME",
    "TIMESTAMP":"DATE","DATE":"DATE","DATE1":"DATE","BIZDT":"DATE",
}

EXCLUDED_EXACT = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","LIQUISETF","NIFTYBEES",
    "JUNIORBEES","BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES",
    "ITBEES","INFRABEES","PHARMABEES","CPSEETF","NIFTYIETF","SETFNIF50",
}
EXCLUDED_SUFFIX = ("ETF","BEES","CASE","SETF","GILT")


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)): return str(obj)
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.bool_):     return bool(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

def r2(n): return round(float(n)*100)/100 if n is not None else None


def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2: return rows
        hdr = [h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]

        def fc(aliases):
            for a in aliases:
                if a in hdr: return hdr.index(a)
            return -1

        i_sym = fc(["SYMBOL","TCKRSYMB"])
        i_ser = fc(["SERIES","SCTYSRS"])
        i_o   = fc(["OPEN","OPNPRIC","OPEN PRICE"])
        i_h   = fc(["HIGH","HGHPRIC","HIGH PRICE"])
        i_l   = fc(["LOW","LWPRIC","LOW PRICE"])
        i_c   = fc(["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"])
        i_v   = fc(["TOTTRDQTY","TTLTRADGVOL","VOLUME"])

        if i_sym < 0 or i_c < 0: return rows
        mc = max(x for x in [i_sym, i_o, i_h, i_l, i_c, i_v] if x >= 0)

        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= mc: continue
            ser = cols[i_ser].strip() if i_ser >= 0 else "EQ"
            if ser not in ("EQ", "BE"): continue
            try:
                sym = cols[i_sym].strip()
                c   = float(cols[i_c])
                o   = float(cols[i_o]) if i_o >= 0 else c
                h   = float(cols[i_h]) if i_h >= 0 else c
                l   = float(cols[i_l]) if i_l >= 0 else c
                v   = float(cols[i_v].replace(",","")) if i_v >= 0 else 0.0
                if c > 0 and o > 0 and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    return rows


def load_all(trading_days):
    print(f"  Loading {len(trading_days)} trading days...")
    rows = []; loaded = 0
    for ds in trading_days:
        y, m, _ = ds.split("-")
        path = DATA_DIR / "equity" / y / m / f"{ds}.csv"
        if not path.exists(): continue
        r = load_csv(path)
        for x in r: x["date"] = ds
        rows.extend(r); loaded += 1
        if loaded % 300 == 0:
            print(f"    {loaded}/{len(trading_days)} files, {len(rows):,} rows...")
    print(f"  {len(rows):,} rows, {loaded} files")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)


def analyse_stock(sym, df):
    """
    For one stock, find all surge events and characterise what happens next.
    Returns a result dict or None if doesn't qualify.
    """
    if sym in EXCLUDED_EXACT: return None
    if any(sym.upper().endswith(s) for s in EXCLUDED_SUFFIX): return None

    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)
    if n < 60: return None

    c, o, h, l, v = df["c"], df["o"], df["h"], df["l"], df["v"]

    # Turnover and price filters
    tv = c * v   # daily turnover in Rs
    avg_tv_60 = float(tv.iloc[-60:].mean()) if n >= 60 else float(tv.mean())

    if avg_tv_60 < MIN_AVG_TURNOVER: return None   # below 1000Cr proxy
    if float(c.iloc[-1]) < MIN_PRICE: return None

    # Previous close for each day
    prev_c = c.shift(1)

    # Date bounds
    all_dates   = df["date"]
    latest_date = all_dates.max()
    cutoff_3yr  = latest_date - pd.DateOffset(years=LOOKBACK_YEARS_OCC)
    cutoff_5yr  = latest_date - pd.DateOffset(years=LOOKBACK_YEARS_FULL)

    occurrences_3yr  = []
    occurrences_all  = []

    for i in range(1, n - 4):   # need at least 4 days after trigger
        day0_c    = float(c.iloc[i])
        day0_prev = float(prev_c.iloc[i])
        if day0_prev <= 0: continue

        # ── Trigger: rise >6% ────────────────────────────────────────────────
        surge_pct = (day0_c - day0_prev) / day0_prev * 100
        if surge_pct < MIN_SURGE: continue

        # ── Signal day quality filters ───────────────────────────────────────
        day0_v     = float(v.iloc[i])
        day0_tv    = day0_c * day0_v
        if day0_tv < MIN_VALUE_CR * 1e7: continue     # Rs 2 Cr minimum
        if day0_c < MIN_PRICE: continue

        # ── Check next 4 days exist ──────────────────────────────────────────
        if i + 4 >= n: continue

        # Extract next 4 rows
        d = {}
        for j in range(1, 5):
            d[j] = {
                "date":  str(df["date"].iloc[i+j].date()),
                "o":     float(o.iloc[i+j]),
                "h":     float(h.iloc[i+j]),
                "l":     float(l.iloc[i+j]),
                "c":     float(c.iloc[i+j]),
                "v":     float(v.iloc[i+j]),
                "tv":    float(c.iloc[i+j] * v.iloc[i+j]),
                "prev_c":float(c.iloc[i+j-1]),
            }

        # ── Consecutive positive check ───────────────────────────────────────
        # Check days 1,2,3 all positive vs previous close
        days_positive = sum(1 for j in [1,2,3] if d[j]["c"] > d[j]["prev_c"])
        day4_positive = d[4]["c"] > d[4]["prev_c"]
        consec        = days_positive  # minimum required
        if consec < MIN_CONSEC_DAYS: continue

        # ── Cumulative return from D0 close over D1+D2+D3 ───────────────────
        cum_d3 = (d[3]["c"] - day0_c) / day0_c * 100
        if cum_d3 < MIN_STREAK_PCT: continue   # total must be >5%

        # ── Day-by-day metrics ───────────────────────────────────────────────
        # Day 1
        d1_open_vs_d0  = r2((d[1]["o"] - day0_c) / day0_c * 100)
        d1_close_ret   = r2((d[1]["c"] - day0_c)  / day0_c * 100)
        d1_low_vs_d0   = r2((d[1]["l"] - day0_c)  / day0_c * 100)   # dip below d0?
        d1_intra_range = r2((d[1]["h"] - d[1]["l"]) / d[1]["l"] * 100)

        # Day 2 CLOSE is the entry price (buy at end of D2)
        entry_px       = d[2]["c"]
        d2_open_vs_d1  = r2((d[2]["o"] - d[1]["c"]) / d[1]["c"] * 100)
        d2_close_ret_from_d0 = r2((d[2]["c"] - day0_c) / day0_c * 100)
        # How much D2 LOW was below D2 OPEN (entry dip opportunity)
        d2_low_vs_open = r2((d[2]["l"] - d[2]["o"]) / d[2]["o"] * 100)

        # Day 3 return from D2 close (entry)
        d3_ret_from_entry = r2((d[3]["c"] - entry_px) / entry_px * 100)
        d3_high_from_entry= r2((d[3]["h"] - entry_px) / entry_px * 100)
        d3_low_from_entry = r2((d[3]["l"] - entry_px) / entry_px * 100)

        # Day 4 return from D2 close (hold)
        d4_ret_from_entry = r2((d[4]["c"] - entry_px) / entry_px * 100)
        d4_high_from_entry= r2((d[4]["h"] - entry_px) / entry_px * 100)

        occ = {
            "date_d0":          str(df["date"].iloc[i].date()),
            "date_d1":          d[1]["date"],
            "date_d2":          d[2]["date"],
            "date_d3":          d[3]["date"],
            "date_d4":          d[4]["date"],
            # D0 trigger
            "d0_prev_close":    r2(day0_prev),
            "d0_close":         r2(day0_c),
            "d0_surge_pct":     r2(surge_pct),
            "d0_volume":        int(day0_v),
            "d0_turnover_cr":   r2(day0_tv / 1e7),
            # D1 — first day after surge
            "d1_open":          r2(d[1]["o"]),
            "d1_close":         r2(d[1]["c"]),
            "d1_low":           r2(d[1]["l"]),
            "d1_open_vs_d0":    d1_open_vs_d0,
            "d1_close_ret":     d1_close_ret,
            "d1_low_dip":       d1_low_vs_d0,
            "d1_positive":      d[1]["c"] > d[1]["prev_c"],
            # D2 — entry day (buy at D2 close)
            "d2_open":          r2(d[2]["o"]),
            "d2_close":         r2(d[2]["c"]),
            "d2_low":           r2(d[2]["l"]),
            "d2_open_vs_d1":    d2_open_vs_d1,
            "d2_cum_from_d0":   d2_close_ret_from_d0,
            "d2_low_vs_open":   d2_low_vs_open,
            "entry_px":         r2(entry_px),
            "d2_positive":      d[2]["c"] > d[2]["prev_c"],
            # D3 — first full day holding from entry
            "d3_close":         r2(d[3]["c"]),
            "d3_ret_from_entry":d3_ret_from_entry,
            "d3_high_from_entry":d3_high_from_entry,
            "d3_low_from_entry": d3_low_from_entry,
            "d3_positive":      d[3]["c"] > d[3]["prev_c"],
            "d3_positive_from_entry": d3_ret_from_entry > 0,
            # D4 — second day holding
            "d4_close":         r2(d[4]["c"]),
            "d4_ret_from_entry":d4_ret_from_entry,
            "d4_high_from_entry":d4_high_from_entry,
            "d4_positive_from_entry": d4_ret_from_entry > 0,
            # Summary
            "cum_d3_from_d0":   r2(cum_d3),
            "total_consec_up":  consec + (1 if day4_positive else 0),
        }

        # Store in appropriate bucket
        ev_date = df["date"].iloc[i]
        occurrences_all.append(occ)
        if ev_date >= cutoff_3yr:
            occurrences_3yr.append(occ)

    if not occurrences_3yr: return None
    if len(occurrences_3yr) < MIN_OCC_3YR: return None

    # ── 100% consistency check across ALL history (5 years) ─────────────────
    all_5yr = [o for o in occurrences_all
                if pd.Timestamp(o["date_d0"]) >= cutoff_5yr]

    n_all     = len(all_5yr)
    n_d3_pos  = sum(1 for o in all_5yr if o["d3_positive_from_entry"])
    n_d4_pos  = sum(1 for o in all_5yr if o["d4_positive_from_entry"])
    wr_d3     = round(n_d3_pos / n_all * 100, 1) if n_all > 0 else 0

    # For strict consistency: at least D3 must be 100% positive from entry
    # (D4 bonus check)
    if wr_d3 < 100.0:
        return None   # not 100% win rate at D3 from D2 close entry

    # ── Aggregated stats ─────────────────────────────────────────────────────
    occ_3yr = occurrences_3yr
    n3      = len(occ_3yr)

    def avg(lst):   return r2(sum(lst)/len(lst)) if lst else None
    def mn(lst):    return r2(min(lst)) if lst else None
    def mx(lst):    return r2(max(lst)) if lst else None

    d3_rets  = [o["d3_ret_from_entry"] for o in occ_3yr]
    d4_rets  = [o["d4_ret_from_entry"] for o in occ_3yr]
    surges   = [o["d0_surge_pct"] for o in occ_3yr]
    d1_opens = [o["d1_open_vs_d0"] for o in occ_3yr]
    d1_close = [o["d1_close_ret"] for o in occ_3yr]
    d1_dips  = [o["d1_low_dip"] for o in occ_3yr]
    d2_lows  = [o["d2_low_vs_open"] for o in occ_3yr]
    cum_d3   = [o["cum_d3_from_d0"] for o in occ_3yr]
    entries  = [o["entry_px"] for o in occ_3yr]
    turnovers= [o["d0_turnover_cr"] for o in occ_3yr]

    # Years spread
    years = sorted(set(o["date_d0"][:4] for o in occ_3yr))

    # Best sell strategy from D2 entry
    # D3 sell: always sells at D3 close
    # D4 sell: better if D4 also positive
    best_sell = "D3 close" if avg(d3_rets) >= avg(d4_rets) else "D4 close"

    # D1 entry opportunity: can you buy cheaper than D0 close on D1?
    # If D1 low < D0 close, you could have bought at D0 close or lower
    n_d1_dips = sum(1 for d in d1_dips if d < 0)  # D1 went below D0 close
    pct_d1_dip= round(n_d1_dips / n3 * 100, 1)
    avg_d1_dip= avg([abs(d) for d in d1_dips if d < 0]) or 0

    # D2 intraday: can you buy cheaper than D2 open on D2?
    n_d2_dips = sum(1 for d in d2_lows if d < 0)
    pct_d2_dip= round(n_d2_dips / n3 * 100, 1)
    avg_d2_dip= avg([abs(d) for d in d2_lows if d < 0]) or 0

    # Risk: what was worst D3 return?
    worst_d3 = min(d3_rets)  # already filtered to 100% positive, so should be >0

    # Latest occurrence
    latest_occ = max(occ_3yr, key=lambda o: o["date_d0"])

    return {
        "sym":                sym,
        "n_occ_3yr":          n3,
        "n_occ_all":          len(occurrences_all),
        "years_3yr":          years,
        "n_years":            len(years),
        "avg_turnover_60d_cr":r2(avg_tv_60 / 1e7),
        "wr_d3_100pct":       True,   # always True — we filtered above
        "wr_d4_pct":          round(n_d4_pos / n_all * 100, 1) if n_all > 0 else 0,
        # D0 trigger stats
        "avg_surge_pct":      avg(surges),
        "min_surge_pct":      mn(surges),
        "max_surge_pct":      mx(surges),
        # D1 — day 1 after surge
        "avg_d1_open_vs_d0":  avg(d1_opens),
        "avg_d1_close_ret":   avg(d1_close),
        "pct_d1_dip_below_d0":pct_d1_dip,
        "avg_d1_dip_pct":     avg_d1_dip,
        # D2 — entry day
        "avg_d2_cum_from_d0": avg([o["d2_cum_from_d0"] for o in occ_3yr]),
        "pct_d2_intra_dip":   pct_d2_dip,
        "avg_d2_intra_dip":   avg_d2_dip,
        # D3 return from D2 close entry (100% positive guaranteed)
        "avg_d3_ret":         avg(d3_rets),
        "min_d3_ret":         mn(d3_rets),
        "max_d3_ret":         mx(d3_rets),
        # D4 return from D2 close entry
        "avg_d4_ret":         avg(d4_rets),
        "min_d4_ret":         mn(d4_rets),
        "wr_d4_positive":     round(sum(1 for r in d4_rets if r > 0)/len(d4_rets)*100,1) if d4_rets else 0,
        # Cumulative
        "avg_cum_d3_from_d0": avg(cum_d3),
        # Trade recommendation
        "best_sell_day":      best_sell,
        "avg_turnover_cr":    avg(turnovers),
        "min_turnover_cr":    mn(turnovers),
        # Latest occurrence
        "latest_date":        latest_occ["date_d0"],
        "latest_entry_px":    latest_occ["entry_px"],
        "latest_surge_pct":   latest_occ["d0_surge_pct"],
        # All occurrences (for table)
        "occurrences":        sorted(occ_3yr, key=lambda o: o["date_d0"], reverse=True),
    }


def main():
    print("="*65)
    print("Momentum Surge Analysis")
    print(f"  Trigger: >{MIN_SURGE}% single-day surge")
    print(f"  Required: {MIN_CONSEC_DAYS}+ consecutive up days, >{MIN_STREAK_PCT}% total gain")
    print(f"  Occurrences: >={MIN_OCC_3YR} in last {LOOKBACK_YEARS_OCC} years")
    print(f"  D3 from D2-close entry: must be 100% positive in 5yr history")
    print(f"  Min trade value: Rs {MIN_VALUE_CR} Cr on signal day")
    print(f"  Avg daily turnover: >=Rs {MIN_AVG_TURNOVER/1e7:.0f} Cr (1000Cr+ mcap proxy)")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f: manifest = json.load(f)
    tds        = sorted(manifest.keys())
    latest_str = tds[-1]
    print(f"\n  {len(tds)} trading days [{tds[0]} -> {latest_str}]")

    print("\n[1] Loading all equity data...")
    df_all = load_all(manifest)
    sym_list = sorted(df_all["sym"].unique())
    print(f"  {len(sym_list)} unique symbols")

    print(f"\n[2] Analysing {len(sym_list)} stocks...")
    results   = []
    skipped   = 0
    qualified = 0

    sym_grps = {s: g.copy() for s, g in df_all.groupby("sym")}
    del df_all

    for i, sym in enumerate(sym_list):
        try:
            res = analyse_stock(sym, sym_grps[sym])
            if res:
                results.append(res)
                qualified += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(sym_list)} processed | qualified: {qualified}")

    del sym_grps

    # Sort: highest occurrences first, then highest avg D3 return
    results.sort(key=lambda x: (-x["n_occ_3yr"], -(x["avg_d3_ret"] or 0)))

    print(f"\n  Qualified: {qualified}")
    print(f"  Top results:")
    for r in results[:10]:
        print(f"    {r['sym']:<14} occ={r['n_occ_3yr']}  "
              f"D3_avg=+{r['avg_d3_ret']:.1f}%  "
              f"D3_min=+{r['min_d3_ret']:.1f}%  "
              f"D4_wr={r['wr_d4_positive']:.0f}%  "
              f"turnover={r['avg_turnover_60d_cr']:.1f}Cr")

    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    out = {
        "generated_at":   now_ist,
        "latest_date":    latest_str,
        "params": {
            "min_surge_pct":      MIN_SURGE,
            "min_consec_days":    MIN_CONSEC_DAYS,
            "min_streak_pct":     MIN_STREAK_PCT,
            "min_occ_3yr":        MIN_OCC_3YR,
            "min_price":          MIN_PRICE,
            "min_value_cr":       MIN_VALUE_CR,
            "min_avg_turnover_cr":MIN_AVG_TURNOVER/1e7,
        },
        "n_qualified": qualified,
        "results":     results,
    }
    jdump(out, OUT_DIR / "momentum_surge.json")
    print(f"\nOK: momentum_surge.json ({qualified} stocks)")

if __name__ == "__main__":
    main()
