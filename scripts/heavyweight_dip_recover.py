#!/usr/bin/env python3
"""
heavyweight_dip_recover.py
===========================
Finds large-cap NSE stocks that have a repeating pattern:
  "Stock drops to a historical floor, then recovers sharply"

Logic:
  1. Large-cap filter: stock must have traded for 4+ years with
     avg daily turnover >= Rs 5 crore (eliminates illiquid stocks)

  2. Floor detection: find dates when the stock was at or near its
     lowest point in a rolling 52-week window (bottom 10% of range)

  3. Recovery verification: after each floor event, check if stock
     recovered by >= 15% in any of next 5/10/20/30 trading days

  4. Consistency: floor+recovery pattern must have occurred in 2+ years
     with no more than 2-year gap between occurrences

  5. Ranking: sort by highest guaranteed minimum recovery % in
     whichever timeframe gives the best result

  6. Alerts: if a floor event occurred in last 20 TRADING days, alert

Outputs (pattern_signals/ folder):
  heavyweight_profiles.json  — all qualifying stocks with full history
  heavyweight_alerts.json    — stocks that hit floor in last 20 trading days
"""

import json, gc, sys, math
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install pandas numpy"); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "pattern_signals"
MANIFEST  = DATA_DIR / "manifest.json"

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
MIN_YEARS_TRADED      = 4        # must have 4+ years of data
MIN_DAILY_TURNOVER_CR = 2.0      # avg daily turnover >= Rs 2 crore (heavyweight)
MIN_TRADING_DAYS      = 800      # ~4 years × ~200 trading days
FLOOR_PCT_THRESHOLD   = 10.0     # price in bottom 10% of 52-week range = floor
FLOOR_LOOKBACK        = 252      # 52-week window for floor detection
RECOVERY_WINDOWS      = [5, 10, 20, 30]  # trading days forward
MIN_RECOVERY_PCT      = 15.0     # minimum recovery in at least one window (%)
MIN_WIN_RATE          = 80.0     # at least 80% of floor events must recover 15%+
MIN_OCC               = 3        # minimum floor events needed
MIN_YEARS_PATTERN     = 2        # pattern must appear in 2+ years
MAX_YR_GAP            = 2        # no more than 2-year gap
ALERT_WINDOW          = 20       # last N trading days for alerts
MIN_VOLUME            = 50_000   # minimum volume on floor day

# ─────────────────────────────────────────────────────────────────────────────
# JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)): return str(obj)
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, np.bool_):      return bool(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIASES
# ─────────────────────────────────────────────────────────────────────────────
SYM_A = ["SYMBOL",    "TCKRSYMB"]
SER_A = ["SERIES",    "SCTYSRS"]
O_A   = ["OPEN",      "OPNPRIC"]
H_A   = ["HIGH",      "HGHPRIC"]
L_A   = ["LOW",       "LWPRIC"]
C_A   = ["CLOSE",     "CLSPRIC", "CLOSE PRICE", "LASTPRIC"]
V_A   = ["TOTTRDQTY", "TTLTRADGVOL", "VOLUME"]
TV_A  = ["TOTTRDVAL", "TTLTRADGVAL"]  # turnover value

def _fc(hdr, aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def _ph(raw):
    return [h.strip().strip('"').strip("'").upper() for h in raw.split(",")]

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2: return rows
        hdr   = _ph(lines[0])
        i_sym = _fc(hdr, SYM_A); i_ser = _fc(hdr, SER_A)
        i_o   = _fc(hdr, O_A);   i_h   = _fc(hdr, H_A)
        i_l   = _fc(hdr, L_A);   i_c   = _fc(hdr, C_A)
        i_v   = _fc(hdr, V_A);   i_tv  = _fc(hdr, TV_A)
        if i_sym < 0 or i_c < 0: return rows
        mc = max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v,i_tv] if x >= 0)
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= mc: continue
            ser = cols[i_ser].strip() if i_ser >= 0 else "EQ"
            if ser not in ("EQ","BE"): continue
            try:
                sym = cols[i_sym].strip()
                c   = float(cols[i_c])
                o   = float(cols[i_o]) if i_o >= 0 else c
                h   = float(cols[i_h]) if i_h >= 0 else c
                l   = float(cols[i_l]) if i_l >= 0 else c
                v   = float(cols[i_v].replace(",","")) if i_v >= 0 else 0.0
                tv  = float(cols[i_tv].replace(",","")) if i_tv >= 0 else (c * v)
                if c > 0 and o > 0 and h >= max(o,c) and l <= min(o,c) and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v,"tv":tv})
            except (ValueError, IndexError): pass
    except Exception: pass
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(trading_days):
    print(f"  Loading {len(trading_days)} trading days...")
    all_rows = []; loaded = 0
    for ds in trading_days:
        y, m, _ = ds.split("-")
        path = DATA_DIR / "equity" / y / m / f"{ds}.csv"
        if not path.exists(): continue
        rows = load_csv(path)
        for r in rows: r["date"] = ds
        all_rows.extend(rows)
        loaded += 1
        if loaded % 300 == 0:
            print(f"    {loaded}/{len(trading_days)} files, {len(all_rows):,} rows...")
    print(f"  Total: {len(all_rows):,} rows, {loaded} files")
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sym","date"]).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# HEAVYWEIGHT FILTER
# ─────────────────────────────────────────────────────────────────────────────
def filter_heavyweights(df):
    """
    Keep only stocks that:
      1. Have 4+ years of data (MIN_TRADING_DAYS rows)
      2. Avg daily turnover >= Rs 5 crore
    Turnover in NSE data is in thousands — we handle both raw and Lakh units.
    """
    counts   = df.groupby("sym")["date"].count()
    valid    = counts[counts >= MIN_TRADING_DAYS].index

    # Calculate avg turnover per stock
    # NSE TOTTRDVAL is in Rupees (not thousands), so divide by 1e7 for Crore
    avg_tv   = df.groupby("sym")["tv"].mean()

    # Try to detect unit: if median turnover looks like it's in Rupees
    # (large numbers) vs Lakhs (smaller numbers)
    # A Rs 5 Cr turnover = Rs 5,00,00,000 = 5e7
    # In NSE format: TOTTRDVAL is in Rupees
    threshold_rs = MIN_DAILY_TURNOVER_CR * 1e7   # Rs 5 crore in rupees

    # If max avg turnover < 1e6, data might be in Lakhs
    if avg_tv[avg_tv > 0].median() < 1e6:
        threshold_rs = MIN_DAILY_TURNOVER_CR * 1e2  # 5 crore in lakhs = 500

    heavy = avg_tv[avg_tv >= threshold_rs].index
    qual  = valid.intersection(heavy)
    print(f"  Heavyweight filter: {len(qual)} stocks "
          f"(from {len(valid)} with enough history)")
    return df[df["sym"].isin(qual)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# FLOOR DETECTION + RECOVERY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyse_stock(g, sym):
    """
    Find all floor events for one stock and check recovery.
    Returns list of floor event records, or None if doesn't qualify.
    """
    g = g.copy().sort_values("date").reset_index(drop=True)
    c, h, l, v = g["c"], g["h"], g["l"], g["v"]
    n = len(g)
    if n < MIN_TRADING_DAYS: return None

    # ── 52-week rolling low and high (shifted so today not included) ──────────
    roll_low  = l.rolling(FLOOR_LOOKBACK, min_periods=60).min().shift(1)
    roll_high = h.rolling(FLOOR_LOOKBACK, min_periods=60).max().shift(1)
    roll_rng  = roll_high - roll_low

    # ── Floor condition ───────────────────────────────────────────────────────
    # Price is in bottom FLOOR_PCT_THRESHOLD % of 52-week range
    # (close - 52w_low) / 52w_range <= threshold/100
    with np.errstate(divide='ignore', invalid='ignore'):
        pos_in_range = np.where(
            roll_rng > 0,
            (c - roll_low) / roll_rng * 100,
            50.0
        )
    g["pos_in_range"] = pos_in_range
    g["is_floor"]     = (g["pos_in_range"] <= FLOOR_PCT_THRESHOLD) & (v >= MIN_VOLUME)

    # ── Forward returns from each floor event ─────────────────────────────────
    for w in RECOVERY_WINDOWS:
        # VERIFIED: fwd_w = close[t+w] / close[t] - 1
        g[f"fwd_{w}"] = c.shift(-w) / c - 1

    # ── Collect floor events ──────────────────────────────────────────────────
    floor_rows = g[g["is_floor"]].copy()
    if len(floor_rows) < MIN_OCC: return None

    # Remove consecutive floor events (keep first of each cluster)
    # A cluster = floor events within 20 trading days of each other
    floor_dates = floor_rows.index.tolist()
    clusters    = []
    i = 0
    while i < len(floor_dates):
        cluster_start = floor_dates[i]
        j = i + 1
        while j < len(floor_dates) and (floor_dates[j] - cluster_start) <= 20:
            j += 1
        clusters.append(cluster_start)
        i = j

    if len(clusters) < MIN_OCC: return None

    # ── Analyse each floor cluster ────────────────────────────────────────────
    events = []
    for idx in clusters:
        row = g.loc[idx]
        evt = {
            "date":          str(row["date"].date()),
            "close":         round(float(row["c"]), 2),
            "low":           round(float(row["l"]), 2),
            "year":          int(row["date"].year),
            "pos_in_range":  round(float(row["pos_in_range"]), 1),
            "vol":           int(row["v"]),
            "recovery": {}
        }
        for w in RECOVERY_WINDOWS:
            fv = row.get(f"fwd_{w}", float("nan"))
            if pd.notna(fv):
                ret_pct = round(float(fv) * 100, 2)
                evt["recovery"][str(w)] = {
                    "return_pct": ret_pct,
                    "positive":   ret_pct > 0,
                    "beats_15pct": ret_pct >= MIN_RECOVERY_PCT,
                }
            else:
                evt["recovery"][str(w)] = None   # no future data (last N days)
        events.append(evt)

    if not events: return None

    # ── Check pattern consistency ─────────────────────────────────────────────
    years_with_events = sorted(set(e["year"] for e in events))
    if len(years_with_events) < MIN_YEARS_PATTERN: return None
    # Year gap check
    for i in range(1, len(years_with_events)):
        if years_with_events[i] - years_with_events[i-1] > MAX_YR_GAP:
            return None

    # ── Compute recovery statistics ───────────────────────────────────────────
    win_stats = {}
    for w in RECOVERY_WINDOWS:
        rets = [e["recovery"].get(str(w)) for e in events
                if e["recovery"].get(str(w)) is not None]
        valid_rets = [r["return_pct"] for r in rets if r is not None]
        if not valid_rets: continue
        n_v     = len(valid_rets)
        pos     = sum(1 for r in valid_rets if r > 0)
        beats15 = sum(1 for r in valid_rets if r >= MIN_RECOVERY_PCT)
        win_stats[w] = {
            "n":               n_v,
            "win_rate":        round(pos/n_v*100, 1),
            "beats15_rate":    round(beats15/n_v*100, 1),  # % of times >= 15%
            "avg_return":      round(sum(valid_rets)/n_v, 2),
            "min_return":      round(min(valid_rets), 2),   # guaranteed floor
            "max_return":      round(max(valid_rets), 2),
            "med_return":      round(sorted(valid_rets)[n_v//2], 2),
        }

    # ── Must have at least one window where beats15_rate >= MIN_WIN_RATE ─────
    best_window     = None
    best_beats15    = 0.0
    best_min_return = -999.0

    for w, ws in win_stats.items():
        if ws["beats15_rate"] >= MIN_WIN_RATE:
            if ws["beats15_rate"] > best_beats15 or (
               ws["beats15_rate"] == best_beats15 and
               ws["min_return"] > best_min_return):
                best_window     = w
                best_beats15    = ws["beats15_rate"]
                best_min_return = ws["min_return"]

    if best_window is None: return None
    if best_min_return <= 0.0: return None   # only show positive recovery

    # ── Compute floor depth stats ─────────────────────────────────────────────
    all_pos = [e["pos_in_range"] for e in events]
    avg_floor_depth = round(sum(all_pos)/len(all_pos), 1)

    # ── Grade ─────────────────────────────────────────────────────────────────
    n_y  = len(years_with_events)
    n_oc = len(events)
    bw   = win_stats[best_window]

    if   n_y>=5 and n_oc>=8 and best_min_return>=20: grade = "A+"
    elif n_y>=4 and n_oc>=6 and best_min_return>=15: grade = "A"
    elif n_y>=3 and n_oc>=4:                          grade = "B"
    else:                                              grade = "C"

    # Score = years × min_return × log(occ+1)
    score = round(n_y * best_min_return * math.log(n_oc+1), 1)

    return {
        "sym":              sym,
        "grade":            grade,
        "score":            score,
        "n_years":          n_y,
        "years":            years_with_events,
        "n_events":         n_oc,
        "best_window_days": best_window,
        "best_beats15_rate":best_beats15,
        "best_min_return":  best_min_return,
        "avg_floor_depth":  avg_floor_depth,
        "win_stats":        {str(k):v for k,v in win_stats.items()},
        "events":           events,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ALERTS — last 20 trading days
# ─────────────────────────────────────────────────────────────────────────────
def generate_alerts(profiles, trading_days_list, df):
    recent_td  = set(trading_days_list[-ALERT_WINDOW:])
    latest_dt  = df["date"].max()
    alerts     = []

    for sym, prof in profiles.items():
        for evt in prof["events"]:
            if evt["date"] not in recent_td: continue
            sig_dt    = pd.Timestamp(evt["date"])
            sig_close = evt["close"]

            # Current price
            tr        = df[(df["sym"]==sym) & (df["date"]==latest_dt)]
            cur_price = float(tr["c"].iloc[0]) if not tr.empty else sig_close
            pct_since = round((cur_price - sig_close)/sig_close*100, 2)

            # Best window stats
            bw     = prof["best_window_days"]
            bwstat = prof["win_stats"].get(str(bw), {})
            avg_r  = bwstat.get("avg_return", 15)
            min_r  = bwstat.get("min_return", 15)

            # Targets based on ACTUAL historical recovery data
            t_conservative = round(sig_close * (1 + min_r/100), 2)
            t_realistic    = round(sig_close * (1 + avg_r/100), 2)

            is_today = sig_dt.date() == latest_dt.date()
            days_ago = int((latest_dt - sig_dt).days)

            alerts.append({
                "sym":               sym,
                "sig_date":          evt["date"],
                "is_today":          is_today,
                "days_ago":          days_ago,
                "grade":             prof["grade"],
                "score":             prof["score"],
                "sig_close":         sig_close,
                "cur_price":         round(cur_price, 2),
                "pct_since":         pct_since,
                "floor_depth_pct":   evt["pos_in_range"],
                "best_window_days":  bw,
                "beats15_rate":      prof["best_beats15_rate"],
                "min_return":        prof["best_min_return"],
                "target_conservative": t_conservative,
                "target_realistic":    t_realistic,
                "n_years":           prof["n_years"],
                "n_events":          prof["n_events"],
                "win_stats":         prof["win_stats"],
                "zone": ("TARGET MET" if pct_since >= avg_r else
                         "RUNNING"    if pct_since > 0      else
                         "BUY ZONE"),
            })

    # Sort: today first, then grade, then score
    grade_ord = {"A+":0,"A":1,"B":2,"C":3}
    alerts.sort(key=lambda a:(
        0 if a["is_today"] else 1,
        grade_ord.get(a["grade"],9),
        -a["score"]
    ))
    return alerts

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Heavyweight Dip-Recover Engine")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Manifest
    print("\n[1] Manifest...")
    if not MANIFEST.exists():
        print(f"  ERROR: {MANIFEST}"); sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    tds        = sorted(manifest.keys())
    latest_str = tds[-1]
    print(f"  {len(tds)} trading days [{tds[0]} -> {latest_str}]")

    # Load data
    print("\n[2] Loading equity data...")
    df = load_data(manifest)

    # Heavyweight filter
    print("\n[3] Heavyweight filter (4+ years, avg turnover >= Rs 5 Cr)...")
    df = filter_heavyweights(df)
    sym_list = sorted(df["sym"].unique())
    print(f"  {len(sym_list)} heavyweight stocks")

    # Analyse each stock
    print("\n[4] Floor detection + recovery analysis...")
    sym_grps = {s:g.copy() for s,g in df.groupby("sym")}
    profiles = {}
    for i, sym in enumerate(sym_list):
        try:
            result = analyse_stock(sym_grps[sym], sym)
            if result:
                profiles[sym] = result
        except Exception as e:
            pass
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(sym_list)} analysed, {len(profiles)} qualifying...")
    del sym_grps; gc.collect()

    # Sort profiles by score
    sorted_profiles = dict(sorted(
        profiles.items(),
        key=lambda x: -x[1]["score"]
    ))

    naplus = sum(1 for p in profiles.values() if p["grade"]=="A+")
    na     = sum(1 for p in profiles.values() if p["grade"]=="A")
    nb     = sum(1 for p in profiles.values() if p["grade"]=="B")
    print(f"\n  {len(profiles)} qualifying stocks | A+:{naplus}  A:{na}  B:{nb}")
    print("  Top 10:")
    for sym, p in list(sorted_profiles.items())[:10]:
        print(f"    {sym:<14} {p['grade']}  "
              f"Best:{p['best_window_days']}d  "
              f"MinRet:{p['best_min_return']:+.1f}%  "
              f"Beats15:{p['best_beats15_rate']:.0f}%  "
              f"Occ:{p['n_events']}  Yrs:{p['n_years']}")

    # Alerts
    print("\n[5] Generating alerts (last 20 trading days)...")
    alerts    = generate_alerts(sorted_profiles, tds, df)
    today_cnt = sum(1 for a in alerts if a["is_today"])
    print(f"  {len(alerts)} alerts | Today: {today_cnt}")

    # Write outputs
    print("\n[6] Writing outputs...")
    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    jdump({
        "generated_at":     now_ist,
        "data_range":       f"{tds[0]} to {latest_str}",
        "thresholds": {
            "min_years_traded":     MIN_YEARS_TRADED,
            "min_turnover_cr":      MIN_DAILY_TURNOVER_CR,
            "floor_threshold_pct":  FLOOR_PCT_THRESHOLD,
            "min_recovery_pct":     MIN_RECOVERY_PCT,
            "min_win_rate_pct":     MIN_WIN_RATE,
        },
        "n_stocks":     len(sorted_profiles),
        "n_aplus":      naplus,
        "n_a":          na,
        "n_b":          nb,
        "profiles":     sorted_profiles,
    }, OUT_DIR / "heavyweight_profiles.json")
    print(f"  OK heavyweight_profiles.json ({len(profiles)} stocks)")

    jdump({
        "generated_at": now_ist,
        "latest_date":  latest_str,
        "alert_date":   latest_str,
        "window_days":  ALERT_WINDOW,
        "total_alerts": len(alerts),
        "today_count":  today_cnt,
        "grade_a":      sum(1 for a in alerts if a.get("grade","") in ("A+","A")),
        "alerts":       alerts,
    }, OUT_DIR / "heavyweight_alerts.json")
    print(f"  OK heavyweight_alerts.json ({len(alerts)} alerts)")

    print("\nDone.")

if __name__ == "__main__":
    main()
