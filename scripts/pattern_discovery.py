#!/usr/bin/env python3
"""
pattern_discovery.py  v1
=========================
Reads ALL NSE equity bhav copy CSVs from the data repo.
Discovers patterns with 100% (and highest) historical win rates.
Handles both old and new NSE CSV formats.

Outputs (pattern_signals/ — raw data/ never touched):
  patterns.json       — all signal patterns ranked by win_rate x avg_return
  alerts.json         — today's active signals with buy/sell zones
  stock_profiles.json — per-stock 100%-win-rate patterns

Run after daily NSE fetch (scheduled 11:45 PM IST).
"""

import json, os, sys, gc, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required. Run: pip install pandas numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "pattern_signals"
MANIFEST   = DATA_DIR / "manifest.json"

MIN_TRADING_DAYS  = 250
MIN_OCCURRENCES   = 5
CROSS_YEAR_FILTER = 2
FWD_WINDOWS       = [5, 10, 20]

# ---------------------------------------------------------------------------
# NSE CSV column aliases — old + new format
# ---------------------------------------------------------------------------
SYMBOL_ALIASES = ["SYMBOL",    "TCKRSYMB"]
SERIES_ALIASES = ["SERIES",    "SCTYSRS"]
OPEN_ALIASES   = ["OPEN",      "OPNPRIC"]
HIGH_ALIASES   = ["HIGH",      "HGHPRIC"]
LOW_ALIASES    = ["LOW",       "LWPRIC"]
CLOSE_ALIASES  = ["CLOSE",     "CLSPRIC",  "CLOSE PRICE", "CLOSE_PRICE", "LASTPRIC"]
VOLUME_ALIASES = ["TOTTRDQTY", "TTLTRADGVOL", "VOLUME"]

def find_col(hdr, aliases):
    for a in aliases:
        if a in hdr:
            return hdr.index(a)
    return -1

def parse_hdr(raw):
    return [h.strip().strip('"').strip("'").upper() for h in raw.split(",")]

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return rows
        hdr = parse_hdr(lines[0])
        i_sym = find_col(hdr, SYMBOL_ALIASES)
        i_ser = find_col(hdr, SERIES_ALIASES)
        i_o   = find_col(hdr, OPEN_ALIASES)
        i_h   = find_col(hdr, HIGH_ALIASES)
        i_l   = find_col(hdr, LOW_ALIASES)
        i_c   = find_col(hdr, CLOSE_ALIASES)
        i_v   = find_col(hdr, VOLUME_ALIASES)
        if i_sym < 0 or i_c < 0:
            return rows
        max_col = max(x for x in [i_sym, i_o, i_h, i_l, i_c, i_v] if x >= 0)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= max_col:
                continue
            series = cols[i_ser].strip() if i_ser >= 0 else "EQ"
            if series not in ("EQ", "BE"):
                continue
            try:
                sym = cols[i_sym].strip()
                c   = float(cols[i_c])
                o   = float(cols[i_o])   if i_o >= 0 else c
                h   = float(cols[i_h])   if i_h >= 0 else c
                l   = float(cols[i_l])   if i_l >= 0 else c
                v   = float(cols[i_v].replace(",","")) if i_v >= 0 else 0.0
                if c > 0 and sym:
                    rows.append({"sym": sym, "o": o, "h": h, "l": l, "c": c, "v": v})
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    return rows

# ---------------------------------------------------------------------------
# Load all CSVs
# ---------------------------------------------------------------------------
def load_all_data(manifest):
    trading_days = sorted(manifest.keys())
    print(f"  Loading {len(trading_days)} trading days...")
    all_rows = []
    loaded   = 0
    for date_str in trading_days:
        y, m, _ = date_str.split("-")
        path = DATA_DIR / "equity" / y / m / f"{date_str}.csv"
        if not path.exists():
            continue
        rows = load_csv(path)
        for r in rows:
            r["date"] = date_str
        all_rows.extend(rows)
        loaded += 1
        if loaded % 200 == 0:
            print(f"    {loaded}/{len(trading_days)} files, {len(all_rows):,} rows...")
    print(f"  Total: {len(all_rows):,} rows from {loaded} files")
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sym", "date"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Compute indicators per stock
# ---------------------------------------------------------------------------
def compute_indicators(g):
    g = g.copy().reset_index(drop=True)

    # Candle
    g["body"]       = (g["c"] - g["o"]).abs()
    g["range"]      = g["h"] - g["l"]
    g["upper_wick"] = g["h"] - g[["o","c"]].max(axis=1)
    g["lower_wick"] = g[["o","c"]].min(axis=1) - g["l"]
    g["close_pos"]  = np.where(g["range"] > 0, (g["c"] - g["l"]) / g["range"], 0.5)
    g["green"]      = (g["c"] >= g["o"]).astype(int)

    # Volume
    g["vol20"]     = g["v"].rolling(20, min_periods=5).mean()
    g["vol_ratio"] = np.where(g["vol20"] > 0, g["v"] / g["vol20"], 0.0)
    g["vol252max"] = g["v"].rolling(252, min_periods=60).max().shift(1)

    # Range
    g["range20"] = g["range"].rolling(20, min_periods=5).mean()
    g["range7min"]= g["range"].rolling(7, min_periods=7).min()
    g["range4min"]= g["range"].rolling(4, min_periods=4).min()

    # MAs
    g["ma20"]  = g["c"].rolling(20,  min_periods=10).mean()
    g["ma50"]  = g["c"].rolling(50,  min_periods=25).mean()
    g["ma200"] = g["c"].rolling(200, min_periods=100).mean()
    g["above_ma200"] = (g["c"] > g["ma200"]).astype(int)
    g["above_ma50"]  = (g["c"] > g["ma50"]).astype(int)
    g["pct_below_200"] = np.where(g["ma200"] > 0, (g["c"] / g["ma200"] - 1) * 100, 0)

    # ATR
    prev_c = g["c"].shift(1)
    tr = pd.concat([g["h"]-g["l"], (g["h"]-prev_c).abs(), (g["l"]-prev_c).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=7).mean()

    # Highs/Lows
    g["high20"]  = g["h"].rolling(20,  min_periods=10).max().shift(1)
    g["high52w"] = g["h"].rolling(252, min_periods=60).max().shift(1)

    # Prev bar
    g["prev_c"] = g["c"].shift(1)
    g["prev_h"] = g["h"].shift(1)
    g["prev_l"] = g["l"].shift(1)
    g["prev_o"] = g["o"].shift(1)
    g["prev_green"] = g["green"].shift(1)
    g["prev_body"]  = g["body"].shift(1)

    # Consecutive red/green
    consec_red = consec_green = 0
    cr_list = []
    cg_list = []
    for green_val in g["green"]:
        if green_val == 0:
            consec_red += 1; consec_green = 0
        else:
            consec_green += 1; consec_red = 0
        cr_list.append(consec_red)
        cg_list.append(consec_green)
    g["consec_red"]   = cr_list
    g["consec_green"] = cg_list

    # MA cross
    g["ma50_above_200"] = (g["ma50"] > g["ma200"]).astype(int)
    g["golden_cross"]   = ((g["ma50_above_200"] == 1) & (g["ma50_above_200"].shift(1) == 0)).astype(int)
    g["below_ma50_10d"] = g["above_ma50"].rolling(10, min_periods=10).sum().shift(1)

    # Vol dry-up windows
    v3_flag         = (g["vol_ratio"] <= 0.3).astype(int)
    g["vol_dryup_3d"] = v3_flag.rolling(3, min_periods=3).sum() >= 2
    g["vol_dryup_5d"] = v3_flag.rolling(5, min_periods=3).sum() >= 2

    # Forward returns
    for w in FWD_WINDOWS:
        g[f"fwd_{w}"] = g["c"].shift(-w) / g["c"] - 1

    g["year"] = g["date"].dt.year
    return g

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
ALL_SIGNALS = {
    "V1":  "Bull Volume Spike 5x",
    "V2":  "Bull Volume Spike 10x",
    "V3":  "Volume Dry-Up (<0.3x avg)",
    "V4":  "Annual Volume Climax",
    "C1":  "Hammer",
    "C2":  "Marubozu Bull",
    "C3":  "Bullish Engulfing",
    "C4":  "Bounce After 3+ Red Days",
    "C5":  "NR7 (Narrowest Range 7 Days)",
    "C6":  "Wide Range Green Day",
    "B1":  "20-Day High Breakout + Volume",
    "B2":  "52-Week High Breakout + Volume",
    "B3":  "Inside Bar",
    "T1":  "Crossed Above 50MA (was below 10d)",
    "T2":  "Golden Cross (50MA > 200MA)",
    "T3":  "Above 200MA + Volume Spike",
    "T4":  "Deep Oversold (>20% below 200MA) + Vol",
    "Q1":  "5+ Consecutive Red Days Entry",
    "Q2":  "7 Consecutive Green Days",
    "K1":  "COMBO: Vol 10x + Hammer",
    "K2":  "COMBO: Vol Dry-Up then 20D Breakout",
    "K3":  "COMBO: 5 Red Days + Vol Spike",
    "K4":  "COMBO: Deep Oversold + Hammer",
    "K5":  "COMBO: 52W High + Marubozu",
    "K6":  "COMBO: NR7 + Above 200MA + Vol",
    "K7":  "COMBO: Engulfing + 20D Breakout + Vol",
    "K8":  "COMBO: 5 Red Days + Hammer + Above 200MA",
    "K9":  "COMBO: Annual Vol Climax + Hammer + Below 200MA",
    "K10": "COMBO: Vol Dry-Up 3D + NR4 + Above 200MA",
}

def define_signals(df):
    d = df
    # Volume
    d["V1"] = (d["vol_ratio"] >= 5)  & (d["c"] > d["o"])
    d["V2"] = (d["vol_ratio"] >= 10) & (d["c"] > d["o"])
    d["V3"] = (d["vol_ratio"] <= 0.3)
    d["V4"] = (d["v"] >= d["vol252max"]) & (d["vol252max"] > 0)

    # Candle
    d["C1"] = ((d["lower_wick"] >= 2 * d["body"]) & (d["body"] > 0) &
               (d["upper_wick"] <= d["body"]) & (d["close_pos"] >= 0.7))
    d["C2"] = ((d["range"] > 0) & (d["body"] / d["range"].replace(0,1) >= 0.85) &
               (d["c"] > d["o"]) & (d["close_pos"] >= 0.90))
    d["C3"] = ((d["c"] > d["o"]) & (d["prev_green"] == 0) &
               (d["o"] <= d["prev_c"]) & (d["c"] >= d["prev_o"]) &
               (d["body"] > d["prev_body"]))
    d["C4"] = (d["consec_red"].shift(1) >= 3) & (d["c"] > d["o"])
    d["C5"] = (d["range"] > 0) & (d["range"] <= d["range7min"])
    d["C6"] = (d["range"] >= 1.5 * d["range20"].replace(0,1)) & (d["c"] > d["o"]) & (d["close_pos"] >= 0.7)

    # Breakout
    d["B1"] = ((d["h"] > d["high20"]) & (d["high20"] > 0) &
               (d["vol_ratio"] >= 1.5) & (d["c"] > d["o"]))
    d["B2"] = ((d["h"] > d["high52w"]) & (d["high52w"] > 0) &
               (d["vol_ratio"] >= 1.5) & (d["c"] > d["o"]))
    d["B3"] = ((d["h"] < d["prev_h"]) & (d["l"] > d["prev_l"]) & (d["prev_h"] > 0))

    # Trend
    d["T1"] = ((d["above_ma50"] == 1) & (d["above_ma50"].shift(1) == 0) &
               (d["below_ma50_10d"] == 0))
    d["T2"] = (d["golden_cross"] == 1)
    d["T3"] = ((d["above_ma200"] == 1) & (d["vol_ratio"] >= 2) & (d["c"] > d["o"]))
    d["T4"] = ((d["pct_below_200"] <= -20) & (d["ma200"] > 0) &
               (d["vol_ratio"] >= 3) & (d["c"] > d["o"]))

    # Sequence
    d["Q1"] = (d["consec_red"].shift(1) >= 5)
    d["Q2"] = (d["consec_green"] >= 7)

    # Combos
    d["K1"]  = d["V2"] & d["C1"]
    d["K2"]  = d["vol_dryup_5d"].shift(1).fillna(False) & d["B1"]
    d["K3"]  = d["Q1"] & d["V2"]
    d["K4"]  = d["T4"] & d["C1"]
    d["K5"]  = d["B2"] & d["C2"]
    d["K6"]  = d["C5"] & (d["above_ma200"] == 1) & (d["vol_ratio"] >= 1.5)
    d["K7"]  = d["C3"] & d["B1"] & d["V1"]
    d["K8"]  = d["Q1"] & d["C1"] & (d["above_ma200"] == 1)
    d["K9"]  = d["V4"] & d["C1"] & (d["above_ma200"] == 0)
    d["K10"] = d["vol_dryup_3d"] & (d["range"] <= d["range4min"].replace(0,1)) & (d["above_ma200"] == 1)
    return d

# ---------------------------------------------------------------------------
# Win rate analysis
# ---------------------------------------------------------------------------
def analyse_signal(df, sig, name):
    mask = df[sig].fillna(False)
    hits = df[mask].copy()
    n    = len(hits)
    if n < MIN_OCCURRENCES:
        return None
    years_active = sorted(hits["year"].unique())
    if len(years_active) < CROSS_YEAR_FILTER:
        return None

    result = {"signal": sig, "name": name, "occurrences": n,
              "years": [int(y) for y in years_active]}
    best_wr = 0
    for w in FWD_WINDOWS:
        col   = f"fwd_{w}"
        valid = hits[col].dropna()
        if len(valid) < MIN_OCCURRENCES:
            result[f"wr_{w}d"] = result[f"avg_{w}d"] = result[f"med_{w}d"] = None
            result[f"n_{w}d"] = 0
            continue
        wins = (valid > 0).sum()
        wr   = round(wins / len(valid) * 100, 1)
        result[f"wr_{w}d"]  = wr
        result[f"avg_{w}d"] = round(valid.mean() * 100, 2)
        result[f"med_{w}d"] = round(valid.median() * 100, 2)
        result[f"n_{w}d"]   = len(valid)
        best_wr = max(best_wr, wr)

    result["best_win_rate"] = best_wr
    best_avg = max((result.get(f"avg_{w}d") or 0) for w in FWD_WINDOWS)
    result["score"] = round(best_wr * best_avg, 1)

    # Grade based on 20d window
    wr20 = result.get("wr_20d") or 0
    n20  = result.get("n_20d")  or 0
    result["grade"] = ("A" if wr20 == 100 and n20 >= 10 else
                       "B" if wr20 == 100 and n20 >= 5  else
                       "C" if wr20 >= 90  and n20 >= 20 else
                       "D" if wr20 >= 80  and n20 >= 30 else "—")

    # Year-wise
    ywise = {}
    for y in years_active:
        yr = hits[hits["year"] == y]["fwd_20"].dropna()
        if len(yr):
            ywise[str(int(y))] = {"occ": len(yr), "wr": round((yr>0).sum()/len(yr)*100,1),
                                   "avg": round(yr.mean()*100, 2)}
    result["yearly"] = ywise

    # Top 5 stocks that contributed most to this signal
    if "sym" in hits.columns:
        top_syms = (hits["sym"].value_counts().head(5).to_dict())
        result["top_stocks"] = top_syms
    return result

# ---------------------------------------------------------------------------
# Stock-specific mining
# ---------------------------------------------------------------------------
def mine_stock(g, sym):
    found = []
    for sig, name in ALL_SIGNALS.items():
        if sig not in g.columns:
            continue
        mask  = g[sig].fillna(False)
        hits  = g[mask]
        if len(hits) < 3:
            continue
        if hits["year"].nunique() < 2:
            continue
        for w in FWD_WINDOWS:
            col   = f"fwd_{w}"
            valid = hits[col].dropna()
            if len(valid) < 3:
                continue
            wr = (valid > 0).sum() / len(valid) * 100
            if wr < 80:
                continue
            avg_r = round(valid.mean() * 100, 2)
            med_r = round(valid.median() * 100, 2)
            found.append({
                "sym": sym, "signal": sig, "name": name,
                "hold_days": w, "occ": int(len(valid)),
                "win_rate": round(wr, 1), "avg_ret": avg_r,
                "med_ret": med_r,
                "max_ret": round(valid.max() * 100, 2),
                "min_ret": round(valid.min() * 100, 2),
                "years":   [int(y) for y in sorted(hits["year"].unique())],
                "grade": ("A" if wr == 100 and len(valid) >= 5 else
                          "B" if wr == 100 else
                          "C" if wr >= 90  else "D"),
            })
    # deduplicate: keep best hold_days per signal
    found.sort(key=lambda x: (-x["win_rate"], -x["avg_ret"]))
    seen = set()
    unique = []
    for f in found:
        k = f["signal"]
        if k not in seen:
            seen.add(k)
            unique.append(f)
    return unique

# ---------------------------------------------------------------------------
# Generate today's alerts
# ---------------------------------------------------------------------------
def generate_alerts(df, patterns):
    latest_date = df["date"].max()
    today       = df[df["date"] == latest_date].copy()
    if today.empty:
        return []
    pat_map = {p["signal"]: p for p in patterns}
    alerts  = []
    for sig in ALL_SIGNALS:
        if sig not in today.columns:
            continue
        active = today[today[sig].fillna(False)]
        if active.empty:
            continue
        pat = pat_map.get(sig)
        if not pat:
            continue
        wr20  = pat.get("wr_20d") or 0
        avg20 = pat.get("avg_20d") or 0
        if wr20 < 60:
            continue
        for _, row in active.iterrows():
            sym   = row["sym"]
            close = round(row["c"], 2)
            atr   = row["atr14"] if not pd.isna(row.get("atr14", float("nan"))) else close * 0.02
            alerts.append({
                "sym":           sym,
                "date":          str(latest_date.date()),
                "signal":        sig,
                "signal_name":   ALL_SIGNALS[sig],
                "close":         close,
                "buy_zone_lo":   round(close - 0.5 * atr, 2),
                "buy_zone_hi":   round(close + 0.3 * atr, 2),
                "stop_loss":     round(close - 1.5 * atr, 2),
                "target_1":      round(close + 1.5 * atr, 2),
                "target_2":      round(close * (1 + max(avg20, 5) / 100), 2),
                "hold_days":     20,
                "win_rate_5d":   pat.get("wr_5d"),
                "win_rate_10d":  pat.get("wr_10d"),
                "win_rate_20d":  wr20,
                "avg_ret_20d":   avg20,
                "grade":         pat.get("grade", "—"),
                "score":         pat.get("score", 0),
                "vol_ratio":     round(float(row.get("vol_ratio", 0)), 1),
                "above_200ma":   bool(row.get("above_ma200", 0)),
            })
    alerts.sort(key=lambda a: (
        {"A":0,"B":1,"C":2,"D":3,"—":4}.get(a["grade"],4),
        -(a["win_rate_20d"] or 0) * (a["avg_ret_20d"] or 1)
    ))
    return alerts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("NSE Pattern Discovery Engine  v1")
    print("=" * 65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Manifest...")
    if not MANIFEST.exists():
        print(f"  ERROR: {MANIFEST} not found"); sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} trading days")

    print("\n[2] Loading all equity CSVs...")
    df = load_all_data(manifest)
    print(f"  {len(df):,} rows, {df['sym'].nunique():,} symbols")

    # Filter stocks with enough history
    sym_counts = df.groupby("sym")["date"].count()
    valid_syms = sym_counts[sym_counts >= MIN_TRADING_DAYS].index
    df = df[df["sym"].isin(valid_syms)].copy()
    print(f"  {len(valid_syms)} stocks have >= {MIN_TRADING_DAYS} trading days")

    print("\n[3] Computing indicators per stock...")
    groups = []
    syms   = sorted(df["sym"].unique())
    for i, sym in enumerate(syms):
        try:
            g = df[df["sym"] == sym].copy()
            groups.append(compute_indicators(g))
        except Exception as e:
            pass
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(syms)}...")
    df = pd.concat(groups, ignore_index=True)
    del groups; gc.collect()
    print(f"  Done — {len(df):,} rows with indicators")

    print("\n[4] Defining signals...")
    df = define_signals(df)
    for sig in list(ALL_SIGNALS)[:5]:
        cnt = int(df[sig].fillna(False).sum())
        print(f"    {sig}: {cnt:,} hits")

    print("\n[5] Cross-stock pattern analysis...")
    patterns = []
    for sig, name in ALL_SIGNALS.items():
        try:
            r = analyse_signal(df, sig, name)
            if r:
                patterns.append(r)
        except Exception as e:
            print(f"    WARN {sig}: {e}")
    patterns.sort(key=lambda p: -(p.get("score") or 0))
    grade_counts = {}
    for p in patterns:
        g = p.get("grade","—"); grade_counts[g] = grade_counts.get(g,0) + 1
    print(f"  {len(patterns)} patterns | Grades: {grade_counts}")
    print(f"  Top 5 by score:")
    for p in patterns[:5]:
        print(f"    {p['signal']:<5} {p['name'][:38]:<38}  "
              f"WR20:{p.get('wr_20d','—')}%  Avg20:{p.get('avg_20d','—')}%  Grade:{p['grade']}")

    print("\n[6] Stock-specific 100% pattern mining...")
    stock_profiles = {}
    all_grade_a    = []
    for sym in syms:
        try:
            g = df[df["sym"] == sym]
            results = mine_stock(g, sym)
            if results:
                stock_profiles[sym] = results
                all_grade_a.extend([r for r in results if r["grade"] == "A"])
        except Exception:
            pass
    all_grade_a.sort(key=lambda x: (-x["avg_ret"], -x["win_rate"]))
    stocks_100pct = len([s for s in stock_profiles
                         if any(r["grade"]=="A" for r in stock_profiles[s])])
    print(f"  {stocks_100pct} stocks with Grade A (100%) patterns")
    print(f"  Top 5 Grade A:")
    for r in all_grade_a[:5]:
        print(f"    {r['sym']:<14} {r['signal']:<5} {r['hold_days']}d  "
              f"WR:{r['win_rate']}%  Avg:{r['avg_ret']:+.1f}%  Occ:{r['occ']}")

    print("\n[7] Generating today's alerts...")
    alerts    = generate_alerts(df, patterns)
    grade_a_n = len([a for a in alerts if a["grade"]=="A"])
    print(f"  {len(alerts)} alerts | Grade A: {grade_a_n}")

    # Write outputs
    print("\n[8] Writing JSON files...")
    ist     = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    with open(OUT_DIR / "patterns.json", "w") as f:
        json.dump({
            "generated_at":     now_ist,
            "stocks_analyzed":  int(len(syms)),
            "trading_days":     len(manifest),
            "data_range":       f"{sorted(manifest)[ 0]} to {sorted(manifest)[-1]}",
            "grade_counts":     grade_counts,
            "patterns":         patterns,
        }, f, indent=2)
    print(f"  OK patterns.json ({len(patterns)} patterns)")

    with open(OUT_DIR / "alerts.json", "w") as f:
        json.dump({
            "generated_at": now_ist,
            "alert_date":   str(df["date"].max().date()),
            "total_alerts": len(alerts),
            "grade_a":      grade_a_n,
            "alerts":       alerts,
        }, f, indent=2)
    print(f"  OK alerts.json ({len(alerts)} alerts)")

    # Only keep Grade A/B/C in profiles to keep file manageable
    filtered = {sym: [r for r in v if r["grade"] in ("A","B","C")]
                for sym, v in stock_profiles.items()
                if any(r["grade"] in ("A","B") for r in v)}
    with open(OUT_DIR / "stock_profiles.json", "w") as f:
        json.dump({
            "generated_at":      now_ist,
            "stocks_with_100pct":stocks_100pct,
            "all_grade_a":       all_grade_a[:300],
            "profiles":          filtered,
        }, f, indent=2)
    print(f"  OK stock_profiles.json ({len(filtered)} stocks, {len(all_grade_a)} Grade A patterns)")
    print(f"\nDone.")

if __name__ == "__main__":
    main()
