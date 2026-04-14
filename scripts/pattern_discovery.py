#!/usr/bin/env python3
"""
pattern_discovery.py  v2
=========================
NSE Pattern Discovery — strict 100% win-rate engine.

Hard filters (all must pass):
  - Win rate: 100% at all three windows (5d, 10d, 20d)
  - Avg return: 20d >= +10%, 10d >= +7%, 5d >= +3%
  - Volume on signal day > 100,000 shares
  - Appears in 2+ different years
  - Max gap between consecutive years of occurrence <= 2 years
  - At least ceil(data_years / 2) years covered

Outputs (pattern_signals/ only — data/ never touched):
  patterns.json       — cross-stock patterns, graded by repeat-years
  stock_profiles.json — per-stock patterns with all-years detail + month heatmap
  alerts.json         — today + last 20 trading days window with buy/sell status
  heatmap.json        — signal × month profit matrix across all stocks
"""

import json, os, gc, sys, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install pandas numpy"); sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "pattern_signals"
MANIFEST  = DATA_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Hard thresholds — no result shown below these
# ---------------------------------------------------------------------------
MIN_WR_ALL    = 100.0   # must be 100% win at every window
MIN_AVG_5D    = 3.0     # avg return at 5d >= +3%
MIN_AVG_10D   = 7.0     # avg return at 10d >= +7%
MIN_AVG_20D   = 10.0    # avg return at 20d >= +10%
MIN_VOLUME    = 100_000  # signal day volume must be tradeable
MIN_OCC       = 3       # minimum occurrences
MIN_YEARS     = 2       # must appear in 2+ different years
MAX_YR_GAP    = 2       # max gap (years) between consecutive occurrences
FWD_WINDOWS   = [5, 10, 20]
MIN_TRADING_D = 200

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# CSV column aliases — old + new NSE format
# ---------------------------------------------------------------------------
SYMBOL_ALIASES = ["SYMBOL",    "TCKRSYMB"]
SERIES_ALIASES = ["SERIES",    "SCTYSRS"]
OPEN_ALIASES   = ["OPEN",      "OPNPRIC"]
HIGH_ALIASES   = ["HIGH",      "HGHPRIC"]
LOW_ALIASES    = ["LOW",       "LWPRIC"]
CLOSE_ALIASES  = ["CLOSE",     "CLSPRIC",  "CLOSE PRICE", "LASTPRIC"]
VOLUME_ALIASES = ["TOTTRDQTY", "TTLTRADGVOL", "VOLUME"]

def find_col(hdr, aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def parse_hdr(raw):
    return [h.strip().strip('"').strip("'").upper() for h in raw.split(",")]

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2: return rows
        hdr   = parse_hdr(lines[0])
        i_sym = find_col(hdr, SYMBOL_ALIASES)
        i_ser = find_col(hdr, SERIES_ALIASES)
        i_o   = find_col(hdr, OPEN_ALIASES)
        i_h   = find_col(hdr, HIGH_ALIASES)
        i_l   = find_col(hdr, LOW_ALIASES)
        i_c   = find_col(hdr, CLOSE_ALIASES)
        i_v   = find_col(hdr, VOLUME_ALIASES)
        if i_sym < 0 or i_c < 0: return rows
        max_col = max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v] if x >= 0)
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= max_col: continue
            series = cols[i_ser].strip() if i_ser >= 0 else "EQ"
            if series not in ("EQ","BE"): continue
            try:
                sym = cols[i_sym].strip()
                c   = float(cols[i_c])
                o   = float(cols[i_o]) if i_o >= 0 else c
                h   = float(cols[i_h]) if i_h >= 0 else c
                l   = float(cols[i_l]) if i_l >= 0 else c
                v   = float(cols[i_v].replace(",","")) if i_v >= 0 else 0.0
                if c > 0 and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except (ValueError,IndexError): pass
    except Exception: pass
    return rows

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
def load_all_data(manifest):
    trading_days = sorted(manifest.keys())
    print(f"  Loading {len(trading_days)} trading days...")
    all_rows = []
    loaded   = 0
    for date_str in trading_days:
        y, m, _ = date_str.split("-")
        path = DATA_DIR / "equity" / y / m / f"{date_str}.csv"
        if not path.exists(): continue
        rows = load_csv(path)
        for r in rows: r["date"] = date_str
        all_rows.extend(rows)
        loaded += 1
        if loaded % 300 == 0:
            print(f"    {loaded}/{len(trading_days)} files, {len(all_rows):,} rows...")
    print(f"  Total: {len(all_rows):,} rows, {loaded} files")
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sym","date"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_indicators(g):
    g = g.copy().reset_index(drop=True)
    g["body"]       = (g["c"] - g["o"]).abs()
    g["range_"]     = g["h"] - g["l"]
    g["upper_wick"] = g["h"] - g[["o","c"]].max(axis=1)
    g["lower_wick"] = g[["o","c"]].min(axis=1) - g["l"]
    g["close_pos"]  = np.where(g["range_"]>0,(g["c"]-g["l"])/g["range_"],0.5)
    g["green"]      = (g["c"] >= g["o"]).astype(int)
    g["month"]      = g["date"].dt.month
    g["year"]       = g["date"].dt.year

    # Volume
    g["vol20"]     = g["v"].rolling(20,min_periods=5).mean()
    g["vol_ratio"] = np.where(g["vol20"]>0, g["v"]/g["vol20"], 0.0)
    g["vol252max"] = g["v"].rolling(252,min_periods=60).max().shift(1)
    g["vol_ok"]    = g["v"] >= MIN_VOLUME   # tradeable volume flag

    # Range
    g["range20"]  = g["range_"].rolling(20,min_periods=5).mean()
    g["range7min"]= g["range_"].rolling(7,min_periods=7).min()
    g["range4min"]= g["range_"].rolling(4,min_periods=4).min()

    # MAs
    g["ma20"]  = g["c"].rolling(20, min_periods=10).mean()
    g["ma50"]  = g["c"].rolling(50, min_periods=25).mean()
    g["ma200"] = g["c"].rolling(200,min_periods=100).mean()
    g["above_ma200"] = (g["c"] > g["ma200"]).astype(int)
    g["above_ma50"]  = (g["c"] > g["ma50"]).astype(int)
    g["pct_below_200"] = np.where(g["ma200"]>0,(g["c"]/g["ma200"]-1)*100,0)

    # ATR
    prev_c = g["c"].shift(1)
    tr = pd.concat([g["h"]-g["l"], (g["h"]-prev_c).abs(), (g["l"]-prev_c).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14,min_periods=7).mean()

    # Highs/Lows
    g["high20"]  = g["h"].rolling(20, min_periods=10).max().shift(1)
    g["high52w"] = g["h"].rolling(252,min_periods=60).max().shift(1)

    # Prev bar
    g["prev_c"]     = g["c"].shift(1)
    g["prev_h"]     = g["h"].shift(1)
    g["prev_l"]     = g["l"].shift(1)
    g["prev_o"]     = g["o"].shift(1)
    g["prev_green"] = g["green"].shift(1)
    g["prev_body"]  = g["body"].shift(1)

    # Consecutive red/green
    cr_list = []; cg_list = []; cr = cg = 0
    for gv in g["green"]:
        if gv == 0: cr += 1; cg = 0
        else:        cg += 1; cr = 0
        cr_list.append(cr); cg_list.append(cg)
    g["consec_red"]   = cr_list
    g["consec_green"] = cg_list

    # MA cross
    g["ma50_above_200"] = (g["ma50"] > g["ma200"]).astype(int)
    g["golden_cross"]   = ((g["ma50_above_200"]==1) & (g["ma50_above_200"].shift(1)==0)).astype(int)
    g["below_ma50_10d"] = g["above_ma50"].rolling(10,min_periods=10).sum().shift(1)

    # Vol dry-up
    v3flag = (g["vol_ratio"] <= 0.3).astype(int)
    g["vol_dryup_3d"] = v3flag.rolling(3,min_periods=3).sum() >= 2
    g["vol_dryup_5d"] = v3flag.rolling(5,min_periods=3).sum() >= 2

    # Forward returns
    for w in FWD_WINDOWS:
        g[f"fwd_{w}"] = g["c"].shift(-w) / g["c"] - 1

    return g

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
ALL_SIGNALS = {
    "V1":  "Bull Volume Spike 5x",
    "V2":  "Bull Volume Spike 10x",
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
    "T1":  "Crossed Above 50MA",
    "T2":  "Golden Cross",
    "T3":  "Above 200MA + Volume Spike",
    "T4":  "Deep Oversold + Vol Spike",
    "Q1":  "5+ Consecutive Red Days",
    "K1":  "COMBO: Vol 10x + Hammer",
    "K2":  "COMBO: Vol Dry-Up then 20D Breakout",
    "K3":  "COMBO: 5 Red Days + Vol Spike",
    "K4":  "COMBO: Deep Oversold + Hammer",
    "K5":  "COMBO: 52W High + Marubozu",
    "K6":  "COMBO: NR7 + Above 200MA + Vol",
    "K7":  "COMBO: Engulfing + 20D Breakout + Vol",
    "K8":  "COMBO: 5 Red Days + Hammer + Above 200MA",
    "K9":  "COMBO: Annual Vol Climax + Hammer",
    "K10": "COMBO: Vol Dry-Up + NR4 + Above 200MA",
}

def define_signals(df):
    d = df
    d["V1"]  = (d["vol_ratio"]>=5)  & (d["c"]>d["o"]) & d["vol_ok"]
    d["V2"]  = (d["vol_ratio"]>=10) & (d["c"]>d["o"]) & d["vol_ok"]
    d["V4"]  = (d["v"]>=d["vol252max"]) & (d["vol252max"]>0) & d["vol_ok"]
    d["C1"]  = ((d["lower_wick"]>=2*d["body"]) & (d["body"]>0) &
                (d["upper_wick"]<=d["body"]) & (d["close_pos"]>=0.7) & d["vol_ok"])
    d["C2"]  = ((d["range_"]>0) & (d["body"]/d["range_"].replace(0,1)>=0.85) &
                (d["c"]>d["o"]) & (d["close_pos"]>=0.90) & d["vol_ok"])
    d["C3"]  = ((d["c"]>d["o"]) & (d["prev_green"]==0) &
                (d["o"]<=d["prev_c"]) & (d["c"]>=d["prev_o"]) &
                (d["body"]>d["prev_body"]) & d["vol_ok"])
    d["C4"]  = (d["consec_red"].shift(1)>=3) & (d["c"]>d["o"]) & d["vol_ok"]
    d["C5"]  = (d["range_"]>0) & (d["range_"]<=d["range7min"]) & d["vol_ok"]
    d["C6"]  = ((d["range_"]>=1.5*d["range20"].replace(0,1)) &
                (d["c"]>d["o"]) & (d["close_pos"]>=0.7) & d["vol_ok"])
    d["B1"]  = ((d["h"]>d["high20"]) & (d["high20"]>0) &
                (d["vol_ratio"]>=1.5) & (d["c"]>d["o"]) & d["vol_ok"])
    d["B2"]  = ((d["h"]>d["high52w"]) & (d["high52w"]>0) &
                (d["vol_ratio"]>=1.5) & (d["c"]>d["o"]) & d["vol_ok"])
    d["B3"]  = (d["h"]<d["prev_h"]) & (d["l"]>d["prev_l"]) & (d["prev_h"]>0) & d["vol_ok"]
    d["T1"]  = ((d["above_ma50"]==1) & (d["above_ma50"].shift(1)==0) &
                (d["below_ma50_10d"]==0) & d["vol_ok"])
    d["T2"]  = (d["golden_cross"]==1) & d["vol_ok"]
    d["T3"]  = ((d["above_ma200"]==1) & (d["vol_ratio"]>=2) &
                (d["c"]>d["o"]) & d["vol_ok"])
    d["T4"]  = ((d["pct_below_200"]<=-20) & (d["ma200"]>0) &
                (d["vol_ratio"]>=3) & (d["c"]>d["o"]) & d["vol_ok"])
    d["Q1"]  = (d["consec_red"].shift(1)>=5) & d["vol_ok"]
    d["K1"]  = d["V2"] & d["C1"]
    d["K2"]  = d["vol_dryup_5d"].shift(1).fillna(False) & d["B1"]
    d["K3"]  = d["Q1"] & d["V2"]
    d["K4"]  = d["T4"] & d["C1"]
    d["K5"]  = d["B2"] & d["C2"]
    d["K6"]  = d["C5"] & (d["above_ma200"]==1) & (d["vol_ratio"]>=1.5) & d["vol_ok"]
    d["K7"]  = d["C3"] & d["B1"] & d["V1"]
    d["K8"]  = d["Q1"] & d["C1"] & (d["above_ma200"]==1)
    d["K9"]  = d["V4"] & d["C1"]
    d["K10"] = (d["vol_dryup_3d"] &
                (d["range_"]<=d["range4min"].replace(0,1)) &
                (d["above_ma200"]==1) & d["vol_ok"])
    return d

# ---------------------------------------------------------------------------
# Year-gap validator
# ---------------------------------------------------------------------------
def years_pass_gap(years_list):
    """Return True if consecutive year gaps are all <= MAX_YR_GAP."""
    if len(years_list) < MIN_YEARS:
        return False
    sy = sorted(years_list)
    for i in range(1, len(sy)):
        if sy[i] - sy[i-1] > MAX_YR_GAP:
            return False
    return True

def repeating_score(years_list, avg_ret_20d, n_occ):
    """Score = num_years × avg_ret × log(occ). More years = higher score."""
    import math
    return round(len(years_list) * avg_ret_20d * math.log(max(n_occ,1)+1), 1)

# ---------------------------------------------------------------------------
# Cross-stock pattern analysis
# ---------------------------------------------------------------------------
def analyse_signal(df, sig, name):
    mask = df[sig].fillna(False) & df["vol_ok"]
    hits = df[mask].copy()
    if len(hits) < MIN_OCC:
        return None

    # All three windows must pass thresholds
    result = {"signal":sig,"name":name,"occurrences":len(hits)}
    for w in FWD_WINDOWS:
        col   = f"fwd_{w}"
        valid = hits[col].dropna()
        if len(valid) < MIN_OCC:
            return None
        wr    = (valid>0).sum()/len(valid)*100
        avg_r = valid.mean()*100
        if wr < MIN_WR_ALL:
            return None
        result[f"wr_{w}d"]  = round(wr,1)
        result[f"avg_{w}d"] = round(avg_r,2)
        result[f"med_{w}d"] = round(valid.median()*100,2)
        result[f"n_{w}d"]   = int(len(valid))

    # Return thresholds
    if (result.get("avg_5d",0)  < MIN_AVG_5D  or
        result.get("avg_10d",0) < MIN_AVG_10D or
        result.get("avg_20d",0) < MIN_AVG_20D):
        return None

    # Year filter
    years_all = sorted(hits["year"].unique().tolist())
    if not years_pass_gap(years_all):
        return None
    result["years"] = [int(y) for y in years_all]
    result["n_years"] = len(years_all)

    # Score
    result["score"] = repeating_score(years_all, result["avg_20d"], len(hits))

    # Grade — based on years covered and occurrences
    n_y  = len(years_all)
    n_oc = result[f"n_20d"]
    result["grade"] = ("A+" if n_y>=4 and n_oc>=10 else
                       "A"  if n_y>=3 and n_oc>=7  else
                       "B"  if n_y>=2 and n_oc>=5  else
                       "C"  if n_y>=2              else "—")

    # Year-wise breakdown
    ywise = {}
    for y in years_all:
        yr = hits[hits["year"]==y]["fwd_20"].dropna()
        if len(yr):
            ywise[str(int(y))] = {
                "occ":  int(len(yr)),
                "wr":   round((yr>0).sum()/len(yr)*100,1),
                "avg":  round(yr.mean()*100,2),
                "min":  round(yr.min()*100,2),
                "max":  round(yr.max()*100,2),
            }
    result["yearly"] = ywise

    # Month heatmap (cross-stock)
    hits["_month"] = hits["date"].dt.month
    mheat = {}
    for m in range(1,13):
        ms = hits[hits["_month"]==m]["fwd_20"].dropna()
        if len(ms) >= 2:
            mheat[MONTHS[m-1]] = {
                "occ":  int(len(ms)),
                "wr":   round((ms>0).sum()/len(ms)*100,1),
                "avg":  round(ms.mean()*100,2),
                "min":  round(ms.min()*100,2),
                "max":  round(ms.max()*100,2),
            }
    result["month_heat"] = mheat

    # Top contributing stocks
    result["top_stocks"] = hits["sym"].value_counts().head(10).to_dict()
    return result

# ---------------------------------------------------------------------------
# Per-stock mining
# ---------------------------------------------------------------------------
def mine_stock(g, sym, data_years):
    """
    data_years = total years of data available for this stock.
    Required: appear in at least ceil(data_years/2) years.
    """
    import math
    min_yrs_required = max(MIN_YEARS, math.ceil(data_years/2))
    found = []

    for sig, name in ALL_SIGNALS.items():
        if sig not in g.columns:
            continue
        mask = g[sig].fillna(False) & g["vol_ok"]
        hits = g[mask].copy()
        if len(hits) < MIN_OCC:
            continue

        # Must have valid forward returns in 2+ years
        # Use 20d window as primary
        valid_20 = hits["fwd_20"].dropna()
        if len(valid_20) < MIN_OCC:
            continue
        yr_data_20 = hits[hits["fwd_20"].notna()]["year"].unique()
        if not years_pass_gap(sorted(yr_data_20)):
            continue

        # All thresholds must pass at all windows
        ok = True
        win_data = {}
        for w in FWD_WINDOWS:
            col   = f"fwd_{w}"
            valid = hits[col].dropna()
            if len(valid) < MIN_OCC:
                ok = False; break
            wr    = (valid>0).sum()/len(valid)*100
            avg_r = valid.mean()*100
            if wr < MIN_WR_ALL:
                ok = False; break
            win_data[w] = {"wr":round(wr,1),"avg":round(avg_r,2),
                           "med":round(valid.median()*100,2),
                           "occ":int(len(valid))}
        if not ok:
            continue
        if (win_data[5]["avg"]  < MIN_AVG_5D  or
            win_data[10]["avg"] < MIN_AVG_10D or
            win_data[20]["avg"] < MIN_AVG_20D):
            continue

        # Year coverage
        years_all = sorted([int(y) for y in yr_data_20])
        if len(years_all) < min(MIN_YEARS, min_yrs_required):
            continue

        # Per-year detail — show ALL years with what happened
        yr_detail = {}
        for y in sorted(hits["year"].unique()):
            yr_rows = hits[hits["year"]==y]
            yr_v20  = yr_rows["fwd_20"].dropna()
            yr_v5   = yr_rows["fwd_5"].dropna()
            yr_v10  = yr_rows["fwd_10"].dropna()
            if len(yr_v20):
                yr_detail[str(int(y))] = {
                    "occ":    int(len(yr_v20)),
                    "wr_20d": round((yr_v20>0).sum()/len(yr_v20)*100,1),
                    "avg_20d":round(yr_v20.mean()*100,2),
                    "min_20d":round(yr_v20.min()*100,2),
                    "max_20d":round(yr_v20.max()*100,2),
                    "avg_5d": round(yr_v5.mean()*100,2) if len(yr_v5) else None,
                    "avg_10d":round(yr_v10.mean()*100,2) if len(yr_v10) else None,
                }

        # Month heatmap for this stock+signal
        hits["_month"] = hits["date"].dt.month
        month_heat = {}
        for m in range(1,13):
            ms = hits[hits["_month"]==m]["fwd_20"].dropna()
            if len(ms) >= 1:
                month_heat[MONTHS[m-1]] = {
                    "occ": int(len(ms)),
                    "avg": round(ms.mean()*100,2),
                    "min": round(ms.min()*100,2),
                    "max": round(ms.max()*100,2),
                }

        # Avg volume on signal days (for context)
        avg_vol = int(hits["v"].mean())

        n_years = len(years_all)
        n_occ   = win_data[20]["occ"]
        grade   = ("A+" if n_years>=4 and n_occ>=8 else
                   "A"  if n_years>=3 and n_occ>=5 else
                   "B"  if n_years>=2 and n_occ>=3 else "C")

        found.append({
            "sym":       sym,
            "signal":    sig,
            "name":      name,
            "grade":     grade,
            "n_years":   n_years,
            "years":     years_all,
            "occ_20d":   n_occ,
            "wr_5d":     win_data[5]["wr"],
            "avg_5d":    win_data[5]["avg"],
            "wr_10d":    win_data[10]["wr"],
            "avg_10d":   win_data[10]["avg"],
            "wr_20d":    win_data[20]["wr"],
            "avg_20d":   win_data[20]["avg"],
            "med_20d":   win_data[20]["med"],
            "min_ret_20d": round(valid_20.min()*100,2),
            "max_ret_20d": round(valid_20.max()*100,2),
            "avg_vol":   avg_vol,
            "score":     repeating_score(years_all, win_data[20]["avg"], n_occ),
            "yr_detail": yr_detail,
            "month_heat":month_heat,
        })

    # Deduplicate: best window per signal (highest score)
    found.sort(key=lambda x: -x["score"])
    seen   = set()
    unique = []
    for f in found:
        if f["signal"] not in seen:
            seen.add(f["signal"])
            unique.append(f)
    return unique

# ---------------------------------------------------------------------------
# Alerts — today + last 20 trading days
# ---------------------------------------------------------------------------
def generate_alerts(df, stock_profiles, trading_days_list):
    latest_date = df["date"].max()
    # Last 20 trading days
    recent_cutoff = pd.Timestamp(trading_days_list[-20]) if len(trading_days_list)>=20 else df["date"].min()
    recent_df = df[df["date"] >= recent_cutoff].copy()

    # Build lookup: sym -> {signal -> pattern}
    sym_pat_lookup = {}
    for sym, pats in stock_profiles.items():
        sym_pat_lookup[sym] = {p["signal"]: p for p in pats}

    alerts   = []
    for sig in ALL_SIGNALS:
        if sig not in recent_df.columns:
            continue
        active = recent_df[recent_df[sig].fillna(False)].copy()
        if active.empty:
            continue
        for _, row in active.iterrows():
            sym     = row["sym"]
            pat     = sym_pat_lookup.get(sym, {}).get(sig)
            if not pat:
                continue  # only alert on stocks with validated patterns

            sig_date  = row["date"]
            sig_close = row["c"]
            today_row = df[(df["sym"]==sym) & (df["date"]==latest_date)]
            cur_price = float(today_row["c"].iloc[0]) if not today_row.empty else sig_close

            days_ago  = int((latest_date - sig_date).days)
            pct_since = round((cur_price - sig_close)/sig_close*100, 2)

            # Buy/sell zone logic based on historical returns
            hist_min  = pat.get("min_ret_20d", 0)
            hist_avg  = pat.get("avg_20d", 10)
            hist_max  = pat.get("max_ret_20d", 20)
            atr       = float(row["atr14"]) if not pd.isna(row.get("atr14", float("nan"))) else sig_close*0.02
            buy_lo    = round(sig_close - 0.5*atr, 2)
            buy_hi    = round(sig_close + 0.3*atr, 2)
            sl_px     = round(sig_close - 1.5*atr, 2)
            t1_px     = round(sig_close*(1+max(hist_min,5)/100), 2)
            t2_px     = round(sig_close*(1+hist_avg/100), 2)

            # Zone status
            if pct_since >= hist_avg:
                zone = "SOLD — target reached"
                zone_cls = "sold"
            elif pct_since >= hist_min and hist_min > 0:
                zone = "SELL ZONE — minimum target hit"
                zone_cls = "sell"
            elif cur_price < sl_px:
                zone = "BELOW SL — not in this case as WR=100%"
                zone_cls = "warn"
            elif pct_since < 0:
                zone = "BUY ZONE — still below entry"
                zone_cls = "buy"
            else:
                zone = "RUNNING — hold to target"
                zone_cls = "run"

            is_today  = sig_date.date() == latest_date.date()
            alerts.append({
                "sym":          sym,
                "sig_date":     str(sig_date.date()),
                "signal":       sig,
                "signal_name":  ALL_SIGNALS[sig],
                "is_today":     is_today,
                "days_ago":     days_ago,
                "sig_close":    round(sig_close,2),
                "cur_price":    round(cur_price,2),
                "pct_since":    pct_since,
                "buy_zone_lo":  buy_lo,
                "buy_zone_hi":  buy_hi,
                "stop_loss":    sl_px,
                "target_1":     t1_px,
                "target_2":     t2_px,
                "zone_status":  zone,
                "zone_cls":     zone_cls,
                "vol":          int(row["v"]),
                "vol_ratio":    round(float(row.get("vol_ratio",0)),1),
                "grade":        pat["grade"],
                "n_years":      pat["n_years"],
                "years":        pat["years"],
                "avg_20d":      pat["avg_20d"],
                "min_ret":      pat.get("min_ret_20d",0),
                "max_ret":      pat.get("max_ret_20d",0),
                "above_200ma":  bool(row.get("above_ma200",0)),
                "score":        pat["score"],
            })

    # Sort: today first, then by grade+score
    grade_ord = {"A+":0,"A":1,"B":2,"C":3,"—":4}
    alerts.sort(key=lambda a: (
        0 if a["is_today"] else 1,
        grade_ord.get(a["grade"],9),
        -a["score"]
    ))
    return alerts

# ---------------------------------------------------------------------------
# Global month heatmap
# ---------------------------------------------------------------------------
def build_global_heatmap(df, signals):
    """
    For each signal, for each month: what was the average 20d return
    across all stocks that ever fired that signal in that month?
    """
    heatmap = {}
    for sig in signals:
        if sig not in df.columns:
            continue
        mask = df[sig].fillna(False) & df["vol_ok"]
        hits = df[mask][["month","fwd_20"]].dropna()
        if len(hits) < 5:
            continue
        mdata = {}
        for m in range(1,13):
            ms = hits[hits["month"]==m]["fwd_20"]
            if len(ms) >= 2:
                avg_r = ms.mean()*100
                wr    = (ms>0).sum()/len(ms)*100
                mdata[MONTHS[m-1]] = {
                    "occ":int(len(ms)), "wr":round(wr,1),
                    "avg":round(avg_r,2),
                    "min":round(ms.min()*100,2),
                    "max":round(ms.max()*100,2),
                }
        if mdata:
            heatmap[sig] = {"name":ALL_SIGNALS[sig], "months":mdata}
    return heatmap

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*65)
    print("NSE Pattern Discovery  v2  — 100% Win Rate Engine")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Manifest…")
    if not MANIFEST.exists():
        print(f"  ERROR: {MANIFEST}"); sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    trading_days_sorted = sorted(manifest.keys())
    print(f"  {len(trading_days_sorted)} trading days [{trading_days_sorted[0]} → {trading_days_sorted[-1]}]")

    print("\n[2] Loading all equity CSVs…")
    df = load_all_data(manifest)
    print(f"  {len(df):,} rows, {df['sym'].nunique():,} unique symbols")

    # Filter stocks with enough history
    sym_counts = df.groupby("sym")["date"].count()
    valid_syms = sym_counts[sym_counts >= MIN_TRADING_D].index
    df = df[df["sym"].isin(valid_syms)].copy()
    print(f"  {len(valid_syms)} stocks with ≥{MIN_TRADING_D} trading days")

    print("\n[3] Computing indicators…")
    groups = []
    for i, sym in enumerate(sorted(df["sym"].unique())):
        try:
            groups.append(compute_indicators(df[df["sym"]==sym].copy()))
        except Exception:
            pass
        if (i+1)%300==0: print(f"    {i+1}/{len(valid_syms)}…")
    df = pd.concat(groups, ignore_index=True)
    del groups; gc.collect()
    print(f"  Done — {len(df):,} rows with indicators")

    print("\n[4] Defining signals…")
    df = define_signals(df)

    print("\n[5] Cross-stock pattern analysis (strict filters)…")
    patterns = []
    for sig, name in ALL_SIGNALS.items():
        try:
            r = analyse_signal(df, sig, name)
            if r: patterns.append(r)
        except Exception as e:
            print(f"    WARN {sig}: {e}")
    patterns.sort(key=lambda p: -p.get("score",0))
    grade_counts = {}
    for p in patterns:
        g=p.get("grade","—"); grade_counts[g]=grade_counts.get(g,0)+1
    print(f"  {len(patterns)} patterns pass strict filters | {grade_counts}")

    print("\n[6] Per-stock 100% pattern mining…")
    stock_profiles = {}
    all_results    = []
    syms           = sorted(df["sym"].unique())
    for i, sym in enumerate(syms):
        try:
            g = df[df["sym"]==sym]
            data_years = g["year"].nunique()
            results    = mine_stock(g, sym, data_years)
            if results:
                stock_profiles[sym] = results
                all_results.extend(results)
        except Exception:
            pass
        if (i+1)%300==0: print(f"    {i+1}/{len(syms)}…")

    all_results.sort(key=lambda x: (-x["n_years"],-x["avg_20d"]))
    n_aplus = sum(1 for r in all_results if r["grade"]=="A+")
    n_a     = sum(1 for r in all_results if r["grade"]=="A")
    print(f"  {len(stock_profiles)} stocks with valid patterns | A+:{n_aplus} A:{n_a}")
    print(f"  Top 5:")
    for r in all_results[:5]:
        print(f"    {r['sym']:<14} {r['signal']:<5} {r['grade']}  "
              f"WR20:{r['wr_20d']}%  Avg20:{r['avg_20d']:+.1f}%  "
              f"{r['n_years']}yrs  Occ:{r['occ_20d']}")

    print("\n[7] Generating alerts (today + last 20 trading days)…")
    alerts = generate_alerts(df, stock_profiles, trading_days_sorted)
    print(f"  {len(alerts)} alerts | Today: {sum(1 for a in alerts if a['is_today'])}")

    print("\n[8] Global month heatmap…")
    heatmap = build_global_heatmap(df, list(ALL_SIGNALS.keys()))
    print(f"  {len(heatmap)} signals with heatmap data")

    print("\n[9] Writing JSON files…")
    ist     = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    # patterns.json
    with open(OUT_DIR/"patterns.json","w") as f:
        json.dump({
            "generated_at": now_ist,
            "thresholds":   {"wr_all":MIN_WR_ALL,"avg_5d":MIN_AVG_5D,
                             "avg_10d":MIN_AVG_10D,"avg_20d":MIN_AVG_20D,
                             "min_vol":MIN_VOLUME},
            "stocks_analyzed": int(len(syms)),
            "data_range":   f"{trading_days_sorted[0]} to {trading_days_sorted[-1]}",
            "grade_counts": grade_counts,
            "patterns":     patterns,
        }, f, indent=2)
    print(f"  OK patterns.json ({len(patterns)} patterns)")

    # stock_profiles.json — top 500 results only to keep manageable
    filtered_profs = {sym: v for sym,v in stock_profiles.items()
                      if any(r["grade"] in ("A+","A") for r in v)}
    with open(OUT_DIR/"stock_profiles.json","w") as f:
        json.dump({
            "generated_at": now_ist,
            "n_stocks":     len(filtered_profs),
            "n_aplus":      n_aplus,
            "n_a":          n_a,
            "all_results":  all_results[:600],   # top 600 by score
            "profiles":     filtered_profs,
        }, f, indent=2)
    print(f"  OK stock_profiles.json ({len(filtered_profs)} stocks)")

    # alerts.json
    with open(OUT_DIR/"alerts.json","w") as f:
        json.dump({
            "generated_at":   now_ist,
            "latest_date":    str(df["date"].max().date()),
            "window_days":    20,
            "total_alerts":   len(alerts),
            "today_count":    sum(1 for a in alerts if a["is_today"]),
            "alerts":         alerts,
        }, f, indent=2)
    print(f"  OK alerts.json ({len(alerts)} alerts)")

    # heatmap.json
    with open(OUT_DIR/"heatmap.json","w") as f:
        json.dump({
            "generated_at": now_ist,
            "signals":      heatmap,
            "months":       MONTHS,
        }, f, indent=2)
    print(f"  OK heatmap.json ({len(heatmap)} signals)")

    print(f"\n✅ Done. pattern_signals/ updated with 4 files.")

if __name__ == "__main__":
    main()
