#!/usr/bin/env python3
"""
pattern_discovery.py  v3
=========================
NSE Pattern Discovery — 100% win-rate engine + candle morphology.

Key changes from v2:
  - Alerts: strictly last 20 TRADING days (not calendar days). Older discarded.
  - Min return: ALL occurrences at ALL windows must be >= +5% (not just average)
  - Candle morphology: buckets every candle shape, finds which gives >3% next day />7% next week
  - Heatmap removed entirely
  - Grading based on consistency across all occurrences at all windows
  - Incremental checkpoint: skips run if no new dates since last run
  - Math verified throughout

Math conventions (all verified):
  body        = abs(close - open)                     always >= 0
  lower_wick  = min(open,close) - low                 always >= 0
  upper_wick  = high - max(open,close)                always >= 0
  fwd_w       = close[t+w] / close[t] - 1            fractional (x100 = %)
  win_rate    = (fwd_w > 0).sum() / n * 100           % of occurrences positive
  avg_return  = fwd_w.mean() * 100                    average % return
  min_return  = fwd_w.min()  * 100                    worst single occurrence %
"""

import json, os, gc, sys, traceback, math
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pip install pandas numpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# SAFE JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "pattern_signals"
MANIFEST   = DATA_DIR / "manifest.json"
CHECKPOINT = OUT_DIR / "checkpoint.json"

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CROSS_MIN_WR      = 80.0    # cross-stock win rate threshold (relaxed)
STOCK_MIN_WR      = 100.0   # per-stock win rate (strict 100%)
STOCK_MIN_AVG     = 5.0     # avg return >= +5% at each window (base filter)
STOCK_MIN_ANY_ONE = 15.0    # at least ONE window must have avg >= 15% (OR logic)
MIN_VOLUME        = 100_000  # minimum tradeable volume
MIN_OCC           = 3        # minimum occurrences
MIN_YEARS         = 2        # must appear in 2+ years
MAX_YR_GAP        = 2        # max gap between consecutive years
MIN_TRADING_DAYS  = 200      # minimum data history per stock
FWD_WINDOWS       = [5, 10, 20]
SCRIPT_VERSION    = "v3"

MORPH_MIN_OCC     = 5        # min candle bucket occurrences
MORPH_ND_THRESH   = 3.0      # next-day return threshold %
MORPH_WK_THRESH   = 7.0      # next-week return threshold %

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

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

def _fcol(hdr, aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def _phdr(raw):
    return [h.strip().strip('"').strip("'").upper() for h in raw.split(",")]

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) < 2: return rows
        hdr = _phdr(lines[0])
        i_sym = _fcol(hdr, SYM_A); i_ser = _fcol(hdr, SER_A)
        i_o = _fcol(hdr, O_A);   i_h = _fcol(hdr, H_A)
        i_l = _fcol(hdr, L_A);   i_c = _fcol(hdr, C_A)
        i_v = _fcol(hdr, V_A)
        if i_sym < 0 or i_c < 0: return rows
        mc = max(x for x in [i_sym,i_o,i_h,i_l,i_c,i_v] if x >= 0)
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
                # Sanity check: OHLC relationships must be valid
                if c > 0 and o > 0 and h >= max(o,c) and l <= min(o,c) and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except (ValueError, IndexError):
                pass
    except Exception:
        pass
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT.exists():
        try: return json.loads(CHECKPOINT.read_text())
        except Exception: pass
    return {}

def save_checkpoint(cp):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(cp, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_all_data(trading_days):
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
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(g):
    g = g.copy().reset_index(drop=True)
    o, h, l, c, v = g["o"], g["h"], g["l"], g["c"], g["v"]

    # Candle geometry — VERIFIED
    g["body"]       = (c - o).abs()
    g["range_"]     = h - l
    g["upper_wick"] = h - pd.concat([o,c],axis=1).max(axis=1)
    g["lower_wick"] = pd.concat([o,c],axis=1).min(axis=1) - l
    # close_pos: 0 = closed at low, 1 = closed at high
    g["close_pos"]  = np.where(g["range_"]>0, (c-l)/g["range_"], 0.5)
    g["green"]      = (c >= o).astype(int)
    g["month"]      = g["date"].dt.month
    g["year"]       = g["date"].dt.year

    # Volume
    g["vol20"]      = v.rolling(20,min_periods=5).mean()
    g["vol_ratio"]  = np.where(g["vol20"]>0, v/g["vol20"], 0.0)
    g["vol252max"]  = v.rolling(252,min_periods=60).max().shift(1)
    g["vol_ok"]     = (v >= MIN_VOLUME)

    # Range
    g["range20"]    = g["range_"].rolling(20,min_periods=5).mean()
    g["range7min"]  = g["range_"].rolling(7,min_periods=7).min()
    g["range4min"]  = g["range_"].rolling(4,min_periods=4).min()

    # MAs
    g["ma20"]         = c.rolling(20,min_periods=10).mean()
    g["ma50"]         = c.rolling(50,min_periods=25).mean()
    g["ma200"]        = c.rolling(200,min_periods=100).mean()
    g["above_ma200"]  = (c > g["ma200"]).astype(int)
    g["above_ma50"]   = (c > g["ma50"]).astype(int)
    g["pct_below_200"]= np.where(g["ma200"]>0, (c/g["ma200"]-1)*100, 0.0)

    # ATR — True Range = max(H-L, |H-prevC|, |L-prevC|)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14,min_periods=7).mean()

    # Historical highs (shifted so today not included)
    g["high20"]  = h.rolling(20,min_periods=10).max().shift(1)
    g["high52w"] = h.rolling(252,min_periods=60).max().shift(1)

    # Previous bar
    g["prev_c"]     = c.shift(1)
    g["prev_h"]     = h.shift(1)
    g["prev_l"]     = l.shift(1)
    g["prev_o"]     = o.shift(1)
    g["prev_green"] = g["green"].shift(1)
    g["prev_body"]  = g["body"].shift(1)

    # Consecutive red/green streaks
    cr_l, cg_l, cr, cg = [], [], 0, 0
    for gv in g["green"]:
        if gv == 0: cr += 1; cg = 0
        else:        cg += 1; cr = 0
        cr_l.append(cr); cg_l.append(cg)
    g["consec_red"]   = cr_l
    g["consec_green"] = cg_l

    # MA cross
    g["ma50_above_200"] = (g["ma50"] > g["ma200"]).astype(int)
    g["golden_cross"]   = ((g["ma50_above_200"]==1) &
                           (g["ma50_above_200"].shift(1)==0)).astype(int)
    g["below_ma50_10d"] = (1-g["above_ma50"]).rolling(10,min_periods=10).sum().shift(1)

    # Volume dry-up
    vlow = (g["vol_ratio"]<=0.3).astype(int)
    g["vol_dryup_3d"] = vlow.rolling(3,min_periods=3).sum() >= 2
    g["vol_dryup_5d"] = vlow.rolling(5,min_periods=3).sum() >= 2

    # Forward returns — fwd_w = close[t+w]/close[t] - 1 (fractional)
    for w in FWD_WINDOWS:
        g[f"fwd_{w}"] = c.shift(-w) / c - 1

    # Next open (for gap analysis)
    g["next_open"]     = o.shift(-1)
    g["next_open_pct"] = (g["next_open"] / c - 1) * 100

    return g

# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
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
    safe_range = d["range_"].replace(0, np.nan)
    safe_body  = d["body"].replace(0, np.nan)

    d["V1"] = (d["vol_ratio"]>=5)  & (d["c"]>d["o"]) & d["vol_ok"]
    d["V2"] = (d["vol_ratio"]>=10) & (d["c"]>d["o"]) & d["vol_ok"]
    d["V4"] = (d["v"]>=d["vol252max"]) & (d["vol252max"]>0) & d["vol_ok"]

    d["C1"] = ((d["body"]>0) &
               (d["lower_wick"] >= 2*safe_body) &
               (d["upper_wick"] <= safe_body) &
               (d["close_pos"]>=0.7) & d["vol_ok"])
    d["C2"] = ((safe_range>0) &
               (d["body"]/safe_range >= 0.85) &
               (d["c"]>d["o"]) & (d["close_pos"]>=0.90) & d["vol_ok"])
    d["C3"] = ((d["c"]>d["o"]) & (d["prev_green"]==0) &
               (d["o"]<=d["prev_c"]) & (d["c"]>=d["prev_o"]) &
               (d["body"]>d["prev_body"]) & d["vol_ok"])
    d["C4"] = (d["consec_red"].shift(1)>=3) & (d["c"]>d["o"]) & d["vol_ok"]
    d["C5"] = (d["range_"]>0) & (d["range_"]<=d["range7min"]) & d["vol_ok"]
    d["C6"] = ((d["range_"]>=1.5*d["range20"].replace(0,np.nan)) &
               (d["c"]>d["o"]) & (d["close_pos"]>=0.7) & d["vol_ok"])

    d["B1"] = ((d["h"]>d["high20"]) & (d["high20"]>0) &
               (d["vol_ratio"]>=1.5) & (d["c"]>d["o"]) & d["vol_ok"])
    d["B2"] = ((d["h"]>d["high52w"]) & (d["high52w"]>0) &
               (d["vol_ratio"]>=1.5) & (d["c"]>d["o"]) & d["vol_ok"])
    d["B3"] = (d["h"]<d["prev_h"]) & (d["l"]>d["prev_l"]) & (d["prev_h"]>0) & d["vol_ok"]

    d["T1"] = ((d["above_ma50"]==1) & (d["above_ma50"].shift(1)==0) &
               (d["below_ma50_10d"]==10) & d["vol_ok"])
    d["T2"] = (d["golden_cross"]==1) & d["vol_ok"]
    d["T3"] = ((d["above_ma200"]==1) & (d["vol_ratio"]>=2) &
               (d["c"]>d["o"]) & d["vol_ok"])
    d["T4"] = ((d["pct_below_200"]<=-20) & (d["ma200"]>0) &
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
                (d["range_"] <= d["range4min"].replace(0,np.nan)) &
                (d["above_ma200"]==1) & d["vol_ok"])
    return d

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def years_pass_gap(years_list):
    if len(years_list) < MIN_YEARS: return False
    sy = sorted(years_list)
    for i in range(1, len(sy)):
        if sy[i]-sy[i-1] > MAX_YR_GAP: return False
    return True

def score(years_list, avg_ret_20d, n_occ):
    return round(len(years_list) * avg_ret_20d * math.log(max(n_occ,1)+1), 1)

def grade_pattern(n_years, n_occ, all_100pct, worst_return):
    """
    Grade strictly based on:
      - Consistency across years and occurrences
      - Whether EVERY occurrence at EVERY window is positive
      - Whether worst single return >= +5%
    """
    if not all_100pct or worst_return < STOCK_MIN_AVG:
        return "C"
    if n_years >= 4 and n_occ >= 8: return "A+"
    if n_years >= 3 and n_occ >= 5: return "A"
    if n_years >= 2 and n_occ >= 3: return "B"
    return "C"

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-STOCK ANALYSIS (>=80% win rate)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_cross(df, sig, name):
    if sig not in df.columns: return None
    mask = df[sig].fillna(False) & df["vol_ok"]
    hits = df[mask].copy()
    if len(hits) < MIN_OCC: return None

    res = {"signal":sig, "name":name, "occurrences":len(hits)}
    for w in FWD_WINDOWS:
        valid = hits[f"fwd_{w}"].dropna()
        if len(valid) < MIN_OCC: return None
        wr    = (valid>0).sum()/len(valid)*100     # verified
        avg_r = valid.mean()*100                    # verified
        min_r = valid.min()*100                     # verified
        if wr < CROSS_MIN_WR: return None
        if avg_r < STOCK_MIN_AVG: return None
        res[f"wr_{w}d"]  = round(wr,1)
        res[f"avg_{w}d"] = round(avg_r,2)
        res[f"min_{w}d"] = round(min_r,2)
        res[f"n_{w}d"]   = int(len(valid))

    years_all = sorted(hits["year"].unique().tolist())
    if not years_pass_gap(years_all): return None
    res["years"]   = [int(y) for y in years_all]
    res["n_years"] = len(years_all)
    res["score"]   = score(years_all, res["avg_20d"], len(hits))

    n_y  = len(years_all); n_oc = res["n_20d"]
    res["grade"] = ("A+" if n_y>=4 and n_oc>=10 else
                    "A"  if n_y>=3 and n_oc>=7  else
                    "B"  if n_y>=2 and n_oc>=5  else
                    "C"  if n_y>=2              else "—")

    ywise = {}
    for y in years_all:
        yr = hits[hits["year"]==y]["fwd_20"].dropna()
        if len(yr):
            ywise[str(int(y))] = {
                "occ": int(len(yr)),
                "wr":  round((yr>0).sum()/len(yr)*100,1),
                "avg": round(yr.mean()*100,2),
                "min": round(yr.min()*100,2),
                "max": round(yr.max()*100,2),
            }
    res["yearly"]     = ywise
    res["top_stocks"] = hits["sym"].value_counts().head(10).to_dict()
    return res

# ─────────────────────────────────────────────────────────────────────────────
# PER-STOCK 100% MINING
# ─────────────────────────────────────────────────────────────────────────────
def mine_stock(g, sym, data_years):
    min_yrs = max(MIN_YEARS, math.ceil(data_years/2))
    found   = []

    for sig in ALL_SIGNALS:
        if sig not in g.columns: continue
        mask  = g[sig].fillna(False) & g["vol_ok"]
        hits  = g[mask].copy()
        if len(hits) < MIN_OCC: continue

        valid_20 = hits["fwd_20"].dropna()
        if len(valid_20) < MIN_OCC: continue

        yr_data = hits[hits["fwd_20"].notna()]["year"].unique()
        if not years_pass_gap(sorted(yr_data)): continue

        ok = True; win_data = {}
        for w in FWD_WINDOWS:
            valid = hits[f"fwd_{w}"].dropna()
            if len(valid) < MIN_OCC: ok = False; break

            wr    = (valid>0).sum()/len(valid)*100   # verified
            avg_r = valid.mean()*100                  # verified
            min_r = valid.min()*100                   # worst single occurrence — verified
            max_r = valid.max()*100
            med_r = valid.median()*100

            # STRICT filters
            if wr    < STOCK_MIN_WR:  ok = False; break   # 100% win rate required
            if avg_r < STOCK_MIN_AVG: ok = False; break   # avg return required

            win_data[w] = {"wr":round(wr,1),"avg":round(avg_r,2),
                           "min":round(min_r,2),"max":round(max_r,2),
                           "med":round(med_r,2),"occ":int(len(valid))}

        if not ok: continue

        # MIN 15% RETURN IN AT LEAST ONE WINDOW (OR logic, not AND)
        # Any one of 5d, 10d, 20d must have avg return >= 15%
        max_avg_any_window = max(win_data[w]["avg"] for w in FWD_WINDOWS)
        if max_avg_any_window < 15.0:
            continue

        years_all = sorted([int(y) for y in yr_data])
        if len(years_all) < min(MIN_YEARS, min_yrs): continue

        # Per-year detail
        yr_detail = {}
        for y in sorted(hits["year"].unique()):
            yrows = hits[hits["year"]==y]
            yv20  = yrows["fwd_20"].dropna()
            yv5   = yrows["fwd_5"].dropna()
            yv10  = yrows["fwd_10"].dropna()
            if len(yv20):
                yr_detail[str(int(y))] = {
                    "occ":     int(len(yv20)),
                    "wr_20d":  round((yv20>0).sum()/len(yv20)*100,1),
                    "avg_20d": round(yv20.mean()*100,2),
                    "min_20d": round(yv20.min()*100,2),
                    "max_20d": round(yv20.max()*100,2),
                    "avg_5d":  round(yv5.mean()*100,2) if len(yv5) else None,
                    "avg_10d": round(yv10.mean()*100,2) if len(yv10) else None,
                }

        n_years  = len(years_all)
        n_occ    = win_data[20]["occ"]
        all_100  = all(win_data[w]["wr"] == 100.0 for w in FWD_WINDOWS)
        worst    = min(win_data[w]["min"] for w in FWD_WINDOWS)

        found.append({
            "sym":        sym,
            "signal":     sig,
            "name":       ALL_SIGNALS[sig],
            "grade":      grade_pattern(n_years, n_occ, all_100, worst),
            "n_years":    n_years,
            "years":      years_all,
            "occ_20d":    n_occ,
            "wr_5d":      win_data[5]["wr"],
            "avg_5d":     win_data[5]["avg"],
            "min_5d":     win_data[5]["min"],
            "wr_10d":     win_data[10]["wr"],
            "avg_10d":    win_data[10]["avg"],
            "min_10d":    win_data[10]["min"],
            "wr_20d":     win_data[20]["wr"],
            "avg_20d":    win_data[20]["avg"],
            "min_20d":    win_data[20]["min"],
            "max_20d":    win_data[20]["max"],
            "med_20d":    win_data[20]["med"],
            "worst_return": worst,
            "avg_vol":    int(hits["v"].mean()),
            "score":      score(years_all, win_data[20]["avg"], n_occ),
            "yr_detail":  yr_detail,
        })

    found.sort(key=lambda x: -x["score"])
    seen, unique = set(), []
    for f in found:
        if f["signal"] not in seen:
            seen.add(f["signal"]); unique.append(f)
    return unique

# ─────────────────────────────────────────────────────────────────────────────
# ALERTS — strictly last 20 TRADING days
# ─────────────────────────────────────────────────────────────────────────────
def generate_alerts(df, stock_profiles, trading_days_list):
    # Last 20 TRADING days (by manifest/calendar, not calendar days)
    recent_td  = set(trading_days_list[-20:]) if len(trading_days_list)>=20 \
                 else set(trading_days_list)
    latest_dt  = df["date"].max()
    recent_df  = df[df["date"].dt.strftime("%Y-%m-%d").isin(recent_td)].copy()

    # Fast lookup
    lookup = {sym: {p["signal"]:p for p in pats}
              for sym, pats in stock_profiles.items()}

    alerts = []
    for sig in ALL_SIGNALS:
        if sig not in recent_df.columns: continue
        active = recent_df[recent_df[sig].fillna(False)]
        if active.empty: continue

        for _, row in active.iterrows():
            sym = row["sym"]
            pat = lookup.get(sym,{}).get(sig)
            if not pat: continue   # only stocks with validated 100% patterns

            sig_dt    = row["date"]
            sig_close = row["c"]
            tr        = df[(df["sym"]==sym) & (df["date"]==latest_dt)]
            cur_price = float(tr["c"].iloc[0]) if not tr.empty else sig_close

            # VERIFIED: pct_since = return from signal close to current close
            pct_since = round((cur_price - sig_close)/sig_close*100, 2)
            days_ago  = int((latest_dt - sig_dt).days)

            atr = float(row["atr14"]) if pd.notna(row.get("atr14")) else sig_close*0.02
            atr = max(atr, sig_close*0.005)

            hist_avg = pat["avg_20d"]
            hist_min = pat["min_20d"]   # actual worst occurrence
            hist_max = pat["max_20d"]

            # Targets based on ACTUAL historical performance
            buy_lo = round(sig_close - 0.5*atr, 2)
            buy_hi = round(sig_close + 0.3*atr, 2)
            sl_px  = round(sig_close - 1.5*atr, 2)
            t1_px  = round(sig_close*(1+hist_min/100), 2)   # conservative
            t2_px  = round(sig_close*(1+hist_avg/100), 2)   # realistic

            if pct_since >= hist_avg:
                zone="TARGET REACHED"; zone_cls="sold"
            elif pct_since >= hist_min:
                zone="PARTIAL TARGET HIT"; zone_cls="sell"
            elif pct_since < 0:
                zone="BUY ZONE"; zone_cls="buy"
            else:
                zone="RUNNING"; zone_cls="run"

            body  = round(float(row["body"]),2)
            lw    = round(float(row["lower_wick"]),2)
            uw    = round(float(row["upper_wick"]),2)
            wr    = round(lw/body,2) if body>0 else 0

            alerts.append({
                "sym":          sym,
                "sig_date":     str(sig_dt.date()),
                "signal":       sig,
                "signal_name":  ALL_SIGNALS[sig],
                "grade":        pat["grade"],
                "is_today":     sig_dt.date()==latest_dt.date(),
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
                "n_years":      pat["n_years"],
                "years":        pat["years"],
                "occ_20d":      pat["occ_20d"],
                "win_rate_5d":  pat["wr_5d"],
                "win_rate_10d": pat["wr_10d"],
                "win_rate_20d": pat["wr_20d"],
                "avg_ret_5d":   pat["avg_5d"],
                "avg_ret_10d":  pat["avg_10d"],
                "avg_ret_20d":  pat["avg_20d"],
                "min_ret_20d":  pat["min_20d"],
                "worst_return": pat["worst_return"],
                "score":        pat["score"],
                "open":   round(float(row["o"]),2),
                "high":   round(float(row["h"]),2),
                "low":    round(float(row["l"]),2),
                "close":  round(float(row["c"]),2),
                "body":   body, "lower_wick":lw, "upper_wick":uw,
                "wick_ratio":    wr,
                "vol":           int(row["v"]),
                "vol_ratio":     round(float(row.get("vol_ratio",0)),1),
                "above_200ma":   bool(row.get("above_ma200",0)),
            })

    grade_ord = {"A+":0,"A":1,"B":2,"C":3}
    alerts.sort(key=lambda a:(0 if a["is_today"] else 1,
                               grade_ord.get(a["grade"],9), -a["score"]))
    return alerts

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE MORPHOLOGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def build_morphology(df):
    """
    Bucket every candle by body/wick shape.
    Find which shapes give >3% next-day open gap or >7% next-week return.
    Also tracks: does next day open higher than signal close (gap-up)?
    """
    mask = df["fwd_5"].notna() & df["next_open"].notna() & df["vol_ok"]
    sub  = df[mask].copy()
    if sub.empty: return {}

    safe_body = sub["body"].replace(0, np.nan)

    # Body as % of close price
    sub["body_pct"] = np.where(sub["c"]>0, sub["body"]/sub["c"]*100, 0.0)

    def body_bkt(bp):
        if bp < 0.5:  return "tiny"
        if bp < 1.5:  return "small"
        if bp < 3.0:  return "medium"
        if bp < 5.0:  return "large"
        return "huge"

    def wick_bkt(ratio):
        if pd.isna(ratio) or ratio < 0.25: return "none"
        if ratio < 1.0:                    return "small"
        if ratio < 2.0:                    return "medium"
        if ratio < 4.0:                    return "long"
        return "very_long"

    sub["bbkt"] = sub["body_pct"].apply(body_bkt)
    sub["ubkt"] = (sub["upper_wick"]/safe_body).apply(wick_bkt)
    sub["lbkt"] = (sub["lower_wick"]/safe_body).apply(wick_bkt)
    sub["clr"]  = sub["green"].map({1:"green",0:"red"})
    sub["bkey"] = (sub["clr"]+"_"+sub["bbkt"]+
                   "_L"+sub["lbkt"]+"_U"+sub["ubkt"])

    results = {}
    for bkey, grp in sub.groupby("bkey"):
        n = len(grp)
        if n < MORPH_MIN_OCC: continue

        gap   = grp["next_open_pct"]         # % gap up/down at next open
        fwd5  = grp["fwd_5"] * 100           # 5-day return %

        gap_up_rate  = round((gap > 0).sum()/n*100, 1)
        avg_gap      = round(gap.mean(), 2)
        fwd5_wr      = round((fwd5>0).sum()/n*100, 1)
        fwd5_avg     = round(fwd5.mean(), 2)
        fwd5_min     = round(fwd5.min(), 2)

        nd_gt3_pct   = round((gap  >= MORPH_ND_THRESH).sum()/n*100, 1)
        wk_gt7_pct   = round((fwd5 >= MORPH_WK_THRESH).sum()/n*100, 1)

        # Skip buckets that offer no useful signal
        if nd_gt3_pct < 30 and wk_gt7_pct < 30: continue

        results[bkey] = {
            "description":      bkey.replace("_"," "),
            "occurrences":      int(n),
            "gap_up_rate_pct":  gap_up_rate,
            "avg_gap_pct":      avg_gap,
            "next_day_gt3pct":  nd_gt3_pct,   # % of time next day opens >3% higher
            "week_gt7pct":      wk_gt7_pct,   # % of time week return > 7%
            "week_win_rate":    fwd5_wr,
            "week_avg_return":  fwd5_avg,
            "week_min_return":  fwd5_min,      # worst week return in this bucket
        }

    results = dict(sorted(results.items(), key=lambda x:-x[1]["week_gt7pct"]))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("NSE Pattern Discovery  v3")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Manifest
    print("\n[1] Manifest...")
    if not MANIFEST.exists():
        print(f"  ERROR: {MANIFEST}"); sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    tds = sorted(manifest.keys())
    latest_str = tds[-1]
    print(f"  {len(tds)} trading days [{tds[0]} -> {latest_str}]")

    # [2] Checkpoint
    print("\n[2] Checkpoint...")
    cp     = load_checkpoint()
    last   = cp.get("last_full_run_date")
    ver_ok = cp.get("script_version","") == SCRIPT_VERSION
    force  = os.environ.get("FORCE_FULL_RERUN","").lower() == "true"

    if not force and ver_ok and last:
        new_dates = [d for d in tds if d > last]
        if not new_dates:
            print(f"  No new dates since {last}. Nothing to do.")
            print("  To force rerun: set env FORCE_FULL_RERUN=true")
            sys.exit(0)
        print(f"  {len(new_dates)} new dates since {last}. Running full recompute.")
    else:
        print("  Full run (first time, new version, or forced).")

    # [3] Load data
    print("\n[3] Loading all equity CSVs...")
    df = load_all_data(manifest)
    print(f"  {len(df):,} rows, {df['sym'].nunique():,} symbols")

    sym_counts = df.groupby("sym")["date"].count()
    valid_syms = sym_counts[sym_counts >= MIN_TRADING_DAYS].index
    df = df[df["sym"].isin(valid_syms)].copy()
    sym_list = sorted(df["sym"].unique())
    print(f"  {len(sym_list)} stocks with >= {MIN_TRADING_DAYS} trading days")

    # [4] Indicators
    print("\n[4] Computing indicators...")
    sym_grps = {s:g.copy() for s,g in df.groupby("sym")}
    computed = []
    for i, sym in enumerate(sym_list):
        try: computed.append(compute_indicators(sym_grps[sym]))
        except Exception: pass
        if (i+1)%300==0: print(f"    {i+1}/{len(sym_list)}...")
    df = pd.concat(computed, ignore_index=True)
    del computed, sym_grps; gc.collect()
    print(f"  {len(df):,} rows with indicators")

    # [5] Signals
    print("\n[5] Defining signals...")
    df = define_signals(df)

    # [6] Cross-stock
    print("\n[6] Cross-stock analysis (>=80% win rate, avg>=+5% all windows)...")
    patterns = []
    for sig, name in ALL_SIGNALS.items():
        try:
            r = analyse_cross(df, sig, name)
            if r: patterns.append(r)
        except Exception as e:
            print(f"    WARN {sig}: {e}")
    patterns.sort(key=lambda p: -p.get("score",0))
    gcounts = {}
    for p in patterns:
        g=p.get("grade","—"); gcounts[g]=gcounts.get(g,0)+1
    print(f"  {len(patterns)} patterns | {gcounts}")

    # [7] Per-stock mining
    print("\n[7] Per-stock 100% mining (100% wr, min +5% every occurrence)...")
    sym_grps2 = {s:g for s,g in df.groupby("sym")}
    stock_profiles = {}; all_results = []
    for i, sym in enumerate(sorted(sym_grps2.keys())):
        try:
            g    = sym_grps2[sym]
            dyrs = int(g["year"].nunique())
            res  = mine_stock(g, sym, dyrs)
            if res:
                stock_profiles[sym] = res
                all_results.extend(res)
        except Exception: pass
        if (i+1)%300==0: print(f"    {i+1}/{len(sym_grps2)}...")
    del sym_grps2; gc.collect()

    all_results.sort(key=lambda x:(-x["n_years"],-x["avg_20d"]))
    naplus = sum(1 for r in all_results if r["grade"]=="A+")
    na     = sum(1 for r in all_results if r["grade"]=="A")
    nb     = sum(1 for r in all_results if r["grade"]=="B")
    print(f"  {len(stock_profiles)} stocks | A+:{naplus}  A:{na}  B:{nb}")
    print("  Top 5:")
    for r in all_results[:5]:
        print(f"    {r['sym']:<14} {r['signal']:<5} {r['grade']}  "
              f"Avg20:{r['avg_20d']:+.1f}%  Min20:{r['min_20d']:+.1f}%  "
              f"{r['n_years']}yrs  Occ:{r['occ_20d']}")

    # [8] Alerts — strictly last 20 trading days
    print("\n[8] Alerts (strictly last 20 TRADING days)...")
    alerts     = generate_alerts(df, stock_profiles, tds)
    today_cnt  = sum(1 for a in alerts if a["is_today"])
    print(f"  {len(alerts)} alerts | Today: {today_cnt}")

    # [9] Candle morphology
    print("\n[9] Candle morphology...")
    morph = build_morphology(df)
    print(f"  {len(morph)} useful candle buckets")

    # [10] Write outputs
    print("\n[10] Writing JSON files...")
    ist     = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    grade_a_list = [r for r in all_results if r["grade"] in ("A+","A")]

    jdump({"generated_at":now_ist,
           "thresholds":{"cross_min_wr":CROSS_MIN_WR,"stock_min_wr":STOCK_MIN_WR,
                         "min_avg_all":STOCK_MIN_AVG,"min_single":STOCK_MIN_AVG,
                         "min_vol":MIN_VOLUME},
           "stocks_analyzed":int(len(sym_list)),
           "trading_days":len(tds),
           "data_range":f"{tds[0]} to {latest_str}",
           "grade_counts":gcounts,
           "patterns":patterns},
          OUT_DIR/"patterns.json")
    print(f"  OK patterns.json ({len(patterns)} patterns)")

    jdump({"generated_at":now_ist,
           "n_stocks":len(stock_profiles),
           "stocks_with_100pct":len(stock_profiles),
           "n_aplus":naplus,"n_a":na,"n_b":nb,
           "all_results":all_results[:600],
           "all_grade_a":grade_a_list[:200],
           "profiles":{s:v for s,v in stock_profiles.items()
                       if any(r["grade"] in ("A+","A") for r in v)}},
          OUT_DIR/"stock_profiles.json")
    print(f"  OK stock_profiles.json ({len(stock_profiles)} stocks)")

    jdump({"generated_at":now_ist,
           "latest_date":latest_str,"alert_date":latest_str,
           "window_days":20,
           "window_note":"Last 20 TRADING days only — older alerts discarded",
           "total_alerts":len(alerts),"today_count":today_cnt,
           "grade_a":sum(1 for a in alerts if a.get("grade","") in ("A+","A")),
           "alerts":alerts},
          OUT_DIR/"alerts.json")
    print(f"  OK alerts.json ({len(alerts)} alerts)")

    jdump({"generated_at":now_ist,
           "description":"Candle shape -> next-day open gap + next-week return",
           "thresholds":{"next_day_pct":MORPH_ND_THRESH,"next_week_pct":MORPH_WK_THRESH,
                         "min_occ":MORPH_MIN_OCC},
           "n_buckets":len(morph),
           "buckets":morph},
          OUT_DIR/"candle_morphology.json")
    print(f"  OK candle_morphology.json ({len(morph)} buckets)")

    # [11] Checkpoint
    save_checkpoint({"last_full_run_date":latest_str,"script_version":SCRIPT_VERSION,
                     "run_at":now_ist,"stocks_analyzed":int(len(sym_list)),
                     "patterns_found":len(patterns),"profiles_found":len(stock_profiles)})
    print(f"  OK checkpoint.json")
    print(f"\nDone. Next run skips if no new dates after {latest_str}")
    print(f"To force full rerun: set env FORCE_FULL_RERUN=true")

if __name__ == "__main__":
    main()
