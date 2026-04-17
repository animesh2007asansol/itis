#!/usr/bin/env python3
"""
Pin Bar Pattern OPTIMIZER
===========================
Goal: Find the combination of filters that gives the highest win rate
on next-day positive close.

Filters tested in combination:
  1. wick_ratio       : lower_wick / body  (2x, 3x, 4x, 5x)
  2. upper_wick_pct   : upper_wick / body  (25%, 15%, 10%, 5%)
  3. min_prev_red     : previous N days must be red closes (0,1,2,3)
  4. trend_filter     : close < N-day SMA (stock in downtrend) (False,10,20)
  5. volume_filter    : volume > N × 20-day avg volume (False,1.5,2.0,2.5)
  6. delivery_filter  : delivery% > threshold (False,40,50,60)
  7. min_price        : ignore stocks below Rs X (0, 50, 100)
  8. min_signals      : only show combos with >= N historical signals (10,20,50)

Output:
  data/analysis/optimizer_results.json   — all tested combos ranked by win rate
  data/analysis/pinbar_signals.json      — today's alerts using BEST combo
  data/analysis/pinbar_backtest.json     — best combo backtest stats
  data/analysis/pinbar_history.json      — all signals under best combo
"""

import json, sys, logging, warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

class SafeEncoder(json.JSONEncoder):
    """Handles date, numpy int/float types that json.dumps chokes on."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = DATA_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("optimizer")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_equity() -> pd.DataFrame:
    log.info("Loading equity data...")
    frames = []

    # Maps every known NSE column alias → standard name
    # Old format (pre mid-2024): SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY
    # New format (mid-2024+):    TckrSymb, SctySrs, OpnPric, HghPric, LwPric,
    #                             ClsPric, TtlTradgVol  (uppercased by pandas)
    COL_ALIASES = {
        # Symbol
        "SYMBOL":       "SYMBOL",
        "TCKRSYMB":     "SYMBOL",
        # Series
        "SERIES":       "SERIES",
        "SCTYSRS":      "SERIES",
        # Open
        "OPEN":         "OPEN",
        "OPNPRIC":      "OPEN",
        "OPEN PRICE":   "OPEN",
        # High
        "HIGH":         "HIGH",
        "HGHPRIC":      "HIGH",
        "HIGH PRICE":   "HIGH",
        # Low
        "LOW":          "LOW",
        "LWPRIC":       "LOW",
        "LOW PRICE":    "LOW",
        # Close — prefer official close over last traded price
        "CLOSE":        "CLOSE",
        "CLSPRIC":      "CLOSE",
        "CLOSE PRICE":  "CLOSE",
        "CLOSEPRICE":   "CLOSE",
        "CLOSE_PRICE":  "CLOSE",
        "LASTPRIC":     "CLOSE",  # fallback only
        "LAST PRICE":   "CLOSE",
        # Volume
        "TOTTRDQTY":    "VOLUME",
        "TTLTRADGVOL":  "VOLUME",
        # Date
        "TIMESTAMP":    "DATE",
        "DATE":         "DATE",
        "DATE1":        "DATE",
        "DATE2":        "DATE",
        "BIZDT":        "DATE",
        "TRADDT":       "DATE",
    }

    for csv_path in sorted((DATA_DIR / "equity").rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            # Strip whitespace and quotes, uppercase ALL column names
            df.columns = (df.columns
                          .str.strip()
                          .str.strip('"')
                          .str.strip("'")
                          .str.upper())
            # Rename every recognised alias to its standard name
            df = df.rename(columns={c: COL_ALIASES[c]
                                     for c in df.columns if c in COL_ALIASES})
            needed = ["SYMBOL","OPEN","HIGH","LOW","CLOSE"]
            if not all(c in df.columns for c in needed): continue
            if "DATE" not in df.columns:
                # Reliable fallback: filename is always YYYY-MM-DD.csv
                df["DATE"] = csv_path.stem[:10]
            keep = needed + ["DATE"]
            if "VOLUME" in df.columns: keep.append("VOLUME")
            if "SERIES" in df.columns: keep.append("SERIES")
            frames.append(df[keep])
        except Exception:
            continue

    if not frames:
        log.error("No equity data!")
        sys.exit(1)

    master = pd.concat(frames, ignore_index=True)
    master["DATE"] = pd.to_datetime(master["DATE"], errors="coerce").dt.date
    master = master.dropna(subset=["DATE"])

    if "SERIES" in master.columns:
        master = master[master["SERIES"].str.strip() == "EQ"]

    for c in ["OPEN","HIGH","LOW","CLOSE"]:
        master[c] = pd.to_numeric(master[c], errors="coerce")
    if "VOLUME" in master.columns:
        master["VOLUME"] = pd.to_numeric(master["VOLUME"], errors="coerce")

    master = master.dropna(subset=["OPEN","HIGH","LOW","CLOSE"])
    master = master.sort_values(["SYMBOL","DATE"]).reset_index(drop=True)
    log.info(f"  {len(master):,} rows | {master['SYMBOL'].nunique()} symbols | "
             f"{master['DATE'].min()} → {master['DATE'].max()}")
    return master


def load_delivery() -> pd.DataFrame:
    """Load delivery % per symbol per date."""
    frames = []
    for csv_path in sorted((DATA_DIR / "delivery").rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.upper()
            if "DELIV_PER" not in df.columns: continue
            if "DATE" not in df.columns:
                df["DATE"] = csv_path.stem[:10]
            if "NAME" in df.columns and "SYMBOL" not in df.columns:
                df = df.rename(columns={"NAME":"SYMBOL"})
            if "SYMBOL" not in df.columns: continue
            df["DATE"]     = pd.to_datetime(df["DATE"], errors="coerce").dt.date
            df["DELIV_PER"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
            frames.append(df[["DATE","SYMBOL","DELIV_PER"]].dropna())
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["DATE","SYMBOL","DELIV_PER"])
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(master: pd.DataFrame, delivery: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns needed for filtering."""
    log.info("Building features...")
    dfs = []

    for sym, grp in master.groupby("SYMBOL"):
        grp = grp.copy().sort_values("DATE").reset_index(drop=True)
        n   = len(grp)
        if n < 25: continue

        o, h, l, c = grp.OPEN, grp.HIGH, grp.LOW, grp.CLOSE

        grp["BODY"]        = (c - o).abs()
        grp["UPPER_WICK"]  = h - pd.concat([o,c],axis=1).max(axis=1)
        grp["LOWER_WICK"]  = pd.concat([o,c],axis=1).min(axis=1) - l
        grp["WICK_RATIO"]  = grp["LOWER_WICK"] / grp["BODY"].replace(0, np.nan)
        grp["UPPER_PCT"]   = grp["UPPER_WICK"]  / grp["BODY"].replace(0, np.nan)

        # Is today red (close < open)?
        grp["IS_RED"] = (c < o).astype(int)
        grp["PREV1_RED"] = grp["IS_RED"].shift(1)
        grp["PREV2_RED"] = grp["IS_RED"].shift(2)
        grp["PREV3_RED"] = grp["IS_RED"].shift(3)

        # SMAs for trend
        grp["SMA10"]  = c.rolling(10).mean()
        grp["SMA20"]  = c.rolling(20).mean()

        # Volume vs 20-day avg
        if "VOLUME" in grp.columns:
            grp["VOL_AVG20"] = grp["VOLUME"].rolling(20).mean()
            grp["VOL_RATIO"] = grp["VOLUME"] / grp["VOL_AVG20"].replace(0, np.nan)
        else:
            grp["VOL_RATIO"] = np.nan

        # Next-day return
        grp["NEXT_CLOSE"]  = c.shift(-1)
        grp["NEXT_DATE"]   = grp["DATE"].shift(-1).astype(str)   # str avoids date serialization issues
        grp["NEXT_RETURN"] = (grp["NEXT_CLOSE"] - c) / c * 100
        grp["POSITIVE"]    = grp["NEXT_RETURN"] > 0

        # Days gap check (must be actual next trading day)
        grp["NEXT_DATE_GAP"] = (
            pd.to_datetime(grp["NEXT_DATE"], errors="coerce") - pd.to_datetime(grp["DATE"].astype(str))
        ).dt.days

        dfs.append(grp)

    combined = pd.concat(dfs, ignore_index=True)

    # Merge delivery %
    if not delivery.empty:
        combined = combined.merge(
            delivery.rename(columns={"DELIV_PER":"DELIV_PCT"}),
            on=["DATE","SYMBOL"], how="left"
        )
    else:
        combined["DELIV_PCT"] = np.nan

    log.info(f"  Features built: {len(combined):,} rows")
    return combined


# ─────────────────────────────────────────────
# SIGNAL DETECTION WITH FILTERS
# ─────────────────────────────────────────────

def detect_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply all filters and return matching rows (excluding last row = no next day)."""
    mask = pd.Series(True, index=df.index)

    # Base pattern
    mask &= df["BODY"] > 0
    mask &= df["WICK_RATIO"]  >= params["wick_ratio"]
    mask &= df["UPPER_PCT"]   <= params["upper_wick_pct"]

    # Min price filter (removes penny stocks)
    mask &= df["CLOSE"] >= params["min_price"]

    # Previous red days
    if params["min_prev_red"] >= 1: mask &= df["PREV1_RED"] == 1
    if params["min_prev_red"] >= 2: mask &= df["PREV2_RED"] == 1
    if params["min_prev_red"] >= 3: mask &= df["PREV3_RED"] == 1

    # Trend filter: stock must be in downtrend (close < SMA)
    if params["trend_filter"] == 10:
        mask &= df["CLOSE"] < df["SMA10"]
    elif params["trend_filter"] == 20:
        mask &= df["CLOSE"] < df["SMA20"]

    # Volume filter
    if params["volume_filter"]:
        mask &= df["VOL_RATIO"] >= params["volume_filter"]

    # Delivery filter
    if params["delivery_filter"]:
        mask &= df["DELIV_PCT"] >= params["delivery_filter"]

    # Must have valid next day (not too far gap)
    mask &= df["NEXT_DATE_GAP"].between(1, 5)
    mask &= df["NEXT_CLOSE"].notna()

    return df[mask].copy()


# ─────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────

PARAM_GRID = {
    "wick_ratio":      [2.0, 2.5, 3.0, 4.0, 5.0],
    "upper_wick_pct":  [0.25, 0.15, 0.10, 0.05],
    "min_prev_red":    [0, 1, 2, 3],
    "trend_filter":    [False, 10, 20],
    "volume_filter":   [False, 1.5, 2.0, 2.5],
    "delivery_filter": [False, 40, 50, 60],
    "min_price":       [50, 100],
}

MIN_SIGNALS = 20   # ignore combos with fewer than this many signals


def run_optimizer(features: pd.DataFrame) -> list:
    log.info("Running optimizer (this takes a few minutes)...")

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))
    total  = len(combos)
    log.info(f"  Testing {total:,} parameter combinations...")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        sigs   = detect_signals(features, params)
        n      = len(sigs)
        if n < MIN_SIGNALS:
            continue
        wins    = sigs["POSITIVE"].sum()
        wr      = round(wins / n * 100, 1)
        avg_ret = round(sigs["NEXT_RETURN"].mean(), 2)
        results.append({
            **params,
            "total_signals": int(n),
            "wins":          int(wins),
            "win_rate":      wr,
            "avg_return":    avg_ret,
        })
        if (i+1) % 500 == 0:
            log.info(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

    results.sort(key=lambda x: (x["win_rate"], x["avg_return"]), reverse=True)
    log.info(f"  Combos with >={MIN_SIGNALS} signals: {len(results)}")
    if results:
        best = results[0]
        log.info(f"  BEST: win_rate={best['win_rate']}% | "
                 f"signals={best['total_signals']} | "
                 f"avg_return={best['avg_return']}%")
        log.info(f"  BEST params: {best}")
    return results


# ─────────────────────────────────────────────
# FULL BACKTEST STATS FOR BEST PARAMS
# ─────────────────────────────────────────────

def full_backtest(features: pd.DataFrame, params: dict) -> dict:
    sigs = detect_signals(features, params)
    if sigs.empty:
        return {}

    total    = len(sigs)
    wins     = int(sigs["POSITIVE"].sum())
    wr       = round(wins / total * 100, 1)
    avg_ret  = round(sigs["NEXT_RETURN"].mean(), 2)
    avg_win  = round(sigs[sigs["POSITIVE"]]["NEXT_RETURN"].mean(), 2)
    avg_loss = round(sigs[~sigs["POSITIVE"]]["NEXT_RETURN"].mean(), 2)

    sigs["year"] = pd.to_datetime(sigs["DATE"]).dt.year
    by_year = (
        sigs.groupby("year")
        .agg(total=("POSITIVE","count"), wins=("POSITIVE","sum"),
             avg_return=("NEXT_RETURN","mean"))
        .round(2).reset_index()
    )
    by_year["win_rate"] = (by_year["wins"]/by_year["total"]*100).round(1)

    top_gains  = (sigs.nlargest(10,"NEXT_RETURN")
                  [["SYMBOL","DATE","NEXT_RETURN","WICK_RATIO","BODY"]]
                  .rename(columns={"SYMBOL":"symbol","DATE":"date",
                                   "NEXT_RETURN":"next_return",
                                   "WICK_RATIO":"wick_ratio","BODY":"body"})
                  .assign(color=lambda d: "green")
                  .to_dict("records"))

    top_losses = (sigs.nsmallest(10,"NEXT_RETURN")
                  [["SYMBOL","DATE","NEXT_RETURN","WICK_RATIO","BODY"]]
                  .rename(columns={"SYMBOL":"symbol","DATE":"date",
                                   "NEXT_RETURN":"next_return",
                                   "WICK_RATIO":"wick_ratio","BODY":"body"})
                  .assign(color=lambda d: "red")
                  .to_dict("records"))

    # History list
    history = []
    for _, r in sigs.iterrows():
        history.append({
            "symbol":      str(r["SYMBOL"]),
            "date":        str(r["DATE"]),
            "open":        round(float(r["OPEN"]),  2),
            "high":        round(float(r["HIGH"]),  2),
            "low":         round(float(r["LOW"]),   2),
            "close":       round(float(r["CLOSE"]), 2),
            "body":        round(float(r["BODY"]),  2),
            "lower_wick":  round(float(r["LOWER_WICK"]), 2),
            "upper_wick":  round(float(r["UPPER_WICK"]), 2),
            "wick_ratio":  round(float(r["WICK_RATIO"]), 2),
            "color":       "green" if r["CLOSE"] >= r["OPEN"] else "red",
            "next_date":   str(r["NEXT_DATE"]) if pd.notna(r["NEXT_DATE"]) else "",
            "next_close":  round(float(r["NEXT_CLOSE"]), 2) if pd.notna(r["NEXT_CLOSE"]) else 0,
            "next_return": round(float(r["NEXT_RETURN"]), 2),
            "positive":    bool(r["POSITIVE"]),
        })

    return {
        "stats": {
            "total_signals":    total,
            "positive_signals": wins,
            "win_rate_pct":     wr,
            "avg_next_return":  avg_ret,
            "avg_win_return":   avg_win,
            "avg_loss_return":  avg_loss,
            "max_gain_pct":     round(float(sigs["NEXT_RETURN"].max()), 2),
            "max_loss_pct":     round(float(sigs["NEXT_RETURN"].min()), 2),
            "by_year":          by_year.to_dict("records"),
            "top_gains":        top_gains,
            "top_losses":       top_losses,
            "best_params":      {k:v for k,v in params.items()
                                 if k not in ("min_signals",)},
            "generated_at":     datetime.utcnow().isoformat() + "Z",
        },
        "history": history,
    }


# ─────────────────────────────────────────────
# TODAY'S LIVE ALERTS
# ─────────────────────────────────────────────

def todays_alerts(features: pd.DataFrame, params: dict) -> dict:
    latest = features["DATE"].max()
    today  = features[features["DATE"] == latest].copy()

    # For live alerts we don't require next_close (it's future)
    mask = pd.Series(True, index=today.index)
    mask &= today["BODY"] > 0
    mask &= today["WICK_RATIO"]  >= params["wick_ratio"]
    mask &= today["UPPER_PCT"]   <= params["upper_wick_pct"]
    mask &= today["CLOSE"]       >= params["min_price"]
    if params["min_prev_red"] >= 1: mask &= today["PREV1_RED"] == 1
    if params["min_prev_red"] >= 2: mask &= today["PREV2_RED"] == 1
    if params["min_prev_red"] >= 3: mask &= today["PREV3_RED"] == 1
    if params["trend_filter"] == 10:  mask &= today["CLOSE"] < today["SMA10"]
    if params["trend_filter"] == 20:  mask &= today["CLOSE"] < today["SMA20"]
    if params["volume_filter"]:       mask &= today["VOL_RATIO"] >= params["volume_filter"]
    if params["delivery_filter"]:     mask &= today["DELIV_PCT"] >= params["delivery_filter"]

    alerts_df = today[mask]
    alerts = []
    for _, r in alerts_df.iterrows():
        alerts.append({
            "symbol":      str(r["SYMBOL"]),
            "date":        str(latest),
            "open":        round(float(r["OPEN"]),  2),
            "high":        round(float(r["HIGH"]),  2),
            "low":         round(float(r["LOW"]),   2),
            "close":       round(float(r["CLOSE"]), 2),
            "body":        round(float(r["BODY"]),  2),
            "lower_wick":  round(float(r["LOWER_WICK"]), 2),
            "upper_wick":  round(float(r["UPPER_WICK"]), 2),
            "wick_ratio":  round(float(r["WICK_RATIO"]), 2),
            "color":       "green" if r["CLOSE"] >= r["OPEN"] else "red",
            "vol_ratio":   round(float(r["VOL_RATIO"]), 2) if pd.notna(r.get("VOL_RATIO")) else None,
            "deliv_pct":   round(float(r["DELIV_PCT"]), 1) if pd.notna(r.get("DELIV_PCT")) else None,
        })

    log.info(f"  Today's signals ({latest}): {len(alerts)}")
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "signal_date":  str(latest),
        "count":        len(alerts),
        "alerts":       alerts,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    master   = load_equity()
    delivery = load_delivery()
    features = build_features(master, delivery)

    # Run optimizer
    results  = run_optimizer(features)

    # Save full optimizer rankings
    (OUT_DIR / "optimizer_results.json").write_text(
        json.dumps(results[:200], indent=2, cls=SafeEncoder)   # top 200
    )
    log.info(f"  Optimizer results saved ({len(results)} combos)")

    if not results:
        log.error("No valid parameter combinations found!")
        sys.exit(1)

    # Use best params for everything else
    best_params = {k: results[0][k] for k in PARAM_GRID}

    # Full backtest with best params
    bt = full_backtest(features, best_params)
    (OUT_DIR / "pinbar_backtest.json").write_text(
        json.dumps(bt["stats"], indent=2, cls=SafeEncoder)
    )
    (OUT_DIR / "pinbar_history.json").write_text(
        json.dumps(bt["history"], indent=2, cls=SafeEncoder)
    )

    # Today's alerts
    alerts = todays_alerts(features, best_params)
    (OUT_DIR / "pinbar_signals.json").write_text(
        json.dumps(alerts, indent=2, cls=SafeEncoder)
    )

    log.info("\n=== FINAL SUMMARY ===")
    log.info(f"  Best win rate  : {results[0]['win_rate']}%")
    log.info(f"  Total signals  : {results[0]['total_signals']}")
    log.info(f"  Avg return     : {results[0]['avg_return']}%")
    log.info(f"  Best params    : {best_params}")
    log.info(f"  Today alerts   : {alerts['count']}")


if __name__ == "__main__":
    main()
