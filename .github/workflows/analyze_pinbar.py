#!/usr/bin/env python3
"""
Pin Bar (Hammer) Pattern Analyzer & Backtester
================================================
Pattern definition:
  - Lower wick  >= 2 × body  (|close - open|)
  - Upper wick  <= 0.25 × body
  - Body > 0 (not a doji)

For each signal:
  - Next-day return = (next_close - signal_close) / signal_close × 100
  - Positive = next_day return > 0

Outputs:
  data/analysis/pinbar_backtest.json   ← historical win-rate stats
  data/analysis/pinbar_signals.json    ← today's signals (live alerts)
  data/analysis/pinbar_history.json    ← all past signals with outcomes
"""

import json
import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
OUT_DIR   = DATA_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pinbar")

# ─────────────────────────────────────────────
# PATTERN DETECTION
# ─────────────────────────────────────────────

def is_pin_bar(row: pd.Series) -> bool:
    """
    Returns True if the candle matches the hammer/pin-bar pattern:
      lower_wick >= 2 × body   AND   upper_wick <= 0.25 × body
    """
    o, h, l, c = row["OPEN"], row["HIGH"], row["LOW"], row["CLOSE"]

    body       = abs(c - o)
    if body == 0:
        return False   # doji — skip

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    return (lower_wick >= 2 * body) and (upper_wick <= 0.25 * body)


def candle_color(row: pd.Series) -> str:
    return "green" if row["CLOSE"] >= row["OPEN"] else "red"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_all_equity() -> pd.DataFrame:
    """
    Load all equity CSV files into a single DataFrame.
    Columns kept: DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY
    """
    log.info("Loading all equity CSV files...")
    frames = []
    eq_dir = DATA_DIR / "equity"

    for csv_path in sorted(eq_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            # Normalise column names across NSE format versions
            col_map = {}
            for col in df.columns:
                cu = col.upper().strip()
                if cu in ("TIMESTAMP", "DATE", "DATE2"):
                    col_map[col] = "DATE"
                elif cu == "TOTTRDQTY":
                    col_map[col] = "VOLUME"
                elif cu == "TOTTRDVAL":
                    col_map[col] = "TURNOVER"
            df = df.rename(columns=col_map)

            needed = ["SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"]
            if not all(c in df.columns for c in needed):
                continue

            # Add date from filename if not in CSV
            if "DATE" not in df.columns:
                date_str = csv_path.stem[:10]   # YYYY-MM-DD
                df["DATE"] = date_str

            df = df[["DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"]
                    + (["VOLUME"] if "VOLUME" in df.columns else [])
                    + (["SERIES"] if "SERIES" in df.columns else [])]

            frames.append(df)
        except Exception as e:
            log.warning(f"  Skip {csv_path.name}: {e}")

    if not frames:
        log.error("No equity data found!")
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True)
    master["DATE"] = pd.to_datetime(master["DATE"], errors="coerce").dt.date
    master = master.dropna(subset=["DATE"])

    # Keep EQ series only if column exists
    if "SERIES" in master.columns:
        master = master[master["SERIES"].str.strip() == "EQ"]

    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    master = master.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    master = master.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

    log.info(f"  Loaded {len(master):,} rows, "
             f"{master['SYMBOL'].nunique()} symbols, "
             f"{master['DATE'].min()} → {master['DATE'].max()}")
    return master


# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────

def run_backtest(master: pd.DataFrame) -> dict:
    """
    For every symbol, scan all historical dates for pin-bar signals.
    Record the next-day outcome (positive / negative).
    """
    log.info("Running backtest...")

    signals = []
    symbols = master["SYMBOL"].unique()

    for sym in symbols:
        df = master[master["SYMBOL"] == sym].copy().reset_index(drop=True)
        if len(df) < 2:
            continue

        for i in range(len(df) - 1):   # -1 because we need next day
            row      = df.iloc[i]
            next_row = df.iloc[i + 1]

            if not is_pin_bar(row):
                continue

            # Next-day must be the actual next trading day (within 5 cal days)
            days_gap = (next_row["DATE"] - row["DATE"]).days
            if days_gap > 5:
                continue   # data gap, skip

            next_ret = (next_row["CLOSE"] - row["CLOSE"]) / row["CLOSE"] * 100
            body     = abs(row["CLOSE"] - row["OPEN"])
            upper_w  = row["HIGH"] - max(row["OPEN"], row["CLOSE"])
            lower_w  = min(row["OPEN"], row["CLOSE"]) - row["LOW"]

            signals.append({
                "symbol":       sym,
                "date":         str(row["DATE"]),
                "open":         round(float(row["OPEN"]),  2),
                "high":         round(float(row["HIGH"]),  2),
                "low":          round(float(row["LOW"]),   2),
                "close":        round(float(row["CLOSE"]), 2),
                "body":         round(float(body),    2),
                "lower_wick":   round(float(lower_w), 2),
                "upper_wick":   round(float(upper_w), 2),
                "wick_ratio":   round(float(lower_w / body), 2),
                "color":        candle_color(row),
                "next_date":    str(next_row["DATE"]),
                "next_close":   round(float(next_row["CLOSE"]), 2),
                "next_return":  round(float(next_ret), 2),
                "positive":     bool(next_ret > 0),
            })

    if not signals:
        log.warning("No pin-bar signals found in historical data.")
        return {"signals": [], "stats": {}}

    df_sig = pd.DataFrame(signals)

    # ── Overall stats ──────────────────────────────────────────
    total       = len(df_sig)
    positive    = df_sig["positive"].sum()
    win_rate    = round(positive / total * 100, 1)
    avg_ret     = round(df_sig["next_return"].mean(), 2)
    avg_ret_win = round(df_sig[df_sig["positive"]]["next_return"].mean(), 2)
    avg_ret_los = round(df_sig[~df_sig["positive"]]["next_return"].mean(), 2)
    max_gain    = round(df_sig["next_return"].max(), 2)
    max_loss    = round(df_sig["next_return"].min(), 2)

    # ── By year ───────────────────────────────────────────────
    df_sig["year"] = pd.to_datetime(df_sig["date"]).dt.year
    by_year = (
        df_sig.groupby("year")
        .agg(
            total=("positive", "count"),
            wins=("positive", "sum"),
            avg_return=("next_return", "mean"),
        )
        .round(2)
        .reset_index()
    )
    by_year["win_rate"] = (by_year["wins"] / by_year["total"] * 100).round(1)

    # ── Top performing signals ─────────────────────────────────
    top_gains = (
        df_sig.nlargest(10, "next_return")
        [["symbol","date","next_return","wick_ratio","color"]]
        .to_dict("records")
    )
    top_losses = (
        df_sig.nsmallest(10, "next_return")
        [["symbol","date","next_return","wick_ratio","color"]]
        .to_dict("records")
    )

    stats = {
        "total_signals":   int(total),
        "positive_signals":int(positive),
        "win_rate_pct":    win_rate,
        "avg_next_return": avg_ret,
        "avg_win_return":  avg_ret_win,
        "avg_loss_return": avg_ret_los,
        "max_gain_pct":    max_gain,
        "max_loss_pct":    max_loss,
        "by_year":         by_year.to_dict("records"),
        "top_gains":       top_gains,
        "top_losses":      top_losses,
        "generated_at":    datetime.utcnow().isoformat() + "Z",
    }

    log.info(f"  Signals: {total}, Win rate: {win_rate}%, Avg return: {avg_ret}%")
    return {"stats": stats, "signals": signals}


# ─────────────────────────────────────────────
# TODAY'S LIVE ALERTS
# ─────────────────────────────────────────────

def find_todays_signals(master: pd.DataFrame) -> list:
    """
    Find pin-bar signals that formed on the latest available trading date.
    These are stocks to watch for next-day positive move.
    """
    if master.empty:
        return []

    latest_date = master["DATE"].max()
    log.info(f"Scanning for signals on latest date: {latest_date}")

    today_df = master[master["DATE"] == latest_date]
    alerts   = []

    for _, row in today_df.iterrows():
        if not is_pin_bar(row):
            continue

        body    = abs(row["CLOSE"] - row["OPEN"])
        upper_w = row["HIGH"] - max(row["OPEN"], row["CLOSE"])
        lower_w = min(row["OPEN"], row["CLOSE"]) - row["LOW"]

        alerts.append({
            "symbol":     row["SYMBOL"],
            "date":       str(latest_date),
            "open":       round(float(row["OPEN"]),  2),
            "high":       round(float(row["HIGH"]),  2),
            "low":        round(float(row["LOW"]),   2),
            "close":      round(float(row["CLOSE"]), 2),
            "body":       round(float(body),    2),
            "lower_wick": round(float(lower_w), 2),
            "upper_wick": round(float(upper_w), 2),
            "wick_ratio": round(float(lower_w / body), 2),
            "color":      candle_color(row),
        })

    log.info(f"  Found {len(alerts)} pin-bar signals today")
    return alerts


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    master = load_all_equity()
    if master.empty:
        sys.exit(1)

    # ── Backtest ───────────────────────────────────────────────
    result  = run_backtest(master)
    stats   = result.get("stats", {})
    signals = result.get("signals", [])

    # Save full history
    history_path = OUT_DIR / "pinbar_history.json"
    with open(history_path, "w") as f:
        json.dump(signals, f, indent=2)
    log.info(f"  History saved: {len(signals)} signals")

    # Save backtest summary
    bt_path = OUT_DIR / "pinbar_backtest.json"
    with open(bt_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Backtest saved → {bt_path.relative_to(BASE_DIR)}")

    # ── Today's alerts ─────────────────────────────────────────
    alerts = find_todays_signals(master)
    alerts_path = OUT_DIR / "pinbar_signals.json"
    with open(alerts_path, "w") as f:
        json.dump({
            "generated_at":  datetime.utcnow().isoformat() + "Z",
            "signal_date":   str(master["DATE"].max()),
            "count":         len(alerts),
            "alerts":        alerts,
        }, f, indent=2)
    log.info(f"  Alerts saved: {len(alerts)} signals → {alerts_path.relative_to(BASE_DIR)}")

    log.info("Done.")


if __name__ == "__main__":
    main()
