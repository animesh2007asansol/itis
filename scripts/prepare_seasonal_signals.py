#!/usr/bin/env python3
"""
prepare_seasonal_signals.py  v3
================================
Reads NSE equity bhav copy CSVs — handles BOTH format variants:

  OLD (pre mid-2024):  SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,...
  NEW (mid-2024+):     SYMBOL,SERIES,DATE1,PREV CLOSE,OPEN PRICE,HIGH PRICE,
                       LOW PRICE,LAST PRICE,CLOSE PRICE,AVG PRICE,...

Outputs (seasonal_signals/ only — raw data/ is NEVER touched):
  latest.json    ← most recent trading day prices
  prev.json      ← previous trading day prices
  backtest.json  ← full 5-year backtest from actual NSE closes

Stop-loss detection: scans every daily close between buy and sell date.
"""

import json, os, sys, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "seasonal_signals"
MANIFEST  = DATA_DIR / "manifest.json"

START_CAPITAL = 100_000

# ---------------------------------------------------------------------------
# Trade definitions
# ---------------------------------------------------------------------------
TRADE_DEFS = [
    # FY22
    dict(fy="FY22", season="AC + Power",          stock="Voltas",
         sym="VOLTAS",     buy_date="2021-03-01", sell_date="2021-05-24",
         dep_amt=30000, sl_pct=12,
         note="Hot summer 2021. Voltas AC surge."),
    dict(fy="FY22", season="AC + Power",          stock="Coal India",
         sym="COALINDIA",  buy_date="2021-03-01", sell_date="2021-05-31",
         dep_amt=25000, sl_pct=12,
         note="Peak summer power grid demand."),
    dict(fy="FY22", season="Jewellery Pre-AT",    stock="Titan",
         sym="TITAN",      buy_date="2021-03-15", sell_date="2021-05-14",
         dep_amt=35000, sl_pct=10,
         note="Akshaya Tritiya May 14 2021."),
    dict(fy="FY22", season="Sugar + Paint",       stock="Balrampur Chini",
         sym="BALRAMCHIN", buy_date="2021-09-15", sell_date="2022-02-28",
         dep_amt=30000, sl_pct=12,
         note="Ethanol push FY22. Strong margins."),
    dict(fy="FY22", season="Sugar + Paint",       stock="Asian Paints",
         sym="ASIANPAINT", buy_date="2021-09-15", sell_date="2021-12-15",
         dep_amt=22000, sl_pct=12,
         note="Post-monsoon decorative demand spike."),
    dict(fy="FY22", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2021-12-15", sell_date="2022-02-01",
         dep_amt=22000, sl_pct=12,
         note="Budget FY22 railway capex Rs2.15L cr."),
    dict(fy="FY22", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2021-12-15", sell_date="2022-02-01",
         dep_amt=18000, sl_pct=12,
         note="IRFC rallied on railway budget euphoria."),

    # FY23
    dict(fy="FY23", season="AC + Power",          stock="Voltas",
         sym="VOLTAS",     buy_date="2022-02-15", sell_date="2022-05-31",
         dep_amt=40000, sl_pct=12,
         note="Decent but RBI rate hike May 2022 cut rally."),
    dict(fy="FY23", season="AC + Power",          stock="NTPC",
         sym="NTPC",       buy_date="2022-02-15", sell_date="2022-05-31",
         dep_amt=30000, sl_pct=12,
         note="NTPC held up — grid demand was real."),
    dict(fy="FY23", season="Kharif Fertilizer",   stock="Coromandel Intl",
         sym="COROMANDEL", buy_date="2022-04-15", sell_date="2022-06-22",
         dep_amt=35000, sl_pct=12,
         note="Russia-Ukraine spiked urea costs."),
    dict(fy="FY23", season="Kharif Fertilizer",   stock="Chambal Fert",
         sym="CHAMBLFERT", buy_date="2022-04-15", sell_date="2022-06-08",
         dep_amt=28000, sl_pct=12,
         note="Same war-driven input cost shock."),
    dict(fy="FY23", season="Sugar (Rabi)",         stock="Balrampur Chini",
         sym="BALRAMCHIN", buy_date="2022-09-15", sell_date="2023-02-28",
         dep_amt=38000, sl_pct=12,
         note="Sugar export quota Nov 2022 — catalyst worked."),
    dict(fy="FY23", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2022-12-15", sell_date="2023-02-01",
         dep_amt=28000, sl_pct=12,
         note="Budget FY23 Rs2.40L cr railway."),
    dict(fy="FY23", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2022-12-15", sell_date="2023-02-01",
         dep_amt=22000, sl_pct=12,
         note="Pre-budget railway stock rally."),

    # FY24
    dict(fy="FY24", season="AC + Power",          stock="Amber Enterprises",
         sym="AMBER",      buy_date="2023-02-15", sell_date="2023-05-31",
         dep_amt=50000, sl_pct=12,
         note="HOTTEST April in 50 years — Amber +44%."),
    dict(fy="FY24", season="AC + Power",          stock="Blue Star",
         sym="BLUESTARCO", buy_date="2023-02-15", sell_date="2023-05-31",
         dep_amt=40000, sl_pct=12,
         note="Record AC sales Q1 FY24."),
    dict(fy="FY24", season="Jewellery Pre-AT",    stock="Titan",
         sym="TITAN",      buy_date="2024-03-25", sell_date="2024-05-10",
         dep_amt=55000, sl_pct=10,
         note="Akshaya Tritiya May 10 2024."),
    dict(fy="FY24", season="Jewellery Pre-AT",    stock="Kalyan Jewellers",
         sym="KALYANKJIL", buy_date="2024-03-25", sell_date="2024-05-10",
         dep_amt=45000, sl_pct=10,
         note="AT 2024 — Kalyan +15.7%."),
    dict(fy="FY24", season="Sugar + Paint + Fert",stock="Balrampur Chini",
         sym="BALRAMCHIN", buy_date="2023-09-15", sell_date="2024-02-28",
         dep_amt=48000, sl_pct=12,
         note="Ethanol expansion FY24."),
    dict(fy="FY24", season="Sugar + Paint + Fert",stock="Asian Paints",
         sym="ASIANPAINT", buy_date="2023-08-15", sell_date="2023-12-15",
         dep_amt=38000, sl_pct=12,
         note="Post-monsoon paint season FY24."),
    dict(fy="FY24", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2023-12-15", sell_date="2024-02-01",
         dep_amt=50000, sl_pct=12,
         note="RVNL +27% in 5 weeks. Budget Rs2.62L cr."),
    dict(fy="FY24", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2023-12-15", sell_date="2024-02-01",
         dep_amt=38000, sl_pct=12,
         note="Record railway allocation — IRFC +21%."),

    # FY25
    dict(fy="FY25", season="AC + Power",          stock="Voltas",
         sym="VOLTAS",     buy_date="2024-02-15", sell_date="2024-05-10",
         dep_amt=60000, sl_pct=12,
         note="Unseasonal rains Apr-May 2024."),
    dict(fy="FY25", season="AC + Power",          stock="NTPC",
         sym="NTPC",       buy_date="2024-02-15", sell_date="2024-04-12",
         dep_amt=45000, sl_pct=12,
         note="Market correction + unseasonal rains."),
    dict(fy="FY25", season="Kharif Fertilizer",   stock="Chambal Fert",
         sym="CHAMBLFERT", buy_date="2024-04-15", sell_date="2024-07-15",
         dep_amt=55000, sl_pct=12,
         note="Good Kharif sowing season FY25."),
    dict(fy="FY25", season="Kharif Fertilizer",   stock="RCF",
         sym="RCF",        buy_date="2024-04-15", sell_date="2024-07-15",
         dep_amt=38000, sl_pct=12,
         note="NBS subsidy revival boosted RCF."),
    dict(fy="FY25", season="Jewellery Pre-Dhanteras", stock="Titan",
         sym="TITAN",      buy_date="2024-09-01", sell_date="2024-10-29",
         dep_amt=65000, sl_pct=10,
         note="Dhanteras Oct 29 2024."),
    dict(fy="FY25", season="Jewellery Pre-Dhanteras", stock="Senco Gold",
         sym="SENCO",      buy_date="2024-09-01", sell_date="2024-10-29",
         dep_amt=42000, sl_pct=10,
         note="Senco +16.3% pre-Dhanteras."),
    dict(fy="FY25", season="Sugar (Rabi)",         stock="Balrampur Chini",
         sym="BALRAMCHIN", buy_date="2024-09-15", sell_date="2025-02-28",
         dep_amt=60000, sl_pct=12,
         note="Ethanol hike Jan 2025 boosted sugar margins."),
    dict(fy="FY25", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2024-12-15", sell_date="2025-02-01",
         dep_amt=70000, sl_pct=12,
         note="Budget FY25 Rs2.52L cr railway capex."),
    dict(fy="FY25", season="Railway Pre-Budget",  stock="Jupiter Wagons",
         sym="JWL",        buy_date="2024-12-15", sell_date="2025-02-01",
         dep_amt=55000, sl_pct=12,
         note="JWL +39% after Dec 26 2024 fare revision."),

    # FY26
    dict(fy="FY26", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2025-12-15", sell_date="2026-02-01",
         dep_amt=80000, sl_pct=12,
         note="Budget FY26 Rs3.02L cr — all-time record."),
    dict(fy="FY26", season="Railway Pre-Budget",  stock="Jupiter Wagons",
         sym="JWL",        buy_date="2025-12-15", sell_date="2026-02-01",
         dep_amt=65000, sl_pct=12,
         note="JWL strong on fare revision expectations."),
    dict(fy="FY26", season="AC + Power (LIVE)",   stock="Amber Enterprises",
         sym="AMBER",      buy_date="2026-03-01", sell_date="2026-05-31",
         dep_amt=55000, sl_pct=12, live=True,
         note="IMD heatwave active. Crude $66 — full position."),
    dict(fy="FY26", season="AC + Power (LIVE)",   stock="Blue Star",
         sym="BLUESTARCO", buy_date="2026-03-01", sell_date="2026-05-31",
         dep_amt=42000, sl_pct=12, live=True,
         note="Monitor closely — near stop loss zone."),
    dict(fy="FY26", season="AC + Power (LIVE)",   stock="Coal India",
         sym="COALINDIA",  buy_date="2026-03-01", sell_date="2026-05-31",
         dep_amt=38000, sl_pct=12, live=True,
         note="Summer peak grid demand. CEA alert active."),
    dict(fy="FY26", season="Jewellery Pre-AT",    stock="Titan",
         sym="TITAN",      buy_date="2026-04-13", sell_date="2026-04-19",
         dep_amt=50000, sl_pct=10, open=True,
         note="Akshaya Tritiya Apr 19 2026. BUY WINDOW OPEN."),
    dict(fy="FY26", season="Jewellery Pre-AT",    stock="Kalyan Jewellers",
         sym="KALYANKJIL", buy_date="2026-04-13", sell_date="2026-04-19",
         dep_amt=40000, sl_pct=10, open=True,
         note="Akshaya Tritiya Apr 19 2026. BUY WINDOW OPEN."),
]

ALL_SYMS = {t["sym"] for t in TRADE_DEFS} | {
    "COROMANDEL","CHAMBLFERT","DEEPAKFERT","TITAGARH",
    "BALRAMCHIN","TRIVENI","EIDPARRY",
    "ASIANPAINT","BERGEPAINT","KANSAINER","SENCO",
}

# ---------------------------------------------------------------------------
# ROBUST CSV PARSER — handles both old and new NSE formats
# ---------------------------------------------------------------------------

# Old NSE format column names
OLD_COL_MAP = {
    "SYMBOL":   ["SYMBOL"],
    "SERIES":   ["SERIES"],
    "CLOSE":    ["CLOSE"],
}

# New NSE format (mid-2024+) — column names changed
NEW_COL_MAP = {
    "SYMBOL":   ["SYMBOL"],
    "SERIES":   ["SERIES"],
    "CLOSE":    ["CLOSE PRICE", "CLOSE_PRICE", "CLOSEPRICE"],
}

# Any of these close column aliases work
ALL_CLOSE_ALIASES = ["CLOSE", "CLOSE PRICE", "CLOSE_PRICE", "CLOSEPRICE",
                     "LAST PRICE", "LAST_PRICE"]


def find_col(hdr, aliases):
    """Return index of first matching alias in header, or -1."""
    for alias in aliases:
        if alias in hdr:
            return hdr.index(alias)
    return -1


def parse_hdr(raw_line):
    """Parse CSV header line — strip whitespace and quotes, uppercase."""
    return [h.strip().strip('"').strip("'").upper()
            for h in raw_line.split(",")]


def parse_csv_file(path, syms_wanted, label=""):
    """
    Parse one NSE equity CSV file.
    Returns (prices_dict, format_detected, diagnostic_str).
    prices_dict = {SYM: close_price}
    """
    prices  = {}
    fmt     = "unknown"
    diag    = []

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            return prices, "empty", "File is empty"

        hdr = parse_hdr(lines[0])
        diag.append(f"Header ({len(hdr)} cols): {hdr[:8]}")

        # Detect SYMBOL column
        i_sym = find_col(hdr, ["SYMBOL"])
        if i_sym < 0:
            return prices, "bad-header", f"No SYMBOL column. Header: {hdr[:10]}"

        # Detect SERIES column
        i_series = find_col(hdr, ["SERIES"])

        # Detect CLOSE column — try all known aliases
        i_close = find_col(hdr, ALL_CLOSE_ALIASES)
        if i_close < 0:
            return prices, "bad-header", f"No close price column. Header: {hdr[:12]}"

        diag.append(f"SYMBOL@{i_sym}  SERIES@{i_series}  CLOSE@{i_close}({hdr[i_close]})")

        # Detect format
        close_col_name = hdr[i_close]
        if close_col_name == "CLOSE":
            fmt = "old-format"
        else:
            fmt = f"new-format({close_col_name})"

        # Collect sample SERIES values for diagnosis
        series_seen = set()
        rows_parsed = 0
        rows_eq     = 0

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            cols = [c.strip().strip('"').strip("'")
                    for c in line.split(",")]
            if len(cols) <= max(i_sym, i_close):
                continue

            rows_parsed += 1

            # Capture series values for diagnosis
            series_val = cols[i_series].strip() if i_series >= 0 else "N/A"
            series_seen.add(series_val)

            # Filter EQ series only — also accept "BE" (trade-to-trade) for some stocks
            if i_series >= 0 and series_val not in ("EQ", "BE"):
                continue
            rows_eq += 1

            sym = cols[i_sym].strip()
            if sym not in syms_wanted:
                continue

            try:
                close_raw = cols[i_close].strip().replace(",", "")
                close = round(float(close_raw), 2)
                if close > 0:
                    prices[sym] = close
            except (ValueError, IndexError):
                pass

        diag.append(f"Rows: total={rows_parsed}  EQ/BE={rows_eq}  "
                    f"SERIES seen={sorted(series_seen)[:8]}  "
                    f"symbols found={len(prices)}")

    except Exception as e:
        return prices, "error", f"Exception: {e}\n{traceback.format_exc()}"

    return prices, fmt, " | ".join(diag)


# ---------------------------------------------------------------------------
# Price DB — uses parse_csv_file instead of ad-hoc parsing
# ---------------------------------------------------------------------------
class PriceDB:
    def __init__(self, data_dir, manifest):
        self.data_dir     = data_dir
        self.trading_days = sorted(manifest.keys())
        self._cache = {}   # date_str -> {sym: close}
        self._miss  = set()
        self._fmts  = {}   # date_str -> format string (for summary)

    def trading_days_in_range(self, start, end):
        return [d for d in self.trading_days if start <= d <= end]

    def nearest_on_or_after(self, target):
        for d in self.trading_days:
            if d >= target:
                return d
        return None

    def nearest_on_or_before(self, target):
        result = None
        for d in self.trading_days:
            if d <= target:
                result = d
            else:
                break
        return result

    def get(self, date_str, sym):
        if not date_str or date_str in self._miss:
            return None
        if date_str not in self._cache:
            self._load(date_str)
        return self._cache.get(date_str, {}).get(sym)

    def _csv_path(self, date_str):
        y, m, _ = date_str.split("-")
        return self.data_dir / "equity" / y / m / f"{date_str}.csv"

    def _load(self, date_str):
        path = self._csv_path(date_str)
        if not path.exists():
            self._miss.add(date_str)
            return
        prices, fmt, _ = parse_csv_file(path, ALL_SYMS, date_str)
        self._fmts[date_str] = fmt
        if prices:
            self._cache[date_str] = prices
        else:
            # Even if 0 seasonal symbols found, cache empty dict so we don't reload
            self._cache[date_str] = {}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def find_price(db, date_target, sym, direction="after", max_delta=7):
    if direction == "after":
        d = db.nearest_on_or_after(date_target)
    else:
        d = db.nearest_on_or_before(date_target)

    if d:
        px = db.get(d, sym)
        if px:
            return d, px

    for delta in range(1, max_delta + 1):
        for sign in ([-1, 1] if direction == "before" else [1, -1]):
            try:
                d2 = (datetime.strptime(date_target, "%Y-%m-%d")
                      + timedelta(days=delta * sign)).strftime("%Y-%m-%d")
            except Exception:
                continue
            if direction == "after":
                cand = db.nearest_on_or_after(d2)
            else:
                cand = db.nearest_on_or_before(d2)
            if cand:
                px = db.get(cand, sym)
                if px:
                    return cand, px
    return None, None


def compute_trade(td, db, latest_prices):
    sym        = td["sym"]
    buy_target = td["buy_date"]
    sell_target= td["sell_date"]
    sl_pct     = td.get("sl_pct", 12)
    dep_amt    = td["dep_amt"]
    is_live    = td.get("live", False)
    is_open    = td.get("open", False)

    r = dict(
        fy=td["fy"], season=td["season"], stock=td["stock"], sym=sym,
        buy_date_target=buy_target, sell_date_target=sell_target,
        dep_amt=dep_amt, sl_pct=sl_pct,
        live=is_live, open=is_open, note=td.get("note",""),
        data_ok=False, data_note="",
        buy_date_actual=None, buy_px=None,
        sell_date_actual=None, sell_px=None, sl_px=None,
        stop_hit=False, stop_hit_date=None, stop_hit_px=None,
        ret_pct=None, pnl=None, win=None, verdict="UNKNOWN",
    )

    # OPEN trade
    if is_open:
        cur = latest_prices.get(sym)
        if cur:
            r.update(buy_px=cur, buy_date_actual="latest",
                     sell_date_actual=sell_target, sl_px=round(cur*(1-sl_pct/100),2),
                     data_ok=True, data_note="Entry est = latest NSE close",
                     verdict="OPEN — enter now", sell_px=cur,
                     ret_pct=0.0, pnl=0)
        else:
            r["verdict"] = "OPEN — price unavailable"
        return r

    # Buy price
    buy_d, buy_px = find_price(db, buy_target, sym, "after")
    if not buy_px:
        r["data_note"] = f"Buy price not found near {buy_target}"
        r["verdict"]   = "DATA MISSING"
        return r

    r["buy_date_actual"] = buy_d
    r["buy_px"]          = buy_px
    sl_price             = round(buy_px * (1 - sl_pct / 100), 2)
    r["sl_px"]           = sl_price

    # Scan for stop loss
    today_str = datetime.now().strftime("%Y-%m-%d")
    scan_end  = today_str if is_live else sell_target
    for d in db.trading_days_in_range(buy_d, scan_end):
        px = db.get(d, sym)
        if px and px <= sl_price:
            r.update(stop_hit=True, stop_hit_date=d, stop_hit_px=px,
                     sell_date_actual=d, sell_px=sl_price,
                     ret_pct=round((sl_price-buy_px)/buy_px*100,2),
                     pnl=round(dep_amt*(sl_price-buy_px)/buy_px),
                     win=False, data_ok=True,
                     data_note=f"Stop: close Rs{px} <= SL Rs{sl_price} on {d}",
                     verdict=f"STOP LOSS HIT ({d}) — actual NSE data")
            return r

    # LIVE trade
    if is_live:
        latest_d = db.nearest_on_or_before(today_str)
        cur_px   = db.get(latest_d, sym) if latest_d else None
        if cur_px is None:
            cur_px = latest_prices.get(sym)
            latest_d = "latest.json"
        if cur_px:
            ret = round((cur_px - buy_px) / buy_px * 100, 2)
            r.update(sell_date_actual=latest_d, sell_px=cur_px,
                     ret_pct=ret, pnl=round(dep_amt*ret/100),
                     win=None, data_ok=True,
                     data_note=f"Buy: {buy_d} NSE close; Current: {latest_d}",
                     verdict=f"LIVE {'UP' if ret>=0 else 'DOWN'} {ret:+.1f}% (SL not hit)")
        else:
            r["verdict"] = "LIVE — current price unavailable"
        return r

    # Sell price
    sell_d, sell_px = find_price(db, sell_target, sym, "before")
    if not sell_px:
        r["data_note"] = f"Sell price not found near {sell_target}"
        r["verdict"]   = "DATA MISSING — sell date CSV missing"
        return r

    ret = round((sell_px - buy_px) / buy_px * 100, 2)
    r.update(sell_date_actual=sell_d, sell_px=sell_px,
             ret_pct=ret, pnl=round(dep_amt*ret/100),
             win=ret > 0, data_ok=True,
             data_note=f"Buy: {buy_d} | Sell: {sell_d} | both NSE closes",
             verdict=f"{'WIN' if ret>0 else 'LOSS'} {ret:+.1f}% — actual NSE data")
    return r


def compute_year_summary(trades, start_capital):
    summary = []
    capital = start_capital
    for fy in ["FY22","FY23","FY24","FY25","FY26"]:
        fy_t  = [t for t in trades if t["fy"] == fy]
        pnl   = sum(t["pnl"] for t in fy_t if t.get("pnl") is not None)
        wins  = sum(1 for t in fy_t if t.get("win") is True)
        losses= sum(1 for t in fy_t if t.get("win") is False)
        stops = sum(1 for t in fy_t if t.get("stop_hit"))
        partial= any(t.get("live") or t.get("open") for t in fy_t)
        end_c = round(capital + pnl)
        ret_p = round(pnl / capital * 100, 1) if capital else 0
        closed= [t for t in fy_t if t.get("ret_pct") is not None
                 and not t.get("live") and not t.get("open")]
        best  = max(closed, key=lambda t: t.get("ret_pct") or -999) if closed else None
        worst = min(closed, key=lambda t: t.get("ret_pct") or  999) if closed else None
        parts = []
        if stops:     parts.append(f"{stops} stop loss(es)")
        if best  and best.get("ret_pct",0)  > 0: parts.append(f"Best: {best['stock']} {best['ret_pct']:+.1f}%")
        if worst and worst.get("ret_pct",0) < 0: parts.append(f"Worst: {worst['stock']} {worst['ret_pct']:+.1f}%")
        if partial: parts.append("FY still open")
        summary.append(dict(
            fy=fy, start_capital=round(capital), pnl=round(pnl),
            ret_pct=ret_p, end_capital=end_c,
            wins=wins, losses=losses, stops=stops,
            partial=partial, note=". ".join(parts) or f"{wins}W {losses}L",
        ))
        capital = end_c
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SeasonalSignals v3 — NSE Historical Backtest Engine")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Manifest
    print("\n[1] Loading manifest...")
    if not MANIFEST.exists():
        print(f"  ERROR: manifest.json not at {MANIFEST}")
        sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    asc  = sorted(manifest.keys())
    desc = list(reversed(asc))
    print(f"  OK: {len(asc)} trading days [{asc[0]} -> {asc[-1]}]")

    # 2. Find latest/prev CSV files
    print("\n[2] Finding latest equity CSVs...")
    found_dates = []
    for d in desc[:20]:
        y, m, _ = d.split("-")
        p = DATA_DIR / "equity" / y / m / f"{d}.csv"
        if p.exists():
            found_dates.append(d)
            if len(found_dates) == 2:
                break
    if not found_dates:
        print("  ERROR: No equity CSVs in recent 20 dates")
        sys.exit(1)
    latest_date = found_dates[0]
    prev_date   = found_dates[1] if len(found_dates) > 1 else None
    print(f"  latest: {latest_date}")
    print(f"  prev:   {prev_date or 'none'}")

    # 3. Parse latest CSV with full diagnostics
    print("\n[3] Parsing latest CSV (with format diagnostics)...")
    latest_path = DATA_DIR / "equity" / latest_date[:4] / latest_date[5:7] / f"{latest_date}.csv"
    latest_raw, latest_fmt, latest_diag = parse_csv_file(latest_path, ALL_SYMS, "latest")
    print(f"  Format: {latest_fmt}")
    print(f"  Diag:   {latest_diag}")
    print(f"  Symbols found: {len(latest_raw)} -> {sorted(latest_raw.keys())}")

    prev_raw = {}
    if prev_date:
        prev_path = DATA_DIR / "equity" / prev_date[:4] / prev_date[5:7] / f"{prev_date}.csv"
        prev_raw, prev_fmt, prev_diag = parse_csv_file(prev_path, ALL_SYMS, "prev")
        print(f"  prev format: {prev_fmt} | symbols: {len(prev_raw)}")

    # 4. Backtest
    print("\n[4] Running backtest...")
    db     = PriceDB(DATA_DIR, manifest)
    trades = [compute_trade(td, db, latest_raw) for td in TRADE_DEFS]

    # Count unique formats seen
    fmts_used = {}
    for fmt in db._fmts.values():
        fmts_used[fmt] = fmts_used.get(fmt, 0) + 1
    print(f"  CSV formats used by PriceDB: {fmts_used}")

    # Print results
    print(f"\n  {'':2} {'Sym':<14} {'FY':5} {'Buy Date':11} {'Buy Rs':9} "
          f"{'Sell Date':11} {'Sell Rs':9} {'Ret%':7}  Result")
    print(f"  {'':2} {'-'*14} {'-'*5} {'-'*11} {'-'*9} {'-'*11} {'-'*9} {'-'*7}  {'-'*30}")
    for t in trades:
        icon = ("OK" if t.get("win") is True else "SL" if t.get("stop_hit")
                else "LV" if t.get("live") else "OP" if t.get("open") else "??")
        bp   = f"{t['buy_px']:9.2f}"  if t.get("buy_px")  else "       ——"
        sp   = f"{t['sell_px']:9.2f}" if t.get("sell_px") else "       ——"
        bd   = (t.get("buy_date_actual")  or "——")[-10:]
        sd   = (t.get("sell_date_actual") or "——")[-10:]
        rp   = f"{t['ret_pct']:+7.2f}%" if t.get("ret_pct") is not None else "      ——"
        vrd  = (t.get("verdict") or "")[:35]
        print(f"  {icon} {t['sym']:<14} {t['fy']:5} {bd:11} {bp} {sd:11} {sp} {rp}  {vrd}")

    # 5. Year summary
    print("\n[5] Year summary...")
    yr_sum = compute_year_summary(trades, START_CAPITAL)
    for y in yr_sum:
        pnl_s = f"+Rs{y['pnl']:,}" if y['pnl'] >= 0 else f"-Rs{abs(y['pnl']):,}"
        print(f"  {y['fy']}  Rs{y['start_capital']:>9,} -> Rs{y['end_capital']:>9,}"
              f"  {y['ret_pct']:>+6.1f}%  {pnl_s:>14}  {y['wins']}W {y['losses']}L  {y['note'][:50]}")

    all_closed = [t for t in trades if not t.get("live") and not t.get("open")
                  and t.get("ret_pct") is not None]
    wins_n    = sum(1 for t in all_closed if t.get("win"))
    hit_rate  = round(wins_n / len(all_closed) * 100, 1) if all_closed else 0
    total_ret = round((yr_sum[-1]["end_capital"] - START_CAPITAL) / START_CAPITAL * 100, 1)
    cagr      = round(((yr_sum[-1]["end_capital"] / START_CAPITAL) ** (1/5) - 1) * 100, 1) if START_CAPITAL else 0
    print(f"\n  Total return: {total_ret:+.1f}%  |  5-yr CAGR: {cagr:+.1f}%  |  "
          f"Win rate: {hit_rate:.0f}% ({wins_n}/{len(all_closed)})")

    # 6. Write JSON files
    ist     = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")
    print("\n[6] Writing output files to seasonal_signals/...")

    # latest.json
    latest_rich = {}
    for sym, close in latest_raw.items():
        entry = {"close": close}
        if sym in prev_raw:
            pc = prev_raw[sym]
            entry["prev_close"] = pc
            entry["chg_pct"]    = round((close - pc) / pc * 100, 2)
        latest_rich[sym] = entry
    with open(OUT_DIR / "latest.json", "w") as f:
        json.dump(dict(label="latest", date=latest_date, generated_at=now_ist,
                       trading_day=True, symbol_count=len(latest_rich),
                       prices=latest_rich,
                       csv_format=latest_fmt,
                       source="NSE CM equity bhav copy — EQ/BE series",
                       note="NSE official closing prices. Updated nightly after 11 PM IST."), f, indent=2)
    print(f"  OK latest.json  ({latest_date}, {len(latest_rich)} symbols, format={latest_fmt})")

    if prev_date:
        prev_rich = {sym: {"close": c} for sym, c in prev_raw.items()}
        with open(OUT_DIR / "prev.json", "w") as f:
            json.dump(dict(label="prev", date=prev_date, generated_at=now_ist,
                           trading_day=True, symbol_count=len(prev_rich),
                           prices=prev_rich, csv_format=prev_fmt,
                           source="NSE CM equity bhav copy — EQ/BE series"), f, indent=2)
        print(f"  OK prev.json    ({prev_date}, {len(prev_rich)} symbols, format={prev_fmt})")

    # backtest.json
    with open(OUT_DIR / "backtest.json", "w") as f:
        json.dump(dict(
            generated_at=now_ist,
            data_source=("NSE CM equity bhav copy CSVs — "
                         "github.com/animesh2007asansol/itis"),
            price_method=("Actual NSE EQ-series closing prices on/near target "
                          "dates. Stop loss detected by scanning daily closes."),
            csv_formats_used=fmts_used,
            start_capital=START_CAPITAL,
            final_capital=yr_sum[-1]["end_capital"],
            total_ret_pct=total_ret, cagr_5yr=cagr,
            win_rate_pct=hit_rate, total_trades=len(all_closed),
            total_wins=wins_n, total_losses=len(all_closed)-wins_n,
            year_summary=yr_sum, trades=trades,
        ), f, indent=2)
    sz = os.path.getsize(OUT_DIR / "backtest.json") // 1024
    print(f"  OK backtest.json ({len(trades)} trades, ~{sz} KB)")

    # Warn about missing data
    missing = [t for t in trades if t.get("verdict","").startswith("DATA MISSING")]
    if missing:
        print(f"\n  WARNING: {len(missing)} trades have missing data:")
        for t in missing:
            print(f"    {t['sym']} {t['fy']}: {t['data_note'] or t['verdict']}")

    print(f"\nDone. seasonal_signals/ has 3 JSON files.")
    print(f"Note: If symbols are still 0 in latest.json, check the")
    print(f"      'Diag' line above for the actual column names in the CSV.")


if __name__ == "__main__":
    main()
