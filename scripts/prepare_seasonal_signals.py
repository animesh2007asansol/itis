#!/usr/bin/env python3
"""
prepare_seasonal_signals.py  v2
=================================
Reads NSE equity bhav copy CSVs from the GitHub data repo.

Outputs (all in seasonal_signals/ — raw data/ is NEVER touched):
  latest.json   <- most recent trading day's prices for seasonal symbols
  prev.json     <- previous trading day's prices
  backtest.json <- full 5-year backtest computed from ACTUAL NSE close prices

Backtest logic per trade
-----------------------
1. Find nearest trading day on-or-after the target buy date -> actual buy price.
2. Scan every trading day from buy to sell date:
   - If close <= buy_px * (1 - sl_pct/100) -> STOP LOSS hit on that date.
3. If no stop: exit price = NSE close on nearest trading day on-or-before sell date.
4. Return = (exit - buy) / buy * 100
5. LIVE/OPEN trades: sell_px = latest available close from NSE data.

Year summary: P&L per trade = dep_amt * ret_pct / 100.
Capital compounds across years.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "seasonal_signals"
MANIFEST  = DATA_DIR / "manifest.json"

START_CAPITAL = 100_000   # Rs 1L initial capital FY22

# ---------------------------------------------------------------------------
# Trade definitions  (authoritative strategy entries)
# buy_date / sell_date = target dates; script finds nearest actual trading day.
# dep_amt = capital allocated to this specific stock in this trade.
# sl_pct  = stop loss % (positive number, e.g. 12 = exit if -12% from buy).
# live    = still holding as of today, sell_date is the planned exit.
# open    = buy signal active but not yet entered.
# ---------------------------------------------------------------------------
TRADE_DEFS = [
    # FY22 -----------------------------------------------------------------
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
         note="Budget FY22 railway capex Rs 2.15L cr."),
    dict(fy="FY22", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2021-12-15", sell_date="2022-02-01",
         dep_amt=18000, sl_pct=12,
         note="IRFC rallied on railway budget euphoria."),

    # FY23 -----------------------------------------------------------------
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
         note="Stop loss — Russia-Ukraine spiked urea import costs."),
    dict(fy="FY23", season="Kharif Fertilizer",   stock="Chambal Fert",
         sym="CHAMBLFERT", buy_date="2022-04-15", sell_date="2022-06-08",
         dep_amt=28000, sl_pct=12,
         note="Stop loss — same war-driven input cost shock."),
    dict(fy="FY23", season="Sugar (Rabi)",         stock="Balrampur Chini",
         sym="BALRAMCHIN", buy_date="2022-09-15", sell_date="2023-02-28",
         dep_amt=38000, sl_pct=12,
         note="Sugar export quota Nov 2022 — catalyst worked."),
    dict(fy="FY23", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2022-12-15", sell_date="2023-02-01",
         dep_amt=28000, sl_pct=12,
         note="Budget FY23 Rs 2.40L cr railway."),
    dict(fy="FY23", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2022-12-15", sell_date="2023-02-01",
         dep_amt=22000, sl_pct=12,
         note="Pre-budget railway stock rally."),

    # FY24 -----------------------------------------------------------------
    dict(fy="FY24", season="AC + Power",          stock="Amber Enterprises",
         sym="AMBER",      buy_date="2023-02-15", sell_date="2023-05-31",
         dep_amt=50000, sl_pct=12,
         note="HOTTEST April in 50 years — Amber +44%."),
    dict(fy="FY24", season="AC + Power",          stock="Blue Star",
         sym="BLUESTARCO", buy_date="2023-02-15", sell_date="2023-05-31",
         dep_amt=40000, sl_pct=12,
         note="Record AC sales Q1 FY24 — Blue Star beat estimates."),
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
         note="RVNL +27% in 5 weeks. Budget Rs 2.62L cr."),
    dict(fy="FY24", season="Railway Pre-Budget",  stock="IRFC",
         sym="IRFC",       buy_date="2023-12-15", sell_date="2024-02-01",
         dep_amt=38000, sl_pct=12,
         note="Record railway allocation — IRFC +21%."),

    # FY25 -----------------------------------------------------------------
    dict(fy="FY25", season="AC + Power",          stock="Voltas",
         sym="VOLTAS",     buy_date="2024-02-15", sell_date="2024-05-10",
         dep_amt=60000, sl_pct=12,
         note="Stop loss — unseasonal rains Apr-May 2024."),
    dict(fy="FY25", season="AC + Power",          stock="NTPC",
         sym="NTPC",       buy_date="2024-02-15", sell_date="2024-04-12",
         dep_amt=45000, sl_pct=12,
         note="Stop loss — market correction + unseasonal rains."),
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
         note="Budget FY25 Rs 2.52L cr railway capex."),
    dict(fy="FY25", season="Railway Pre-Budget",  stock="Jupiter Wagons",
         sym="JWL",        buy_date="2024-12-15", sell_date="2025-02-01",
         dep_amt=55000, sl_pct=12,
         note="JWL +39% after Dec 26 2024 fare revision."),

    # FY26 -----------------------------------------------------------------
    dict(fy="FY26", season="Railway Pre-Budget",  stock="RVNL",
         sym="RVNL",       buy_date="2025-12-15", sell_date="2026-02-01",
         dep_amt=80000, sl_pct=12,
         note="Budget FY26 Rs 3.02L cr — all-time record."),
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
         note="Akshaya Tritiya Apr 19 2026. BUY WINDOW OPEN — exit by Apr 19."),
    dict(fy="FY26", season="Jewellery Pre-AT",    stock="Kalyan Jewellers",
         sym="KALYANKJIL", buy_date="2026-04-13", sell_date="2026-04-19",
         dep_amt=40000, sl_pct=10, open=True,
         note="Akshaya Tritiya Apr 19 2026. BUY WINDOW OPEN — exit by Apr 19."),
]

ALL_SYMS = {t["sym"] for t in TRADE_DEFS} | {
    "COROMANDEL","CHAMBLFERT","DEEPAKFERT","TITAGARH",
    "BALRAMCHIN","TRIVENI","EIDPARRY",
    "ASIANPAINT","BERGEPAINT","KANSAINER","SENCO",
}

# ---------------------------------------------------------------------------
# Price DB: lazy-loading cached NSE CSV reader
# ---------------------------------------------------------------------------
class PriceDB:
    def __init__(self, data_dir, manifest):
        self.data_dir    = data_dir
        self.trading_days = sorted(manifest.keys())   # ascending
        self._cache = {}   # date_str -> {sym: close}
        self._miss  = set()

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
        prices = {}
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            if not lines:
                self._miss.add(date_str); return
            hdr = [h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
            i_sym    = hdr.index("SYMBOL") if "SYMBOL" in hdr else -1
            i_series = hdr.index("SERIES") if "SERIES" in hdr else -1
            i_close  = hdr.index("CLOSE")  if "CLOSE"  in hdr else -1
            if i_sym < 0 or i_close < 0:
                self._miss.add(date_str); return
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
                if len(cols) <= max(i_sym, i_close): continue
                if i_series >= 0 and cols[i_series] != "EQ": continue
                sym = cols[i_sym]
                try:
                    close = float(cols[i_close])
                    if close > 0:
                        prices[sym] = round(close, 2)
                except ValueError:
                    pass
        except Exception as e:
            print(f"    WARN: error loading {date_str}: {e}")
            self._miss.add(date_str); return
        self._cache[date_str] = prices


# ---------------------------------------------------------------------------
# Backtest engine: compute one trade
# ---------------------------------------------------------------------------
def find_price(db, date_target, sym, direction="after", max_delta=5):
    """Find price for sym near date_target. direction = 'after' or 'before'."""
    if direction == "after":
        d = db.nearest_on_or_after(date_target)
    else:
        d = db.nearest_on_or_before(date_target)
    if d:
        px = db.get(d, sym)
        if px: return d, px
    # Widen search
    for delta in range(1, max_delta + 1):
        for sign in [-1, 1]:
            try:
                d2 = (datetime.strptime(date_target, "%Y-%m-%d") + timedelta(days=delta * sign)).strftime("%Y-%m-%d")
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
        fy             = td["fy"],
        season         = td["season"],
        stock          = td["stock"],
        sym            = sym,
        buy_date_target= buy_target,
        sell_date_target= sell_target,
        dep_amt        = dep_amt,
        sl_pct         = sl_pct,
        live           = is_live,
        open           = is_open,
        note           = td.get("note", ""),
        data_ok        = False,
        data_note      = "",
        buy_date_actual= None,
        buy_px         = None,
        sell_date_actual= None,
        sell_px        = None,
        sl_px          = None,
        stop_hit       = False,
        stop_hit_date  = None,
        stop_hit_px    = None,
        ret_pct        = None,
        pnl            = None,
        win            = None,
        verdict        = "UNKNOWN",
    )

    # -- OPEN trade: not yet entered ------------------------------------------
    if is_open:
        buy_date_actual, buy_px = find_price(db, buy_target, sym, "before")
        if not buy_px:
            # use latest
            buy_px = latest_prices.get(sym)
            buy_date_actual = "latest"
        if not buy_px:
            r["verdict"] = "OPEN — price unavailable"
            return r
        r["buy_date_actual"] = buy_date_actual
        r["buy_px"]          = buy_px
        r["sl_px"]           = round(buy_px * (1 - sl_pct / 100), 2)
        r["sell_date_actual"]= sell_target
        r["data_ok"]         = True
        r["data_note"]       = "Not yet entered — entry price = latest NSE close"
        r["verdict"]         = "OPEN — enter now"
        cur = latest_prices.get(sym)
        if cur:
            r["sell_px"] = cur
            ret = round((cur - buy_px) / buy_px * 100, 2)
            r["ret_pct"] = ret
            r["pnl"]     = round(dep_amt * ret / 100)
        return r

    # -- Find buy price -------------------------------------------------------
    buy_date_actual, buy_px = find_price(db, buy_target, sym, "after")
    if not buy_px:
        r["data_note"] = f"Buy price not found for {sym} near {buy_target}"
        r["verdict"]   = "DATA MISSING"
        return r

    r["buy_date_actual"] = buy_date_actual
    r["buy_px"]          = buy_px
    sl_price             = round(buy_px * (1 - sl_pct / 100), 2)
    r["sl_px"]           = sl_price

    # -- Scan for stop loss from buy date to scan_end -------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    scan_end  = today_str if is_live else sell_target
    scan_days = db.trading_days_in_range(buy_date_actual, scan_end)

    stop_hit = False
    for d in scan_days:
        px = db.get(d, sym)
        if px is None:
            continue
        if px <= sl_price:
            stop_hit = True
            r["stop_hit"]      = True
            r["stop_hit_date"] = d
            r["stop_hit_px"]   = px
            break

    if stop_hit:
        exit_px = sl_price   # exit at SL level (strategy rule)
        r["sell_date_actual"] = r["stop_hit_date"]
        r["sell_px"]          = exit_px
        ret = round((exit_px - buy_px) / buy_px * 100, 2)
        r["ret_pct"]  = ret
        r["pnl"]      = round(dep_amt * ret / 100)
        r["win"]      = False
        r["data_ok"]  = True
        r["data_note"]= (f"Stop triggered: NSE close Rs{r['stop_hit_px']} "
                         f"<= SL Rs{sl_price} on {r['stop_hit_date']}")
        r["verdict"]  = f"STOP LOSS HIT ({r['stop_hit_date']}) — actual NSE data"
        return r

    # -- LIVE trade: no stop hit yet, show current P&L -----------------------
    if is_live:
        latest_day = db.nearest_on_or_before(today_str)
        cur_px = db.get(latest_day, sym) if latest_day else None
        if cur_px is None:
            cur_px = latest_prices.get(sym)
            latest_day = "latest.json"
        if cur_px:
            r["sell_date_actual"] = latest_day
            r["sell_px"]          = cur_px
            ret = round((cur_px - buy_px) / buy_px * 100, 2)
            r["ret_pct"]  = ret
            r["pnl"]      = round(dep_amt * ret / 100)
            r["win"]      = None
            r["data_ok"]  = True
            r["data_note"]= f"Buy: NSE close {buy_date_actual}; Current: {latest_day}"
            chg_icon = "UP" if ret >= 0 else "DOWN"
            r["verdict"]  = f"LIVE — {chg_icon} {ret:+.1f}% (holding, SL not hit)"
        else:
            r["verdict"] = "LIVE — current price unavailable"
        return r

    # -- Closed trade: find sell price ----------------------------------------
    sell_date_actual, sell_px = find_price(db, sell_target, sym, "before")
    if not sell_px:
        r["data_note"] = f"Sell price not found for {sym} near {sell_target}"
        r["verdict"]   = "DATA MISSING"
        return r

    r["sell_date_actual"] = sell_date_actual
    r["sell_px"]          = sell_px
    ret = round((sell_px - buy_px) / buy_px * 100, 2)
    r["ret_pct"]  = ret
    r["pnl"]      = round(dep_amt * ret / 100)
    r["win"]      = ret > 0
    r["data_ok"]  = True
    r["data_note"]= f"Buy: NSE close {buy_date_actual} | Sell: NSE close {sell_date_actual}"
    verdict_tag   = "WIN" if ret > 0 else "LOSS"
    r["verdict"]  = f"{verdict_tag} {ret:+.1f}% — actual NSE data"
    return r


# ---------------------------------------------------------------------------
# Year summary from computed trades
# ---------------------------------------------------------------------------
def compute_year_summary(trades, start_capital):
    years   = ["FY22", "FY23", "FY24", "FY25", "FY26"]
    summary = []
    capital = start_capital
    for fy in years:
        fy_t   = [t for t in trades if t["fy"] == fy]
        pnl    = sum(t["pnl"] for t in fy_t if t.get("pnl") is not None)
        wins   = sum(1 for t in fy_t if t.get("win") is True)
        losses = sum(1 for t in fy_t if t.get("win") is False)
        stops  = sum(1 for t in fy_t if t.get("stop_hit"))
        partial= any(t.get("live") or t.get("open") for t in fy_t)
        end_cap= round(capital + pnl)
        ret_pct= round(pnl / capital * 100, 1) if capital else 0

        completed = [t for t in fy_t
                     if t.get("ret_pct") is not None
                     and not t.get("live") and not t.get("open")]
        best  = max(completed, key=lambda t: t.get("ret_pct") or -9999) if completed else None
        worst = min(completed, key=lambda t: t.get("ret_pct") or  9999) if completed else None

        parts = []
        if stops:
            parts.append(f"{stops} stop loss(es)")
        if best and best.get("ret_pct", 0) > 0:
            parts.append(f"Best: {best['stock']} {best['ret_pct']:+.1f}%")
        if worst and worst.get("ret_pct", 0) < 0:
            parts.append(f"Worst: {worst['stock']} {worst['ret_pct']:+.1f}%")
        if partial:
            parts.append("FY still open (unrealised P&L included)")
        note = ". ".join(parts) if parts else f"{wins}W {losses}L"

        summary.append(dict(
            fy           = fy,
            start_capital= round(capital),
            pnl          = round(pnl),
            ret_pct      = ret_pct,
            end_capital  = end_cap,
            wins         = wins,
            losses       = losses,
            stops        = stops,
            partial      = partial,
            note         = note,
        ))
        capital = end_cap
    return summary


# ---------------------------------------------------------------------------
# CSV reader for latest/prev (lightweight)
# ---------------------------------------------------------------------------
def read_csv_prices(path, syms):
    prices = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines: return prices
        hdr = [h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_sym    = hdr.index("SYMBOL") if "SYMBOL" in hdr else -1
        i_series = hdr.index("SERIES") if "SERIES" in hdr else -1
        i_close  = hdr.index("CLOSE")  if "CLOSE"  in hdr else -1
        if i_sym < 0 or i_close < 0: return prices
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols) <= max(i_sym, i_close): continue
            if i_series >= 0 and cols[i_series] != "EQ": continue
            sym = cols[i_sym]
            if sym not in syms: continue
            try:
                close = round(float(cols[i_close]), 2)
                if close > 0: prices[sym] = close
            except ValueError: pass
    except Exception: pass
    return prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SeasonalSignals v2 — NSE Historical Backtest Engine")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest
    print("\n[1] Loading manifest...")
    if not MANIFEST.exists():
        print(f"  ERROR: manifest.json not found at {MANIFEST}")
        sys.exit(1)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    asc  = sorted(manifest.keys())
    desc = list(reversed(asc))
    print(f"  OK: {len(asc)} trading days  [{asc[0]} -> {asc[-1]}]")

    # 2. Identify latest/prev CSV files
    print("\n[2] Finding latest equity CSVs...")
    found = []
    for d in desc[:20]:
        y, m, _ = d.split("-")
        p = DATA_DIR / "equity" / y / m / f"{d}.csv"
        if p.exists():
            found.append(d)
            if len(found) == 2: break
    if not found:
        print("  ERROR: No equity CSVs found")
        sys.exit(1)
    latest_date = found[0]
    prev_date   = found[1] if len(found) > 1 else None
    print(f"  latest: {latest_date}")
    print(f"  prev:   {prev_date or 'none'}")

    latest_raw = read_csv_prices(
        DATA_DIR / "equity" / latest_date[:4] / latest_date[5:7] / f"{latest_date}.csv",
        ALL_SYMS)
    prev_raw = {}
    if prev_date:
        prev_raw = read_csv_prices(
            DATA_DIR / "equity" / prev_date[:4] / prev_date[5:7] / f"{prev_date}.csv",
            ALL_SYMS)
    print(f"  latest: {len(latest_raw)} symbols | prev: {len(prev_raw)} symbols")

    # 3. Build price DB and run backtest
    print("\n[3] Running backtest from NSE historical CSV files...")
    db     = PriceDB(DATA_DIR, manifest)
    trades = [compute_trade(td, db, latest_raw) for td in TRADE_DEFS]

    # Print trade summary table
    print(f"\n  {'':2} {'Sym':<14} {'FY':5} {'Buy Date':11} {'Buy Rs':9} "
          f"{'Sell Date':11} {'Sell Rs':9} {'Ret%':7}  Verdict")
    print(f"  {'':2} {'-'*14} {'-'*5} {'-'*11} {'-'*9} {'-'*11} {'-'*9} {'-'*7}  {'-'*35}")
    for t in trades:
        icon = ("OK" if t.get("win") is True
                else "SL" if t.get("stop_hit")
                else "LV" if t.get("live")
                else "OP" if t.get("open")
                else "??" )
        bp  = f"{t['buy_px']:9.2f}"  if t.get("buy_px")  else "     ——  "
        sp  = f"{t['sell_px']:9.2f}" if t.get("sell_px") else "     ——  "
        bd  = (t.get("buy_date_actual") or "——")[-10:]
        sd  = (t.get("sell_date_actual") or "——")[-10:]
        rp  = f"{t['ret_pct']:+7.2f}%" if t.get("ret_pct") is not None else "   ——   "
        vrd = (t.get("verdict") or "")[:40]
        print(f"  {icon} {t['sym']:<14} {t['fy']:5} {bd:11} {bp} {sd:11} {sp} {rp}  {vrd}")

    # 4. Year summary
    print("\n[4] Year summary...")
    yr_sum = compute_year_summary(trades, START_CAPITAL)
    for y in yr_sum:
        pnl_s = f"+Rs{y['pnl']:,}" if y["pnl"] >= 0 else f"-Rs{abs(y['pnl']):,}"
        print(f"  {y['fy']}  Rs{y['start_capital']:>9,} -> Rs{y['end_capital']:>9,}  "
              f"{y['ret_pct']:>+6.1f}%  {pnl_s:>14}  {y['wins']}W {y['losses']}L  {y['note'][:50]}")

    all_closed = [t for t in trades
                  if not t.get("live") and not t.get("open")
                  and t.get("ret_pct") is not None]
    wins_total = sum(1 for t in all_closed if t.get("win"))
    hit_rate   = round(wins_total / len(all_closed) * 100, 1) if all_closed else 0
    total_ret  = round((yr_sum[-1]["end_capital"] - START_CAPITAL) / START_CAPITAL * 100, 1)
    cagr       = round(((yr_sum[-1]["end_capital"] / START_CAPITAL) ** (1/5) - 1) * 100, 1)
    print(f"\n  Total return: {total_ret:+.1f}%  |  5-yr CAGR: {cagr:+.1f}%  |  "
          f"Win rate: {hit_rate:.0f}% ({wins_total}/{len(all_closed)})")

    # 5. Write output files
    ist     = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    print("\n[5] Writing output files...")

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
        json.dump(dict(
            label="latest", date=latest_date, generated_at=now_ist,
            trading_day=True, symbol_count=len(latest_rich),
            prices=latest_rich,
            source="NSE CM segment equity bhav copy — EQ series",
            note="NSE official closing prices. Updated nightly after 11 PM IST.",
        ), f, indent=2)
    print(f"  OK latest.json  ({latest_date}, {len(latest_rich)} symbols)")

    # prev.json
    if prev_date:
        with open(OUT_DIR / "prev.json", "w") as f:
            json.dump(dict(
                label="prev", date=prev_date, generated_at=now_ist,
                trading_day=True, symbol_count=len(prev_raw),
                prices={sym: {"close": c} for sym, c in prev_raw.items()},
                source="NSE CM segment equity bhav copy — EQ series",
            ), f, indent=2)
        print(f"  OK prev.json    ({prev_date}, {len(prev_raw)} symbols)")

    # backtest.json
    with open(OUT_DIR / "backtest.json", "w") as f:
        json.dump(dict(
            generated_at  = now_ist,
            data_source   = ("NSE CM equity bhav copy CSVs — "
                             "github.com/animesh2007asansol/itis"),
            price_method  = ("Actual NSE EQ-series closing prices on/near target "
                             "dates. Stop loss detected by scanning daily closes."),
            start_capital = START_CAPITAL,
            final_capital = yr_sum[-1]["end_capital"],
            total_ret_pct = total_ret,
            cagr_5yr      = cagr,
            win_rate_pct  = hit_rate,
            total_trades  = len(all_closed),
            total_wins    = wins_total,
            total_losses  = len(all_closed) - wins_total,
            year_summary  = yr_sum,
            trades        = trades,
        ), f, indent=2)
    sz = os.path.getsize(OUT_DIR / "backtest.json") // 1024
    print(f"  OK backtest.json ({len(trades)} trades, ~{sz} KB)")

    print(f"\nDone.  seasonal_signals/ updated with 3 JSON files.")


if __name__ == "__main__":
    main()
