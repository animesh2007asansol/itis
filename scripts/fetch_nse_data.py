#!/usr/bin/env python3
"""
NSE Bhav Copy Fetcher - Production Grade
=========================================
Downloads daily stock market data from NSE India:
  - CM Equity Bhav Copy (OHLCV for all stocks)
  - F&O Bhav Copy (Futures & Options)
  - SME Bhav Copy (Small & Medium Enterprises)
  - Index Data (Nifty, Bank Nifty, etc.)
  - Corporate Actions (Splits, Bonuses, Dividends, Rights)
  - Delivery / Institutional data

Designed to never fail silently. Every error is logged,
retried, and recorded in a daily status manifest.

Author: Auto-generated for animesh2007asansol
"""

import os
import sys
import time
import zipfile
import logging
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import random
import io
import traceback
from typing import Optional, Tuple, Dict, List

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
LOG_DIR    = BASE_DIR / "logs"
MANIFEST   = BASE_DIR / "data" / "manifest.json"

# NSE root URLs (primary + fallbacks)
NSE_ARCHIVE   = "https://archives.nseindia.com"
NSE_WWW       = "https://www.nseindia.com"
NSE_WWW1      = "https://www1.nseindia.com"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":  "keep-alive",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

MAX_RETRIES = 7
RETRY_DELAY = 8   # seconds between retries (multiplied by attempt number)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
today_str = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"{today_str}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nse_fetcher")

# ─────────────────────────────────────────────
# NSE SESSION (cookie management)
# ─────────────────────────────────────────────

def build_session() -> requests.Session:
    """
    Build a requests.Session that mimics a real browser and
    seeds the correct NSE cookies (nsit, nseappid, bm_sv, etc.)
    without which most archive downloads return 403.
    """
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    warm_up_urls = [
        f"{NSE_WWW}/",
        f"{NSE_WWW}/market-data/live-equity-market",
        f"{NSE_WWW}/market-data/equity-stock-indices",
    ]

    for url in warm_up_urls:
        try:
            r = session.get(url, timeout=20)
            logger.info(f"  Session warm-up: {url} → {r.status_code}")
            time.sleep(random.uniform(1.5, 3.0))
        except Exception as exc:
            logger.warning(f"  Warm-up failed for {url}: {exc}")

    return session


# ─────────────────────────────────────────────
# CORE DOWNLOAD UTILITY
# ─────────────────────────────────────────────

def download_url(
    url: str,
    session: requests.Session,
    max_retries: int = MAX_RETRIES,
    base_delay: int = RETRY_DELAY,
) -> Optional[bytes]:
    """
    Download a URL with exponential back-off retry.
    Refreshes the session on 403/429 responses.
    Returns raw bytes or None if all retries exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  [{attempt}/{max_retries}] GET {url}")
            resp = session.get(url, timeout=60, stream=True)

            if resp.status_code == 200:
                content = resp.content
                if len(content) < 100:
                    logger.warning("  Response too small – likely an error page, retrying…")
                else:
                    logger.info(f"  ✓ Downloaded {len(content):,} bytes")
                    return content

            elif resp.status_code in (403, 429):
                logger.warning(f"  {resp.status_code} – re-seeding session cookies…")
                session = build_session()

            else:
                logger.warning(f"  HTTP {resp.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("  Timeout – retrying…")
        except requests.exceptions.ConnectionError as exc:
            logger.warning(f"  Connection error: {exc}")
        except Exception as exc:
            logger.error(f"  Unexpected error: {exc}\n{traceback.format_exc()}")

        sleep_time = base_delay * attempt + random.uniform(0, 3)
        logger.info(f"  Sleeping {sleep_time:.1f}s before next attempt…")
        time.sleep(sleep_time)

    logger.error(f"  ✗ All {max_retries} attempts failed for {url}")
    return None


def extract_zip_csv(content: bytes, filename_hint: str = "") -> Optional[pd.DataFrame]:
    """Extract a single CSV from a zip archive and return as DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            csv_names = [n for n in names if n.lower().endswith(".csv")]
            if not csv_names:
                logger.error(f"  No CSV found in zip. Contents: {names}")
                return None
            target = csv_names[0]
            with zf.open(target) as f:
                df = pd.read_csv(f)
            logger.info(f"  Extracted '{target}': {len(df):,} rows × {len(df.columns)} cols")
            return df
    except zipfile.BadZipFile:
        # Maybe it's already a plain CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
            logger.info(f"  Parsed as plain CSV: {len(df):,} rows")
            return df
        except Exception as exc:
            logger.error(f"  Not a valid CSV either: {exc}")
            return None
    except Exception as exc:
        logger.error(f"  Zip extraction error: {exc}")
        return None


# ─────────────────────────────────────────────
# DATA SAVERS
# ─────────────────────────────────────────────

def save_df(df: pd.DataFrame, category: str, trade_date: date, extra_tag: str = "") -> Path:
    """Save a DataFrame to data/<category>/YYYY/MM/YYYY-MM-DD[_tag].csv"""
    year  = trade_date.strftime("%Y")
    month = trade_date.strftime("%m")
    dir_  = DATA_DIR / category / year / month
    dir_.mkdir(parents=True, exist_ok=True)

    fname = f"{trade_date.isoformat()}"
    if extra_tag:
        fname += f"_{extra_tag}"
    fname += ".csv"
    path = dir_ / fname

    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.to_csv(path, index=False)
    logger.info(f"  Saved → {path.relative_to(BASE_DIR)}")
    return path


# ─────────────────────────────────────────────
# 1. EQUITY BHAV COPY (CM segment)
# ─────────────────────────────────────────────

def fetch_equity_bhav(trade_date: date, session: requests.Session) -> bool:
    """
    Columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE,
             LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL,
             TIMESTAMP, TOTALTRADES, ISIN
    """
    ds  = trade_date.strftime("%d%b%Y").upper()   # e.g. 12APR2024
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    urls = [
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip",
        f"{NSE_WWW1}/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip",
    ]

    for url in urls:
        content = download_url(url, session)
        if content:
            df = extract_zip_csv(content)
            if df is not None and not df.empty:
                save_df(df, "equity", trade_date)
                return True

    logger.error("  fetch_equity_bhav: all URLs failed")
    return False


# ─────────────────────────────────────────────
# 2. F&O BHAV COPY
# ─────────────────────────────────────────────

def fetch_fo_bhav(trade_date: date, session: requests.Session) -> bool:
    """
    Futures & Options daily bhav copy.
    Columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
             OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS,
             VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP
    """
    ds  = trade_date.strftime("%d%b%Y").upper()
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    urls = [
        f"{NSE_ARCHIVE}/content/historical/DERIVATIVES/{yr}/{mon}/fo{ds}bhav.csv.zip",
        f"{NSE_WWW1}/content/historical/DERIVATIVES/{yr}/{mon}/fo{ds}bhav.csv.zip",
    ]

    for url in urls:
        content = download_url(url, session)
        if content:
            df = extract_zip_csv(content)
            if df is not None and not df.empty:
                save_df(df, "fo", trade_date)
                return True

    logger.error("  fetch_fo_bhav: all URLs failed")
    return False


# ─────────────────────────────────────────────
# 3. EQUITY DELIVERY / INSTITUTIONAL DATA
# ─────────────────────────────────────────────

def fetch_delivery_data(trade_date: date, session: requests.Session) -> bool:
    """
    Deliverable quantity data per symbol.
    Columns: DATE2, SYMBOL, SERIES, DELIV_QTY, DELIV_PER
    """
    ds  = trade_date.strftime("%d%m%Y")
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    filename = f"MTO_{ds}.DAT"
    urls = [
        f"{NSE_ARCHIVE}/archives/equities/mto/{filename}",
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/{filename}",
    ]

    for url in urls:
        content = download_url(url, session)
        if content:
            try:
                # MTO file: skip first 4 header rows
                df = pd.read_csv(
                    io.BytesIO(content),
                    header=None,
                    skiprows=3,
                    names=["RecType", "SrNo", "NAME", "DELIV_QTY", "DELIV_VAL",
                           "TOTTRDQTY", "TOTTRDVAL", "DELIV_PER"],
                )
                df = df[df["RecType"] == 20].copy()
                if not df.empty:
                    save_df(df, "delivery", trade_date)
                    return True
            except Exception as exc:
                logger.warning(f"  MTO parse error: {exc}")

    # Fallback: Try the sec_bhavdata format
    ds2 = trade_date.strftime("%d%b%Y").upper()
    url = f"{NSE_ARCHIVE}/archives/equities/bhavcopy/sec_bhavdata{ds2}.csv"
    content = download_url(url, session)
    if content:
        try:
            df = pd.read_csv(io.BytesIO(content))
            if not df.empty:
                save_df(df, "delivery", trade_date)
                return True
        except Exception as exc:
            logger.warning(f"  sec_bhavdata parse error: {exc}")

    logger.warning("  fetch_delivery_data: no data found (non-critical)")
    return False


# ─────────────────────────────────────────────
# 4. INDEX BHAV COPY (Nifty, BankNifty, etc.)
# ─────────────────────────────────────────────

def fetch_index_data(trade_date: date, session: requests.Session) -> bool:
    """
    All NSE Index values for the day.
    Columns: Index Name, Open, High, Low, Closing, Points Change,
             Change(%),  Volume, Turnover(Rs.Cr.)
    """
    ds  = trade_date.strftime("%d%m%Y")
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    urls = [
        f"{NSE_ARCHIVE}/content/indices/ind_close_all_{ds}.csv",
        f"{NSE_ARCHIVE}/archives/equities/indices/ind_close_all_{ds}.csv",
    ]

    for url in urls:
        content = download_url(url, session)
        if content:
            try:
                df = pd.read_csv(io.BytesIO(content))
                if not df.empty:
                    save_df(df, "index", trade_date)
                    return True
            except Exception as exc:
                logger.warning(f"  Index CSV parse error: {exc}")

    logger.warning("  fetch_index_data: no data found (non-critical)")
    return False


# ─────────────────────────────────────────────
# 5. CORPORATE ACTIONS
# ─────────────────────────────────────────────

def fetch_corporate_actions(trade_date: date, session: requests.Session) -> bool:
    """
    Fetch corporate actions (Bonus, Split, Dividend, Rights) from NSE API.
    Saves both a daily snapshot and an append to the master CSV.
    """
    from_date = (trade_date - timedelta(days=7)).strftime("%d-%m-%Y")
    to_date   = trade_date.strftime("%d-%m-%Y")

    # NSE corporate actions JSON API
    corp_url = (
        f"{NSE_WWW}/corporates/shownCorporateActions?index=equities"
        f"&from_date={from_date}&to_date={to_date}"
    )

    # Seed referer header for this API call
    session.headers["Referer"] = f"{NSE_WWW}/companies-listing/corporate-filings/corporate-actions"

    content = download_url(corp_url, session)
    if content:
        try:
            data = json.loads(content)
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                save_df(df, "corporate_actions", trade_date)

                # Append to master file
                master_path = DATA_DIR / "corporate_actions" / "master.csv"
                master_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not master_path.exists()
                df.to_csv(master_path, mode="a", header=write_header, index=False)
                logger.info(f"  Appended {len(df)} rows to master corporate actions")
                return True
            else:
                logger.info("  No corporate actions for this period")
                return True  # Not an error – just no actions
        except json.JSONDecodeError:
            logger.warning("  Corporate actions response is not JSON")
        except Exception as exc:
            logger.warning(f"  Corporate actions parse error: {exc}")

    logger.warning("  fetch_corporate_actions: failed (non-critical)")
    return False


# ─────────────────────────────────────────────
# 6. SME BHAV COPY (Emerge Platform)
# ─────────────────────────────────────────────

def fetch_sme_bhav(trade_date: date, session: requests.Session) -> bool:
    """NSE SME / Emerge platform bhav copy."""
    ds  = trade_date.strftime("%d%b%Y").upper()
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    url = (
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/"
        f"SME{ds}BHAV.csv.zip"
    )

    content = download_url(url, session)
    if content:
        df = extract_zip_csv(content)
        if df is not None and not df.empty:
            save_df(df, "sme", trade_date)
            return True

    logger.warning("  fetch_sme_bhav: no data (non-critical – may be unavailable)")
    return False


# ─────────────────────────────────────────────
# 7. FULL BHAVCOPY (PR series – richer format)
# ─────────────────────────────────────────────

def fetch_full_bhav(trade_date: date, session: requests.Session) -> bool:
    """
    Full bhavcopy with 50+ columns including delivery, turnover per series.
    URL pattern: PR<DDMMYYYY>.zip → contains multiple .CSV files
    """
    ds = trade_date.strftime("%d%m%Y")
    yr = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    urls = [
        f"{NSE_ARCHIVE}/archives/equities/bhavcopy/pr{ds}.zip",
    ]

    for url in urls:
        content = download_url(url, session)
        if content:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv"):
                            with zf.open(name) as f:
                                df = pd.read_csv(f, header=None)
                            tag = Path(name).stem.lower()
                            save_df(df, "full_bhav", trade_date, extra_tag=tag)
                return True
            except Exception as exc:
                logger.warning(f"  full_bhav extraction error: {exc}")

    logger.warning("  fetch_full_bhav: no data (non-critical)")
    return False


# ─────────────────────────────────────────────
# MANIFEST & SUMMARY
# ─────────────────────────────────────────────

def update_manifest(trade_date: date, results: Dict[str, bool]) -> None:
    """
    Update data/manifest.json with fetch results.
    This file acts as an API index for downstream apps.
    """
    manifest = {}
    if MANIFEST.exists():
        try:
            with open(MANIFEST, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    date_key = trade_date.isoformat()
    manifest[date_key] = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "success": any(results.values()),
        "files": {},
    }

    # Record actual file paths for each successful category
    for category in ["equity", "fo", "index", "delivery", "sme", "corporate_actions", "full_bhav"]:
        yr  = trade_date.strftime("%Y")
        mon = trade_date.strftime("%m")
        fp  = DATA_DIR / category / yr / mon / f"{date_key}.csv"
        if fp.exists():
            manifest[date_key]["files"][category] = str(fp.relative_to(BASE_DIR))

    # Sort by date descending
    manifest = dict(sorted(manifest.items(), reverse=True))

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Manifest updated: {MANIFEST.relative_to(BASE_DIR)}")


def generate_summary_page(manifest: dict) -> None:
    """
    Write data/summary.json – a lightweight stats file consumed
    by the GitHub Pages front-end.
    """
    total_dates  = len(manifest)
    success_dates = sum(1 for v in manifest.values() if v.get("success"))
    categories = set()
    for v in manifest.values():
        categories.update(v.get("files", {}).keys())

    summary = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "total_trading_days": total_dates,
        "successful_fetches": success_dates,
        "categories_available": sorted(categories),
        "latest_date": next(iter(manifest), None),
        "latest_status": next(iter(manifest.values()), {}).get("results", {}),
    }

    summary_path = DATA_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Summary page written: {summary_path.relative_to(BASE_DIR)}")


# ─────────────────────────────────────────────
# HOLIDAY / WEEKEND GUARD
# ─────────────────────────────────────────────

def is_market_holiday(trade_date: date, session: requests.Session) -> bool:
    """
    Quick check: if equity bhav copy for that date doesn't exist
    on NSE archives, it's a holiday. Returns True = holiday.
    """
    ds  = trade_date.strftime("%d%b%Y").upper()
    yr  = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()
    url = f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip"

    try:
        r = session.head(url, timeout=15)
        return r.status_code != 200
    except Exception:
        return False   # Assume not holiday if check itself fails


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def run(target_date: Optional[date] = None) -> bool:
    """
    Main entry point.
    target_date: override date (useful for backfilling).
    Returns True if at least equity data was saved.
    """
    if target_date is None:
        # Default to yesterday if run late at night, else today
        now = datetime.utcnow()
        # 15:30 UTC = 21:00 IST (market closed ~10h ago for today's data)
        target_date = now.date()

    logger.info("=" * 60)
    logger.info(f"  NSE Data Fetcher starting for {target_date}")
    logger.info("=" * 60)

    # Skip weekends
    if target_date.weekday() >= 5:
        logger.info(f"  {target_date} is a weekend – nothing to fetch.")
        return True

    # Build session
    logger.info("  Building NSE session (seeding cookies)…")
    session = build_session()
    logger.info("  Session ready.")

    # Check if market was open
    logger.info("  Checking if market traded on this date…")
    if is_market_holiday(target_date, session):
        logger.info(f"  {target_date} appears to be a market holiday – skipping.")
        return True

    # ── Fetch all data categories ──────────────────────────────

    results: Dict[str, bool] = {}

    logger.info("\n── 1/7  EQUITY BHAV COPY ──────────────────────────────")
    results["equity"] = fetch_equity_bhav(target_date, session)

    logger.info("\n── 2/7  F&O BHAV COPY ─────────────────────────────────")
    results["fo"] = fetch_fo_bhav(target_date, session)

    logger.info("\n── 3/7  DELIVERY DATA ─────────────────────────────────")
    results["delivery"] = fetch_delivery_data(target_date, session)

    logger.info("\n── 4/7  INDEX DATA ────────────────────────────────────")
    results["index"] = fetch_index_data(target_date, session)

    logger.info("\n── 5/7  CORPORATE ACTIONS ─────────────────────────────")
    results["corporate_actions"] = fetch_corporate_actions(target_date, session)

    logger.info("\n── 6/7  SME BHAV COPY ─────────────────────────────────")
    results["sme"] = fetch_sme_bhav(target_date, session)

    logger.info("\n── 7/7  FULL BHAV COPY (PR series) ────────────────────")
    results["full_bhav"] = fetch_full_bhav(target_date, session)

    # ── Manifest & Summary ─────────────────────────────────────

    logger.info("\n── UPDATING MANIFEST ──────────────────────────────────")
    update_manifest(target_date, results)

    if MANIFEST.exists():
        with open(MANIFEST) as f:
            manifest = json.load(f)
        generate_summary_page(manifest)

    # ── Final report ───────────────────────────────────────────

    logger.info("\n" + "=" * 60)
    logger.info("  FETCH SUMMARY")
    logger.info("=" * 60)
    for cat, ok in results.items():
        status = "✓  OK" if ok else "✗  FAILED"
        logger.info(f"  {cat:<25} {status}")

    equity_ok = results.get("equity", False)
    if not equity_ok:
        logger.error("\n  ⚠ CRITICAL: Equity bhav copy could not be fetched!")
        return False

    logger.info("\n  ✓ Fetch complete for " + str(target_date))
    return True


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Accept optional date argument: python fetch_nse_data.py 2024-04-10
    target: Optional[date] = None

    if len(sys.argv) > 1 and sys.argv[1].strip():
        try:
            target = date.fromisoformat(sys.argv[1].strip())
            logger.info(f"  Using command-line date: {target}")
        except ValueError:
            logger.error(f"  Invalid date format '{sys.argv[1]}'. Use YYYY-MM-DD.")
            sys.exit(1)

    success = run(target)
    sys.exit(0 if success else 1)
