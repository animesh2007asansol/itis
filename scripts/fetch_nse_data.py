#!/usr/bin/env python3
"""
NSE Bhav Copy Fetcher — v2 (Cloudflare-proof)
===============================================
Root cause of empty data folder:
  NSE's Cloudflare WAF blocks GitHub Actions IP ranges when using
  plain `requests`, even with perfect browser headers. The TLS
  fingerprint (JA3 hash) of Python's ssl module is trivially
  detected and rejected before any cookie logic runs.

Fix strategy (in order of attempt):
  1. curl_cffi   — impersonates Chrome's TLS fingerprint exactly
  2. cloudscraper — JS-challenge solver fallback
  3. requests     — last resort (works on some NSE archive URLs)
  4. yfinance     — for equity/index OHLCV only; bypasses NSE entirely

Data downloaded:
  • Equity Bhav Copy  (OHLCV, ISIN, turnover)
  • F&O Bhav Copy     (futures + options)
  • Index Bhav Copy   (Nifty 50, Bank Nifty, all indices)
  • Delivery Data     (deliverable qty per stock)
  • Corporate Actions (split, bonus, dividend, rights)
  • SME / Emerge Bhav Copy
"""

import os, sys, io, json, time, random, zipfile, logging, traceback
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
MANIFEST = DATA_DIR / "manifest.json"

NSE_ARCHIVE = "https://archives.nseindia.com"
NSE_WWW     = "https://www.nseindia.com"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"{date.today().isoformat()}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("nse")

# ─────────────────────────────────────────────
# HTTP CLIENT — tries 3 engines in order
# ─────────────────────────────────────────────

def _make_curl_session():
    """curl_cffi: spoofs Chrome 124 TLS fingerprint (JA3 + ALPN)."""
    from curl_cffi.requests import Session
    s = Session(impersonate="chrome124")
    return s, "curl_cffi"


def _make_cloudscraper_session():
    """cloudscraper: solves Cloudflare JS challenges."""
    import cloudscraper
    s = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    return s, "cloudscraper"


def _make_requests_session():
    """Plain requests — works for raw NSE archive files that bypass CF."""
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return s, "requests"


def build_session():
    """Return the best available HTTP session."""
    for factory in [_make_curl_session, _make_cloudscraper_session, _make_requests_session]:
        try:
            s, name = factory()
            log.info(f"  HTTP engine: {name}")
            # Warm up — seeds NSE cookies
            try:
                r = s.get(NSE_WWW, timeout=20)
                log.info(f"  Warm-up {NSE_WWW} → {r.status_code}")
                time.sleep(random.uniform(2, 4))
                r2 = s.get(f"{NSE_WWW}/market-data/live-equity-market", timeout=20)
                log.info(f"  Warm-up market-data → {r2.status_code}")
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                log.warning(f"  Warm-up error (non-fatal): {e}")
            return s, name
        except ImportError:
            continue
        except Exception as exc:
            log.warning(f"  {factory.__name__} failed: {exc}")
            continue

    raise RuntimeError("No HTTP engine available. Install curl_cffi or cloudscraper.")


def download(url: str, session, retries: int = 6) -> Optional[bytes]:
    """Download URL with retries. Returns bytes or None."""
    for attempt in range(1, retries + 1):
        try:
            log.info(f"  [{attempt}/{retries}] GET {url}")
            r = session.get(url, timeout=60)
            code = r.status_code

            if code == 200:
                data = r.content
                if len(data) < 200:
                    log.warning(f"  Response only {len(data)} bytes — likely error page")
                else:
                    log.info(f"  OK {len(data):,} bytes")
                    return data

            elif code in (403, 429):
                log.warning(f"  {code} — Cloudflare/rate-limit. Re-warming session...")
                try:
                    session.get(NSE_WWW, timeout=20)
                    time.sleep(random.uniform(5, 10))
                except Exception:
                    pass

            elif code == 404:
                log.warning(f"  404 — file not on NSE (holiday or wrong date)")
                return None   # no point retrying a 404

            else:
                log.warning(f"  HTTP {code}")

        except Exception as exc:
            log.warning(f"  Error: {exc}")

        wait = (attempt * 8) + random.uniform(0, 4)
        log.info(f"  Sleeping {wait:.1f}s...")
        time.sleep(wait)

    log.error(f"  All {retries} attempts failed")
    return None


def zip_to_df(content: bytes) -> Optional[pd.DataFrame]:
    """Extract first CSV from a zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                log.error(f"  No CSV in zip. Contents: {zf.namelist()}")
                return None
            with zf.open(csvs[0]) as f:
                df = pd.read_csv(f)
            log.info(f"  Extracted '{csvs[0]}': {len(df):,} rows x {len(df.columns)} cols")
            return df
    except zipfile.BadZipFile:
        try:
            df = pd.read_csv(io.BytesIO(content))
            log.info(f"  Plain CSV: {len(df):,} rows")
            return df
        except Exception as e:
            log.error(f"  Not a zip or CSV: {e}")
            return None
    except Exception as e:
        log.error(f"  Zip error: {e}")
        return None


def save(df: pd.DataFrame, category: str, d: date, tag: str = "") -> Path:
    yr, mo = d.strftime("%Y"), d.strftime("%m")
    out = DATA_DIR / category / yr / mo
    out.mkdir(parents=True, exist_ok=True)
    name = d.isoformat() + (f"_{tag}" if tag else "") + ".csv"
    path = out / name
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.to_csv(path, index=False)
    log.info(f"  Saved -> {path.relative_to(BASE_DIR)}")
    return path


# ─────────────────────────────────────────────
# 1. EQUITY BHAV COPY
# ─────────────────────────────────────────────

def fetch_equity_bhav(d: date, session) -> bool:
    ds  = d.strftime("%d%b%Y").upper()
    yr  = d.strftime("%Y")
    mon = d.strftime("%b").upper()

    urls = [
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip",
        f"https://www1.nseindia.com/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip",
    ]

    for url in urls:
        raw = download(url, session)
        if raw:
            df = zip_to_df(raw)
            if df is not None and not df.empty:
                save(df, "equity", d)
                return True

    # yfinance fallback
    log.warning("  NSE direct failed -- trying yfinance fallback...")
    return _equity_via_yfinance(d)


def _equity_via_yfinance(d: date) -> bool:
    """
    Download OHLCV for Nifty 50 stocks from Yahoo Finance.
    Yahoo Finance never blocks GitHub Actions.
    NOTE: covers Nifty 50 only — NSE bhav covers ~2000 stocks.
          Full NSE bhav requires curl_cffi to pass Cloudflare.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("  yfinance not installed")
        return False

    tickers = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
        "SUNPHARMA.NS","ULTRACEMCO.NS","BAJFINANCE.NS","NESTLEIND.NS","WIPRO.NS",
        "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","ONGC.NS",
        "JSWSTEEL.NS","TATAMOTORS.NS","M&M.NS","BAJAJFINSV.NS","COALINDIA.NS",
        "GRASIM.NS","ADANIPORTS.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS",
        "BRITANNIA.NS","EICHERMOT.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS",
        "SBILIFE.NS","HDFCLIFE.NS","TATACONSUM.NS","BPCL.NS","IOC.NS",
        "TATASTEEL.NS","UPL.NS","APOLLOHOSP.NS","BAJAJ-AUTO.NS","LTIM.NS",
    ]

    try:
        start = d.isoformat()
        end   = (d + timedelta(days=1)).isoformat()
        data  = yf.download(tickers, start=start, end=end, interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        if data.empty:
            log.warning("  yfinance: no data (holiday?)")
            return False

        rows = []
        for t in tickers:
            sym = t.replace(".NS", "")
            try:
                row = data[t].dropna()
                if row.empty:
                    continue
                r = row.iloc[0]
                rows.append({
                    "SYMBOL":  sym, "SERIES": "EQ",
                    "OPEN":    round(float(r["Open"]),  2),
                    "HIGH":    round(float(r["High"]),  2),
                    "LOW":     round(float(r["Low"]),   2),
                    "CLOSE":   round(float(r["Close"]), 2),
                    "VOLUME":  int(r["Volume"]),
                    "DATE":    d.isoformat(),
                    "SOURCE":  "yfinance_fallback",
                })
            except Exception:
                continue

        if not rows:
            return False

        save(pd.DataFrame(rows), "equity", d)
        log.info(f"  yfinance fallback: {len(rows)} Nifty-50 stocks saved")
        return True

    except Exception as exc:
        log.error(f"  yfinance error: {exc}")
        return False


# ─────────────────────────────────────────────
# 2. F&O BHAV COPY
# ─────────────────────────────────────────────

def fetch_fo_bhav(d: date, session) -> bool:
    ds  = d.strftime("%d%b%Y").upper()
    yr  = d.strftime("%Y")
    mon = d.strftime("%b").upper()

    for url in [
        f"{NSE_ARCHIVE}/content/historical/DERIVATIVES/{yr}/{mon}/fo{ds}bhav.csv.zip",
        f"https://www1.nseindia.com/content/historical/DERIVATIVES/{yr}/{mon}/fo{ds}bhav.csv.zip",
    ]:
        raw = download(url, session)
        if raw:
            df = zip_to_df(raw)
            if df is not None and not df.empty:
                save(df, "fo", d)
                return True
    return False


# ─────────────────────────────────────────────
# 3. INDEX DATA
# ─────────────────────────────────────────────

def fetch_index_data(d: date, session) -> bool:
    ds = d.strftime("%d%m%Y")

    for url in [
        f"{NSE_ARCHIVE}/content/indices/ind_close_all_{ds}.csv",
        f"{NSE_ARCHIVE}/archives/equities/indices/ind_close_all_{ds}.csv",
    ]:
        raw = download(url, session)
        if raw:
            try:
                df = pd.read_csv(io.BytesIO(raw))
                if not df.empty:
                    save(df, "index", d)
                    return True
            except Exception as e:
                log.warning(f"  Index parse: {e}")

    return _index_via_yfinance(d)


def _index_via_yfinance(d: date) -> bool:
    try:
        import yfinance as yf
        indices = {
            "^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY BANK",
            "^CNXIT": "NIFTY IT", "^CNXPHARMA": "NIFTY PHARMA",
        }
        rows = []
        for ticker, name in indices.items():
            try:
                df = yf.download(ticker, start=d.isoformat(),
                                 end=(d+timedelta(days=1)).isoformat(), progress=False)
                if df.empty:
                    continue
                r = df.iloc[0]
                rows.append({"Index Name": name,
                             "Open": round(float(r["Open"]), 2),
                             "High": round(float(r["High"]), 2),
                             "Low":  round(float(r["Low"]),  2),
                             "Close":round(float(r["Close"]),2),
                             "Date": d.isoformat(), "Source": "yfinance_fallback"})
            except Exception:
                continue
        if rows:
            save(pd.DataFrame(rows), "index", d)
            return True
    except Exception as exc:
        log.warning(f"  Index yfinance: {exc}")
    return False


# ─────────────────────────────────────────────
# 4. DELIVERY DATA
# ─────────────────────────────────────────────

def fetch_delivery_data(d: date, session) -> bool:
    ds  = d.strftime("%d%m%Y")
    ds2 = d.strftime("%d%b%Y").upper()

    for url in [
        f"{NSE_ARCHIVE}/archives/equities/mto/MTO_{ds}.DAT",
        f"{NSE_ARCHIVE}/archives/equities/bhavcopy/sec_bhavdata{ds2}.csv",
    ]:
        raw = download(url, session)
        if raw:
            try:
                if url.endswith(".DAT"):
                    df = pd.read_csv(io.BytesIO(raw), header=None, skiprows=3,
                                     names=["RecType","SrNo","NAME","DELIV_QTY",
                                            "DELIV_VAL","TOTTRDQTY","TOTTRDVAL","DELIV_PER"])
                    df = df[df["RecType"] == 20]
                else:
                    df = pd.read_csv(io.BytesIO(raw))
                if not df.empty:
                    save(df, "delivery", d)
                    return True
            except Exception as e:
                log.warning(f"  Delivery parse: {e}")
    return False


# ─────────────────────────────────────────────
# 5. CORPORATE ACTIONS
# ─────────────────────────────────────────────

def fetch_corporate_actions(d: date, session) -> bool:
    from_d = (d - timedelta(days=7)).strftime("%d-%m-%Y")
    to_d   = d.strftime("%d-%m-%Y")
    url    = (f"{NSE_WWW}/corporates/shownCorporateActions?index=equities"
              f"&from_date={from_d}&to_date={to_d}")
    try:
        session.headers.update(
            {"Referer": f"{NSE_WWW}/companies-listing/corporate-filings/corporate-actions"}
        )
    except Exception:
        pass

    raw = download(url, session)
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                save(df, "corporate_actions", d)
                master = DATA_DIR / "corporate_actions" / "master.csv"
                master.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(master, mode="a", header=not master.exists(), index=False)
                log.info(f"  Corporate actions: {len(df)} rows")
                return True
            log.info("  No corporate actions this period.")
            return True
        except Exception as e:
            log.warning(f"  Corp actions parse: {e}")
    return False


# ─────────────────────────────────────────────
# 6. SME BHAV COPY
# ─────────────────────────────────────────────

def fetch_sme_bhav(d: date, session) -> bool:
    ds  = d.strftime("%d%b%Y").upper()
    yr  = d.strftime("%Y")
    mon = d.strftime("%b").upper()
    url = (f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/"
           f"SME{ds}BHAV.csv.zip")
    raw = download(url, session)
    if raw:
        df = zip_to_df(raw)
        if df is not None and not df.empty:
            save(df, "sme", d)
            return True
    return False


# ─────────────────────────────────────────────
# HOLIDAY CHECK
# ─────────────────────────────────────────────

def is_holiday(d: date, session) -> bool:
    if d.weekday() >= 5:
        return True
    ds  = d.strftime("%d%b%Y").upper()
    yr  = d.strftime("%Y")
    mon = d.strftime("%b").upper()
    url = f"{NSE_ARCHIVE}/content/historical/EQUITIES/{yr}/{mon}/cm{ds}bhav.csv.zip"
    try:
        r = session.head(url, timeout=15)
        return r.status_code == 404
    except Exception:
        return False


# ─────────────────────────────────────────────
# MANIFEST
# ─────────────────────────────────────────────

def update_manifest(d: date, results: Dict[str, bool]) -> None:
    manifest = {}
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text())
        except Exception:
            pass

    key = d.isoformat()
    manifest[key] = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "success": any(results.values()),
        "files": {},
    }
    for cat in ["equity", "fo", "index", "delivery", "corporate_actions", "sme"]:
        yr, mo = d.strftime("%Y"), d.strftime("%m")
        fp = DATA_DIR / cat / yr / mo / f"{key}.csv"
        if fp.exists():
            manifest[key]["files"][cat] = str(fp.relative_to(BASE_DIR))

    manifest = dict(sorted(manifest.items(), reverse=True))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    summary = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "total_trading_days": len(manifest),
        "successful_fetches": sum(1 for v in manifest.values() if v.get("success")),
        "categories_available": sorted(
            {cat for v in manifest.values() for cat in v.get("files", {})}
        ),
        "latest_date": next(iter(manifest), None),
        "latest_status": next(iter(manifest.values()), {}).get("results", {}),
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("  Manifest + summary updated.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(target: Optional[date] = None) -> bool:
    if target is None:
        target = date.today()

    log.info("=" * 60)
    log.info(f"  NSE Fetcher v2 (Cloudflare-proof) -- {target}")
    log.info("=" * 60)

    # NSE_SKIP_HOLIDAYS=true bypasses all weekend/holiday guards.
    # Set via the "skip_holidays" input on manual workflow runs.
    skip_holidays = os.environ.get("NSE_SKIP_HOLIDAYS", "false").lower() == "true"

    if target.weekday() >= 5:
        if skip_holidays:
            log.info(f"  {target} is a weekend but NSE_SKIP_HOLIDAYS=true -- proceeding.")
        else:
            log.info("  Weekend -- nothing to fetch. (Use skip_holidays=true to override)")
            return True

    session, engine = build_session()
    log.info(f"  Session engine: {engine}")

    if not skip_holidays and is_holiday(target, session):
        log.info(f"  {target} is a market holiday. (Use skip_holidays=true to override)")
        return True
    elif skip_holidays:
        log.info("  Holiday check skipped (NSE_SKIP_HOLIDAYS=true).")

    results: Dict[str, bool] = {}

    log.info("\n-- 1/6  EQUITY --")
    results["equity"] = fetch_equity_bhav(target, session)

    log.info("\n-- 2/6  F&O --")
    results["fo"] = fetch_fo_bhav(target, session)

    log.info("\n-- 3/6  INDEX --")
    results["index"] = fetch_index_data(target, session)

    log.info("\n-- 4/6  DELIVERY --")
    results["delivery"] = fetch_delivery_data(target, session)

    log.info("\n-- 5/6  CORPORATE ACTIONS --")
    results["corporate_actions"] = fetch_corporate_actions(target, session)

    log.info("\n-- 6/6  SME --")
    results["sme"] = fetch_sme_bhav(target, session)

    log.info("\n-- MANIFEST --")
    update_manifest(target, results)

    log.info("\n" + "=" * 60)
    for cat, ok in results.items():
        log.info(f"  {'OK' if ok else 'FAIL'}  {cat}")
    log.info("=" * 60)

    return results.get("equity", False)


if __name__ == "__main__":
    target = None
    if len(sys.argv) > 1:
        try:
            target = date.fromisoformat(sys.argv[1].strip())
        except ValueError:
            log.error(f"Bad date '{sys.argv[1]}'. Use YYYY-MM-DD.")
            sys.exit(1)
    sys.exit(0 if run(target) else 1)
