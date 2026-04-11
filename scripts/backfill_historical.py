#!/usr/bin/env python3
"""
NSE Historical Backfill — Production Grade
============================================
Downloads NSE bhav copy data for a date range.

Features:
  • Checkpoint file — resumes from last successful date if interrupted
  • Skips dates where data already exists on disk
  • Configurable delay to be polite to NSE servers
  • Per-year progress tracking
  • Detailed final report

Usage:
  python scripts/backfill_historical.py 2020-01-01 2020-12-31
  python scripts/backfill_historical.py 2020-01-01          # to today
  python scripts/backfill_historical.py --year 2022         # whole year
  python scripts/backfill_historical.py --last-n-years 5    # last 5 years
"""

import sys
import os
import json
import time
import random
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path

# ── path setup ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from fetch_nse_data import run, build_session, is_holiday as is_market_holiday, DATA_DIR

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backfill")

# ── checkpoint ────────────────────────────────────────────────
CHECKPOINT_FILE = BASE_DIR / "data" / ".backfill_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(cp: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# ── helpers ───────────────────────────────────────────────────

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def equity_exists(d: date) -> bool:
    """Return True if equity CSV for this date already on disk."""
    yr  = d.strftime("%Y")
    mon = d.strftime("%m")
    fp  = DATA_DIR / "equity" / yr / mon / f"{d.isoformat()}.csv"
    return fp.exists() and fp.stat().st_size > 500   # >500 bytes = real data


# ── main backfill ─────────────────────────────────────────────

def backfill(start: date, end: date, delay_min: float = 2.0, delay_max: float = 5.0) -> dict:
    """
    Download NSE data for every trading day in [start, end].

    Returns a summary dict:
      success / skipped / failed / holidays / total
    """
    all_dates = [d for d in daterange(start, end) if d.weekday() < 5]
    total = len(all_dates)

    logger.info("=" * 65)
    logger.info(f"  BACKFILL: {start} → {end}")
    logger.info(f"  Weekday dates : {total}")
    logger.info("=" * 65)

    cp = load_checkpoint()
    resume_from_str = cp.get("last_failed") or cp.get("last_success")
    resume_from = date.fromisoformat(resume_from_str) if resume_from_str else None

    counts = dict(success=0, skipped=0, failed=0, holiday=0)
    failed_dates: list[str] = []

    session = build_session()   # one warm session for the whole run

    for idx, d in enumerate(all_dates, 1):
        d_str = d.isoformat()

        # ── Progress header ───────────────────────────────────
        pct = idx / total * 100
        logger.info(f"\n[{idx}/{total}  {pct:.1f}%]  {d_str}")

        # ── Skip if already downloaded ────────────────────────
        if equity_exists(d):
            logger.info("  ✓ Already on disk — skipping.")
            counts["skipped"] += 1
            continue

        # ── Market holiday check ──────────────────────────────
        if is_market_holiday(d, session):
            logger.info("  — Market holiday — skipping.")
            counts["holiday"] += 1
            cp["last_success"] = d_str
            save_checkpoint(cp)
            continue

        # ── Fetch ─────────────────────────────────────────────
        ok = run(d)

        if ok:
            counts["success"] += 1
            cp["last_success"] = d_str
            cp.pop("last_failed", None)
        else:
            counts["failed"] += 1
            failed_dates.append(d_str)
            cp["last_failed"] = d_str
            logger.warning(f"  ✗ FAILED: {d_str}")

        save_checkpoint(cp)

        # ── Polite delay ──────────────────────────────────────
        if idx < total:
            sleep = random.uniform(delay_min, delay_max)
            logger.info(f"  Sleeping {sleep:.1f}s…")
            time.sleep(sleep)

        # ── Refresh session every 50 dates to avoid expiry ────
        if idx % 50 == 0:
            logger.info("  Refreshing NSE session…")
            session = build_session()

    # ── Report ────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  BACKFILL COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Total weekdays  : {total}")
    logger.info(f"  ✓ Downloaded    : {counts['success']}")
    logger.info(f"  ↷ Already had   : {counts['skipped']}")
    logger.info(f"  — Holidays      : {counts['holiday']}")
    logger.info(f"  ✗ Failed        : {counts['failed']}")
    if failed_dates:
        logger.info("  Failed dates:")
        for fd in failed_dates:
            logger.info(f"    • {fd}")
    logger.info("=" * 65)

    if counts["failed"] == 0:
        clear_checkpoint()
        logger.info("  Checkpoint cleared (clean run).")

    return {**counts, "total": total, "failed_dates": failed_dates}


# ── CLI ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NSE Historical Backfill")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--year",          type=int,  help="Download a whole year, e.g. --year 2022")
    group.add_argument("--last-n-years",  type=int,  help="Download last N years, e.g. --last-n-years 5")
    group.add_argument("start_date",      nargs="?", help="Start date YYYY-MM-DD")

    p.add_argument("end_date", nargs="?", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--delay-min", type=float, default=2.0, help="Min sleep between dates (s)")
    p.add_argument("--delay-max", type=float, default=5.0, help="Max sleep between dates (s)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    today = date.today()

    if args.year:
        start = date(args.year, 1, 1)
        end   = date(args.year, 12, 31)
        if end > today:
            end = today

    elif args.last_n_years:
        start = date(today.year - args.last_n_years, today.month, today.day)
        end   = today

    else:
        start = date.fromisoformat(args.start_date)
        end   = date.fromisoformat(args.end_date) if args.end_date else today

    result = backfill(start, end, args.delay_min, args.delay_max)

    # Exit non-zero only if critical failures
    if result["failed"] > result["success"] + result["skipped"]:
        sys.exit(1)
    sys.exit(0)
