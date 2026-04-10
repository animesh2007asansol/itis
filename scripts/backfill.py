#!/usr/bin/env python3
"""
NSE Historical Backfill Script
================================
Run this ONCE to populate historical data.

Usage:
  python scripts/backfill.py                      # Last 30 days
  python scripts/backfill.py 2024-01-01           # From date to today
  python scripts/backfill.py 2024-01-01 2024-03-31  # Date range

WARNING: GitHub Actions free tier has limits. Run this locally
or in a GitHub Actions job with a long timeout.
"""

import sys
from datetime import date, timedelta
from fetch_nse_data import run
import logging, time, random

logger = logging.getLogger("backfill")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

if __name__ == "__main__":
    today = date.today()

    if len(sys.argv) == 1:
        start_date = today - timedelta(days=30)
        end_date   = today
    elif len(sys.argv) == 2:
        start_date = date.fromisoformat(sys.argv[1])
        end_date   = today
    elif len(sys.argv) == 3:
        start_date = date.fromisoformat(sys.argv[1])
        end_date   = date.fromisoformat(sys.argv[2])
    else:
        print("Usage: python backfill.py [start_date] [end_date]")
        sys.exit(1)

    logger.info(f"Backfilling {start_date} → {end_date}")
    success_count = fail_count = skip_count = 0

    for d in daterange(start_date, end_date):
        if d.weekday() >= 5:
            logger.info(f"  {d} – weekend, skip")
            skip_count += 1
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"  Processing {d}")
        logger.info(f"{'='*50}")

        ok = run(d)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        # Be polite to NSE servers
        sleep = random.uniform(3, 7)
        logger.info(f"  Sleeping {sleep:.1f}s…")
        time.sleep(sleep)

    logger.info("\n" + "="*50)
    logger.info(f"  BACKFILL COMPLETE")
    logger.info(f"  Success : {success_count}")
    logger.info(f"  Failed  : {fail_count}")
    logger.info(f"  Skipped : {skip_count}")
    logger.info("="*50)
