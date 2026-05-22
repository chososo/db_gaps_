#!/usr/bin/env python
"""Daily data fetch entry point. Called by GitHub Actions cron."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.data.pipeline import run_daily_update
from db_gaps.utils.logging import get_logger

LOG = get_logger("db_gaps.cli")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch all ETF data and build processed matrices.")
    ap.add_argument("--full", action="store_true", help="Force full re-fetch (ignore cache).")
    args = ap.parse_args()

    summary = run_daily_update(force_full=args.full)
    LOG.info("Daily update summary: %s", summary)
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
