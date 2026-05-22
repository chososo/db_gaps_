#!/usr/bin/env python
"""End-to-end: fetch → build report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.data.pipeline import run_daily_update
from db_gaps.report import build_site
from db_gaps.utils.logging import get_logger

LOG = get_logger("db_gaps.full")


def main() -> int:
    summary = run_daily_update(force_full=False)
    LOG.info("Fetch summary: %s", summary)
    out = build_site()
    LOG.info("Report built at %s", out)
    return 0 if summary.get("fail", 0) == 0 else 0  # don't fail the pipeline on partial fetch errors


if __name__ == "__main__":
    sys.exit(main())
