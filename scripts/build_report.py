#!/usr/bin/env python
"""Build the multi-page HTML report into docs/ for GitHub Pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.report import build_site
from db_gaps.utils.logging import get_logger

LOG = get_logger("db_gaps.report_cli")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=None, help="strategies to include (default: settings active or all built-ins)")
    ap.add_argument("--out", default=None, help="output directory (default: docs/)")
    args = ap.parse_args()

    out = build_site(strategies=args.strategies, output_dir=args.out)
    LOG.info("Site built at %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
