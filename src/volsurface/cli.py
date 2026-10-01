"""Command-line entry point: ``volsurface collect`` and ``volsurface build``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from volsurface.collect import DATA_DIR, collect_snapshot
from volsurface.pipeline import DERIVED_DIR, process_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="volsurface")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="fetch today's option-chain snapshot")
    p_collect.add_argument("--out", type=Path, default=DATA_DIR)
    p_collect.add_argument("--tickers", nargs="*", default=None)
    p_collect.add_argument("--force", action="store_true", help="collect on non-session days")

    p_proc = sub.add_parser("process", help="clean, fit and analyse every snapshot")
    p_proc.add_argument("--data", type=Path, default=DATA_DIR)
    p_proc.add_argument("--derived", type=Path, default=DERIVED_DIR)
    p_proc.add_argument("--force", action="store_true", help="ignore cached results")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "collect":
        snap = collect_snapshot(args.out, args.tickers, args.force)
        print(snap if snap else "skipped")
    elif args.cmd == "process":
        process_all(args.data, args.derived, args.force)


if __name__ == "__main__":
    main()
