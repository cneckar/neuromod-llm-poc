#!/usr/bin/env python3
"""
DEPRECATED — forwards to the single reproduction path, ``scripts/reproduce.py``.

The reproduction pipeline was consolidated into one tiered playbook that regenerates the
paper's text **and** visual findings and writes a REPRODUCTION_REPORT.md mapping every
artifact to the claim it supports. This shim preserves the old "Golden Path" command so
existing links/scripts keep working; it just translates the flags and forwards.

Mapping
-------
    reproduce_results.py                 -> scripts/reproduce.py --tier 2   (paper model, paper scale)
    reproduce_results.py --test-mode     -> scripts/reproduce.py --tier 1   (ungated gpt2, quick)
    reproduce_results.py --model X        -> ... --model X
    reproduce_results.py --output-dir D   -> ... --outdir D

See REPRODUCIBILITY.md for the tiers and the artifact->claim map, or run:
    python scripts/reproduce.py --tier 2 --list
"""

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_ROOT, "scripts", "reproduce.py")


def main():
    ap = argparse.ArgumentParser(
        description="DEPRECATED shim -> scripts/reproduce.py (see REPRODUCIBILITY.md)")
    ap.add_argument("--test-mode", action="store_true",
                    help="Quick, ungated run (forwards to --tier 1, gpt2)")
    ap.add_argument("--model", default=None, help="Text model override (forwarded)")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="(Legacy; per-condition N is fixed by the battery — not forwarded)")
    ap.add_argument("--output-dir", default="outputs/reproduction",
                    help="Output directory (forwarded as --outdir)")
    args, extra = ap.parse_known_args()

    tier = "1" if args.test_mode else "2"
    cmd = [sys.executable, _TARGET, "--tier", tier, "--outdir", args.output_dir]
    if args.model:
        cmd += ["--model", args.model]
    if extra:
        cmd += extra  # pass through any other reproduce.py flags verbatim

    sys.stderr.write(
        "\n[deprecated] reproduce_results.py now forwards to scripts/reproduce.py "
        "(the single reproduction path).\n"
        f"[deprecated] Running: {' '.join(cmd[1:])}\n"
        "[deprecated] See REPRODUCIBILITY.md; use scripts/reproduce.py directly going forward.\n\n")
    if args.n_samples is not None:
        sys.stderr.write(
            "[deprecated] --n-samples is ignored: per-condition N is fixed inside the test "
            "battery; use --tier to choose scale.\n\n")

    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
