#!/usr/bin/env python
"""CLI: build or verify the frozen 5-fold patient-level split manifest.

Usage
-----
Build and write the manifest (server-side, from the real labels.csv):

    python scripts/make_split.py --labels /path/to/labels.csv

Verify an existing manifest against labels.csv:

    python scripts/make_split.py --verify --labels /path/to/labels.csv

Optional flags
--------------
--labels PATH    Path to labels.csv [default: inferred from server layout
                 /home/maia-user/Andre/data/Train/labels.csv if omitted,
                 but the flag is required to prevent silent path mistakes].
--out PATH       Output manifest path [default: configs/splits/fold_split_5cv.json].
--verify         Re-derive and check; do not write.

Exit codes
----------
0  Success.
1  Failure (bad checksum, mismatch, wrong case count, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_and_write(labels_path: str, out_path: str) -> None:
    from abus.data.split import (  # noqa: PLC0415
        N_FOLDS,
        SPLIT_SEED,
        make_fold_split,
        write_manifest,
    )

    print(f"Building 5-fold split from: {labels_path}")
    split = make_fold_split(labels_path, n_folds=N_FOLDS, seed=SPLIT_SEED)

    # Per-fold summary table
    n_total = sum(len(f) for f in split.folds)
    n_b_global = sum(1 for v in split.label_of.values() if v == "B")
    global_b_frac = n_b_global / n_total

    print(
        f"\n{'fold':>4}  {'n_cases':>7}  {'n_benign':>8}  {'n_malignant':>11}  "
        f"{'benign_fraction':>15}"
    )
    print("-" * 55)
    for k, fold in enumerate(split.folds):
        n_b = sum(1 for cid in fold if split.label_of[cid] == "B")
        n_m = len(fold) - n_b
        b_frac = n_b / len(fold)
        marker = " <-- DEGENERATE (>0.10 from global)" if abs(b_frac - global_b_frac) > 0.10 else ""
        print(f"{k:>4}  {len(fold):>7}  {n_b:>8}  {n_m:>11}  {b_frac:>15.3f}{marker}")
    print("-" * 55)
    print(
        f"{'ALL':>4}  {n_total:>7}  {n_b_global:>8}  {n_total-n_b_global:>11}  "
        f"{global_b_frac:>15.3f}  (global)"
    )

    print(f"\nWriting manifest to: {out_path}")
    write_manifest(split, out_path, labels_csv_path=labels_path)
    print(f"Manifest written: {out_path}")
    print("Done.")


def _verify(labels_path: str, manifest_path: str) -> None:
    from abus.data.split import verify_manifest  # noqa: PLC0415

    print(f"Verifying manifest:  {manifest_path}")
    print(f"Against labels.csv:  {labels_path}")
    result = verify_manifest(labels_path, manifest_path)
    if result:
        print(f"MANIFEST OK: sha256 matches — {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen 5-fold patient-level split manifest."
    )
    parser.add_argument(
        "--labels",
        required=True,
        metavar="PATH",
        help="Path to labels.csv (must have 'case_id' and 'label' columns, 100 rows).",
    )
    parser.add_argument(
        "--out",
        default="configs/splits/fold_split_5cv.json",
        metavar="PATH",
        help="Output manifest path [default: configs/splits/fold_split_5cv.json].",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-derive the split and verify the manifest; do not write.",
    )
    args = parser.parse_args()

    labels_path = str(Path(args.labels).resolve())

    try:
        if args.verify:
            _verify(labels_path, str(Path(args.out).resolve()))
        else:
            _build_and_write(labels_path, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
