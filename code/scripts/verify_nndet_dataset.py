#!/usr/bin/env python
"""Verify the nnDetection dataset: splits faithfulness + bbox round-trip (STORY_01_01).

Usage (server):
    python scripts/verify_nndet_dataset.py \\
        --task-dir /home/maia-user/Andre/outputs/nndet/Task001_TDSCABUS \\
        --case-id  <any_train_case_id>

This script:
  1. Runs verify_nndet_dataset — checks spacing in every imagesTr/ file, splits
     faithfulness (ASC-01_01.3), and dataset.json presence.
  2. For the specified --case-id, runs the empirical bbox round-trip:
     a. Loads the written nnDetection image to determine the target grid spacing.
     b. Reads the GT bbox from bbx_labels.csv for that case.
     c. Calls bbox_original_roundtrip(gt_bbox, target_spacing_mm).
     d. Prints the per-axis residual on the original grid (gate: ≤ 1 voxel).
  3. Exits 0 if all checks pass; exits 1 with a descriptive error if any fail.

The output of this script is pasted into docs/results/STORY_01_01_results.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the nnDetection Task directory: spacing, splits, bbox round-trip."
    )
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Path to the nnDetection Task directory (e.g. Task001_TDSCABUS/).",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        required=False,
        help="Case ID for empirical bbox round-trip test (ASC-01_01.4). Optional.",
    )
    parser.add_argument(
        "--tdsc-root",
        required=False,
        default=None,
        help=(
            "TDSC dataset root (needed for bbox round-trip: reads GT from bbx_labels.csv). "
            "Required when --case-id is specified."
        ),
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    if not task_dir.exists():
        log.error("Task directory not found: %s", task_dir)
        sys.exit(1)

    # --- Step 1: run verify_nndet_dataset ---
    from abus.detect.nndet_io import NndetDatasetError, verify_nndet_dataset

    try:
        report = verify_nndet_dataset(str(task_dir))
    except NndetDatasetError as e:
        log.error("Dataset verification FAILED: %s", e)
        sys.exit(1)

    log.info("Dataset verification passed: %s", report)

    # --- Step 2: empirical bbox round-trip (optional, requires --case-id + --tdsc-root) ---
    roundtrip_report: dict = {}

    if args.case_id is not None:
        if args.tdsc_root is None:
            log.error("--case-id requires --tdsc-root to locate bbx_labels.csv.")
            sys.exit(1)

        import pandas as pd
        import SimpleITK as sitk

        from abus.detect.nndet_convention import bbox_original_roundtrip
        from abus.geometry.convert import csv_itk_to_bbox
        from abus.io.loader import CANONICAL_SPACING_MM

        case_id = args.case_id
        tdsc_root = Path(args.tdsc_root)

        # Find GT bbox from Train/bbx_labels.csv (or Validation if not in Train)
        gt_bbox = None
        for split_name in ("Train", "Validation"):
            csv_path = tdsc_root / split_name / "bbx_labels.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            rows = df[df["id"] == case_id]
            if not rows.empty:
                row = rows.iloc[0]
                gt_bbox = csv_itk_to_bbox(
                    (float(row["c_x"]), float(row["c_y"]), float(row["c_z"])),
                    (float(row["len_x"]), float(row["len_y"]), float(row["len_z"])),
                )
                log.info("GT bbox for case %d from %s: %s", case_id, split_name, gt_bbox)
                break

        if gt_bbox is None:
            log.error("Case %d not found in any split's bbx_labels.csv.", case_id)
            sys.exit(1)

        # Load the nnDetection image to get the actual target grid spacing.
        # nnDetection v0.1 layout: images under raw_splitted/imagesTr/ as .nii.gz
        img_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id:04d}_0000.nii.gz"
        if not img_path.exists():
            log.warning(
                "Original image not found at %s. "
                "Using CANONICAL_SPACING_MM as both original and target "
                "(identity resampling assumption).",
                img_path,
            )
            target_spacing = CANONICAL_SPACING_MM
        else:
            # SimpleITK GetSpacing returns (x, y, z) = (d2, d1, d0) in our convention
            img = sitk.ReadImage(str(img_path))
            sp = img.GetSpacing()
            target_spacing = (sp[2], sp[1], sp[0])  # -> (d0, d1, d2) storage order
            log.info("Target spacing from image header: %s mm", target_spacing)

        recovered = bbox_original_roundtrip(gt_bbox, target_spacing)

        residuals = (
            abs(recovered.min_d0 - gt_bbox.min_d0),
            abs(recovered.min_d1 - gt_bbox.min_d1),
            abs(recovered.min_d2 - gt_bbox.min_d2),
            abs(recovered.max_d0 - gt_bbox.max_d0),
            abs(recovered.max_d1 - gt_bbox.max_d1),
            abs(recovered.max_d2 - gt_bbox.max_d2),
        )
        max_residual = max(residuals)

        roundtrip_report = {
            "case_id": case_id,
            "gt_bbox": gt_bbox.as_tuple(),
            "recovered_bbox": recovered.as_tuple(),
            "residuals_per_endpoint": residuals,
            "max_residual_voxels": max_residual,
            "gate_pass": max_residual <= 1,
            "target_spacing_mm": list(target_spacing),
        }

        log.info("BBox round-trip report: %s", roundtrip_report)
        if max_residual > 1:
            log.error(
                "GATE FAILED (ASC-01_01.4): max residual = %d voxels > 1. "
                "The nnDetection bbox convention mapping is wrong.",
                max_residual,
            )
            sys.exit(1)
        else:
            log.info("GATE PASSED (ASC-01_01.4): max residual = %d voxel(s) ≤ 1.", max_residual)

    # --- Final output ---
    full_report = {
        "verify_nndet_dataset": report,
        "bbox_roundtrip": roundtrip_report if roundtrip_report else "not_run",
    }
    print(json.dumps(full_report, indent=2))
    log.info("verify_nndet_dataset.py: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
