#!/usr/bin/env python
"""Verify the nnDetection dataset: splits faithfulness + bbox round-trip (STORY_01_01).

Usage (server):
    # Single case:
    python scripts/verify_nndet_dataset.py \\
        --task-dir  /home/maia-user/nndet_data/Task001_TDSCABUS \\
        --tdsc-root /home/maia-user/Andre/data \\
        --case-id   0

    # All 100 train cases (Checkpoint 7 D01.8 re-run):
    python scripts/verify_nndet_dataset.py \\
        --task-dir  /home/maia-user/nndet_data/Task001_TDSCABUS \\
        --tdsc-root /home/maia-user/Andre/data \\
        --case-id   all

This script:
  1. Runs verify_nndet_dataset — checks spacing in every imagesTr/ file, splits
     faithfulness (ASC-01_01.3), and dataset.json presence.
  2. For the specified --case-id (int or "all"), runs the empirical bbox round-trip
     using bbox_roundtrip_residuals (D01.8 two-clause gate):
     a. Loads the written nnDetection image to determine the target grid spacing.
     b. Reads the GT bbox from bbx_labels.csv for that case.
     c. Calls bbox_roundtrip_residuals(gt_bbox, target_spacing_mm).
     d. Evaluates both gate clauses:
        - primary: max_residual_mm <= 0.5 mm (D01_8_PRIMARY_MM)
        - structural: per-axis residuals_vx within analytic envelope (3,2,2) vx
  3. When --case-id all: iterates 0..99, prints per-case pass/fail, then an
     aggregate summary (n_pass, n_fail, max_residual_mm across all cases).
  4. Exits 0 if all checks pass; exits 1 with a descriptive error if any fail.

The output of this script is pasted into docs/results/STORY_01_01_results.md
under "Checkpoint 7 — verification (D01.8 re-run)".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abus.geometry.bbox import BBox

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
        type=str,
        required=False,
        default=None,
        help=(
            "Case ID for empirical bbox round-trip test (ASC-01_01.4, D01.8). "
            "Pass an integer (e.g. 0) for a single case, or 'all' to iterate "
            "over all 100 train cases and print per-case + aggregate summaries."
        ),
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

        from abus.detect.nndet_convention import D01_8_PRIMARY_MM, bbox_roundtrip_residuals
        from abus.geometry.convert import csv_itk_to_bbox
        from abus.io.loader import CANONICAL_SPACING_MM

        tdsc_root = Path(args.tdsc_root)

        # Determine which case IDs to process
        run_all = args.case_id.strip().lower() == "all"
        if run_all:
            case_ids = list(range(100))
            log.info("Running bbox round-trip for all 100 train cases (D01.8 re-run).")
        else:
            try:
                case_ids = [int(args.case_id)]
            except ValueError:
                log.error("--case-id must be an integer or 'all', got: %r", args.case_id)
                sys.exit(1)

        def _load_gt_bbox(case_id: int) -> BBox | None:
            """Load GT bbox for a single case from bbx_labels.csv."""
            for split_name in ("Train", "Validation"):
                csv_path = tdsc_root / split_name / "bbx_labels.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                rows = df[df["id"] == case_id]
                if not rows.empty:
                    row = rows.iloc[0]
                    return csv_itk_to_bbox(
                        (float(row["c_x"]), float(row["c_y"]), float(row["c_z"])),
                        (float(row["len_x"]), float(row["len_y"]), float(row["len_z"])),
                    )
            return None

        def _load_target_spacing(case_id: int) -> tuple:
            """Read target spacing from the written nnDetection image header."""
            img_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id:04d}_0000.nii.gz"
            if not img_path.exists():
                log.warning(
                    "Image not found at %s; falling back to CANONICAL_SPACING_MM.",
                    img_path,
                )
                return CANONICAL_SPACING_MM
            img = sitk.ReadImage(str(img_path))
            sp = img.GetSpacing()  # (x, y, z) = (d2, d1, d0)
            return (sp[2], sp[1], sp[0])  # -> (d0, d1, d2) storage order

        per_case_results = []
        any_failed = False

        for case_id in case_ids:
            gt_bbox = _load_gt_bbox(case_id)
            if gt_bbox is None:
                log.error("Case %d not found in any split's bbx_labels.csv.", case_id)
                sys.exit(1)

            target_spacing = _load_target_spacing(case_id)
            if len(case_ids) == 1:
                log.info(
                    "GT bbox for case %d: %s; target spacing: %s mm",
                    case_id,
                    gt_bbox,
                    target_spacing,
                )

            result = bbox_roundtrip_residuals(gt_bbox, target_spacing)
            result["case_id"] = case_id
            result["target_spacing_mm"] = list(target_spacing)
            per_case_results.append(result)

            gate_pass = result["gate_pass"]
            primary_pass = result["primary_gate_pass"]
            structural_pass = result["structural_gate_pass"]
            mrm = result["max_residual_mm"]
            rvx = result["residuals_vx"]
            evx = result["envelope_vx_per_axis"]

            if gate_pass:
                log.info(
                    "GATE PASSED (ASC-01_01.4, D01.8): max_residual_mm = %.3f mm"
                    " <= %.1f mm; per-axis residuals_vx = %s within envelope %s.",
                    mrm,
                    D01_8_PRIMARY_MM,
                    rvx,
                    evx,
                )
            else:
                any_failed = True
                if not primary_pass:
                    log.error(
                        "GATE FAILED (ASC-01_01.4, D01.8, primary): max_residual_mm"
                        " = %.3f mm > %.1f mm. Residual is above the physical"
                        " round-trip envelope — bbox convention or resampling chain"
                        " is wrong. [case %d]",
                        mrm,
                        D01_8_PRIMARY_MM,
                        case_id,
                    )
                if not structural_pass:
                    # Find the offending axis
                    per_axis_vx = (
                        max(rvx[0], rvx[3]),
                        max(rvx[1], rvx[4]),
                        max(rvx[2], rvx[5]),
                    )
                    for axis in range(3):
                        if per_axis_vx[axis] > evx[axis]:
                            log.error(
                                "GATE FAILED (ASC-01_01.4, D01.8, structural):"
                                " per-axis residual %d on axis d%d exceeds envelope %d."
                                " Pattern suggests an axis-swap or"
                                " inclusive/exclusive-max bug, not pure quantisation."
                                " [case %d]",
                                per_axis_vx[axis],
                                axis,
                                evx[axis],
                                case_id,
                            )

        if run_all:
            n_pass = sum(1 for r in per_case_results if r["gate_pass"])
            n_fail = len(per_case_results) - n_pass
            all_max_mrm = max(r["max_residual_mm"] for r in per_case_results)
            aggregate = {
                "n_cases": len(per_case_results),
                "n_pass": n_pass,
                "n_fail": n_fail,
                "max_residual_mm_across_all": all_max_mrm,
                "primary_mm_threshold": D01_8_PRIMARY_MM,
                "envelope_vx_per_axis": (
                    per_case_results[0]["envelope_vx_per_axis"] if per_case_results else None
                ),
                "all_pass": not any_failed,
            }
            log.info(
                "D01.8 aggregate: %d/%d pass; worst max_residual_mm = %.3f mm.",
                n_pass,
                len(per_case_results),
                all_max_mrm,
            )
            roundtrip_report = {
                "mode": "all",
                "aggregate": aggregate,
                "per_case": per_case_results,
            }
        else:
            roundtrip_report = per_case_results[0]

    # --- Final output --- (always printed, even on gate failure, for diagnostics)
    full_report = {
        "verify_nndet_dataset": report,
        "bbox_roundtrip": roundtrip_report if roundtrip_report else "not_run",
    }
    print(json.dumps(full_report, indent=2))

    if args.case_id is not None and any_failed:
        log.error("verify_nndet_dataset.py: GATE FAILED — do not proceed to STORY_01_02.")
        sys.exit(1)

    log.info("verify_nndet_dataset.py: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
