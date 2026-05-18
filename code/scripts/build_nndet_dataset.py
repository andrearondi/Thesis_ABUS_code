#!/usr/bin/env python
"""Build the nnDetection dataset from TDSC-ABUS-2023 (STORY_01_01, server-side).

Usage (server):
    python scripts/build_nndet_dataset.py \\
        --tdsc-root /home/maia-user/Andre/data \\
        --out-root  /home/maia-user/Andre/outputs/nndet \\
        --config    configs/detect/nndet_dataset.yaml

This script:
  1. Reads the pinned config from ``--config`` (requires pyyaml; install via runbook).
  2. Runs export_dataset over all 200 TDSC cases (100 Train + 30 Val + 70 Test).
  3. Writes the nnDetection Task directory under ``<out_root>/<task_name>/``.
  4. Writes dataset.json and splits_final.json (derived from frozen 5-fold manifest).
  5. Prints a conversion summary.

Resource estimate: CPU-bound; ~10–30 wall-clock minutes for 200 NRRD reads/writes;
0 GPU-hours.

See configs/detect/nndet_planning_notes.md for the full nnDetection CLI invocation
plan that follows this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict[str, Any]:
    """Load the YAML config file. Requires pyyaml (install via runbook)."""
    try:
        import yaml
    except ImportError:
        log.error(
            "pyyaml is not installed. Install it with: pip install 'pyyaml>=6.0.2,<7'\n"
            "(authorized by epic-approval gate 2026-05-18, docs/decisions_log.md D01.5)"
        )
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TDSC-ABUS-2023 to nnDetection Task format."
    )
    parser.add_argument(
        "--tdsc-root",
        required=True,
        help="Root of TDSC-ABUS-2023 dataset (contains Train/, Validation/, Test/).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root; the Task directory is created here.",
    )
    parser.add_argument(
        "--config",
        default="configs/detect/nndet_dataset.yaml",
        help="Path to nndet_dataset.yaml (default: configs/detect/nndet_dataset.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths only; do not convert.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from abus.detect.nndet_io import NndetDatasetSpec, export_dataset
    from abus.io.loader import CANONICAL_SPACING_MM

    # Validate spacing matches CANONICAL_SPACING_MM (fail loud if config drifted)
    cfg_spacing = tuple(float(x) for x in cfg["spacing_mm"])
    if cfg_spacing != CANONICAL_SPACING_MM:
        log.error(
            "Config spacing_mm %s does not match CANONICAL_SPACING_MM %s. "
            "This would corrupt the dataset. Aborting.",
            cfg_spacing,
            CANONICAL_SPACING_MM,
        )
        sys.exit(1)

    spec = NndetDatasetSpec(
        task_id=int(cfg["task_id"]),
        task_name=str(cfg["task_name"]),
        n_train_cases=int(cfg["n_train_cases"]),
        n_val_cases=int(cfg["n_val_cases"]),
        n_test_cases=int(cfg["n_test_cases"]),
        spacing_mm=cfg_spacing,  # type: ignore[arg-type]
        modality=str(cfg["modality"]),
        label_semantics=str(cfg["label_semantics"]),
    )

    log.info("Task: %s (id=%d)", spec.task_name, spec.task_id)
    log.info("TDSC root: %s", args.tdsc_root)
    log.info("Output root: %s", args.out_root)
    log.info("Spacing to write: %s mm", spec.spacing_mm)

    if args.dry_run:
        log.info("--dry-run: config validated. No files written.")
        print(json.dumps({"status": "dry-run-ok", "spec": spec.task_name}, indent=2))
        return

    result = export_dataset(
        tdsc_root=args.tdsc_root,
        out_root=args.out_root,
        spec=spec,
    )

    log.info("Conversion complete.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
