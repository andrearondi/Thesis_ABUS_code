#!/usr/bin/env python
"""Server CLI: train one or all five nnDetection fold detectors (STORY_01_02).

Usage
-----
Train a single fold:
    python scripts/train_all_folds.py --fold 0

Train all folds sequentially:
    python scripts/train_all_folds.py --all

Train fold 0 only and print the cost-checkpoint extrapolation:
    python scripts/train_all_folds.py --fold 0 --cost-checkpoint

Required environment variables:
    NNDET_PREPROCESSED   — path to the nnDetection preprocessed Task directory
    DET_MODELS           — path for nnDetection training artefacts (checkpoints, logs)
    NNDET_DIR            — path to the nnDetection source repo (for commit hash)

All five folds are trained strictly sequentially on a single GPU (user decision
D01.5, 2026-05-18). Job 1 (fold 0) is always run alone first to enable the
cost-checkpoint after fold 0 (ASC-01_02.2, EPIC_01 risk #1).

After fold 0 completes, this script:
  1. Reports fold-0 wall-clock and GPU-hours.
  2. Extrapolates the five-fold total GPU-hours.
  3. Checks against the ~1300 GPU-h trigger (configs/detect/training.yaml).
  4. Prints an explicit GO / FALLBACK prompt for the user.
  5. If running --all and the extrapolation exceeds the trigger, STOPS before
     fold 1 and instructs the user to make the GO / FALLBACK decision.

After each fold completes, intermediate periodic checkpoints are pruned:
only the best-metric and the final ("last") checkpoint are retained.

Reproducibility: the nnDetection commit hash is recorded in the DetectorRun
metadata and printed at the start of each fold run (agent_rules §12).

GPU note: this script runs on the server (GPU required). It is NOT run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add src/ to path for local import when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abus.detect.train import DetectorRun, train_fold_detector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAME = "Task001_TDSCABUS"
COST_TRIGGER_GPU_H = 1300.0
N_FOLDS = 5
ARCHITECTURE = "RetinaUNet000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nndet_commit(nndet_dir: str | None) -> str:
    """Get the nnDetection commit hash for reproducibility."""
    if nndet_dir is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", nndet_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # short hash
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def _prune_intermediate_checkpoints(ckpt_dir: Path) -> None:
    """Remove nnDetection's intermediate periodic checkpoints.

    Retains:
      - model_best.ckpt  (best-metric)
      - model_final.ckpt or checkpoint_final.pth  (last epoch)

    Prunes anything matching epoch_*.ckpt / checkpoint_ep*.pth patterns.
    """
    retain_patterns = {"model_best", "model_final", "checkpoint_final"}
    pruned = 0
    for ckpt_file in ckpt_dir.glob("*.ckpt"):
        if not any(pat in ckpt_file.stem for pat in retain_patterns):
            ckpt_file.unlink()
            pruned += 1
    for ckpt_file in ckpt_dir.glob("*.pth"):
        if not any(pat in ckpt_file.stem for pat in retain_patterns):
            ckpt_file.unlink()
            pruned += 1
    if pruned:
        print(f"  Pruned {pruned} intermediate checkpoint(s) from {ckpt_dir}")


def _cost_checkpoint(fold0_run: DetectorRun, trigger: float) -> dict:
    """Compute and print the fold-0 cost checkpoint.

    Returns a dict with the extrapolation and whether the trigger was exceeded.
    """
    five_fold_extrap = fold0_run.gpu_hours * N_FOLDS
    print("\n" + "=" * 60)
    print("FOLD-0 COST CHECKPOINT (ASC-01_02.2)")
    print("=" * 60)
    print(f"  Fold 0 wall-clock: {fold0_run.wall_clock_hours:.2f} h")
    print(f"  Fold 0 GPU-hours:  {fold0_run.gpu_hours:.2f} GPU-h")
    print(f"  Five-fold extrapolation: {five_fold_extrap:.0f} GPU-h")
    print(f"  Trigger threshold: ~{trigger:.0f} GPU-h")

    exceeded = five_fold_extrap > trigger
    if exceeded:
        print("\n  *** TRIGGER EXCEEDED — FALLBACK LADDER ACTIVATED ***")
        print(f"  Extrapolated {five_fold_extrap:.0f} GPU-h > {trigger:.0f} GPU-h limit.")
        print("  STOP: Do NOT start fold 1 until the GO / FALLBACK decision is recorded.")
        print("  See IMPLEMENTATION_PLAN §'Compute budget envelope' for the fallback ladder.")
        print("  Paste this output into docs/results/STORY_01_02_results.md and decide.")
    else:
        print("\n  GO: Extrapolation within budget. Proceed with folds 1–4.")

    print("=" * 60)
    return {
        "fold0_wall_clock_hours": fold0_run.wall_clock_hours,
        "fold0_gpu_hours": fold0_run.gpu_hours,
        "five_fold_extrap_gpu_hours": five_fold_extrap,
        "trigger_gpu_hours": trigger,
        "trigger_exceeded": exceeded,
        "final_val_metric_fold0": fold0_run.final_val_metric,
    }


def _intra_fold0_probe_checkpoint(
    probe_epoch: int,
    gpu_h_per_epoch: float,
    planned_epochs: int,
    trigger: float,
) -> None:
    """Print the intra-fold-0 epoch-rate cost probe (D01.6 requirement).

    This is called within fold 0 training (if supported) to provide an early
    five-fold cost extrapolation BEFORE fold 0 completes. The runbook includes
    a checkpoint for this probe.

    Parameters
    ----------
    probe_epoch:
        The epoch at which the probe was taken.
    gpu_h_per_epoch:
        Measured GPU-hours per epoch.
    planned_epochs:
        Total planned epochs for one fold.
    trigger:
        Cost trigger threshold in GPU-h.
    """
    per_fold_extrap = gpu_h_per_epoch * planned_epochs
    five_fold_extrap = per_fold_extrap * N_FOLDS
    print("\n" + "-" * 60)
    print(f"INTRA-FOLD-0 COST PROBE (epoch {probe_epoch}) — D01.6 requirement")
    print("-" * 60)
    print(f"  GPU-h/epoch (measured):  {gpu_h_per_epoch:.4f}")
    print(f"  Planned epochs/fold:      {planned_epochs}")
    print(f"  Per-fold extrapolation:   {per_fold_extrap:.0f} GPU-h")
    print(f"  Five-fold extrapolation:  {five_fold_extrap:.0f} GPU-h")
    print(f"  Trigger threshold:        ~{trigger:.0f} GPU-h")
    if five_fold_extrap > trigger:
        print("  WARNING: Five-fold extrapolation EXCEEDS trigger threshold.")
        print("  Paste this early probe into docs/results/STORY_01_02_results.md.")
        print("  Do not wait for fold 0 to complete before escalating.")
    else:
        print("  Within budget at current epoch rate.")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train nnDetection fold detectors for TDSC-ABUS-2023 (STORY_01_02)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fold",
        type=int,
        choices=list(range(N_FOLDS)),
        help="Train a single fold (0–4).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all 5 folds sequentially. Stops after fold 0 if trigger exceeded.",
    )
    parser.add_argument(
        "--cost-checkpoint",
        action="store_true",
        default=True,
        help=(
            "Print fold-0 cost checkpoint extrapolation (default: True). "
            "Only meaningful for --fold 0 or --all."
        ),
    )
    parser.add_argument(
        "--nndet-preprocessed",
        default=os.environ.get("NNDET_PREPROCESSED", ""),
        help="nnDetection preprocessed Task directory (env: NNDET_PREPROCESSED).",
    )
    parser.add_argument(
        "--det-models",
        default=os.environ.get("DET_MODELS", ""),
        help="nnDetection model output root (env: DET_MODELS).",
    )
    parser.add_argument(
        "--nndet-dir",
        default=os.environ.get("NNDET_DIR", None),
        help="nnDetection source repo root (for commit hash; env: NNDET_DIR).",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional path to write a JSON summary of all fold runs.",
    )
    args = parser.parse_args()

    if not args.nndet_preprocessed:
        parser.error(
            "--nndet-preprocessed or $NNDET_PREPROCESSED must be set "
            "(path to the nnDetection preprocessed Task directory)."
        )
    if not args.det_models:
        parser.error(
            "--det-models or $DET_MODELS must be set " "(path for nnDetection training artefacts)."
        )

    nndet_commit = _get_nndet_commit(args.nndet_dir)
    print(f"nnDetection commit: {nndet_commit}")
    print(f"Task: {TASK_NAME}")
    print(f"Architecture: {ARCHITECTURE}")

    folds_to_train = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    all_runs: list[DetectorRun] = []

    for fold in folds_to_train:
        print(f"\n{'=' * 60}")
        print(f"TRAINING FOLD {fold} / {N_FOLDS - 1}")
        print(f"{'=' * 60}")

        try:
            run = train_fold_detector(
                fold=fold,
                nndet_dataset_root=args.nndet_preprocessed,
                out_root=args.det_models,
                task_name=TASK_NAME,
                nndet_commit=nndet_commit,
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"\nERROR training fold {fold}: {e}")
            sys.exit(1)

        all_runs.append(run)
        print(f"\nFold {fold} complete.")
        print(f"  Checkpoint: {run.checkpoint_path}")
        print(f"  Wall-clock: {run.wall_clock_hours:.2f} h")
        print(f"  GPU-hours:  {run.gpu_hours:.2f} GPU-h")
        print(f"  Final val metric: {run.final_val_metric:.4f}")

        # Prune intermediate checkpoints (user decision D01.5)
        ckpt_dir = Path(args.det_models) / TASK_NAME / ARCHITECTURE / str(fold)
        if ckpt_dir.exists():
            _prune_intermediate_checkpoints(ckpt_dir)

        # Cost checkpoint after fold 0
        if fold == 0 and args.cost_checkpoint:
            checkpoint_result = _cost_checkpoint(run, COST_TRIGGER_GPU_H)
            if checkpoint_result["trigger_exceeded"] and args.all:
                print(
                    "\nSTOP: Cost trigger exceeded. Exiting before fold 1. "
                    "Record the GO / FALLBACK decision in "
                    "docs/results/STORY_01_02_results.md before continuing."
                )
                sys.exit(2)  # exit code 2 = trigger exceeded (not an error)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    total_gpu_h = sum(r.gpu_hours for r in all_runs)
    for run in all_runs:
        print(
            f"  Fold {run.fold}: {run.gpu_hours:.2f} GPU-h, "
            f"val_metric={run.final_val_metric:.4f}, "
            f"ckpt={run.checkpoint_path}"
        )
    print(f"  Total GPU-hours: {total_gpu_h:.2f}")
    print("=" * 60)

    # Write JSON summary if requested
    if args.results_json:
        summary = {
            "folds": [
                {
                    "fold": r.fold,
                    "checkpoint_path": r.checkpoint_path,
                    "wall_clock_hours": r.wall_clock_hours,
                    "gpu_hours": r.gpu_hours,
                    "final_val_metric": r.final_val_metric,
                    "nndet_commit": r.nndet_commit,
                }
                for r in all_runs
            ],
            "total_gpu_hours": total_gpu_h,
            "nndet_commit": nndet_commit,
        }
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nResults written to: {args.results_json}")


if __name__ == "__main__":
    main()
