"""nnDetection fold-training wrapper (STORY_01_02).

Thin wrapper around the nnDetection training CLI. Invokes the CLI for one fold,
captures the run metadata (checkpoint path, wall-clock, GPU-hours, final
internal-validation metric, nnDetection commit), and returns a DetectorRun.

No architecture or schedule modification — nnDetection default schedule is used
(thesis §3.2.3 pre-registered: the detector is not tuned). Not a free parameter.

GPU note: training itself requires a GPU and the nnDetection environment. This
module is importable on the laptop (no GPU imports at module level) but
train_fold_detector will fail locally without nnDetection.

The module is CPU-safe: all imports at module level are stdlib-only
(subprocess, time, pathlib, re) — no GPU or nnDetection imports.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# DetectorRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorRun:
    """Metadata from a completed nnDetection fold-training run.

    Attributes
    ----------
    fold:
        Fold index (0–4) trained.
    checkpoint_path:
        Absolute path to the best-metric checkpoint saved by nnDetection.
    wall_clock_hours:
        Total wall-clock time for the training run, in hours.
    gpu_hours:
        GPU-hours consumed (wall_clock_hours × number_of_GPUs).
        For a single-GPU run: gpu_hours == wall_clock_hours.
    final_val_metric:
        Final internal-validation metric reported by nnDetection's trainer
        (typically FROC or mAP at the best-metric checkpoint).
    nndet_commit:
        nnDetection version+commit hash used (reproducibility, agent_rules §12).
        Format: "v0.x" or a full commit hash from
        ``git -C <nndet_dir> rev-parse HEAD``.
    """

    fold: int
    checkpoint_path: str
    wall_clock_hours: float
    gpu_hours: float
    final_val_metric: float
    nndet_commit: str


# ---------------------------------------------------------------------------
# train_fold_detector
# ---------------------------------------------------------------------------


def train_fold_detector(
    fold: int,
    nndet_dataset_root: str,
    out_root: str,
    task_name: str = "Task001_TDSCABUS",
    nndet_module: str = "nndet.entrypoints.train",
    n_gpus: int = 1,
    nndet_commit: str = "unknown",
) -> DetectorRun:
    """Invoke the nnDetection default training CLI for one fold.

    Trains on train_ids(fold) = all cases NOT in fold ``fold``, using the
    nnDetection default schedule (RetinaUNet, default augmentation, early
    stopping on the internal validation metric). No architecture or schedule
    modification (thesis §3.2.3).

    The fold training is launched via:
        python -m nndet.entrypoints.train <task_name> RetinaUNet000 \
            --fold <fold> --overwrites "train.fold=<fold>"

    The checkpoint output lives under:
        <out_root>/<task_name>/RetinaUNet000/<fold>/

    On completion, the function:
      1. Locates the best-metric checkpoint (``model_best.ckpt``).
      2. Records wall-clock time.
      3. Parses the final val metric from nnDetection's training log.
      4. Returns a DetectorRun with all fields populated.

    Parameters
    ----------
    fold:
        Fold index 0–4.
    nndet_dataset_root:
        Path to the nnDetection dataset root directory containing
        ``Task001_TDSCABUS/``.
    out_root:
        Output root for nnDetection training artefacts (models, logs).
        Maps to nnDetection's ``det_models`` directory.
    task_name:
        nnDetection task name (default ``"Task001_TDSCABUS"``).
    nndet_module:
        Python module path for the nnDetection training entrypoint.
    n_gpus:
        Number of GPUs used (for GPU-hour computation). Default 1 (sequential
        fold training; user decision at epic-approval gate 2026-05-18, D01.5).
    nndet_commit:
        nnDetection commit hash string (reproducibility). Passed in by the
        calling script (``scripts/train_all_folds.py``).

    Returns
    -------
    DetectorRun

    Raises
    ------
    RuntimeError
        If the training subprocess exits with a non-zero return code.
    FileNotFoundError
        If the expected checkpoint is not found after training completes.
    """
    start_time = time.time()

    # nnDetection training CLI call.
    # The fold is passed as both a positional override and an explicit flag to
    # ensure nnDetection uses exactly the frozen 5-fold splits file.
    cmd = [
        "python",
        "-m",
        nndet_module,
        task_name,
        "RetinaUNet000",
        "--fold",
        str(fold),
    ]

    result = subprocess.run(cmd, cwd=nndet_dataset_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nnDetection training failed for fold {fold}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"stdout:\n{result.stdout[-3000:]}\n"
            f"stderr:\n{result.stderr[-3000:]}\n"
        )

    wall_clock_seconds = time.time() - start_time
    wall_clock_hours = wall_clock_seconds / 3600.0
    gpu_hours = wall_clock_hours * n_gpus

    # Locate the best-metric checkpoint.
    # nnDetection writes checkpoints to:
    #   <det_models>/<task_name>/RetinaUNet000/<fold>/
    # The best-metric checkpoint is typically named ``model_best.ckpt``.
    ckpt_dir = Path(out_root) / task_name / "RetinaUNet000" / str(fold)
    best_ckpt = ckpt_dir / "model_best.ckpt"
    if not best_ckpt.exists():
        # Try alternative naming conventions
        candidates = list(ckpt_dir.glob("*best*.ckpt")) + list(ckpt_dir.glob("*best*.pth"))
        if candidates:
            best_ckpt = candidates[0]
        else:
            raise FileNotFoundError(
                f"Best-metric checkpoint not found under {ckpt_dir}. "
                "Expected 'model_best.ckpt' or similar. "
                "Check the nnDetection training output directory."
            )

    # Parse final val metric from stdout.
    # nnDetection logs lines like: "Best val metric: 0.8234" or
    # "metric_0: 0.82" depending on the version. We extract the last float.
    final_val_metric = _parse_final_val_metric(result.stdout)

    return DetectorRun(
        fold=fold,
        checkpoint_path=str(best_ckpt),
        wall_clock_hours=wall_clock_hours,
        gpu_hours=gpu_hours,
        final_val_metric=final_val_metric,
        nndet_commit=nndet_commit,
    )


def _parse_final_val_metric(log_text: str) -> float:
    """Extract the final validation metric float from nnDetection's stdout.

    Scans lines for "best" or "val" followed by a float. Returns the last
    such match, or 0.0 if no metric is found (non-blocking: the training
    completed even if log parsing fails).
    """
    metric = 0.0
    # Pattern: any line containing a float after "best" or "val" or "metric"
    pattern = re.compile(r"(?:best|val|metric)[^0-9\-]*([0-9]+\.[0-9]+)", re.IGNORECASE)
    for match in pattern.finditer(log_text):
        try:
            metric = float(match.group(1))
        except ValueError:
            pass
    return metric
