"""nnDetection fold-training wrapper (STORY_01_02, D01.9).

Thin wrapper around the nnDetection training console script. Invokes
``nndet_train <task_name> -o exp.fold=<N>`` for one fold, captures the run
metadata (checkpoint path, wall-clock, GPU-hours, final internal-validation
metric, nnDetection commit), and returns a DetectorRun.

nnDetection 0.1 CLI surface (verified against commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
  Console script: ``nndet_train``
  Signature: ``nndet_train <TASK> [-o OVERWRITES ...] [--sweep]``
  Fold is passed as a Hydra override: ``-o exp.fold=<N>``
  There is NO model positional, NO ``--fold`` flag.

nnDetection 0.1 checkpoint output layout (from scripts/train.py:init_train_dir):
  ``$det_models/<task_name>/<exp.id>/fold<N>/``
  where exp.id defaults to ``<exp.model>_<planner_id>``
      = ``RetinaUNetV001_D3V001_3d`` for our task.
  Fold subdir is ``fold0``, NOT ``0``.

Env-var contract (lowercase — nnDetection reads these via os.environ):
  ``det_data`` — nnDetection dataset root
  ``det_models`` — nnDetection model output root
  Do NOT use uppercase DET_MODELS / DET_DATA; nnDetection silently ignores them.

No architecture or schedule modification — nnDetection default schedule is used
(thesis §3.2.3 pre-registered: the detector is not tuned). Not a free parameter.

GPU note: training requires a GPU and the nnDetection environment. This module is
importable on the laptop (all module-level imports are stdlib-only: subprocess,
time, pathlib, re). train_fold_detector will fail without nnDetection in PATH.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (pinned; changes here invalidate existing checkpoints)
# ---------------------------------------------------------------------------

#: nnDetection experiment id (model_id + planner_id), verified by user:
#: preprocessed/D3V001_3d/ is present → planner_id = D3V001_3d.
#: D01.9: was "RetinaUNet000" (incorrect); correct value is "RetinaUNetV001_D3V001_3d".
EXP_ID: str = "RetinaUNetV001_D3V001_3d"

#: nnDetection task name (matches STORY_01_01 build).
TASK_NAME: str = "Task001_TDSCABUS"

#: nnDetection 0.1 commit hash (reproducibility, agent_rules §12).
NNDET_COMMIT_PINNED: str = "97a58f3110b71caf1b4bcc1851e67cf11e987fc5"


# ---------------------------------------------------------------------------
# DetectorRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorRun:
    """Metadata from a completed nnDetection fold-training run.

    Attributes
    ----------
    fold : int
        Fold index (0–4) trained.
    checkpoint_path : str
        Absolute path to the best-metric checkpoint saved by nnDetection.
        Under ``$det_models/<task_name>/<exp.id>/fold<N>/``.
    wall_clock_hours : float
        Total wall-clock time for the training run, in hours.
    gpu_hours : float
        GPU-hours consumed (wall_clock_hours × n_gpus).
        Single-GPU run (user decision D01.5): gpu_hours == wall_clock_hours.
    final_val_metric : float
        Final internal-validation metric reported by nnDetection's trainer
        (typically FROC or mAP at the best-metric checkpoint).
    nndet_commit : str
        nnDetection version+commit used (reproducibility, agent_rules §12).
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
    task_name: str = TASK_NAME,
    exp_id: str = EXP_ID,
    n_gpus: int = 1,
    nndet_commit: str = NNDET_COMMIT_PINNED,
) -> DetectorRun:
    """Invoke the nnDetection default training console script for one fold.

    Trains on train_ids(fold) = all 80 cases NOT in fold ``fold``, using the
    nnDetection default schedule (Retina U-Net, default augmentation, early
    stopping on the internal validation metric). No architecture or schedule
    modification (thesis §3.2.3).

    Mechanics (D01.9 — verified against nnDetection 0.1 commit 97a58f3):
      Calls ``nndet_train <task_name> -o exp.fold=<fold>`` via subprocess.
      No model positional, no ``--fold`` flag — fold is a Hydra override.
      Requires ``det_data`` and ``det_models`` env vars set (lowercase).

    Output layout (nnDetection 0.1, from scripts/train.py:init_train_dir):
      ``$det_models/<task_name>/<exp_id>/fold<N>/``
      Note: fold subdir is ``fold0``, not ``0``.

    Parameters
    ----------
    fold : int
        Fold index 0–4.
    nndet_dataset_root : str
        Path to the nnDetection dataset root containing ``<task_name>/``.
        This is the ``det_data`` directory (passed as env var or via the
        runbook; this argument is used as the subprocess cwd).
    out_root : str
        Output root for nnDetection training artefacts.
        Maps to nnDetection's ``det_models`` env var.
    task_name : str
        nnDetection task name (default ``"Task001_TDSCABUS"``).
    exp_id : str
        nnDetection experiment id (``<exp.model>_<planner_id>``).
        Default ``"RetinaUNetV001_D3V001_3d"``.
    n_gpus : int
        Number of GPUs (for GPU-hour computation). Default 1 (sequential
        fold training; user decision D01.5, 2026-05-18).
    nndet_commit : str
        nnDetection commit hash string (reproducibility).

    Returns
    -------
    DetectorRun

    Raises
    ------
    RuntimeError
        If the training subprocess exits with a non-zero return code.
    FileNotFoundError
        If no best-metric checkpoint is found after training completes.
    """
    start_time = time.time()

    # nnDetection 0.1 training CLI (D01.13):
    #   nndet_train <TASK> -o exp.fold=<N> --sweep
    # No model positional. No --fold flag. Fold is a Hydra override.
    # --sweep is REQUIRED: produces plan_inference.pkl and sweep_predictions/
    # that predict_dir and nndet_consolidate need (D01.13 points 3-5).
    cmd = [
        "nndet_train",
        task_name,
        "-o",
        f"exp.fold={fold}",
        "--sweep",
    ]

    result = subprocess.run(  # noqa: S603  (trusted project-only command)
        cmd,
        cwd=nndet_dataset_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nnDetection training failed for fold {fold}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"stdout (last 3000 chars):\n{result.stdout[-3000:]}\n"
            f"stderr (last 3000 chars):\n{result.stderr[-3000:]}\n"
        )

    wall_clock_seconds = time.time() - start_time
    wall_clock_hours = wall_clock_seconds / 3600.0
    gpu_hours = wall_clock_hours * n_gpus

    # Locate the best-metric checkpoint.
    # nnDetection 0.1 writes to:
    #   $det_models/<task_name>/<exp_id>/fold<N>/
    # The fold subdir is f"fold{N}", not str(N).
    # Checkpoint filename is nnDetection's default (typically "model_best.ckpt");
    # we search for the canonical name first, then fall back to a glob.
    fold_dir = Path(out_root) / task_name / exp_id / f"fold{fold}"
    best_ckpt = _locate_best_checkpoint(fold_dir)

    # Parse the final val metric from nnDetection's stdout.
    final_val_metric = _parse_final_val_metric(result.stdout)

    return DetectorRun(
        fold=fold,
        checkpoint_path=str(best_ckpt),
        wall_clock_hours=wall_clock_hours,
        gpu_hours=gpu_hours,
        final_val_metric=final_val_metric,
        nndet_commit=nndet_commit,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _locate_best_checkpoint(fold_dir: Path) -> Path:
    """Find the best-metric checkpoint in ``fold_dir``.

    nnDetection 0.1 checkpoint filename is the default from its checkpoint
    callback — typically ``model_best.ckpt``. We try canonical names first,
    then fall back to a glob.

    [SERVER-SIDE AUDIT REQUIRED: confirm the exact checkpoint filename against
    nnDetection 0.1's checkpoint callback at
    ``nndet/training/callbacks/checkpoint.py`` before the first server run.]
    """
    # Canonical candidates in priority order
    for name in ("model_best.ckpt", "best.ckpt", "best_model.ckpt"):
        candidate = fold_dir / name
        if candidate.exists():
            return candidate

    # Fallback: glob for any *best* checkpoint
    glob_hits = sorted(fold_dir.glob("*best*.ckpt")) + sorted(fold_dir.glob("*best*.pth"))
    if glob_hits:
        return glob_hits[0]

    raise FileNotFoundError(
        f"Best-metric checkpoint not found under {fold_dir}. "
        "Expected a file matching 'model_best.ckpt' or '*best*.ckpt'. "
        "Check the nnDetection 0.1 training output directory. "
        "[SERVER-SIDE AUDIT: confirm the exact checkpoint filename from the "
        "nnDetection 0.1 checkpoint callback before the first server run.]"
    )


def _parse_final_val_metric(log_text: str) -> float:
    """Extract the final validation metric float from nnDetection's stdout.

    Scans for lines containing a float after 'best', 'val', or 'metric'.
    Returns the last such match, or 0.0 if none found. Non-blocking: training
    completed even if log parsing fails (returns 0.0 as a sentinel).
    """
    metric = 0.0
    pattern = re.compile(r"(?:best|val|metric)[^0-9\-]*([0-9]+\.[0-9]+)", re.IGNORECASE)
    for match in pattern.finditer(log_text):
        try:
            metric = float(match.group(1))
        except ValueError:
            pass
    return metric
