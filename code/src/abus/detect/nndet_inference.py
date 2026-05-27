"""Internal wrapper around nnDetection 0.1's predict_dir helper (STORY_01_02, D01.9).

This module is INTERNAL to the detect package — downstream stories (01_03, 01_04)
and EPIC_02+ do NOT import from it. Only candidates.py uses it for the OOF path.

Two public objects:

  RawDetections
    Intermediate per-case detection record parsed from disk.
    Boxes are in nnDetection's resampled-grid convention (float32, shape (N,6)).
    The public per-candidate type is RawCandidate in candidates.py.

  parse_predictions_dir(pred_dir) -> dict[int, RawDetections]
    Discovers *_boxes.pkl files in pred_dir and parses them.

  predict_oof(fold, case_ids, task_dir, model_dir, ...) -> dict[int, RawDetections]
    Drives nnDetection 0.1's predict_dir(case_ids=...) helper for the OOF path.
    Requires the nnDetection conda env (server-only). Lazily imported so the
    module is importable on the laptop without nnDetection.

nnDetection 0.1 output schema (commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
  Files written by both the CLI path (nndet_predict) and the Python helper
  (predict_dir) under <training_dir>/test_predictions/:
    <case_id_4d>_boxes.pkl  — pickle of dict with keys:
      'boxes':      np.ndarray shape (N, 6) float32  — resampled-grid coords
      'scores':     np.ndarray shape (N,)   float32  — detection confidence
      'embeddings': np.ndarray shape (N, D) float32  — backbone-pooled (optional)

  Schema-drift mitigation (Risk #5, D01.9):
    These tests are the canary. If the real nnDetection output schema differs,
    the unit tests in tests/test_nndet_inference.py will fail before this code
    is ever run end-to-end. The senior engineer must confirm schema parity
    between the CLI path and the Python helper path before the first server run.

CPU safety:
  parse_predictions_dir is pure-Python + numpy; no GPU required.
  predict_oof lazy-imports nnDetection inside the function body.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RawDetections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDetections:
    """Raw nnDetection predictions for one case, parsed from disk.

    Internal-only. The public per-candidate output type is RawCandidate
    in candidates.py; this is the intermediate dict-of-arrays that
    generate_oof_candidates / generate_ensemble_candidates convert.

    Attributes
    ----------
    case_id : int
    boxes : np.ndarray
        Shape (N, 6), nnDetection box convention, resampled grid. float32.
        Format: (z1, y1, x1, z2, y2, x2) — inclusive coordinates on the
        resampled voxel grid. This is the format nnDetection 0.1 writes.
        [SERVER-SIDE AUDIT REQUIRED: confirm exact axis order against the
        nnDetection 0.1 source at scripts/predict.py and inference/helper.py
        before end-to-end execution.]
    scores : np.ndarray
        Shape (N,), float32.
    embeddings : np.ndarray or None
        Shape (N, D) float32 if available in the pickle, else None.
        A zero-vector placeholder of shape (D,) is NOT inserted here —
        the placeholder is inserted by the candidate pipeline (candidates.py)
        which knows the expected embedding dimension D from the architecture.
    """

    case_id: int
    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray | None


# ---------------------------------------------------------------------------
# parse_predictions_dir
# ---------------------------------------------------------------------------


def parse_predictions_dir(pred_dir: str) -> dict[int, RawDetections]:
    """Discover and parse nnDetection 0.1's ``*_boxes.pkl`` outputs in ``pred_dir``.

    Returns a dict keyed by case_id (int, parsed from the nnDetection 0.1
    filename stem ``<case_id_4d>_boxes.pkl``).

    Parsing rules:
      - Only files matching the glob ``*_boxes.pkl`` are processed.
      - Filename stem format: ``<case_id_4d>_boxes`` where ``<case_id_4d>``
        is a zero-padded integer (e.g. ``0042``). The integer is extracted by
        stripping the ``_boxes`` suffix and calling int().
      - Files whose stem (before ``_boxes``) cannot be parsed as an integer
        are skipped with a WARNING log. They do NOT raise an exception
        (defect-prevention echo of the STORY_01_01 silent-zero failure —
        D01.9: malformed filenames are logged, not silently absorbed).
      - Each valid pickle is expected to contain a dict with keys:
          'boxes':       np.ndarray (N, 6) float32
          'scores':      np.ndarray (N,)   float32
          'embeddings':  np.ndarray (N, D) float32  [optional]
        Boxes and scores are coerced to float32. Missing 'embeddings' →
        RawDetections.embeddings = None.

    Pure-Python; CPU-safe; no nnDetection required.

    Parameters
    ----------
    pred_dir : str
        Path to the directory containing ``*_boxes.pkl`` files.

    Returns
    -------
    dict[int, RawDetections]
        Keyed by case_id. Empty dict if no matching files found.
    """
    pred_path = Path(pred_dir)
    result: dict[int, RawDetections] = {}

    pkl_files = sorted(pred_path.glob("*_boxes.pkl"))

    for pkl_file in pkl_files:
        stem = pkl_file.stem  # e.g. "0042_boxes"
        # Strip the "_boxes" suffix to get the case_id string
        if not stem.endswith("_boxes"):
            logger.warning(
                "parse_predictions_dir: skipping file %s — stem does not end with '_boxes'",
                pkl_file.name,
            )
            continue

        case_id_str = stem[: -len("_boxes")]
        try:
            case_id = int(case_id_str)
        except ValueError:
            logger.warning(
                "parse_predictions_dir: skipping file %s — cannot parse integer case_id "
                "from stem %r",
                pkl_file.name,
                case_id_str,
            )
            continue

        with open(pkl_file, "rb") as f:
            payload = pickle.load(f)  # noqa: S301  (trusted project-internal files)

        boxes = np.asarray(payload["boxes"], dtype=np.float32)
        scores = np.asarray(payload["scores"], dtype=np.float32)

        embeddings: np.ndarray | None = None
        if "embeddings" in payload and payload["embeddings"] is not None:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)

        result[case_id] = RawDetections(
            case_id=case_id,
            boxes=boxes,
            scores=scores,
            embeddings=embeddings,
        )

    return result


# ---------------------------------------------------------------------------
# predict_oof
# ---------------------------------------------------------------------------


def predict_oof(
    fold: int,
    case_ids: list[int],
    task_dir: str,
    model_dir: str,
    num_models: int = 1,
    num_tta: int = 4,
) -> dict[int, RawDetections]:
    """Run the fold-``fold`` detector on ``case_ids`` via the nnDetection Python helper.

    The OOF path (per-fold inference over a specific case_ids list) has no CLI
    surface in nnDetection 0.1. This function drives it via the documented Python
    helper ``nndet.inference.helper.predict_dir(case_ids=...)``.

    Caller is responsible for the leakage guard — generate_oof_candidates asserts
    ``set(case_ids) ⊆ oof_ids(fold)`` BEFORE calling predict_oof.

    This function writes predictions to a temporary directory, calls
    parse_predictions_dir on that directory, and returns the parsed dict before
    the tempdir is cleaned up.

    Requires the nnDetection conda env (server-only). nnDetection is lazily
    imported inside this function so the module is importable on the laptop.

    Parameters
    ----------
    fold : int
        Fold index (0–4). Used only for logging/traceability.
    case_ids : list[int]
        Integer case IDs to score (OOF ids for this fold, pre-validated by caller).
    task_dir : str
        Path to the nnDetection task directory
        (``$det_data/<task_name>``).
    model_dir : str
        Path to the fold's training output directory
        (``$det_models/<task_name>/<exp.id>/fold<N>``).
    num_models : int
        Number of models to ensemble (1 for a single fold). Default 1.
    num_tta : int
        Number of test-time augmentation passes. Default 4 (nnDetection default).

    Returns
    -------
    dict[int, RawDetections]
        Keyed by case_id.

    Raises
    ------
    ImportError
        If nnDetection is not available in the current environment (laptop).
    RuntimeError
        If predict_dir fails.
    """
    # Lazy import — nnDetection is server-only; the module must remain importable
    # on the laptop (thesis §3.2.3, D01.9 dependency note).
    try:
        from nndet.inference.helper import predict_dir  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "nnDetection is not installed in the current environment. "
            "predict_oof requires the nnDetection conda env (server-only). "
            "If running on the laptop, use the synthetic stub via generate_oof_candidates "
            "with an explicit inference_fn."
        ) from exc

    logger.info(
        "predict_oof: fold=%d, %d case_ids, task_dir=%s, model_dir=%s",
        fold,
        len(case_ids),
        task_dir,
        model_dir,
    )

    with tempfile.TemporaryDirectory() as out_dir:
        # Call nnDetection 0.1's predict_dir with the case_ids list so only
        # the OOF cases are scored.
        # [SERVER-SIDE AUDIT REQUIRED: verify the exact signature of
        #  predict_dir(case_ids=...) against nnDetection 0.1 source
        #  nndet/inference/helper.py before first server run.]
        predict_dir(
            task_dir=task_dir,
            model_dir=model_dir,
            output_dir=out_dir,
            case_ids=case_ids,
            num_models=num_models,
            num_tta=num_tta,
        )

        detections = parse_predictions_dir(out_dir)

    logger.info(
        "predict_oof: fold=%d — parsed %d cases from predictions",
        fold,
        len(detections),
    )
    return detections
