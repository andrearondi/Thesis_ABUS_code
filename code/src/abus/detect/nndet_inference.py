"""Internal wrapper around nnDetection 0.1's predict_dir helper (STORY_01_02, D01.13).

This module is INTERNAL to the detect package — downstream stories (01_03, 01_04)
and EPIC_02+ do NOT import from it. Only candidates.py uses it for the OOF path.

Two public objects:

  RawDetections
    Intermediate per-case detection record parsed from disk.
    Boxes are in nnDetection's resampled-grid convention (float32, shape (N,6)).
    Box axis order: (x1, y1, x2, y2, z1, z2) — nndet/core/boxes/ops.py line 34.
    The public per-candidate type is RawCandidate in candidates.py.

  parse_predictions_dir(pred_dir) -> dict[int, RawDetections]
    Discovers *_pred_boxes.pkl files in pred_dir and parses them.

  predict_oof(fold, case_ids, preprocessed_dir, fold_dir, ...) -> dict[int, RawDetections]
    Drives nnDetection 0.1's predict_dir(case_ids=...) helper for the OOF path.
    Requires the nnDetection conda env (server-only). Lazily imported so the
    module is importable on the laptop without nnDetection.

nnDetection 0.1 output schema (D01.13, source-grounded, commit 97a58f3):
  predict_dir(save_state=False) writes ONE FILE PER KEY:
    <case_id>_pred_boxes.pkl  — pickle of np.ndarray (N, 6) float32
    <case_id>_pred_scores.pkl — pickle of np.ndarray (N,) float32
    <case_id>_pred_labels.pkl — pickle of np.ndarray (N,) int
  Source: helper.py:103-110 (to_numpy(result) → save_pickle(item, target_dir/f"{cid}_{key}.pkl"))
  get_case_result keys: pred_boxes, pred_scores, pred_labels, restore, ...
  NO embeddings key — embeddings are set to None and a zero placeholder is filled
  by candidates.py until STORY_01_04 wires backbone extraction.

  Box axis (x1,y1,x2,y2,z1,z2):
  Source: nndet/core/boxes/ops.py line 34, detection.py _apply_offsets_to_boxes line 228.

  Schema-drift mitigation (Risk #5, D01.13):
    These tests are the canary. Any schema change surfaces as a test failure.

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
    in candidates.py; this is the intermediate array record that
    generate_oof_candidates / generate_ensemble_candidates convert.

    Attributes
    ----------
    case_id : int
    boxes : np.ndarray
        Shape (N, 6), nnDetection box convention, resampled grid. float32.
        Axis order: (x1, y1, x2, y2, z1, z2) — per nndet/core/boxes/ops.py
        line 34 and detection.py _apply_offsets_to_boxes (D01.13 confirmed).
    scores : np.ndarray
        Shape (N,), float32.
    embeddings : np.ndarray or None
        Always None from predict_dir per-key output (D01.13: no embeddings
        key in get_case_result). Zero-vector placeholder is filled by
        candidates.py (which knows embedding dim D from config).
    """

    case_id: int
    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray | None


# ---------------------------------------------------------------------------
# parse_predictions_dir
# ---------------------------------------------------------------------------


def parse_predictions_dir(pred_dir: str) -> dict[int, RawDetections]:
    """Discover and parse nnDetection 0.1's per-key outputs in ``pred_dir``.

    D01.13 schema (source-grounded, commit 97a58f3):
      predict_dir(save_state=False) writes one file per result key:
        <case_id>_pred_boxes.pkl  — np.ndarray (N, 6) float32
        <case_id>_pred_scores.pkl — np.ndarray (N,) float32
        <case_id>_pred_labels.pkl — np.ndarray (N,) int  (not used here)
      Source: helper.py:103-110 ``for key, item in to_numpy(result).items(): ...``
      Box axis: (x1, y1, x2, y2, z1, z2) — nndet/core/boxes/ops.py line 34.
      NO embeddings key. RawDetections.embeddings is always None.

    Parsing rules:
      - Only files matching ``*_pred_boxes.pkl`` are processed (anchors case discovery).
      - Filename format: ``<case_id_str>_pred_boxes.pkl`` where ``<case_id_str>``
        is a zero-padded integer (e.g. ``0042``). The integer is extracted by
        stripping ``_pred_boxes`` and calling int().
      - Files whose prefix cannot be parsed as an integer are skipped with a
        WARNING log (defect-prevention: logged, not silently absorbed — echo of
        STORY_01_01 D01.9 silent-zero lesson).
      - The corresponding ``*_pred_scores.pkl`` is loaded from the same directory.
        If the scores file is missing, a WARNING is logged and the case is skipped.
      - Boxes and scores are coerced to float32.

    Pure-Python; CPU-safe; no nnDetection required.

    Parameters
    ----------
    pred_dir : str
        Path to the directory containing per-key pickle files.

    Returns
    -------
    dict[int, RawDetections]
        Keyed by case_id. Empty dict if no matching files found.
    """
    pred_path = Path(pred_dir)
    result: dict[int, RawDetections] = {}

    # Anchor on *_pred_boxes.pkl — one per case (D01.13 per-key schema)
    boxes_files = sorted(pred_path.glob("*_pred_boxes.pkl"))

    for boxes_file in boxes_files:
        stem = boxes_file.stem  # e.g. "0042_pred_boxes"
        suffix = "_pred_boxes"
        if not stem.endswith(suffix):
            logger.warning(
                "parse_predictions_dir: skipping file %s — unexpected stem %r",
                boxes_file.name,
                stem,
            )
            continue

        case_id_str = stem[: -len(suffix)]
        try:
            case_id = int(case_id_str)
        except ValueError:
            logger.warning(
                "parse_predictions_dir: skipping file %s — cannot parse integer case_id "
                "from prefix %r",
                boxes_file.name,
                case_id_str,
            )
            continue

        # Load the corresponding scores file
        scores_file = pred_path / f"{case_id_str}_pred_scores.pkl"
        if not scores_file.exists():
            logger.warning(
                "parse_predictions_dir: skipping case_id=%d — scores file %s not found",
                case_id,
                scores_file.name,
            )
            continue

        with open(boxes_file, "rb") as f:
            boxes_raw = pickle.load(f)  # noqa: S301
        with open(scores_file, "rb") as f:
            scores_raw = pickle.load(f)  # noqa: S301

        boxes = np.asarray(boxes_raw, dtype=np.float32)
        scores = np.asarray(scores_raw, dtype=np.float32)

        # Embeddings are NOT in the per-key predict_dir output (D01.13).
        # Zero-vector placeholder is filled by candidates.py.
        result[case_id] = RawDetections(
            case_id=case_id,
            boxes=boxes,
            scores=scores,
            embeddings=None,
        )

    return result


# ---------------------------------------------------------------------------
# predict_oof
# ---------------------------------------------------------------------------


def predict_oof(
    fold: int,
    case_ids: list[int],
    preprocessed_dir: str,
    fold_dir: str,
    num_models: int = 1,
) -> dict[int, RawDetections]:
    """Run the fold-``fold`` detector on ``case_ids`` via the nnDetection Python helper.

    The OOF path (per-fold inference over a specific case_ids list) has no CLI
    surface in nnDetection 0.1. This function drives it via the documented Python
    helper ``nndet.inference.helper.predict_dir(case_ids=...)``.

    D01.13 implementation (source-grounded, helper.py:29-42):

    Real predict_dir signature::

        predict_dir(source_dir, target_dir, cfg, plan, source_models,
                    model_fn=load_final_model, num_models=None,
                    num_tta_transforms=None, restore=False,
                    case_ids=None, save_state=False, **kwargs)

    This function:
      1. Loads ``cfg`` from ``<fold_dir>/config.yaml`` via OmegaConf.load.
      2. Loads ``plan`` from ``<fold_dir>/plan_inference.pkl`` via nndet's
         load_pickle (plan_inference.pkl only exists after ``--sweep`` ran,
         which is why retraining with ``--sweep`` is required — D01.13 point 7).
      3. Calls predict_dir with:
         - source_dir   = preprocessed_dir (preprocessed .npz files location)
         - target_dir   = a tempdir
         - source_models = fold_dir (as Path)
         - model_fn     = partial(load_final_model, identifier="last")
           (SWA model_last is the pre-registered detector — thesis §3.2.3)
         - num_models   = 1
         - restore      = True  (restore boxes to original image space)
         - case_ids     = [f"{cid:04d}" for cid in case_ids]
           (4-digit zero-padded strings matching preprocessed .npz stems)
         - save_state   = False (write per-key pkls; D01.13 per-key schema)
      4. Parses the tempdir via parse_predictions_dir and returns the dict.

    Caller (generate_oof_candidates) is responsible for the leakage guard —
    it asserts ``set(case_ids) ⊆ oof_ids(fold)`` BEFORE calling predict_oof.

    Candidate operating point (D01.13 point 6): use high-recall ensembler
    defaults (model_score_thresh=0.0, ensemble_score_thresh=0.0,
    model_topk=1000, model_detections_per_image=100). All four params are
    explicitly overridden so a swept plan_inference.pkl cannot silently lower
    topk/detections_per_image. STORY_01_03 owns the single project-wide
    calibration; we must not use the FROC-optimal swept params here.

    Requires the nnDetection conda env (server-only). nnDetection is lazily
    imported inside this function so the module is importable on the laptop.

    Parameters
    ----------
    fold : int
        Fold index (0–4). Used only for logging/traceability.
    case_ids : list[int]
        Integer case IDs to score (OOF ids for this fold, pre-validated by caller).
    preprocessed_dir : str
        Path to the preprocessed imagesTr directory containing ``.npz`` files.
        (``$det_data/<task_name>/preprocessed/D3V001_3d/imagesTr``)
    fold_dir : str
        Path to the fold's training output directory containing
        ``config.yaml`` and ``plan_inference.pkl``.
        (``$det_models/<task_name>/<exp.id>/fold<N>``)
    num_models : int
        Number of models (1 for a single fold). Default 1.

    Returns
    -------
    dict[int, RawDetections]
        Keyed by case_id (int). Boxes in (x1,y1,x2,y2,z1,z2) axis.
        embeddings is always None (D01.13: not in per-key output).

    Raises
    ------
    ImportError
        If nnDetection is not available in the current environment (laptop).
    RuntimeError
        If predict_dir fails.
    """
    # Lazy imports — nnDetection is server-only; the module must remain importable
    # on the laptop (thesis §3.2.3, D01.13 dependency note).
    try:
        from functools import partial as _partial

        from nndet.inference.helper import predict_dir
        from nndet.inference.loading import load_final_model
        from nndet.io.load import load_pickle
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "nnDetection / omegaconf is not installed in the current environment. "
            "predict_oof requires the nnDetection conda env (server-only). "
            "If running on the laptop, use the synthetic stub via generate_oof_candidates "
            "with an explicit inference_fn."
        ) from exc

    fold_path = Path(fold_dir)

    # Load config + plan from the fold's training output (created by --sweep).
    cfg_path = fold_path / "config.yaml"
    plan_path = fold_path / "plan_inference.pkl"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"plan_inference.pkl not found at {plan_path}. "
            "This file is only created when nndet_train runs with --sweep. "
            "Re-train this fold with --sweep (D01.13 requirement)."
        )
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    plan = load_pickle(plan_path)

    # Override swept inference params to high-recall defaults (D01.13 point 6).
    # STORY_01_03 owns calibration; we must not use the FROC-optimal swept params here.
    # Set ALL four ensembler params explicitly so sweep cannot silently lower topk/detections.
    if "inference_plan" in plan:
        plan["inference_plan"]["model_score_thresh"] = 0.0
        plan["inference_plan"]["ensemble_score_thresh"] = 0.0
        plan["inference_plan"]["model_topk"] = 1000
        plan["inference_plan"]["model_detections_per_image"] = 100

    # model_fn loads model_last (SWA checkpoint — pre-registered detector, thesis §3.2.3)
    model_fn = _partial(load_final_model, identifier="last")

    # case_ids as 4-digit zero-padded strings (matches preprocessed .npz stems)
    case_ids_str = [f"{cid:04d}" for cid in case_ids]

    logger.info(
        "predict_oof: fold=%d, %d cases, preprocessed_dir=%s, fold_dir=%s",
        fold,
        len(case_ids),
        preprocessed_dir,
        fold_dir,
    )

    with tempfile.TemporaryDirectory() as out_dir:
        predict_dir(
            source_dir=preprocessed_dir,
            target_dir=out_dir,
            cfg=cfg,
            plan=plan,
            source_models=fold_path,
            model_fn=model_fn,
            num_models=num_models,
            restore=True,
            case_ids=case_ids_str,
            save_state=False,
        )
        detections = parse_predictions_dir(out_dir)

    logger.info(
        "predict_oof: fold=%d — parsed %d cases from predictions",
        fold,
        len(detections),
    )
    return detections
