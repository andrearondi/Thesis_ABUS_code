"""Internal wrapper around nnDetection 0.1's predict_dir helper (STORY_01_02, D01.13/D01.14).

This module is INTERNAL to the detect package — downstream stories (01_03, 01_04)
and EPIC_02+ do NOT import from it. Only candidates.py uses it for the OOF and
embedding-extraction paths.

Public objects:

  D_EMB : int = 128
    Pinned embedding dimension (fpn_channels=128 per nnDetection build log, D01.14).
    Single source of truth; carried through RawCandidateSet and NodeFeatureSpec.

  RawDetections
    Intermediate per-case detection record (boxes/scores only; embeddings=None).
    Used by parse_predictions_dir (CLI per-key reader) and predict_oof.
    Box axis order: (x1, y1, x2, y2, z1, z2) — nndet/core/boxes/ops.py line 34.

  RawDetectionsWithEmb
    Like RawDetections but embeddings is always a real (N, D_EMB) float32 array.
    Produced by predict_with_embeddings (D01.14 embedding-extraction path).

  point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2) -> np.ndarray (C,)
    Trilinear point-pooling of a (C, D0, D1, D2) feature map at a centroid given
    in tile-pixel frame as (cx_d0, cx_d1, cx_d2). CPU-only; no nnDetection required.
    D01.14 axis convention: feat axis 1=d0=z, axis 2=d1=y, axis 3=d2=x.

  parse_predictions_dir(pred_dir) -> dict[int, RawDetections]
    Discovers *_pred_boxes.pkl files in pred_dir and parses them (CLI schema).

  predict_oof(fold, case_ids, preprocessed_dir, fold_dir, ...) -> dict[int, RawDetections]
    Drives nnDetection 0.1's predict_dir(case_ids=...) helper for the OOF path.
    Requires the nnDetection conda env (server-only). Lazily imported.

  predict_with_embeddings(fold, case_ids, preprocessed_dir, fold_dir, ...)
      -> dict[int, RawDetectionsWithEmb]
    D01.14 embedding-extraction path. Loads the model, registers a forward hook
    on model.decoder, pools the finest FPN level (decoder_levels[0]) at each
    detection's centroid using trilinear interpolation, and returns boxes + scores +
    real (N, 128) embeddings. Requires the nnDetection conda env (server-only).
    Lazy imports.

  preprocess_val_test(fold_dir, num_processes) -> None
    D01.14b val/test preprocessing step. Runs the nnDetection planner's
    run_preprocessing_test to populate preprocessed/<data_identifier>/imagesTs/
    from the raw test set (raw_splitted/imagesTs/). Must be called BEFORE
    predict_with_embeddings for val/test cases. Lazy imports; server-only.

nnDetection 0.1 output schema (D01.13, source-grounded, commit 97a58f3):
  predict_dir(save_state=False) writes ONE FILE PER KEY:
    <case_id>_pred_boxes.pkl  — pickle of np.ndarray (N, 6) float32
    <case_id>_pred_scores.pkl — pickle of np.ndarray (N,) float32
    <case_id>_pred_labels.pkl — pickle of np.ndarray (N,) int
  Source: helper.py:103-110.
  NO embeddings key — embeddings are set to None from parse_predictions_dir.

  Box axis (x1,y1,x2,y2,z1,z2):
  Source: nndet/core/boxes/ops.py line 34, detection.py _apply_offsets_to_boxes line 228.

CPU safety:
  parse_predictions_dir, point_pool_trilinear are pure-Python + numpy; no GPU required.
  predict_oof and predict_with_embeddings lazy-import nnDetection inside the function body.
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
# D01.14 — pinned embedding dimension
# ---------------------------------------------------------------------------

D_EMB: int = 128
"""Pinned backbone-embedding dimension (fpn_channels=128 per nnDetection build log, D01.14).

Single source of truth. Flows:
  nndet_inference.D_EMB
  → RawDetectionsWithEmb.embeddings shape (N, D_EMB)
  → candidates.RawCandidate.embedding shape (D_EMB,)  [asserted in _raw_detections_to_candidates]
  → STORY_01_03 RetainedCandidate.embedding (pass-through)
  → STORY_01_04 NodeFeatureSpec.embedding_dim = 128
  → graph node features x: (n_nodes, 128 + 7) float32
"""

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


@dataclass(frozen=True)
class RawDetectionsWithEmb:
    """Raw nnDetection predictions for one case, WITH real backbone embeddings (D01.14).

    Produced exclusively by predict_with_embeddings. Unlike RawDetections,
    embeddings is ALWAYS a real (N, D_EMB) float32 array — never None.

    Attributes
    ----------
    case_id : int
    boxes : np.ndarray
        Shape (N, 6), nnDetection box convention. float32.
        Axis order: (x1, y1, x2, y2, z1, z2) — per nndet/core/boxes/ops.py line 34.
    scores : np.ndarray
        Shape (N,), float32.
    embeddings : np.ndarray
        Shape (N, D_EMB) float32. Always real; never None.
        Pooled from the finest FPN decoder level (decoder_levels[0]) at each
        detection's centroid using trilinear interpolation (D01.14).
    """

    case_id: int
    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray  # always (N, D_EMB) float32; NOT Optional


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
# point_pool_trilinear  (D01.14 — CPU-only, no nnDetection required)
# ---------------------------------------------------------------------------


def point_pool_trilinear(
    feat_map: np.ndarray,
    cx_d0: float,
    cx_d1: float,
    cx_d2: float,
) -> np.ndarray:
    """Trilinear point-pooling of a (C, D0, D1, D2) feature map at a centroid.

    D01.14 axis convention (grounded in nnDetection source):
      feat_map axis 0 : C channels
      feat_map axis 1 : d0 (storage axis 0, acoustic depth, z in nndet box notation)
      feat_map axis 2 : d1 (storage axis 1, lateral y in nndet box notation)
      feat_map axis 3 : d2 (storage axis 2, elevation x in nndet box notation)

    nnDetection box axis: (x1, y1, x2, y2, z1, z2) where x=d2, y=d1, z=d0
    (nndet/core/boxes/ops.py line 34). The caller maps box centroid to:
      cx_d0 = (z1 + z2) / 2   → feat axis 1
      cx_d1 = (y1 + y2) / 2   → feat axis 2
      cx_d2 = (x1 + x2) / 2   → feat axis 3

    Trilinear interpolation: each corner voxel weighted by the product of
    (1 - frac) or frac along each axis, where frac is the fractional distance
    from the floor voxel to the centroid. Boundary centroids are clamped to the
    valid feature map extent before flooring so out-of-bounds centroids degrade
    gracefully (producing the nearest-boundary value).

    Parameters
    ----------
    feat_map : np.ndarray
        Shape (C, D0, D1, D2), float32. Feature map for ONE tile from one TTA pass.
    cx_d0 : float
        Centroid coordinate along axis 1 (d0/z) in tile-pixel space.
    cx_d1 : float
        Centroid coordinate along axis 2 (d1/y) in tile-pixel space.
    cx_d2 : float
        Centroid coordinate along axis 3 (d2/x) in tile-pixel space.

    Returns
    -------
    np.ndarray
        Shape (C,), float32. Trilinearly interpolated channel vector.
    """
    _, D0, D1, D2 = feat_map.shape

    # Clamp centroid to valid range so boundary detections don't go out of bounds
    cx_d0 = float(np.clip(cx_d0, 0.0, D0 - 1.0))
    cx_d1 = float(np.clip(cx_d1, 0.0, D1 - 1.0))
    cx_d2 = float(np.clip(cx_d2, 0.0, D2 - 1.0))

    # Floor indices and fractional distances
    f0 = int(cx_d0)
    f1 = int(cx_d1)
    f2 = int(cx_d2)
    # Ceiling indices (clamped so floor+1 stays in bounds)
    c0 = min(f0 + 1, D0 - 1)
    c1 = min(f1 + 1, D1 - 1)
    c2 = min(f2 + 1, D2 - 1)
    # Fractional parts
    fd0 = cx_d0 - f0
    fd1 = cx_d1 - f1
    fd2 = cx_d2 - f2

    # Trilinear interpolation across 8 corners: weight = product of (1-frac) or frac
    # along each of the three axes.
    result = (
        feat_map[:, f0, f1, f2] * (1 - fd0) * (1 - fd1) * (1 - fd2)
        + feat_map[:, c0, f1, f2] * fd0 * (1 - fd1) * (1 - fd2)
        + feat_map[:, f0, c1, f2] * (1 - fd0) * fd1 * (1 - fd2)
        + feat_map[:, f0, f1, c2] * (1 - fd0) * (1 - fd1) * fd2
        + feat_map[:, c0, c1, f2] * fd0 * fd1 * (1 - fd2)
        + feat_map[:, c0, f1, c2] * fd0 * (1 - fd1) * fd2
        + feat_map[:, f0, c1, c2] * (1 - fd0) * fd1 * fd2
        + feat_map[:, c0, c1, c2] * fd0 * fd1 * fd2
    )
    return np.asarray(result, dtype=np.float32)


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


# ---------------------------------------------------------------------------
# _predict_single_case_with_embeddings  (D01.14 — CPU-safe helper, testable)
# ---------------------------------------------------------------------------


def _predict_single_case_with_embeddings(
    tiles: list[dict],
    feat_maps_per_tile: list[np.ndarray],
    boxes_per_tile: list[np.ndarray],
    scores_per_tile: list[np.ndarray],
    fpn_level_index: int = 0,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tile embedding pooling + WBC ensemble for one case.

    This is the testable core of D01.14's per-tile embedding extraction, separated
    from the nnDetection I/O so it can be tested CPU-only without a GPU or model.

    The caller (predict_with_embeddings) is responsible for:
    - Running inference per tile with the decoder hook active
    - Passing the per-tile feature maps and per-tile boxes/scores

    This function:
    1. For each tile: pools embeddings at tile-LOCAL box centroids from the
       tile's own feature map — correct because both are in the same tile-pixel
       coordinate frame.
    2. Applies tile_origin offset to boxes → case-preprocessed-space boxes,
       carrying embeddings alongside (no coordinate change to embeddings).
    3. Builds RawCandidate-like (bbox, score, embedding) records and runs
       ensemble_combine (WBC + score+embedding averaging) across all tiles.
    4. Returns (boxes, scores, embeddings) arrays for the case.

    Fix for code-review Must-fix #1+2:
    - Must-fix #1 (coordinate mismatch): pooling at tile-LOCAL centroids, not
      case-space centroids. The feat_map is in tile-pixel space; so are the
      per-tile boxes from inference_step before tile_origin is applied.
    - Must-fix #2 (single feat map for all tiles): each tile uses its own
      feat_maps_per_tile[i], not a single shared last-captured map.

    Parameters
    ----------
    tiles : list[dict]
        List of tile dicts from predictor.tile_case, each with "tile_origin".
        tile_origin is in nnDetection (x,y,z) = (d2,d1,d0) order.
    feat_maps_per_tile : list[np.ndarray]
        Per-tile feature maps, one per tile. Each is (C, D0, D1, D2) float32
        in tile-local pixel space, where axis 1=d0, axis 2=d1, axis 3=d2.
    boxes_per_tile : list[np.ndarray]
        Per-tile detected boxes, one per tile. Each is (N_i, 6) float32 in
        tile-local pixel space, axis (x1,y1,x2,y2,z1,z2).
    scores_per_tile : list[np.ndarray]
        Per-tile detection scores, one per tile. Each is (N_i,) float32.
    fpn_level_index : int
        Which FPN level to pool from. Default 0 = finest.
    iou_threshold : float
        IoU threshold for ensemble_combine WBC. Default 0.5 (provisional;
        calibrated params come from STORY_01_03).

    Returns
    -------
    boxes : np.ndarray  shape (N, 6) float32, case-preprocessed-space
    scores : np.ndarray shape (N,) float32
    embeddings : np.ndarray shape (N, D_EMB) float32
    """
    # Import ensemble_combine lazily to avoid circular imports at module load time.
    # (This module is imported by generate_candidates.py which imports from ensemble.py.)
    from abus.detect.candidates import RawCandidate
    from abus.detect.ensemble import ensemble_combine
    from abus.geometry.bbox import BBox

    all_proposals: list[RawCandidate] = []

    # NOTE: no zip(strict=) — that kwarg is Python 3.10+, and this module runs in the
    # server's nndet env (Python 3.8). Default zip() truncates to the shortest iterable,
    # which matches the intended strict=False semantics here. See check_py38_compat.py.
    tile_iter = zip(tiles, feat_maps_per_tile, boxes_per_tile, scores_per_tile)  # noqa: B905
    for tile_idx, (tile, feat_map_raw, tile_boxes, tile_scores) in enumerate(tile_iter):
        if tile_boxes.shape[0] == 0:
            continue

        # Extract the feature map for the specified FPN level.
        if isinstance(feat_map_raw, (list, tuple)):  # noqa: UP038 (py38: no X|Y in isinstance)
            if fpn_level_index >= len(feat_map_raw):
                logger.warning(
                    "_predict_single_case_with_embeddings: tile %d: fpn_level_index=%d "
                    "out of range (%d levels); skipping tile embeddings",
                    tile_idx,
                    fpn_level_index,
                    len(feat_map_raw),
                )
                continue
            feat_map: np.ndarray = np.asarray(feat_map_raw[fpn_level_index])
        else:
            feat_map = np.asarray(feat_map_raw)

        # Remove batch dim if present (B, C, D0, D1, D2) → (C, D0, D1, D2)
        if feat_map.ndim == 5:
            feat_map = feat_map[0]

        # tile_origin in nnDetection (x,y,z) order = (d2, d1, d0).
        # _apply_offsets_to_boxes convention (detection.py:242-249):
        #   boxes[:,0] += offset[0]   (x1 → d2_min)
        #   boxes[:,1] += offset[1]   (y1 → d1_min)
        #   boxes[:,2] += offset[0]   (x2 → d2_max)
        #   boxes[:,3] += offset[1]   (y2 → d1_max)
        #   boxes[:,4] += offset[2]   (z1 → d0_min)
        #   boxes[:,5] += offset[2]   (z2 → d0_max)
        tile_origin = tile.get("tile_origin", (0, 0, 0))
        x_offset = float(tile_origin[0])  # d2
        y_offset = float(tile_origin[1])  # d1
        z_offset = float(tile_origin[2])  # d0

        for i in range(tile_boxes.shape[0]):
            box_tile = tile_boxes[i]  # tile-local (x1,y1,x2,y2,z1,z2)

            # Pool at tile-LOCAL centroid — CORRECT because feat_map and box
            # are both in the same tile-pixel coordinate frame.
            cx_x = (box_tile[0] + box_tile[2]) / 2.0  # d2 in tile space
            cx_y = (box_tile[1] + box_tile[3]) / 2.0  # d1 in tile space
            cx_z = (box_tile[4] + box_tile[5]) / 2.0  # d0 in tile space

            emb = point_pool_trilinear(feat_map, cx_d0=cx_z, cx_d1=cx_y, cx_d2=cx_x)

            # Apply tile_origin offset → case-space box
            x1_case = box_tile[0] + x_offset
            y1_case = box_tile[1] + y_offset
            x2_case = box_tile[2] + x_offset
            y2_case = box_tile[3] + y_offset
            z1_case = box_tile[4] + z_offset
            z2_case = box_tile[5] + z_offset

            # Map (x1,y1,x2,y2,z1,z2) → project BBox (min_d0,min_d1,min_d2,max_d0,max_d1,max_d2)
            # x=d2, y=d1, z=d0
            bbox = BBox(
                min_d0=int(round(float(z1_case))),
                min_d1=int(round(float(y1_case))),
                min_d2=int(round(float(x1_case))),
                max_d0=max(int(round(float(z2_case))), int(round(float(z1_case))) + 1),
                max_d1=max(int(round(float(y2_case))), int(round(float(y1_case))) + 1),
                max_d2=max(int(round(float(x2_case))), int(round(float(x1_case))) + 1),
            )

            all_proposals.append(
                RawCandidate(
                    case_id=0,  # placeholder — caller replaces
                    split="ens_tmp",
                    bbox=bbox,
                    score=float(tile_scores[i]),
                    embedding=emb,
                    source_detectors=(0,),  # placeholder — caller replaces
                )
            )

    if not all_proposals:
        empty = np.zeros((0, 6), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros((0, D_EMB), dtype=np.float32)

    # WBC across all tiles: union → IoU-cluster → score+embedding averaging
    combined = ensemble_combine(all_proposals, iou_threshold=iou_threshold)

    n = len(combined)
    boxes_out = np.zeros((n, 6), dtype=np.float32)
    scores_out = np.zeros(n, dtype=np.float32)
    embeddings_out = np.zeros((n, D_EMB), dtype=np.float32)

    for i, c in enumerate(combined):
        b = c.bbox
        # Map back to (x1,y1,x2,y2,z1,z2) for RawDetectionsWithEmb convention
        boxes_out[i] = [b.min_d2, b.min_d1, b.max_d2, b.max_d1, b.min_d0, b.max_d0]
        scores_out[i] = c.score
        embeddings_out[i] = c.embedding.astype(np.float32)

    return boxes_out, scores_out, embeddings_out


# ---------------------------------------------------------------------------
# predict_with_embeddings  (D01.14 — server-only; lazy nnDetection import)
# ---------------------------------------------------------------------------


def predict_with_embeddings(
    fold: int,
    case_ids: list[int],
    preprocessed_dir: str,
    fold_dir: str,
    fpn_level_index: int = 0,
    num_models: int = 1,
) -> dict[int, RawDetectionsWithEmb]:
    """Run the fold-``fold`` detector on ``case_ids`` with FPN embedding extraction.

    D01.14 embedding-extraction path (post-code-review fix). Uses a custom
    per-tile inference loop (NOT ``predict_dir``) so that:

    1. The decoder hook captures the feature map for each tile in tile-local space.
    2. Per-tile boxes from ``model.inference_step`` are in the SAME tile-local
       coordinate frame as the captured feature map.
    3. Embeddings are pooled at tile-LOCAL box centroids — correct because both
       the feature map and the boxes are in tile-pixel space at this point.
    4. After pooling, ``tile_origin`` is applied to shift boxes to case-preprocessed
       space. Embeddings are carried alongside (they are opaque 128-vecs; the
       coordinate shift applies only to boxes).
    5. ``_predict_single_case_with_embeddings`` handles WBC across tiles and returns
       (boxes, scores, embeddings) for the case.
    6. ``restore_detection`` converts boxes from preprocessed space to original
       image space. Embeddings are not affected.

    **Why NOT predict_dir** (code-review Must-fix #1+2):
    ``predict_dir`` runs the full tiling loop inside nnDetection's ensembler, which:
    (a) applies ``_apply_offsets_to_boxes`` (tile_origin) to every box AFTER inference,
    (b) runs WBC across all tiles, and
    (c) calls ``restore_detection`` to produce original-image-space boxes.
    At that point the hook's feature map is from the LAST tile processed and in TILE
    space — completely mismatched with the case-space / restored boxes.

    **TTA**: uses single NoOp transform (no TTA flips). This avoids TTA-inverse
    complexity on feature maps and is thesis-appropriate — the embedding is extracted
    from one forward pass per tile, which avoids flip-averaging artifacts. Box quality
    from the FROC sweep (predict_oof / predict_dir) is unaffected.

    Axis convention (D01.14, grounded in nnDetection source):
      nnDetection box: (x1,y1,x2,y2,z1,z2) — ops.py line 34
      x=d2, y=d1, z=d0 — centroid mapped to feat map axes [1→d0, 2→d1, 3→d2]
    as implemented in ``point_pool_trilinear``.

    Requires the nnDetection conda env (server-only). nnDetection is lazily
    imported inside this function so the module is importable on the laptop.

    Parameters
    ----------
    fold : int
        Fold index (0–4). Used for logging/traceability.
    case_ids : list[int]
        Integer case IDs to score.
    preprocessed_dir : str
        Path to the preprocessed imagesTr directory containing ``.npz`` files.
    fold_dir : str
        Path to the fold's training output directory (contains config.yaml +
        plan_inference.pkl + model_last.ckpt).
    fpn_level_index : int
        Index into ``model.decoder_levels`` selecting the FPN level to pool from.
        Default 0 = finest detection level (largest spatial resolution, D01.14).
    num_models : int
        Number of models (1 for a single fold). Default 1.

    Returns
    -------
    dict[int, RawDetectionsWithEmb]
        Keyed by case_id (int). Boxes in (x1,y1,x2,y2,z1,z2) axis, original
        image space (restored via restore_detection).
        embeddings is always a real (N, D_EMB) float32 array.

    Raises
    ------
    ImportError
        If nnDetection / torch is not available in the current environment (laptop).
    FileNotFoundError
        If plan_inference.pkl is missing (requires --sweep retrain, D01.13).
    """
    # Lazy imports — nnDetection + torch are server-only; the module must remain
    # importable on the laptop (thesis §3.2.3, D01.13 dependency note).
    try:
        import torch as _torch  # noqa: F401 — verify torch is available before proceeding
        from nndet.inference.loading import load_final_model
        from nndet.inference.restore import restore_detection
        from nndet.io.load import load_pickle
        from nndet.io.patching import create_grid, save_get_crop
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "nnDetection / torch / omegaconf is not installed in the current environment. "
            "predict_with_embeddings requires the nnDetection conda env (server-only). "
            f"Original error: {exc}"
        ) from exc

    fold_path = Path(fold_dir)

    # Load config + plan from fold training output (created by --sweep, D01.13).
    cfg_path = fold_path / "config.yaml"
    plan_path = fold_path / "plan_inference.pkl"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"plan_inference.pkl not found at {plan_path}. "
            "This file is only created when nndet_train runs with --sweep. "
            "Re-train this fold with --sweep (D01.13/D01.14 requirement)."
        )
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    plan = load_pickle(plan_path)

    # Override swept inference params to high-recall defaults (D01.13 point 6).
    inference_params = {}
    if "inference_plan" in plan:
        inference_params = dict(plan["inference_plan"])
    inference_params["model_score_thresh"] = 0.0
    inference_params["ensemble_score_thresh"] = 0.0
    inference_params["model_topk"] = 1000
    inference_params["model_detections_per_image"] = 100

    # Load model (SWA model_last — pre-registered detector, thesis §3.2.3)
    model_list = load_final_model(
        source_models=fold_path,
        cfg=cfg,
        plan=plan,
        num_models=1,
        identifier="last",
    )
    model = model_list[0]["model"]
    model.eval()

    # D01.14 fix: load_final_model returns a CPU model (torch.load map_location="cpu")
    # and never moves it to GPU — nnDetection's normal predict path relies on the
    # Predictor to do that, but this custom tiling loop does not use the Predictor.
    # Without this, next(model.parameters()).device is "cpu" and the entire 3D
    # detector runs on CPU (Step 13 audit silently grinds for many minutes; Steps
    # 14/16 at scale become infeasible). Move to CUDA so inference_step runs on GPU.
    if _torch.cuda.is_available():
        model.cuda()

    # Plan parameters for tiling (same as predictor.create uses)
    crop_size = plan["patch_size"]
    overlap_frac = 0.5
    overlap = [int(c * overlap_frac) for c in crop_size]

    logger.info(
        "predict_with_embeddings: fold=%d, %d cases, preprocessed_dir=%s, fold_dir=%s",
        fold,
        len(case_ids),
        preprocessed_dir,
        fold_dir,
    )

    # --- Hook setup ---
    # Captures the FPN feature map list (features_maps_all) from model.decoder
    # output during each tile forward pass. The hook fires ONCE per forward call,
    # in tile-local pixel space — before any tile_origin offset or restore.
    _tile_feat_maps: list[list[np.ndarray]] = []  # per forward call

    def _decoder_hook(module: object, input: object, output: object) -> None:  # noqa: A002
        """Capture FPN feature maps per tile forward call."""
        if isinstance(output, (list, tuple)):  # noqa: UP038 (py38: no X|Y in isinstance)
            captured = [
                np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t) for t in output
            ]
        else:
            arr = output.detach().cpu().numpy() if hasattr(output, "detach") else np.array(output)
            captured = [arr]
        _tile_feat_maps.append(captured)

    # nnDetection's loaded object is the Lightning module (RetinaUNetModule), whose
    # actual network is at `.model` (base_module.py: self.model = from_config_plan(...))
    # and whose decoder is `network.decoder` (retina.py: self.decoder = decoder).
    # So the decoder is at `model.model.decoder`, NOT `model.decoder`. Resolve the
    # inner network first, then fall back to the top-level object for robustness
    # across nnDetection versions. Getting this wrong yields all-zero embeddings.
    _decoder_module = None
    _inner = getattr(model, "model", None)
    if _inner is not None and hasattr(_inner, "decoder"):
        _decoder_module = _inner.decoder
    elif hasattr(model, "decoder"):
        _decoder_module = model.decoder

    if _decoder_module is not None:
        _hook_handle = _decoder_module.register_forward_hook(_decoder_hook)
    else:
        logger.warning(
            "predict_with_embeddings: could not locate a .decoder attribute on "
            "model.model or model; embeddings will be zeros. Check nnDetection version."
        )
        _hook_handle = None

    result: dict[int, RawDetectionsWithEmb] = {}
    case_ids_str = [f"{cid:04d}" for cid in case_ids]
    preprocessed_path = Path(preprocessed_dir)

    try:
        for case_id_str in case_ids_str:
            case_id = int(case_id_str)

            # Load preprocessed case data
            npz_path = preprocessed_path / f"{case_id_str}.npz"
            if not npz_path.exists():
                npz_path = preprocessed_path / f"{case_id_str}.npy"

            case_data = np.load(str(npz_path), allow_pickle=True)
            if hasattr(case_data, "files"):
                case_arr = case_data["data"]  # npz: shape (1, D0, D1, D2) or (C, D0, D1, D2)
            else:
                case_arr = case_data  # npy

            # Load properties for restoration
            props_path = preprocessed_path / f"{case_id_str}.pkl"
            properties = load_pickle(str(props_path))
            properties["transpose_backward"] = plan["transpose_backward"]

            # Create tiles using nnDetection's tiling (same grid as predict_dir uses)
            dshape = case_arr.shape
            tiles_crops = create_grid(
                cshape=crop_size,
                dshape=dshape[1:],
                overlap=overlap,
                mode="symmetric",
            )

            # Per-case: reset the hook's capture list
            _tile_feat_maps.clear()

            tiles: list[dict] = []
            for crop in tiles_crops:
                try:
                    tile_data, tile_origin, tile_crop = save_get_crop(case_arr, crop, mode="shift")
                except RuntimeError:
                    tile_data, tile_origin, tile_crop = save_get_crop(
                        case_arr, crop, mode="symmetric"
                    )
                tiles.append(
                    {
                        "data": tile_data,
                        "tile_origin": tile_origin,
                        "crop": tile_crop,
                    }
                )

            # Per-tile inference with hook active
            boxes_per_tile: list[np.ndarray] = []
            scores_per_tile: list[np.ndarray] = []
            feat_maps_per_tile: list[np.ndarray] = []

            import torch  # already available (lazy-imported above via _torch)  # noqa: PLC0415

            model_device = (
                next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
            )

            with torch.no_grad():
                for tile in tiles:
                    tile_data_tensor = torch.from_numpy(
                        tile["data"][None].astype(np.float32)  # add batch dim
                    ).to(model_device)

                    # inference_step returns dict with pred_boxes/pred_scores/pred_labels
                    # in tile-LOCAL pixel space (NOT yet offset by tile_origin)
                    result_tile = model.inference_step(tile_data_tensor, batch_num=0)

                    # The hook fired during this inference_step call (via model.forward).
                    # feat_maps_per_tile: one entry per tile (the LAST hook call for this tile).
                    if _tile_feat_maps:
                        feat_maps_per_tile.append(_tile_feat_maps[-1])
                        _tile_feat_maps.clear()
                    else:
                        # No hook capture (decoder hook not attached) — use empty
                        feat_maps_per_tile.append([])

                    # Extract per-tile boxes/scores (list with one element for batch_size=1)
                    tile_boxes_list = result_tile.get("pred_boxes", [])
                    tile_scores_list = result_tile.get("pred_scores", [])

                    if tile_boxes_list and len(tile_boxes_list) > 0:
                        b = tile_boxes_list[0]
                        s = tile_scores_list[0]
                        b_np = b.detach().cpu().numpy() if hasattr(b, "detach") else np.array(b)
                        s_np = s.detach().cpu().numpy() if hasattr(s, "detach") else np.array(s)
                        boxes_per_tile.append(b_np.astype(np.float32))
                        scores_per_tile.append(s_np.astype(np.float32))
                    else:
                        boxes_per_tile.append(np.zeros((0, 6), dtype=np.float32))
                        scores_per_tile.append(np.zeros(0, dtype=np.float32))

            # Per-tile embedding pooling + WBC across tiles
            boxes_case, scores_case, embeddings_case = _predict_single_case_with_embeddings(
                tiles=tiles,
                feat_maps_per_tile=feat_maps_per_tile,
                boxes_per_tile=boxes_per_tile,
                scores_per_tile=scores_per_tile,
                fpn_level_index=fpn_level_index,
                iou_threshold=0.5,
            )

            # boxes_case is in case-preprocessed space; restore to original image space.
            if boxes_case.shape[0] > 0:
                boxes_restored = restore_detection(
                    boxes_case.astype(np.float64),
                    transpose_backward=properties["transpose_backward"],
                    original_spacing=properties["original_spacing"],
                    spacing_after_resampling=properties["spacing_after_resampling"],
                    crop_bbox=properties["crop_bbox"],
                ).astype(np.float32)
            else:
                boxes_restored = boxes_case

            result[case_id] = RawDetectionsWithEmb(
                case_id=case_id,
                boxes=boxes_restored,
                scores=scores_case,
                embeddings=embeddings_case,
            )

            logger.info(
                "predict_with_embeddings: fold=%d case %s — %d detections",
                fold,
                case_id_str,
                boxes_restored.shape[0],
            )

    finally:
        # Always remove the hook to avoid memory leaks / interference
        if _hook_handle is not None:
            _hook_handle.remove()

    logger.info(
        "predict_with_embeddings: fold=%d — processed %d cases with embeddings",
        fold,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# preprocess_val_test  (D01.14b — server-only; lazy nnDetection import)
# ---------------------------------------------------------------------------


def preprocess_val_test(
    fold_dir: str,
    num_processes: int = 0,
) -> None:
    """Preprocess val/test raw images into preprocessed/<data_identifier>/imagesTs/.

    D01.14b fix: val/test cases (0100–0199) are NOT in imagesTr — they live in
    raw_splitted/imagesTs/ (raw NIfTI). predict_with_embeddings reads preprocessed
    .npz files; it cannot operate on raw images. This function runs the planner's
    run_preprocessing_test to populate preprocessed/<data_identifier>/imagesTs/ before
    the val/test ensemble step.

    Mirrors scripts/predict.py:74-81 (nnDetection-main, commit 97a58f3):
        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=cfg["host"]["preprocessed_output_dir"],
            splitted_4d_output_dir=cfg["host"]["splitted_4d_output_dir"],
            plan=plan,
            num_processes=num_processes,
        )

    After this call, preprocessed/<data_identifier>/imagesTs/ contains .npz + .pkl
    files for val/test cases (4-digit zero-padded stems: 0100.npz, ..., 0199.npz).
    predict_with_embeddings can then be pointed at imagesTs to score these cases.

    This function is idempotent: run_preprocessing_test skips already-processed
    cases (it calls get_case_ids_from_dir first and passes them as remove_ids).

    Parameters
    ----------
    fold_dir : str
        Path to any fold's training output directory (e.g.
        ``$det_models/<task>/<exp>/fold0``). Provides config.yaml (for host paths)
        and plan_inference.pkl (for planner_id and preprocessing plan).
    num_processes : int
        Number of parallel preprocessing workers. 0 = sequential (safe on laptop
        for testing; use >=1 on the server). Default 0.

    Raises
    ------
    ImportError
        If nnDetection / omegaconf is not available in the current environment (laptop).
    FileNotFoundError
        If plan_inference.pkl is missing (requires --sweep retrain, D01.13).
    """
    # Lazy imports — nnDetection is server-only; the module must remain importable
    # on the laptop (thesis §3.2.3, D01.14b).
    try:
        from nndet.io.load import load_pickle
        from nndet.planning import PLANNER_REGISTRY
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "nnDetection / omegaconf is not installed in the current environment. "
            "preprocess_val_test requires the nnDetection conda env (server-only). "
            f"Original error: {exc}"
        ) from exc

    fold_path = Path(fold_dir)
    cfg_path = fold_path / "config.yaml"
    plan_path = fold_path / "plan_inference.pkl"

    if not plan_path.exists():
        raise FileNotFoundError(
            f"plan_inference.pkl not found at {plan_path}. "
            "This file is only created when nndet_train runs with --sweep. "
            "Re-train with --sweep (D01.13 requirement)."
        )

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    plan = load_pickle(plan_path)

    planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])

    logger.info(
        "preprocess_val_test: planner=%s, preprocessed_output_dir=%s, "
        "splitted_4d_output_dir=%s, num_processes=%d",
        plan["planner_id"],
        cfg["host"]["preprocessed_output_dir"],
        cfg["host"]["splitted_4d_output_dir"],
        num_processes,
    )

    planner_cls.run_preprocessing_test(
        preprocessed_output_dir=cfg["host"]["preprocessed_output_dir"],
        splitted_4d_output_dir=cfg["host"]["splitted_4d_output_dir"],
        plan=plan,
        num_processes=num_processes,
    )

    logger.info("preprocess_val_test: done")
