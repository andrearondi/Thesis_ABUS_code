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
    Used by parse_predictions_dir (consolidated-dict reader) and predict_oof.
    Box axis order: (x1, y1, x2, y2, z1, z2) — nndet/core/boxes/ops.py line 34.

  RawDetectionsWithEmb
    Like RawDetections but embeddings is always a real (N, D_EMB) float32 array.
    Produced by predict_with_embeddings (D01.14 embedding-extraction path).

  point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2) -> np.ndarray (C,)
    Trilinear point-pooling of a (C, D0, D1, D2) feature map at a centroid given
    in tile-pixel frame as (cx_d0, cx_d1, cx_d2). CPU-only; no nnDetection required.
    D01.14 axis convention: feat axis 1=d0=z, axis 2=d1=y, axis 3=d2=x.

  parse_predictions_dir(pred_dir) -> dict[int, RawDetections]
    Discovers *_boxes.pkl files in pred_dir and parses them (consolidated-dict schema,
    server-verified 2026-06-23 against nnDetection commit 97a58f3).

  predict_oof(fold, case_ids, preprocessed_dir, fold_dir, ...) -> dict[int, RawDetections]
    Drives nnDetection 0.1's predict_dir(case_ids=...) helper for the OOF path.
    Requires the nnDetection conda env (server-only). Lazily imported.

  predict_with_embeddings(fold, case_ids, preprocessed_dir, fold_dir, ...)
      -> dict[int, RawDetectionsWithEmb]
    [RETIRED D01.17] Per-tile embedding-extraction loop. REPLACED by the decoupled
    pool_embeddings_at_boxes. Kept for reference and unit tests only.
    Do NOT call in new code.

  pool_embeddings_at_boxes(fold, case_id, boxes_preprocessed, preprocessed_dir,
                            fold_dir, *, pooling="centroid", fpn_level_index=0)
      -> np.ndarray (N, D_EMB)
    D01.17 post-hoc decoupled embedding pooler. Loads the fold's model, hooks
    model.model.decoder, re-tiles the preprocessed case, and pools the finest FPN
    level at each native box (from predict_oof(restore=False)) via grid_pool.
    RAISES RuntimeError if decoder not found (no silent zeros). Requires nnDetection
    conda env (server-only). Lazy imports.

    OOM fix (D01.14-OOM, 2026-06-20): streaming per-tile pooling.  The hook
    now retains only the level-0 (finest) feature map; pooling happens
    immediately after each tile's forward pass so that the tile's feature map
    is released before the next tile begins.  Peak host-RAM is bounded to
    ~1 tile's worth of level-0 feature map rather than all tiles simultaneously.

  restore_boxes_for_case(boxes_preprocessed, case_id, preprocessed_dir, fold_dir)
      -> np.ndarray (N, 6)
    D01.18 helper: applies nnDetection's restore_detection to convert boxes from
    preprocessed/storage space (as returned by predict_oof(restore=False)) to
    original-image-grid space. Used by generate_candidates.py to build the
    RawCandidate.bbox in the original-grid frame required by STORY_01_03 GT matching.
    Loads plan["transpose_backward"] from plan_inference.pkl and per-case properties
    from <preprocessed_dir>/<case_id_str>.pkl (the per-case metadata written by
    nndet_prep). Requires nnDetection conda env (server-only). Lazy imports.

  preprocess_val_test(fold_dir, num_processes) -> None
    D01.14b val/test preprocessing step. Runs the nnDetection planner's
    run_preprocessing_test to populate preprocessed/<data_identifier>/imagesTs/
    from the raw test set (raw_splitted/imagesTs/). Must be called BEFORE
    predict_with_embeddings for val/test cases. Lazy imports; server-only.

nnDetection 0.1 output schema (server-verified 2026-06-23, commit 97a58f3):
  predict_dir(save_state=False) writes ONE CONSOLIDATED FILE PER CASE:
    <case_id>_boxes.pkl  — pickle of dict with keys:
      "pred_boxes":   np.ndarray (N, 6) float32 — box axis (x1, y1, x2, y2, z1, z2)
      "pred_scores":  np.ndarray (N,)   float32
      "pred_labels":  np.ndarray (N,)   float32
      "restore":      bool scalar
      "original_size_of_raw_data": np.ndarray (3,) int64
      "itk_origin":   np.ndarray (3,)   float64
      "itk_spacing":  np.ndarray (3,)   float64
      "itk_direction":np.ndarray (9,)   float64

  Verified by loading the actual file for case 5 on the server:
    0005_boxes.pkl  ->  dict, keys as above
    pred_boxes shape (265, 6) float32, pred_scores shape (265,) float32

  The previously documented D01.13 per-key schema (predict_dir writing separate
  <case>_pred_boxes.pkl + <case>_pred_scores.pkl files) was WRONG. The local
  nnDetection-main checkout (also commit 97a58f3) uses a different code path
  (helper.py in the local copy writes per-key files); the server's installed
  package at commit 97a58f3 writes the consolidated dict. Trust the server
  probe, not the local source.

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
    """Discover and parse nnDetection 0.1's consolidated outputs in ``pred_dir``.

    Real schema (server-verified 2026-06-23, commit 97a58f3):
      predict_dir(save_state=False) writes ONE FILE PER CASE:
        <case_id>_boxes.pkl  — pickle of dict containing:
          "pred_boxes":   np.ndarray (N, 6) float32  (x1,y1,x2,y2,z1,z2)
          "pred_scores":  np.ndarray (N,)   float32
          "pred_labels":  np.ndarray (N,)   float32
          "restore":      bool scalar
          "original_size_of_raw_data": np.ndarray (3,) int64
          "itk_origin":   np.ndarray (3,)   float64
          "itk_spacing":  np.ndarray (3,)   float64
          "itk_direction":np.ndarray (9,)   float64
      Box axis: (x1, y1, x2, y2, z1, z2) — nndet/core/boxes/ops.py line 34.
      NO embeddings key. RawDetections.embeddings is always None.

    NOTE: The D01.13 docstring previously claimed per-key files
    (<case>_pred_boxes.pkl + <case>_pred_scores.pkl). That was WRONG. The server
    probe (2026-06-23) showed the consolidated dict schema above. The glob pattern
    was changed from ``*_pred_boxes.pkl`` to ``*_boxes.pkl`` to match reality.

    Parsing rules:
      - Only files matching ``*_boxes.pkl`` are processed (anchors case discovery).
      - Filename format: ``<case_id_str>_boxes.pkl`` where ``<case_id_str>`` is a
        zero-padded integer (e.g. ``0042``). The integer is extracted by stripping
        ``_boxes`` and calling int().
      - Files whose prefix cannot be parsed as an integer are skipped with a
        WARNING log (defect-prevention: logged, not silently absorbed).
      - Each .pkl is loaded as a dict. If "pred_boxes" or "pred_scores" keys are
        missing, a WARNING is logged and the case is skipped.
      - A case with pred_boxes shape (0, 6) (genuinely empty detection set) is
        valid and produces a RawDetections with empty arrays — distinct from
        file-not-found or malformed dict.
      - Boxes and scores are coerced to float32.

    Pure-Python; CPU-safe; no nnDetection required.

    Parameters
    ----------
    pred_dir : str
        Path to the directory containing consolidated pickle files.

    Returns
    -------
    dict[int, RawDetections]
        Keyed by case_id. Empty dict if no matching files found.
    """
    pred_path = Path(pred_dir)
    result: dict[int, RawDetections] = {}

    # Anchor on *_boxes.pkl — one per case (server-verified consolidated schema,
    # 2026-06-23, commit 97a58f3). Do NOT use *_pred_boxes.pkl; that glob matches
    # nothing on the server.
    boxes_files = sorted(pred_path.glob("*_boxes.pkl"))

    for boxes_file in boxes_files:
        stem = boxes_file.stem  # e.g. "0042_boxes"
        suffix = "_boxes"
        # The glob guarantees stems end with "_boxes", but this guard is
        # belt-and-suspenders: if the glob pattern ever changes, the suffix
        # check prevents silent misparses rather than silently producing wrong case_ids.
        if not stem.endswith(suffix):
            logger.warning(
                "parse_predictions_dir: skipping file %s — unexpected stem %r",
                boxes_file.name,
                stem,
            )
            continue

        case_id_str = stem[: -len(suffix)]
        # Strict integer check: nnDetection emits 4-digit zero-padded names only.
        # Guard against Python's int() accepting underscores, sign chars, whitespace.
        if not case_id_str.isdigit():
            logger.warning(
                "parse_predictions_dir: skipping file %s — prefix %r is not a "
                "pure decimal integer (expected 4-digit zero-padded case_id)",
                boxes_file.name,
                case_id_str,
            )
            continue
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

        with open(boxes_file, "rb") as f:
            pred_dict = pickle.load(f)  # noqa: S301

        if not isinstance(pred_dict, dict):
            logger.warning(
                "parse_predictions_dir: skipping case_id=%d — expected dict in %s, " "got %s",
                case_id,
                boxes_file.name,
                type(pred_dict).__name__,
            )
            continue

        if "pred_boxes" not in pred_dict or "pred_scores" not in pred_dict:
            logger.warning(
                "parse_predictions_dir: skipping case_id=%d — file %s missing "
                "'pred_boxes' or 'pred_scores' key (keys found: %s)",
                case_id,
                boxes_file.name,
                sorted(pred_dict.keys()),
            )
            continue

        boxes = np.asarray(pred_dict["pred_boxes"], dtype=np.float32)
        scores = np.asarray(pred_dict["pred_scores"], dtype=np.float32)

        # A case with pred_boxes shape (0, 6) is valid (no detections) — keep it.
        # Shape validation: boxes must be 2-D with last dim 6.
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            logger.warning(
                "parse_predictions_dir: skipping case_id=%d — pred_boxes has unexpected "
                "shape %s (expected (N, 6))",
                case_id,
                boxes.shape,
            )
            continue

        # Embeddings are NOT in the consolidated predict_dir output.
        # pool_embeddings_at_boxes fills them in the decoupled D01.17 path.
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
    restore: bool = False,  # D01.17: False = boxes in preprocessed space (same as FPN maps)
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
         - restore      = False (D01.17: boxes kept in preprocessed space,
                          same space as FPN feature maps; pool_embeddings_at_boxes
                          receives these boxes directly — no affine inversion needed)
         - case_ids     = [f"{cid:04d}" for cid in case_ids]
           (4-digit zero-padded strings matching preprocessed .npz stems)
         - save_state   = False (write consolidated per-case dicts; server schema 2026-06-23)
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
        Keyed by case_id (int). Boxes in (x1,y1,x2,y2,z1,z2) axis in
        PREPROCESSED space (restore=False, D01.17). Pass directly to
        pool_embeddings_at_boxes without coordinate conversion.
        embeddings is always None (not in consolidated predict_dir output, 2026-06-23).

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
            restore=restore,  # D01.17 default=False: boxes in preprocessed space (= FPN space)
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
# pool_embeddings_at_boxes  (D01.17 — POST-HOC DECOUPLED embedding pooler)
# ---------------------------------------------------------------------------


def pool_embeddings_at_boxes(
    fold: int,
    case_id: int,
    boxes_preprocessed: np.ndarray,
    preprocessed_dir: str,
    fold_dir: str,
    *,
    pooling: str = "centroid",
    fpn_level_index: int = 0,
) -> np.ndarray:
    """Post-hoc decoupled 128-D embedding pooler for ONE case (D01.17).

    Pools a 128-D embedding from the Retina U-Net's FPN decoder feature map at
    each native consensus box (from ``predict_oof(restore=False)``).

    D01.17 design:
      1. Loads fold's SWA model_last, moves to GPU.
      2. Registers a forward hook on ``model.model.decoder`` capturing
         ``decoder_levels[fpn_level_index]`` as a CPU numpy array.
         RAISES RuntimeError if the decoder module cannot be resolved —
         NEVER returns silent zeros (D01.17 inversion path #2, removes the
         D01.14 silent-zero fallback).
      3. Re-tiles the SAME preprocessed case on the SAME native grid (create_grid
         + save_get_crop) and runs each tile's forward under the SAME
         ``torch.cuda.amp.autocast()`` the native predictor uses (predictor.py:305).
      4. For each input native box (preprocessed space): selects the tile whose
         centre is nearest the box centroid (deterministic), maps the box into that
         tile's feature coords by dividing by per-axis stride (= patch_size /
         feature_map_size, anchors.py:225), and pools via
         ``grid_pool(..., mode=pooling)`` → a 128-D vector.
      5. Returns (N, 128) float32. Row i ↔ input box i (1:1, no re-match, no drop).

    Memory: streaming — captures only one tile's level-0 map at a time, released
    before the next tile (same D01.14-OOM bound).
    Coordinate space: PREPROCESSED (restore=False); restore_detection is NEVER inverted.
    Level PINNED to ``decoder_levels[fpn_level_index]`` (default 0 = finest). Anchor-level
    recovery is intentionally NOT implemented (D01.17 §4).

    Parameters
    ----------
    fold : int
        Fold index (0–4). Used for logging/traceability.
    case_id : int
        Integer case ID; converted to f"{case_id:04d}" .npz stem internally.
    boxes_preprocessed : np.ndarray
        Shape (N, 6) float32. Native final boxes in preprocessed space from
        ``predict_oof(restore=False)`` in axis (x1,y1,x2,y2,z1,z2).
    preprocessed_dir : str
        Path to the preprocessed imagesTr or imagesTs directory (the SAME dir
        passed to ``predict_oof``).
    fold_dir : str
        Path to the fold's training output directory (``config.yaml`` +
        ``plan_inference.pkl`` + checkpoints).
    pooling : str
        Pooling mode. "centroid" (default) or "roi_align".
    fpn_level_index : int
        Index into ``decoder_levels`` for the FPN level to pool from.
        PINNED = 0 (finest detection level). Do not change without a new D01.xx.

    Returns
    -------
    np.ndarray
        Shape (N, 128) float32. One embedding row per input box.

    Raises
    ------
    RuntimeError
        If ``model.model.decoder`` cannot be resolved (build error, not a warning).
    ImportError
        If nnDetection / torch is not available (laptop).
    FileNotFoundError
        If ``plan_inference.pkl`` is missing (requires ``--sweep`` retrain).
    """
    # Lazy imports — nnDetection + torch are server-only; the module must remain
    # importable on the laptop (thesis §3.2.3, D01.17).
    try:
        import torch as _torch  # type: ignore[import-not-found]
        from nndet.inference.loading import load_final_model
        from nndet.io.load import load_pickle
        from nndet.io.patching import create_grid, save_get_crop
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "nnDetection / torch / omegaconf is not installed in the current environment. "
            "pool_embeddings_at_boxes requires the nnDetection conda env (server-only). "
            f"Original error: {exc}"
        ) from exc

    # Lazy import of the grid_pool operator (also lazy-importable on laptop)
    from abus.detect.grid_pool import grid_pool as _grid_pool

    fold_path = Path(fold_dir)
    case_id_str = f"{case_id:04d}"

    # Load config + plan
    cfg_path = fold_path / "config.yaml"
    plan_path = fold_path / "plan_inference.pkl"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"plan_inference.pkl not found at {plan_path}. "
            "This file is only created when nndet_train runs with --sweep. "
            "Re-train this fold with --sweep (D01.17/D01.13 requirement)."
        )
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    plan = load_pickle(plan_path)

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

    # Move to GPU (load_final_model returns CPU model; GPU needed for inference)
    if _torch.cuda.is_available():
        model.cuda()

    model_device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    # Resolve decoder module and decoder_levels (D01.17: RAISE if not found — no silent zeros).
    #
    # CRITICAL — decoder output indexing (ML-review B1, confirmed from nnDetection source):
    #   The decoder returns ALL resolution feature maps: output[0] is the FINEST resolution,
    #   but the detection head only consumes output[decoder_levels[i]].
    #   For ABUS volumes (num_resolutions ≥ 6), decoder_levels[0] = 2 (never 0).
    #   Source: retina.py:219-220 `feature_maps_head = [features_maps_all[i] for i in
    #   self.decoder_levels]`; planning/base.py:534 `decoder_levels_start =
    #   min(max(0, num_resolutions - 4), 2)`.
    #   Boxes from predict_oof are in the coordinate frame of feature_maps_head[0]
    #   = output[decoder_levels[0]], NOT output[0]. Using output[0] gives a ~4× stride
    #   error and wrong channel count (32 vs 128).
    _decoder_module = None
    _decoder_levels = None
    _inner = getattr(model, "model", None)
    if _inner is not None and hasattr(_inner, "decoder"):
        _decoder_module = _inner.decoder
        if hasattr(_inner, "decoder_levels"):
            _decoder_levels = list(_inner.decoder_levels)
    elif hasattr(model, "decoder"):
        _decoder_module = model.decoder
        if hasattr(model, "decoder_levels"):
            _decoder_levels = list(model.decoder_levels)

    if _decoder_module is None:
        raise RuntimeError(
            "pool_embeddings_at_boxes: cannot resolve decoder module. "
            "Expected model.model.decoder or model.decoder. "
            "Check nnDetection version (D01.17 inversion path #2). "
            "Do NOT fallback to zeros — this is a build error."
        )

    if _decoder_levels is None:
        raise RuntimeError(
            "pool_embeddings_at_boxes: cannot resolve model.model.decoder_levels. "
            "Expected model.model.decoder_levels (BaseRetinaNet attribute). "
            "Check nnDetection version (D01.17 B1 fix)."
        )

    # The hook must select the decoder output at decoder_levels[fpn_level_index],
    # not at fpn_level_index itself — the decoder returns all levels, head uses a subset.
    _hook_output_idx = _decoder_levels[fpn_level_index]

    logger.info(
        "pool_embeddings_at_boxes: decoder_levels=%s, fpn_level_index=%d, "
        "selecting output index %d (D01.17 B1 fix).",
        _decoder_levels,
        fpn_level_index,
        _hook_output_idx,
    )

    # Hook: capture the correct FPN level feature map per tile (streaming — one tile at a time)
    _tile_feat: list[np.ndarray] = []

    def _hook(module: object, input: object, output: object) -> None:  # noqa: A002
        if isinstance(output, (list, tuple)):  # noqa: UP038 (py38: no X|Y in isinstance)
            if _hook_output_idx >= len(output):
                # Should not happen if decoder_levels is self-consistent with model output
                return
            t = output[_hook_output_idx]
        else:
            t = output
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        _tile_feat.append(arr)

    _hook_handle = _decoder_module.register_forward_hook(_hook)

    # Tiling parameters (same as predict_dir / predict_with_embeddings)
    crop_size = plan["patch_size"]
    overlap_frac = 0.5
    overlap = [int(c * overlap_frac) for c in crop_size]

    # Load preprocessed case
    preprocessed_path = Path(preprocessed_dir)
    npz_path = preprocessed_path / f"{case_id_str}.npz"
    if not npz_path.exists():
        npz_path = preprocessed_path / f"{case_id_str}.npy"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed case not found: {preprocessed_path}/{case_id_str}.npz/.npy"
        )

    case_data = np.load(str(npz_path), allow_pickle=True)
    if hasattr(case_data, "files"):
        case_arr = case_data["data"]  # (C or 1, D0, D1, D2)
    else:
        case_arr = case_data

    # Build tile grid (same as predict_dir does — deterministic)
    tiles_crops = create_grid(
        cshape=crop_size,
        dshape=case_arr.shape[1:],
        overlap=overlap,
        mode="symmetric",
    )

    # Compute tile origins in preprocessed space for nearest-tile assignment.
    # tile_origin from save_get_crop is in STORAGE ORDER: (d0_start, d1_start, d2_start).
    tile_origins = []
    try:
        for crop in tiles_crops:
            _, tile_origin, _ = save_get_crop(case_arr, crop, mode="shift")
            tile_origins.append(tile_origin)  # storage order: (d0, d1, d2)
    except RuntimeError:
        tile_origins = []
        for crop in tiles_crops:
            _, tile_origin, _ = save_get_crop(case_arr, crop, mode="symmetric")
            tile_origins.append(tile_origin)

    n_boxes = boxes_preprocessed.shape[0]
    embeddings_out = np.zeros((n_boxes, D_EMB), dtype=np.float32)
    # Track which boxes were successfully filled (M2: no silent-zero rows).
    _filled = np.zeros(n_boxes, dtype=bool)

    if n_boxes == 0:
        _hook_handle.remove()
        return embeddings_out

    # For each box: identify the nearest tile by centroid distance in preprocessed space.
    #
    # AXIS CONVENTION (D01.17, code-review M1, confirmed from nnDetection source):
    #   restore=False boxes from predict_oof are in STORAGE ORDER:
    #     box slot 0 (labeled "x1") → storage axis d0
    #     box slot 1 (labeled "y1") → storage axis d1
    #     box slot 4 (labeled "z1") → storage axis d2
    #   (Physical labels — acoustic depth / lateral / elevational — depend on
    #   transpose_forward for the specific dataset and are NOT asserted here.)
    #
    #   Source: nnDetection detection.py:242-249 — _apply_offsets_to_boxes adds
    #     tile_origin[0] → slot 0,2; tile_origin[1] → slot 1,3; tile_origin[2] → slot 4,5
    #     where tile_origin is from save_get_crop → origin=[int(x.start)...], storage order.
    #
    #   The nnDetection "x/y/z" labels on box columns are column-pair identifiers,
    #   NOT physical axis assignments. They do NOT pin box-slot-0 to storage axis d2.
    #   (This was the source of the D01.17 M1 bug in the original implementation.)
    #
    #   tile_origin is in storage order (d0, d1, d2).
    #   crop_size (plan["patch_size"]) is in storage order (d0, d1, d2).

    # Box centroids in preprocessed space, using correct slot-to-axis mapping:
    box_cx_d0 = (boxes_preprocessed[:, 0] + boxes_preprocessed[:, 2]) / 2.0  # slot 0,2 → d0
    box_cx_d1 = (boxes_preprocessed[:, 1] + boxes_preprocessed[:, 3]) / 2.0  # slot 1,3 → d1
    box_cx_d2 = (boxes_preprocessed[:, 4] + boxes_preprocessed[:, 5]) / 2.0  # slot 4,5 → d2

    # tile_origin in storage order (d0, d1, d2); crop_size likewise
    tile_origin_arr = np.array(tile_origins, dtype=np.float32)  # (n_tiles, 3)
    tile_cx_d0 = tile_origin_arr[:, 0] + crop_size[0] / 2.0  # d0
    tile_cx_d1 = tile_origin_arr[:, 1] + crop_size[1] / 2.0  # d1
    tile_cx_d2 = tile_origin_arr[:, 2] + crop_size[2] / 2.0  # d2

    # nearest tile index for each box
    dists = (
        (box_cx_d0[:, None] - tile_cx_d0[None, :]) ** 2
        + (box_cx_d1[:, None] - tile_cx_d1[None, :]) ** 2
        + (box_cx_d2[:, None] - tile_cx_d2[None, :]) ** 2
    )
    nearest_tile = dists.argmin(axis=1)  # (N,)

    # Group boxes by tile to run each tile's forward only once
    from collections import defaultdict

    boxes_by_tile: dict = defaultdict(list)
    for box_idx, tile_idx_val in enumerate(nearest_tile):
        boxes_by_tile[int(tile_idx_val)].append(box_idx)

    # Run forward per tile, pool embeddings
    try:
        with _torch.no_grad():
            for tile_idx, crop in enumerate(tiles_crops):
                if tile_idx not in boxes_by_tile:
                    continue  # no boxes assigned to this tile — skip

                # Load and run this tile
                _tile_feat.clear()
                try:
                    tile_data, tile_origin, _ = save_get_crop(case_arr, crop, mode="shift")
                except RuntimeError:
                    tile_data, tile_origin, _ = save_get_crop(case_arr, crop, mode="symmetric")

                tile_tensor = _torch.from_numpy(tile_data[None].astype(np.float32)).to(model_device)

                with _torch.cuda.amp.autocast():  # noqa: SIM117 — autocast needed for dtype match
                    _ = model.inference_step(tile_tensor, batch_num=0)

                # Grab captured feature map (level-0 only; hook already selected it).
                # If hook did not fire: log and skip — _filled stays False for these
                # boxes, triggering RuntimeError after the tile loop (M2 guard).
                if not _tile_feat:
                    logger.warning(
                        "pool_embeddings_at_boxes: hook did not fire for tile %d "
                        "(case %s, fold %d). Will raise RuntimeError after tile loop.",
                        tile_idx,
                        case_id_str,
                        fold,
                    )
                    continue

                feat_raw = _tile_feat[-1]
                _tile_feat.clear()

                # feat_raw shape: (B, C, D0, D1, D2) or (C, D0, D1, D2)
                if feat_raw.ndim == 5:
                    feat_raw = feat_raw[0]  # drop batch dim → (C, D0, D1, D2)

                feat_shape = feat_raw.shape  # (C, D0f, D1f, D2f)
                C_feat, D0f, D1f, D2f = feat_shape

                # Guard: the correct detection level must have exactly D_EMB=128 channels.
                # A mismatch means the hook selected the wrong decoder output level (B1).
                if C_feat != D_EMB:
                    raise RuntimeError(
                        f"pool_embeddings_at_boxes: captured feature map has {C_feat} channels, "
                        f"expected {D_EMB} (D_EMB). This means the FPN hook selected the wrong "
                        f"decoder level (decoder_levels[{fpn_level_index}] = {_hook_output_idx}). "
                        f"Captured shape: {feat_raw.shape}. "
                        "Do NOT proceed — embeddings from the wrong level corrupt all GNN inputs."
                    )

                # Per-axis stride: how many preprocessed voxels per feature voxel.
                # crop_size from plan["patch_size"] is in storage order (d0, d1, d2).
                # Feature map (C, D0f, D1f, D2f) likewise in storage order.
                stride_d0 = crop_size[0] / D0f  # d0 stride
                stride_d1 = crop_size[1] / D1f  # d1 stride
                stride_d2 = crop_size[2] / D2f  # d2 stride

                # Map each assigned box into this tile's feature space.
                #
                # Box column-to-axis mapping (D01.17 M1 fix, confirmed from nnDetection source):
                #   box slot 0 ("x1") → storage d0   box slot 2 ("x2") → storage d0
                #   box slot 1 ("y1") → storage d1   box slot 3 ("y2") → storage d1
                #   box slot 4 ("z1") → storage d2   box slot 5 ("z2") → storage d2
                #
                # tile_origin is storage order (d0_start, d1_start, d2_start).
                for box_idx in boxes_by_tile[tile_idx]:
                    box = boxes_preprocessed[box_idx]  # (slot0..5)
                    # Unpack using correct storage-axis names
                    d0_1_pre = box[0]  # slot 0 = d0 lower
                    d1_1_pre = box[1]  # slot 1 = d1 lower
                    d0_2_pre = box[2]  # slot 2 = d0 upper
                    d1_2_pre = box[3]  # slot 3 = d1 upper
                    d2_1_pre = box[4]  # slot 4 = d2 lower
                    d2_2_pre = box[5]  # slot 5 = d2 upper

                    # Shift to tile-local preprocessed coords
                    d0_1_tile = d0_1_pre - tile_origin[0]  # d0 - d0_start
                    d0_2_tile = d0_2_pre - tile_origin[0]
                    d1_1_tile = d1_1_pre - tile_origin[1]  # d1 - d1_start
                    d1_2_tile = d1_2_pre - tile_origin[1]
                    d2_1_tile = d2_1_pre - tile_origin[2]  # d2 - d2_start
                    d2_2_tile = d2_2_pre - tile_origin[2]

                    # Convert to feature coords by dividing by per-axis stride
                    d0_1_feat = d0_1_tile / stride_d0  # feat axis 1 (D0f)
                    d0_2_feat = d0_2_tile / stride_d0
                    d1_1_feat = d1_1_tile / stride_d1  # feat axis 2 (D1f)
                    d1_2_feat = d1_2_tile / stride_d1
                    d2_1_feat = d2_1_tile / stride_d2  # feat axis 3 (D2f)
                    d2_2_feat = d2_2_tile / stride_d2

                    if pooling == "centroid":
                        # grid_pool centroid signature: point = (x=cx_d2, y=cx_d1, z=cx_d0)
                        cx_d0_feat = (d0_1_feat + d0_2_feat) / 2.0
                        cx_d1_feat = (d1_1_feat + d1_2_feat) / 2.0
                        cx_d2_feat = (d2_1_feat + d2_2_feat) / 2.0
                        point = np.array([cx_d2_feat, cx_d1_feat, cx_d0_feat], dtype=np.float32)
                        emb = _grid_pool(feat_raw, point, mode="centroid", align_corners=False)
                    else:
                        # grid_pool roi_align box signature: (x1,y1,x2,y2,z1,z2)
                        # = (d2_lower, d1_lower, d2_upper, d1_upper, d0_lower, d0_upper)
                        box_feat = np.array(
                            [
                                d2_1_feat,
                                d1_1_feat,
                                d2_2_feat,
                                d1_2_feat,
                                d0_1_feat,
                                d0_2_feat,
                            ],
                            dtype=np.float32,
                        )
                        emb = _grid_pool(feat_raw, box_feat, mode="roi_align", align_corners=False)

                    embeddings_out[box_idx] = emb
                    _filled[box_idx] = True

                # Release tile data
                del tile_data, tile_tensor, feat_raw

    finally:
        _hook_handle.remove()

    # M2: raise if any box that was assigned to a tile ended up unfilled
    # (hook did not fire for that tile, silent-zero prevention).
    unfilled_indices = np.where(~_filled)[0]
    if len(unfilled_indices) > 0:
        raise RuntimeError(
            f"pool_embeddings_at_boxes: {len(unfilled_indices)} box(es) were assigned "
            f"to a tile but the FPN hook did not fire for that tile "
            f"(case {case_id_str}, fold {fold}). "
            f"Unfilled box indices: {unfilled_indices[:10].tolist()}"
            f"{'...' if len(unfilled_indices) > 10 else ''}. "
            "This is a hard error to prevent silent zero embeddings propagating to H1/H2/H3."
        )

    logger.info(
        "pool_embeddings_at_boxes: fold=%d case %s — pooled %d embeddings (mode=%s)",
        fold,
        case_id_str,
        n_boxes,
        pooling,
    )
    return embeddings_out


# ---------------------------------------------------------------------------
# restore_boxes_for_case  (D01.18 — original-grid bbox helper)
# ---------------------------------------------------------------------------


def restore_boxes_for_case(
    boxes_preprocessed: np.ndarray,
    case_id: int,
    preprocessed_dir: str,
    fold_dir: str,
) -> np.ndarray:
    """Convert boxes from preprocessed space to original-image-grid space (D01.18).

    Applies nnDetection's ``restore_detection`` using per-case properties from the
    preprocessed dir and ``transpose_backward`` from the fold's plan_inference.pkl.

    This is the restore step that was always part of the D01.17 design but was
    omitted from the implementation (D01.18 design confirmation). The candidate
    BBox in ``RawCandidate`` must be in original-image-grid space because
    STORY_01_03 IoU-matches candidates against original-grid GT lesion bboxes
    (thesis §3.2.2/§3.2.9, D01.18 ground 2). The slot→axis map in
    ``_raw_detections_to_candidates`` (x=d0, y=d1, z=d2) is correct for
    RESTORED boxes; do NOT pass preprocessed/storage-frame boxes to that function.

    Two-frame D01.17 design:
      pool_embeddings_at_boxes ← boxes_preprocessed  (storage frame, same as FPN maps)
      _raw_detections_to_candidates ← restore_boxes_for_case(boxes_preprocessed, ...)
                                      (original-grid frame, paired 1:1 by box index)

    Parameters
    ----------
    boxes_preprocessed : np.ndarray
        Shape (N, 6) float32. Native boxes from ``predict_oof(restore=False)`` in
        preprocessed/storage space, axis (x1,y1,x2,y2,z1,z2).
    case_id : int
        Integer case ID; converted to f"{case_id:04d}" .pkl stem internally.
    preprocessed_dir : str
        Path to the preprocessed imagesTr or imagesTs directory.
        The per-case ``<case_id_str>.pkl`` properties file lives here.
    fold_dir : str
        Path to the fold's training output. Contains ``plan_inference.pkl``
        (source of ``transpose_backward``).

    Returns
    -------
    np.ndarray
        Shape (N, 6) float32. Boxes in original-image-grid space.
        Axis order: (x1,y1,x2,y2,z1,z2) — same nnDetection convention (detection.py:263).
        After restoring, x=d0, y=d1, z=d2 in the original NRRD storage grid.

    Raises
    ------
    ImportError
        If nnDetection / omegaconf not available (laptop).
    FileNotFoundError
        If ``plan_inference.pkl`` or per-case .pkl is missing.
    """
    # Lazy imports — nnDetection is server-only.
    try:
        from nndet.inference.restore import restore_detection
        from nndet.io.load import load_pickle
    except ImportError as exc:
        raise ImportError(
            "nnDetection is not installed in the current environment. "
            "restore_boxes_for_case requires the nnDetection conda env (server-only). "
            f"Original error: {exc}"
        ) from exc

    if boxes_preprocessed.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    case_id_str = f"{case_id:04d}"
    preprocessed_path = Path(preprocessed_dir)

    # Load plan to get transpose_backward
    plan_path = Path(fold_dir) / "plan_inference.pkl"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"plan_inference.pkl not found at {plan_path}. "
            "Required for restore_boxes_for_case (provides transpose_backward)."
        )
    plan = load_pickle(plan_path)
    transpose_backward = plan["transpose_backward"]

    # Load per-case properties (written by nndet_prep, lives alongside the .npz)
    props_path = preprocessed_path / f"{case_id_str}.pkl"
    if not props_path.exists():
        raise FileNotFoundError(
            f"Per-case properties not found: {props_path}. "
            "This file is written by nndet_prep alongside the .npz file."
        )
    props = load_pickle(props_path)

    boxes_original = restore_detection(
        boxes=boxes_preprocessed,
        transpose_backward=transpose_backward,
        original_spacing=props["original_spacing"],
        spacing_after_resampling=props["spacing_after_resampling"],
        crop_bbox=props["crop_bbox"],
    )
    return np.asarray(boxes_original, dtype=np.float32)


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
    # Import lazily to avoid circular imports at module load time.
    # (This module is imported by generate_candidates.py which imports from ensemble.py.)
    from abus.detect.candidates import RawCandidate
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

    # WBC across all tiles via the shared helper (eliminates duplication and keeps
    # iou_threshold consistent between batch and streaming paths).
    # _ensemble_proposals_to_arrays is defined later in this module; Python resolves
    # function references at call time so forward definition is fine.
    return _ensemble_proposals_to_arrays(all_proposals, iou_threshold=iou_threshold)


# ---------------------------------------------------------------------------
# _ensemble_proposals_to_arrays  (D01.14-OOM — WBC + array packing)
# ---------------------------------------------------------------------------


def _ensemble_proposals_to_arrays(
    all_proposals: list,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WBC ensemble a list of RawCandidate proposals and return (boxes, scores, embeddings).

    D01.14-OOM streaming fix: this is the second half of what
    ``_predict_single_case_with_embeddings`` does (WBC + array packing).
    Factored out so the streaming path in ``predict_with_embeddings`` can pass
    already-pooled proposals directly without re-materialising per-tile feature
    maps.

    Parameters
    ----------
    all_proposals : list[RawCandidate]
        Proposals for one case, already pooled at tile-LOCAL centroids and
        shifted to case-space (produced by repeated calls to
        ``_pool_tile_proposals``).
    iou_threshold : float
        IoU threshold for WBC. Default 0.5 (provisional; STORY_01_03 wires a
        calibrated project-wide threshold).

    Returns
    -------
    boxes : np.ndarray  shape (N, 6) float32, case-preprocessed-space
    scores : np.ndarray shape (N,) float32
    embeddings : np.ndarray shape (N, D_EMB) float32
    """
    from abus.detect.ensemble import ensemble_combine

    if not all_proposals:
        empty = np.zeros((0, 6), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros((0, D_EMB), dtype=np.float32)

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
# _pool_tile_proposals  (D01.14-OOM — CPU-safe streaming helper)
# ---------------------------------------------------------------------------


def _pool_tile_proposals(
    feat_map_raw: object,
    tile_boxes: np.ndarray,
    tile_scores: np.ndarray,
    tile_origin: tuple,
    fpn_level_index: int = 0,
    tile_idx: int = 0,
) -> list:
    """Pool embeddings for one tile and return a list of RawCandidate proposals.

    D01.14-OOM streaming fix: called immediately after each tile's forward pass so
    the feature map can be released before the next tile runs.  Only the tiny
    per-tile (bbox, score, embedding) records accumulate across tiles.

    Semantics are IDENTICAL to the per-tile loop body inside
    ``_predict_single_case_with_embeddings`` — this is a factored-out version of
    that loop body so it can be called in the streaming path AND so the existing
    ``_predict_single_case_with_embeddings`` interface (used by unit tests) is
    preserved unchanged.

    Parameters
    ----------
    feat_map_raw : list[np.ndarray] | np.ndarray
        Raw feature map from the decoder hook for this tile.
        If a list/tuple, index ``fpn_level_index`` selects the FPN level.
    tile_boxes : np.ndarray
        Per-tile detected boxes in tile-LOCAL pixel space, shape (N, 6) float32,
        axis (x1, y1, x2, y2, z1, z2).
    tile_scores : np.ndarray
        Per-tile detection scores, shape (N,) float32.
    tile_origin : tuple
        Tile origin in nnDetection (x, y, z) = (d2, d1, d0) order.
    fpn_level_index : int
        FPN level to pool from (default 0 = finest).
    tile_idx : int
        Tile index for logging only.

    Returns
    -------
    list[RawCandidate]
        Zero or more proposals for this tile, with embeddings pooled at
        tile-LOCAL centroids and boxes shifted to case-space by tile_origin.
    """
    from abus.detect.candidates import RawCandidate
    from abus.geometry.bbox import BBox

    if tile_boxes.shape[0] == 0:
        return []

    # Extract the FPN level requested.
    if isinstance(feat_map_raw, (list, tuple)):  # noqa: UP038 (py38: no X|Y in isinstance)
        if fpn_level_index >= len(feat_map_raw):
            logger.warning(
                "_pool_tile_proposals: tile %d: fpn_level_index=%d out of range "
                "(%d levels); skipping tile embeddings",
                tile_idx,
                fpn_level_index,
                len(feat_map_raw),
            )
            return []
        feat_map: np.ndarray = np.asarray(feat_map_raw[fpn_level_index])
    else:
        feat_map = np.asarray(feat_map_raw)

    # Remove batch dim if present: (B, C, D0, D1, D2) → (C, D0, D1, D2)
    if feat_map.ndim == 5:
        feat_map = feat_map[0]

    # tile_origin in nnDetection (x, y, z) order = (d2, d1, d0).
    # _apply_offsets_to_boxes convention (detection.py:242-249):
    #   boxes[:,0] += offset[0]   (x1 → d2_min)
    #   boxes[:,1] += offset[1]   (y1 → d1_min)
    #   boxes[:,2] += offset[0]   (x2 → d2_max)
    #   boxes[:,3] += offset[1]   (y2 → d1_max)
    #   boxes[:,4] += offset[2]   (z1 → d0_min)
    #   boxes[:,5] += offset[2]   (z2 → d0_max)
    x_offset = float(tile_origin[0])  # d2
    y_offset = float(tile_origin[1])  # d1
    z_offset = float(tile_origin[2])  # d0

    proposals = []
    for i in range(tile_boxes.shape[0]):
        box_tile = tile_boxes[i]  # tile-local (x1, y1, x2, y2, z1, z2)

        # Pool at tile-LOCAL centroid — CORRECT: feat_map and box are both in tile space.
        cx_x = (box_tile[0] + box_tile[2]) / 2.0  # d2 in tile space
        cx_y = (box_tile[1] + box_tile[3]) / 2.0  # d1 in tile space
        cx_z = (box_tile[4] + box_tile[5]) / 2.0  # d0 in tile space

        emb = point_pool_trilinear(feat_map, cx_d0=cx_z, cx_d1=cx_y, cx_d2=cx_x)

        # Apply tile_origin offset → case-space box.
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

        proposals.append(
            RawCandidate(
                case_id=0,  # placeholder — caller replaces
                split="ens_tmp",
                bbox=bbox,
                score=float(tile_scores[i]),
                embedding=emb,
                source_detectors=(0,),  # placeholder — caller replaces
            )
        )

    return proposals


# ---------------------------------------------------------------------------
# predict_with_embeddings  (RETIRED — D01.17; use pool_embeddings_at_boxes instead)
# ---------------------------------------------------------------------------


def predict_with_embeddings(
    fold: int,
    case_ids: list[int],
    preprocessed_dir: str,
    fold_dir: str,
    fpn_level_index: int = 0,
    num_models: int = 1,
) -> dict[int, RawDetectionsWithEmb]:
    """[RETIRED D01.17] Per-tile embedding-extraction loop — do NOT use in new code.

    This function was RETIRED by D01.17 (decision log, 2026-06-22) after three
    production failures:
      1. CPU execution: model not moved to GPU (predict_dir uses Predictor to do that;
         this custom loop did not).
      2. 64 GB OOM: all tiles' FPN feature maps accumulated before pooling.
      3. O(N²) WBC hang: per-tile detection cap exploded N at score_thresh=0.0.

    D01.17 REPLACEMENT: use the decoupled design instead:
      boxes = predict_oof(fold, case_ids, preprocessed_dir, fold_dir, restore=False)
      for case_id in case_ids:
          emb = pool_embeddings_at_boxes(fold, case_id, boxes[case_id].boxes,
                                         preprocessed_dir, fold_dir, pooling="centroid")

    This function is KEPT (not deleted) for reference and to preserve existing unit
    tests. It is NOT called by generate_candidates.py or any production path.

    *** Do NOT add new calls to this function. ***

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

    # --- Hook setup (D01.14-OOM streaming fix) ---
    # The hook now captures ONLY the level-0 (finest) FPN feature map immediately
    # as a CPU numpy array, discarding all other levels.  This eliminates the
    # multi-level tensor accumulation that was the dominant source of host RAM
    # usage (levels 1-3 are never used downstream; only level 0 is pooled).
    #
    # _tile_feat_maps holds at most ONE entry at a time: the level-0 array from
    # the most recent forward call.  It is cleared immediately after pooling so
    # the memory is released before the next tile runs.
    _tile_feat_maps: list[np.ndarray] = []  # at most one entry between clears

    def _decoder_hook(module: object, input: object, output: object) -> None:  # noqa: A002
        """Capture only the level-0 FPN feature map per tile forward call.

        D01.14-OOM fix: retain only the finest level (index 0) as a numpy array;
        discard levels 1-3 immediately so their GPU/CPU tensors can be freed.
        """
        if isinstance(output, (list, tuple)):  # noqa: UP038 (py38: no X|Y in isinstance)
            if len(output) == 0:
                return
            t = output[fpn_level_index] if fpn_level_index < len(output) else output[0]
            arr = np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
        else:
            arr = np.asarray(output.detach().cpu().numpy() if hasattr(output, "detach") else output)
        _tile_feat_maps.append(arr)

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

            # D01.14-OOM streaming fix: do NOT materialise all tiles up front.
            # Instead, iterate crops one at a time, run inference, pool immediately,
            # and release the feature map before the next tile.
            #
            # Only the tiny per-tile RawCandidate proposals (bbox + score + 128-D
            # embedding) accumulate across tiles.  A typical tile's level-0 feature
            # map is ~(128, ~48, ~48, ~48) float32 ≈ 170 MB; ~300 tiles × 170 MB =
            # ~50 GB.  With streaming, peak host RAM from feature maps is ~170 MB
            # (one tile at a time).

            import torch  # already available (lazy-imported above via _torch)  # noqa: PLC0415

            model_device = (
                next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
            )

            # Accumulates only small (bbox, score, embedding) records — O(N_dets).
            all_proposals_case: list = []

            with torch.no_grad():
                for tile_idx, crop in enumerate(tiles_crops):
                    # Load one tile (one crop of the volume) — released at end of loop body.
                    try:
                        tile_data, tile_origin, _tile_crop = save_get_crop(
                            case_arr, crop, mode="shift"
                        )
                    except RuntimeError:
                        tile_data, tile_origin, _tile_crop = save_get_crop(
                            case_arr, crop, mode="symmetric"
                        )

                    tile_data_tensor = torch.from_numpy(
                        tile_data[None].astype(np.float32)  # add batch dim
                    ).to(model_device)

                    # inference_step returns dict with pred_boxes/pred_scores/pred_labels
                    # in tile-LOCAL pixel space (NOT yet offset by tile_origin).
                    result_tile = model.inference_step(tile_data_tensor, batch_num=0)

                    # The hook fired during inference_step.  Grab the captured level-0
                    # feature map (already numpy on CPU), then clear the buffer immediately
                    # so the array can be freed after pooling.
                    if _tile_feat_maps:
                        feat_map_raw: object = _tile_feat_maps[-1]
                        _tile_feat_maps.clear()
                    else:
                        # Hook did not fire (decoder not found) — empty placeholder.
                        feat_map_raw = []

                    # On the first tile of the first case, log the captured map shape
                    # and model device so server-side Step 13 audits can verify the
                    # hook produced a real (C, D0, D1, D2) array (not (B,C,D0,D1,D2)
                    # which would need batch-dim stripping) and that the model is on GPU.
                    if tile_idx == 0 and case_id_str == case_ids_str[0]:
                        fm_shape = getattr(feat_map_raw, "shape", "no-hook")
                        logger.debug(
                            "predict_with_embeddings: tile 0 debug — "
                            "feat_map_raw.shape=%s model_device=%s",
                            fm_shape,
                            model_device,
                        )

                    # Extract per-tile boxes/scores (list with one element for batch_size=1).
                    tile_boxes_list = result_tile.get("pred_boxes", [])
                    tile_scores_list = result_tile.get("pred_scores", [])

                    if tile_boxes_list and len(tile_boxes_list) > 0:
                        b = tile_boxes_list[0]
                        s = tile_scores_list[0]
                        b_np = b.detach().cpu().numpy() if hasattr(b, "detach") else np.array(b)
                        s_np = s.detach().cpu().numpy() if hasattr(s, "detach") else np.array(s)
                        tile_boxes = b_np.astype(np.float32)
                        tile_scores = s_np.astype(np.float32)
                    else:
                        tile_boxes = np.zeros((0, 6), dtype=np.float32)
                        tile_scores = np.zeros(0, dtype=np.float32)

                    # Pool per-tile embeddings immediately at tile-LOCAL centroids, then
                    # shift boxes to case space — feat_map_raw is no longer needed after this.
                    # The hook already captured only level 0, so fpn_level_index=0 applies to
                    # the single-level array directly (the helper handles list or array input).
                    tile_proposals = _pool_tile_proposals(
                        feat_map_raw=feat_map_raw,
                        tile_boxes=tile_boxes,
                        tile_scores=tile_scores,
                        tile_origin=tile_origin,
                        fpn_level_index=0,  # hook already selected level 0
                        tile_idx=tile_idx,
                    )
                    all_proposals_case.extend(tile_proposals)

                    # Release references so GC can reclaim: tile raw data + feature map.
                    del tile_data, tile_data_tensor, feat_map_raw, result_tile

            # WBC ensemble across all tiles (tiny proposals only — no large arrays here).
            boxes_case, scores_case, embeddings_case = _ensemble_proposals_to_arrays(
                all_proposals_case
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
