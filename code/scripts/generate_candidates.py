#!/usr/bin/env python
"""Server CLI: OOF + ensemble candidate generation (STORY_01_02, D01.17).

D01.17 redesign (DECOUPLED): RETIRED the hand-rolled predict_with_embeddings per-tile
loop that caused three production failures (CPU execution, 64 GB OOM, O(N²) WBC hang).
REPLACED with a two-step DECOUPLED design:

  Step 1 — NATIVE BOXES: predict_oof(fold, case_ids, preprocessed_dir, fold_dir,
                                     restore=False) via nnDetection's untouched predict_dir.
            → boxes + scores in PREPROCESSED space; embeddings=None. ≤100/case.
            → recall/provenance/Gate A owned by nnDetection.

  Step 2 — POST-HOC EMBEDDING: pool_embeddings_at_boxes(fold, case_id,
                                boxes_preprocessed, preprocessed_dir, fold_dir,
                                pooling=<operator>) via torch.nn.functional.grid_sample.
            → (N, 128) float32; row i ↔ native box i. FPN level pinned to
              decoder_levels[0]. RAISES RuntimeError if decoder not found (no silent zeros).

Inference paths:

  OOF path (--splits train):
    For each fold k in 0..4:
      1. predict_oof(k, oof_ids(k), imagesTr, fold_dir_k, restore=False) → native boxes.
      2. pool_embeddings_at_boxes(k, case_id, those_boxes, imagesTr, fold_dir_k,
                                   pooling=<op>) → (N, 128).
    The leakage guard in generate_oof_candidates fires BEFORE inference.

  Ensemble path (--splits val test):
    For each fold k in 0..4 and for each val/test case:
      1. predict_oof(k, val_test_ids, imagesTs, fold_dir_k, restore=False)
      2. pool_embeddings_at_boxes(k, case_id, those_boxes, imagesTs, fold_dir_k,
                                   pooling=<op>)
    Then ensemble_combine(per_case_proposals) across folds:
      union → IoU-cluster → score+embedding-weighted average.
    source_detectors for each candidate = set of fold ids in its cluster (1..5).

  Run both --pooling centroid AND --pooling roi_align separately before Job 3d
  (the discriminativeness audit, STORY_01_04), which selects the operator.

Pooling operator governance (D01.17 frozen selection rule):
  Run embedding_discriminativeness for BOTH operators on the OOF set (BEFORE
  any H1/H2/H3 evaluation). Select:
    roi_align if roi_align.embedding_auc > centroid.embedding_auc + 0.0
    centroid  otherwise (ties → centroid = thesis §3.2.3 default)
  The selected operator is written to embedding_extraction.yaml and frozen.
  Do NOT select the operator based on H1/H2/H3 outcomes.

nnDetection 0.1 surface used (commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
  nndet_train <TASK> -o exp.fold=N --sweep   (training; see train_all_folds.py)
  predict_dir(case_ids=..., restore=False)   (Python helper; called by predict_oof)
  load_final_model(identifier="last")        (SWA model_last — pre-registered, D01.13)

Env-var contract (lowercase — nnDetection reads these via os.environ):
  det_data   — nnDetection dataset root (e.g. /home/maia-user/nndet_data)
  det_models — nnDetection model output root (same value for our task)
  Both MUST be exported before running this script.

Leakage control (thesis §3.2.1, §3.2.6):
  OOF: fold-k detector is applied ONLY to oof_ids(k). The generate_oof_candidates
    leakage guard raises ProvenanceError if a train_ids case is passed (ASC-01_02.7).
  Ensemble: all five fold detectors applied to val/test cases independently,
    then combined. No 6th all-data detector (provenance_check enforces this).
    source_detectors is REAL per-cluster fold provenance (1..5 contributing folds).

Serialisation:
  <out_dir>/<split>_candidates.npz + <split>_candidates.json
  (per operator; re-run with --pooling centroid and --pooling roi_align separately)

After generation, provenance_check is run and the summary table + embedding
variance gap are printed. Pass --provenance-check <out_dir> to re-run the
check on existing files (laptop-safe, no GPU needed).

GPU note: this script calls nnDetection inference (GPU required). NOT run locally.
The nnDetection conda env must be active and det_data/det_models env vars set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add src/ to path for local import when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collections.abc import Callable

from abus.data.split import load_split
from abus.detect.candidates import (
    RawCandidate,
    RawCandidateSet,
    embedding_variance_gap,
    generate_oof_candidates,
    provenance_check,
)
from abus.detect.ensemble import ensemble_combine
from abus.detect.nndet_inference import (
    D_EMB,
    RawDetections,
    RawDetectionsWithEmb,
    pool_embeddings_at_boxes,
    predict_oof,
    preprocess_val_test,
    restore_boxes_for_case,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAME = "Task001_TDSCABUS"
EXP_ID = "RetinaUNetV001_D3V001_3d"
NNDET_COMMIT = "97a58f3110b71caf1b4bcc1851e67cf11e987fc5"


# ---------------------------------------------------------------------------
# OOF inference function (D01.17: decoupled predict_oof + pool_embeddings_at_boxes)
# ---------------------------------------------------------------------------


def _make_oof_inference_fn(
    fold: int,
    preprocessed_dir: str,
    fold_dir: str,
    pooling: str = "centroid",
) -> Callable[[list[int]], list[RawCandidate]]:
    """Return an inference callable for OOF generation using the D01.17 decoupled design.

    D01.17: uses predict_oof(restore=False) for native boxes then
    pool_embeddings_at_boxes for post-hoc embeddings. The two steps are
    decoupled: native inference is nnDetection-untouched (≤100/case, recall-maximal),
    embedding pooling is post-hoc via torch.nn.functional.grid_sample.

    The callable signature: (case_ids: list[int]) -> list[RawCandidate].

    Parameters
    ----------
    fold : int
        Fold index (0–4).
    preprocessed_dir : str
        Path to preprocessed imagesTr directory (contains .npz case files).
        e.g. ``$det_data/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTr``
    fold_dir : str
        Path to fold's training output (contains config.yaml + plan_inference.pkl +
        model_last.ckpt).
        e.g. ``$det_models/Task001_TDSCABUS/RetinaUNetV001_D3V001_3d/fold<N>``
    pooling : str
        Pooling operator: "centroid" (default) or "roi_align" (audit-selected).
    """

    def inference_fn(case_ids: list[int]) -> list[RawCandidate]:
        """D01.17/D01.18: native boxes → post-hoc embeddings → restore → RawCandidate list.

        Two-frame design (D01.18):
          boxes_preprocessed → pool_embeddings_at_boxes (storage frame = FPN frame)
          boxes_preprocessed → restore_boxes_for_case   (original-grid frame for BBox)
        Both frames come from the SAME predict_oof(restore=False) call; pairing is 1:1 by row.
        """
        # Step 1: NATIVE BOXES — untouched nnDetection, ≤100/case, preprocessed space
        raw_dets = predict_oof(
            fold=fold,
            case_ids=case_ids,
            preprocessed_dir=preprocessed_dir,
            fold_dir=fold_dir,
            restore=False,  # D01.17: boxes in preprocessed space (= FPN space)
        )

        candidates: list[RawCandidate] = []
        for case_id, rd in raw_dets.items():
            # Step 2: POST-HOC EMBEDDING — pool at each native box (storage frame)
            emb = pool_embeddings_at_boxes(
                fold=fold,
                case_id=case_id,
                boxes_preprocessed=rd.boxes,
                preprocessed_dir=preprocessed_dir,
                fold_dir=fold_dir,
                pooling=pooling,
            )

            # Step 3: RESTORE — convert storage-frame boxes to original-grid frame (D01.18).
            # _raw_detections_to_candidates expects RESTORED boxes (slot0→d0, slot4→d2).
            # The embedding (step 2) was pooled from storage-frame boxes and does not
            # change — embeddings are opaque 128-D vectors, not spatial coordinates.
            boxes_original = restore_boxes_for_case(
                boxes_preprocessed=rd.boxes,
                case_id=case_id,
                preprocessed_dir=preprocessed_dir,
                fold_dir=fold_dir,
            )

            # Attach embeddings to RESTORED boxes
            rd_with_emb = RawDetectionsWithEmb(
                case_id=case_id,
                boxes=boxes_original,  # D01.18: original-grid frame for BBox
                scores=rd.scores,
                embeddings=emb,
            )
            candidates.extend(
                _raw_detections_to_candidates(rd_with_emb, split="train", source_detectors=(fold,))
            )
        return candidates

    return inference_fn


# ---------------------------------------------------------------------------
# Ensemble inference functions (nndet_consolidate + nndet_predict -f -1)
# ---------------------------------------------------------------------------


def _run_consolidate(task_name: str, exp_id: str) -> None:
    """Run nndet_consolidate to create the 5-fold ensemble metadata.

    D01.13: --sweep_boxes is required (consolidate.py:130-132).
    Without it, nndet_consolidate raises ValueError("Export needs new parameter sweep!")
    because it consumes each fold's sweep_predictions/ dir (produced by --sweep).

    CLI: nndet_consolidate <TASK> <MODEL> --sweep_boxes
    """
    cmd = ["nndet_consolidate", task_name, exp_id, "--sweep_boxes"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nndet_consolidate failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
        )
    print("  nndet_consolidate complete.")


def _run_ensemble_predict(task_name: str, exp_id: str) -> None:
    """Run nndet_predict with -f -1 (all-folds ensemble on imagesTs/).

    CLI: nndet_predict <TASK> <MODEL> -f -1
    Output lands at $det_models/<TASK>/<EXP_ID>/test_predictions/
    """
    cmd = ["nndet_predict", task_name, exp_id, "-f", "-1"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nndet_predict -f -1 failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
        )
    print("  nndet_predict (ensemble) complete.")


# ---------------------------------------------------------------------------
# Convert RawDetections → RawCandidate list
# ---------------------------------------------------------------------------


def _raw_detections_to_candidates(
    rd: RawDetections | RawDetectionsWithEmb,
    split: str,
    source_detectors: tuple[int, ...],
) -> list[RawCandidate]:
    """Convert a RawDetectionsWithEmb record to a list of RawCandidate objects.

    D01.14: embeddings must be REAL (N, D_EMB) float32 arrays — not None.
    Passing a RawDetections with embeddings=None raises ValueError (the
    zero-placeholder of D01.13 is retired; the candidate stage now always
    uses predict_with_embeddings which produces real embeddings).

    Box axis (D01.18 two-frame design): rd.boxes MUST be in ORIGINAL-IMAGE-GRID
    space (produced by ``restore_boxes_for_case``).

    After restore_detection, the column format is ``(x1, y1, x2, y2, z1, z2)``
    where x/y/z refer to the ORIGINAL storage axes: x=d0, y=d1, z=d2.
    This is nnDetection's documented output convention (detection.py:263).
    The slot→axis mapping is therefore: x1=slot0→d0, y1=slot1→d1, z1=slot4→d2.

    Do NOT pass storage-frame (preprocessed-space) boxes — those are for
    ``pool_embeddings_at_boxes``. STORY_01_03 IoU-matches the BBox against
    original-grid GT lesion bboxes (thesis §3.2.2/§3.2.9).
    """
    import numpy as np

    from abus.geometry.bbox import BBox

    # D01.14/D01.17: embeddings must not be None — the zero-placeholder is retired.
    if rd.embeddings is None:
        raise ValueError(
            f"_raw_detections_to_candidates: rd.embeddings is None for case_id={rd.case_id}. "
            "D01.17 requires real (N, D_EMB) embeddings from pool_embeddings_at_boxes. "
            "Do not pass a bare RawDetections (embeddings=None) to this function — "
            "use predict_oof then pool_embeddings_at_boxes which returns a RawDetectionsWithEmb "
            "with real embeddings (D01.17 decoupled design)."
        )

    candidates: list[RawCandidate] = []

    if rd.boxes.shape[0] == 0:
        return candidates

    for i in range(rd.boxes.shape[0]):
        # D01.18: boxes are RESTORED — output of restore_detection.
        # nnDetection documents restore_detection output as (x1,y1,x2,y2,z1,z2)
        # in ORIGINAL image space (detection.py:263).  After restore, x/y/z map
        # back to original storage axes: x=d0, y=d1, z=d2.
        # Source: restore.py:54 permute_boxes(boxes, transpose_backward) undoes
        # transpose_forward, so original axis 0 → slot 0 (x), axis 1 → slot 1 (y),
        # axis 2 → slot 4 (z).
        box = rd.boxes[i]
        x1, y1, x2, y2, z1, z2 = (int(round(float(v))) for v in box)

        # Guard: skip degenerate boxes
        if x2 < x1 or y2 < y1 or z2 < z1:
            logger.warning(
                "case %d: degenerate box [%d,%d,%d,%d,%d,%d] skipped",
                rd.case_id,
                x1,
                y1,
                x2,
                y2,
                z1,
                z2,
            )
            continue

        # Map restored (x1,y1,x2,y2,z1,z2) → project BBox
        # x=d0, y=d1, z=d2  (original storage-axis order, post-restore)
        bbox = BBox(
            min_d0=x1,
            min_d1=y1,
            min_d2=z1,
            max_d0=max(x2, x1 + 1),
            max_d1=max(y2, y1 + 1),
            max_d2=max(z2, z1 + 1),
        )

        emb = rd.embeddings[i].astype(np.float32)
        # Assert D_EMB shape — catch a dead hook or wrong-level pooling early.
        if emb.shape != (D_EMB,):
            raise ValueError(
                f"Embedding shape mismatch for case_id={rd.case_id} detection {i}: "
                f"expected ({D_EMB},), got {emb.shape}. "
                "Check that predict_with_embeddings is using fpn_channels=128 level."
            )

        candidates.append(
            RawCandidate(
                case_id=rd.case_id,
                split=split,
                bbox=bbox,
                score=float(rd.scores[i]),
                embedding=emb,
                source_detectors=source_detectors,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Ensemble path: branch (a) — RETIRED (D01.14)
# ---------------------------------------------------------------------------
# Branch (a) (nndet_consolidate --sweep_boxes + nndet_predict -f -1) is
# ABANDONED for candidate generation (D01.14 decision 4). The CLI path
# cannot hook the model to extract FPN embeddings. The nndet_consolidate
# step may still be run independently for a separate idiomatic-mAP sanity
# number, but it is NOT the candidate-gen source.
#
# _run_consolidate and _run_ensemble_predict are retained below for reference
# (the runbook may call them for the sanity number) but _generate_ensemble_branch_a
# is removed — there is no caller for it in the D01.14 pipeline.


# ---------------------------------------------------------------------------
# Ensemble path: branch (b) fallback — per-fold predict_oof + ensemble_combine
# ---------------------------------------------------------------------------


def _generate_ensemble_with_embeddings(
    det_models_root: str,
    task_name: str,
    exp_id: str,
    split_case_ids: dict[str, list[int]],
    preprocessed_ts_dir: str,
    nndet_commit: str,
    pooling: str = "centroid",
) -> dict[str, RawCandidateSet]:
    """D01.17 DECOUPLED per-fold ensemble path.

    For each of the 5 fold detectors and each val/test case:
      1. Native boxes: predict_oof(fold, case_ids, imagesTs, fold_dir, restore=False)
         — untouched nnDetection, ≤100/case, preprocessed space.
      2. Post-hoc embedding: pool_embeddings_at_boxes(fold, case_id, those_boxes,
         imagesTs, fold_dir, pooling=<op>) → (N, 128) float32.
      3. Convert to RawCandidate list with source_detectors=(fold,).
      4. Collect proposals per case across all 5 folds.
    Then for each case:
      5. ensemble_combine(per_case_proposals) → union + IoU-cluster +
         score+embedding-weighted average. source_detectors = set of fold ids
         from each cluster (1..5, genuine).

    D01.17 replaces the retired predict_with_embeddings per-tile loop that caused
    three production failures. The decoupled design eliminates:
      - CPU execution (predict_oof delegates to nnDetection's GPU-aware predictor)
      - OOM (no tiling needed for embedding pooling — one forward per case, not tile)
      - O(N²) WBC hang (predict_oof uses ≤100/case, not per-tile thousands)

    D01.14b: val/test preprocessed files are in imagesTs (NOT imagesTr).
    imagesTr contains only the 100 training cases (0000-0099). The val/test cases
    (0100-0199) must be preprocessed separately via preprocess_val_test into
    preprocessed/<data_identifier>/imagesTs/ before this function is called.

    Parameters
    ----------
    preprocessed_ts_dir : str
        Path to the preprocessed imagesTs directory containing .npz files for
        val/test cases. Populated by preprocess_val_test before this call.
        e.g. ``$det_data/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTs``
    pooling : str
        Pooling operator: "centroid" (default) or "roi_align". Audit-selected.
    """
    all_splits = list(split_case_ids.keys())
    all_case_ids = sorted({cid for ids in split_case_ids.values() for cid in ids})

    # D01.14b guard: imagesTs must exist before ensemble
    ts_path = Path(preprocessed_ts_dir)
    if not ts_path.exists():
        raise FileNotFoundError(
            f"Preprocessed imagesTs directory not found: {preprocessed_ts_dir}\n"
            "D01.14b: val/test cases must be preprocessed into imagesTs before "
            "running the ensemble step. Call preprocess_val_test() first.\n"
            "The OOF/train path uses imagesTr (training cases only); val/test "
            "cases (0100+) require a separate imagesTs preprocessing step."
        )

    proposals_by_case: dict[int, list[RawCandidate]] = {cid: [] for cid in all_case_ids}

    for fold in range(5):
        fold_dir = str(Path(det_models_root) / task_name / exp_id / f"fold{fold}")
        print(
            f"  Fold {fold}: predict_oof (native boxes) for "
            f"{len(all_case_ids)} val/test cases ..."
        )

        # Step 1: NATIVE BOXES — untouched nnDetection, preprocessed space
        raw_dets = predict_oof(
            fold=fold,
            case_ids=all_case_ids,
            preprocessed_dir=preprocessed_ts_dir,
            fold_dir=fold_dir,
            restore=False,  # D01.17: boxes in preprocessed space
        )

        print(
            f"  Fold {fold}: pool_embeddings_at_boxes ({pooling}) for " f"{len(raw_dets)} cases ..."
        )

        for case_id, rd in raw_dets.items():
            # Step 2: POST-HOC EMBEDDING — pool at each native box (storage frame)
            emb = pool_embeddings_at_boxes(
                fold=fold,
                case_id=case_id,
                boxes_preprocessed=rd.boxes,
                preprocessed_dir=preprocessed_ts_dir,
                fold_dir=fold_dir,
                pooling=pooling,
            )

            # Step 3: RESTORE — convert storage-frame boxes to original-grid frame (D01.18).
            # _raw_detections_to_candidates expects RESTORED boxes (slot0→d0, slot4→d2)
            # so the BBox is in original-grid space (required by STORY_01_03 GT matching).
            boxes_original = restore_boxes_for_case(
                boxes_preprocessed=rd.boxes,
                case_id=case_id,
                preprocessed_dir=preprocessed_ts_dir,
                fold_dir=fold_dir,
            )

            # Attach embeddings to RESTORED boxes (D01.18: original-grid frame for BBox)
            rd_with_emb = RawDetectionsWithEmb(
                case_id=case_id,
                boxes=boxes_original,  # D01.18: original-grid frame
                scores=rd.scores,
                embeddings=emb,
            )

            # Per-fold source_detectors = (fold,); ensemble_combine will merge them
            fold_candidates = _raw_detections_to_candidates(
                rd_with_emb, split="ens_tmp", source_detectors=(fold,)
            )
            proposals_by_case.setdefault(case_id, []).extend(fold_candidates)

    result: dict[str, RawCandidateSet] = {}

    for spl in all_splits:
        case_ids_for_split = set(split_case_ids[spl])
        combined: list[RawCandidate] = []

        for case_id in sorted(case_ids_for_split):
            props = proposals_by_case.get(case_id, [])
            if not props:
                continue
            # Re-tag with the final split label
            retagged = [
                RawCandidate(
                    case_id=p.case_id,
                    split=spl,
                    bbox=p.bbox,
                    score=p.score,
                    embedding=p.embedding,
                    source_detectors=p.source_detectors,
                )
                for p in props
            ]
            # ensemble_combine: union → IoU-cluster → score+embedding weighted avg
            # source_detectors for each cluster = set of fold ids from its members.
            combined.extend(ensemble_combine(retagged, iou_threshold=0.5))

        result[spl] = RawCandidateSet(
            split=spl,
            candidates=combined,
            detector_commit=nndet_commit,
            _fold_of={},
        )

    return result


# ---------------------------------------------------------------------------
# Summary table + helpers
# ---------------------------------------------------------------------------


def _print_summary_table(split: str, cset: RawCandidateSet) -> None:
    """Print the candidate summary table for one split (D01.14: includes embedding info)."""
    import numpy as np

    n_vols = len({c.case_id for c in cset.candidates})
    total = len(cset.candidates)
    mean_per_vol = total / n_vols if n_vols > 0 else 0.0
    mean_score = float(np.mean([c.score for c in cset.candidates])) if cset.candidates else 0.0
    unique_src = {s for c in cset.candidates for s in c.source_detectors}

    # D01.14 Step 17: report embedding_dim and mean_embedding_l2_norm
    emb_dim = cset.candidates[0].embedding.shape[0] if cset.candidates else 0
    mean_emb_l2 = (
        float(np.mean([float(np.linalg.norm(c.embedding)) for c in cset.candidates]))
        if cset.candidates
        else 0.0
    )

    print(
        f"  {split:5s} | vols={n_vols:4d} | cands={total:6d} | "
        f"mean/vol={mean_per_vol:6.1f} | mean_score={mean_score:.3f} | "
        f"src_folds={sorted(unique_src)} | emb_dim={emb_dim} | mean_emb_l2={mean_emb_l2:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OOF + ensemble raw candidates (STORY_01_02, D01.13).",
    )
    parser.add_argument(
        "--det-data",
        default=os.environ.get("det_data", ""),
        help=(
            "nnDetection dataset root (env: det_data — lowercase). "
            "Example: /home/maia-user/nndet_data"
        ),
    )
    parser.add_argument(
        "--det-models",
        default=os.environ.get("det_models", ""),
        help=(
            "nnDetection model output root (env: det_models — lowercase). "
            "For our task: same as det_data."
        ),
    )
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help=(
            "Path to preprocessed imagesTr directory for OOF (train split) inference. "
            "D01.13: predict_with_embeddings reads preprocessed .npz files, not raw images. "
            "Default: <det_data>/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTr. "
            "D01.14b: val/test inference uses imagesTs (derived from <det_data>; no override arg). "
            "imagesTs is populated by preprocess_val_test before the ensemble step."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for serialised RawCandidateSet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which splits to generate (default: all three).",
    )
    parser.add_argument(
        "--nndet-commit",
        default=NNDET_COMMIT,
        help="nnDetection commit hash (for reproducibility).",
    )
    parser.add_argument(
        "--pooling",
        choices=["centroid", "roi_align"],
        default="centroid",
        help=(
            "D01.17: pooling operator for post-hoc embedding extraction. "
            "'centroid' = single trilinear sample at box centroid (thesis §3.2.3 default). "
            "'roi_align' = 3D RoIAlign of box extent → 1x1x1 (§3.2.3 contingency). "
            "Run BOTH operators before the discriminativeness audit (Job 3d, STORY_01_04). "
            "Do NOT select based on H1/H2/H3 outcomes (D01.17 researcher-DoF firewall)."
        ),
    )
    # --ensemble-branch retired; kept for backward-compat of existing call-sites (no-op)
    parser.add_argument(
        "--ensemble-branch",
        choices=["embedding"],
        default="embedding",
        help=(
            "D01.17 (formerly D01.14): only 'embedding' path supported. "
            "Retained as a no-op for backward-compat."
        ),
    )
    parser.add_argument(
        "--provenance-check",
        metavar="RAW_CANDIDATES_DIR",
        help="Re-run provenance_check on existing candidate files in the given directory.",
    )
    parser.add_argument(
        "--num-processes-preprocessing",
        type=int,
        default=0,
        help=(
            "D01.14b: number of parallel preprocessing workers for preprocess_val_test. "
            "0 = sequential (safe default). On the server, use 3 (matching det_num_threads). "
            "Only affects the val/test preprocessing step (Step 14a); OOF inference is unaffected."
        ),
    )
    args = parser.parse_args()

    # -- Provenance-check-only mode (laptop-safe)
    if args.provenance_check:
        from abus.detect.candidates import _cli_provenance_check

        sys.exit(_cli_provenance_check(args.provenance_check))

    if not args.det_data:
        parser.error(
            "--det-data or $det_data must be set (lowercase env var; "
            "nnDetection silently ignores the uppercase DET_DATA)."
        )
    if not args.det_models:
        parser.error("--det-models or $det_models must be set (lowercase env var).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOF/train inference reads preprocessed .npz files from imagesTr.
    # imagesTr contains the 100 training cases (0000-0099) preprocessed by nndet_prep.
    preprocessed_dir = args.preprocessed_dir or str(
        Path(args.det_data) / TASK_NAME / "preprocessed" / "D3V001_3d" / "imagesTr"
    )
    print(f"Preprocessed dir (OOF/train source, imagesTr): {preprocessed_dir}")

    # D01.14b: val/test ensemble reads from imagesTs (NOT imagesTr).
    # imagesTr contains ONLY training cases (0000-0099); val/test cases (0100-0199)
    # are NOT there. They must be preprocessed into imagesTs first via preprocess_val_test.
    preprocessed_ts_dir = str(
        Path(args.det_data) / TASK_NAME / "preprocessed" / "D3V001_3d" / "imagesTs"
    )
    print(f"Preprocessed dir (val/test source, imagesTs): {preprocessed_ts_dir}")

    # Load the frozen 5-fold split for provenance and case-id discovery
    split_manifest = load_split()

    # Build the case-id map for each split
    # train = all 100 training cases (0..99 by convention)
    train_ids = sorted(split_manifest.fold_of.keys())
    # val/test are the 30 + 70 cases in raw_splitted/imagesTs/
    # We discover them from the manifest if available; otherwise use known ranges.
    # STORY_01_01 confirmed val = 100..129, test = 130..199.
    val_ids = list(range(100, 130))
    test_ids = list(range(130, 200))

    split_case_ids: dict[str, list[int]] = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    candidate_sets: dict[str, RawCandidateSet] = {}

    # -----------------------------------------------------------------------
    # OOF generation (train split)
    # -----------------------------------------------------------------------
    if "train" in args.splits:
        print("\n=== OOF candidate generation (train split) ===")
        all_oof: list[RawCandidate] = []

        for fold_id in range(5):
            fold_dir = str(Path(args.det_models) / TASK_NAME / EXP_ID / f"fold{fold_id}")
            print(f"\nFold {fold_id}: {fold_dir}")

            oof_case_ids = split_manifest.oof_ids(fold_id)
            print(f"  OOF case_ids: {len(oof_case_ids)} cases")

            # The leakage guard in generate_oof_candidates fires BEFORE inference
            # (ASC-01_02.7). It raises ProvenanceError if any case_ids are
            # in train_ids(fold).
            # D01.17: decoupled design — predict_oof(restore=False) + pool_embeddings_at_boxes
            inference_fn = _make_oof_inference_fn(
                fold=fold_id,
                preprocessed_dir=preprocessed_dir,
                fold_dir=fold_dir,
                pooling=args.pooling,
            )

            fold_candidates = generate_oof_candidates(
                fold=fold_id,
                detector_ckpt=fold_dir,  # fold_dir contains model_last.ckpt (D01.13)
                nndet_dataset_root=args.det_data,
                inference_fn=inference_fn,
                split_override=split_manifest,
            )
            print(f"  Fold {fold_id}: {len(fold_candidates)} OOF candidates.")
            all_oof.extend(fold_candidates)

        train_set = RawCandidateSet(
            split="train",
            candidates=all_oof,
            detector_commit=args.nndet_commit,
            _fold_of=dict(split_manifest.fold_of),
        )
        path = str(out_dir / "train_candidates")
        train_set.save(path)
        print(f"\nSaved train candidates: {path}.npz + .json")
        candidate_sets["train"] = train_set

    # -----------------------------------------------------------------------
    # Ensemble generation (val + test splits) — D01.17 decoupled path
    # -----------------------------------------------------------------------
    ens_splits = [s for s in ("val", "test") if s in args.splits]
    if ens_splits:
        print("\n=== Ensemble candidate generation (val + test splits) ===")
        print(
            f"D01.17: decoupled predict_oof(restore=False) + "
            f"pool_embeddings_at_boxes({args.pooling})"
        )
        print("        (real 128-D embeddings; genuine 1..5 source_detectors)")
        print("D01.14b: preprocessing val/test cases into imagesTs first ...")

        # D01.14b: preprocess val/test raw images into imagesTs BEFORE ensemble.
        # Use fold0's plan_inference.pkl (all folds share the same planner + plan
        # since they all trained on the same task with the same data identifier).
        fold0_dir = str(Path(args.det_models) / TASK_NAME / EXP_ID / "fold0")
        preprocess_val_test(
            fold_dir=fold0_dir,
            num_processes=args.num_processes_preprocessing,
        )
        print(f"D01.14b: val/test preprocessing complete → {preprocessed_ts_dir}")

        requested_split_ids = {spl: split_case_ids[spl] for spl in ens_splits}

        ens_sets = _generate_ensemble_with_embeddings(
            det_models_root=args.det_models,
            task_name=TASK_NAME,
            exp_id=EXP_ID,
            split_case_ids=requested_split_ids,
            preprocessed_ts_dir=preprocessed_ts_dir,
            nndet_commit=args.nndet_commit,
            pooling=args.pooling,
        )
        print("D01.14b ensemble generation complete.")

        for spl, cset in ens_sets.items():
            path = str(out_dir / f"{spl}_candidates")
            cset.save(path)
            print(f"Saved {spl} candidates: {path}.npz + .json")
            candidate_sets[spl] = cset

    # -----------------------------------------------------------------------
    # Provenance check
    # -----------------------------------------------------------------------
    print("\n--- Provenance check ---")
    all_ok = True
    for spl, cset in candidate_sets.items():
        try:
            result = provenance_check(cset)
            print(f"  {spl}: PROVENANCE OK ({result['n_checked']} candidates checked)")
        except Exception as e:
            print(f"  {spl}: PROVENANCE FAIL — {e}")
            all_ok = False

    if all_ok:
        print("PROVENANCE OK")

    # -----------------------------------------------------------------------
    # Candidate summary table
    # -----------------------------------------------------------------------
    print("\n--- Candidate summary ---")
    print(f"  {'split':5s} | {'vols':>4s} | {'cands':>6s} | mean/vol | mean_score | src_folds")
    for spl, cset in candidate_sets.items():
        _print_summary_table(spl, cset)

    # -----------------------------------------------------------------------
    # Embedding variance gap (OOF vs val — D01.6 diagnostic)
    # -----------------------------------------------------------------------
    if "train" in candidate_sets and "val" in candidate_sets:
        if candidate_sets["train"].candidates and candidate_sets["val"].candidates:
            print("\n--- OOF-vs-ensemble embedding-variance gap (D01.6) ---")
            gap = embedding_variance_gap(candidate_sets["train"], candidate_sets["val"])
            print(f"  pooled_mean_ratio (OOF/ensemble): {gap['pooled_mean_ratio']:.4f}")
            print(f"  per_dim_oof_var mean:      {gap['per_dim_oof_var'].mean():.6f}")
            print(f"  per_dim_ensemble_var mean: {gap['per_dim_ensemble_var'].mean():.6f}")

            gap_path = out_dir / "embedding_variance_gap.json"
            gap_report = {
                "pooled_mean_ratio": float(gap["pooled_mean_ratio"]),
                "per_dim_oof_var_mean": float(gap["per_dim_oof_var"].mean()),
                "per_dim_ensemble_var_mean": float(gap["per_dim_ensemble_var"].mean()),
            }
            with open(str(gap_path), "w", encoding="utf-8") as f:
                json.dump(gap_report, f, indent=2)
                f.write("\n")
            print(f"  Gap report saved: {gap_path}")

    if not all_ok:
        sys.exit(1)

    print("\nCandidate generation complete.")


if __name__ == "__main__":
    main()
