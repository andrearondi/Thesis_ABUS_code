#!/usr/bin/env python
"""Server CLI: OOF + ensemble candidate generation (STORY_01_02, D01.9).

Drives two separate nnDetection inference paths:

  OOF path (--splits train):
    For each fold k in 0..4, calls predict_oof(k, oof_ids(k), ...) which
    internally invokes nndet.inference.helper.predict_dir(case_ids=...).
    This is the ONLY nnDetection call that bypasses the CLI — per-fold OOF
    over a specified case_ids list has no CLI surface in nnDetection 0.1.
    The leakage guard in generate_oof_candidates fires BEFORE inference.

  Ensemble path (--splits val test):
    Step 1: nndet_consolidate Task001_TDSCABUS RetinaUNetV001_D3V001_3d
    Step 2: nndet_predict Task001_TDSCABUS RetinaUNetV001_D3V001_3d -f -1
      (consolidated 5-fold ensemble on raw_splitted/imagesTs/)
    Step 3: parse_predictions_dir on the consolidated test_predictions/ dir.
    Step 4 (FALLBACK): if consolidated metadata is insufficient for
      source_detectors, fall back to per-fold predict_oof for all val/test
      cases + ensemble_combine.

nnDetection 0.1 CLI surface (commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
  nndet_train <TASK> -o exp.fold=N            (training; see train_all_folds.py)
  nndet_consolidate <TASK> <MODEL>            (consolidate 5 folds)
  nndet_predict <TASK> <MODEL> -f -1          (ensemble on imagesTs/)
  nndet_predict <TASK> <MODEL> -f N           (single-fold on imagesTs/)

Env-var contract (lowercase — nnDetection reads these via os.environ):
  det_data   — nnDetection dataset root (e.g. /home/maia-user/nndet_data)
  det_models — nnDetection model output root (same value for our task)
  Both MUST be exported before running this script.

Leakage control (thesis §3.2.1, §3.2.6):
  OOF: fold-k detector is applied ONLY to oof_ids(k). The generate_oof_candidates
    leakage guard raises ProvenanceError if a train_ids case is passed (ASC-01_02.7).
  Ensemble: all five fold detectors applied to val/test cases independently,
    then combined. No 6th all-data detector (provenance_check enforces this).

Serialisation:
  <out_dir>/<split>_candidates.npz + <split>_candidates.json

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
from abus.detect.nndet_inference import RawDetections, parse_predictions_dir, predict_oof

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAME = "Task001_TDSCABUS"
EXP_ID = "RetinaUNetV001_D3V001_3d"
NNDET_COMMIT = "97a58f3110b71caf1b4bcc1851e67cf11e987fc5"


# ---------------------------------------------------------------------------
# OOF inference function (per fold via predict_oof Python helper)
# ---------------------------------------------------------------------------


def _make_oof_inference_fn(
    fold: int,
    preprocessed_dir: str,
    fold_dir: str,
) -> Callable[[list[int]], list[RawCandidate]]:
    """Return an inference callable for OOF generation using predict_oof.

    The callable signature: (case_ids: list[int]) -> list[RawCandidate].
    Internally calls predict_oof(fold, case_ids, preprocessed_dir, fold_dir)
    which uses nndet.inference.helper.predict_dir(case_ids=...) with the real
    nnDetection 0.1 signature (D01.13).

    Parameters
    ----------
    fold : int
        Fold index (0–4).
    preprocessed_dir : str
        Path to preprocessed imagesTr directory (contains .npz case files).
        e.g. ``$det_data/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTr``
    fold_dir : str
        Path to fold's training output (contains config.yaml + plan_inference.pkl).
        e.g. ``$det_models/Task001_TDSCABUS/RetinaUNetV001_D3V001_3d/fold<N>``
    """

    def inference_fn(case_ids: list[int]) -> list[RawCandidate]:
        """Run fold-``fold`` detector on ``case_ids`` via predict_oof (D01.13)."""
        # predict_oof returns dict[int, RawDetections] keyed by case_id
        raw_dets = predict_oof(
            fold=fold,
            case_ids=case_ids,
            preprocessed_dir=preprocessed_dir,
            fold_dir=fold_dir,
        )

        candidates: list[RawCandidate] = []
        for _case_id, rd in raw_dets.items():
            candidates.extend(
                _raw_detections_to_candidates(rd, split="train", source_detectors=(fold,))
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
    rd: RawDetections,
    split: str,
    source_detectors: tuple[int, ...],
    embedding_dim: int = 1,
) -> list[RawCandidate]:
    """Convert a RawDetections record to a list of RawCandidate objects.

    Boxes are left in nnDetection's resampled-grid convention here. The
    grid→original conversion (using nndet_convention.py) is deferred to
    STORY_01_04 which wires the full feature-extraction pipeline.

    STORY_01_02's goal is to produce the raw candidate set with correct
    provenance; the exact box coordinates on the original grid are verified
    by STORY_01_01's round-trip test and STORY_01_04's bbox extraction.

    If rd.embeddings is None (not available in nnDetection 0.1's output),
    a zero-vector placeholder of shape (1,) is used until STORY_01_04 wires
    the backbone-pooled embedding extraction (story spec note on placeholder).
    """
    import numpy as np

    from abus.geometry.bbox import BBox

    candidates: list[RawCandidate] = []

    if rd.boxes.shape[0] == 0:
        return candidates

    for i in range(rd.boxes.shape[0]):
        # D01.13 confirmed box axis: (x1, y1, x2, y2, z1, z2)
        # Source: nndet/core/boxes/ops.py line 34, detection.py _apply_offsets_to_boxes
        # Project axis vocabulary (EPIC_00 C1): x↔d2, y↔d1, z↔d0
        box = rd.boxes[i]
        x1, y1, x2, y2, z1, z2 = (int(round(float(v))) for v in box)

        # Guard: skip degenerate boxes (inverted coords are not produced by nnDetection's
        # NMS but are checked defensively — a silent wrong BBox is worse than a skip).
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

        # Map (x1,y1,x2,y2,z1,z2) → project BBox (min_d0,min_d1,min_d2,max_d0,max_d1,max_d2)
        # z→d0, y→d1, x→d2
        bbox = BBox(
            min_d0=z1,
            min_d1=y1,
            min_d2=x1,
            max_d0=max(z2, z1 + 1),
            max_d1=max(y2, y1 + 1),
            max_d2=max(x2, x1 + 1),
        )

        if rd.embeddings is not None:
            emb = rd.embeddings[i].astype(np.float32)
        else:
            emb = np.zeros(embedding_dim, dtype=np.float32)

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
# Ensemble path: branch (a) — consolidated nndet_predict -f -1
# ---------------------------------------------------------------------------


def _generate_ensemble_branch_a(
    det_models_root: str,
    task_name: str,
    exp_id: str,
    split_case_ids: dict[str, list[int]],
    nndet_commit: str,
) -> dict[str, RawCandidateSet]:
    """Consolidated nndet_predict -f -1 path.

    Runs consolidate + ensemble predict, parses the output, assigns
    source_detectors from the consolidated output.

    NOTE: nnDetection 0.1's consolidated output may or may not carry per-cluster
    fold-contribution metadata. If it does, source_detectors is populated from
    that metadata. If not, the consolidated output treats all 5 folds as a unit
    and source_detectors is set to (0,1,2,3,4) for every candidate.

    [SERVER-SIDE AUDIT REQUIRED: verify whether nnDetection 0.1's consolidated
    test_predictions/<case>_boxes.pkl includes per-cluster fold-membership info.
    If it does, update the source_detectors assignment below accordingly.]
    """
    print("\nStep 1: nndet_consolidate ...")
    _run_consolidate(task_name, exp_id)

    print("\nStep 2: nndet_predict -f -1 (ensemble) ...")
    _run_ensemble_predict(task_name, exp_id)

    # Consolidated predictions land at:
    # $det_models/<task_name>/<exp_id>/test_predictions/
    pred_dir = Path(det_models_root) / task_name / exp_id / "test_predictions"
    print(f"\nStep 3: parsing predictions from {pred_dir} ...")
    raw_dets = parse_predictions_dir(str(pred_dir))
    print(f"  Parsed {len(raw_dets)} cases.")

    # Assign source_detectors.
    # [SERVER-SIDE AUDIT: if the consolidated output carries per-cluster fold
    # metadata, parse it here. For now, conservatively assign all 5 folds since
    # the consolidated ensemble uses all 5 fold detectors.]
    all_source_detectors: tuple[int, ...] = (0, 1, 2, 3, 4)

    result: dict[str, RawCandidateSet] = {}

    for spl in ("val", "test"):
        if spl not in split_case_ids:
            continue
        case_ids_for_split = set(split_case_ids[spl])
        candidates: list[RawCandidate] = []

        for case_id, rd in raw_dets.items():
            if case_id not in case_ids_for_split:
                continue
            # Re-use the shared converter; all_source_detectors = (0,1,2,3,4) for
            # the consolidated ensemble path (all 5 fold detectors contributed).
            candidates.extend(
                _raw_detections_to_candidates(rd, split=spl, source_detectors=all_source_detectors)
            )

        result[spl] = RawCandidateSet(
            split=spl,
            candidates=candidates,
            detector_commit=nndet_commit,
            _fold_of={},
        )

    return result


# ---------------------------------------------------------------------------
# Ensemble path: branch (b) fallback — per-fold predict_oof + ensemble_combine
# ---------------------------------------------------------------------------


def _generate_ensemble_branch_b(
    det_models_root: str,
    task_name: str,
    exp_id: str,
    split_case_ids: dict[str, list[int]],
    preprocessed_dir: str,
    nndet_commit: str,
) -> dict[str, RawCandidateSet]:
    """Per-fold fallback: five predict_oof calls + ensemble_combine.

    Used when branch (a) cannot assign per-cluster source_detectors reliably,
    OR when the consolidated output is unavailable.

    D01.13: uses real predict_dir signature; predict_oof takes preprocessed_dir
    and fold_dir (not task_dir + model_dir). Val/test preprocessed files are in
    imagesTr (all 200 cases were preprocessed into imagesTr for nndet_prep).
    """

    # Collect per-case proposals from all 5 fold detectors
    all_splits = list(split_case_ids.keys())
    all_case_ids = sorted({cid for ids in split_case_ids.values() for cid in ids})

    proposals_by_case: dict[int, list[RawCandidate]] = {cid: [] for cid in all_case_ids}

    for fold in range(5):
        fold_dir = str(Path(det_models_root) / task_name / exp_id / f"fold{fold}")
        print(f"  Fold {fold}: running predict_oof for {len(all_case_ids)} val/test cases ...")

        raw_dets = predict_oof(
            fold=fold,
            case_ids=all_case_ids,
            preprocessed_dir=preprocessed_dir,
            fold_dir=fold_dir,
        )

        for case_id, rd in raw_dets.items():
            fold_candidates = _raw_detections_to_candidates(
                rd, split="ens_tmp", source_detectors=(fold,)
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
            # Re-tag split
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
    """Print the candidate summary table for one split."""
    import numpy as np

    n_vols = len({c.case_id for c in cset.candidates})
    total = len(cset.candidates)
    mean_per_vol = total / n_vols if n_vols > 0 else 0.0
    mean_score = float(np.mean([c.score for c in cset.candidates])) if cset.candidates else 0.0
    unique_src = {s for c in cset.candidates for s in c.source_detectors}
    print(
        f"  {split:5s} | vols={n_vols:4d} | cands={total:6d} | "
        f"mean/vol={mean_per_vol:6.1f} | mean_score={mean_score:.3f} | "
        f"src_folds={sorted(unique_src)}"
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
            "Path to preprocessed imagesTr directory for OOF inference. "
            "D01.13: predict_dir reads preprocessed .npz files, not raw images. "
            "Default: <det_data>/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTr"
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
        "--ensemble-branch",
        choices=["a", "b", "auto"],
        default="auto",
        help=(
            "Ensemble path for val/test: "
            "'a' = consolidated nndet_predict -f -1 (preferred); "
            "'b' = per-fold fallback via predict_oof + ensemble_combine; "
            "'auto' = try branch (a), fall back to (b) on error (default)."
        ),
    )
    parser.add_argument(
        "--provenance-check",
        metavar="RAW_CANDIDATES_DIR",
        help="Re-run provenance_check on existing candidate files in the given directory.",
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

    # D01.13: OOF inference reads preprocessed .npz files, not raw images.
    preprocessed_dir = args.preprocessed_dir or str(
        Path(args.det_data) / TASK_NAME / "preprocessed" / "D3V001_3d" / "imagesTr"
    )
    print(f"Preprocessed dir (OOF source): {preprocessed_dir}")

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
            # D01.13: predict_oof now takes preprocessed_dir + fold_dir (real signature)
            inference_fn = _make_oof_inference_fn(
                fold=fold_id,
                preprocessed_dir=preprocessed_dir,
                fold_dir=fold_dir,
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
    # Ensemble generation (val + test splits)
    # -----------------------------------------------------------------------
    ens_splits = [s for s in ("val", "test") if s in args.splits]
    if ens_splits:
        print("\n=== Ensemble candidate generation (val + test splits) ===")

        requested_split_ids = {spl: split_case_ids[spl] for spl in ens_splits}

        if args.ensemble_branch in ("a", "auto"):
            try:
                ens_sets = _generate_ensemble_branch_a(
                    det_models_root=args.det_models,
                    task_name=TASK_NAME,
                    exp_id=EXP_ID,
                    split_case_ids=requested_split_ids,
                    nndet_commit=args.nndet_commit,
                )
                print("Ensemble branch (a): consolidated nndet_predict -f -1 complete.")
            except Exception as exc:
                if args.ensemble_branch == "auto":
                    print(
                        f"  WARNING: branch (a) failed ({exc}). "
                        "Falling back to branch (b): per-fold predict_oof + ensemble_combine."
                    )
                    ens_sets = _generate_ensemble_branch_b(
                        det_models_root=args.det_models,
                        task_name=TASK_NAME,
                        exp_id=EXP_ID,
                        split_case_ids=requested_split_ids,
                        preprocessed_dir=preprocessed_dir,
                        nndet_commit=args.nndet_commit,
                    )
                    print("Ensemble branch (b) fallback complete.")
                else:
                    raise
        else:
            # --ensemble-branch b
            ens_sets = _generate_ensemble_branch_b(
                det_models_root=args.det_models,
                task_name=TASK_NAME,
                exp_id=EXP_ID,
                split_case_ids=requested_split_ids,
                preprocessed_dir=preprocessed_dir,
                nndet_commit=args.nndet_commit,
            )
            print("Ensemble branch (b) complete.")

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
