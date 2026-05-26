#!/usr/bin/env python
"""Server CLI: OOF + ensemble candidate generation (STORY_01_02).

Usage
-----
Generate all candidates (OOF train + ensemble val + ensemble test):
    python scripts/generate_candidates.py \\
        --nndet-dataset-root $NNDET_PREPROCESSED \\
        --checkpoints-root /home/maia-user/Andre/checkpoints \\
        --out-dir /home/maia-user/Andre/candidates/raw \\
        --nndet-commit <hash>

Generate only OOF candidates (train split):
    python scripts/generate_candidates.py ... --splits train

Generate only ensemble candidates (val + test splits):
    python scripts/generate_candidates.py ... --splits val test

After generation, run the provenance check on the laptop:
    python -m abus.detect.candidates --provenance-check <out-dir>

And the lap-top variant:
    pytest tests/test_candidates.py

Required environment variables (or pass via CLI):
    NNDET_PREPROCESSED   — nnDetection preprocessed Task directory
    DET_MODELS           — path for nnDetection training artefacts (checkpoints)

GPU note: this script calls nnDetection inference (GPU required). It is NOT run
locally. The nnDetection inference CLI must be available in the active conda env.

Leakage control (thesis §3.2.1, §3.2.6):
  - OOF: fold-k detector is applied ONLY to oof_ids(k). The generate_oof_candidates
    leakage guard raises if a train_ids case is passed (in-code enforcement).
  - Ensemble: all five fold detectors are applied to val/test cases independently,
    then ensemble_combine (union → WBC → score/embedding averaging) is applied.
    No 6th all-data detector is used (provenance_check enforces this post-hoc).

Serialisation:
  Each RawCandidateSet is saved as:
    <out_dir>/<split>_candidates.npz   — arrays
    <out_dir>/<split>_candidates.json  — metadata

After generation, the provenance_check is run automatically and the summary
table is printed (split | n_volumes | total_candidates | mean/vol | mean_score).
The embedding_variance_gap is also printed (OOF vs val set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src/ to path for local import when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abus.data.split import load_split
from abus.detect.candidates import (
    RawCandidate,
    RawCandidateSet,
    embedding_variance_gap,
    generate_ensemble_candidates,
    generate_oof_candidates,
    provenance_check,
)

# ---------------------------------------------------------------------------
# nnDetection inference wrapper
# ---------------------------------------------------------------------------


def _make_oof_inference_fn(
    fold: int,
    ckpt_path: str,
    nndet_dataset_root: str,
    nndet_commit: str,
):
    """Return an inference callable for OOF generation.

    On the server this wraps the nnDetection inference CLI (nndet predict).
    The output is parsed to build RawCandidate objects.

    The callable signature is: (case_ids: list[int]) -> list[RawCandidate].
    """

    def inference_fn(case_ids: list[int]) -> list[RawCandidate]:
        """Run nnDetection inference for fold ``fold`` on ``case_ids``."""
        import subprocess
        import tempfile

        # Write a temporary case-list file for nnDetection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for cid in case_ids:
                f.write(f"{cid:04d}\n")
            case_list_path = f.name

        try:
            # nnDetection predict CLI:
            # python -m nndet.entrypoints.predict <task> RetinaUNet000 --fold <fold>
            # --checkpoint <ckpt> --output <out_dir> --case-ids <case_list>
            with tempfile.TemporaryDirectory() as out_dir:
                cmd = [
                    "python",
                    "-m",
                    "nndet.entrypoints.predict",
                    "Task001_TDSCABUS",
                    "RetinaUNet000",
                    "--fold",
                    str(fold),
                    "--checkpoint",
                    ckpt_path,
                    "--output",
                    out_dir,
                    "--case-ids",
                    case_list_path,
                ]
                result = subprocess.run(
                    cmd,
                    cwd=nndet_dataset_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"nnDetection inference failed for fold {fold}.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"Return code: {result.returncode}\n"
                        f"stderr:\n{result.stderr[-3000:]}\n"
                    )

                # Parse nnDetection inference output files into RawCandidate objects.
                # nnDetection writes per-case box+score+embedding files.
                candidates = _parse_nndet_predictions(
                    out_dir, case_ids, split="train", fold=fold, nndet_commit=nndet_commit
                )
        finally:
            os.unlink(case_list_path)

        return candidates

    return inference_fn


def _make_ensemble_inference_fn(
    fold: int,
    ckpt_path: str,
    nndet_dataset_root: str,
    nndet_commit: str,
    split: str,
):
    """Return an inference callable for ensemble generation (one fold at a time)."""

    def inference_fn(fold_id: int, case_ids: list[int]) -> list[RawCandidate]:
        """Run nnDetection inference for one fold on all val/test cases."""
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as out_dir:
            cmd = [
                "python",
                "-m",
                "nndet.entrypoints.predict",
                "Task001_TDSCABUS",
                "RetinaUNet000",
                "--fold",
                str(fold_id),
                "--checkpoint",
                ckpt_path,
                "--output",
                out_dir,
                "--split",
                split,
            ]
            result = subprocess.run(
                cmd,
                cwd=nndet_dataset_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"nnDetection inference failed for fold {fold_id}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Return code: {result.returncode}\n"
                    f"stderr:\n{result.stderr[-3000:]}\n"
                )

            candidates = _parse_nndet_predictions(
                out_dir, case_ids, split=split, fold=fold_id, nndet_commit=nndet_commit
            )
        return candidates

    return inference_fn


def _parse_nndet_predictions(
    out_dir: str,
    case_ids: list[int],
    split: str,
    fold: int,
    nndet_commit: str,
) -> list[RawCandidate]:
    """Parse nnDetection prediction output files into RawCandidate objects.

    nnDetection writes per-case JSON or pickle files with boxes, scores, and
    (optionally) backbone embeddings. This function reads them and constructs
    RawCandidate objects.

    The exact output format depends on the nnDetection version. This implementation
    reads the standard nnDetection v0.1 prediction output format:
      <out_dir>/<case_id_4d>_boxes.npy   — (N, 6) float array (y1,x1,y2,x2,z1,z2)
      <out_dir>/<case_id_4d>_scores.npy  — (N,) float array
      <out_dir>/<case_id_4d>_emb.npy     — (N, D) float32 array (optional)

    If embedding files are absent, a zero vector of shape (1,) is used as a
    placeholder (embedding extraction may be a separate post-processing step).
    """
    import numpy as np

    from abus.geometry.convert import nndet_to_bbox

    candidates: list[RawCandidate] = []
    out_path = Path(out_dir)

    # Discover case ids from output files if case_ids is empty
    if not case_ids:
        box_files = sorted(out_path.glob("*_boxes.npy"))
        file_case_ids = []
        for bf in box_files:
            stem = bf.stem.replace("_boxes", "")
            try:
                file_case_ids.append(int(stem))
            except ValueError:
                pass
        case_ids = file_case_ids

    target_spacing = (0.3, 0.3, 0.5)  # D01.7 target spacing

    for case_id in case_ids:
        boxes_file = out_path / f"{case_id:04d}_boxes.npy"
        scores_file = out_path / f"{case_id:04d}_scores.npy"
        emb_file = out_path / f"{case_id:04d}_emb.npy"

        if not boxes_file.exists() or not scores_file.exists():
            # No predictions for this case — skip (valid: no detections)
            continue

        boxes = np.load(str(boxes_file))  # (N, 6) — nndet (y1,x1,y2,x2,z1,z2) resampled
        scores = np.load(str(scores_file))  # (N,)

        if emb_file.exists():
            embs = np.load(str(emb_file)).astype(np.float32)  # (N, D)
        else:
            # Placeholder embedding (D=1) until embedding extraction is implemented
            embs = np.zeros((len(scores), 1), dtype=np.float32)

        for i in range(len(scores)):
            # Convert resampled-grid nndet box back to original-grid project BBox.
            # nndet box: (y1, x1, y2, x2, z1, z2) exclusive max, resampled grid.
            # Step 1: convert to original grid via the spacing ratio.
            y1, x1, y2, x2, z1, z2 = (int(v) for v in boxes[i])
            # Scale back to original grid (resampled -> original)
            # orig_coord = resamp_coord * (target_spacing / orig_spacing)
            from abus.io.loader import CANONICAL_SPACING_MM as ORIG

            def _back(coord: int, axis: int) -> int:
                return round(coord * target_spacing[axis] / ORIG[axis])

            # d0->y, d1->x, d2->z
            oy1 = _back(y1, 0)
            ox1 = _back(x1, 1)
            oy2 = _back(y2, 0)
            ox2 = _back(x2, 1)
            oz1 = _back(z1, 2)
            oz2 = _back(z2, 2)

            # Ensure valid inclusive-max BBox (exclusive -> inclusive for nndet_to_bbox)
            oy2 = max(oy2, oy1 + 1)
            ox2 = max(ox2, ox1 + 1)
            oz2 = max(oz2, oz1 + 1)

            bbox = nndet_to_bbox((oy1, ox1, oy2, ox2, oz1, oz2))

            candidates.append(
                RawCandidate(
                    case_id=case_id,
                    split=split,
                    bbox=bbox,
                    score=float(scores[i]),
                    embedding=embs[i].astype(np.float32),
                    source_detectors=(fold,),
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary_table(
    split: str,
    cset: RawCandidateSet,
) -> None:
    """Print the candidate summary table for one split."""
    import numpy as np

    n_vols = len({c.case_id for c in cset.candidates})
    total = len(cset.candidates)
    mean_per_vol = total / n_vols if n_vols > 0 else 0.0
    mean_score = float(np.mean([c.score for c in cset.candidates])) if cset.candidates else 0.0
    # source_detector check
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
        description="Generate OOF + ensemble raw candidates (STORY_01_02).",
    )
    parser.add_argument(
        "--nndet-dataset-root",
        default=os.environ.get("NNDET_PREPROCESSED", ""),
        help="nnDetection preprocessed Task directory (env: NNDET_PREPROCESSED).",
    )
    parser.add_argument(
        "--checkpoints-root",
        required=True,
        help="Root directory containing fold checkpoints: <root>/fold_<k>/model_best.ckpt",
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
        default="unknown",
        help="nnDetection commit hash (for reproducibility).",
    )
    parser.add_argument(
        "--provisional-ensemble-iou",
        type=float,
        default=0.5,
        help=(
            "Provisional WBC IoU for val/test ensemble combination (default 0.5). "
            "Superseded by STORY_01_03 calibrated params; this value is only for "
            "an intermediate inspection."
        ),
    )
    args = parser.parse_args()

    if not args.nndet_dataset_root:
        parser.error("--nndet-dataset-root or $NNDET_PREPROCESSED must be set.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the frozen 5-fold split for provenance
    split = load_split()
    fold_of: dict[int, int] = split.fold_of

    # Build checkpoint paths
    ckpt_root = Path(args.checkpoints_root)
    ckpt_paths: dict[int, str] = {}
    for fold_id in range(5):
        ckpt = ckpt_root / f"fold_{fold_id}" / "model_best.ckpt"
        if not ckpt.exists():
            # Try alternative layout
            ckpt = (
                ckpt_root / "Task001_TDSCABUS" / "RetinaUNet000" / str(fold_id) / "model_best.ckpt"
            )
        if not ckpt.exists():
            print(f"WARNING: checkpoint for fold {fold_id} not found at {ckpt}")
        else:
            ckpt_paths[fold_id] = str(ckpt)

    candidate_sets: dict[str, RawCandidateSet] = {}

    # --- OOF generation (train split) ---
    if "train" in args.splits:
        print("\nGenerating OOF candidates (train split)...")
        all_oof: list[RawCandidate] = []
        for fold_id in range(5):
            if fold_id not in ckpt_paths:
                print(f"  Skipping fold {fold_id} (checkpoint not found).")
                continue
            inference_fn = _make_oof_inference_fn(
                fold=fold_id,
                ckpt_path=ckpt_paths[fold_id],
                nndet_dataset_root=args.nndet_dataset_root,
                nndet_commit=args.nndet_commit,
            )
            fold_candidates = generate_oof_candidates(
                fold=fold_id,
                detector_ckpt=ckpt_paths[fold_id],
                nndet_dataset_root=args.nndet_dataset_root,
                inference_fn=inference_fn,
            )
            print(f"  Fold {fold_id}: {len(fold_candidates)} OOF candidates.")
            all_oof.extend(fold_candidates)

        train_set = RawCandidateSet(
            split="train",
            candidates=all_oof,
            detector_commit=args.nndet_commit,
            _fold_of=dict(fold_of),
        )
        path = str(out_dir / "train_candidates")
        train_set.save(path)
        print(f"Saved train candidates: {path}.npz + .json")
        candidate_sets["train"] = train_set

    # --- Ensemble generation (val + test splits) ---
    for spl in ("val", "test"):
        if spl not in args.splits:
            continue
        print(f"\nGenerating ensemble candidates ({spl} split)...")

        def ensemble_inference_fn(
            fold_id: int, case_ids: list[int], _spl: str = spl
        ) -> list[RawCandidate]:
            if fold_id not in ckpt_paths:
                return []
            fn = _make_ensemble_inference_fn(
                fold=fold_id,
                ckpt_path=ckpt_paths[fold_id],
                nndet_dataset_root=args.nndet_dataset_root,
                nndet_commit=args.nndet_commit,
                split=_spl,
            )
            return fn(fold_id, case_ids)

        ensemble_candidates = generate_ensemble_candidates(
            split=spl,
            detector_ckpts=ckpt_paths,
            nndet_dataset_root=args.nndet_dataset_root,
            inference_fn=ensemble_inference_fn,
        )
        ens_set = RawCandidateSet(
            split=spl,
            candidates=ensemble_candidates,
            detector_commit=args.nndet_commit,
            _fold_of={},  # val/test cases have no fold membership
        )
        path = str(out_dir / f"{spl}_candidates")
        ens_set.save(path)
        print(f"Saved {spl} candidates: {path}.npz + .json")
        candidate_sets[spl] = ens_set

    # --- Provenance check ---
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

    # --- Candidate summary table ---
    print("\n--- Candidate summary ---")
    print(f"  {'split':5s} | {'vols':>4s} | {'cands':>6s} | mean/vol | mean_score | src_folds")
    for spl, cset in candidate_sets.items():
        _print_summary_table(spl, cset)

    # --- Embedding variance gap (OOF vs val) ---
    if "train" in candidate_sets and "val" in candidate_sets:
        if candidate_sets["train"].candidates and candidate_sets["val"].candidates:
            print("\n--- OOF-vs-ensemble embedding-variance gap (D01.6) ---")
            gap = embedding_variance_gap(candidate_sets["train"], candidate_sets["val"])
            print(f"  pooled_mean_ratio (OOF/ensemble): {gap['pooled_mean_ratio']:.4f}")
            print(f"  per_dim_oof_var mean:      {gap['per_dim_oof_var'].mean():.6f}")
            print(f"  per_dim_ensemble_var mean: {gap['per_dim_ensemble_var'].mean():.6f}")

            # Save gap report
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
