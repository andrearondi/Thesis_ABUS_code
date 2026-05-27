"""RawCandidate — one raw detector proposal before NMS/tau/WBC (STORY_01_02).

This module implements:
  - RawCandidate: frozen dataclass for one raw detector proposal.
  - RawCandidateSet: serialisable container of candidates for one split.
  - generate_oof_candidates: apply fold-k detector to oof_ids(k) ONLY.
  - generate_ensemble_candidates: apply all 5 fold detectors + ensemble_combine.
  - provenance_check: assert leakage-control invariants.
  - embedding_variance_gap: OOF-vs-ensemble variance diagnostic (D01.6).
  - ProvenanceError: raised when a provenance invariant is violated.

Leakage-control invariants (thesis §3.2.1, §3.2.6):
  OOF (train split):
    Each candidate has exactly one source_detector.
    That detector's fold IS the candidate's case's fold
    (detector k trains on all folds EXCEPT k, scores fold-k cases).
    Equivalently: source_detector[0] == fold_of[case_id].
    The in-code guard in generate_oof_candidates raises immediately if asked to
    score a train-ids case.

  Ensemble (val/test split):
    Each candidate has 1..5 source_detectors, all in {0,1,2,3,4}.
    No 6th all-data detector (id >= 5 or a trained joint model).
    Scores and embeddings are averaged across contributing detectors.

Serialisation:
  RawCandidateSet.save(path): writes <path>.npz (embedding matrix, bbox matrix,
    scores, case_ids, source_detectors) and <path>.json (metadata + fold_of).
  RawCandidateSet.load(path): reconstructs the full object, round-trips exactly.

CLI:
  python -m abus.detect.candidates --provenance-check <raw_candidates_dir>
  Loads all RawCandidateSets in the directory, runs provenance_check on each,
  prints PROVENANCE OK or reports the failure, and prints the
  OOF-vs-ensemble embedding-variance gap.

GPU / nnDetection note:
  generate_oof_candidates and generate_ensemble_candidates call an ``inference_fn``
  callable injected at call time — on the server this wraps the nnDetection
  inference CLI; in tests it is a synthetic stub. The module itself is
  CPU-only and importable on the laptop.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from abus.data.split import FoldSplit, load_split
from abus.geometry.bbox import BBox

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProvenanceError(RuntimeError):
    """Raised when a RawCandidateSet violates a leakage-control invariant.

    Provenance invariants (thesis §3.2.1, §3.2.6):
      - OOF (train split): every candidate has source_detectors == (fold_of[case_id],).
        That is, detector k scored a case that is in fold k (its OOF fold),
        meaning the detector was trained on the OTHER four folds and never saw
        this case during training.
      - Ensemble (val/test split): every candidate has 1..5 source_detectors,
        all in {0,1,2,3,4}. A 6th detector (id >= 5) is forbidden.
    """


# ---------------------------------------------------------------------------
# RawCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawCandidate:
    """One raw detector proposal BEFORE NMS / tau / WBC.

    Boxes are in the project BBox convention on the ORIGINAL volume grid
    (storage-axis, voxel, inclusive-max), converted back from nnDetection's
    resampled grid via nndet_convention.py on the server.

    Attributes
    ----------
    case_id:
        Integer case identifier (TDSC-ABUS-2023 convention: 0–99 for train,
        100–129 for val, 130–199 for test).
    split:
        "train" | "val" | "test"
    bbox:
        Project BBox on the original (pre-resample) volume grid.
    score:
        Raw detector confidence in [0, 1].
    embedding:
        1-D float32 array — the Retina U-Net backbone-pooled feature.
    source_detectors:
        Tuple of fold ids that produced / contributed to this proposal.
        OOF: exactly one element (the fold detector that scored this case).
        Ensemble: 1..5 elements (the fold detectors whose proposals clustered
        into this candidate after WBC).
    """

    case_id: int
    split: str
    bbox: BBox
    score: float
    embedding: np.ndarray
    source_detectors: tuple[int, ...]


# ---------------------------------------------------------------------------
# RawCandidateSet
# ---------------------------------------------------------------------------


@dataclass
class RawCandidateSet:
    """All raw candidates for one split, plus provenance metadata.

    Serialised to disk via save() / loaded via load().
    The ``_fold_of`` mapping is carried for provenance checking on the laptop
    after server-generated sets are copied back.

    Attributes
    ----------
    split:
        "train" | "val" | "test"
    candidates:
        All raw candidates for this split (may be empty before inference).
    detector_commit:
        nnDetection version+commit used (reproducibility, agent_rules §12).
    _fold_of:
        {case_id: fold_index} mapping from the frozen manifest.
        For val/test splits, may be empty (no fold membership).
    """

    split: str
    candidates: list[RawCandidate]
    detector_commit: str
    _fold_of: dict[int, int]

    def save(self, path: str) -> None:
        """Serialise to <path>.npz (arrays) + <path>.json (metadata).

        The .npz contains:
          embeddings  — float32 (N, D)
          bboxes      — int32   (N, 6) layout: (min_d0,min_d1,min_d2,max_d0,max_d1,max_d2)
          scores      — float64 (N,)
          case_ids    — int32   (N,)
          splits_arr  — object array of str (N,) for per-candidate split labels
          n_src_det   — int32   (N,) number of source_detectors per candidate
          src_det_flat — int32  (sum(n_src_det),) flattened source_detectors

        The .json contains: split, detector_commit, fold_of, n_candidates.
        """
        n = len(self.candidates)
        if n > 0:
            emb_dim = self.candidates[0].embedding.shape[0]
            embeddings = np.zeros((n, emb_dim), dtype=np.float32)
            bboxes = np.zeros((n, 6), dtype=np.int32)
            scores = np.zeros(n, dtype=np.float64)
            case_ids_arr = np.zeros(n, dtype=np.int32)
            n_src_det = np.zeros(n, dtype=np.int32)
            src_det_list: list[int] = []
            splits_arr = np.empty(n, dtype=object)

            for i, c in enumerate(self.candidates):
                embeddings[i] = c.embedding.astype(np.float32)
                bboxes[i] = [
                    c.bbox.min_d0,
                    c.bbox.min_d1,
                    c.bbox.min_d2,
                    c.bbox.max_d0,
                    c.bbox.max_d1,
                    c.bbox.max_d2,
                ]
                scores[i] = c.score
                case_ids_arr[i] = c.case_id
                splits_arr[i] = c.split
                n_src_det[i] = len(c.source_detectors)
                src_det_list.extend(c.source_detectors)

            src_det_flat = np.array(src_det_list, dtype=np.int32)
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
            bboxes = np.zeros((0, 6), dtype=np.int32)
            scores = np.zeros(0, dtype=np.float64)
            case_ids_arr = np.zeros(0, dtype=np.int32)
            splits_arr = np.empty(0, dtype=object)
            n_src_det = np.zeros(0, dtype=np.int32)
            src_det_flat = np.zeros(0, dtype=np.int32)

        npz_path = path + ".npz"
        np.savez(
            npz_path,
            embeddings=embeddings,
            bboxes=bboxes,
            scores=scores,
            case_ids=case_ids_arr,
            splits_arr=splits_arr,
            n_src_det=n_src_det,
            src_det_flat=src_det_flat,
        )

        json_path = path + ".json"
        meta: dict[str, Any] = {
            "split": self.split,
            "detector_commit": self.detector_commit,
            "fold_of": {str(k): v for k, v in self._fold_of.items()},
            "n_candidates": n,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> RawCandidateSet:
        """Load from <path>.npz + <path>.json.

        Reconstructs every RawCandidate field exactly:
        bbox ints, score float, embedding float32, source_detectors tuple.
        """
        npz_path = path + ".npz"
        json_path = path + ".json"

        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        # allow_pickle=True required for the object-dtype splits_arr field.
        # The file is written by this project only (not untrusted input).
        data = np.load(npz_path, allow_pickle=True)
        n = int(meta["n_candidates"])

        fold_of: dict[int, int] = {int(k): int(v) for k, v in meta["fold_of"].items()}

        if n == 0:
            return cls(
                split=meta["split"],
                candidates=[],
                detector_commit=meta["detector_commit"],
                _fold_of=fold_of,
            )

        embeddings = data["embeddings"].astype(np.float32)
        bboxes = data["bboxes"].astype(np.int32)
        scores = data["scores"]
        case_ids_arr = data["case_ids"].astype(np.int32)
        splits_arr = data["splits_arr"]
        n_src_det = data["n_src_det"].astype(np.int32)
        src_det_flat = data["src_det_flat"].astype(np.int32)

        # Reconstruct ragged source_detectors
        src_det_offset = 0
        candidates: list[RawCandidate] = []
        for i in range(n):
            k = int(n_src_det[i])
            src_dets = tuple(int(x) for x in src_det_flat[src_det_offset : src_det_offset + k])
            src_det_offset += k

            b = bboxes[i]
            bbox = BBox(int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5]))
            split_str = str(splits_arr[i]) if splits_arr[i] is not None else meta["split"]

            candidates.append(
                RawCandidate(
                    case_id=int(case_ids_arr[i]),
                    split=split_str,
                    bbox=bbox,
                    score=float(scores[i]),
                    embedding=embeddings[i].astype(np.float32),
                    source_detectors=src_dets,
                )
            )

        return cls(
            split=meta["split"],
            candidates=candidates,
            detector_commit=meta["detector_commit"],
            _fold_of=fold_of,
        )


# ---------------------------------------------------------------------------
# load_raw_candidates
# ---------------------------------------------------------------------------


def load_raw_candidates(split: str, candidates_dir: str) -> RawCandidateSet:
    """Load the serialised RawCandidateSet for ``split`` from ``candidates_dir``.

    Expects files named ``<split>_candidates.npz`` and ``<split>_candidates.json``
    in ``candidates_dir``.

    Parameters
    ----------
    split:
        "train" | "val" | "test"
    candidates_dir:
        Directory containing the serialised candidate files.

    Returns
    -------
    RawCandidateSet
    """
    path = os.path.join(candidates_dir, f"{split}_candidates")
    return RawCandidateSet.load(path)


# ---------------------------------------------------------------------------
# provenance_check
# ---------------------------------------------------------------------------


def provenance_check(cset: RawCandidateSet) -> dict:
    """Assert leakage-control invariants on a RawCandidateSet.

    OOF invariant (split == "train"):
      Every candidate has exactly one source_detector.
      source_detector[0] == fold_of[case_id].
      This means: the detector that scored the candidate is the fold detector
      whose OOF fold contains that case (detector k trained on all folds != k,
      then scored fold-k cases). No detector ever scored its own training cases.

    Ensemble invariant (split in {"val", "test"}):
      Every candidate has 1..5 source_detectors, all in {0,1,2,3,4}.
      No 6th detector (id >= 5).

    Parameters
    ----------
    cset:
        The RawCandidateSet to check.

    Returns
    -------
    dict
        {"ok": True, "n_checked": int} on success.

    Raises
    ------
    ProvenanceError
        On any invariant violation. The message names the failing candidate.
    """
    if cset.split == "train":
        # OOF invariant
        for c in cset.candidates:
            if len(c.source_detectors) != 1:
                raise ProvenanceError(
                    f"OOF leakage: case_id={c.case_id} has "
                    f"{len(c.source_detectors)} source_detectors, expected exactly 1. "
                    "Each OOF candidate must come from exactly one fold detector."
                )
            src = c.source_detectors[0]
            expected_fold = cset._fold_of.get(c.case_id)
            if expected_fold is None:
                raise ProvenanceError(
                    f"OOF leakage: case_id={c.case_id} is not in the fold_of mapping. "
                    "Cannot verify OOF invariant without fold membership."
                )
            if src != expected_fold:
                raise ProvenanceError(
                    f"OOF leakage: case_id={c.case_id} has source_detector={src} "
                    f"but fold_of[{c.case_id}]={expected_fold}. "
                    f"Detector {src} trains on folds != {src}, which includes fold "
                    f"{expected_fold}, so it trained on case {c.case_id}. "
                    "This is a leakage violation (thesis §3.2.1)."
                )
    else:
        # Ensemble invariant (val or test)
        for c in cset.candidates:
            if len(c.source_detectors) == 0:
                raise ProvenanceError(
                    f"Ensemble provenance: case_id={c.case_id} split='{cset.split}' "
                    "has empty source_detectors. "
                    "Each val/test candidate must list 1..5 contributing fold detectors."
                )
            for fd in c.source_detectors:
                if fd >= 5:
                    raise ProvenanceError(
                        f"Ensemble provenance: case_id={c.case_id} split='{cset.split}' "
                        f"has source_detector={fd} which is a sixth (or higher) detector "
                        "id. Only fold detectors 0..4 are allowed — no 6th all-data "
                        "detector (thesis §3.2.1)."
                    )

    return {"ok": True, "n_checked": len(cset.candidates)}


# ---------------------------------------------------------------------------
# embedding_variance_gap
# ---------------------------------------------------------------------------


def embedding_variance_gap(
    oof_set: RawCandidateSet,
    ensemble_set: RawCandidateSet,
) -> dict:
    """OOF-vs-ensemble embedding-variance gap diagnostic (D01.6).

    Computes per-embedding-dimension variance of OOF candidate features vs
    ensemble val/test candidate features, plus a pooled summary statistic.

    This is a transparency diagnostic: it lets the Stage-0 audit state how
    conservative the OOF-measured relational signal is relative to the ensemble
    set the downstream verdicts use. A pooled_mean_ratio < 1.0 means OOF
    embeddings are less variable (more conservative) than ensemble embeddings.

    Ensemble averaging over multiple detectors reduces embedding variance; a
    ratio < 1.0 is expected and benign.

    Parameters
    ----------
    oof_set:
        OOF RawCandidateSet (split == "train").
    ensemble_set:
        Ensemble RawCandidateSet (split in {"val", "test"}).

    Returns
    -------
    dict with keys:
        "per_dim_oof_var"      : np.ndarray (D,) — per-dimension OOF variance
        "per_dim_ensemble_var" : np.ndarray (D,) — per-dimension ensemble variance
        "per_dim_ratio"        : np.ndarray (D,) — OOF_var / ensemble_var
                                 (inf/nan where ensemble_var == 0)
        "pooled_mean_ratio"    : float — mean of per_dim_ratio over finite values;
                                 nan if all ensemble dims have zero variance.
    """
    if not oof_set.candidates:
        raise ValueError("oof_set has no candidates — cannot compute variance gap.")
    if not ensemble_set.candidates:
        raise ValueError("ensemble_set has no candidates — cannot compute variance gap.")

    oof_embs = np.stack([c.embedding for c in oof_set.candidates], axis=0).astype(np.float64)
    ens_embs = np.stack([c.embedding for c in ensemble_set.candidates], axis=0).astype(np.float64)

    # Population variance (ddof=0) — consistent with a descriptive diagnostic.
    oof_var = np.var(oof_embs, axis=0)
    ens_var = np.var(ens_embs, axis=0)

    # Per-dim ratio: OOF / ensemble. Where ensemble_var == 0, ratio is inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ens_var > 0, oof_var / ens_var, np.inf)

    # Pooled mean: mean over finite values only.
    finite_mask = np.isfinite(ratio)
    if finite_mask.any():
        pooled_mean_ratio = float(np.mean(ratio[finite_mask]))
    else:
        pooled_mean_ratio = float("nan")

    return {
        "per_dim_oof_var": oof_var.astype(np.float64),
        "per_dim_ensemble_var": ens_var.astype(np.float64),
        "per_dim_ratio": ratio.astype(np.float64),
        "pooled_mean_ratio": pooled_mean_ratio,
    }


# ---------------------------------------------------------------------------
# generate_oof_candidates
# ---------------------------------------------------------------------------


def generate_oof_candidates(
    fold: int,
    detector_ckpt: str,
    nndet_dataset_root: str,
    inference_fn: Callable[[list[int]], list[RawCandidate]] | None = None,
    split_override: FoldSplit | None = None,
    case_ids_override: list[int] | None = None,
) -> list[RawCandidate]:
    """Apply fold-``fold`` detector to oof_ids(fold) ONLY.

    Each returned candidate has source_detectors == (fold,) and
    case_id in oof_ids(fold). Raises if asked to score a case in train_ids(fold).

    The leakage guard is enforced in code: before calling inference_fn, this
    function verifies that every case_id to be scored is in oof_ids(fold) and
    raises ValueError if any is in train_ids(fold).

    On the server, ``inference_fn`` wraps the nnDetection inference CLI and
    returns the raw proposals with source_detectors set. In tests, a synthetic
    stub is passed.

    If ``inference_fn`` is None and we are in a context where nnDetection is
    available, the function can be extended to call the CLI directly. For now,
    the function is usable only with an explicit inference_fn (the server
    scripts/generate_candidates.py provides this).

    Parameters
    ----------
    fold:
        OOF fold index (0–4). The detector trained on all OTHER folds.
    detector_ckpt:
        Path to the fold detector checkpoint (server path).
    nndet_dataset_root:
        Path to the nnDetection dataset root (server path).
    inference_fn:
        Callable(case_ids: list[int]) -> list[RawCandidate]. Performs the
        actual nnDetection inference for the given case ids and returns raw
        proposals (with source_detectors == (fold,) set by the caller).
        If None, raises NotImplementedError.
    split_override:
        Inject a synthetic FoldSplit for testing. If None, loads the frozen
        manifest via load_split().
    case_ids_override:
        If provided, score these case_ids instead of oof_ids(fold). Used by
        tests to verify the leakage guard raises for train-ids cases.

    Returns
    -------
    list[RawCandidate]
        Raw proposals for oof_ids(fold), each with source_detectors == (fold,).

    Raises
    ------
    ProvenanceError
        If any case_id to be scored is in train_ids(fold) (leakage guard, ASC-01_02.7).
        Fires BEFORE inference_fn is called (pre-condition, not post-condition).
    NotImplementedError
        If inference_fn is None.
    """
    split = split_override if split_override is not None else load_split()
    train_ids = set(split.train_ids(fold))
    oof_ids = set(split.oof_ids(fold))

    if case_ids_override is not None:
        case_ids_to_score = case_ids_override
    else:
        case_ids_to_score = sorted(oof_ids)

    # Leakage guard (ASC-01_02.7): refuse to score any train-ids case.
    # Raises ProvenanceError (not ValueError) so the in-code guard is the same
    # exception type as provenance_check — callers can catch one exception class
    # for all leakage violations.  This is a PRE-CONDITION: it fires BEFORE
    # inference_fn is ever called.
    leakage_cases = [cid for cid in case_ids_to_score if cid in train_ids]
    if leakage_cases:
        raise ProvenanceError(
            f"OOF leakage guard triggered for fold={fold}: "
            f"the following case_ids are in train_ids({fold}) and must NOT be scored "
            f"by detector {fold}: {leakage_cases}. "
            f"Detector {fold} was trained on these cases — scoring them would violate "
            "the OOF leakage-control invariant (thesis §3.2.1)."
        )

    if inference_fn is None:
        raise NotImplementedError(
            "generate_oof_candidates requires an explicit inference_fn. "
            "On the server, pass a callable that wraps the nnDetection inference CLI "
            "(see scripts/generate_candidates.py)."
        )

    raw = inference_fn(sorted(case_ids_to_score))

    # Ensure every returned candidate has source_detectors == (fold,)
    # and case_id in oof_ids.
    validated: list[RawCandidate] = []
    for c in raw:
        # Skip oof_ids membership check when case_ids_override is set (test mode):
        # tests may inject non-OOF cases to exercise the leakage guard above.
        if c.case_id not in oof_ids and case_ids_override is None:
            raise ValueError(
                f"inference_fn returned a candidate for case_id={c.case_id} "
                f"which is not in oof_ids({fold}) = {sorted(oof_ids)}. "
                "Check the inference_fn implementation."
            )
        if c.source_detectors != (fold,):
            # Override source_detectors if the inference_fn didn't set them correctly.
            c = RawCandidate(
                case_id=c.case_id,
                split=c.split,
                bbox=c.bbox,
                score=c.score,
                embedding=c.embedding,
                source_detectors=(fold,),
            )
        validated.append(c)

    return validated


# ---------------------------------------------------------------------------
# generate_ensemble_candidates
# ---------------------------------------------------------------------------


def generate_ensemble_candidates(
    split: str,
    detector_ckpts: dict[int, str],
    nndet_dataset_root: str,
    inference_fn: Callable[[int, list[int]], list[RawCandidate]] | None = None,
) -> list[RawCandidate]:
    """Apply all 5 fold detectors to val/test cases, then ensemble_combine.

    For split in {"val", "test"}: apply ALL FIVE fold detectors to every case,
    then ensemble_combine (union → WBC → per-cluster score+embedding averaging).
    Each returned candidate's source_detectors lists the fold detectors whose
    proposals survived into its cluster.

    NEVER trains or uses a 6th all-data detector (thesis §3.2.1).

    The provisional WBC IoU threshold (0.5) is used here; the val/test ensemble
    combination will be re-run in STORY_01_03 with the calibrated WBCParams so
    the val/test candidate set uses the same frozen WBC as everything else.

    Parameters
    ----------
    split:
        "val" or "test" (train is handled by generate_oof_candidates).
    detector_ckpts:
        {fold_id: checkpoint_path} for all five fold detectors.
    nndet_dataset_root:
        Path to the nnDetection dataset root (server path).
    inference_fn:
        Callable(fold: int, case_ids: list[int]) -> list[RawCandidate].
        Convention: an empty ``case_ids`` list means "all cases for this split"
        (the callable discovers the case set from the split manifest).
        If None, raises NotImplementedError.

    Returns
    -------
    list[RawCandidate]
        Ensemble-combined candidates for all val/test cases.

    Raises
    ------
    ValueError
        If split is "train" (wrong function — use generate_oof_candidates).
    NotImplementedError
        If inference_fn is None.
    """
    if split == "train":
        raise ValueError(
            "generate_ensemble_candidates does not handle the train split. "
            "Use generate_oof_candidates for OOF candidate generation."
        )
    if inference_fn is None:
        raise NotImplementedError(
            "generate_ensemble_candidates requires an explicit inference_fn. "
            "On the server, pass a callable that wraps the nnDetection inference CLI."
        )

    from abus.detect.ensemble import ensemble_combine

    # Apply each fold detector and collect per-case proposals.
    # proposals_by_case: {case_id: [list of proposals from all detectors]}
    proposals_by_case: dict[int, list[RawCandidate]] = {}

    for fold_id, _ckpt in sorted(detector_ckpts.items()):
        fold_proposals = inference_fn(fold_id, [])  # empty list = all val/test cases
        for c in fold_proposals:
            if c.case_id not in proposals_by_case:
                proposals_by_case[c.case_id] = []
            # Ensure source_detectors is tagged with this fold.
            tagged = RawCandidate(
                case_id=c.case_id,
                split=split,
                bbox=c.bbox,
                score=c.score,
                embedding=c.embedding,
                source_detectors=(fold_id,),
            )
            proposals_by_case[c.case_id].append(tagged)

    # Ensemble combine per case
    combined: list[RawCandidate] = []
    for case_id in sorted(proposals_by_case.keys()):
        case_proposals = proposals_by_case[case_id]
        combined.extend(ensemble_combine(case_proposals, iou_threshold=0.5))

    return combined


# ---------------------------------------------------------------------------
# CLI — provenance check entrypoint
# ---------------------------------------------------------------------------


def _cli_provenance_check(raw_candidates_dir: str) -> int:
    """Load all RawCandidateSets in dir, run provenance_check, print summary."""
    splits = ["train", "val", "test"]
    found = False
    oof_set: RawCandidateSet | None = None
    ensemble_set: RawCandidateSet | None = None

    for split in splits:
        path = os.path.join(raw_candidates_dir, f"{split}_candidates")
        npz = path + ".npz"
        js = path + ".json"
        if not (os.path.exists(npz) and os.path.exists(js)):
            continue
        found = True
        print(f"Loading {split} candidates from {path}...")
        cset = RawCandidateSet.load(path)
        try:
            result = provenance_check(cset)
            print(f"  {split}: PROVENANCE OK ({result['n_checked']} candidates)")
        except ProvenanceError as e:
            print(f"  {split}: PROVENANCE FAIL — {e}")
            return 1

        if split == "train":
            oof_set = cset
        elif split in ("val", "test") and ensemble_set is None:
            ensemble_set = cset

    if not found:
        print(f"No candidate files found in {raw_candidates_dir!r}.")
        return 1

    print("PROVENANCE OK")

    # Embedding variance gap
    if (
        oof_set is not None
        and ensemble_set is not None
        and oof_set.candidates
        and ensemble_set.candidates
    ):
        print("\n--- OOF-vs-ensemble embedding-variance gap ---")
        gap = embedding_variance_gap(oof_set, ensemble_set)
        print(f"  pooled_mean_ratio (OOF/ensemble): {gap['pooled_mean_ratio']:.4f}")
        print(f"  per_dim_oof_var (first 5 dims):  {gap['per_dim_oof_var'][:5]}")
        print(f"  per_dim_ensemble_var (first 5 dims): {gap['per_dim_ensemble_var'][:5]}")

    return 0


def main() -> None:
    """CLI entry point for ``python -m abus.detect.candidates``."""
    parser = argparse.ArgumentParser(
        description="abus.detect.candidates — provenance check and variance gap."
    )
    parser.add_argument(
        "--provenance-check",
        metavar="RAW_CANDIDATES_DIR",
        help="Run provenance_check on all RawCandidateSets in the given directory.",
    )
    args = parser.parse_args()

    if args.provenance_check:
        exit_code = _cli_provenance_check(args.provenance_check)
        raise SystemExit(exit_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
