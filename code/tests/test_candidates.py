"""Tests for STORY_01_02: RawCandidate, RawCandidateSet, provenance_check,
ensemble_combine, embedding_variance_gap — all on synthetic data, CPU-only, no nnDetection.

Test plan (from story spec):
  - test_provenance_oof: a synthetic OOF candidate whose source detector trained on its
    case raises ProvenanceError.
  - test_provenance_ensemble: a val candidate with 0 source_detectors or a 6th-id detector
    raises ProvenanceError.
  - test_ensemble_averaging: ensemble_combine averages scores and embeddings correctly
    (closed-form check on 3 overlapping proposals from 3 detectors).
  - test_oof_leakage_guard: generate_oof_candidates raises when asked (via stub) to score
    a case in train_ids(fold).
  - test_serialisation_roundtrip: RawCandidateSet.save then .load reproduces every field.
  - test_embedding_variance_gap: closed-form check on known-variance synthetic sets.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from abus.detect.candidates import (
    ProvenanceError,
    RawCandidate,
    RawCandidateSet,
    embedding_variance_gap,
    generate_ensemble_candidates,
    generate_oof_candidates,
    provenance_check,
)

# ASC-01_02.7: the leakage guard must raise ProvenanceError before inference is called.
from abus.detect.ensemble import ensemble_combine
from abus.geometry.bbox import BBox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 8


def _bbox(lo: int = 0, hi: int = 5) -> BBox:
    return BBox(lo, lo, lo, hi, hi, hi)


def _candidate(
    case_id: int,
    split: str,
    source_detectors: tuple[int, ...],
    score: float = 0.9,
    emb: np.ndarray | None = None,
) -> RawCandidate:
    if emb is None:
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    return RawCandidate(
        case_id=case_id,
        split=split,
        bbox=_bbox(),
        score=score,
        embedding=emb,
        source_detectors=source_detectors,
    )


def _make_oof_set(
    fold_membership: dict[int, int],  # case_id -> fold that produced this candidate
    include_leakage_case: bool = False,
) -> RawCandidateSet:
    """Build a synthetic OOF RawCandidateSet.

    fold_membership: {case_id: source_fold} — source_fold is the fold detector used.
    For OOF to be clean, case_id must NOT be in source_fold's train set.
    The frozen manifest assigns each case to exactly one fold.

    With include_leakage_case=True, we add a candidate where the source detector's fold
    IS the fold the case belongs to — simulating the leakage scenario.
    """
    # Minimal synthetic fold map: 5 cases, one per fold.
    # fold_of[case_id] = the fold that *holds* the case (its OOF fold).
    fold_of = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    candidates = []
    for case_id, src_fold in fold_membership.items():
        candidates.append(_candidate(case_id, "train", (src_fold,)))

    if include_leakage_case:
        # Leakage: case_id=0 belongs to fold 0; source detector is also fold 0.
        # This means detector 0 scored case 0 which was in fold 0 = its OOF set.
        # WAIT: OOF means the detector was trained on all OTHER folds, so fold_0
        # detector trains on folds 1,2,3,4 and scores fold_0 cases.
        # Leakage = detector fold K scoring a case that IS in its training folds
        # (i.e., source_fold = K but case is in folds != K).
        # Simpler: leakage = case_id 0 is in fold 0, but we say source detector = 1
        # (which trained on fold 0 cases). Actually the invariant is:
        #   for OOF candidate: fold_of[case_id] == source_detector[0]
        # (detector k should only score fold k cases = its OOF fold)
        # Leakage means source_detector == some fold that TRAINED on this case.
        # i.e., source_detector != fold_of[case_id]
        # Let's add case_id=0 (fold 0) with source_detector=1
        # Detector 1 trains on folds 0,2,3,4 — so it trained on case 0. Leakage!
        candidates.append(_candidate(case_id=0, split="train", source_detectors=(1,)))

    return RawCandidateSet(
        split="train",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )


# ---------------------------------------------------------------------------
# test_provenance_oof
# ---------------------------------------------------------------------------


def test_provenance_oof_clean():
    """A clean OOF set where each candidate's source detector is its fold passes."""
    # case_id=k was scored by fold detector k (which trained on all OTHER folds).
    fold_of = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    candidates = [_candidate(k, "train", (k,)) for k in range(5)]
    cset = RawCandidateSet(
        split="train",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )
    result = provenance_check(cset)
    assert result["ok"] is True


def test_provenance_oof_leakage_raises():
    """OOF candidate whose source detector trained on that case raises ProvenanceError.

    Scenario: case_id=0 belongs to fold 0. Detector 1 trains on folds {0,2,3,4},
    i.e., it trained on case_id=0. If detector 1 scored case_id=0, that is leakage.
    The invariant: for train-split, source_detector[0] == fold_of[case_id].
    """
    fold_of = {0: 0, 1: 1}
    # case_id=0 in fold 0, but source_detector=1 (wrong — detector 1 trained on fold 0)
    candidates = [
        _candidate(0, "train", (1,)),  # leakage
        _candidate(1, "train", (1,)),  # clean
    ]
    cset = RawCandidateSet(
        split="train",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )
    with pytest.raises(ProvenanceError, match="leakage"):
        provenance_check(cset)


# ---------------------------------------------------------------------------
# test_provenance_ensemble
# ---------------------------------------------------------------------------


def test_provenance_ensemble_clean():
    """Val candidates with 1–5 source detectors from folds 0–4 pass."""
    fold_of: dict[int, int] = {}  # val split has no fold membership
    candidates = [
        _candidate(100, "val", (0, 1, 2)),
        _candidate(101, "val", (0,)),
        _candidate(102, "val", (0, 1, 2, 3, 4)),
    ]
    cset = RawCandidateSet(
        split="val",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )
    result = provenance_check(cset)
    assert result["ok"] is True


def test_provenance_ensemble_empty_source_detectors_raises():
    """Val candidate with 0 source_detectors raises ProvenanceError."""
    fold_of: dict[int, int] = {}
    candidates = [_candidate(100, "val", ())]  # empty source_detectors
    cset = RawCandidateSet(
        split="val",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )
    with pytest.raises(ProvenanceError, match="source_detectors"):
        provenance_check(cset)


def test_provenance_ensemble_sixth_detector_raises():
    """Val candidate with a detector id >= 5 raises ProvenanceError."""
    fold_of: dict[int, int] = {}
    candidates = [_candidate(100, "val", (0, 5))]  # detector 5 doesn't exist
    cset = RawCandidateSet(
        split="val",
        candidates=candidates,
        detector_commit="abc123",
        _fold_of=fold_of,
    )
    with pytest.raises(ProvenanceError, match="sixth"):
        provenance_check(cset)


# ---------------------------------------------------------------------------
# test_ensemble_averaging
# ---------------------------------------------------------------------------


def test_ensemble_averaging_three_overlapping_detectors():
    """ensemble_combine averages score and embedding across 3 overlapping proposals.

    Setup: 3 proposals all covering the same location from detectors 0, 1, 2.
    All boxes overlap with IoU > 0 (same bbox).
    Expected cluster: 1 combined candidate.
    Score: mean of the 3 individual scores.
    Embedding: mean of the 3 individual embeddings.
    source_detectors: (0, 1, 2).
    """
    emb0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    proposals = [
        _candidate(100, "val", (0,), score=0.9, emb=emb0),
        _candidate(100, "val", (1,), score=0.7, emb=emb1),
        _candidate(100, "val", (2,), score=0.5, emb=emb2),
    ]

    # WBC IoU threshold = 0.0 ensures all overlapping boxes cluster together.
    combined = ensemble_combine(proposals, iou_threshold=0.0)

    assert len(combined) == 1, f"Expected 1 combined candidate, got {len(combined)}"
    c = combined[0]
    expected_score = (0.9 + 0.7 + 0.5) / 3
    assert abs(c.score - expected_score) < 1e-5, f"Score {c.score} != {expected_score}"

    expected_emb = (emb0 + emb1 + emb2) / 3
    np.testing.assert_allclose(c.embedding, expected_emb, atol=1e-5)

    assert set(c.source_detectors) == {0, 1, 2}


def test_ensemble_averaging_non_overlapping_stays_separate():
    """Non-overlapping proposals from different detectors remain separate clusters."""
    emb = np.ones(4, dtype=np.float32)

    proposals = [
        RawCandidate(
            case_id=100,
            split="val",
            bbox=BBox(0, 0, 0, 2, 2, 2),
            score=0.9,
            embedding=emb.copy(),
            source_detectors=(0,),
        ),
        RawCandidate(
            case_id=100,
            split="val",
            bbox=BBox(100, 100, 100, 102, 102, 102),  # far away, no overlap
            score=0.8,
            embedding=emb.copy(),
            source_detectors=(1,),
        ),
    ]

    combined = ensemble_combine(proposals, iou_threshold=0.5)
    assert len(combined) == 2, f"Expected 2 separate candidates, got {len(combined)}"


# ---------------------------------------------------------------------------
# test_oof_leakage_guard
# ---------------------------------------------------------------------------


def test_oof_leakage_guard_raises_for_train_case():
    """generate_oof_candidates raises if asked to score a train_ids(fold) case.

    We use a stub: the function accepts a callable `inference_fn` that maps
    case_id -> list[RawCandidate]. The leakage guard runs before calling it.
    If a case_id is in train_ids(fold), it raises without invoking inference_fn.
    """

    # fold 0 OOF ids = fold-0 cases; fold 0 train ids = folds 1..4 cases.
    # We'll pass a small synthetic split that makes case_id=1 a train case for fold 0.

    # Provide a simple inference stub that should NOT be called for train cases.
    call_log = []

    def inference_stub(case_ids: list[int]) -> list[RawCandidate]:
        call_log.extend(case_ids)
        return []

    # Build a minimal split with only 2 cases: case 0 in fold 0, case 1 in fold 1.
    # fold=0 detector trains on fold 1 cases, scores fold 0 cases.
    # Asking it to score case 1 (train case for fold 0 detector) should raise.
    from abus.data.split import FoldSplit

    synthetic_split = FoldSplit(
        folds=[[0], [1]],  # fold 0 = [case 0], fold 1 = [case 1]
        fold_of={0: 0, 1: 1},
        seed=42,
        splitter_version="1.0",
        label_of={0: "M", 1: "B"},
    )

    # The leakage guard must raise ProvenanceError (ASC-01_02.7) — not a generic
    # ValueError. This ensures the guard fires as a PRE-CONDITION before inference.
    with pytest.raises(ProvenanceError, match="leakage"):
        generate_oof_candidates(
            fold=0,
            detector_ckpt="fake_ckpt",
            nndet_dataset_root="fake_root",
            inference_fn=inference_stub,
            split_override=synthetic_split,
            # ask it to score case_id=1, which is a TRAIN case for fold=0
            case_ids_override=[1],
        )

    # The inference function should NOT have been called (pre-condition guard)
    assert call_log == []


# ---------------------------------------------------------------------------
# test_serialisation_roundtrip
# ---------------------------------------------------------------------------


def test_serialisation_roundtrip():
    """RawCandidateSet.save then .load reproduces every candidate field exactly."""
    emb_a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    emb_b = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    fold_of = {0: 0, 1: 0, 100: -1}  # -1 = val/test (no fold)

    original = RawCandidateSet(
        split="train",
        candidates=[
            RawCandidate(
                case_id=0,
                split="train",
                bbox=BBox(1, 2, 3, 4, 5, 6),
                score=0.87,
                embedding=emb_a,
                source_detectors=(0,),
            ),
            RawCandidate(
                case_id=1,
                split="train",
                bbox=BBox(10, 20, 30, 40, 50, 60),
                score=0.42,
                embedding=emb_b,
                source_detectors=(1,),
            ),
        ],
        detector_commit="deadbeef",
        _fold_of=fold_of,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_candidates")
        original.save(path)

        loaded = RawCandidateSet.load(path)

    assert loaded.split == original.split
    assert loaded.detector_commit == original.detector_commit
    assert len(loaded.candidates) == len(original.candidates)

    for orig_c, load_c in zip(original.candidates, loaded.candidates, strict=False):
        assert orig_c.case_id == load_c.case_id
        assert orig_c.split == load_c.split
        assert orig_c.bbox == load_c.bbox
        assert abs(orig_c.score - load_c.score) < 1e-6
        np.testing.assert_array_equal(orig_c.embedding, load_c.embedding)
        assert orig_c.embedding.dtype == np.float32
        assert load_c.embedding.dtype == np.float32
        assert orig_c.source_detectors == load_c.source_detectors


# ---------------------------------------------------------------------------
# test_embedding_variance_gap
# ---------------------------------------------------------------------------


def test_embedding_variance_gap_known_variances():
    """embedding_variance_gap returns correct per-dim variances and pooled ratio.

    OOF set: all embeddings = [1, 2, 3] and [4, 5, 6] → per-dim var = [4.5, 4.5, 4.5].
    Ensemble set: all embeddings = [1, 1, 1] and [1, 1, 1] → per-dim var = [0, 0, 0].

    Pooled mean ratio = mean(oof_var / ensemble_var). With ensemble_var = 0 on all dims,
    ratio is inf (or handled specially). Let's use non-zero ensemble variance.

    OOF embeddings:       [[0, 0], [2, 2]]   → var = [1, 1] (each dim)
    Ensemble embeddings:  [[0, 0], [4, 4]]   → var = [4, 4] (each dim)
    Per-dim ratio (OOF / ensemble) = [0.25, 0.25]. Pooled mean = 0.25.
    """
    fold_of = {0: 0, 1: 0}

    oof_set = RawCandidateSet(
        split="train",
        candidates=[
            _candidate(0, "train", (0,), emb=np.array([0.0, 0.0], dtype=np.float32)),
            _candidate(1, "train", (0,), emb=np.array([2.0, 2.0], dtype=np.float32)),
        ],
        detector_commit="abc",
        _fold_of=fold_of,
    )

    ensemble_set = RawCandidateSet(
        split="val",
        candidates=[
            _candidate(100, "val", (0,), emb=np.array([0.0, 0.0], dtype=np.float32)),
            _candidate(101, "val", (0,), emb=np.array([4.0, 4.0], dtype=np.float32)),
        ],
        detector_commit="abc",
        _fold_of={},
    )

    result = embedding_variance_gap(oof_set, ensemble_set)

    assert "per_dim_oof_var" in result
    assert "per_dim_ensemble_var" in result
    assert "per_dim_ratio" in result
    assert "pooled_mean_ratio" in result

    # OOF: mean=[1,1], var=[(0-1)²+(2-1)²]/2=1 per dim
    np.testing.assert_allclose(result["per_dim_oof_var"], [1.0, 1.0], atol=1e-5)
    # Ensemble: mean=[2,2], var=[(0-2)²+(4-2)²]/2=4 per dim
    np.testing.assert_allclose(result["per_dim_ensemble_var"], [4.0, 4.0], atol=1e-5)
    # Per-dim ratio (OOF/ensemble) = 0.25 each
    np.testing.assert_allclose(result["per_dim_ratio"], [0.25, 0.25], atol=1e-5)
    # Pooled mean ratio
    assert abs(result["pooled_mean_ratio"] - 0.25) < 1e-5


def test_embedding_variance_gap_zero_ensemble_variance_handled():
    """When ensemble variance is 0 on a dim, ratio for that dim is inf or nan (not crash)."""
    fold_of = {0: 0, 1: 0}

    oof_set = RawCandidateSet(
        split="train",
        candidates=[
            _candidate(0, "train", (0,), emb=np.array([0.0, 1.0], dtype=np.float32)),
            _candidate(1, "train", (0,), emb=np.array([2.0, 1.0], dtype=np.float32)),
        ],
        detector_commit="abc",
        _fold_of=fold_of,
    )

    # All ensemble embeddings identical → zero variance on both dims
    ensemble_set = RawCandidateSet(
        split="val",
        candidates=[
            _candidate(100, "val", (0,), emb=np.array([1.0, 1.0], dtype=np.float32)),
            _candidate(101, "val", (0,), emb=np.array([1.0, 1.0], dtype=np.float32)),
        ],
        detector_commit="abc",
        _fold_of={},
    )

    # Should not raise; result is numerically valid (inf or nan where div-by-zero)
    result = embedding_variance_gap(oof_set, ensemble_set)
    assert "pooled_mean_ratio" in result
    # The function must return a dict with all required keys
    for key in ("per_dim_oof_var", "per_dim_ensemble_var", "per_dim_ratio", "pooled_mean_ratio"):
        assert key in result


# ---------------------------------------------------------------------------
# ASC-01_02.7 — explicit test: generate_oof_candidates leakage guard is a
#               PRE-CONDITION that raises ProvenanceError BEFORE inference
# ---------------------------------------------------------------------------


def test_asc_01_02_7_leakage_guard_is_precondition_raises_before_inference():
    """ASC-01_02.7: generate_oof_candidates raises ProvenanceError BEFORE calling
    inference_fn when any case_id is in train_ids(fold).

    This is the in-code leakage guard that ASC-01_02.3 relies on. It must:
      1. Raise ProvenanceError (not ValueError or any other exception type).
      2. Fire BEFORE the inference function is called (pre-condition, not post).
      3. Include "leakage" in the message so callers can identify it.
    """
    from abus.data.split import FoldSplit

    # 3-fold synthetic split for simplicity:
    #   fold 0 = [0, 1], fold 1 = [2, 3], fold 2 = [4, 5]
    # Detector for fold=0 trains on folds 1+2 (cases 2,3,4,5).
    # OOF cases for fold=0 are cases 0 and 1.
    # Asking fold=0 detector to score case 2 (a train case) must raise.
    synthetic_split = FoldSplit(
        folds=[[0, 1], [2, 3], [4, 5]],
        fold_of={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2},
        seed=42,
        splitter_version="1.0",
        label_of={0: "M", 1: "B", 2: "M", 3: "B", 4: "M", 5: "B"},
    )

    inference_called = []

    def inference_stub(case_ids: list[int]) -> list[RawCandidate]:
        inference_called.extend(case_ids)
        return []

    # case_ids_override=[2] → case 2 is in train_ids(0) → must raise ProvenanceError
    with pytest.raises(ProvenanceError) as exc_info:
        generate_oof_candidates(
            fold=0,
            detector_ckpt="fake_ckpt",
            nndet_dataset_root="fake_root",
            inference_fn=inference_stub,
            split_override=synthetic_split,
            case_ids_override=[2],  # train case for fold=0
        )

    # Must be ProvenanceError specifically
    assert isinstance(exc_info.value, ProvenanceError)
    assert (
        "leakage" in str(exc_info.value).lower()
    ), f"Expected 'leakage' in error message, got: {exc_info.value}"

    # The inference function must NOT have been called (pre-condition, not post)
    assert (
        inference_called == []
    ), f"inference_fn was called before the leakage guard fired: {inference_called}"


# ---------------------------------------------------------------------------
# Additional: RawCandidate is frozen (immutable)
# ---------------------------------------------------------------------------


def test_raw_candidate_is_frozen():
    """RawCandidate is a frozen dataclass — mutation raises."""
    c = _candidate(0, "train", (0,))
    with pytest.raises((TypeError, AttributeError)):
        c.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional: RawCandidateSet serialisation preserves fold_of
# ---------------------------------------------------------------------------


def test_serialisation_preserves_fold_of():
    """Round-trip preserves the _fold_of mapping."""
    fold_of = {0: 0, 1: 1, 2: 2}
    cset = RawCandidateSet(
        split="train",
        candidates=[_candidate(k, "train", (k,)) for k in range(3)],
        detector_commit="abc",
        _fold_of=fold_of,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_cands")
        cset.save(path)
        loaded = RawCandidateSet.load(path)

    assert loaded._fold_of == fold_of


# ---------------------------------------------------------------------------
# ASC-01_02.4 (D01.12) — source_detector cross-validation helper
# ---------------------------------------------------------------------------


def test_source_detector_cross_validation_agrees():
    """Synthetic per-detector proposals produce consistent source_detectors sets
    whether combined via ensemble_combine (branch-b fallback) or simply assigned.

    This tests the *helper logic* that validates branch-(a) vs branch-(b) agreement
    on the server. The synthetic version: create per-fold proposals for one case,
    combine them via ensemble_combine, verify that the resulting source_detectors
    equals the expected set of contributing folds.

    On the server, this test's logic is exercised by comparing the branch-(a)
    consolidated nndet_predict output against branch-(b)'s per-fold predict_oof
    calls on five specific val cases (D01.12 clause in ASC-01_02.4).
    """
    # Five fold detectors each contribute one proposal for case_id=100.
    # All proposals overlap (same bbox), so WBC should collapse them into one cluster.
    embs = [np.full(4, float(i), dtype=np.float32) for i in range(5)]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]

    proposals = [_candidate(100, "val", (i,), score=scores[i], emb=embs[i]) for i in range(5)]

    # IoU threshold = 0 ensures all identical-bbox proposals cluster together.
    combined = ensemble_combine(proposals, iou_threshold=0.0)

    assert len(combined) == 1, "All 5 overlapping proposals should form 1 cluster"
    cluster = combined[0]

    # All 5 fold detectors should be listed as source_detectors
    assert set(cluster.source_detectors) == {
        0,
        1,
        2,
        3,
        4,
    }, f"Expected {{0,1,2,3,4}} source_detectors, got {cluster.source_detectors}"

    # Score should be mean of the 5 scores
    expected_score = float(np.mean(scores))
    assert (
        abs(cluster.score - expected_score) < 1e-5
    ), f"Expected score {expected_score}, got {cluster.score}"

    # Embedding should be mean of the 5 embeddings
    expected_emb = np.mean(np.stack(embs), axis=0).astype(np.float32)
    np.testing.assert_allclose(cluster.embedding, expected_emb, atol=1e-5)


# ---------------------------------------------------------------------------
# generate_ensemble_candidates public API (S4 — was untested)
# ---------------------------------------------------------------------------


def test_generate_ensemble_candidates_val_split():
    """generate_ensemble_candidates routes val cases through ensemble_combine.

    Verifies the public API is exercised: five fold detectors each propose one
    overlapping candidate for case_id=100 in the val split. With IoU threshold=0
    all proposals cluster into one combined candidate with source_detectors={0..4}.
    """
    # Each "fold detector" returns one proposal for case_id=100.
    fold_proposals = {
        fold: [_candidate(100, "val", (fold,), score=0.9 - fold * 0.1)] for fold in range(5)
    }

    def inference_stub(fold_id: int, case_ids: list[int]) -> list[RawCandidate]:
        # Re-tag with the correct fold id (generate_ensemble_candidates overrides anyway)
        return fold_proposals[fold_id]

    # detector_ckpts: fold_id -> ckpt path (paths are unused by the stub)
    detector_ckpts = {fold: f"/fake/fold{fold}/model_best.ckpt" for fold in range(5)}

    result = generate_ensemble_candidates(
        split="val",
        detector_ckpts=detector_ckpts,
        nndet_dataset_root="/fake/root",
        inference_fn=inference_stub,
    )

    # WBC default threshold (0.5) — all proposals have the same bbox (overlap > 0.5)
    # so they should collapse into one cluster.
    assert len(result) == 1, f"Expected 1 combined candidate, got {len(result)}"
    c = result[0]
    assert c.split == "val"
    assert c.case_id == 100
    assert set(c.source_detectors) == {0, 1, 2, 3, 4}
    # Score should be the mean of all 5 fold scores
    expected_score = float(np.mean([0.9, 0.8, 0.7, 0.6, 0.5]))
    assert abs(c.score - expected_score) < 1e-5


def test_generate_ensemble_candidates_raises_for_train_split():
    """generate_ensemble_candidates raises ValueError for split='train'."""
    with pytest.raises(ValueError, match="train"):
        generate_ensemble_candidates(
            split="train",
            detector_ckpts={},
            nndet_dataset_root="/fake",
            inference_fn=lambda fold, cids: [],
        )


def test_generate_ensemble_candidates_raises_without_inference_fn():
    """generate_ensemble_candidates raises NotImplementedError when inference_fn is None."""
    with pytest.raises(NotImplementedError):
        generate_ensemble_candidates(
            split="val",
            detector_ckpts={0: "/fake"},
            nndet_dataset_root="/fake",
            inference_fn=None,
        )
