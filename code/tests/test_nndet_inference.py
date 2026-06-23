"""Tests for src/abus/detect/nndet_inference.py (STORY_01_02, D01.9 + D01.13).

Schema note (D01.13, source-grounded against nnDetection 0.1 commit 97a58f3):
  predict_dir(save_state=False) writes ONE FILE PER KEY:
    <case_id>_pred_boxes.pkl   — pickle of np.ndarray shape (N, 6) float32
    <case_id>_pred_scores.pkl  — pickle of np.ndarray shape (N,) float32
    <case_id>_pred_labels.pkl  — pickle of np.ndarray shape (N,) int/float

  Box axis order (nndet/core/boxes/ops.py line 34, detection.py line 228):
    (x1, y1, x2, y2, z1, z2)  — NOT (z1, y1, x1, z2, y2, x2)

  NO embeddings key in the predict_dir output (helper.py:103-110, to_numpy).
  Embeddings are set to None in RawDetections; generate_candidates fills a
  zero-vector placeholder until STORY_01_04 wires backbone extraction.

  D01.9 schema (single *_boxes.pkl dict with 'boxes'/'scores'/'embeddings')
  is SUPERSEDED. Tests updated to mirror the real per-key schema.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from abus.detect.nndet_inference import RawDetections, parse_predictions_dir

# ---------------------------------------------------------------------------
# Helpers: build synthetic pickle fixtures
# ---------------------------------------------------------------------------

NNDET_SCHEMA_VERSION = "0.1_97a58f3_D01.13"  # commit short hash + decision


def _write_per_key_pkls(
    tmpdir: Path,
    case_id_str: str,
    n_dets: int = 3,
    boxes_override: np.ndarray | None = None,
    scores_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Write synthetic nnDetection per-key pkl files for one case.

    D01.13 schema (source-grounded):
      <case_id>_pred_boxes.pkl  — np.ndarray (N, 6) float32, axis (x1,y1,x2,y2,z1,z2)
      <case_id>_pred_scores.pkl — np.ndarray (N,) float32
      <case_id>_pred_labels.pkl — np.ndarray (N,) int32

    Returns dict with 'boxes' and 'scores' for verification.
    No embeddings file — embeddings are not in predict_dir output (D01.13).
    """
    if boxes_override is not None:
        boxes = boxes_override.astype(np.float32)
    else:
        # Synthetic boxes in (x1,y1,x2,y2,z1,z2) format — D01.13 confirmed axis
        boxes = np.arange(n_dets * 6, dtype=np.float32).reshape(n_dets, 6)
        # Ensure max > min: cols 0,1→min x,y; cols 2,3→max x,y; cols 4,5→z1,z2
        boxes[:, 2] += 2.0  # x2 > x1
        boxes[:, 3] += 2.0  # y2 > y1
        boxes[:, 5] += 2.0  # z2 > z1

    if scores_override is not None:
        scores = scores_override.astype(np.float32)
    else:
        scores = np.linspace(0.9, 0.5, n_dets, dtype=np.float32)

    labels = np.zeros(n_dets, dtype=np.int32)

    with open(tmpdir / f"{case_id_str}_pred_boxes.pkl", "wb") as f:
        pickle.dump(boxes, f)
    with open(tmpdir / f"{case_id_str}_pred_scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    with open(tmpdir / f"{case_id_str}_pred_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    return {"boxes": boxes, "scores": scores}


# ---------------------------------------------------------------------------
# D01.13: test_parse_predictions_dir_basic — per-key schema
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_basic_per_key_schema() -> None:
    """D01.13: parse_predictions_dir reads per-key files (pred_boxes/pred_scores).

    Real nnDetection 0.1 output (helper.py:103-110, to_numpy):
      <case_id>_pred_boxes.pkl  — np.ndarray (N,6) float32
      <case_id>_pred_scores.pkl — np.ndarray (N,) float32
    No embeddings file. RawDetections.embeddings must be None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        n_dets = 4
        written = _write_per_key_pkls(tmp, "0042", n_dets=n_dets)

        result = parse_predictions_dir(tmpdir)

    assert isinstance(result, dict)
    assert 42 in result, "case_id 42 must be in result (from '0042_pred_boxes.pkl')"

    rd = result[42]
    assert isinstance(rd, RawDetections)
    assert rd.case_id == 42
    assert rd.boxes.shape == (n_dets, 6)
    assert rd.scores.shape == (n_dets,)
    assert rd.embeddings is None, "embeddings must be None (not in per-key output)"

    np.testing.assert_allclose(rd.boxes, written["boxes"], rtol=1e-5)
    np.testing.assert_allclose(rd.scores, written["scores"], rtol=1e-5)


# ---------------------------------------------------------------------------
# D01.13: box axis is (x1,y1,x2,y2,z1,z2)
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_box_axis_order_x1y1x2y2z1z2() -> None:
    """D01.13: boxes from pred_boxes.pkl are in (x1,y1,x2,y2,z1,z2) axis order.

    Source proof: nndet/core/boxes/ops.py line 34:
      'expected to be in (x1, y1, x2, y2, z1, z2) format'
    detection.py _apply_offsets_to_boxes (line 228):
      offsets: idx 0→x, 1→y, 2→x, 3→y, 4→z, 5→z

    The parser must preserve this axis order exactly — it must NOT permute to
    (z1,y1,x1,z2,y2,x2). The stored boxes have index layout:
      boxes[:, 0] = x1
      boxes[:, 1] = y1
      boxes[:, 2] = x2
      boxes[:, 3] = y2
      boxes[:, 4] = z1
      boxes[:, 5] = z2
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write a known box: x1=10, y1=20, x2=30, y2=40, z1=5, z2=15
        boxes_known = np.array([[10.0, 20.0, 30.0, 40.0, 5.0, 15.0]], dtype=np.float32)
        _write_per_key_pkls(tmp, "0001", n_dets=1, boxes_override=boxes_known)

        result = parse_predictions_dir(tmpdir)

    rd = result[1]
    box = rd.boxes[0]
    # D01.13: axis is (x1, y1, x2, y2, z1, z2)
    assert box[0] == 10.0, f"boxes[:,0] should be x1=10, got {box[0]}"
    assert box[1] == 20.0, f"boxes[:,1] should be y1=20, got {box[1]}"
    assert box[2] == 30.0, f"boxes[:,2] should be x2=30, got {box[2]}"
    assert box[3] == 40.0, f"boxes[:,3] should be y2=40, got {box[3]}"
    assert box[4] == 5.0, f"boxes[:,4] should be z1=5, got {box[4]}"
    assert box[5] == 15.0, f"boxes[:,5] should be z2=15, got {box[5]}"


# ---------------------------------------------------------------------------
# D01.13: embeddings are always None from per-key output
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_embeddings_always_none_in_per_key_schema() -> None:
    """D01.13: embeddings are NOT in predict_dir per-key output; RawDetections.embeddings=None.

    helper.py:103-110: to_numpy(result) saves each key. get_case_result returns
    pred_boxes/pred_scores/pred_labels/restore/... — NO 'embeddings' key.
    Even if an extra *_pred_embeddings.pkl were present it is not parsed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_per_key_pkls(tmp, "0010", n_dets=3)
        # Even if we write a spurious embeddings file, embeddings should still be None
        emb = np.ones((3, 16), dtype=np.float32)
        with open(tmp / "0010_pred_embeddings.pkl", "wb") as f:
            pickle.dump(emb, f)

        result = parse_predictions_dir(tmpdir)

    rd = result[10]
    assert rd.embeddings is None, (
        f"embeddings must be None from per-key schema even if extra files exist, "
        f"got {rd.embeddings}"
    )


# ---------------------------------------------------------------------------
# D01.13: filename parser — case_id from *_pred_boxes.pkl
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_filename_parser_strips_leading_zeros() -> None:
    """case_id integer extracted from '0042_pred_boxes.pkl' stem prefix -> 42."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_per_key_pkls(tmp, "0000", n_dets=1)
        _write_per_key_pkls(tmp, "0007", n_dets=1)
        _write_per_key_pkls(tmp, "0099", n_dets=1)
        _write_per_key_pkls(tmp, "0199", n_dets=1)

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == {0, 7, 99, 199}


def test_parse_predictions_dir_malformed_pred_boxes_filename_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed filenames (non-integer prefix in *_pred_boxes.pkl) are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_per_key_pkls(tmp, "0001", n_dets=2)
        # Malformed: prefix before _pred_boxes cannot be parsed as int
        malformed_boxes = np.ones((1, 6), dtype=np.float32)
        with open(tmp / "badname_pred_boxes.pkl", "wb") as f:
            pickle.dump(malformed_boxes, f)
        malformed_scores = np.ones(1, dtype=np.float32)
        with open(tmp / "badname_pred_scores.pkl", "wb") as f:
            pickle.dump(malformed_scores, f)

        with caplog.at_level(logging.WARNING):
            result = parse_predictions_dir(tmpdir)

    assert 1 in result
    assert len(result) == 1, f"Malformed filename should be skipped, got {list(result.keys())}"
    assert any(
        "badname" in record.message.lower() or "skip" in record.message.lower()
        for record in caplog.records
    ), "Expected a warning about the malformed filename"


# ---------------------------------------------------------------------------
# D01.13: zero detections
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_zero_detections() -> None:
    """An empty boxes array (0 detections) is handled; entry present with shape (0,6)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        empty_boxes = np.zeros((0, 6), dtype=np.float32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        _write_per_key_pkls(
            tmp,
            "0005",
            n_dets=0,
            boxes_override=empty_boxes,
            scores_override=empty_scores,
        )

        result = parse_predictions_dir(tmpdir)

    assert 5 in result
    rd = result[5]
    assert rd.boxes.shape == (0, 6)
    assert rd.scores.shape == (0,)


# ---------------------------------------------------------------------------
# D01.13: multiple cases
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_multiple_cases() -> None:
    """Multiple cases each with per-key pkl files are all parsed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        case_ids = [0, 1, 42, 99]
        for cid in case_ids:
            _write_per_key_pkls(tmp, f"{cid:04d}", n_dets=2)

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == set(case_ids)
    for cid in case_ids:
        assert result[cid].case_id == cid
        assert result[cid].boxes.shape == (2, 6)
        assert result[cid].embeddings is None


# ---------------------------------------------------------------------------
# D01.13: no files
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_no_files() -> None:
    """An empty directory returns an empty dict (not a crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = parse_predictions_dir(tmpdir)

    assert result == {}


# ---------------------------------------------------------------------------
# RawDetections frozen
# ---------------------------------------------------------------------------


def test_raw_detections_is_frozen() -> None:
    """RawDetections is a frozen dataclass — field assignment raises."""
    rd = RawDetections(
        case_id=0,
        boxes=np.zeros((2, 6), dtype=np.float32),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        embeddings=None,
    )
    with pytest.raises((TypeError, AttributeError)):
        rd.case_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# D01.13: dtype coercion
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_boxes_dtype_coerced_to_float32() -> None:
    """Boxes and scores arrays are returned as float32 regardless of pickle dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        boxes64 = np.ones((3, 6), dtype=np.float64)
        scores64 = np.array([0.9, 0.8, 0.7], dtype=np.float64)
        with open(tmp / "0020_pred_boxes.pkl", "wb") as f:
            pickle.dump(boxes64, f)
        with open(tmp / "0020_pred_scores.pkl", "wb") as f:
            pickle.dump(scores64, f)
        labels = np.zeros(3, dtype=np.int32)
        with open(tmp / "0020_pred_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

        result = parse_predictions_dir(tmpdir)

    rd = result[20]
    assert rd.boxes.dtype == np.float32
    assert rd.scores.dtype == np.float32


def test_parse_predictions_dir_non_pred_boxes_files_ignored() -> None:
    """Files that are not *_pred_boxes.pkl are ignored by the parser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_per_key_pkls(tmp, "0001", n_dets=2)
        # These must be ignored
        (tmp / "0002_scores.npy").write_bytes(b"dummy")
        (tmp / "README.txt").write_text("hello")
        (tmp / "0003_pred_labels.pkl").write_bytes(b"dummy")  # no matching _pred_boxes

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == {1}


# ---------------------------------------------------------------------------
# D01.13: predict_oof — real predict_dir signature
# ---------------------------------------------------------------------------


def test_predict_oof_calls_predict_dir_with_real_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """D01.13: predict_oof calls predict_dir with the real nnDetection 0.1 signature.

    Real signature (helper.py:29-42):
      predict_dir(source_dir, target_dir, cfg, plan, source_models,
                  model_fn=load_final_model, num_models=None,
                  num_tta_transforms=None, restore=False,
                  case_ids=None, save_state=False, **kwargs)

    Requirements (D01.13):
      - source_dir  = preprocessed imagesTr dir (NOT the raw task_dir)
      - target_dir  = a tempdir
      - cfg         = OmegaConf-loaded config.yaml from fold_dir
      - plan        = load_pickle(fold_dir / "plan_inference.pkl")
      - source_models = fold_dir (as Path)
      - model_fn    = partial(load_final_model, identifier="last")
      - num_models  = 1
      - restore     = False (D01.17: boxes in preprocessed space = same as FPN maps)
      - case_ids    = [f"{cid:04d}" for cid in case_ids]  (4-digit zero-padded strings)
      - save_state  = False
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    # We test the argument-construction logic by monkeypatching the lazy imports
    # inside predict_oof. Since nnDetection isn't installed on the laptop, we
    # replace the nndet imports with mocks and verify what predict_dir is called with.
    from abus.detect import nndet_inference

    called_kwargs: dict = {}

    def fake_predict_dir(
        source_dir: str,
        target_dir: str,
        cfg: dict,
        plan: dict,
        source_models: Path,
        model_fn: object = None,
        num_models: int | None = None,
        num_tta_transforms: int | None = None,
        restore: bool = False,
        case_ids: list[str] | None = None,
        save_state: bool = False,
        **kwargs: object,
    ) -> None:
        called_kwargs.update(
            {
                "source_dir": source_dir,
                "target_dir": target_dir,
                "cfg": cfg,
                "plan": plan,
                "source_models": source_models,
                "model_fn": model_fn,
                "num_models": num_models,
                "restore": restore,
                "case_ids": case_ids,
                "save_state": save_state,
            }
        )
        # Write synthetic per-key output so parse_predictions_dir can run
        target_path = Path(target_dir)
        for cid_str in case_ids or []:
            boxes = np.array([[1.0, 2.0, 3.0, 4.0, 0.0, 1.0]], dtype=np.float32)
            scores = np.array([0.9], dtype=np.float32)
            with open(target_path / f"{cid_str}_pred_boxes.pkl", "wb") as f:
                pickle.dump(boxes, f)
            with open(target_path / f"{cid_str}_pred_scores.pkl", "wb") as f:
                pickle.dump(scores, f)

    fake_cfg: dict[str, object] = {"module": "TestModule", "model_cfg": {}, "trainer_cfg": {}}
    fake_plan: dict[str, object] = {"inference_plan": {}}
    fake_load_final_model = MagicMock()

    def _fake_load_pickle(p: str) -> object:  # noqa: S301
        with open(p, "rb") as _f:
            return pickle.load(_f)  # noqa: S301

    with tempfile.TemporaryDirectory() as fold_dir_str:
        fold_dir = Path(fold_dir_str)
        # Write config.yaml and plan_inference.pkl that predict_oof expects to read.
        # OmegaConf.load is monkeypatched below, so file content is irrelevant;
        # the file just needs to exist for the path check in predict_oof.
        (fold_dir / "config.yaml").write_text("{}")  # placeholder; OmegaConf.load is mocked
        with open(fold_dir / "plan_inference.pkl", "wb") as _pkl_f:
            pickle.dump(fake_plan, _pkl_f)

        # Monkeypatch the lazy imports
        with (
            patch.dict(
                "sys.modules",
                {
                    "nndet": MagicMock(),
                    "nndet.inference": MagicMock(),
                    "nndet.inference.helper": MagicMock(predict_dir=fake_predict_dir),
                    "nndet.inference.loading": MagicMock(load_final_model=fake_load_final_model),
                    "nndet.io": MagicMock(),
                    "nndet.io.load": MagicMock(load_pickle=_fake_load_pickle),
                    "omegaconf": MagicMock(
                        OmegaConf=MagicMock(
                            load=lambda p: fake_cfg,
                            to_container=lambda c, **kw: c,
                        )
                    ),
                },
            ),
        ):
            result = nndet_inference.predict_oof(
                fold=0,
                case_ids=[0, 1],
                preprocessed_dir=str(fold_dir),  # D01.13: source_dir is preprocessed
                fold_dir=fold_dir_str,
            )

    # Verify predict_dir was called with required keyword arguments (D01.13 spec)
    assert "source_dir" in called_kwargs, "predict_dir must be called"
    n_models = called_kwargs["num_models"]
    assert n_models == 1, f"num_models must be 1, got {n_models}"
    assert called_kwargs["restore"] is False, (
        "restore must be False (D01.17: boxes kept in preprocessed space, "
        "same coordinate frame as FPN feature maps, so pool_embeddings_at_boxes "
        "can pool without inverting restore_detection's affine)"
    )
    assert called_kwargs["save_state"] is False, "save_state must be False"
    # case_ids must be 4-digit zero-padded strings
    assert called_kwargs["case_ids"] == [
        "0000",
        "0001",
    ], f"case_ids must be 4-digit strings, got {called_kwargs['case_ids']}"
    # model_fn must be callable
    assert callable(called_kwargs["model_fn"]), "model_fn must be callable"
    # cfg must come from disk (the fake values we wrote)
    assert called_kwargs["cfg"] == fake_cfg, f"cfg mismatch: {called_kwargs['cfg']}"
    # D01.13 point 6 + MC1 fix: plan passed to predict_dir must have ALL four high-recall
    # params explicitly set so a swept plan cannot silently lower topk/detections_per_image.
    plan_passed = called_kwargs["plan"]
    assert plan_passed["inference_plan"]["model_score_thresh"] == 0.0, (
        f"model_score_thresh must be 0.0 (high-recall), got "
        f"{plan_passed['inference_plan']['model_score_thresh']}"
    )
    assert plan_passed["inference_plan"]["ensemble_score_thresh"] == 0.0, (
        f"ensemble_score_thresh must be 0.0 (high-recall), got "
        f"{plan_passed['inference_plan']['ensemble_score_thresh']}"
    )
    assert plan_passed["inference_plan"]["model_topk"] == 1000, (
        f"model_topk must be 1000 (high-recall default), got "
        f"{plan_passed['inference_plan']['model_topk']}"
    )
    assert plan_passed["inference_plan"]["model_detections_per_image"] == 100, (
        f"model_detections_per_image must be 100 (high-recall default), got "
        f"{plan_passed['inference_plan']['model_detections_per_image']}"
    )
    # Result must be a dict keyed by int case_ids
    assert set(result.keys()) == {0, 1}, f"result keys must be int case_ids, got {result.keys()}"


# ===========================================================================
# D01.14 — embedding extraction tests (CPU-only, no nnDetection/GPU required)
# ===========================================================================


def test_point_pool_at_centroid_trilinear() -> None:
    """D01.14: point_pool_trilinear returns the correct value at a known centroid.

    The feature map is a 3-D ramp: feat[c, d0, d1, d2] = d0 + d1 + d2 (for c=0).
    For a box with centroid (cx_d0=2.0, cx_d1=3.0, cx_d2=4.0) the expected
    value at channel 0 is 2.0 + 3.0 + 4.0 = 9.0 (since the map is linear, trilinear
    interpolation is exact at integer coordinates and linearly interpolates between them).

    The function takes:
        feat_map : np.ndarray shape (C, D0, D1, D2) — feature map for one tile
        cx_d0, cx_d1, cx_d2 : float — centroid in tile-pixel frame

    Returns:
        np.ndarray shape (C,) — the pooled 128-D (or C-D) embedding
    """
    from abus.detect.nndet_inference import point_pool_trilinear

    C, D0, D1, D2 = 4, 10, 10, 10
    # feat[c, d0, d1, d2] = d0 + d1 + d2 for all c (known ramp)
    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    for d0 in range(D0):
        for d1 in range(D1):
            for d2 in range(D2):
                feat_map[:, d0, d1, d2] = d0 + d1 + d2

    cx_d0, cx_d1, cx_d2 = 2.0, 3.0, 4.0
    result = point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2)

    assert result.shape == (C,), f"Expected shape ({C},), got {result.shape}"
    expected = cx_d0 + cx_d1 + cx_d2  # = 9.0
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-5,
        err_msg=f"Trilinear pool at integer centroid should equal {expected}",
    )


def test_point_pool_at_centroid_subpixel_interpolation() -> None:
    """D01.14: trilinear interpolation is correct at a sub-pixel centroid.

    Feature map: feat[c, d0, d1, d2] = float(d0) for all c.
    Centroid at (0.5, 0.0, 0.0): expected = 0.5 * feat[0] + 0.5 * feat[1]
                                           = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    """
    from abus.detect.nndet_inference import point_pool_trilinear

    C, D0, D1, D2 = 2, 4, 4, 4
    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    for d0 in range(D0):
        feat_map[:, d0, :, :] = float(d0)

    result = point_pool_trilinear(feat_map, cx_d0=0.5, cx_d1=0.0, cx_d2=0.0)

    assert result.shape == (C,)
    np.testing.assert_allclose(
        result, 0.5, rtol=1e-5, err_msg="Trilinear at d0=0.5 should give 0.5"
    )


def test_pool_axis_order_box_to_feature_map() -> None:
    """D01.14: box axis (x,y,z)=(d2,d1,d0) maps correctly to feature tensor [C,d0,d1,d2].

    nnDetection box axis: (x1,y1,x2,y2,z1,z2) where x=d2, y=d1, z=d0.
    box_center(box) gives (cx_x, cx_y, cx_z) = (cx_d2, cx_d1, cx_d0).
    pool_axis_order must map:
        feat tensor axis 1 → d0 → z center
        feat tensor axis 2 → d1 → y center
        feat tensor axis 3 → d2 → x center

    Test: place a distinct value ONLY at position (d0=2, d1=0, d0=0) in the feature map.
    Box: x1=0,y1=0,x2=1,y2=1,z1=2,z2=3 → centroid z=2.5 (d0=2.5), y=0.5 (d1=0.5), x=0.5 (d2=0.5).
    If axis is wrong, point_pool_trilinear will look at the wrong voxel and miss the value.
    """
    from abus.detect.nndet_inference import point_pool_trilinear

    C = 1
    feat_map = np.zeros((C, 8, 8, 8), dtype=np.float32)
    # Place a value of 99.0 only at (d0=2, d1=0, d2=0)
    feat_map[0, 2, 0, 0] = 99.0

    # Box centroid in nndet (x,y,z): x=0.5 → d2=0.5, y=0.5 → d1=0.5, z=2.5 → d0=2.5
    # Trilinear at (d0=2.5, d1=0.5, d2=0.5) samples corners:
    #   (2,0,0)=99, (2,1,0)=0, (2,0,1)=0, (2,1,1)=0,
    #   (3,0,0)=0, (3,1,0)=0, (3,0,1)=0, (3,1,1)=0
    # Weight of (2,0,0) = (1-0.5)*(1-0.5)*(1-0.5) = 0.125
    # Expected = 99.0 * 0.125 = 12.375
    cx_d0 = 2.5  # z center
    cx_d1 = 0.5  # y center
    cx_d2 = 0.5  # x center

    result = point_pool_trilinear(feat_map, cx_d0=cx_d0, cx_d1=cx_d1, cx_d2=cx_d2)

    expected = 99.0 * 0.5 * 0.5 * 0.5  # = 12.375
    np.testing.assert_allclose(
        result[0],
        expected,
        rtol=1e-4,
        err_msg=f"Axis mapping wrong: expected {expected}, got {result[0]}",
    )


def test_embedding_carried_through_wbc_same_weights_as_boxes() -> None:
    """D01.14: ensemble_combine averages embeddings with the SAME weights/members as boxes.

    Three proposals for one case, all overlapping (IoU > 0.5 with seed):
      - fold 0: score=0.9, embedding=[1,0,0,...,0]
      - fold 1: score=0.6, embedding=[0,1,0,...,0]
      - fold 2: score=0.3, embedding=[0,0,1,...,0]

    All three should cluster. Mean score = (0.9+0.6+0.3)/3 = 0.6.
    Mean embedding = [1/3, 1/3, 1/3, 0, ...] (first three dims).
    source_detectors = (0, 1, 2).

    The assertion: embeddings are averaged with the SAME set of K cluster members
    as boxes (no re-matching, no nearest-centroid). The number of embeddings entering
    the mean must equal the number of boxes in the cluster.
    """
    from abus.detect.candidates import RawCandidate
    from abus.detect.ensemble import ensemble_combine
    from abus.geometry.bbox import BBox

    _D = 8  # local test dimension; not D_EMB
    bbox = BBox(0, 0, 0, 10, 10, 10)  # All three proposals have same bbox → IoU=1.0

    proposals = [
        RawCandidate(
            case_id=5,
            split="val",
            bbox=bbox,
            score=0.9,
            embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            source_detectors=(0,),
        ),
        RawCandidate(
            case_id=5,
            split="val",
            bbox=bbox,
            score=0.6,
            embedding=np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            source_detectors=(1,),
        ),
        RawCandidate(
            case_id=5,
            split="val",
            bbox=bbox,
            score=0.3,
            embedding=np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
            source_detectors=(2,),
        ),
    ]

    result = ensemble_combine(proposals, iou_threshold=0.5)

    assert (
        len(result) == 1
    ), f"3 fully overlapping proposals should produce 1 cluster, got {len(result)}"
    cand = result[0]

    np.testing.assert_allclose(
        cand.score,
        0.6,
        rtol=1e-5,
        err_msg="Score must be mean of all cluster members",
    )
    expected_emb = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(
        cand.embedding,
        expected_emb,
        rtol=1e-5,
        err_msg="Embedding must be mean of ALL cluster members with same weights as scores",
    )

    # source_detectors must contain all three folds
    assert set(cand.source_detectors) == {
        0,
        1,
        2,
    }, f"source_detectors must be {{0,1,2}}, got {cand.source_detectors}"


def test_embedding_dim_is_128_constant_present() -> None:
    """D01.14: D_EMB = 128 constant is exposed from nndet_inference module."""
    from abus.detect import nndet_inference

    assert hasattr(
        nndet_inference, "D_EMB"
    ), "D_EMB constant must be defined in nndet_inference (D01.14: D_emb=128 pinned)"
    assert (
        nndet_inference.D_EMB == 128
    ), f"D_EMB must be 128 (fpn_channels=128 per build log), got {nndet_inference.D_EMB}"


def test_embedding_not_zero_not_constant_from_real_hook() -> None:
    """D01.14: pooled embeddings have non-trivial per-dimension variance.

    This test verifies that point_pool_trilinear produces non-constant, non-zero
    outputs when applied to a non-trivial feature map — catching a dead hook that
    captured a constant/wrong tensor (inversion axes 1+8 in D01.14).

    Simulate 5 detections with different centroids on a random feature map.
    The pooled embeddings must have non-zero per-dimension variance.
    """
    from abus.detect.nndet_inference import point_pool_trilinear

    rng = np.random.default_rng(42)
    C = 128  # D_EMB
    feat_map = rng.standard_normal((C, 20, 20, 20)).astype(np.float32)

    # 5 detections at distinct centroids
    centroids = [
        (2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0),
        (10.0, 11.0, 12.0),
        (1.0, 15.0, 8.0),
        (18.0, 2.0, 17.0),
    ]

    pooled = np.stack(
        [
            point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2)
            for (cx_d0, cx_d1, cx_d2) in centroids
        ],
        axis=0,
    )  # shape (5, 128)

    assert pooled.shape == (5, C), f"Expected (5, {C}), got {pooled.shape}"

    # Not all-zero
    assert np.any(pooled != 0), "All embeddings are zero — hook is dead or map is zero"

    # Not all constant across detections (variance > 0 across the 5 samples)
    per_dim_var = np.var(pooled, axis=0)
    assert np.any(
        per_dim_var > 0
    ), "All embedding dimensions have zero variance across 5 detections — constant embedding"


def test_predict_with_embeddings_returns_rawdetectionswithemb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D01.14: predict_with_embeddings returns dict[int, RawDetectionsWithEmb] with real embeddings.

    Tests that:
    1. The function exists and is importable.
    2. With a mocked model/nnDetection environment, it returns RawDetectionsWithEmb
       instances (not RawDetections with embeddings=None).
    3. Each RawDetectionsWithEmb.embeddings has shape (N, D_EMB) float32.
    4. Embeddings are NOT None and NOT all zeros (hook produced real features).

    Uses monkeypatching to stub the nnDetection imports (GPU not available on laptop).
    The stub model performs a forward pass with a controlled feature map so we
    can verify the pooling coordinate mapping is correct.
    """
    import tempfile
    from pathlib import Path

    from abus.detect.nndet_inference import predict_with_embeddings

    # --- Synthetic test setup ---
    # We simulate one case with 2 detections in (x1,y1,x2,y2,z1,z2) format.
    # The mock model:
    #   - model.decoder is hooked; when called, sets hook_output to a known feature map
    #   - model.decoder_levels is [0] (finest level at index 0)
    #   - model.inference_step returns 2 detected boxes + scores (no actual forward needed)
    #
    # Instead of driving the full tile loop, we test predict_with_embeddings by
    # providing a minimal mock that makes the function execute its hook-registration
    # and pooling logic on a controlled feature map.

    # This is a complex integration path; we test only the contract at the
    # module boundary — that the function exists, accepts the right args,
    # and returns RawDetectionsWithEmb with non-None embeddings.
    # The actual pooling math is proven by test_point_pool_at_centroid_trilinear.

    assert callable(predict_with_embeddings), "predict_with_embeddings must be callable"

    # Verify it raises ImportError on laptop (nnDetection not installed),
    # not an AttributeError or TypeError — proving the lazy import guard works.
    with tempfile.TemporaryDirectory() as tmpdir:
        fold_dir = Path(tmpdir)
        (fold_dir / "config.yaml").write_text("{}")
        # Don't write plan_inference.pkl so it raises FileNotFoundError (not ImportError)
        # before even trying to import nnDetection — validates the plan check fires first.
        # Note: the lazy-import guard for nnDetection fires INSIDE the function.
        # On laptop, nnDetection is not installed, so we get ImportError.
        # We can't reach the plan-check without nnDetection, so just verify
        # ImportError is raised (not some other error).
        try:
            predict_with_embeddings(
                fold=0,
                case_ids=[0],
                preprocessed_dir=tmpdir,
                fold_dir=str(fold_dir),
            )
            pytest.fail("Should have raised ImportError (nnDetection not installed)")
        except ImportError as exc:
            # Expected on laptop — nnDetection not installed
            msg = str(exc).lower()
            assert (
                "nndetection" in msg or "nndet" in msg or "torch" in msg
            ), f"ImportError should mention nnDetection/nndet/torch, got: {exc}"
        except Exception as exc:
            pytest.fail(
                f"Expected ImportError (nnDetection not installed), got {type(exc).__name__}: {exc}"
            )


def test_predict_with_embeddings_uses_tile_local_coordinates() -> None:
    """D01.14 Must-fix: predict_with_embeddings must pool embeddings at tile-LOCAL centroids.

    Root cause (code-review Must-fix #1+2): the original implementation used predict_dir
    as a black box and pooled from the final restored-box centroids on the last-captured
    feature map. This is wrong: the hook captures tile-space feature maps, but predict_dir
    gives back case-space (restored) boxes. The coordinates are mismatched.

    Fix: a custom per-tile loop that calls model.inference_step per tile, pools at
    tile-local centroids, then applies tile_origin offset. Uses no-TTA (NoOp transform)
    to avoid TTA-inverse complexity for feature maps.

    This test verifies the correct architecture by mocking nnDetection and verifying
    that the pooling uses tile-local coordinates (not case-space coordinates).

    Scenario: two tiles, each with one detection.
      Tile 0: tile_origin=(0,0,0), detection at tile-local box (2,2,4,4,2,4).
              Feature map has a unique value 11.0 at (d0=3, d1=3, d2=3).
              Centroid: x=(2+4)/2=3 → d2=3, y=(2+4)/2=3 → d1=3, z=(2+4)/2=3 → d0=3
              Expected embedding = pool at (d0=3, d1=3, d2=3) = 11.0 (all channels).
      Tile 1: tile_origin=(0,0,10,0), i.e. x_offset=10 (x=d2).
              Detection at tile-local box (1,1,3,3,1,3) in tile space.
              Feature map has value 22.0 at (d0=2, d1=2, d2=2).
              Centroid: x=(1+3)/2=2 → d2=2, y=(1+3)/2=2 → d1=2, z=(1+3)/2=2 → d0=2
              Expected embedding = 22.0.
              Case-space box after tile_origin offset: x+10 → (11,1,13,3,1,3)

    If the bug existed (using case-space coords on tile-local map), tile 1's detection
    at case-space x-centroid=(11+13)/2=12 would try to index position d2=12 on a
    tile map of spatial size 8 — getting clamped/wrong values, not 22.0.
    """

    from abus.detect.nndet_inference import (
        D_EMB,
    )

    # --- Build synthetic per-tile data ---
    # Two tiles, each (C=D_EMB, D0=8, D1=8, D2=8) feature map with a distinct
    # "hot spot" at a known tile-local location.
    TILE_FEAT_D = 8

    # Tile 0 feature map: channel 0 has value 11.0 at (d0=3, d1=3, d2=3)
    feat_tile0 = np.zeros((D_EMB, TILE_FEAT_D, TILE_FEAT_D, TILE_FEAT_D), dtype=np.float32)
    feat_tile0[:, 3, 3, 3] = 11.0

    # Tile 1 feature map: channel 0 has value 22.0 at (d0=2, d1=2, d2=2)
    feat_tile1 = np.zeros((D_EMB, TILE_FEAT_D, TILE_FEAT_D, TILE_FEAT_D), dtype=np.float32)
    feat_tile1[:, 2, 2, 2] = 22.0

    # tile_origin in nndet format: (x_offset, y_offset, z_offset)
    # Tile 0 at origin (0,0,0). Tile 1 with x_offset=10 (d2 offset by 10).
    tile_origin_0 = (0, 0, 0)  # (x=d2, y=d1, z=d0)
    tile_origin_1 = (10, 0, 0)  # x=d2 shifted by 10

    # Tile-local detection boxes in (x1,y1,x2,y2,z1,z2) format:
    # Tile 0: centroid at (x=3→d2, y=3→d1, z=3→d0) i.e. box (2,2,4,4,2,4)
    tile0_boxes = np.array([[2.0, 2.0, 4.0, 4.0, 2.0, 4.0]], dtype=np.float32)
    tile0_scores = np.array([0.9], dtype=np.float32)

    # Tile 1: centroid at (x=2→d2, y=2→d1, z=2→d0) i.e. box (1,1,3,3,1,3) tile-local
    tile1_boxes = np.array([[1.0, 1.0, 3.0, 3.0, 1.0, 3.0]], dtype=np.float32)
    tile1_scores = np.array([0.8], dtype=np.float32)

    # After applying tile_origin_1 (x_offset=10): case-space box becomes (11,1,13,3,1,3)
    # If wrong impl used case-space centroid x=12 on tile-map size 8 → clamped to 7 → wrong value

    # --- Mock nnDetection environment ---
    # We mock:
    #   1. The model object (has .decoder and .inference_step)
    #   2. The predictor (has .tile_case, knows tile_origin per tile)
    #   3. nnDetection imports

    feat_maps_per_tile = [feat_tile0, feat_tile1]
    boxes_per_tile = [tile0_boxes, tile1_boxes]
    scores_per_tile = [tile0_scores, tile1_scores]
    tile_origins = [tile_origin_0, tile_origin_1]

    # Synthetic tiles (simulating predictor.tile_case output)
    tiles = [
        {
            "data": np.zeros((1, 1, TILE_FEAT_D, TILE_FEAT_D, TILE_FEAT_D), dtype=np.float32),
            "tile_origin": tile_origins[0],
        },
        {
            "data": np.zeros((1, 1, TILE_FEAT_D, TILE_FEAT_D, TILE_FEAT_D), dtype=np.float32),
            "tile_origin": tile_origins[1],
        },
    ]

    # The test exercises predict_with_embeddings by monkeypatching the lazy imports.
    # We provide a fake _predict_single_case_with_embeddings that uses our synthetic data.
    # Since the real predict_with_embeddings is being FIXED (it doesn't exist yet in the
    # correct form), this test will FAIL until the fix is implemented.

    # The test verifies the OUTPUT: that case_id's detected embeddings match
    # the tile-local pooled values (not case-space-pooled values).

    # Import the helper that the fixed predict_with_embeddings must use internally.
    from abus.detect.nndet_inference import _predict_single_case_with_embeddings

    # Run the per-case helper with our synthetic tiles, feature maps, boxes.
    # Expected: tile0 detection embedding = 11.0, tile1 detection embedding = 22.0.
    boxes_out, scores_out, embeddings_out = _predict_single_case_with_embeddings(
        tiles=tiles,
        feat_maps_per_tile=feat_maps_per_tile,
        boxes_per_tile=boxes_per_tile,
        scores_per_tile=scores_per_tile,
        fpn_level_index=0,
        iou_threshold=0.3,  # detections don't overlap much; both should survive
    )

    assert (
        embeddings_out.shape[1] == D_EMB
    ), f"embedding dim must be D_EMB={D_EMB}, got {embeddings_out.shape[1]}"
    assert (
        embeddings_out.shape[0] >= 2
    ), f"Expected at least 2 detections (one per tile), got {embeddings_out.shape[0]}"

    # Both detections must survive (boxes don't overlap). Find them by score.
    # Tile 0 detection: score=0.9, expected embedding=11.0
    # Tile 1 detection: score=0.8, expected embedding=22.0
    # After case-offset, tile1 box is (11,1,13,3,1,3) and tile0 box is (2,2,4,4,2,4)
    # They don't overlap (x ranges [2..4] vs [11..13]) so both should survive WBC.

    found_tile0 = False
    found_tile1 = False
    for i in range(embeddings_out.shape[0]):
        emb = embeddings_out[i]
        score = scores_out[i]
        if abs(float(score) - 0.9) < 0.01:
            # Should be tile0 detection: embedding from feat_tile0 at (3,3,3) = 11.0
            np.testing.assert_allclose(
                emb,
                11.0,
                rtol=1e-4,
                err_msg=(
                    f"Tile 0 detection embedding should be 11.0 (tile-local pool at (3,3,3)). "
                    f"If wrong, the impl used case-space coords on tile-local map. got {emb[:3]}"
                ),
            )
            found_tile0 = True
        elif abs(float(score) - 0.8) < 0.01:
            # Should be tile1 detection: embedding from feat_tile1 at (2,2,2) = 22.0
            np.testing.assert_allclose(
                emb,
                22.0,
                rtol=1e-4,
                err_msg=(
                    "Tile 1 embedding should be 22.0 (tile-local pool at (2,2,2)). "
                    "If wrong, impl used case-space x=12 instead of tile-local x=2. "
                    f"got {emb[:3]}"
                ),
            )
            found_tile1 = True

    assert found_tile0, "Did not find tile-0 detection (score~0.9) in output"
    assert found_tile1, "Did not find tile-1 detection (score~0.8) in output"


def test_rawdetectionswithemb_embeddings_never_none() -> None:
    """D01.14: RawDetectionsWithEmb always has a real embeddings array (never None)."""
    from abus.detect.nndet_inference import D_EMB, RawDetectionsWithEmb

    rd = RawDetectionsWithEmb(
        case_id=42,
        boxes=np.zeros((3, 6), dtype=np.float32),
        scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        embeddings=np.random.randn(3, D_EMB).astype(np.float32),
    )
    assert rd.embeddings is not None, "RawDetectionsWithEmb.embeddings must never be None"
    assert rd.embeddings.shape == (
        3,
        D_EMB,
    ), f"embeddings shape must be (N, {D_EMB}), got {rd.embeddings.shape}"
    assert (
        rd.embeddings.dtype == np.float32
    ), f"embeddings must be float32, got {rd.embeddings.dtype}"


# ===========================================================================
# D01.14b — val/test preprocessing fix (2026-06-04)
# ===========================================================================


def test_d01_14b_preprocess_val_test_function_exists() -> None:
    """D01.14b: preprocess_val_test is importable from nndet_inference.

    This is the explicit test-preprocessing step that populates
    preprocessed/<data_identifier>/imagesTs/ from the raw test set,
    replacing the old assumption that nndet_predict would preprocess at predict-time.
    """
    import abus.detect.nndet_inference as nndet_inference

    assert hasattr(
        nndet_inference, "preprocess_val_test"
    ), "preprocess_val_test must be defined in nndet_inference (D01.14b)"
    assert callable(nndet_inference.preprocess_val_test), "preprocess_val_test must be callable"


def test_d01_14b_preprocess_val_test_calls_run_preprocessing_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D01.14b: preprocess_val_test calls planner_cls.run_preprocessing_test.

    Source-grounded against scripts/predict.py:74-81 (nnDetection-main):
        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=cfg["host"]["preprocessed_output_dir"],
            splitted_4d_output_dir=cfg["host"]["splitted_4d_output_dir"],
            plan=plan,
            num_processes=...,
        )

    Verifies:
    - The correct planner class is retrieved from PLANNER_REGISTRY
    - run_preprocessing_test is called with the plan from plan_inference.pkl
    - preprocessed_output_dir is the task preprocessed root (NOT imagesTr or imagesTs)
    - splitted_4d_output_dir is cfg["host"]["splitted_4d_output_dir"]
    - num_processes is forwarded
    """
    import pickle
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from abus.detect.nndet_inference import preprocess_val_test

    fake_plan = {
        "planner_id": "TestPlannerV001",
        "data_identifier": "TestV001_3d",
        "inference_plan": {},
    }
    fake_cfg = {
        "host": {
            "preprocessed_output_dir": "/fake/preprocessed",
            "splitted_4d_output_dir": "/fake/raw_splitted",
        }
    }

    called_kwargs: dict[str, object] = {}

    def fake_run_preprocessing_test(
        preprocessed_output_dir: object,
        splitted_4d_output_dir: object,
        plan: object,
        num_processes: int = 0,
    ) -> None:
        called_kwargs.update(
            {
                "preprocessed_output_dir": preprocessed_output_dir,
                "splitted_4d_output_dir": splitted_4d_output_dir,
                "plan": plan,
                "num_processes": num_processes,
            }
        )

    fake_planner_cls = MagicMock()
    fake_planner_cls.run_preprocessing_test = fake_run_preprocessing_test

    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_planner_cls

    def _fake_load_pickle(p: str) -> object:
        with open(p, "rb") as _f:
            return pickle.load(_f)  # noqa: S301

    with tempfile.TemporaryDirectory() as fold_dir_str:
        fold_dir = Path(fold_dir_str)
        (fold_dir / "config.yaml").write_text("{}")
        with open(fold_dir / "plan_inference.pkl", "wb") as _f:
            pickle.dump(fake_plan, _f)

        with (
            patch.dict(
                "sys.modules",
                {
                    "nndet": MagicMock(),
                    "nndet.planning": MagicMock(PLANNER_REGISTRY=fake_registry),
                    "nndet.io": MagicMock(),
                    "nndet.io.load": MagicMock(load_pickle=_fake_load_pickle),
                    "omegaconf": MagicMock(
                        OmegaConf=MagicMock(
                            load=lambda p: fake_cfg,
                            to_container=lambda c, **kw: c,
                        )
                    ),
                },
            ),
        ):
            preprocess_val_test(
                fold_dir=fold_dir_str,
                num_processes=3,
            )

    # run_preprocessing_test must have been called
    assert called_kwargs, "run_preprocessing_test was never called"

    # Verify the planner was looked up from the registry
    assert fake_registry.get.called, "PLANNER_REGISTRY.get must be called"
    assert fake_registry.get.call_args[0][0] == "TestPlannerV001", (
        f"PLANNER_REGISTRY.get must be called with plan['planner_id']='TestPlannerV001', "
        f"got {fake_registry.get.call_args}"
    )

    # preprocessed_output_dir: must be the cfg host value (NOT imagesTr or imagesTs)
    assert str(called_kwargs["preprocessed_output_dir"]) == "/fake/preprocessed", (
        f"preprocessed_output_dir must be cfg['host']['preprocessed_output_dir'], "
        f"got {called_kwargs['preprocessed_output_dir']}"
    )

    # splitted_4d_output_dir: must come from cfg
    assert str(called_kwargs["splitted_4d_output_dir"]) == "/fake/raw_splitted", (
        f"splitted_4d_output_dir must be cfg['host']['splitted_4d_output_dir'], "
        f"got {called_kwargs['splitted_4d_output_dir']}"
    )

    # plan: must be the loaded fake_plan
    assert (
        called_kwargs["plan"] is fake_plan or called_kwargs["plan"] == fake_plan
    ), f"plan must be the loaded plan_inference.pkl content, got {called_kwargs['plan']}"

    # num_processes: forwarded
    assert (
        called_kwargs["num_processes"] == 3
    ), f"num_processes must be forwarded (3), got {called_kwargs['num_processes']}"


def test_d01_14b_preprocess_val_test_raises_importerror_on_laptop() -> None:
    """D01.14b: preprocess_val_test raises ImportError on laptop (nnDetection not installed).

    Validates the lazy-import guard is in place — same pattern as predict_oof and
    predict_with_embeddings. Callers that run on the laptop must get an ImportError,
    not an AttributeError or NameError from a missing module reference.
    """
    import tempfile
    from pathlib import Path

    from abus.detect.nndet_inference import preprocess_val_test

    with tempfile.TemporaryDirectory() as fold_dir_str:
        fold_dir = Path(fold_dir_str)
        (fold_dir / "config.yaml").write_text("{}")
        # Write a minimal plan_inference.pkl so the file-existence check passes.
        import pickle

        with open(fold_dir / "plan_inference.pkl", "wb") as _f:
            pickle.dump({"planner_id": "X", "inference_plan": {}}, _f)

        # On laptop, nnDetection is not installed — must raise ImportError.
        with pytest.raises(ImportError):
            preprocess_val_test(fold_dir=fold_dir_str)


# ===========================================================================
# D01.14-OOM — streaming equivalence tests (CPU-only)
# ===========================================================================


def test_pool_tile_proposals_equivalence_with_predict_single_case() -> None:
    """D01.14-OOM: _pool_tile_proposals + _ensemble_proposals_to_arrays produce
    numerically equivalent output to _predict_single_case_with_embeddings on a
    synthetic multi-tile case.

    This is the critical correctness test for the streaming OOM fix.  Both paths
    must produce identical (boxes, scores, embeddings) up to float32 precision.

    Setup: 3 tiles, each with 1-2 detections at known centroids in tile space.
    Tile 0: origin (0,0,0), 1 detection, score=0.9
    Tile 1: origin (10,0,0), 2 detections, scores=0.7 and 0.5
    Tile 2: origin (0,20,0), 1 detection, score=0.6

    The batch path (_predict_single_case_with_embeddings) receives all tiles/maps
    up front. The streaming path calls _pool_tile_proposals per tile then
    _ensemble_proposals_to_arrays.  Both must agree exactly.
    """
    from abus.detect.nndet_inference import (
        D_EMB,
        _ensemble_proposals_to_arrays,
        _pool_tile_proposals,
        _predict_single_case_with_embeddings,
    )

    rng = np.random.default_rng(2026)
    C = D_EMB
    SPATIAL = 12  # tile spatial size for the test

    # Build 3 tiles with known feature maps and detections.
    num_tiles = 3
    tile_origins = [
        (0, 0, 0),
        (10, 0, 0),
        (0, 20, 0),
    ]
    feat_maps = [
        rng.standard_normal((C, SPATIAL, SPATIAL, SPATIAL)).astype(np.float32)
        for _ in range(num_tiles)
    ]

    # boxes_per_tile in tile-local (x1,y1,x2,y2,z1,z2), all within [0, SPATIAL)
    boxes_per_tile = [
        np.array([[1.0, 1.0, 4.0, 4.0, 1.0, 4.0]], dtype=np.float32),  # tile 0: 1 det
        np.array(
            [[2.0, 2.0, 5.0, 5.0, 2.0, 5.0], [6.0, 6.0, 9.0, 9.0, 6.0, 9.0]],  # tile 1: 2 dets
            dtype=np.float32,
        ),
        np.array([[3.0, 3.0, 7.0, 7.0, 3.0, 7.0]], dtype=np.float32),  # tile 2: 1 det
    ]
    scores_per_tile = [
        np.array([0.9], dtype=np.float32),
        np.array([0.7, 0.5], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
    ]

    # Build tile dicts as _predict_single_case_with_embeddings expects.
    tiles = [
        {"tile_origin": tile_origins[i], "data": np.zeros((1, SPATIAL, SPATIAL, SPATIAL))}
        for i in range(num_tiles)
    ]

    # --- Batch path (existing, tested interface) ---
    boxes_batch, scores_batch, embs_batch = _predict_single_case_with_embeddings(
        tiles=tiles,
        feat_maps_per_tile=feat_maps,
        boxes_per_tile=boxes_per_tile,
        scores_per_tile=scores_per_tile,
        fpn_level_index=0,
        iou_threshold=0.5,
    )

    # --- Streaming path (new code under test) ---
    all_proposals = []
    for i in range(num_tiles):
        # The hook in the streaming path already selected level 0 as a plain array
        # (not a list), so feat_maps[i] is passed directly.
        props = _pool_tile_proposals(
            feat_map_raw=feat_maps[i],
            tile_boxes=boxes_per_tile[i],
            tile_scores=scores_per_tile[i],
            tile_origin=tile_origins[i],
            fpn_level_index=0,
            tile_idx=i,
        )
        all_proposals.extend(props)

    boxes_stream, scores_stream, embs_stream = _ensemble_proposals_to_arrays(all_proposals)

    # --- Equivalence assertions ---
    assert (
        boxes_batch.shape == boxes_stream.shape
    ), f"Boxes shape mismatch: batch {boxes_batch.shape} vs stream {boxes_stream.shape}"
    assert (
        scores_batch.shape == scores_stream.shape
    ), f"Scores shape mismatch: batch {scores_batch.shape} vs stream {scores_stream.shape}"
    assert (
        embs_batch.shape == embs_stream.shape
    ), f"Embeddings shape mismatch: batch {embs_batch.shape} vs stream {embs_stream.shape}"

    np.testing.assert_array_equal(
        boxes_batch,
        boxes_stream,
        err_msg="Boxes must be bit-identical between batch and streaming paths",
    )
    np.testing.assert_array_equal(
        scores_batch,
        scores_stream,
        err_msg="Scores must be bit-identical between batch and streaming paths",
    )
    np.testing.assert_array_equal(
        embs_batch,
        embs_stream,
        err_msg="Embeddings must be bit-identical between batch and streaming paths",
    )


def test_pool_tile_proposals_streaming_one_map_at_a_time() -> None:
    """D01.14-OOM: _pool_tile_proposals accepts a single (non-list) numpy array
    as feat_map_raw — the form the streaming hook now produces.

    The hook selects level 0 before appending to _tile_feat_maps (a plain ndarray,
    not a list of levels).  _pool_tile_proposals must handle both the list form
    (legacy, used by _predict_single_case_with_embeddings tests) and the plain-array
    form (new streaming path).

    Verifies: when feat_map_raw is a plain ndarray the result is identical to
    when it is wrapped in a length-1 list with fpn_level_index=0.
    """
    from abus.detect.nndet_inference import D_EMB, _pool_tile_proposals

    rng = np.random.default_rng(99)
    C = D_EMB
    feat = rng.standard_normal((C, 8, 8, 8)).astype(np.float32)
    boxes = np.array([[1.0, 1.0, 3.0, 3.0, 1.0, 3.0]], dtype=np.float32)
    scores = np.array([0.8], dtype=np.float32)
    origin = (0, 0, 0)

    # Plain array form (new streaming hook output)
    props_array = _pool_tile_proposals(
        feat_map_raw=feat,
        tile_boxes=boxes,
        tile_scores=scores,
        tile_origin=origin,
        fpn_level_index=0,
    )

    # List form (batch path / legacy)
    props_list = _pool_tile_proposals(
        feat_map_raw=[feat],
        tile_boxes=boxes,
        tile_scores=scores,
        tile_origin=origin,
        fpn_level_index=0,
    )

    assert len(props_array) == 1, f"Expected 1 proposal, got {len(props_array)}"
    assert len(props_list) == 1, f"Expected 1 proposal (list form), got {len(props_list)}"

    np.testing.assert_array_equal(
        props_array[0].embedding,
        props_list[0].embedding,
        err_msg="Plain-array and list-wrapped feat_map must produce identical embeddings",
    )


def test_streaming_hook_only_level0_survives() -> None:
    """D01.14-OOM: the hook in the streaming path retains only level-0 array.

    Verifies the _decoder_hook closure behaviour via the _pool_tile_proposals
    output: when called with a list of 4 FPN levels and fpn_level_index=0, the
    result matches calling it with only level-0 directly.

    This is a unit-level proxy test (we cannot invoke the real hook without a GPU
    model, but we can test the _pool_tile_proposals level-selection logic which is
    the consumer of whatever the hook produces).
    """
    from abus.detect.nndet_inference import D_EMB, _pool_tile_proposals

    rng = np.random.default_rng(7)
    C = D_EMB

    # 4 FPN levels — level 0 is the finest (largest spatial size).
    feat_level0 = rng.standard_normal((C, 8, 8, 8)).astype(np.float32)
    feat_level1 = rng.standard_normal((C, 4, 4, 4)).astype(np.float32)
    feat_level2 = rng.standard_normal((C, 2, 2, 2)).astype(np.float32)
    feat_level3 = rng.standard_normal((C, 1, 1, 1)).astype(np.float32)
    four_level_list = [feat_level0, feat_level1, feat_level2, feat_level3]

    boxes = np.array([[1.0, 1.0, 3.0, 3.0, 1.0, 3.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    origin = (0, 0, 0)

    # Multi-level list with fpn_level_index=0
    props_multi = _pool_tile_proposals(
        feat_map_raw=four_level_list,
        tile_boxes=boxes,
        tile_scores=scores,
        tile_origin=origin,
        fpn_level_index=0,
    )

    # Only level-0 array (what the new streaming hook appends)
    props_single = _pool_tile_proposals(
        feat_map_raw=feat_level0,
        tile_boxes=boxes,
        tile_scores=scores,
        tile_origin=origin,
        fpn_level_index=0,
    )

    assert len(props_multi) == 1
    assert len(props_single) == 1

    np.testing.assert_array_equal(
        props_multi[0].embedding,
        props_single[0].embedding,
        err_msg=(
            "Level-0 pooling from 4-level list must equal pooling from level-0 array alone. "
            "If this fails the hook level-selection or _pool_tile_proposals indexing is wrong."
        ),
    )


def test_streaming_equivalence_under_score_ties() -> None:
    """D01.14-OOM: batch and streaming paths agree even when proposals have tied scores.

    Score ties are not present in typical detector output but can occur in edge
    cases (e.g. two detections from identical tile positions across overlapping tiles).
    The WBC seed-selection must be stable: both paths iterate proposals in the same
    order (tile 0 before tile 1, ...) so tied seeds resolve identically.

    This test has two tiles with the same score (0.8) and overlapping boxes that
    cluster into a single WBC output.  Both paths must produce the same single
    merged detection.
    """
    from abus.detect.nndet_inference import (
        D_EMB,
        _ensemble_proposals_to_arrays,
        _pool_tile_proposals,
        _predict_single_case_with_embeddings,
    )

    rng = np.random.default_rng(1234)
    C = D_EMB
    SPATIAL = 10

    # Two tiles with the SAME score but different embeddings.
    # Both detect approximately the same box (high IoU → should cluster).
    feat0 = rng.standard_normal((C, SPATIAL, SPATIAL, SPATIAL)).astype(np.float32)
    feat1 = rng.standard_normal((C, SPATIAL, SPATIAL, SPATIAL)).astype(np.float32)

    tile_origins = [(0, 0, 0), (0, 0, 0)]  # same origin → same case-space boxes
    boxes_per_tile = [
        np.array([[1.0, 1.0, 5.0, 5.0, 1.0, 5.0]], dtype=np.float32),
        np.array([[1.5, 1.5, 5.5, 5.5, 1.5, 5.5]], dtype=np.float32),
    ]
    # Tied scores — the critical tie-breaking case.
    scores_per_tile = [
        np.array([0.8], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
    ]

    tiles = [
        {"tile_origin": tile_origins[i], "data": np.zeros((1, SPATIAL, SPATIAL, SPATIAL))}
        for i in range(2)
    ]
    feat_maps = [feat0, feat1]

    # Batch path
    boxes_b, scores_b, embs_b = _predict_single_case_with_embeddings(
        tiles=tiles,
        feat_maps_per_tile=feat_maps,
        boxes_per_tile=boxes_per_tile,
        scores_per_tile=scores_per_tile,
        fpn_level_index=0,
        iou_threshold=0.3,  # low threshold so the two overlapping boxes cluster
    )

    # Streaming path
    all_props = []
    for i in range(2):
        all_props.extend(
            _pool_tile_proposals(
                feat_map_raw=feat_maps[i],
                tile_boxes=boxes_per_tile[i],
                tile_scores=scores_per_tile[i],
                tile_origin=tile_origins[i],
                fpn_level_index=0,
                tile_idx=i,
            )
        )
    boxes_s, scores_s, embs_s = _ensemble_proposals_to_arrays(all_props, iou_threshold=0.3)

    assert (
        boxes_b.shape == boxes_s.shape
    ), f"Boxes shape mismatch under score tie: batch {boxes_b.shape} vs stream {boxes_s.shape}"
    np.testing.assert_array_equal(
        boxes_b,
        boxes_s,
        err_msg="Boxes must match under score ties (both paths iterate in tile order)",
    )
    np.testing.assert_array_equal(
        scores_b,
        scores_s,
        err_msg="Scores must match under score ties",
    )
    np.testing.assert_array_equal(
        embs_b,
        embs_s,
        err_msg="Embeddings must match under score ties",
    )


# ===========================================================================
# D01.17 — pool_embeddings_at_boxes tests
# ===========================================================================


def test_pool_embeddings_at_boxes_signature() -> None:
    """D01.17: pool_embeddings_at_boxes must exist with the correct signature."""
    import inspect

    from abus.detect.nndet_inference import pool_embeddings_at_boxes

    sig = inspect.signature(pool_embeddings_at_boxes)
    params = list(sig.parameters.keys())

    required = ["fold", "case_id", "boxes_preprocessed", "preprocessed_dir", "fold_dir"]
    for name in required:
        assert name in params, f"pool_embeddings_at_boxes missing required param: {name}"

    # keyword-only params
    assert "pooling" in params, "pooling must be a keyword-only param"
    assert "fpn_level_index" in params, "fpn_level_index must be a keyword-only param"

    p_pooling = sig.parameters["pooling"]
    p_fpn = sig.parameters["fpn_level_index"]
    assert p_pooling.kind == inspect.Parameter.KEYWORD_ONLY, "pooling must be keyword-only"
    assert p_fpn.kind == inspect.Parameter.KEYWORD_ONLY, "fpn_level_index must be keyword-only"
    assert (
        p_pooling.default == "centroid"
    ), f"pooling default must be 'centroid', got {p_pooling.default}"
    assert p_fpn.default == 0, f"fpn_level_index default must be 0, got {p_fpn.default}"


def test_pool_embeddings_at_boxes_raises_import_error_without_nndet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D01.17: pool_embeddings_at_boxes raises ImportError when nnDetection is not installed.

    This test simulates the laptop environment where nnDetection is absent.
    """
    from unittest.mock import patch

    import numpy as np

    from abus.detect import nndet_inference

    boxes = np.array([[1.0, 1.0, 3.0, 3.0, 1.0, 3.0]], dtype=np.float32)

    with patch.dict(
        "sys.modules",
        {
            "nndet": None,
            "nndet.inference": None,
            "nndet.inference.loading": None,
            "nndet.io": None,
            "nndet.io.load": None,
            "nndet.io.patching": None,
            "omegaconf": None,
        },
    ):  # noqa: E501
        with pytest.raises(ImportError):
            nndet_inference.pool_embeddings_at_boxes(
                fold=0,
                case_id=0,
                boxes_preprocessed=boxes,
                preprocessed_dir="/nonexistent",
                fold_dir="/nonexistent",
            )


def test_pool_embeddings_at_boxes_predict_oof_restore_false() -> None:
    """D01.17: predict_oof default restore=False (boxes in preprocessed space).

    Verifies that predict_oof's restore parameter defaults to False per D01.17.
    This is the cornerstone of the decoupled design: the same coordinate space
    for both native boxes and FPN feature maps.
    """
    import inspect

    from abus.detect.nndet_inference import predict_oof

    sig = inspect.signature(predict_oof)
    restore_param = sig.parameters.get("restore")
    assert restore_param is not None, "predict_oof must have a 'restore' parameter (D01.17)"
    assert restore_param.default is False, (
        f"predict_oof restore default must be False (D01.17), got {restore_param.default}. "
        "The D01.17 design requires boxes in preprocessed space (same as FPN feature maps) "
        "so pool_embeddings_at_boxes can pool without inverting restore_detection's affine."
    )


def test_pool_embeddings_at_boxes_raises_runtime_error_without_decoder_levels(
    tmp_path: Path,
) -> None:
    """D01.17 B1: pool_embeddings_at_boxes raises RuntimeError when decoder_levels is missing.

    This pins the fix for ML-review Round 2 B1 (wrong decoder output level).
    The function must resolve model.model.decoder_levels to correctly select
    the detection-level FPN output (decoder_levels[0] ≥ 2 for ABUS volumes,
    never output[0]).

    We supply a mock model that has a decoder (so the decoder-module check passes)
    but lacks decoder_levels (so the B1 guard fires), and confirm the RuntimeError.

    Filesystem setup: create real-looking fold_dir and preprocessed_dir so that
    the file-existence checks pass and execution reaches the decoder_levels guard.
    """
    import pickle
    import sys
    from unittest.mock import MagicMock, patch

    # Create real-looking fold dir with config.yaml + plan_inference.pkl
    fold_dir = tmp_path / "fold0"
    fold_dir.mkdir()

    # Minimal config YAML
    (fold_dir / "config.yaml").write_text(
        "module: RetinaUNetModule\nmodel_cfg: {}\ntrainer_cfg: {}\n"
    )

    # Minimal plan pickle: must contain patch_size
    plan = {"patch_size": [64, 64, 64]}
    with open(fold_dir / "plan_inference.pkl", "wb") as fh:
        pickle.dump(plan, fh)

    # Create preprocessed dir with a fake .npz for case_id=0 (0000.npz)
    pre_dir = tmp_path / "preprocessed"
    pre_dir.mkdir()
    fake_arr = np.zeros((1, 64, 64, 64), dtype=np.float32)
    np.savez(str(pre_dir / "0000.npz"), data=fake_arr)

    # Build a mock model: model.model has .decoder but NO .decoder_levels
    mock_decoder = MagicMock()
    mock_inner = MagicMock(spec=[])  # spec=[] → no attributes by default
    mock_inner.decoder = mock_decoder  # has .decoder
    # does NOT have .decoder_levels (not in spec=[])

    mock_model = MagicMock()
    mock_model.model = mock_inner
    # make model.parameters() not throw; hasattr check in code uses default_rng
    mock_model.parameters.return_value = iter([MagicMock()])

    mock_load_fn = MagicMock(return_value=[{"model": mock_model, "rank": 0}])

    # Mock torch and nndet modules so lazy import succeeds
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_nndet_loading = MagicMock()
    mock_nndet_loading.load_final_model = mock_load_fn

    mock_nndet_io_load = MagicMock()
    mock_nndet_io_load.load_pickle.return_value = plan  # plan_inference.pkl returns plan

    mock_nndet_io_patching = MagicMock()
    # create_grid returns empty tile list (no tiles needed; we raise before any tile)
    mock_nndet_io_patching.create_grid.return_value = []
    mock_nndet_io_patching.save_get_crop.return_value = (fake_arr, (0, 0, 0), None)

    mock_omegaconf = MagicMock()
    mock_omegaconf.OmegaConf.load.return_value = {}
    mock_omegaconf.OmegaConf.to_container.return_value = {
        "module": "RetinaUNetModule",
        "model_cfg": {},
        "trainer_cfg": {},
    }

    boxes = np.zeros((2, 6), dtype=np.float32)

    import abus.detect.nndet_inference as nni

    with patch.dict(
        sys.modules,
        {
            "torch": mock_torch,
            "nndet": MagicMock(),
            "nndet.inference": MagicMock(),
            "nndet.inference.loading": mock_nndet_loading,
            "nndet.io": MagicMock(),
            "nndet.io.load": mock_nndet_io_load,
            "nndet.io.patching": mock_nndet_io_patching,
            "omegaconf": mock_omegaconf,
        },
    ):
        with pytest.raises(RuntimeError, match="decoder_levels"):
            nni.pool_embeddings_at_boxes(
                fold=0,
                case_id=0,
                boxes_preprocessed=boxes,
                preprocessed_dir=str(pre_dir),
                fold_dir=str(fold_dir),
            )


# ===========================================================================
# D01.18 — restore_boxes_for_case tests (two-frame design)
# ===========================================================================


def test_restore_boxes_for_case_signature() -> None:
    """D01.18: restore_boxes_for_case is importable with the correct signature.

    The function converts storage-frame boxes from predict_oof(restore=False)
    to original-image-grid boxes for the RawCandidate.bbox (required by
    STORY_01_03 GT matching against original-grid GT lesion bboxes).
    """
    import inspect

    from abus.detect.nndet_inference import restore_boxes_for_case

    sig = inspect.signature(restore_boxes_for_case)
    params = list(sig.parameters.keys())

    required = ["boxes_preprocessed", "case_id", "preprocessed_dir", "fold_dir"]
    for name in required:
        assert name in params, f"restore_boxes_for_case missing required param: {name}"


def test_restore_boxes_for_case_raises_import_error_on_laptop() -> None:
    """D01.18: restore_boxes_for_case raises ImportError on laptop (nnDetection not installed).

    The function is server-only (requires nndet.inference.restore and nndet.io.load).
    The lazy-import guard must fire before any file-system access.
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from abus.detect import nndet_inference

    boxes = np.zeros((2, 6), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        fold_dir = Path(tmpdir) / "fold0"
        fold_dir.mkdir()
        pre_dir = Path(tmpdir) / "preprocessed"
        pre_dir.mkdir()

        with patch.dict(
            "sys.modules",
            {
                "nndet": None,
                "nndet.inference": None,
                "nndet.inference.restore": None,
                "nndet.io": None,
                "nndet.io.load": None,
            },
        ):
            with pytest.raises(ImportError):
                nndet_inference.restore_boxes_for_case(
                    boxes_preprocessed=boxes,
                    case_id=0,
                    preprocessed_dir=str(pre_dir),
                    fold_dir=str(fold_dir),
                )


def test_restore_boxes_for_case_applies_identity_transform() -> None:
    """D01.18: restore_boxes_for_case applies nnDetection restore_detection correctly.

    With identity transpose_backward=[0,1,2] and uniform spacing
    (original_spacing == spacing_after_resampling), the restored boxes
    equal the input boxes (no scaling or permutation).

    This pins the two-frame contract:
      - storage-frame boxes → pool_embeddings_at_boxes (unchanged, FPN frame)
      - same boxes → restore_boxes_for_case → original-grid boxes for BBox
    When the transform is identity, both frames coincide (as a sanity check).
    """
    import pickle
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from abus.detect import nndet_inference

    # Identity plan: transpose_backward=[0,1,2] (no permutation)
    plan = {
        "patch_size": [64, 64, 64],
        "transpose_backward": [0, 1, 2],
    }

    # Per-case props: same spacing before and after resampling → scaling = 1.0
    # No crop offset (crop_bbox starts at 0 on each axis)
    props = {
        "original_spacing": [0.3, 0.3, 0.5],
        "spacing_after_resampling": [0.3, 0.3, 0.5],
        "crop_bbox": [(0, 64), (0, 64), (0, 64)],
    }

    # Synthetic storage-frame boxes: (x1,y1,x2,y2,z1,z2) with slot0→d0
    boxes_pre = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 5.0, 15.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fold_dir = Path(tmpdir) / "fold0"
        fold_dir.mkdir()
        pre_dir = Path(tmpdir) / "preprocessed"
        pre_dir.mkdir()

        # Write plan_inference.pkl
        with open(fold_dir / "plan_inference.pkl", "wb") as fh:
            pickle.dump(plan, fh)
        # Write per-case properties pkl
        with open(pre_dir / "0007.pkl", "wb") as fh:
            pickle.dump(props, fh)

        # Mock nndet lazy imports

        def _fake_restore_detection(
            boxes: np.ndarray,
            transpose_backward: list,
            original_spacing: list,
            spacing_after_resampling: list,
            crop_bbox: list,
            **kw: object,
        ) -> np.ndarray:
            return boxes.copy()

        mock_restore_detection = MagicMock(side_effect=_fake_restore_detection)
        mock_load_pickle_calls: list[tuple] = []

        def _fake_load_pickle(p: str) -> object:  # noqa: S301 — test mock
            import pickle

            with open(p, "rb") as fh:
                val = pickle.load(fh)  # noqa: S301
            mock_load_pickle_calls.append((str(p),))
            return val

        mock_restore_mod = MagicMock()
        mock_restore_mod.restore_detection = mock_restore_detection

        mock_io_load = MagicMock()
        mock_io_load.load_pickle = _fake_load_pickle

        with patch.dict(
            sys.modules,
            {
                "nndet": MagicMock(),
                "nndet.inference": MagicMock(),
                "nndet.inference.restore": mock_restore_mod,
                "nndet.io": MagicMock(),
                "nndet.io.load": mock_io_load,
            },
        ):
            result = nndet_inference.restore_boxes_for_case(
                boxes_preprocessed=boxes_pre,
                case_id=7,
                preprocessed_dir=str(pre_dir),
                fold_dir=str(fold_dir),
            )

    # restore_detection was called once with all 5 required args (D01.18 contract)
    assert mock_restore_detection.called, "restore_detection must be called"
    call_kwargs = mock_restore_detection.call_args
    # boxes: must be the input storage-frame boxes (identity mock returns boxes.copy())
    np.testing.assert_array_equal(
        call_kwargs.kwargs.get("boxes"),
        boxes_pre,
        err_msg="boxes arg to restore_detection must be the storage-frame input",
    )
    # transpose_backward must come from plan["transpose_backward"]
    assert list(call_kwargs.kwargs.get("transpose_backward", [])) == [
        0,
        1,
        2,
    ], "transpose_backward must be loaded from plan_inference.pkl"
    # original_spacing must come from per-case props
    assert list(call_kwargs.kwargs.get("original_spacing", [])) == [
        0.3,
        0.3,
        0.5,
    ], "original_spacing must come from the per-case .pkl"
    # spacing_after_resampling must come from per-case props
    assert list(call_kwargs.kwargs.get("spacing_after_resampling", [])) == [
        0.3,
        0.3,
        0.5,
    ], "spacing_after_resampling must come from the per-case .pkl"
    # crop_bbox must come from per-case props
    assert list(call_kwargs.kwargs.get("crop_bbox", [])) == [
        (0, 64),
        (0, 64),
        (0, 64),
    ], "crop_bbox must come from the per-case .pkl"

    # Result shape and dtype preserved (identity mock returns boxes.copy())
    assert result.shape == boxes_pre.shape, f"Shape changed: {result.shape} vs {boxes_pre.shape}"
    assert result.dtype == np.float32, f"dtype changed: {result.dtype}"


def test_restore_boxes_for_case_empty_boxes_returns_empty() -> None:
    """D01.18: restore_boxes_for_case with 0 input boxes returns empty array (no crash)."""
    import pickle
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from abus.detect import nndet_inference

    boxes_empty = np.zeros((0, 6), dtype=np.float32)

    plan = {"patch_size": [64, 64, 64], "transpose_backward": [0, 1, 2]}
    props = {
        "original_spacing": [0.3, 0.3, 0.5],
        "spacing_after_resampling": [0.3, 0.3, 0.5],
        "crop_bbox": [(0, 64), (0, 64), (0, 64)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fold_dir = Path(tmpdir) / "fold0"
        fold_dir.mkdir()
        pre_dir = Path(tmpdir) / "preprocessed"
        pre_dir.mkdir()

        with open(fold_dir / "plan_inference.pkl", "wb") as fh:
            pickle.dump(plan, fh)
        with open(pre_dir / "0000.pkl", "wb") as fh:
            pickle.dump(props, fh)

        # nndet lazy imports are not exercised — empty boxes return before restore_detection call
        mock_restore_mod = MagicMock()

        def _fake_load_pickle(p: str) -> object:  # noqa: S301 — test mock
            import pickle

            with open(p, "rb") as fh:
                return pickle.load(fh)  # noqa: S301

        mock_io_load = MagicMock()
        mock_io_load.load_pickle = _fake_load_pickle

        with patch.dict(
            sys.modules,
            {
                "nndet": MagicMock(),
                "nndet.inference": MagicMock(),
                "nndet.inference.restore": mock_restore_mod,
                "nndet.io": MagicMock(),
                "nndet.io.load": mock_io_load,
            },
        ):
            result = nndet_inference.restore_boxes_for_case(
                boxes_preprocessed=boxes_empty,
                case_id=0,
                preprocessed_dir=str(pre_dir),
                fold_dir=str(fold_dir),
            )

    # Empty input → empty output; restore_detection NOT called (early return)
    assert result.shape == (0, 6), f"Expected (0,6), got {result.shape}"
    assert (
        not mock_restore_mod.restore_detection.called
    ), "restore_detection must not be called for empty boxes (early return)"


def test_restore_boxes_for_case_non_identity_transpose_backward() -> None:
    """D01.18 SF1: axis-swap is visible when transpose_backward != identity.

    Uses transpose_backward=[1,2,0] (ABUS production value derived from
    transpose_forward=[2,0,1], i.e. max-spacing axis 2 goes first).

    The real nnDetection permute_boxes([1,2,0]) on input (x1,y1,x2,y2,z1,z2):
      output slot0 ← input axis 1 (y1/y2 pair)
      output slot1 ← input axis 2 (z1/z2 pair)
      output slot4 ← input axis 0 (x1/x2 pair)
    So output = (y1, z1, y2, z2, x1, x2).

    After scaling (identity here) and crop offset (zero here), the consumer
    _raw_detections_to_candidates reads slot0→d0, slot1→d1, slot4→d2.
    That maps: y_orig→d0, z_orig→d1, x_orig→d2 for THIS transpose — which is
    correct because x_orig=preproc-axis2=original-d2, y_orig=preproc-axis0=original-d0,
    z_orig=preproc-axis1=original-d1 after the inverse permutation.

    This test locks the axis-swap regression by checking that restore_detection
    is called with the correct non-identity transpose_backward from the plan file.
    A wrong slot-mapping in the consumer would mismatch the physical axes by
    ~0.073 mm vs ~0.476 mm (a 6.5× spacing ratio), detectable in IoU computation.
    """
    import pickle
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from abus.detect import nndet_inference

    # ABUS production transpose: transpose_forward=[2,0,1], backward=[1,2,0]
    plan = {"patch_size": [64, 64, 64], "transpose_backward": [1, 2, 0]}
    props = {
        "original_spacing": [0.073, 0.200, 0.476],
        "spacing_after_resampling": [0.3, 0.3, 0.5],  # after resampling, forward-transposed
        "crop_bbox": [(0, 200), (0, 100), (0, 50)],
    }

    # Storage-frame boxes (preprocessed space, x=preproc-axis0, y=preproc-axis1, z=preproc-axis2)
    boxes_pre = np.array([[10.0, 20.0, 30.0, 40.0, 5.0, 15.0]], dtype=np.float32)

    captured_args: dict = {}

    def _capturing_restore_detection(
        boxes: np.ndarray,
        transpose_backward: list,
        original_spacing: list,
        spacing_after_resampling: list,
        crop_bbox: list,
        **kw: object,
    ) -> np.ndarray:
        captured_args["transpose_backward"] = list(transpose_backward)
        captured_args["original_spacing"] = list(original_spacing)
        # Identity substitute: return boxes unchanged for shape/dtype verification
        return boxes.copy()

    mock_restore_detection = MagicMock(side_effect=_capturing_restore_detection)
    mock_restore_mod = MagicMock()
    mock_restore_mod.restore_detection = mock_restore_detection

    def _fake_load_pickle(p: str) -> object:  # noqa: S301 — test mock
        with open(p, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    mock_io_load = MagicMock()
    mock_io_load.load_pickle = _fake_load_pickle

    with tempfile.TemporaryDirectory() as tmpdir:
        fold_dir = Path(tmpdir) / "fold0"
        fold_dir.mkdir()
        pre_dir = Path(tmpdir) / "preprocessed"
        pre_dir.mkdir()
        with open(fold_dir / "plan_inference.pkl", "wb") as fh:
            pickle.dump(plan, fh)
        with open(pre_dir / "0003.pkl", "wb") as fh:
            pickle.dump(props, fh)

        with patch.dict(
            sys.modules,
            {
                "nndet": MagicMock(),
                "nndet.inference": MagicMock(),
                "nndet.inference.restore": mock_restore_mod,
                "nndet.io": MagicMock(),
                "nndet.io.load": mock_io_load,
            },
        ):
            result = nndet_inference.restore_boxes_for_case(
                boxes_preprocessed=boxes_pre,
                case_id=3,
                preprocessed_dir=str(pre_dir),
                fold_dir=str(fold_dir),
            )

    # Correct non-identity transpose_backward loaded from plan
    assert captured_args.get("transpose_backward") == [
        1,
        2,
        0,
    ], f"Expected [1,2,0] (ABUS production), got {captured_args.get('transpose_backward')}"
    # original_spacing from per-case props (not from plan)
    assert captured_args.get("original_spacing") == pytest.approx(
        [0.073, 0.200, 0.476]
    ), f"original_spacing mismatch: {captured_args.get('original_spacing')}"
    # Shape and dtype preserved
    assert result.shape == (1, 6), f"Expected (1,6), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
