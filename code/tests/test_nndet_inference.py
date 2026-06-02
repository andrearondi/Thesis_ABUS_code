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
      - restore     = True  (restore boxes to original image space)
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
    assert called_kwargs["restore"] is True, "restore must be True (boxes in original space)"
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
