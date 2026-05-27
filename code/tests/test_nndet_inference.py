"""Tests for src/abus/detect/nndet_inference.py (STORY_01_02, D01.9).

Test plan (from story spec — ASC-01_02.6, Local validation plan):

  test_parse_predictions_dir_basic
    Write a synthetic <case>_boxes.pkl into a tempdir; parse_predictions_dir
    returns a dict keyed by int case_id with boxes/scores arrays of the
    expected shape and dtype.

  test_parse_predictions_dir_filename_parser
    The parser correctly extracts the case_id integer from nnDetection 0.1's
    filename convention (e.g. 0042_boxes.pkl → 42). Malformed filenames are
    skipped with a logged warning, not silently absorbed.

  test_parse_predictions_dir_missing_embeddings
    If the pickle schema has no embeddings field, RawDetections.embeddings is
    None and the parser does not crash.

  test_parse_predictions_dir_zero_detections
    An empty boxes array (no detections for a case) is handled correctly;
    the entry is still present in the returned dict with shape (0, 6) boxes.

  test_parse_predictions_dir_multiple_cases
    Multiple *_boxes.pkl files in the same directory are all parsed; the
    returned dict has one entry per file.

  test_parse_predictions_dir_schema_modes
    The schema used by nnDetection 0.1 (dict with 'boxes', 'scores' and
    optionally 'embeddings' keys) is parsed correctly. Also verifies that the
    boxes dtype is float32 and scores dtype is float32/float64.

  test_raw_detections_is_frozen
    RawDetections is a frozen dataclass — field reassignment raises.

  test_parse_predictions_dir_no_files
    An empty directory returns an empty dict (not a crash).

CPU-only; no nnDetection import required.

Schema note (nnDetection 0.1, commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
  The nnDetection 0.1 predict path writes per-case pickle files. Based on the
  nnDetection source structure and the D01.9 diagnostics, the output is a dict
  containing:
    'boxes':   np.ndarray shape (N, 6) float32 — bounding boxes in nnDetection's
               internal format (z1, y1, x1, z2, y2, x2) on the resampled grid
    'scores':  np.ndarray shape (N,) float32/float64 — detection confidence scores
    'embeddings': np.ndarray shape (N, D) float32 — backbone-pooled features
                  (may be absent)
  This schema is what the synthetic fixtures in these tests mirror. If the actual
  server schema differs, the tests will fail first (Risk #5 mitigation, D01.9).
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

NNDET_SCHEMA_VERSION = "0.1_97a58f3"  # commit short hash, for traceability


def _write_boxes_pkl(
    path: Path,
    n_dets: int = 3,
    emb_dim: int = 16,
    include_embeddings: bool = True,
    boxes_override: np.ndarray | None = None,
    scores_override: np.ndarray | None = None,
) -> dict:
    """Write a synthetic nnDetection-style *_boxes.pkl and return the dict written.

    Schema matches nnDetection 0.1's output:
      {'boxes': np.ndarray (N,6) float32,
       'scores': np.ndarray (N,) float32,
       'embeddings': np.ndarray (N, D) float32}   ← optional
    """
    if boxes_override is not None:
        boxes = boxes_override.astype(np.float32)
    else:
        # Synthetic boxes: (z1, y1, x1, z2, y2, x2) style, resampled grid coords
        boxes = np.arange(n_dets * 6, dtype=np.float32).reshape(n_dets, 6)
        boxes[:, 3:] += 1.0  # ensure max > min

    if scores_override is not None:
        scores = scores_override.astype(np.float32)
    else:
        scores = np.linspace(0.9, 0.5, n_dets, dtype=np.float32)

    payload: dict = {"boxes": boxes, "scores": scores}
    if include_embeddings:
        embeddings = np.random.default_rng(0).random((n_dets, emb_dim), dtype=np.float32)
        payload["embeddings"] = embeddings

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    return payload


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_basic
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_basic():
    """parse_predictions_dir returns a dict keyed by int case_id with correct shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        n_dets = 4
        emb_dim = 16
        written = _write_boxes_pkl(tmp / "0042_boxes.pkl", n_dets=n_dets, emb_dim=emb_dim)

        result = parse_predictions_dir(tmpdir)

    assert isinstance(result, dict), "result must be a dict"
    assert 42 in result, "case_id 42 (from filename 0042_boxes.pkl) must be in result"

    rd = result[42]
    assert isinstance(rd, RawDetections)
    assert rd.case_id == 42
    assert rd.boxes.shape == (n_dets, 6), f"Expected shape ({n_dets}, 6), got {rd.boxes.shape}"
    assert rd.scores.shape == (n_dets,), f"Expected shape ({n_dets},), got {rd.scores.shape}"
    assert rd.embeddings is not None
    assert rd.embeddings.shape == (n_dets, emb_dim)
    assert rd.embeddings.dtype == np.float32

    # Verify values round-trip correctly
    np.testing.assert_allclose(rd.boxes, written["boxes"], rtol=1e-5)
    np.testing.assert_allclose(rd.scores, written["scores"], rtol=1e-5)


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_filename_parser
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_filename_parser_strips_leading_zeros():
    """case_id integer is extracted from the zero-padded stem: 0042_boxes.pkl -> 42."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Various zero-padded filenames
        _write_boxes_pkl(tmp / "0000_boxes.pkl", n_dets=1)
        _write_boxes_pkl(tmp / "0007_boxes.pkl", n_dets=1)
        _write_boxes_pkl(tmp / "0099_boxes.pkl", n_dets=1)
        _write_boxes_pkl(tmp / "0199_boxes.pkl", n_dets=1)

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == {0, 7, 99, 199}


def test_parse_predictions_dir_malformed_filename_skipped(caplog):
    """Malformed filenames (non-integer stems) are skipped with a warning, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_boxes_pkl(tmp / "0001_boxes.pkl", n_dets=2)
        # Malformed: cannot parse integer from stem
        malformed = tmp / "badname_boxes.pkl"
        _write_boxes_pkl(malformed, n_dets=1)

        with caplog.at_level(logging.WARNING):
            result = parse_predictions_dir(tmpdir)

    # Only the valid case is in the result
    assert 1 in result
    assert (
        len(result) == 1
    ), f"Malformed filename should be skipped, got keys: {list(result.keys())}"
    # A warning was logged about the skipped file
    assert any(
        "badname" in record.message.lower() or "skip" in record.message.lower()
        for record in caplog.records
    ), "Expected a warning about the malformed filename"


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_missing_embeddings
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_missing_embeddings():
    """If the pickle has no 'embeddings' key, RawDetections.embeddings is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_boxes_pkl(tmp / "0010_boxes.pkl", n_dets=3, include_embeddings=False)

        result = parse_predictions_dir(tmpdir)

    assert 10 in result
    rd = result[10]
    assert (
        rd.embeddings is None
    ), f"embeddings should be None when missing from pickle, got {rd.embeddings}"
    # Other fields still correct
    assert rd.boxes.shape == (3, 6)
    assert rd.scores.shape == (3,)


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_zero_detections
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_zero_detections():
    """An empty boxes array (0 detections) is handled; entry present with shape (0,6)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        empty_boxes = np.zeros((0, 6), dtype=np.float32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        _write_boxes_pkl(
            tmp / "0005_boxes.pkl",
            n_dets=0,
            boxes_override=empty_boxes,
            scores_override=empty_scores,
        )

        result = parse_predictions_dir(tmpdir)

    assert 5 in result
    rd = result[5]
    assert rd.boxes.shape == (0, 6), f"Expected shape (0,6), got {rd.boxes.shape}"
    assert rd.scores.shape == (0,), f"Expected shape (0,), got {rd.scores.shape}"


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_multiple_cases
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_multiple_cases():
    """Multiple *_boxes.pkl files are all parsed; one entry per file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        case_ids = [0, 1, 42, 99]
        for cid in case_ids:
            _write_boxes_pkl(tmp / f"{cid:04d}_boxes.pkl", n_dets=2)

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == set(
        case_ids
    ), f"Expected case_ids {case_ids}, got {list(result.keys())}"
    for cid in case_ids:
        assert result[cid].case_id == cid
        assert result[cid].boxes.shape == (2, 6)


# ---------------------------------------------------------------------------
# test_parse_predictions_dir_no_files
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_no_files():
    """An empty directory returns an empty dict (not a crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = parse_predictions_dir(tmpdir)

    assert result == {}, f"Empty directory should return empty dict, got {result}"


# ---------------------------------------------------------------------------
# test_raw_detections_is_frozen
# ---------------------------------------------------------------------------


def test_raw_detections_is_frozen():
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
# test_parse_predictions_dir_schema_modes
# ---------------------------------------------------------------------------


def test_parse_predictions_dir_boxes_dtype_coerced_to_float32():
    """Boxes and scores arrays are returned as float32 regardless of pickle dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write with float64 deliberately (should be coerced to float32)
        payload = {
            "boxes": np.ones((3, 6), dtype=np.float64),
            "scores": np.array([0.9, 0.8, 0.7], dtype=np.float64),
        }
        with open(tmp / "0020_boxes.pkl", "wb") as f:
            pickle.dump(payload, f)

        result = parse_predictions_dir(tmpdir)

    rd = result[20]
    assert rd.boxes.dtype == np.float32, f"Expected float32 boxes, got {rd.boxes.dtype}"
    assert rd.scores.dtype == np.float32, f"Expected float32 scores, got {rd.scores.dtype}"


def test_parse_predictions_dir_non_pickle_files_ignored():
    """Non-.pkl files in the directory are ignored (only *_boxes.pkl are processed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_boxes_pkl(tmp / "0001_boxes.pkl", n_dets=2)
        # Write some other files that should be ignored
        (tmp / "0002_scores.npy").write_bytes(b"dummy")
        (tmp / "README.txt").write_text("hello")
        (tmp / "0003_embeddings.pkl").write_bytes(b"dummy")  # not *_boxes.pkl

        result = parse_predictions_dir(tmpdir)

    assert set(result.keys()) == {
        1
    }, f"Only 0001_boxes.pkl should be parsed, got {list(result.keys())}"
