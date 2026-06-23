"""Tests for scripts/inspect_detector_quality.py (STORY_01_02 detector quality inspection).

Schema ground truth (server-verified 2026-06-23, commit 97a58f3):

val_results/results_boxes.json
  FLAT dict. Keys include:
    "FROC_score_IoU_0.10" ... "FROC_score_IoU_0.90"
    "mAP_IoU_0.10_0.50_0.05_MaxDet_100"
    "AP_IoU_0.10_MaxDet_100" ... "AP_IoU_0.50_MaxDet_100"
  VALUES ARE STRINGS (including the literal "nan"). Coerce with float().
  Keys prefixed "0_" are per-class duplicates — skip or de-dup.

sweep/<param>.json  (5 files)
  Dict {param_label: {"state": str, "overwrite": dict, "scores": str}}
  "scores" field is a PYTHON-REPR string of a dict (ast.literal_eval, NOT json.loads).

val_predictions/<case>_boxes.pkl
  Consolidated dict schema same as nnDetection output:
    "pred_boxes": np.ndarray (N, 6) float32
    "pred_scores": np.ndarray (N,) float32
    "pred_labels": np.ndarray (N,) float32
    "restore": bool
    "original_size_of_raw_data": np.ndarray (3,) int64
    "itk_origin": np.ndarray (3,) float64
    "itk_spacing": np.ndarray (3,) float64
    "itk_direction": np.ndarray (9,) float64

train.log
  Plain text. Lines containing "training_epoch_end" mark epoch completions.
  Also contains loss values per epoch.

Python 3.8-compatible: no X | Y union syntax, no walrus in assignments.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable.  The script itself does sys.path manipulation
# but for tests we add the scripts dir explicitly.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Fixture helpers: build synthetic on-disk fold layout
# ---------------------------------------------------------------------------


def _make_results_boxes_json(
    tmpdir: Path,
    *,
    froc_10: str = "0.7200",
    froc_50: str = "0.5100",
    froc_90: str = "0.3000",
    ap_10: str = "0.6500",
    ap_50: str = "0.4100",
    include_nan: bool = False,
    include_class_prefixed: bool = True,
) -> Path:
    """Write a synthetic val_results/results_boxes.json replicating the real server schema.

    Values are STRINGS (real server schema — always coerce with float()).
    Keys prefixed '0_' are per-class duplicates.
    """
    val_results_dir = tmpdir / "val_results"
    val_results_dir.mkdir(parents=True, exist_ok=True)

    d: dict = {
        "FROC_score_IoU_0.10": froc_10,
        "FROC_score_IoU_0.20": "0.6800",
        "FROC_score_IoU_0.30": "0.6200",
        "FROC_score_IoU_0.40": "0.5700",
        "FROC_score_IoU_0.50": froc_50,
        "FROC_score_IoU_0.60": "0.4500",
        "FROC_score_IoU_0.70": "0.4000",
        "FROC_score_IoU_0.80": "0.3500",
        "FROC_score_IoU_0.90": froc_90,
        "mAP_IoU_0.10_0.50_0.05_MaxDet_100": "0.5500",
        "AP_IoU_0.10_MaxDet_100": ap_10,
        "AP_IoU_0.20_MaxDet_100": "0.6000",
        "AP_IoU_0.30_MaxDet_100": "0.5600",
        "AP_IoU_0.40_MaxDet_100": "0.4800",
        "AP_IoU_0.50_MaxDet_100": ap_50,
    }

    if include_nan:
        d["FROC_score_IoU_0.80"] = "nan"
        d["AP_IoU_0.40_MaxDet_100"] = "nan"

    if include_class_prefixed:
        # Per-class duplicate keys (should be ignored / de-duped)
        d["0_FROC_score_IoU_0.10"] = froc_10
        d["0_AP_IoU_0.10_MaxDet_100"] = ap_10

    path = val_results_dir / "results_boxes.json"
    path.write_text(json.dumps(d))
    return path


def _make_sweep_json(tmpdir: Path, param_name: str = "model_score_thresh") -> Path:
    """Write a synthetic sweep/<param>.json replicating the real server schema.

    The 'scores' field is a PYTHON-REPR string (ast.literal_eval), NOT JSON.
    """
    sweep_dir = tmpdir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Real schema: {param_label: {"state": str, "overwrite": dict, "scores": str}}
    # The scores value is repr() of a dict, e.g. "{'FROC_score_IoU_0.10': 0.72}"
    scores_dict = {
        "FROC_score_IoU_0.10": 0.72,
        "FROC_score_IoU_0.50": 0.51,
    }
    # Crucially: repr() produces a Python-literal string, NOT valid JSON
    scores_repr = repr(scores_dict)

    data = {
        "0.05": {
            "state": "finished",
            "overwrite": {param_name: 0.05},
            "scores": scores_repr,
        },
        "0.10": {
            "state": "finished",
            "overwrite": {param_name: 0.10},
            "scores": repr({"FROC_score_IoU_0.10": 0.68, "FROC_score_IoU_0.50": 0.49}),
        },
        "0.50": {
            "state": "finished",
            "overwrite": {param_name: 0.50},
            "scores": repr({"FROC_score_IoU_0.10": 0.55, "FROC_score_IoU_0.50": 0.42}),
        },
    }

    path = sweep_dir / f"sweep_{param_name}.json"
    path.write_text(json.dumps(data))
    return path


def _write_val_prediction_pkl(
    tmpdir: Path,
    case_id_str: str,
    n_dets: int = 5,
) -> Path:
    """Write a synthetic val_predictions/<case>_boxes.pkl.

    Uses the same consolidated dict schema as nnDetection 0.1 output.
    Boxes in (x1,y1,x2,y2,z1,z2) float32.
    """
    val_pred_dir = tmpdir / "val_predictions"
    val_pred_dir.mkdir(parents=True, exist_ok=True)

    boxes = np.zeros((n_dets, 6), dtype=np.float32)
    for i in range(n_dets):
        boxes[i] = [
            float(i * 10),
            float(i * 5),
            float(i * 10 + 20),
            float(i * 5 + 10),
            float(i * 3),
            float(i * 3 + 15),
        ]

    pred_dict = {
        "pred_boxes": boxes,
        "pred_scores": np.linspace(0.9, 0.5, n_dets, dtype=np.float32),
        "pred_labels": np.zeros(n_dets, dtype=np.float32),
        "restore": False,
        "original_size_of_raw_data": np.array([865, 470, 348], dtype=np.int64),
        "itk_origin": np.zeros(3, dtype=np.float64),
        "itk_spacing": np.array([0.073, 0.200, 0.476], dtype=np.float64),
        "itk_direction": np.eye(3, dtype=np.float64).ravel(),
    }

    path = val_pred_dir / f"{case_id_str}_boxes.pkl"
    with open(path, "wb") as f:
        pickle.dump(pred_dict, f)
    return path


def _make_train_log(
    tmpdir: Path,
    n_epochs: int = 60,
    plateau_at: int = 50,
) -> Path:
    """Write a synthetic train.log replicating the real nnDetection loguru format.

    Real format (commit 97a58f3, loguru default):
      "2026-01-01 00:01:00 | INFO | nndet.training.trainer:training_epoch_end:311
       | training_epoch_end | Train loss reached: 0.1234"
    The parser looks for "training_epoch_end" on the line AND
    "Train loss reached: <float>" for the loss value.
    """
    lines = []
    for ep in range(1, n_epochs + 1):
        # Simulate decreasing loss with plateau
        if ep <= plateau_at:
            loss = max(0.05, 1.0 - ep * 0.017)
        else:
            loss = max(0.05, 1.0 - plateau_at * 0.017)  # plateaued
        lines.append(
            f"2026-01-01 00:{ep:02d}:00.000 | INFO     | "
            f"nndet.training.trainer:training_epoch_end:311 | "
            f"training_epoch_end | Train loss reached: {loss:.4f}"
        )

    path = tmpdir / "train.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_complete_fold_dir(
    base_dir: Path,
    fold_idx: int,
    n_val_cases: int = 3,
    n_epochs: int = 60,
) -> Path:
    """Create a complete synthetic fold directory with all required artifacts."""
    fold_dir = base_dir / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # val_results/results_boxes.json
    _make_results_boxes_json(fold_dir)

    # sweep files (at least one)
    _make_sweep_json(fold_dir, "model_score_thresh")

    # val_predictions (multiple cases)
    for i in range(n_val_cases):
        case_id_str = f"{(fold_idx * 20 + i):04d}"
        _write_val_prediction_pkl(fold_dir, case_id_str)

    # train.log
    _make_train_log(fold_dir, n_epochs=n_epochs)

    return fold_dir


# ===========================================================================
# Tests for parse_results_boxes_json
# ===========================================================================


class TestParseResultsBoxesJson:
    """Tests for the val_results/results_boxes.json parser."""

    def test_parses_froc_scores_as_float(self, tmp_path: Path) -> None:
        """Values are strings on disk; parser must coerce to float."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path, froc_10="0.7200")
        result = parse_results_boxes_json(str(results_path))
        assert "FROC_score_IoU_0.10" in result
        assert isinstance(result["FROC_score_IoU_0.10"], float)
        assert abs(result["FROC_score_IoU_0.10"] - 0.72) < 1e-6

    def test_parses_ap_scores_as_float(self, tmp_path: Path) -> None:
        """AP keys must also be coerced from strings to float."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path, ap_10="0.6500")
        result = parse_results_boxes_json(str(results_path))
        assert "AP_IoU_0.10_MaxDet_100" in result
        assert abs(result["AP_IoU_0.10_MaxDet_100"] - 0.65) < 1e-6

    def test_handles_nan_string_value(self, tmp_path: Path) -> None:
        """String 'nan' values must be coerced to float('nan'), not raise."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path, include_nan=True)
        result = parse_results_boxes_json(str(results_path))
        # The 'nan'-valued key should be present and be a NaN float
        assert "FROC_score_IoU_0.80" in result
        import math

        assert math.isnan(result["FROC_score_IoU_0.80"])

    def test_excludes_class_prefixed_keys(self, tmp_path: Path) -> None:
        """Keys prefixed '0_' are per-class duplicates and must be excluded."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path, include_class_prefixed=True)
        result = parse_results_boxes_json(str(results_path))
        # No key should start with '0_'
        for key in result:
            assert not key.startswith("0_"), f"Class-prefixed key leaked: {key}"

    def test_returns_all_froc_iou_keys(self, tmp_path: Path) -> None:
        """All FROC_score_IoU_X.XX keys (0.10-0.90) must be present."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path)
        result = parse_results_boxes_json(str(results_path))
        for iou in ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]:
            assert f"FROC_score_IoU_{iou}" in result, f"Missing FROC key for IoU={iou}"

    def test_returns_ap_keys(self, tmp_path: Path) -> None:
        """AP_IoU_X.XX_MaxDet_100 keys must be present."""
        from inspect_detector_quality import parse_results_boxes_json

        results_path = _make_results_boxes_json(tmp_path)
        result = parse_results_boxes_json(str(results_path))
        for iou in ["0.10", "0.20", "0.30", "0.40", "0.50"]:
            assert f"AP_IoU_{iou}_MaxDet_100" in result, f"Missing AP key for IoU={iou}"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError on a missing results_boxes.json."""
        from inspect_detector_quality import parse_results_boxes_json

        with pytest.raises(FileNotFoundError):
            parse_results_boxes_json(str(tmp_path / "nonexistent.json"))


# ===========================================================================
# Tests for parse_sweep_json
# ===========================================================================


class TestParseSweepJson:
    """Tests for the sweep/<param>.json parser.

    Critical: 'scores' field is a PYTHON-REPR string (ast.literal_eval), NOT JSON.
    """

    def test_parses_scores_with_ast_literal_eval(self, tmp_path: Path) -> None:
        """The 'scores' field must be parsed with ast.literal_eval, not json.loads."""
        from inspect_detector_quality import parse_sweep_json

        sweep_path = _make_sweep_json(tmp_path, "model_score_thresh")
        result = parse_sweep_json(str(sweep_path))
        # Must return a dict of {param_label: parsed_scores_dict}
        assert "0.05" in result
        scores = result["0.05"]["scores_parsed"]
        assert isinstance(scores, dict)
        assert "FROC_score_IoU_0.10" in scores
        assert isinstance(scores["FROC_score_IoU_0.10"], float)

    def test_json_loads_would_fail_on_scores_repr(self, tmp_path: Path) -> None:
        """Regression: json.loads must FAIL on the real repr() scores string.

        This test proves the real schema requires ast.literal_eval, not json.loads.
        """
        # The repr() of a Python dict uses single quotes — invalid JSON.
        scores_repr = repr({"FROC_score_IoU_0.10": 0.72})
        assert "'" in scores_repr  # repr uses single-quotes
        with pytest.raises(ValueError):  # json.JSONDecodeError is a subclass of ValueError
            json.loads(scores_repr)

    def test_parses_all_param_labels(self, tmp_path: Path) -> None:
        """All param_label entries in the sweep file must be returned."""
        from inspect_detector_quality import parse_sweep_json

        sweep_path = _make_sweep_json(tmp_path, "model_score_thresh")
        result = parse_sweep_json(str(sweep_path))
        assert set(result.keys()) == {"0.05", "0.10", "0.50"}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError on a missing sweep file."""
        from inspect_detector_quality import parse_sweep_json

        with pytest.raises(FileNotFoundError):
            parse_sweep_json(str(tmp_path / "nonexistent.json"))


# ===========================================================================
# Tests for parse_val_predictions_dir (reusing nndet_inference schema)
# ===========================================================================


class TestParseValPredictionsDir:
    """Tests for val_predictions/<case>_boxes.pkl parser.

    Must REUSE abus.detect.nndet_inference.parse_predictions_dir logic —
    NOT reinvent the wheel. The schema is the same consolidated dict format.
    """

    def test_parses_boxes_and_scores(self, tmp_path: Path) -> None:
        """Parser reads pred_boxes and pred_scores from the consolidated pkl.

        _write_val_prediction_pkl writes to tmp_path/val_predictions/<case>_boxes.pkl.
        parse_val_predictions_dir is called on the val_predictions/ subdirectory.
        """
        from inspect_detector_quality import parse_val_predictions_dir

        _write_val_prediction_pkl(tmp_path, "0004", n_dets=5)
        # val_predictions/ is created by the fixture helper
        result = parse_val_predictions_dir(str(tmp_path / "val_predictions"))
        assert 4 in result
        rd = result[4]
        assert rd.boxes.shape == (5, 6)
        assert rd.scores.shape == (5,)
        assert rd.boxes.dtype == np.float32
        assert rd.scores.dtype == np.float32

    def test_parses_multiple_cases(self, tmp_path: Path) -> None:
        """Multiple <case>_boxes.pkl files are all discovered."""
        from inspect_detector_quality import parse_val_predictions_dir

        for cid in ["0004", "0009", "0015"]:
            _write_val_prediction_pkl(tmp_path, cid)
        result = parse_val_predictions_dir(str(tmp_path / "val_predictions"))
        assert set(result.keys()) == {4, 9, 15}

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        """Empty val_predictions dir returns {} (no error)."""
        from inspect_detector_quality import parse_val_predictions_dir

        # Create an empty directory to search
        empty_dir = tmp_path / "val_predictions"
        empty_dir.mkdir()
        result = parse_val_predictions_dir(str(empty_dir))
        assert result == {}

    def test_top_n_scores_ordering(self, tmp_path: Path) -> None:
        """Top-N boxes by score are in descending score order."""
        from inspect_detector_quality import parse_val_predictions_dir

        _write_val_prediction_pkl(tmp_path, "0004", n_dets=10)
        result = parse_val_predictions_dir(str(tmp_path / "val_predictions"))
        rd = result[4]
        # The synthetic fixture generates linspace(0.9, 0.5) — check decreasing
        assert rd.scores[0] >= rd.scores[-1]


# ===========================================================================
# Tests for parse_train_log
# ===========================================================================


class TestParseTrainLog:
    """Tests for train.log convergence parser."""

    def test_counts_training_epoch_end_lines(self, tmp_path: Path) -> None:
        """Parser counts lines containing 'training_epoch_end'."""
        from inspect_detector_quality import parse_train_log

        log_path = _make_train_log(tmp_path, n_epochs=60)
        result = parse_train_log(str(log_path))
        assert result["epochs_completed"] == 60

    def test_incomplete_training_detected(self, tmp_path: Path) -> None:
        """Fewer than 60 epochs is flagged as incomplete."""
        from inspect_detector_quality import parse_train_log

        log_path = _make_train_log(tmp_path, n_epochs=42)
        result = parse_train_log(str(log_path))
        assert result["epochs_completed"] == 42
        assert result["training_complete"] is False

    def test_complete_training_flagged(self, tmp_path: Path) -> None:
        """60/60 epochs is flagged as complete."""
        from inspect_detector_quality import parse_train_log

        log_path = _make_train_log(tmp_path, n_epochs=60)
        result = parse_train_log(str(log_path))
        assert result["training_complete"] is True

    def test_extracts_loss_per_epoch(self, tmp_path: Path) -> None:
        """Loss values per epoch are extracted into a list."""
        from inspect_detector_quality import parse_train_log

        log_path = _make_train_log(tmp_path, n_epochs=60)
        result = parse_train_log(str(log_path))
        assert "loss_per_epoch" in result
        assert len(result["loss_per_epoch"]) == 60
        # Each element is a float
        assert all(isinstance(v, float) for v in result["loss_per_epoch"])

    def test_missing_log_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError for a missing train.log."""
        from inspect_detector_quality import parse_train_log

        with pytest.raises(FileNotFoundError):
            parse_train_log(str(tmp_path / "nonexistent.log"))

    def test_empty_log_returns_zero_epochs(self, tmp_path: Path) -> None:
        """An empty train.log is handled gracefully (0 epochs, incomplete)."""
        from inspect_detector_quality import parse_train_log

        log_path = tmp_path / "train.log"
        log_path.write_text("")
        result = parse_train_log(str(log_path))
        assert result["epochs_completed"] == 0
        assert result["training_complete"] is False


# ===========================================================================
# Tests for compute_fold_table (FROC/AP table builder)
# ===========================================================================


class TestComputeFoldTable:
    """Tests for the per-fold + pooled FROC/AP table builder."""

    def test_per_fold_rows_present(self, tmp_path: Path) -> None:
        """Table contains one row per fold."""
        from inspect_detector_quality import compute_fold_table

        fold_dirs = []
        for k in range(5):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            _make_results_boxes_json(fd)
            fold_dirs.append(str(fd))

        table = compute_fold_table(fold_dirs)
        # Should have 5 fold rows + 1 pooled row
        fold_rows = [r for r in table if r["fold"] != "pooled"]
        assert len(fold_rows) == 5

    def test_pooled_row_present(self, tmp_path: Path) -> None:
        """Table contains a 'pooled' row with mean across folds."""
        from inspect_detector_quality import compute_fold_table

        fold_dirs = []
        for k in range(5):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            _make_results_boxes_json(fd, froc_10=str(0.70 + k * 0.01))
            fold_dirs.append(str(fd))

        table = compute_fold_table(fold_dirs)
        pooled_rows = [r for r in table if r["fold"] == "pooled"]
        assert len(pooled_rows) == 1

    def test_pooled_froc_is_mean(self, tmp_path: Path) -> None:
        """Pooled FROC@IoU0.1 = mean of per-fold values."""
        from inspect_detector_quality import compute_fold_table

        froc_values = [0.72, 0.68, 0.74, 0.70, 0.71]
        fold_dirs = []
        for k, fv in enumerate(froc_values):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            _make_results_boxes_json(fd, froc_10=str(fv))
            fold_dirs.append(str(fd))

        table = compute_fold_table(fold_dirs)
        pooled = next(r for r in table if r["fold"] == "pooled")
        expected_mean = sum(froc_values) / len(froc_values)
        assert abs(pooled["FROC_score_IoU_0.10"] - expected_mean) < 1e-5

    def test_nan_values_excluded_from_mean(self, tmp_path: Path) -> None:
        """NaN fold values must be excluded from the pooled mean (not propagate NaN)."""
        from inspect_detector_quality import compute_fold_table

        fold_dirs = []
        for k in range(5):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            # fold 2 has a NaN
            _make_results_boxes_json(fd, include_nan=(k == 2))
            fold_dirs.append(str(fd))

        table = compute_fold_table(fold_dirs)
        pooled = next(r for r in table if r["fold"] == "pooled")
        import math

        # NaN-affected key: FROC_score_IoU_0.80 — fold 2 is NaN
        # Pooled must still be a valid float (4 non-NaN values averaged)
        val = pooled.get("FROC_score_IoU_0.80")
        assert val is not None
        assert not math.isnan(val)

    def test_missing_val_results_dir_raises(self, tmp_path: Path) -> None:
        """Fold dir without val_results/results_boxes.json raises FileNotFoundError."""
        from inspect_detector_quality import compute_fold_table

        fd = tmp_path / "fold0"
        fd.mkdir()
        # No val_results/ subdirectory created
        with pytest.raises(FileNotFoundError):
            compute_fold_table([str(fd)])


# ===========================================================================
# Tests for build_convergence_summary
# ===========================================================================


class TestBuildConvergenceSummary:
    """Tests for per-fold training convergence summary."""

    def test_returns_summary_per_fold(self, tmp_path: Path) -> None:
        """One convergence summary entry per fold."""
        from inspect_detector_quality import build_convergence_summary

        fold_dirs = []
        for k in range(5):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            _make_train_log(fd)
            fold_dirs.append(str(fd))

        summaries = build_convergence_summary(fold_dirs)
        assert len(summaries) == 5

    def test_complete_fold_marked(self, tmp_path: Path) -> None:
        """Fold with 60/60 epochs is marked training_complete=True."""
        from inspect_detector_quality import build_convergence_summary

        fd = tmp_path / "fold0"
        fd.mkdir()
        _make_train_log(fd, n_epochs=60)
        summaries = build_convergence_summary([str(fd)])
        assert summaries[0]["training_complete"] is True
        assert summaries[0]["epochs_completed"] == 60

    def test_incomplete_fold_marked(self, tmp_path: Path) -> None:
        """Fold with fewer than 60 epochs is marked training_complete=False."""
        from inspect_detector_quality import build_convergence_summary

        fd = tmp_path / "fold0"
        fd.mkdir()
        _make_train_log(fd, n_epochs=37)
        summaries = build_convergence_summary([str(fd)])
        assert summaries[0]["training_complete"] is False
        assert summaries[0]["epochs_completed"] == 37

    def test_missing_train_log_flagged(self, tmp_path: Path) -> None:
        """Fold without train.log is flagged as error, not crash."""
        from inspect_detector_quality import build_convergence_summary

        fd = tmp_path / "fold0"
        fd.mkdir()
        # No train.log created
        summaries = build_convergence_summary([str(fd)])
        assert summaries[0]["error"] is not None


# ===========================================================================
# Tests for render_detection_overlays
# ===========================================================================


class TestRenderDetectionOverlays:
    """Tests for the detection-vs-GT overlay PNG renderer.

    These tests use a minimal synthetic numpy volume — no real NRRD or server data.
    Matplotlib is imported but renders headlessly (Agg backend).
    """

    def _make_synthetic_volume(self, shape: tuple) -> np.ndarray:
        """Simple synthetic grayscale volume for rendering tests."""
        vol = np.random.randint(0, 256, shape, dtype=np.uint8).astype(np.float32)
        return vol

    def _make_gt_bbox_dict(self) -> dict:
        """Minimal GT bbox in the project's BBox-like dict format."""
        return {
            "min_d0": 50,
            "max_d0": 150,
            "min_d1": 30,
            "max_d1": 100,
            "min_d2": 20,
            "max_d2": 80,
        }

    def test_renders_png_for_each_case(self, tmp_path: Path) -> None:
        """One PNG per (fold, case) rendered without error."""
        from inspect_detector_quality import render_detection_overlays

        out_dir = tmp_path / "overlays"
        fold_dir = tmp_path / "fold0"
        fold_dir.mkdir()

        # Write a few val prediction pkls
        _write_val_prediction_pkl(fold_dir, "0004", n_dets=3)

        # Synthetic volume: (50, 60, 70) — small enough for fast test
        vol = self._make_synthetic_volume((50, 60, 70))
        gt_bboxes = {4: self._make_gt_bbox_dict()}
        spacing_mm = (0.073, 0.200, 0.476)

        render_detection_overlays(
            fold_idx=0,
            fold_dir=str(fold_dir),
            volume=vol,
            spacing_mm=spacing_mm,
            gt_bboxes=gt_bboxes,
            out_dir=str(out_dir),
            top_n=3,
            case_ids=[4],
        )

        # Expect at least one PNG
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_top_n_limits_displayed_boxes(self, tmp_path: Path) -> None:
        """top_n parameter limits the number of predicted boxes drawn."""
        from inspect_detector_quality import render_detection_overlays

        out_dir = tmp_path / "overlays"
        fold_dir = tmp_path / "fold0"
        fold_dir.mkdir()

        # 10 detections in the pkl
        _write_val_prediction_pkl(fold_dir, "0004", n_dets=10)
        vol = self._make_synthetic_volume((50, 60, 70))
        gt_bboxes = {4: self._make_gt_bbox_dict()}

        # Should not raise even with top_n < n_dets
        render_detection_overlays(
            fold_idx=0,
            fold_dir=str(fold_dir),
            volume=vol,
            spacing_mm=(0.073, 0.200, 0.476),
            gt_bboxes=gt_bboxes,
            out_dir=str(out_dir),
            top_n=3,
            case_ids=[4],
        )
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_missing_case_in_gt_skipped(self, tmp_path: Path) -> None:
        """If a case has no GT bbox, it is skipped without error."""
        from inspect_detector_quality import render_detection_overlays

        out_dir = tmp_path / "overlays"
        fold_dir = tmp_path / "fold0"
        fold_dir.mkdir()

        _write_val_prediction_pkl(fold_dir, "0004", n_dets=3)
        vol = self._make_synthetic_volume((50, 60, 70))
        # GT bboxes deliberately empty — case 4 has no GT entry
        gt_bboxes: dict = {}

        # Should not raise
        render_detection_overlays(
            fold_idx=0,
            fold_dir=str(fold_dir),
            volume=vol,
            spacing_mm=(0.073, 0.200, 0.476),
            gt_bboxes=gt_bboxes,
            out_dir=str(out_dir),
            top_n=3,
            case_ids=[4],
        )


# ===========================================================================
# Integration: test that the script is importable and has the expected CLI
# ===========================================================================


class TestScriptImportability:
    """Ensure the script can be imported and exposes the expected public API."""

    def test_module_importable(self) -> None:
        """scripts/inspect_detector_quality.py is importable without GPU/nnDetection."""
        import importlib

        spec = importlib.util.find_spec("inspect_detector_quality")
        assert spec is not None, (
            "inspect_detector_quality module not found in scripts/. "
            "Ensure scripts/ is in sys.path."
        )

    def test_parse_results_boxes_json_exists(self) -> None:
        """parse_results_boxes_json function is exported."""
        from inspect_detector_quality import parse_results_boxes_json

        assert callable(parse_results_boxes_json)

    def test_parse_sweep_json_exists(self) -> None:
        """parse_sweep_json function is exported."""
        from inspect_detector_quality import parse_sweep_json

        assert callable(parse_sweep_json)

    def test_parse_val_predictions_dir_exists(self) -> None:
        """parse_val_predictions_dir function is exported."""
        from inspect_detector_quality import parse_val_predictions_dir

        assert callable(parse_val_predictions_dir)

    def test_parse_train_log_exists(self) -> None:
        """parse_train_log function is exported."""
        from inspect_detector_quality import parse_train_log

        assert callable(parse_train_log)

    def test_compute_fold_table_exists(self) -> None:
        """compute_fold_table function is exported."""
        from inspect_detector_quality import compute_fold_table

        assert callable(compute_fold_table)

    def test_build_convergence_summary_exists(self) -> None:
        """build_convergence_summary function is exported."""
        from inspect_detector_quality import build_convergence_summary

        assert callable(build_convergence_summary)

    def test_render_detection_overlays_exists(self) -> None:
        """render_detection_overlays function is exported."""
        from inspect_detector_quality import render_detection_overlays

        assert callable(render_detection_overlays)

    def test_main_function_exists(self) -> None:
        """main() function is exported (entry point for server run)."""
        from inspect_detector_quality import main

        assert callable(main)

    def test_select_top_boxes_exists(self) -> None:
        """select_top_boxes helper is exported."""
        from inspect_detector_quality import select_top_boxes

        assert callable(select_top_boxes)


# ===========================================================================
# Tests for select_top_boxes helper
# ===========================================================================


class TestSelectTopBoxes:
    """Tests for the top-N box selector."""

    def test_returns_top_n_by_score(self) -> None:
        """Returns exactly top_n boxes, sorted by descending score."""
        from inspect_detector_quality import select_top_boxes

        boxes = np.arange(20, dtype=np.float32).reshape(4, 5)
        # pad to (4, 6)
        boxes = np.hstack([boxes, np.zeros((4, 1), dtype=np.float32)])
        scores = np.array([0.3, 0.9, 0.5, 0.7], dtype=np.float32)
        result = select_top_boxes(boxes, scores, top_n=2)
        assert result.shape == (2, 6)
        # Row 1 (score 0.9) should come first
        assert np.allclose(result[0], boxes[1])
        # Row 3 (score 0.7) should come second
        assert np.allclose(result[1], boxes[3])

    def test_fewer_boxes_than_top_n(self) -> None:
        """Returns all boxes if fewer than top_n are available."""
        from inspect_detector_quality import select_top_boxes

        boxes = np.ones((2, 6), dtype=np.float32)
        scores = np.array([0.8, 0.6], dtype=np.float32)
        result = select_top_boxes(boxes, scores, top_n=10)
        assert result.shape == (2, 6)

    def test_empty_boxes_returns_empty(self) -> None:
        """Empty input returns empty array with same shape (0, 6)."""
        from inspect_detector_quality import select_top_boxes

        boxes = np.zeros((0, 6), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        result = select_top_boxes(boxes, scores, top_n=5)
        assert result.shape == (0, 6)

    def test_top_n_zero_returns_empty(self) -> None:
        """top_n=0 returns empty array."""
        from inspect_detector_quality import select_top_boxes

        boxes = np.ones((5, 6), dtype=np.float32)
        scores = np.ones(5, dtype=np.float32)
        result = select_top_boxes(boxes, scores, top_n=0)
        assert result.shape[0] == 0

    def test_nan_score_sorts_last_not_first(self) -> None:
        """A NaN score must NOT appear as the top box — it must sort to the end.

        Regression for Code-reviewer S1: np.argsort places NaN first in
        descending order without the nan_to_num/-inf guard.
        """
        from inspect_detector_quality import select_top_boxes

        boxes = np.arange(18, dtype=np.float32).reshape(3, 6)
        # box index 1 has NaN score — must NOT come out as rank #1
        scores = np.array([0.5, float("nan"), 0.9], dtype=np.float32)
        result = select_top_boxes(boxes, scores, top_n=2)
        assert result.shape == (2, 6)
        # rank 0 must be box index 2 (score=0.9), not box index 1 (score=NaN)
        assert np.allclose(result[0], boxes[2]), "NaN-scored box should not be selected as rank 1"


# ===========================================================================
# Additional regression tests for findings S3 and S4
# ===========================================================================


class TestRegressionS3S4:
    """Additional regression tests from Round 1 reviewer findings S3 and S4."""

    def test_compute_fold_table_excludes_0_prefix_keys(self, tmp_path: Path) -> None:
        """0_-prefixed per-class duplicate keys must NOT appear in the table rows.

        Regression for finding S3: compute_fold_table calls parse_results_boxes_json
        which already strips 0_ keys. Verify end-to-end in the table output.
        """
        from inspect_detector_quality import compute_fold_table

        fd = tmp_path / "fold0"
        fd.mkdir()
        _make_results_boxes_json(fd, include_class_prefixed=True)

        table = compute_fold_table([str(fd)])
        for row in table:
            for key in row:
                assert not str(key).startswith("0_"), f"0_-prefixed key leaked into table: {key}"

    def test_parse_sweep_json_non_dict_scores_replaced_with_empty(self, tmp_path: Path) -> None:
        """If ast.literal_eval produces a non-dict (e.g. a list), replace with {}.

        Regression for finding S4: isinstance guard after eval.
        """
        import json as _json

        from inspect_detector_quality import parse_sweep_json

        sweep_dir = tmp_path / "sweep"
        sweep_dir.mkdir()
        # Craft a sweep file where "scores" is repr() of a list — not a dict
        data = {
            "0.05": {
                "state": "finished",
                "overwrite": {"model_score_thresh": 0.05},
                "scores": repr([0.72, 0.51]),  # a list, NOT a dict
            }
        }
        path = sweep_dir / "sweep_model_score_thresh.json"
        path.write_text(_json.dumps(data))

        result = parse_sweep_json(str(path))
        assert result["0.05"]["scores_parsed"] == {}

    def test_parse_sweep_json_bad_repr_replaced_with_empty(self, tmp_path: Path) -> None:
        """Unparseable 'scores' string falls back to {} without raising.

        Regression for finding S3: bad-scores repr branch.
        """
        import json as _json

        from inspect_detector_quality import parse_sweep_json

        sweep_dir = tmp_path / "sweep"
        sweep_dir.mkdir()
        data = {
            "0.05": {
                "state": "finished",
                "overwrite": {"model_score_thresh": 0.05},
                "scores": "THIS IS NOT VALID PYTHON REPR @@@@",
            }
        }
        path = sweep_dir / "sweep_model_score_thresh.json"
        path.write_text(_json.dumps(data))

        result = parse_sweep_json(str(path))
        assert result["0.05"]["scores_parsed"] == {}

    def test_pooled_row_has_min_n_folds_key(self, tmp_path: Path) -> None:
        """Pooled row must carry a min_n_folds key indicating min contributing fold count (A1)."""
        from inspect_detector_quality import compute_fold_table

        fold_dirs = []
        for k in range(5):
            fd = tmp_path / f"fold{k}"
            fd.mkdir()
            _make_results_boxes_json(fd)
            fold_dirs.append(str(fd))

        table = compute_fold_table(fold_dirs)
        pooled = next(r for r in table if r["fold"] == "pooled")
        assert "min_n_folds" in pooled
        assert pooled["min_n_folds"] == 5

    def test_loss_extraction_mismatch_flagged_in_error(self, tmp_path: Path) -> None:
        """When loss extraction produces fewer values than epoch markers, error is set (S2)."""
        from inspect_detector_quality import parse_train_log

        # Write a log with epoch markers but no loss lines
        lines = []
        for ep in range(10):
            lines.append(
                f"2026-01-01 00:{ep:02d}:00 | INFO | "
                f"nndet.training.trainer:training_epoch_end:311 | "
                f"training_epoch_end | some other message"
            )
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join(lines) + "\n")

        result = parse_train_log(str(log_path))
        # 10 epoch markers found, 0 loss values — error should be set
        assert result["epochs_completed"] == 10
        assert result["loss_extraction_complete"] is False
        assert result["error"] is not None
        assert "loss" in result["error"].lower()
