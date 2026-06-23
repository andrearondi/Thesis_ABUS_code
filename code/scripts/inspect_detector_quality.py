#!/usr/bin/env python
"""Detector quality inspection tool for STORY_01_02 (READ-ONLY, no GPU, no inference).

Run BEFORE Job 3a (candidate generation) to sanity-check all 5 trained fold detectors.
Reads artifacts already produced by nnDetection --sweep. Does NOT retrain or tune anything.

What this script does:
  (a) Per-fold + pooled FROC/AP table from val_results/results_boxes.json.
      Emphasises loose-IoU FROC (0.10-0.30) as the candidate-recall-relevant lens.
      IMPORTANT: FROC_score is mean-sensitivity-over-FP-volume-levels computed over the
      sweep-frozen detection set (score_thresh=0, <=100 boxes/case, WBC-clustered; the
      score axis is swept inside the FROC curve). It is a FLOOR indicator for the fixed
      detector at its detection-quality operating point. The gap from Gate A (STORY_01_03)
      is driven by the IoU>=0.30 hit threshold baked into FROC, remove_small_boxes=5,
      per-image cap, and WBC clustering — NOT a score-threshold difference.
      Gate A measures candidate recall on the actual candidate set at a permissive threshold.
  (b) Training-curve convergence summary from each fold's train.log.
  (c) Detection-vs-GT visual overlays: predicted boxes (top-N by score, green) +
      GT bbox (red) on mid-slice PNGs through the lesion, saved to output dir.

Usage (server):
  conda activate /home/maia-user/Andre/envs/thesis
  cd /home/maia-user/Andre/Thesis_ABUS_code/code
  python scripts/inspect_detector_quality.py \\
    --det-models $det_models \\
    --out-dir /home/maia-user/Andre/outputs/detector_quality_inspection \\
    --gt-bbx-csv /home/maia-user/Andre/data/Train/bbx_labels.csv \\
    [--k-cases 3] [--top-n 5] [--loss-csv /path/to/losses.csv]

Python 3.8-compatible: no X | Y union syntax, no match/case, no walrus assignments.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src/ is importable when invoked directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAME = "Task001_TDSCABUS"
EXP_ID = "RetinaUNetV001_D3V001_3d"
EXPECTED_EPOCHS = 60  # max_num_epochs 50 + swa_epochs 10

# IoU thresholds present in val_results/results_boxes.json (server-verified 2026-06-23)
FROC_IOUs = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
AP_IOUs = ["0.10", "0.20", "0.30", "0.40", "0.50"]

# Loose-IoU thresholds that matter for candidate recall (before Gate A in STORY_01_03)
LOOSE_FROC_IOUs = ["0.10", "0.20", "0.30"]


# ---------------------------------------------------------------------------
# Parser: val_results/results_boxes.json
# ---------------------------------------------------------------------------


def parse_results_boxes_json(path: str) -> dict[str, float]:
    """Parse val_results/results_boxes.json into a dict of float values.

    Real server schema (2026-06-23, commit 97a58f3):
      Flat dict. ALL values are STRINGS (including the literal "nan").
      Keys prefixed "0_" are per-class duplicates — excluded by this parser.

    Parameters
    ----------
    path : str
        Absolute path to results_boxes.json.

    Returns
    -------
    Dict[str, float]
        Keyed by metric name (class-prefix-free). NaN strings become float('nan').

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"results_boxes.json not found: {p}")

    with open(p) as f:
        raw = json.load(f)

    result: dict[str, float] = {}
    for key, val in raw.items():
        # Skip per-class duplicate keys (prefixed "0_")
        if key.startswith("0_"):
            continue
        try:
            result[key] = float(val)  # coerce string "nan" → float('nan')
        except (ValueError, TypeError):
            logger.warning(
                "parse_results_boxes_json: cannot convert key=%r val=%r to float", key, val
            )
    return result


# ---------------------------------------------------------------------------
# Parser: sweep/<param>.json
# ---------------------------------------------------------------------------


def parse_sweep_json(path: str) -> dict[str, Any]:
    """Parse one sweep/<param>.json file.

    Real server schema (2026-06-23):
      {param_label: {"state": str, "overwrite": dict, "scores": str}}
      CRITICAL: "scores" is a PYTHON-REPR string of a dict — use ast.literal_eval,
      NOT json.loads. json.loads will fail because repr() uses single quotes.

    Parameters
    ----------
    path : str
        Absolute path to a sweep_<param>.json file.

    Returns
    -------
    Dict[str, Any]
        Keyed by param_label. Each value is the original dict PLUS a
        "scores_parsed" key containing the parsed scores dict (float values).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sweep file not found: {p}")

    with open(p) as f:
        raw = json.load(f)

    result: dict[str, Any] = {}
    for param_label, entry in raw.items():
        parsed_entry = dict(entry)
        scores_raw = entry.get("scores", "{}")
        try:
            scores_parsed = ast.literal_eval(scores_raw)
        except (ValueError, SyntaxError) as exc:
            logger.warning(
                "parse_sweep_json: cannot ast.literal_eval scores for param_label=%r: %s",
                param_label,
                exc,
            )
            scores_parsed = {}
        if not isinstance(scores_parsed, dict):
            logger.warning(
                "parse_sweep_json: scores_parsed is not a dict for param_label=%r"
                " (got %s); replacing with {}.",
                param_label,
                type(scores_parsed).__name__,
            )
            scores_parsed = {}
        parsed_entry["scores_parsed"] = scores_parsed
        result[param_label] = parsed_entry

    return result


# ---------------------------------------------------------------------------
# Parser: val_predictions/<case>_boxes.pkl
# (REUSES abus.detect.nndet_inference.parse_predictions_dir)
# ---------------------------------------------------------------------------


def parse_val_predictions_dir(val_pred_dir: str) -> dict[int, Any]:
    """Parse val_predictions directory using the shared nnDetection schema parser.

    REUSES abus.detect.nndet_inference.parse_predictions_dir — do NOT reinvent.
    Schema: <case>_boxes.pkl consolidated dict (server-verified 2026-06-23).

    Parameters
    ----------
    val_pred_dir : str
        Path to a fold's val_predictions/ directory.

    Returns
    -------
    Dict[int, RawDetections]
        Keyed by case_id (int). Empty dict if directory has no matching files.
    """
    from abus.detect.nndet_inference import parse_predictions_dir

    return parse_predictions_dir(val_pred_dir)


# ---------------------------------------------------------------------------
# Parser: train.log
# ---------------------------------------------------------------------------


def parse_train_log(log_path: str) -> dict[str, Any]:
    """Parse a fold's train.log for convergence information.

    Looks for lines containing 'training_epoch_end' and extracts loss values.

    Parameters
    ----------
    log_path : str
        Absolute path to train.log.

    Returns
    -------
    Dict[str, Any]
        "epochs_completed": int — count of training_epoch_end lines
        "training_complete": bool — True iff epochs_completed == EXPECTED_EPOCHS
        "loss_per_epoch": List[float] — loss values per epoch (may be empty)
        "final_loss": Optional[float] — last loss value, or None
        "error": Optional[str] — set if parse failed

    Raises
    ------
    FileNotFoundError
        If the log file does not exist.
    """
    p = Path(log_path)
    if not p.exists():
        raise FileNotFoundError(f"train.log not found: {p}")

    epochs_completed = 0
    loss_per_epoch: list[float] = []

    with open(p) as f:
        for line in f:
            if "training_epoch_end" not in line:
                continue
            epochs_completed += 1
            # Try to extract loss= value from the line
            loss_val = _extract_loss_from_log_line(line)
            if loss_val is not None:
                loss_per_epoch.append(loss_val)

    final_loss: float | None = loss_per_epoch[-1] if loss_per_epoch else None

    # Warn when the log contains epoch markers but loss extraction produced fewer
    # values than epoch markers — indicates the regex didn't match some lines.
    loss_count = len(loss_per_epoch)
    if epochs_completed > 0 and loss_count < epochs_completed:
        error_msg: str | None = (
            f"loss extracted for only {loss_count}/{epochs_completed} epochs "
            "(train.log format may differ from expected 'Train loss reached: <float>')"
        )
    else:
        error_msg = None

    return {
        "epochs_completed": epochs_completed,
        "training_complete": epochs_completed == EXPECTED_EPOCHS,
        "loss_per_epoch": loss_per_epoch,
        "final_loss": final_loss,
        "loss_extraction_complete": loss_count == epochs_completed,
        "error": error_msg,
    }


def _extract_loss_from_log_line(line: str) -> float | None:
    """Extract a loss value from a loguru-format train.log line.

    nnDetection real format (loguru default, commit 97a58f3):
      "... training_epoch_end | Train loss reached: 0.12345"
    Matches the pattern 'Train loss reached: <float>' case-sensitively.
    """
    import re

    m = re.search(r"Train loss reached:\s*([0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)", line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# FROC/AP table builder
# ---------------------------------------------------------------------------


def compute_fold_table(fold_dirs: list[str]) -> list[dict[str, Any]]:
    """Build per-fold + pooled FROC/AP table from val_results/results_boxes.json.

    NaN values from individual folds are excluded from the pooled mean (not propagated).

    Parameters
    ----------
    fold_dirs : List[str]
        List of fold directory paths, one per fold (5 total).

    Returns
    -------
    List[Dict[str, Any]]
        One dict per fold (fold=0..4) + one "pooled" dict.
        Each dict contains fold identifier plus all FROC and AP metric keys.

    Raises
    ------
    FileNotFoundError
        If any fold's val_results/results_boxes.json is missing.
    """
    # Collect all metric keys we care about
    froc_keys = [f"FROC_score_IoU_{iou}" for iou in FROC_IOUs]
    ap_keys = [f"AP_IoU_{iou}_MaxDet_100" for iou in AP_IOUs]
    map_key = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"
    all_metric_keys = froc_keys + ap_keys + [map_key]

    rows: list[dict[str, Any]] = []
    per_fold_values: dict[str, list[float]] = {k: [] for k in all_metric_keys}

    for k, fold_dir in enumerate(fold_dirs):
        results_path = Path(fold_dir) / "val_results" / "results_boxes.json"
        metrics = parse_results_boxes_json(str(results_path))

        row: dict[str, Any] = {"fold": k}
        for key in all_metric_keys:
            val = metrics.get(key, float("nan"))
            row[key] = val
            if not math.isnan(val):
                per_fold_values[key].append(val)
        rows.append(row)

    # Pooled row: mean over non-NaN fold values
    # min_n_folds: minimum contributing-fold count across metrics.
    # Per-metric means each exclude only their own NaN folds; this single scalar
    # shows the worst-case fold coverage (use per-key len(per_fold_values[k]) for
    # full detail).
    min_n_folds = min(
        (len(per_fold_values[k]) for k in all_metric_keys if per_fold_values[k]),
        default=0,
    )
    pooled: dict[str, Any] = {"fold": "pooled", "min_n_folds": min_n_folds}
    for key in all_metric_keys:
        vals = per_fold_values[key]
        pooled[key] = float(np.mean(vals)) if vals else float("nan")
    rows.append(pooled)

    return rows


# ---------------------------------------------------------------------------
# Training convergence summary
# ---------------------------------------------------------------------------


def build_convergence_summary(fold_dirs: list[str]) -> list[dict[str, Any]]:
    """Build per-fold training convergence summaries from train.log files.

    Parameters
    ----------
    fold_dirs : List[str]
        List of fold directory paths.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per fold with convergence info.
        "error" is set (non-None) if train.log is missing or unreadable.
    """
    summaries: list[dict[str, Any]] = []

    for k, fold_dir in enumerate(fold_dirs):
        log_path = Path(fold_dir) / "train.log"
        try:
            summary = parse_train_log(str(log_path))
        except FileNotFoundError as exc:
            summary = {
                "epochs_completed": 0,
                "training_complete": False,
                "loss_per_epoch": [],
                "final_loss": None,
                "loss_extraction_complete": False,
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            summary = {
                "epochs_completed": 0,
                "training_complete": False,
                "loss_per_epoch": [],
                "final_loss": None,
                "loss_extraction_complete": False,
                "error": f"Unexpected error parsing train.log: {exc}",
            }
        summary["fold"] = k
        summaries.append(summary)

    return summaries


# ---------------------------------------------------------------------------
# Box-selection helper
# ---------------------------------------------------------------------------


def select_top_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Return the top-N boxes sorted by descending score.

    Parameters
    ----------
    boxes : np.ndarray
        Shape (N, 6) — box coordinates.
    scores : np.ndarray
        Shape (N,) — confidence scores.
    top_n : int
        Maximum number of boxes to return.

    Returns
    -------
    np.ndarray
        Shape (min(N, top_n), 6) sorted by descending score.
    """
    if boxes.shape[0] == 0 or top_n <= 0:
        return boxes[:0]  # zero-row array preserving dtype and column count
    # Replace NaN scores with -inf so they always sort to the end, not the front.
    safe_scores = np.where(np.isnan(scores), -np.inf, scores)
    idx: np.ndarray = np.argsort(safe_scores)[::-1][:top_n]
    result: np.ndarray = boxes[idx]
    return result


# ---------------------------------------------------------------------------
# Detection-vs-GT overlay renderer
# ---------------------------------------------------------------------------


def render_detection_overlays(
    fold_idx: int,
    fold_dir: str,
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    gt_bboxes: dict[int, dict[str, int]],
    out_dir: str,
    top_n: int = 5,
    case_ids: list[int] | None = None,
) -> None:
    """Render detection-vs-GT overlay PNGs for a fold's val cases.

    Loads val_predictions/<case>_boxes.pkl, takes top-N boxes by score,
    overlays predicted boxes (green) and GT bbox (red) on the volume mid-slice
    through the lesion centroid. Saves PNGs to out_dir.

    Axis convention (ABUS original-grid, storage order):
      d0 = acoustic depth  (0.073 mm/vox, ~865 vox)
      d1 = lateral         (0.200 mm/vox, ~470 vox)
      d2 = elevation       (0.476 mm/vox, ~348 vox)

    Mid-slice is taken through the GT lesion centroid (d0, d1, d2 separately).
    Predicted boxes are in original-grid space after restore_boxes_for_case
    (same space as GT bboxes).

    Note: val_predictions/ is the restore=True original-image-space export
    (train.py:311 — nnDetection writes to <fold>/val_predictions/ via
    restore_boxes_for_case). Boxes are in the same voxel space as the GT
    bbox CSV, so overlays are geometrically valid.

    Parameters
    ----------
    fold_idx : int
    fold_dir : str
        Path to fold directory (contains val_predictions/).
    volume : np.ndarray
        3-D array (d0, d1, d2) float32/uint8 — the NRRD volume for this case.
        May be a shared volume if case_ids are from the same volume (sanity use).
    spacing_mm : Tuple[float, float, float]
        Physical spacing (d0, d1, d2) in mm/voxel.
    gt_bboxes : Dict[int, Dict[str, int]]
        GT bboxes keyed by case_id. Each value is a dict with keys:
        min_d0, max_d0, min_d1, max_d1, min_d2, max_d2.
    out_dir : str
        Directory to save PNGs.
    top_n : int
        Number of top-scoring predicted boxes to overlay.
    case_ids : Optional[List[int]]
        Which case IDs to render. If None, renders all found in val_predictions/.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless — no display required on server
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    val_pred_dir = Path(fold_dir) / "val_predictions"
    if not val_pred_dir.exists():
        logger.warning("render_detection_overlays: val_predictions/ not found at %s", fold_dir)
        return

    # Parse all val predictions for this fold
    all_preds = parse_val_predictions_dir(str(val_pred_dir))

    render_case_ids = case_ids if case_ids is not None else sorted(all_preds.keys())

    for cid in render_case_ids:
        if cid not in gt_bboxes:
            logger.info(
                "render_detection_overlays: fold=%d case=%d skipped (no GT bbox)", fold_idx, cid
            )
            continue

        gt = gt_bboxes[cid]
        rd = all_preds.get(cid)

        # Get top-N predicted boxes by score
        if rd is not None and rd.boxes.shape[0] > 0:
            top_boxes = select_top_boxes(rd.boxes, rd.scores, top_n)
        else:
            top_boxes = np.zeros((0, 6), dtype=np.float32)

        # Render one PNG per slice axis through GT centroid
        gt_cx_d0 = (gt["min_d0"] + gt["max_d0"]) // 2
        gt_cx_d1 = (gt["min_d1"] + gt["max_d1"]) // 2
        gt_cx_d2 = (gt["min_d2"] + gt["max_d2"]) // 2

        vol_arr = volume

        for axis in range(3):
            if axis == 0:
                mid = min(gt_cx_d0, vol_arr.shape[0] - 1)
                slice_img = vol_arr[mid, :, :]
                # GT rect in (d1, d2) — matplotlib: col=d2, row=d1
                gt_rect_col, gt_rect_row = gt["min_d2"], gt["min_d1"]
                gt_rect_w = gt["max_d2"] - gt["min_d2"] + 1
                gt_rect_h = gt["max_d1"] - gt["min_d1"] + 1
                # Predicted: box (x1,y1,x2,y2,z1,z2) where on restore x=d0,y=d1,z=d2
                # For axis-0 slice: show (d1, d2) — col=box[4..5]=z=d2, row=box[1..3]=y=d1
                pred_rects = [
                    (b[4], b[1], b[5] - b[4], b[3] - b[1]) for b in top_boxes
                ]  # (col_start, row_start, w, h) using z=d2 as col
                xlabel, ylabel = "d2 (elevation)", "d1 (lateral)"
            elif axis == 1:
                mid = min(gt_cx_d1, vol_arr.shape[1] - 1)
                slice_img = vol_arr[:, mid, :]
                # GT rect in (d0, d2)
                gt_rect_col, gt_rect_row = gt["min_d2"], gt["min_d0"]
                gt_rect_w = gt["max_d2"] - gt["min_d2"] + 1
                gt_rect_h = gt["max_d0"] - gt["min_d0"] + 1
                # axis-1 slice: show (d0, d2) — col=z=d2, row=x=d0
                pred_rects = [(b[4], b[0], b[5] - b[4], b[2] - b[0]) for b in top_boxes]
                xlabel, ylabel = "d2 (elevation)", "d0 (depth)"
            else:  # axis == 2
                mid = min(gt_cx_d2, vol_arr.shape[2] - 1)
                slice_img = vol_arr[:, :, mid]
                # GT rect in (d0, d1)
                gt_rect_col, gt_rect_row = gt["min_d1"], gt["min_d0"]
                gt_rect_w = gt["max_d1"] - gt["min_d1"] + 1
                gt_rect_h = gt["max_d0"] - gt["min_d0"] + 1
                # axis-2 slice: show (d0, d1) — col=y=d1, row=x=d0
                pred_rects = [(b[1], b[0], b[3] - b[1], b[2] - b[0]) for b in top_boxes]
                xlabel, ylabel = "d1 (lateral)", "d0 (depth)"

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(slice_img, cmap="gray", origin="upper", aspect="auto")

            # GT bbox (red)
            gt_patch = patches.Rectangle(
                (gt_rect_col, gt_rect_row),
                gt_rect_w,
                gt_rect_h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="GT bbox",
            )
            ax.add_patch(gt_patch)

            # Predicted boxes (green, decreasing alpha for rank)
            n_pred = len(pred_rects)
            for rank, (col, row, w, h) in enumerate(pred_rects):
                alpha = 0.9 - rank * (0.6 / max(n_pred, 1))
                pred_patch = patches.Rectangle(
                    (col, row),
                    max(w, 1),
                    max(h, 1),
                    linewidth=1.5,
                    edgecolor="lime",
                    facecolor="none",
                    alpha=alpha,
                    label=f"pred rank {rank + 1}" if rank == 0 else None,
                )
                ax.add_patch(pred_patch)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"Fold {fold_idx} | Case {cid:04d} | axis={axis} (slice={mid}) | "
                f"GT (red) + top-{top_n} pred (green)"
            )
            ax.legend(loc="upper right", fontsize=7)

            spacing_str = (
                f"spacing: d0={spacing_mm[0]:.3f}"
                f" d1={spacing_mm[1]:.3f} d2={spacing_mm[2]:.4f} mm/vox"
            )
            fig.text(0.5, -0.01, spacing_str, ha="center", va="top", family="monospace", fontsize=8)

            out_png = out_path / f"fold{fold_idx:02d}_case{cid:04d}_axis{axis}.png"
            fig.savefig(str(out_png), dpi=100, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved overlay PNG: %s", out_png.name)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_froc_ap_table(table: list[dict[str, Any]]) -> None:
    """Print the FROC/AP table to stdout, emphasising loose-IoU metrics.

    IMPORTANT NOTE printed in output:
      FROC_score is mean-sensitivity-over-FP-rates at the SWEPT threshold.
      It is a FLOOR indicator for the fixed detector at its operating point.
      The binding recall test is Gate A in STORY_01_03 which runs at a
      permissive threshold on the actual candidate set.
    """
    print("\n" + "=" * 80)
    print("DETECTOR QUALITY INSPECTION — FROC/AP TABLE")
    print("=" * 80)
    print()
    print("NOTE: FROC_score = mean-sensitivity-over-FP-volume-levels computed over the")
    print("  sweep-frozen detection set (score_thresh=0, <=100 boxes/case, WBC-clustered).")
    print("  The score axis is swept INSIDE the FROC curve — this is NOT a single-")
    print("  threshold metric. It is a FLOOR indicator for the fixed detector.")
    print("  The gap from Gate A (STORY_01_03) is driven by IoU>=0.30 hit threshold,")
    print("  remove_small_boxes=5, per-image cap, and WBC clustering — NOT by the")
    print("  score threshold. Gate A measures candidate recall at a permissive threshold.")
    print("  Loose-IoU FROC (0.10-0.30) is the candidate-recall-relevant lens.")
    print("  This table is a PRE-FLIGHT sanity check only.")
    print()

    # Header (fold label is widened to 16 chars to fit "pooled(n=5)")
    col_w = 8
    header = f"{'fold':<16}"
    for iou in LOOSE_FROC_IOUs:
        header += f"{'FROC@' + iou:^{col_w}}"
    for iou in AP_IOUs[:3]:
        header += f"{'AP@' + iou:^{col_w}}"
    print("--- Loose-IoU view (recall-relevant) ---")
    print(header)
    print("-" * len(header))

    for row in table:
        fold_label = str(row["fold"])
        if fold_label == "pooled":
            print("-" * len(header))
            n_folds = row.get("min_n_folds", "?")
            fold_label = f"pooled(n={n_folds})"
        line = f"{fold_label:<16}"
        for iou in LOOSE_FROC_IOUs:
            key = f"FROC_score_IoU_{iou}"
            val = row.get(key, float("nan"))
            line += f"{val:^{col_w}.4f}" if not math.isnan(val) else f"{'nan':^{col_w}}"
        for iou in AP_IOUs[:3]:
            key = f"AP_IoU_{iou}_MaxDet_100"
            val = row.get(key, float("nan"))
            line += f"{val:^{col_w}.4f}" if not math.isnan(val) else f"{'nan':^{col_w}}"
        print(line)

    print()
    print("--- Full FROC across all IoU thresholds ---")
    froc_header = f"{'fold':<16}" + "".join(f"{'@' + iou:^9}" for iou in FROC_IOUs)
    print(froc_header)
    print("-" * len(froc_header))

    for row in table:
        fold_label = str(row["fold"])
        if fold_label == "pooled":
            print("-" * len(froc_header))
            n_folds = row.get("min_n_folds", "?")
            fold_label = f"pooled(n={n_folds})"
        line = f"{fold_label:<16}"
        for iou in FROC_IOUs:
            key = f"FROC_score_IoU_{iou}"
            val = row.get(key, float("nan"))
            line += f"{val:^9.4f}" if not math.isnan(val) else f"{'nan':^9}"
        print(line)

    print()


def print_convergence_summary(summaries: list[dict[str, Any]]) -> None:
    """Print training convergence summary to stdout."""
    print("=" * 80)
    print("TRAINING CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"{'Fold':<6} {'Epochs':>8} {'Complete':>10} {'Final loss':>12} {'Status'}")
    print("-" * 60)

    all_complete = True
    for s in summaries:
        fold = s.get("fold", "?")
        epochs = s.get("epochs_completed", 0)
        complete = s.get("training_complete", False)
        final_loss = s.get("final_loss")
        error = s.get("error")

        if error:
            status = f"ERROR: {error[:40]}"
            all_complete = False
        elif not complete:
            status = f"INCOMPLETE ({epochs}/{EXPECTED_EPOCHS} epochs)"
            all_complete = False
        else:
            loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
            status = "OK"
            fold_str = str(fold)
            print(
                f"{fold_str:<6} {epochs:>8}/{EXPECTED_EPOCHS:<3} {'YES':>8}"
                f" {loss_str:>12} {status}"
            )
            continue

        fold_str = str(fold)
        loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
        print(f"{fold_str:<6} {epochs:>8}/{EXPECTED_EPOCHS:<3} {'NO':>8} {loss_str:>12} {status}")

    print()
    if all_complete:
        print("All folds: training complete (60/60 epochs). Safe to proceed to Job 3a.")
    else:
        print("WARNING: Some folds are INCOMPLETE. Do NOT proceed to Job 3a.")
    print()


# ---------------------------------------------------------------------------
# CSV loss dump
# ---------------------------------------------------------------------------


def dump_loss_csv(summaries: list[dict[str, Any]], csv_path: str) -> None:
    """Dump per-epoch loss values to CSV for external plotting.

    Parameters
    ----------
    summaries : List[Dict[str, Any]]
        Output of build_convergence_summary.
    csv_path : str
        Output path for the CSV file.
    """
    rows = []
    max_epochs = max((len(s.get("loss_per_epoch", [])) for s in summaries), default=0)
    for ep in range(max_epochs):
        row: dict[str, Any] = {"epoch": ep + 1}
        for s in summaries:
            fold = s.get("fold", "?")
            losses = s.get("loss_per_epoch", [])
            row[f"fold{fold}_loss"] = losses[ep] if ep < len(losses) else ""
        rows.append(row)

    if not rows:
        logger.info("No loss data to dump.")
        return

    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Loss CSV written: %s", csv_path)


# ---------------------------------------------------------------------------
# Sweep summary printer
# ---------------------------------------------------------------------------


def print_sweep_summary(fold_dirs: list[str]) -> None:
    """Print a brief summary of sweep files per fold (how score varies with param)."""
    print("=" * 80)
    print("SWEEP PARAMETER SUMMARY (FROC@IoU0.10 across operating points)")
    print("=" * 80)

    sweep_param_names = [
        "ensemble_iou",
        "model_iou",
        "model_nms_fn",
        "model_score_thresh",
        "remove_small_boxes",
    ]

    for k, fold_dir in enumerate(fold_dirs):
        sweep_dir = Path(fold_dir) / "sweep"
        if not sweep_dir.exists():
            print(f"  Fold {k}: sweep/ directory not found.")
            continue

        print(f"\n  Fold {k}:")
        for param in sweep_param_names:
            sweep_file = sweep_dir / f"sweep_{param}.json"
            if not sweep_file.exists():
                continue
            try:
                data = parse_sweep_json(str(sweep_file))
            except Exception as exc:  # noqa: BLE001
                print(f"    {param}: parse error — {exc}")
                continue

            # Show FROC@0.10 range across param values
            froc_vals = []
            for label, entry in data.items():
                scores = entry.get("scores_parsed", {})
                v = scores.get("FROC_score_IoU_0.10")
                if v is not None:
                    froc_vals.append((label, float(v)))

            if froc_vals:
                froc_vals_sorted = sorted(froc_vals, key=lambda x: x[1], reverse=True)
                best_label, best_froc = froc_vals_sorted[0]
                worst_label, worst_froc = froc_vals_sorted[-1]
                print(
                    f"    {param}: FROC@0.10 range [{worst_froc:.4f}, {best_froc:.4f}]"
                    f"  (best: {param}={best_label})"
                )
            else:
                print(f"    {param}: no FROC@0.10 scores in sweep file.")

    print()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, run all inspections, save outputs."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Read-only detector quality inspection for STORY_01_02. "
            "Run BEFORE Job 3a. No GPU, no inference, no retraining."
        )
    )
    parser.add_argument(
        "--det-models",
        required=True,
        help=(
            "nnDetection det_models root. Fold dirs are at "
            "det_models/Task001_TDSCABUS/RetinaUNetV001_D3V001_3d/fold{0..4}/."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for PNGs and CSV.",
    )
    parser.add_argument(
        "--gt-bbx-csv",
        default="",
        help=(
            "Path to GT bbx_labels.csv (Train/bbx_labels.csv on server). "
            "Required for visual overlays. If not provided, overlays are skipped."
        ),
    )
    parser.add_argument(
        "--nrrd-vol",
        default="",
        help=(
            "Path to a single NRRD volume to use for overlays (e.g. one val case). "
            "If not provided, overlays are skipped."
        ),
    )
    parser.add_argument(
        "--k-cases",
        type=int,
        default=3,
        help="Number of val cases per fold to render overlays for (default: 3).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top-scoring predicted boxes to overlay (default: 5).",
    )
    parser.add_argument(
        "--loss-csv",
        default="",
        help="If provided, dump per-epoch loss values to this CSV path.",
    )
    parser.add_argument(
        "--task",
        default=TASK_NAME,
        help=f"nnDetection task name (default: {TASK_NAME}).",
    )
    parser.add_argument(
        "--exp-id",
        default=EXP_ID,
        help=f"nnDetection experiment id (default: {EXP_ID}).",
    )

    args = parser.parse_args()

    # Build fold dir list
    fold_root = Path(args.det_models) / args.task / args.exp_id
    fold_dirs = [str(fold_root / f"fold{k}") for k in range(5)]

    # Validate fold dirs exist
    missing = [fd for fd in fold_dirs if not Path(fd).exists()]
    if missing:
        logger.error("Missing fold directories: %s", missing)
        sys.exit(1)

    out_dir = args.out_dir

    # -----------------------------------------------------------------------
    # (a) FROC/AP table
    # -----------------------------------------------------------------------
    print("\nParsing val_results/results_boxes.json for all 5 folds ...")
    try:
        table = compute_fold_table(fold_dirs)
    except FileNotFoundError as exc:
        logger.error("FROC table error: %s", exc)
        sys.exit(1)

    print_froc_ap_table(table)

    # -----------------------------------------------------------------------
    # (b) Sweep summary
    # -----------------------------------------------------------------------
    print_sweep_summary(fold_dirs)

    # -----------------------------------------------------------------------
    # (c) Training convergence
    # -----------------------------------------------------------------------
    print("\nParsing train.log for convergence ...")
    summaries = build_convergence_summary(fold_dirs)
    print_convergence_summary(summaries)

    if args.loss_csv:
        dump_loss_csv(summaries, args.loss_csv)

    # -----------------------------------------------------------------------
    # (d) Visual overlays (optional — only if NRRD vol + GT CSV provided)
    # -----------------------------------------------------------------------
    if args.nrrd_vol and args.gt_bbx_csv:
        print(
            f"\nRendering detection-vs-GT overlays"
            f" (k_cases={args.k_cases}, top_n={args.top_n}) ..."
        )
        _run_overlays(fold_dirs, args.nrrd_vol, args.gt_bbx_csv, out_dir, args.k_cases, args.top_n)
    else:
        print("\nOverlays skipped: --nrrd-vol and --gt-bbx-csv both required for overlays.")

    print(f"\nInspection complete. Outputs in: {out_dir}")
    print(
        "\nNext step: if all folds show 60/60 epochs + plausible FROC values, "
        "proceed to Job 3a (candidate generation)."
    )
    print("The BINDING recall test is Gate A in STORY_01_03 — this is a pre-flight check only.")


def _run_overlays(
    fold_dirs: list[str],
    nrrd_vol_path: str,
    gt_bbx_csv_path: str,
    out_dir: str,
    k_cases: int,
    top_n: int,
) -> None:
    """Load a single NRRD volume + GT bboxes and render overlays for each fold.

    Design constraint: only ONE volume is loaded (from --nrrd-vol). The volume's
    case_id is parsed from its filename (DATA_<NNN>.nrrd). Overlays are rendered
    only for that case — rendering boxes from a different case over the wrong
    volume would be geometrically meaningless. This constraint is intentional.

    For a multi-case overlay pass, re-run the script with a different --nrrd-vol.
    """
    # Import project loader — lazy so the script remains importable without
    # the full abus package installed on every machine.
    try:
        from abus.io.loader import load_volume
    except ImportError:
        logger.error(
            "Cannot import abus.io.loader. "
            "Ensure src/ is in PYTHONPATH or abus is installed. "
            "Overlays skipped."
        )
        return

    try:
        vol = load_volume(nrrd_vol_path)
        vol_arr = vol.array.astype(np.float32)
        spacing_mm = vol.spacing_mm
        single_case_id = vol.case_id
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load NRRD volume %s: %s", nrrd_vol_path, exc)
        return

    logger.info(
        "Overlays: single-volume mode — case_id=%d, shape=%s, spacing=%s mm/vox",
        single_case_id,
        vol_arr.shape,
        spacing_mm,
    )

    # Load GT bboxes from CSV
    gt_bboxes = _load_gt_bboxes_for_overlays(gt_bbx_csv_path)

    for k, fold_dir in enumerate(fold_dirs):
        val_pred_dir = Path(fold_dir) / "val_predictions"
        if not val_pred_dir.exists():
            logger.warning("Fold %d: val_predictions/ not found, skipping overlays.", k)
            continue

        # Check whether the loaded case is a val case for this fold
        all_preds = parse_val_predictions_dir(str(val_pred_dir))
        if single_case_id not in all_preds:
            logger.info(
                "Fold %d: case_id=%d is not in this fold's val set — skipping.",
                k,
                single_case_id,
            )
            continue

        # Render exactly the one case whose volume we loaded
        fold_out_dir = str(Path(out_dir) / f"fold{k}_overlays")
        render_detection_overlays(
            fold_idx=k,
            fold_dir=fold_dir,
            volume=vol_arr,
            spacing_mm=spacing_mm,
            gt_bboxes=gt_bboxes,
            out_dir=fold_out_dir,
            top_n=top_n,
            case_ids=[single_case_id],
        )

    print(f"Overlays saved to: {out_dir}/fold*_overlays/")


def _load_gt_bboxes_for_overlays(bbx_csv_path: str) -> dict[int, dict[str, int]]:
    """Load GT bboxes from bbx_labels.csv for overlay rendering.

    Delegates to abus.data.labels.load_gt_bboxes — do NOT reinvent the CSV parser.
    The raw CSV has a non-trivial units convention (mm vs voxels, axis permutation)
    that is authoritative in that module; any raw-CSV fallback here would be wrong.
    abus is always available on the server (src/ on PYTHONPATH).

    Raises
    ------
    ImportError
        If abus.data.labels is not importable (should not happen on server).
    """
    from abus.data.labels import load_gt_bboxes

    gt = load_gt_bboxes(bbx_csv_path)
    result: dict[int, dict[str, int]] = {}
    for cid, bbox in gt.items():
        result[cid] = {
            "min_d0": bbox.min_d0,
            "max_d0": bbox.max_d0,
            "min_d1": bbox.min_d1,
            "max_d1": bbox.max_d1,
            "min_d2": bbox.min_d2,
            "max_d2": bbox.max_d2,
        }
    return result


if __name__ == "__main__":
    main()
