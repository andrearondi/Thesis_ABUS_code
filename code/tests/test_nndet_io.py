"""Tests for STORY_01_01: nnDetection dataset converter + verifier.

TDD cycle: these tests were written before the implementation modules exist.
Each test references modules that don't exist yet; every test must fail with
ImportError or AttributeError until the implementation is in place.

Test inventory (per story acceptance criteria):

  test_bbox_nndet_roundtrip_synthetic      — ASC-01_01.4 / local half:
      random project BBoxes survive bbox_to_nndet → nndet_to_bbox exactly on
      the original grid (the resampled-grid mapping is identity when target==original).

  test_bbox_original_roundtrip_identity_spacing — nndet_convention.bbox_original_roundtrip:
      when target_spacing_mm == CANONICAL_SPACING_MM (identity resampling), the
      round-trip preserves the box exactly.

  test_splits_file_faithful              — ASC-01_01.3 leakage guard:
      write_nndet_splits produces a splits file whose fold membership equals
      load_split() case-for-case.

  test_spacing_written                   — ASC-01_01.2:
      convert_case writes CANONICAL_SPACING_MM into the nnDetection image file.

  test_convert_case_lesion_count         — ASC-01_01.1 (single-case):
      convert_case writes a label whose lesion count matches the CSV row.

  test_loader_guard_propagates           — Risk mitigation 1:
      a synthetic NRRD with non-identity space directions makes convert_case raise
      SpacingPlaceholderError (EPIC_00 guard propagates; spacing never silently wrong).

  test_verify_detects_split_mismatch     — ASC-01_01.3 error path:
      a hand-corrupted nnDetection splits file makes verify_nndet_dataset raise
      NndetDatasetError.

  test_dataset_spec_fields               — NndetDatasetSpec fields are present and typed.

  test_build_dataset_json                — build_dataset_json writes a valid JSON file.

  test_dry_run_entrypoint                — the --dry-run CLI path of nndet_io exits 0
      on a synthetic volume (local-data-sanity smoke).

  ---- Schema-fix tests (Round 2, STORY_01_01 re-run) ----

  test_build_dataset_json_nndet_v01_schema — regression for missing 'task' key server error.
  test_export_dataset_raw_splitted_layout  — raw_splitted/ subdirectory layout.
  test_export_dataset_per_case_json_sidecar — labelsTr/*.nrrd has sibling .json.
  test_dataset_json_passes_nndet_check_replica — mini-replica of nnDetection check_dataset_file.
  test_splits_content_unchanged_after_layout_fix — splits content invariant to path (ASC-01_01.3).
  test_convert_case_multi_lesion_raises_before_sidecar — S1 fix: multi-lesion guard fires
      before _write_per_case_json so wrong-encoding sidecar is never written.
  test_verify_nndet_dataset_detects_missing_sidecar — S2 fix: missing .json sidecar detected
      by verify_nndet_dataset before nndet_prep is executed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import nrrd
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test — these will fail until the modules are implemented.
# ---------------------------------------------------------------------------
from abus.detect.nndet_convention import (
    NNDET_BOX_AXES,
    bbox_original_roundtrip,
)
from abus.detect.nndet_io import (
    NndetDatasetError,
    NndetDatasetSpec,
    build_dataset_json,
    convert_case,
    export_dataset,
    verify_nndet_dataset,
    write_nndet_splits,
)
from abus.geometry.bbox import BBox
from abus.geometry.convert import bbox_to_nndet, nndet_to_bbox
from abus.io.loader import CANONICAL_SPACING_MM, SpacingPlaceholderError

# ---------------------------------------------------------------------------
# Helpers — synthetic NRRD factory (reuses logic from tests/fixtures/synthetic_nrrd.py
# but inline here for test isolation; no dependency on implementation details).
# ---------------------------------------------------------------------------

_SMALL_SHAPE = (8, 10, 12)  # (d0, d1, d2) — small enough for fast tests


def _write_identity_volume_nrrd(path: str, shape: tuple[int, int, int] = _SMALL_SHAPE) -> None:
    """Write a minimal uint8 DATA_<N>.nrrd with identity placeholder header."""
    array = np.zeros(shape, dtype=np.uint8)
    array[2, 3, 4] = 50  # non-zero voxel
    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(path, array, header)


def _write_non_identity_volume_nrrd(path: str, shape: tuple[int, int, int] = _SMALL_SHAPE) -> None:
    """Write a uint8 DATA_<N>.nrrd with non-identity space directions (triggers guard)."""
    array = np.ones(shape, dtype=np.uint8) * 10
    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(path, array, header)


def _write_identity_mask_nrrd(
    path: str,
    shape: tuple[int, int, int] = _SMALL_SHAPE,
    lesion_box: BBox | None = None,
) -> None:
    """Write a minimal uint8 MASK_<N>.nrrd with a small tumor region."""
    array = np.zeros(shape, dtype=np.uint8)
    if lesion_box is not None:
        array[
            lesion_box.min_d0 : lesion_box.max_d0 + 1,
            lesion_box.min_d1 : lesion_box.max_d1 + 1,
            lesion_box.min_d2 : lesion_box.max_d2 + 1,
        ] = 1
    else:
        # Default small lesion at [2:5, 3:7, 4:8]
        array[2:5, 3:7, 4:8] = 1
    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(path, array, header)


def _make_csv_row(
    case_id: int,
    bbox: BBox,
) -> dict:
    """Build a bbx_labels.csv-style dict for one lesion from a storage-axis BBox.

    Applies the inverse (2,1,0) permutation to produce ITK-order CSV values.
    """
    # storage -> ITK: d0->z, d1->y, d2->x
    len_x = float(bbox.max_d2 - bbox.min_d2)
    len_y = float(bbox.max_d1 - bbox.min_d1)
    len_z = float(bbox.max_d0 - bbox.min_d0)
    c_x = bbox.min_d2 + len_x / 2.0
    c_y = bbox.min_d1 + len_y / 2.0
    c_z = bbox.min_d0 + len_z / 2.0
    return {
        "id": case_id,
        "c_x": c_x,
        "c_y": c_y,
        "c_z": c_z,
        "len_x": len_x,
        "len_y": len_y,
        "len_z": len_z,
    }


# ===========================================================================
# Test 1 — bbox round-trip on the original grid (synthetic boxes)
# ===========================================================================


def test_bbox_nndet_roundtrip_synthetic() -> None:
    """Random project BBoxes survive bbox_to_nndet -> nndet_to_bbox on the original grid.

    On the original (unresampled) grid the round-trip must be exact (residual == 0
    on every axis). This is the local half of ASC-01_01.4; the resampled-grid half
    is tested on the server in the runbook.
    """
    rng = np.random.default_rng(seed=42)
    max_coord = 800
    n_boxes = 50

    for _ in range(n_boxes):
        # Generate random valid BBox (min <= max per axis)
        coords = rng.integers(0, max_coord, size=6)
        min_d0 = int(min(coords[0], coords[3]))
        max_d0 = int(max(coords[0], coords[3]))
        min_d1 = int(min(coords[1], coords[4]))
        max_d1 = int(max(coords[1], coords[4]))
        min_d2 = int(min(coords[2], coords[5]))
        max_d2 = int(max(coords[2], coords[5]))
        # Ensure non-degenerate (at least 1 voxel per axis)
        max_d0 = max(max_d0, min_d0 + 1)
        max_d1 = max(max_d1, min_d1 + 1)
        max_d2 = max(max_d2, min_d2 + 1)

        original = BBox(min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)
        nndet_fmt = bbox_to_nndet(original)
        recovered = nndet_to_bbox(nndet_fmt)

        assert recovered == original, (
            f"Round-trip failed: original={original}, recovered={recovered}, "
            f"nndet_fmt={nndet_fmt}"
        )


# ===========================================================================
# Test 2 — bbox_original_roundtrip from nndet_convention (identity spacing)
# ===========================================================================


def test_bbox_original_roundtrip_identity_spacing() -> None:
    """bbox_original_roundtrip is exact when target_spacing == CANONICAL_SPACING_MM.

    When nnDetection's target spacing equals the original spacing (no resampling),
    the resampled-grid mapping is the identity and the round-trip must be exact.
    This tests nndet_convention.bbox_original_roundtrip's documented behaviour.
    """
    # A box representative of a small lesion in the dataset
    original = BBox(min_d0=50, min_d1=100, min_d2=80, max_d0=100, max_d1=150, max_d2=120)

    recovered = bbox_original_roundtrip(original, target_spacing_mm=CANONICAL_SPACING_MM)

    # With no resampling the round-trip must be exact (residual 0 per axis)
    assert recovered == original, (
        f"Round-trip failed with identity resampling: "
        f"original={original}, recovered={recovered}"
    )


# ===========================================================================
# Test 3 — write_nndet_splits produces a splits file faithful to load_split()
# ===========================================================================


def test_splits_file_faithful(tmp_path: Path) -> None:
    """write_nndet_splits produces a splits file whose fold membership equals load_split().

    This is the leakage guard (ASC-01_01.3): the nnDetection CV split must be
    byte-faithful to the frozen 5-fold manifest (configs/splits/fold_split_5cv.json).
    The function must consume load_split() and never call make_fold_split.
    """
    from abus.data.split import load_split

    splits_path = str(tmp_path / "splits.json")
    write_nndet_splits(splits_path)

    with open(splits_path, encoding="utf-8") as f:
        produced = json.load(f)

    frozen = load_split()  # reads configs/splits/fold_split_5cv.json

    # nnDetection splits file format: a list of dicts with keys 'train' and 'val'
    assert isinstance(produced, list), "Splits file must be a JSON list"
    assert len(produced) == len(
        frozen.folds
    ), f"Fold count mismatch: {len(produced)} vs {len(frozen.folds)}"

    for k, entry in enumerate(produced):
        expected_train = frozen.train_ids(k)
        expected_val = frozen.oof_ids(k)

        # nnDetection uses string case IDs in some versions; accept int or str
        produced_train = sorted(int(x) for x in entry["train"])
        produced_val = sorted(int(x) for x in entry["val"])

        assert (
            produced_train == expected_train
        ), f"Fold {k} train mismatch: produced={produced_train}, expected={expected_train}"
        assert (
            produced_val == expected_val
        ), f"Fold {k} val mismatch: produced={produced_val}, expected={expected_val}"


# ===========================================================================
# Test 4 — convert_case writes CANONICAL_SPACING_MM into the nnDetection image
# ===========================================================================


def test_spacing_written(tmp_path: Path) -> None:
    """convert_case writes CANONICAL_SPACING_MM into the nnDetection image file.

    The nnDetection fingerprint reads spacing from the written image; if the
    converter writes identity spacing nnDetection will resample on a wrong grid
    (Risk 1 in the story). This test reads back the written NRRD and asserts
    the space directions diagonal equals CANONICAL_SPACING_MM.
    """
    # Setup synthetic case 42
    case_id = 42
    vol_path = str(tmp_path / "DATA_0042.nrrd")
    mask_path = str(tmp_path / "MASK_0042.nrrd")

    lesion_box = BBox(2, 3, 4, 4, 6, 7)
    _write_identity_volume_nrrd(vol_path)
    _write_identity_mask_nrrd(mask_path, lesion_box=lesion_box)

    csv_row = _make_csv_row(case_id, lesion_box)

    out_image = str(tmp_path / "image_0042_0000.nrrd")
    out_label = str(tmp_path / "label_0042.nrrd")

    convert_case(vol_path, mask_path, csv_row, out_image, out_label)

    # Read back the written image and inspect its header spacing
    _, header = nrrd.read(out_image)
    space_dirs = np.array(header["space directions"], dtype=float)

    # Diagonal must equal CANONICAL_SPACING_MM
    for i, expected_sp in enumerate(CANONICAL_SPACING_MM):
        actual_sp = space_dirs[i, i]
        assert abs(actual_sp - expected_sp) < 1e-9, (
            f"Axis {i} spacing mismatch: got {actual_sp}, expected {expected_sp}. "
            "convert_case must write CANONICAL_SPACING_MM, not identity spacing."
        )

    # Off-diagonal must be zero (pure diagonal spacing, no shear)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(space_dirs[i, j]) < 1e-9, (
                    f"Off-diagonal spacing[{i},{j}]={space_dirs[i,j]} != 0 — "
                    "spacing matrix must be diagonal."
                )


# ===========================================================================
# Test 5 — convert_case writes a label whose lesion count matches the CSV
# ===========================================================================


def test_convert_case_lesion_count(tmp_path: Path) -> None:
    """convert_case writes a label whose lesion count matches the CSV row.

    The story specifies: one CSV row = one lesion. The written label file must
    contain exactly one foreground instance. The label format uses the mask
    directly (one connected component = one lesion) for a single-lesion case.
    This is the local half of ASC-01_01.1.
    """
    case_id = 7
    vol_path = str(tmp_path / "DATA_0007.nrrd")
    mask_path = str(tmp_path / "MASK_0007.nrrd")

    lesion_box = BBox(1, 2, 3, 3, 5, 6)
    _write_identity_volume_nrrd(vol_path)
    _write_identity_mask_nrrd(mask_path, lesion_box=lesion_box)

    csv_row = _make_csv_row(case_id, lesion_box)

    out_image = str(tmp_path / "image_0007_0000.nrrd")
    out_label = str(tmp_path / "label_0007.nrrd")

    result = convert_case(vol_path, mask_path, csv_row, out_image, out_label)

    assert result["n_lesions"] == 1, (
        f"Expected 1 lesion, got n_lesions={result['n_lesions']}. "
        "The convert_case summary dict must report the correct lesion count."
    )
    assert result["case_id"] == case_id, f"Expected case_id={case_id}, got {result['case_id']}"
    assert (
        result["spacing_written"] == CANONICAL_SPACING_MM
    ), f"Expected spacing_written={CANONICAL_SPACING_MM}, got {result['spacing_written']}"

    # The label file must exist
    assert Path(out_label).exists(), f"Label file not written: {out_label}"


# ===========================================================================
# Test 6 — SpacingPlaceholderError propagates from convert_case
# ===========================================================================


def test_loader_guard_propagates(tmp_path: Path) -> None:
    """A non-identity NRRD header makes convert_case raise SpacingPlaceholderError.

    The EPIC_00 loader guards against a non-placeholder header. convert_case
    must use load_volume (which applies this guard), so if any source NRRD
    ever carries a real (non-placeholder) spacing, the error propagates
    immediately instead of silently corrupting all downstream coordinates.
    """
    case_id = 99
    vol_path = str(tmp_path / "DATA_0099.nrrd")
    mask_path = str(tmp_path / "MASK_0099.nrrd")

    _write_non_identity_volume_nrrd(vol_path)
    _write_identity_mask_nrrd(mask_path)

    lesion_box = BBox(1, 2, 3, 3, 5, 6)
    csv_row = _make_csv_row(case_id, lesion_box)

    out_image = str(tmp_path / "image_0099_0000.nrrd")
    out_label = str(tmp_path / "label_0099.nrrd")

    with pytest.raises(SpacingPlaceholderError):
        convert_case(vol_path, mask_path, csv_row, out_image, out_label)


# ===========================================================================
# Test 7 — verify_nndet_dataset raises NndetDatasetError on a corrupted splits file
# ===========================================================================


def test_verify_detects_split_mismatch(tmp_path: Path) -> None:
    """A hand-corrupted nnDetection splits file makes verify_nndet_dataset raise NndetDatasetError.

    verify_nndet_dataset must check that the nnDetection splits file stored in the
    dataset directory equals the frozen manifest case-for-case.  A mismatch is a
    hard failure (ASC-01_01.3).
    """
    from abus.data.split import load_split

    frozen = load_split()

    # Build a minimal fake nnDetection dataset directory
    task_dir = tmp_path / "Task001_TDSCABUS"
    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    # Write a corrupted splits file (swap case IDs between folds)
    all_ids = [cid for fold in frozen.folds for cid in fold]
    # Corrupt: put all IDs in fold 0, leave rest empty
    corrupted_splits = [{"train": [], "val": all_ids}] + [
        {"train": all_ids, "val": []} for _ in range(len(frozen.folds) - 1)
    ]
    splits_path = task_dir / "splits_final.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(corrupted_splits, f)

    # Write a minimal dataset.json using the correct nnDetection v0.1 schema
    # so verify_nndet_dataset proceeds to the splits check (not blocked by schema check).
    # test_labels is required since Round 3 (preprocess.py line 394 contract).
    ds_json = {
        "task": "Task001_TDSCABUS",
        "dim": 3,
        "modalities": {"0": "US"},
        "labels": {"0": "tumor"},
        "test_labels": False,
    }
    with open(task_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(ds_json, f)

    with pytest.raises(NndetDatasetError, match="split"):
        verify_nndet_dataset(str(task_dir))


# ===========================================================================
# Test 8 — NndetDatasetSpec fields are present and typed correctly
# ===========================================================================


def test_dataset_spec_fields() -> None:
    """NndetDatasetSpec is a frozen dataclass with the expected fields and types."""
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=100,
        n_val_cases=30,
        n_test_cases=70,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="single foreground class: tumor",
    )
    assert spec.task_id == 1
    assert spec.task_name == "Task001_TDSCABUS"
    assert spec.n_train_cases == 100
    assert spec.n_val_cases == 30
    assert spec.n_test_cases == 70
    assert spec.spacing_mm == CANONICAL_SPACING_MM
    assert spec.modality == "US"

    # Must be frozen (immutable)
    with pytest.raises((AttributeError, TypeError)):
        spec.task_id = 999  # type: ignore[misc]


# ===========================================================================
# Test 9 — build_dataset_json writes a valid JSON file
# ===========================================================================


def test_build_dataset_json(tmp_path: Path) -> None:
    """build_dataset_json writes a JSON file with required nnDetection v0.1 keys.

    Updated to match the nnDetection v0.1 schema (not nnUNet):
      - 'task' (str), 'dim' (int=3), 'modalities' (dict), 'labels' (foreground-only)
    The old nnUNet keys ('name', 'modality', 'numTraining', etc.) are absent.
    """
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=100,
        n_val_cases=30,
        n_test_cases=70,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="single foreground class: tumor",
    )
    out_path = str(tmp_path / "dataset.json")
    build_dataset_json(spec, out_path)

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    # Required nnDetection v0.1 keys (NOT nnUNet keys)
    assert "task" in data, "dataset.json must have 'task' (nnDetection v0.1)"
    assert data["task"] == "Task001_TDSCABUS"
    assert "dim" in data, "dataset.json must have 'dim'"
    assert data["dim"] == 3
    assert "modalities" in data, "dataset.json must have 'modalities' (plural)"
    assert data["modalities"]["0"] == "US"
    assert "labels" in data, "dataset.json must have 'labels'"
    # Foreground-only labels: {"0": "tumor"}, background implicit
    assert "0" in data["labels"], "Foreground class '0' must be present"
    assert data["labels"]["0"] == "tumor", "First foreground class must be 'tumor'"
    assert "1" not in data["labels"], "nnDetection v0.1 labels must be foreground-only (no index 1)"


# ===========================================================================
# Test 10 — NNDET_BOX_AXES constant is present in nndet_convention
# ===========================================================================


def test_nndet_box_axes_constant() -> None:
    """nndet_convention.NNDET_BOX_AXES documents the axis mapping as a constant."""
    # The constant must exist and document the (y1,x1,y2,x2,z1,z2) ordering
    assert isinstance(NNDET_BOX_AXES, tuple) or isinstance(
        NNDET_BOX_AXES, str
    ), "NNDET_BOX_AXES must be a tuple or string documenting the axis ordering"


# ===========================================================================
# Test 11 — dry-run entrypoint exits 0 on a synthetic volume
# ===========================================================================


def test_dry_run_entrypoint(tmp_path: Path) -> None:
    """The --dry-run CLI path of nndet_io exits 0 on a synthetic volume.

    This is the local-data-sanity smoke test. In --dry-run mode the module
    runs convert_case on the provided case directory and exits 0 on success.
    This test uses a synthetic (all-zeros) case in tmp_path.
    """
    import subprocess
    import sys

    # Create a synthetic case (case ID 0042)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    vol_path = case_dir / "DATA_0042.nrrd"
    mask_path = case_dir / "MASK_0042.nrrd"
    lesion_box = BBox(2, 3, 4, 4, 6, 7)
    _write_identity_volume_nrrd(str(vol_path))
    _write_identity_mask_nrrd(str(mask_path), lesion_box=lesion_box)

    # Write a minimal bbx_labels.csv
    csv_row = _make_csv_row(42, lesion_box)
    csv_path = case_dir / "bbx_labels.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["id", "c_x", "c_y", "c_z", "len_x", "len_y", "len_z"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_row)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "abus.detect.nndet_io",
            "--dry-run",
            "--case-dir",
            str(case_dir),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--dry-run exited with code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ===========================================================================
# Test 12 — convert_case raises ValueError on an empty-list csv_row (M1 fix)
# ===========================================================================


def test_convert_case_empty_csv_row_raises(tmp_path: Path) -> None:
    """convert_case raises ValueError when bbox_csv_row is an empty list.

    The empty-list case represents a case with no CSV rows — not expected in TDSC
    but must produce a clear error rather than silently writing an empty dict or
    crashing with a KeyError.
    """
    vol_path = str(tmp_path / "DATA_0055.nrrd")
    mask_path = str(tmp_path / "MASK_0055.nrrd")
    _write_identity_volume_nrrd(vol_path)
    _write_identity_mask_nrrd(mask_path)

    out_image = str(tmp_path / "image_0055_0000.nrrd")
    out_label = str(tmp_path / "label_0055.nrrd")

    with pytest.raises(ValueError, match="empty list"):
        convert_case(vol_path, mask_path, [], out_image, out_label)


# ===========================================================================
# Test 13 — convert_case raises ValueError on a csv_row missing required keys
# ===========================================================================


def test_convert_case_invalid_csv_row_raises(tmp_path: Path) -> None:
    """convert_case raises ValueError when bbox_csv_row is missing required keys.

    Tests the _validate_csv_row guard added to fix M1 (code review Round 1).
    A dict missing 'c_x' etc. must produce a clear error, not a KeyError.
    """
    vol_path = str(tmp_path / "DATA_0056.nrrd")
    mask_path = str(tmp_path / "MASK_0056.nrrd")
    _write_identity_volume_nrrd(vol_path)
    _write_identity_mask_nrrd(mask_path)

    out_image = str(tmp_path / "image_0056_0000.nrrd")
    out_label = str(tmp_path / "label_0056.nrrd")

    incomplete_row = {"id": 56, "c_x": 5.0}  # missing c_y, c_z, len_x, len_y, len_z

    with pytest.raises(ValueError, match="missing required keys"):
        convert_case(vol_path, mask_path, incomplete_row, out_image, out_label)


# ===========================================================================
# Helper for end-to-end export_dataset tests (Test 14, 15, 16 below)
# ===========================================================================


def _build_synthetic_tdsc_root(
    root: Path,
    train_shards: dict[str, list[int]],  # subfolder name -> case ids (DATA only)
    val_ids: list[int],
    test_ids: list[int],
    mask_padding: int = 3,  # zero-pad width for MASK_*.nrrd filenames in Train
) -> None:
    """Materialise a minimal TDSC-like directory tree under ``root``.

    The Train split uses the nested-shard layout discovered on the official
    distribution (Train/DATA/<shard>/DATA_<NNN>.nrrd) with a flat MASK folder;
    Validation and Test use the documented flat layout. A bbx_labels.csv with
    one lesion per case is written under each split. Volumes and masks share
    a small synthetic lesion so convert_case succeeds end-to-end.

    Parameters
    ----------
    train_shards:
        Mapping from shard subfolder name (e.g. ``"DATA00_49"``) to the list
        of case IDs whose DATA files live in that shard. The MASK_*.nrrd files
        for these IDs always live flat in ``Train/MASK/``.
    mask_padding:
        Zero-padding width for Train MASK filenames. The on-disk Train MASK
        files are 3-digit-padded (``MASK_000.nrrd``); val/test masks use
        no padding for IDs ≥ 100 (``MASK_100.nrrd``). Default 3 mirrors Train.
    """
    lesion_box = BBox(2, 3, 4, 4, 6, 7)
    csv_header = ["id", "c_x", "c_y", "c_z", "len_x", "len_y", "len_z"]

    def _write_pair(data_dir: Path, mask_dir: Path, cid: int, pad: int) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        data_name = f"DATA_{cid:0{pad}d}.nrrd"
        mask_name = f"MASK_{cid:0{pad}d}.nrrd"
        _write_identity_volume_nrrd(str(data_dir / data_name))
        _write_identity_mask_nrrd(str(mask_dir / mask_name), lesion_box=lesion_box)

    def _write_csv(csv_path: Path, ids: list[int]) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            for cid in ids:
                writer.writerow(_make_csv_row(cid, lesion_box))

    # --- Train: nested DATA shards + flat MASK ---
    train_dir = root / "Train"
    train_mask_dir = train_dir / "MASK"
    all_train_ids: list[int] = []
    for shard_name, ids in train_shards.items():
        shard_data_dir = train_dir / "DATA" / shard_name
        for cid in ids:
            _write_pair(shard_data_dir, train_mask_dir, cid, mask_padding)
            all_train_ids.append(cid)
    _write_csv(train_dir / "bbx_labels.csv", all_train_ids)

    # --- Validation: flat DATA + flat MASK (use no-pad — IDs are 3 digits anyway) ---
    val_dir = root / "Validation"
    for cid in val_ids:
        _write_pair(val_dir / "DATA", val_dir / "MASK", cid, pad=1)
    _write_csv(val_dir / "bbx_labels.csv", val_ids)

    # --- Test: flat DATA + flat MASK ---
    test_dir = root / "Test"
    for cid in test_ids:
        _write_pair(test_dir / "DATA", test_dir / "MASK", cid, pad=1)
    _write_csv(test_dir / "bbx_labels.csv", test_ids)


# ===========================================================================
# Test 14 — export_dataset handles nested Train/DATA shards (2026-05-20 fix)
# ===========================================================================


def test_export_dataset_handles_nested_train_shards(tmp_path: Path) -> None:
    """export_dataset discovers Train cases laid out in nested DATA shards.

    The official TDSC distribution ships ``Train/DATA/DATA00_49/`` and
    ``Train/DATA/DATA50_99/`` (two shards), while Validation and Test are
    flat. Prior to the 2026-05-20 fix, ``data_dir.glob("DATA_*.nrrd")`` (non-
    recursive) silently returned 0 matches under ``Train/DATA/``, so the
    builder produced a dataset with 0 training cases without raising any
    error. The fix uses ``rglob`` and a case-id keyed dict; this test pins
    that behaviour. Regression-guards against the runbook STORY_01_01
    Checkpoint-4 failure of 2026-05-20.
    """
    tdsc_root = tmp_path / "tdsc"
    _build_synthetic_tdsc_root(
        tdsc_root,
        train_shards={
            "DATA00_49": [0, 1, 2],  # 3-digit-padded: DATA_000.nrrd, DATA_001.nrrd, DATA_002.nrrd
            "DATA50_99": [50, 99],  # DATA_050.nrrd, DATA_099.nrrd
        },
        val_ids=[100, 101],
        test_ids=[200],
    )

    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=5,  # 3 + 2 across the two shards
        n_val_cases=2,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )

    result = export_dataset(str(tdsc_root), str(out_root), spec)

    assert result["n_train"] == 5, (
        f"Expected n_train=5 across two nested shards, got {result['n_train']}. "
        "Recursive glob over Train/DATA/**/DATA_*.nrrd appears broken."
    )
    assert result["n_val"] == 2
    assert result["n_test"] == 1

    # Every training case must have produced both an image and a label
    # nnDetection v0.1 layout: under raw_splitted/
    task_dir = Path(result["task_dir"])
    images_tr = sorted((task_dir / "raw_splitted" / "imagesTr").glob("*.nrrd"))
    labels_tr = sorted((task_dir / "raw_splitted" / "labelsTr").glob("*.nrrd"))
    assert len(images_tr) == 5, f"Expected 5 imagesTr files, got {len(images_tr)}"
    assert len(labels_tr) == 5, f"Expected 5 labelsTr files, got {len(labels_tr)}"

    # Val + test images live under raw_splitted/imagesTs (no labels)
    images_ts = sorted((task_dir / "raw_splitted" / "imagesTs").glob("*.nrrd"))
    assert len(images_ts) == 3, f"Expected 3 imagesTs files (val+test), got {len(images_ts)}"


# ===========================================================================
# Test 15 — export_dataset refuses to write a partial dataset (silent-zero guard)
# ===========================================================================


def test_export_dataset_raises_on_train_count_mismatch(tmp_path: Path) -> None:
    """export_dataset raises NndetDatasetError when discovered train count != spec.

    The 2026-05-20 Checkpoint-4 failure produced a dataset with n_train=0
    while spec.n_train_cases=100 because the non-recursive glob silently
    returned an empty iterator. This test pins the discovery-phase count
    assertion that makes such a partial conversion impossible.
    """
    tdsc_root = tmp_path / "tdsc"
    _build_synthetic_tdsc_root(
        tdsc_root,
        train_shards={"DATA00_49": [0, 1]},  # only 2 train cases on disk
        val_ids=[100],
        test_ids=[200],
    )

    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=5,  # spec lies — disk has only 2
        n_val_cases=1,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )

    with pytest.raises(NndetDatasetError, match="Train"):
        export_dataset(str(tdsc_root), str(out_root), spec)


# ===========================================================================
# Test 16 — case-id matching across DATA and MASK tolerates padding-width skew
# ===========================================================================


def test_export_dataset_tolerates_mask_padding_skew(tmp_path: Path) -> None:
    """Mixed zero-padding widths across DATA and MASK files are tolerated.

    Real-world TDSC Train ships ``DATA_000.nrrd`` and ``MASK_000.nrrd``
    (3-digit padded). The previous mask-lookup heuristic tried
    ``MASK_{cid:04d}.nrrd`` then ``MASK_{cid}.nrrd`` — neither matches a
    3-digit-padded filename for low case IDs, so the fallback silently
    failed in production. The new implementation keys by parsed integer
    case_id, which is padding-agnostic. This test asserts that DATA files
    3-digit-padded with MASK files 4-digit-padded still pair correctly.
    """
    tdsc_root = tmp_path / "tdsc"
    # Create a Train split with 3-digit DATA filenames AND 4-digit MASK
    # filenames (artificial mix to exercise the padding-agnostic match)
    train_dir = tdsc_root / "Train"
    train_data_dir = train_dir / "DATA" / "DATA00_49"
    train_mask_dir = train_dir / "MASK"
    train_data_dir.mkdir(parents=True)
    train_mask_dir.mkdir(parents=True)
    lesion_box = BBox(2, 3, 4, 4, 6, 7)
    case_ids = [0, 7, 42]
    for cid in case_ids:
        _write_identity_volume_nrrd(str(train_data_dir / f"DATA_{cid:03d}.nrrd"))
        _write_identity_mask_nrrd(
            str(train_mask_dir / f"MASK_{cid:04d}.nrrd"),  # 4-digit-padded
            lesion_box=lesion_box,
        )
    csv_header = ["id", "c_x", "c_y", "c_z", "len_x", "len_y", "len_z"]
    with open(train_dir / "bbx_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        for cid in case_ids:
            writer.writerow(_make_csv_row(cid, lesion_box))

    # Trivial Val + Test (one case each, flat layout)
    for split_name, cid in (("Validation", 100), ("Test", 200)):
        split_dir = tdsc_root / split_name
        (split_dir / "DATA").mkdir(parents=True)
        (split_dir / "MASK").mkdir(parents=True)
        _write_identity_volume_nrrd(str(split_dir / "DATA" / f"DATA_{cid}.nrrd"))
        _write_identity_mask_nrrd(
            str(split_dir / "MASK" / f"MASK_{cid}.nrrd"),
            lesion_box=lesion_box,
        )
        with open(split_dir / "bbx_labels.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            writer.writerow(_make_csv_row(cid, lesion_box))

    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=3,
        n_val_cases=1,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )

    result = export_dataset(str(tdsc_root), str(out_root), spec)
    assert result["n_train"] == 3


# ===========================================================================
# NEW TESTS FOR NNDETECTION v0.1 SCHEMA FIX (STORY_01_01 re-run defects)
# Written RED-first: these fail on the OLD build_dataset_json / export_dataset.
# ===========================================================================


# ===========================================================================
# Test 17 — build_dataset_json writes nnDetection v0.1 schema (regression test)
#
# TDD evidence: test written BEFORE fix; fails on OLD code because:
#   - OLD code has 'modality' not 'modalities'
#   - OLD code lacks 'task' and 'dim'
#   - OLD code has labels {"0":"background","1":"tumor"} not {"0":"tumor"}
# ===========================================================================


def test_build_dataset_json_nndet_v01_schema(tmp_path: Path) -> None:
    """build_dataset_json produces a nnDetection v0.1-conformant dataset.json.

    Checks every key the nnDetection v0.1 check.py / check_dataset_file
    validates:
      - 'task' present and a str
      - 'dim' present and == 3
      - 'modalities' present (NOT 'modality') with key "0" == "US"
      - 'labels' foreground-only: {"0": "tumor"} (NOT {"0":"background","1":"tumor"})
      - '_project' provenance dict preserved
      - nnUNet-only keys absent: 'name', 'numTraining', 'numTest',
        'tensorImageSize', 'training', 'test'

    This is the regression test that would have caught the original ValueError:
        ValueError: Dataset information did not contain 'task' key,
        found ['_project','labels','modality','name','numTest','numTraining',
               'tensorImageSize','test','training']
    """
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=100,
        n_val_cases=30,
        n_test_cases=70,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="single foreground class: tumor",
    )
    out_path = str(tmp_path / "dataset.json")
    build_dataset_json(spec, out_path)

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    # --- Required nnDetection v0.1 keys ---
    assert "task" in data, (
        "dataset.json must have 'task' key (nnDetection v0.1 check.py line 86 "
        "_check_key_missing(cfg, 'task', ktype=str)); "
        f"found keys: {sorted(data.keys())}"
    )
    assert isinstance(data["task"], str), f"'task' must be a str, got {type(data['task']).__name__}"
    assert data["task"] == "Task001_TDSCABUS", f"'task' must equal task_name, got {data['task']!r}"

    assert (
        "dim" in data
    ), f"dataset.json must have 'dim' key for 3D volumes; found: {sorted(data.keys())}"
    assert data["dim"] == 3, f"'dim' must be 3 for 3D ABUS volumes, got {data['dim']}"

    # --- modalities (plural) replaces modality ---
    assert "modalities" in data, (
        "dataset.json must have 'modalities' (plural) key, NOT 'modality'. "
        f"Found keys: {sorted(data.keys())}"
    )
    assert "0" in data["modalities"], f"'modalities' must have key '0', got {data['modalities']}"
    assert (
        data["modalities"]["0"] == "US"
    ), f"'modalities'[\"0\"] must be 'US', got {data['modalities']['0']!r}"
    assert "modality" not in data, (
        "nnUNet key 'modality' must NOT be present in nnDetection v0.1 dataset.json; "
        f"found keys: {sorted(data.keys())}"
    )

    # --- foreground-only labels ---
    assert "labels" in data, f"dataset.json must have 'labels'; found: {sorted(data.keys())}"
    assert data["labels"] == {"0": "tumor"}, (
        "nnDetection v0.1 labels must be foreground-only, zero-indexed: "
        '{"0": "tumor"}. '
        f"Got: {data['labels']}. "
        "Background is implicit in nnDetection; it must NOT appear in labels."
    )

    # --- _project provenance dict preserved ---
    assert (
        "_project" in data
    ), f"'_project' provenance dict must be preserved; found: {sorted(data.keys())}"

    # --- nnUNet-only keys must be absent ---
    nnunet_keys = {"name", "numTraining", "numTest", "tensorImageSize", "training", "test"}
    present_nnunet = nnunet_keys & set(data.keys())
    assert not present_nnunet, (
        f"nnUNet-only keys must not appear in nnDetection dataset.json: {present_nnunet}. "
        "These keys encode nnUNet semantics and may confuse nnDetection's validation."
    )


# ===========================================================================
# Test 18 — export_dataset produces raw_splitted/ subdirectory layout
#
# TDD evidence: test written BEFORE fix; fails on OLD code because:
#   - OLD export_dataset writes imagesTr/labelsTr/imagesTs directly under task_dir
#   - NEW layout requires task_dir/raw_splitted/imagesTr etc.
# ===========================================================================


def test_export_dataset_raw_splitted_layout(tmp_path: Path) -> None:
    """export_dataset writes imagesTr/labelsTr/imagesTs under raw_splitted/ subdir.

    nnDetection v0.1 expects:
        Task001_TDSCABUS/
        ├── dataset.json         (task root — already correct)
        └── raw_splitted/
            ├── imagesTr/
            ├── labelsTr/
            └── imagesTs/

    The current (wrong) layout writes imagesTr/labelsTr/imagesTs directly
    under the task root, causing nndet_prep to fail with a FileNotFoundError
    when it looks for raw_splitted/.
    """
    tdsc_root = tmp_path / "tdsc"
    _build_synthetic_tdsc_root(
        tdsc_root,
        train_shards={"DATA00_49": [0, 1]},
        val_ids=[100],
        test_ids=[200],
    )

    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=2,
        n_val_cases=1,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )

    result = export_dataset(str(tdsc_root), str(out_root), spec)
    task_dir = Path(result["task_dir"])

    # raw_splitted/ must exist under the task root
    raw_splitted = task_dir / "raw_splitted"
    assert raw_splitted.exists() and raw_splitted.is_dir(), (
        f"raw_splitted/ subdirectory not found under {task_dir}. "
        "nnDetection v0.1 nndet_prep looks for raw_splitted/imagesTr etc."
    )

    # imagesTr, labelsTr, imagesTs must be UNDER raw_splitted/
    assert (
        raw_splitted / "imagesTr"
    ).is_dir(), f"imagesTr/ must be under raw_splitted/, not directly under {task_dir}"
    assert (
        raw_splitted / "labelsTr"
    ).is_dir(), f"labelsTr/ must be under raw_splitted/, not directly under {task_dir}"
    assert (
        raw_splitted / "imagesTs"
    ).is_dir(), f"imagesTs/ must be under raw_splitted/, not directly under {task_dir}"

    # The task root must NOT have imagesTr/labelsTr/imagesTs directly (wrong layout)
    assert not (
        task_dir / "imagesTr"
    ).exists(), f"imagesTr/ must NOT exist directly under {task_dir} (old wrong layout)"

    # Correct file counts
    images_tr = sorted((raw_splitted / "imagesTr").glob("*.nrrd"))
    labels_tr = sorted((raw_splitted / "labelsTr").glob("*.nrrd"))
    images_ts = sorted((raw_splitted / "imagesTs").glob("*.nrrd"))
    assert len(images_tr) == 2, f"Expected 2 imagesTr files, got {len(images_tr)}"
    assert len(labels_tr) == 2, f"Expected 2 labelsTr files, got {len(labels_tr)}"
    assert len(images_ts) == 2, f"Expected 2 imagesTs files (val+test), got {len(images_ts)}"

    # dataset.json must remain at task root (not inside raw_splitted)
    assert (task_dir / "dataset.json").exists(), "dataset.json must remain at the task root"


# ===========================================================================
# Test 19 — every labelsTr NRRD has a sibling .json with instances mapping
#
# TDD evidence: test written BEFORE fix; fails on OLD code because:
#   - OLD convert_case writes only the .nrrd label, no .json sidecar
#   - nnDetection v0.1 load.py reads seg_props_file = f"{stem}.json"
#     and expects {"instances": {"1": 0}}
# ===========================================================================


def test_export_dataset_per_case_json_sidecar(tmp_path: Path) -> None:
    """Every labelsTr/*.nrrd has a sibling .json with instances mapping.

    nnDetection v0.1 load.py:
        seg_props_file = f"{str(seg_file).split('.')[0]}.json"
        properties_json["instances"] = {str(k): int(v) ...}

    For single-lesion TDSC cases the correct content is:
        {"instances": {"1": 0}}
    where key "1" is the binary mask value (lesion voxels = 1) and
    value 0 is the class index (labels is {"0": "tumor"} — zero-indexed
    foreground, consistent with the fixed labels dict from defect 1).
    """
    tdsc_root = tmp_path / "tdsc"
    _build_synthetic_tdsc_root(
        tdsc_root,
        train_shards={"DATA00_49": [0, 1, 2]},
        val_ids=[100],
        test_ids=[200],
    )

    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=3,
        n_val_cases=1,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )

    export_dataset(str(tdsc_root), str(out_root), spec)

    task_dir = tmp_path / "out" / "Task001_TDSCABUS"
    labels_tr = task_dir / "raw_splitted" / "labelsTr"

    nrrd_files = sorted(labels_tr.glob("*.nrrd"))
    assert len(nrrd_files) == 3, f"Expected 3 labelsTr NRRD files, got {len(nrrd_files)}"

    for nrrd_file in nrrd_files:
        json_sidecar = nrrd_file.with_suffix(".json")
        assert json_sidecar.exists(), (
            f"Missing per-case JSON sidecar for {nrrd_file.name}. "
            "nnDetection v0.1 load.py reads seg_props_file = stem + '.json' "
            "and crashes if it is absent."
        )

        with open(json_sidecar, encoding="utf-8") as f:
            props = json.load(f)

        assert "instances" in props, (
            f"Per-case JSON {json_sidecar.name} must have 'instances' key. " f"Got: {props}"
        )
        assert props["instances"] == {"1": 0}, (
            f'Per-case JSON {json_sidecar.name} instances must be {{"1": 0}} '
            f"(instance value 1 -> class index 0 for single-lesion tumor). "
            f"Got: {props['instances']}"
        )


# ===========================================================================
# Test 20 — nnDetection v0.1 mini-replica check_dataset_file passes on output
#
# This is the regression test that would have caught the original server error.
# It re-implements the exact key+type checks from nndet/utils/check.py that
# fired with the ValueError on the server.
# ===========================================================================


def test_dataset_json_passes_nndet_check_replica(tmp_path: Path) -> None:
    """The produced dataset.json passes a mini-replica of nnDetection's check_dataset_file.

    Re-implements the key + type checks from nndet/utils/check.py at commit
    97a58f3110b71caf1b4bcc1851e67cf11e987fc5 (the pinned server version).

    The original server failure was:
        ValueError: Dataset information did not contain 'task' key,
        found ['_project','labels','modality','name','numTest','numTraining',
               'tensorImageSize','test','training']

    This test runs the same checks so any future schema regression is caught
    locally before reaching the server.
    """
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=100,
        n_val_cases=30,
        n_test_cases=70,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="single foreground class: tumor",
    )
    out_path = str(tmp_path / "dataset.json")
    build_dataset_json(spec, out_path)

    with open(out_path, encoding="utf-8") as f:
        cfg = json.load(f)

    def _check_key_missing(d: dict, key: str, ktype: type) -> None:
        """Mini-replica of nndet/utils/check.py::_check_key_missing."""
        if key not in d:
            raise ValueError(
                f"Dataset information did not contain {key!r} key, " f"found {sorted(d.keys())}"
            )
        if not isinstance(d[key], ktype):
            raise TypeError(
                f"Key {key!r} must be of type {ktype.__name__}, " f"got {type(d[key]).__name__}"
            )

    # These are the exact checks from nndet/utils/check.py check_dataset_file
    # that the server ran and failed on.
    _check_key_missing(cfg, "task", str)  # line 86 — the one that fired
    _check_key_missing(cfg, "dim", int)  # line 87
    _check_key_missing(cfg, "modalities", dict)  # line 88 (plural)
    _check_key_missing(cfg, "labels", dict)  # line 89

    # Semantic checks that follow the key presence checks
    assert cfg["dim"] == 3, f"dim must be 3 for 3D data, got {cfg['dim']}"
    assert "0" in cfg["modalities"], f"modalities must have key '0', got {cfg['modalities']}"
    # labels must be foreground-only (background is implicit in nnDetection)
    assert (
        "0" in cfg["labels"]
    ), f"labels must have at least key '0' (first foreground class), got {cfg['labels']}"
    assert "background" not in cfg["labels"].values(), (
        "Background must not appear as a label value in nnDetection v0.1 dataset.json; "
        f"got labels: {cfg['labels']}"
    )


# ===========================================================================
# Test 21 — splits_final.json content is bit-for-bit identical regardless of location
#
# This guards ASC-01_01.3 / Risk 3: the splits file CONTENT must be unchanged.
# Only the location can move (within the task dir).
# ===========================================================================


def test_splits_content_unchanged_after_layout_fix(tmp_path: Path) -> None:
    """The splits_final.json content is bit-for-bit identical before and after the layout fix.

    ASC-01_01.3 / Risk 3: only the FILE LOCATION may change (raw_splitted/ vs task root).
    The fold membership must be byte-faithfully preserved from the frozen manifest.

    This test writes splits to two different paths and asserts their JSON content
    is identical, proving the content is invariant to path changes.
    """
    from abus.data.split import load_split

    path_a = str(tmp_path / "splits_a.json")
    path_b = str(tmp_path / "subdir" / "splits_b.json")

    write_nndet_splits(path_a)
    write_nndet_splits(path_b)

    with open(path_a, encoding="utf-8") as f:
        content_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        content_b = json.load(f)

    assert content_a == content_b, (
        "splits_final.json content must be identical regardless of output path. "
        "Only the location may change; the fold membership is immutable."
    )

    # Also verify content matches frozen manifest (re-run faithfulness check)
    frozen = load_split()
    assert len(content_a) == len(
        frozen.folds
    ), f"Fold count mismatch: {len(content_a)} vs {len(frozen.folds)}"
    for k, entry in enumerate(content_a):
        assert sorted(int(x) for x in entry["train"]) == frozen.train_ids(k)
        assert sorted(int(x) for x in entry["val"]) == frozen.oof_ids(k)


# ===========================================================================
# Test 22 — convert_case raises NndetDatasetError for multi-lesion CSV rows
#           BEFORE writing the per-case JSON sidecar (S1 fix guard)
#
# TDD evidence: WRITTEN AFTER fix was applied (S1 adds the guard).
# The guard must fire before _write_per_case_json so no wrong-encoding sidecar
# is left on disk from an interrupted multi-lesion conversion.
# ===========================================================================


def test_convert_case_multi_lesion_raises_before_sidecar(tmp_path: Path) -> None:
    """convert_case raises NndetDatasetError for multi-lesion CSV rows (no sidecar written).

    The {"instances": {"1": 0}} encoding is valid ONLY for single-lesion binary masks.
    A multi-lesion case requires instance-index encoding (mask values 1, 2, 3, …).
    The guard must raise BEFORE writing the per-case JSON sidecar so a wrong-encoding
    .json is never left on disk from a failed conversion.
    """
    vol_path = str(tmp_path / "DATA_0010.nrrd")
    mask_path = str(tmp_path / "MASK_0010.nrrd")
    _write_identity_volume_nrrd(vol_path)
    lesion_box = BBox(2, 3, 4, 4, 6, 7)
    _write_identity_mask_nrrd(mask_path, lesion_box=lesion_box)

    out_image = str(tmp_path / "0010_0000.nrrd")
    out_label = str(tmp_path / "0010.nrrd")
    expected_sidecar = tmp_path / "0010.json"

    # Two CSV rows = two lesions
    row1 = _make_csv_row(10, lesion_box)
    row2 = _make_csv_row(10, BBox(1, 1, 1, 2, 2, 2))

    with pytest.raises(NndetDatasetError, match="multi-lesion"):
        convert_case(vol_path, mask_path, [row1, row2], out_image, out_label)

    # The sidecar must NOT exist — the guard fired before _write_per_case_json
    assert not expected_sidecar.exists(), (
        f"Per-case JSON sidecar {expected_sidecar} must NOT be written when "
        "convert_case raises for a multi-lesion case. "
        "The guard must fire BEFORE _write_per_case_json to avoid leaving a "
        "wrong-encoding sidecar on disk."
    )


# ===========================================================================
# Test 23 — verify_nndet_dataset raises NndetDatasetError if a sidecar is missing
#           (S2 fix check)
#
# TDD evidence: WRITTEN AFTER fix was applied (S2 adds the sidecar presence check).
# An interrupted export that leaves .nrrd without .json must be caught here,
# before nndet_prep on the server.
# ===========================================================================


def test_verify_nndet_dataset_detects_missing_sidecar(tmp_path: Path) -> None:
    """verify_nndet_dataset raises NndetDatasetError if a labelsTr .nrrd lacks a sibling .json.

    An interrupted or partial export_dataset run can leave label NRRDs without
    their per-case JSON sidecars. These cases would pass spacing and splits checks
    but fail at nndet_prep time on the server (nnDetection v0.1 load.py crashes).
    The sidecar presence check catches this offline before the runbook is executed.
    """
    # Build a minimal valid dataset so splits + spacing checks pass
    tdsc_root = tmp_path / "tdsc"
    _build_synthetic_tdsc_root(
        tdsc_root,
        train_shards={"DATA00_49": [0, 1]},
        val_ids=[100],
        test_ids=[200],
    )
    out_root = tmp_path / "out"
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=2,
        n_val_cases=1,
        n_test_cases=1,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="tumor",
    )
    result = export_dataset(str(tdsc_root), str(out_root), spec)
    task_dir = Path(result["task_dir"])

    # Confirm export succeeded and sidecars exist
    labels_tr = task_dir / "raw_splitted" / "labelsTr"
    sidecars = sorted(labels_tr.glob("*.json"))
    assert len(sidecars) == 2, f"Export should have produced 2 sidecars, got {len(sidecars)}"

    # Simulate a missing sidecar (e.g. interrupted export)
    sidecars[0].unlink()

    # verify_nndet_dataset must detect the missing sidecar
    with pytest.raises(NndetDatasetError, match="Missing per-case JSON sidecar"):
        verify_nndet_dataset(str(task_dir))


# ===========================================================================
# Test 24 — dataset.json contains all keys read by preprocess.py pipeline
#           (Round 3 fix: preprocess.py read contract — full coverage)
#
# TDD evidence: WRITTEN BEFORE fix is applied.
# The test FAILS on the current build_dataset_json because test_labels is absent.
# This is the regression gap that allowed the Round-2 server failure:
#   omegaconf.errors.ConfigKeyError: Missing key test_labels (preprocess.py line 394)
#
# The mini-replica in test_dataset_json_passes_nndet_check_replica (Test 20) only
# covers check_dataset_file's contract (task, dim, modalities, labels). preprocess.py
# reads additional keys DOWNSTREAM of check_dataset_file with no defaults.
#
# OmegaConf composition note (confirmed from server traceback preprocess.py 287–401):
#   nndet's OmegaConf composes cfg from dataset.json by nesting all non-'task'
#   keys under "data": cfg["task"] -> file["task"]; cfg["data"]["X"] -> file["X"].
#   Accesses confirmed from the traceback + user-provided code context at commit
#   97a58f3110b71caf1b4bcc1851e67cf11e987fc5:
#     line 287: data_info = cfg["data"]
#     line 289: data_info["modalities"]   (already in schema)
#     line 374: data_info["dim"]          (already in schema)
#     line 394: data_info["test_labels"]  (MISSING — this test catches it)
#   Also confirmed present in the check_dataset_file contract:
#     cfg["task"]           (already in schema)
#     cfg["data"]["labels"] (already in schema)
# ===========================================================================


def test_dataset_json_has_preprocess_read_contract(tmp_path: Path) -> None:
    """dataset.json contains ALL keys read by preprocess.py up to and including run().

    This is a STRONGER contract than test_dataset_json_passes_nndet_check_replica
    (Test 20), which only covers check_dataset_file. preprocess.py also reads
    cfg["data"]["test_labels"] (line 394) downstream, with no default.

    OmegaConf composition (confirmed from server traceback at commit 97a58f3):
        cfg["task"]             -> dataset.json key "task"
        cfg["data"]["X"]        -> dataset.json key "X" for all other keys

    Keys confirmed accessed by preprocess.py (source: server traceback + user-provided
    code context at commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5):
        cfg["task"]                    line ~86 check_dataset_file
        cfg["data"]["modalities"]      line 289
        cfg["data"]["dim"]             line 374
        cfg["data"]["labels"]          line ~89 check_dataset_file
        cfg["data"]["test_labels"]     line 394  <-- the one that failed on server

    Round-2 regression test only: checked keys validated by check_dataset_file.
    This test adds test_labels to the required key set so the full preprocess.py
    contract is covered locally.
    """
    spec = NndetDatasetSpec(
        task_id=1,
        task_name="Task001_TDSCABUS",
        n_train_cases=100,
        n_val_cases=30,
        n_test_cases=70,
        spacing_mm=CANONICAL_SPACING_MM,
        modality="US",
        label_semantics="single foreground class: tumor",
    )
    out_path = str(tmp_path / "dataset.json")
    build_dataset_json(spec, out_path)

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    # Keys confirmed accessed by check_dataset_file (already covered by Test 20)
    for required_key in ("task", "dim", "modalities", "labels"):
        assert required_key in data, (
            f"dataset.json must contain '{required_key}' "
            f"(check_dataset_file contract); found: {sorted(data.keys())}"
        )

    # Key confirmed accessed DOWNSTREAM of check_dataset_file in preprocess.py
    # at line 394: if cfg["data"]["test_labels"]: — no default, raises ConfigKeyError.
    # Value must be False: val/test ground truth lives in TDSC source, accessed
    # via our own evaluation pipeline, NOT via nnDetection's labelsTs/ mechanism.
    # Setting True would cause nndet_prep to call check_data_and_label_splitted(
    # test=True, labels=True) and demand labelsTs/<case>.nrrd + .json, which we
    # do not provide (thesis §3.2 held-out evaluation policy).
    assert "test_labels" in data, (
        "dataset.json must contain 'test_labels' key (preprocess.py line 394 reads "
        "cfg['data']['test_labels'] with no default; missing key raises "
        "omegaconf.errors.ConfigKeyError). This was the Round-3 server failure. "
        f"Found keys: {sorted(data.keys())}"
    )
    # Type check first: a string "false" would pass the is-False check on some Python versions
    # but is NOT a valid JSON boolean — the more informative error comes from the isinstance check.
    assert isinstance(data["test_labels"], bool), (
        f"test_labels must be a JSON boolean (bool), got {type(data['test_labels']).__name__}. "
        "A string 'false' serialised into JSON would be read as a str by json.load, "
        "not a bool, and OmegaConf does not coerce JSON strings to bool in this context."
    )
    assert data["test_labels"] is False, (
        f"test_labels must be False (no labelsTs/ provided; thesis §3.2 held-out eval). "
        f"Got: {data['test_labels']!r}"
    )
