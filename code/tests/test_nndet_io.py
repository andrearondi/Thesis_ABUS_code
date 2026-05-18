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

    # Write a minimal dataset.json so directory looks like a Task
    ds_json = {
        "name": "Task001_TDSCABUS",
        "tensorImageSize": "3D",
        "modality": {"0": "US"},
        "labels": {"0": "background", "1": "tumor"},
        "numTraining": 0,
        "numTest": 0,
        "training": [],
        "test": [],
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
    """build_dataset_json writes a JSON file with required nnDetection keys."""
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

    # Required nnDetection dataset.json keys
    assert "name" in data, "dataset.json must have 'name'"
    assert "modality" in data, "dataset.json must have 'modality'"
    assert "labels" in data, "dataset.json must have 'labels'"
    assert "numTraining" in data, "dataset.json must have 'numTraining'"
    assert data["numTraining"] == 100
    assert "0" in data["labels"], "Background class '0' must be present"
    assert "1" in data["labels"], "Foreground class '1' must be present"


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
