"""Tests for abus.geometry.convert and abus.data.labels (STORY_00_03).

Covers:
  - csv_itk_to_bbox / bbox_to_csv_itk round-trip (hypothesis property test)
  - bbox_to_nndet / nndet_to_bbox round-trip (hypothesis property test)
  - voxel_to_mm / mm_to_voxel round-trip
  - csv_itk_to_bbox on real case-100 CSV values (reference from verification_report.json)
  - bbox_center_mm on a known box
  - load_gt_bboxes: reads local bbx_labels.csv, 30 boxes, case 100 matches reference
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Deferred imports
# ---------------------------------------------------------------------------


def _import_convert():  # type: ignore[no-untyped-def]
    from abus.geometry import convert as conv_mod  # noqa: PLC0415

    return conv_mod


def _import_bbox():  # type: ignore[no-untyped-def]
    from abus.geometry import bbox as bbox_mod  # noqa: PLC0415

    return bbox_mod


# ---------------------------------------------------------------------------
# Local data paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
_BBX_CSV = _DATA_ROOT / "bbx_labels.csv"
_LOCAL_DATA_AVAILABLE = _BBX_CSV.exists()

_skip_if_no_data = pytest.mark.skipif(
    not _LOCAL_DATA_AVAILABLE,
    reason="Local dataset bbx_labels.csv not found",
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for random BBoxes
# ---------------------------------------------------------------------------

# Small integers to keep arithmetic fast and avoid overflow.
_COORD = st.integers(min_value=0, max_value=1000)


@st.composite
def valid_bbox_strategy(draw: st.DrawFn):  # type: ignore[no-untyped-def]
    """Generate a random BBox where min_di <= max_di on every axis."""
    bbox_mod = _import_bbox()
    mn0 = draw(_COORD)
    mn1 = draw(_COORD)
    mn2 = draw(_COORD)
    ext0 = draw(st.integers(min_value=0, max_value=500))
    ext1 = draw(st.integers(min_value=0, max_value=500))
    ext2 = draw(st.integers(min_value=0, max_value=500))
    return bbox_mod.BBox(mn0, mn1, mn2, mn0 + ext0, mn1 + ext1, mn2 + ext2)


# ---------------------------------------------------------------------------
# csv_itk_to_bbox round-trip (hypothesis)
# ---------------------------------------------------------------------------


class TestCsvItkRoundTrip:
    """csv_itk_to_bbox ∘ bbox_to_csv_itk must be the identity for every BBox."""

    @given(b=valid_bbox_strategy())
    @settings(max_examples=200)
    def test_round_trip_csv_itk(self, b) -> None:  # type: ignore[no-untyped-def]
        conv = _import_convert()
        c_xyz, len_xyz = conv.bbox_to_csv_itk(b)
        b2 = conv.csv_itk_to_bbox(c_xyz, len_xyz)
        assert b == b2, f"Round-trip mismatch: {b} -> csv_itk -> {b2}"


# ---------------------------------------------------------------------------
# bbox_to_nndet round-trip (hypothesis)
# ---------------------------------------------------------------------------


class TestNndetRoundTrip:
    """bbox_to_nndet ∘ nndet_to_bbox must be the identity for every BBox."""

    @given(b=valid_bbox_strategy())
    @settings(max_examples=200)
    def test_round_trip_nndet(self, b) -> None:  # type: ignore[no-untyped-def]
        conv = _import_convert()
        t = conv.bbox_to_nndet(b)
        b2 = conv.nndet_to_bbox(t)
        assert b == b2, f"Round-trip mismatch: {b} -> nndet -> {b2}"

    def test_nndet_format_is_y1x1y2x2z1z2_exclusive(self) -> None:
        """bbox_to_nndet must produce (y1, x1, y2, x2, z1, z2) with exclusive upper bound.

        Convention: nnDetection / MDT use (y1, x1, y2, x2, z1, z2) on the resampled grid
        with EXCLUSIVE upper bounds (numpy-style slicing).
        Storage axes: d0->y, d1->x, d2->z.
        Exclusive upper bound: max_nndet = max_bbox + 1.

        Reference: STORY_00_03 spec, decision D00.4 in EPIC_00.
        Source confirmed at implementation time against pinned nnDetection release.
        """
        bbox_mod = _import_bbox()
        conv = _import_convert()
        b = bbox_mod.BBox(10, 20, 30, 40, 50, 60)
        t = conv.bbox_to_nndet(b)
        # d0 -> y: y1=10, y2=41 (exclusive)
        # d1 -> x: x1=20, x2=51 (exclusive)
        # d2 -> z: z1=30, z2=61 (exclusive)
        # nndet format: (y1, x1, y2, x2, z1, z2)
        assert t == (10, 20, 41, 51, 30, 61), f"Expected (10,20,41,51,30,61) got {t}"


# ---------------------------------------------------------------------------
# voxel_to_mm / mm_to_voxel round-trip
# ---------------------------------------------------------------------------


class TestVoxelMmRoundTrip:
    def test_round_trip_identity_spacing(self) -> None:
        conv = _import_convert()
        point = (10.0, 20.0, 30.0)
        spacing = (1.0, 1.0, 1.0)
        assert conv.mm_to_voxel(conv.voxel_to_mm(point, spacing), spacing) == pytest.approx(point)

    def test_round_trip_canonical_spacing(self) -> None:
        from abus.io.loader import CANONICAL_SPACING_MM  # noqa: PLC0415

        conv = _import_convert()
        point = (100.0, 200.0, 50.0)
        assert conv.mm_to_voxel(
            conv.voxel_to_mm(point, CANONICAL_SPACING_MM), CANONICAL_SPACING_MM
        ) == pytest.approx(point)

    def test_voxel_to_mm_known(self) -> None:
        """voxel_to_mm: origin + index * spacing, per axis."""
        conv = _import_convert()
        point = (2.0, 3.0, 4.0)
        spacing = (0.5, 1.0, 2.0)
        origin = (1.0, 0.0, -1.0)
        mm = conv.voxel_to_mm(point, spacing, origin)
        assert mm == pytest.approx((1.0 + 2.0 * 0.5, 0.0 + 3.0 * 1.0, -1.0 + 4.0 * 2.0))

    def test_mm_to_voxel_known(self) -> None:
        """mm_to_voxel: (mm - origin) / spacing, per axis."""
        conv = _import_convert()
        mm_point = (2.0, 3.0, 4.0)
        spacing = (0.5, 1.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        vox = conv.mm_to_voxel(mm_point, spacing, origin)
        assert vox == pytest.approx((4.0, 3.0, 2.0))


# ---------------------------------------------------------------------------
# csv_itk_to_bbox on case-100 reference values
# ---------------------------------------------------------------------------


class TestCsvCase100Reference:
    """csv_itk_to_bbox on case-100 CSV row must yield the reference storage-order bbox.

    Reference (docs/local_data.md and verification_report.json, 0.0-voxel residual):
      CSV row 100: c_x=189.0, c_y=242.0, c_z=314.0, len_x=72.0, len_y=368.0, len_z=302.0
      After (2,1,0) permutation:
        min (d0,d1,d2) = (163, 58, 153)
        max (d0,d1,d2) = (465, 426, 225)   (inclusive)
    """

    def test_case100_csv_to_storage_bbox(self) -> None:
        conv = _import_convert()
        bbox_mod = _import_bbox()
        c_xyz = (189.0, 242.0, 314.0)
        len_xyz = (72.0, 368.0, 302.0)
        b = conv.csv_itk_to_bbox(c_xyz, len_xyz)
        expected = bbox_mod.BBox(163, 58, 153, 465, 426, 225)
        assert b == expected, f"Expected {expected}, got {b}"

    def test_case100_round_trip_is_exact(self) -> None:
        conv = _import_convert()
        c_xyz = (189.0, 242.0, 314.0)
        len_xyz = (72.0, 368.0, 302.0)
        b = conv.csv_itk_to_bbox(c_xyz, len_xyz)
        c2, len2 = conv.bbox_to_csv_itk(b)
        assert c2 == pytest.approx(c_xyz)
        assert len2 == pytest.approx(len_xyz)


# ---------------------------------------------------------------------------
# bbox_center_mm
# ---------------------------------------------------------------------------


class TestBboxCenterMm:
    def test_center_mm_known(self) -> None:
        """bbox_center_mm((0,0,0,1,1,1), (1,1,1)) = (0.5, 0.5, 0.5)."""
        conv = _import_convert()
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 1)
        spacing = (1.0, 1.0, 1.0)
        c = conv.bbox_center_mm(b, spacing)
        assert c == pytest.approx((0.5, 0.5, 0.5))

    def test_center_mm_with_origin(self) -> None:
        conv = _import_convert()
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 3, 3, 3)  # center voxel = (1.5, 1.5, 1.5)
        spacing = (2.0, 2.0, 2.0)
        origin = (10.0, 0.0, -5.0)
        c = conv.bbox_center_mm(b, spacing, origin)
        # center_voxel * spacing + origin = (1.5*2+10, 1.5*2+0, 1.5*2-5) = (13, 3, -2)
        assert c == pytest.approx((13.0, 3.0, -2.0))


# ---------------------------------------------------------------------------
# load_gt_bboxes (real data — skipped if dataset absent)
# ---------------------------------------------------------------------------


@_skip_if_no_data
class TestLoadGtBboxes:
    """load_gt_bboxes reads bbx_labels.csv and applies the (2,1,0) permutation."""

    def test_returns_dict_of_bboxes(self) -> None:
        from abus.data.labels import load_gt_bboxes  # noqa: PLC0415
        from abus.geometry.bbox import BBox  # noqa: PLC0415

        bboxes = load_gt_bboxes(str(_BBX_CSV))
        assert isinstance(bboxes, dict)
        assert len(bboxes) == 30, f"Expected 30 boxes (validation split), got {len(bboxes)}"
        for case_id, b in bboxes.items():
            assert isinstance(case_id, int)
            assert isinstance(b, BBox)

    def test_case100_matches_reference(self) -> None:
        """load_gt_bboxes case 100 must match the storage-order bbox in verification_report.json."""
        from abus.data.labels import load_gt_bboxes  # noqa: PLC0415
        from abus.geometry.bbox import BBox  # noqa: PLC0415

        bboxes = load_gt_bboxes(str(_BBX_CSV))
        assert 100 in bboxes, "Case 100 not found in loaded bboxes"
        b = bboxes[100]
        expected = BBox(163, 58, 153, 465, 426, 225)
        assert b == expected, (
            f"Case 100 bbox mismatch: expected {expected}, got {b}. "
            "Check (2,1,0) permutation and inclusive-max at I/O boundary."
        )

    def test_all_boxes_have_non_negative_coords(self) -> None:
        from abus.data.labels import load_gt_bboxes  # noqa: PLC0415

        bboxes = load_gt_bboxes(str(_BBX_CSV))
        for case_id, b in bboxes.items():
            assert b.min_d0 >= 0, f"Case {case_id}: min_d0 < 0"
            assert b.min_d1 >= 0, f"Case {case_id}: min_d1 < 0"
            assert b.min_d2 >= 0, f"Case {case_id}: min_d2 < 0"

    def test_all_boxes_have_min_le_max(self) -> None:
        from abus.data.labels import load_gt_bboxes  # noqa: PLC0415

        bboxes = load_gt_bboxes(str(_BBX_CSV))
        for case_id, b in bboxes.items():
            assert b.min_d0 <= b.max_d0, f"Case {case_id}: min_d0 > max_d0"
            assert b.min_d1 <= b.max_d1, f"Case {case_id}: min_d1 > max_d1"
            assert b.min_d2 <= b.max_d2, f"Case {case_id}: min_d2 > max_d2"
