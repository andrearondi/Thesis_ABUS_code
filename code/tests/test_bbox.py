"""Tests for abus.geometry.bbox (STORY_00_03).

Covers:
  - BBox construction and as_tuple / from_tuple
  - volume (inclusive-max convention; regression locks it)
  - extent, center
  - iou_3d (self, disjoint, known analytic value)
  - shape_stats (elongation, anisotropy)
  - clip
  - contains
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Deferred import so collection succeeds even before bbox.py exists.
# ---------------------------------------------------------------------------


def _import_bbox():  # type: ignore[no-untyped-def]
    from abus.geometry import bbox as bbox_mod  # noqa: PLC0415

    return bbox_mod


# ---------------------------------------------------------------------------
# BBox construction
# ---------------------------------------------------------------------------


class TestBBoxConstruction:
    """BBox is a frozen dataclass; as_tuple / from_tuple must round-trip."""

    def test_as_tuple_order(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(1, 2, 3, 4, 5, 6)
        assert b.as_tuple() == (1, 2, 3, 4, 5, 6)

    def test_from_tuple(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox.from_tuple((10, 20, 30, 40, 50, 60))
        assert b.min_d0 == 10
        assert b.min_d1 == 20
        assert b.min_d2 == 30
        assert b.max_d0 == 40
        assert b.max_d1 == 50
        assert b.max_d2 == 60

    def test_frozen(self) -> None:
        """BBox must be immutable (frozen dataclass)."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 1)
        with pytest.raises((AttributeError, TypeError)):
            b.min_d0 = 99  # type: ignore[misc]

    def test_invalid_min_gt_max_raises(self) -> None:
        """BBox with min > max on any axis must raise ValueError."""
        bbox_mod = _import_bbox()
        with pytest.raises(ValueError, match="min > max"):
            bbox_mod.BBox(5, 0, 0, 3, 10, 10)  # min_d0=5 > max_d0=3

    def test_equal_min_max_is_valid(self) -> None:
        """min == max on an axis is valid (single-voxel on that axis)."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(5, 5, 5, 5, 5, 5)  # all equal: single voxel box
        assert bbox_mod.volume(b) == 1


# ---------------------------------------------------------------------------
# volume — inclusive-max convention
# ---------------------------------------------------------------------------


class TestVolume:
    """volume = prod(max_di - min_di + 1), the +1 because max IS inside the box."""

    def test_unit_box_has_volume_1(self) -> None:
        """A box where min == max on every axis contains exactly 1 voxel."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(5, 5, 5, 5, 5, 5)
        assert bbox_mod.volume(b) == 1

    def test_2x2x2_box_has_volume_8(self) -> None:
        """BBox (0,0,0,1,1,1) spans 2 voxels per axis -> 8 total."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 1)
        assert bbox_mod.volume(b) == 8

    def test_known_volume(self) -> None:
        """(0,0,0,2,3,4) -> extents (3,4,5) -> volume 60."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 2, 3, 4)
        assert bbox_mod.volume(b) == 3 * 4 * 5

    def test_inclusive_max_regression(self) -> None:
        """REGRESSION LOCK: volume with exclusive-max gives WRONG answer.

        This test DELIBERATELY checks that computing volume as
        prod(max - min)  — WITHOUT the +1 — gives a DIFFERENT value
        from the correct inclusive formula. This locks the convention:
        if someone silently changes volume() to use exclusive max, this
        test will flip from 'wrong value != correct value' to
        'wrong value == correct value', which would cause the test to fail.

        In other words: the exclusive interpretation is provably wrong
        here, so the convention cannot silently change back.
        """
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 1)
        correct_inclusive = bbox_mod.volume(b)  # must be 8
        # If someone accidentally computes exclusive volume: (1-0)*(1-0)*(1-0) = 1
        exclusive_wrong = (b.max_d0 - b.min_d0) * (b.max_d1 - b.min_d1) * (b.max_d2 - b.min_d2)
        assert correct_inclusive == 8, f"volume() returned {correct_inclusive}, expected 8"
        assert exclusive_wrong == 1, (
            "Sanity: exclusive formula should give 1 for this box, got " f"{exclusive_wrong}"
        )
        assert correct_inclusive != exclusive_wrong, (
            "CONVENTION BUG: inclusive and exclusive formulas agree — "
            "volume() is using the exclusive (wrong) formula."
        )


# ---------------------------------------------------------------------------
# extent
# ---------------------------------------------------------------------------


class TestExtent:
    def test_unit_box_extent(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(5, 5, 5, 5, 5, 5)
        assert bbox_mod.extent(b) == (1, 1, 1)

    def test_known_extent(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(1, 2, 3, 4, 6, 8)
        # (4-1+1, 6-2+1, 8-3+1) = (4, 5, 6)
        assert bbox_mod.extent(b) == (4, 5, 6)


# ---------------------------------------------------------------------------
# center
# ---------------------------------------------------------------------------


class TestCenter:
    def test_symmetric_box_center(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 4, 6, 8)
        c = bbox_mod.center(b)
        assert c == (2.0, 3.0, 4.0)

    def test_half_integer_center(self) -> None:
        """Odd-extent boxes have half-integer centers."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 2)  # extents (2, 2, 3)
        c = bbox_mod.center(b)
        assert c == (0.5, 0.5, 1.0)


# ---------------------------------------------------------------------------
# iou_3d
# ---------------------------------------------------------------------------


class TestIou3d:
    def test_iou_self_is_one(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(10, 20, 30, 50, 60, 70)
        assert bbox_mod.iou_3d(b, b) == pytest.approx(1.0)

    def test_iou_disjoint_is_zero(self) -> None:
        bbox_mod = _import_bbox()
        a = bbox_mod.BBox(0, 0, 0, 5, 5, 5)
        b = bbox_mod.BBox(10, 10, 10, 15, 15, 15)
        assert bbox_mod.iou_3d(a, b) == 0.0

    def test_iou_adjacent_gap_zero(self) -> None:
        """Two boxes separated by a 1-voxel gap along d0: IoU must be 0.
        a.max_d0=5, b.min_d0=6 — there is a gap of 1 voxel; no shared voxels."""
        bbox_mod = _import_bbox()
        a = bbox_mod.BBox(0, 0, 0, 5, 5, 5)
        b = bbox_mod.BBox(6, 0, 0, 10, 5, 5)
        assert bbox_mod.iou_3d(a, b) == 0.0

    def test_iou_adjacent_touching_inclusive(self) -> None:
        """Two boxes that share exactly one column of voxels along d0.
        a.max_d0=5, b.min_d0=5 — with inclusive max they share the d0=5 plane.
        intersection extent along d0 = min(5,10) - max(0,5) + 1 = 5-5+1 = 1."""
        bbox_mod = _import_bbox()
        a = bbox_mod.BBox(0, 0, 0, 5, 5, 5)  # extent (6,6,6) = 216 voxels
        b = bbox_mod.BBox(5, 0, 0, 10, 5, 5)  # extent (6,6,6) = 216 voxels
        # intersection: d0 extent=1, d1 extent=6, d2 extent=6 -> 36 voxels
        # union = 216 + 216 - 36 = 396
        expected = 36.0 / 396.0
        assert bbox_mod.iou_3d(a, b) == pytest.approx(expected, rel=1e-9)

    def test_iou_known_analytic(self) -> None:
        """Hand-computed IoU for a known overlapping pair.

        a = (0,0,0,3,3,3): extent (4,4,4), volume 64
        b = (2,2,2,5,5,5): extent (4,4,4), volume 64
        intersection per axis: max(0, min(3,5) - max(0,2) + 1) = max(0, 3-2+1) = 2
        intersection volume = 2*2*2 = 8
        union = 64 + 64 - 8 = 120
        IoU = 8/120 = 1/15
        """
        bbox_mod = _import_bbox()
        a = bbox_mod.BBox(0, 0, 0, 3, 3, 3)
        b = bbox_mod.BBox(2, 2, 2, 5, 5, 5)
        expected = 8.0 / 120.0
        assert bbox_mod.iou_3d(a, b) == pytest.approx(expected, rel=1e-9)

    def test_iou_partial_overlap(self) -> None:
        """a fully contains b: IoU = volume(b) / volume(a)."""
        bbox_mod = _import_bbox()
        a = bbox_mod.BBox(0, 0, 0, 9, 9, 9)  # 10^3 = 1000 voxels
        b = bbox_mod.BBox(1, 1, 1, 4, 4, 4)  # 4^3 = 64 voxels
        vol_b = bbox_mod.volume(b)
        vol_a = bbox_mod.volume(a)
        expected = vol_b / vol_a  # union = vol_a because b inside a
        assert bbox_mod.iou_3d(a, b) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# shape_stats
# ---------------------------------------------------------------------------


class TestShapeStats:
    """shape_stats returns volume_mm3, elongation, anisotropy for a BBox."""

    def test_cube_elongation_is_one(self) -> None:
        """A cube (equal extents * equal spacing) has elongation = 1 and anisotropy = 1."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 9, 9, 9)  # extent (10,10,10) in voxels
        spacing = (1.0, 1.0, 1.0)
        stats = bbox_mod.shape_stats(b, spacing)
        assert stats["elongation"] == pytest.approx(1.0)
        assert stats["anisotropy"] == pytest.approx(1.0)

    def test_volume_mm3_known(self) -> None:
        """Physical volume = product of physical extents per axis."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(0, 0, 0, 1, 1, 1)  # extent (2,2,2) in voxels
        spacing = (0.5, 1.0, 2.0)
        # physical extents: 2*0.5=1.0, 2*1.0=2.0, 2*2.0=4.0
        # volume_mm3 = 1.0 * 2.0 * 4.0 = 8.0
        stats = bbox_mod.shape_stats(b, spacing)
        assert stats["volume_mm3"] == pytest.approx(8.0)

    def test_elongation_anisotropy_known(self) -> None:
        """Anisotropic box with known physical edges: elongation and anisotropy analytic."""
        bbox_mod = _import_bbox()
        # extent (2,4,8) in voxels, spacing (1,1,1) -> physical (2,4,8)
        b = bbox_mod.BBox(0, 0, 0, 1, 3, 7)
        spacing = (1.0, 1.0, 1.0)
        stats = bbox_mod.shape_stats(b, spacing)
        # sorted physical edges: [2, 4, 8]
        # elongation = longest / shortest = 8/2 = 4.0
        # anisotropy = longest / middle = 8/4 = 2.0
        assert stats["elongation"] == pytest.approx(4.0)
        assert stats["anisotropy"] == pytest.approx(2.0)

    def test_shape_stats_with_canonical_spacing(self) -> None:
        """shape_stats using CANONICAL_SPACING_MM produces a physically meaningful result."""
        bbox_mod = _import_bbox()
        from abus.io.loader import CANONICAL_SPACING_MM  # noqa: PLC0415

        # Case 100 bbox: extent (303, 369, 73) voxels (from verification_report.json)
        b = bbox_mod.BBox(163, 58, 153, 465, 426, 225)
        stats = bbox_mod.shape_stats(b, CANONICAL_SPACING_MM)
        # physical extents: 303*0.073=22.119, 369*0.200=73.8, 73*0.475674=34.724
        # sorted: [22.119, 34.724, 73.8]
        # elongation = 73.8/22.119 ≈ 3.337
        # anisotropy = 73.8/34.724 ≈ 2.125
        assert stats["elongation"] == pytest.approx(73.8 / 22.119, rel=1e-3)
        assert stats["anisotropy"] == pytest.approx(73.8 / 34.724, rel=1e-3)
        assert stats["volume_mm3"] > 0


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------


class TestClip:
    def test_clip_identity_when_inside(self) -> None:
        """A box fully inside the volume is unchanged."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(1, 1, 1, 8, 8, 8)
        shape = (10, 10, 10)
        clipped = bbox_mod.clip(b, shape)
        assert clipped == b

    def test_clip_truncates_to_valid_range(self) -> None:
        """A box extending beyond the volume is clipped to [0, shape_di - 1]."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(-2, -3, 0, 15, 20, 8)
        shape = (10, 10, 10)
        clipped = bbox_mod.clip(b, shape)
        assert clipped.min_d0 == 0
        assert clipped.min_d1 == 0
        assert clipped.min_d2 == 0
        assert clipped.max_d0 == 9  # shape[0] - 1
        assert clipped.max_d1 == 9
        assert clipped.max_d2 == 8  # was already in range

    def test_clip_partially_overlapping_box(self) -> None:
        """A box that partially overlaps the volume clips to the overlap region."""
        bbox_mod = _import_bbox()
        # box starts at d0=7, extends to d0=12; shape has 10 voxels (0..9)
        b = bbox_mod.BBox(7, 0, 0, 12, 5, 5)
        shape = (10, 10, 10)
        clipped = bbox_mod.clip(b, shape)
        assert clipped.min_d0 == 7  # was in range
        assert clipped.max_d0 == 9  # clipped to shape[0]-1
        assert clipped.min_d0 <= clipped.max_d0  # result must be valid


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


class TestContains:
    def test_center_voxel_is_contained(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(2, 2, 2, 8, 8, 8)
        assert bbox_mod.contains(b, (5, 5, 5)) is True

    def test_corner_voxels_are_contained(self) -> None:
        """All eight corners of the inclusive box must be inside."""
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(1, 2, 3, 5, 6, 7)
        for d0 in (b.min_d0, b.max_d0):
            for d1 in (b.min_d1, b.max_d1):
                for d2 in (b.min_d2, b.max_d2):
                    assert (
                        bbox_mod.contains(b, (d0, d1, d2)) is True
                    ), f"Corner ({d0},{d1},{d2}) should be inside {b}"

    def test_point_outside_is_not_contained(self) -> None:
        bbox_mod = _import_bbox()
        b = bbox_mod.BBox(2, 2, 2, 8, 8, 8)
        assert bbox_mod.contains(b, (9, 5, 5)) is False
        assert bbox_mod.contains(b, (5, 1, 5)) is False
        assert bbox_mod.contains(b, (5, 5, 9)) is False
