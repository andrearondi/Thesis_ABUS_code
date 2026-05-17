"""Axis-aligned 3D bounding box type and operators for ABUS.

Project-wide convention (fixed here, enforced by STORY_00_03):
  - Axes are NRRD storage axes: d0, d1, d2 (numpy array-indexing order).
  - Units are integer voxel indices.
  - Both endpoints are INCLUSIVE: the voxel at index max IS inside the box.
  - Layout: (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2).

This convention was empirically verified to 0.0-voxel round-trip residual on
6 cases from the TDSC-ABUS-2023 validation split (docs/local_data.md,
docs/local_data_check/verification_report.json, 2026-05-14).

All operators (iou_3d, volume, shape_stats, clip, contains) work in this
storage-axis, inclusive-max representation. Conversion to/from other
conventions (ITK CSV order, nnDetection format) lives exclusively in
abus.geometry.convert — callers never touch raw ITK tuples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned 3D bounding box.

    CONVENTION (project-wide, fixed in STORY_00_03):
      - Axes are NRRD storage axes (d0, d1, d2).
      - Units are integer voxel indices.
      - Both endpoints are INCLUSIVE: the voxel at index max IS inside the box.
      - Layout: (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2).

    Attributes
    ----------
    min_d0, min_d1, min_d2:
        Lower-corner voxel indices (inclusive) along each storage axis.
    max_d0, max_d1, max_d2:
        Upper-corner voxel indices (inclusive) along each storage axis.
    """

    min_d0: int
    min_d1: int
    min_d2: int
    max_d0: int
    max_d1: int
    max_d2: int

    def __post_init__(self) -> None:
        """Validate that min_di <= max_di on every axis."""
        if self.min_d0 > self.max_d0 or self.min_d1 > self.max_d1 or self.min_d2 > self.max_d2:
            raise ValueError(
                f"BBox min > max: "
                f"d0 ({self.min_d0}>{self.max_d0}), "
                f"d1 ({self.min_d1}>{self.max_d1}), "
                f"d2 ({self.min_d2}>{self.max_d2}). "
                "A bounding box must have min_di <= max_di on every axis."
            )

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        """Return (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)."""
        return (self.min_d0, self.min_d1, self.min_d2, self.max_d0, self.max_d1, self.max_d2)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int, int, int, int]) -> BBox:
        """Construct from (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)."""
        return cls(t[0], t[1], t[2], t[3], t[4], t[5])


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def volume(b: BBox) -> int:
    """Voxel count = prod(max_di - min_di + 1).

    The +1 arises because max IS inside the box (inclusive endpoint).
    A single-voxel box where min == max has volume 1, not 0.
    """
    return (b.max_d0 - b.min_d0 + 1) * (b.max_d1 - b.min_d1 + 1) * (b.max_d2 - b.min_d2 + 1)


def extent(b: BBox) -> tuple[int, int, int]:
    """Voxel count per axis: (max_di - min_di + 1) for each i.

    Returns
    -------
    (ext_d0, ext_d1, ext_d2) in voxels.
    """
    return (b.max_d0 - b.min_d0 + 1, b.max_d1 - b.min_d1 + 1, b.max_d2 - b.min_d2 + 1)


def center(b: BBox) -> tuple[float, float, float]:
    """Voxel-coordinate center of the box.

    Returns ((min_di + max_di) / 2) per axis; may be half-integer for
    even-sized extents.
    """
    return (
        (b.min_d0 + b.max_d0) / 2.0,
        (b.min_d1 + b.max_d1) / 2.0,
        (b.min_d2 + b.max_d2) / 2.0,
    )


def iou_3d(a: BBox, b: BBox) -> float:
    """3D intersection-over-union in voxel units (inclusive-max convention).

    Intersection extent per axis:
        max(0, min(a.max_di, b.max_di) - max(a.min_di, b.min_di) + 1)

    The +1 is required by the inclusive-max convention: two boxes that
    share a single face column of voxels (min of one == max of other)
    have a non-zero intersection.

    Returns 0.0 when the boxes do not overlap.  Returns 1.0 when a == b.

    This is the operator used for the IoU > 0.30 candidate-lesion matching
    (thesis §3.2.2) and for NMS (EPIC_01).
    """
    inter_d0 = max(0, min(a.max_d0, b.max_d0) - max(a.min_d0, b.min_d0) + 1)
    inter_d1 = max(0, min(a.max_d1, b.max_d1) - max(a.min_d1, b.min_d1) + 1)
    inter_d2 = max(0, min(a.max_d2, b.max_d2) - max(a.min_d2, b.min_d2) + 1)
    inter = inter_d0 * inter_d1 * inter_d2
    if inter == 0:
        return 0.0
    vol_a = volume(a)
    vol_b = volume(b)
    return inter / (vol_a + vol_b - inter)


def shape_stats(b: BBox, spacing_mm: tuple[float, float, float]) -> dict[str, float]:
    """Physical shape statistics for a BBox — node features for the GNN.

    Computes per-axis physical edge lengths from the inclusive voxel extent
    and the injected physical spacing.

    Parameters
    ----------
    b:
        The bounding box (storage-axis order, inclusive-max).
    spacing_mm:
        Physical voxel spacing (d0, d1, d2) in millimetres (e.g.
        CANONICAL_SPACING_MM = (0.073, 0.200, 0.475674)).

    Returns
    -------
    dict with keys:
      'volume_mm3'  — physical volume in mm³
      'elongation'  — longest / shortest physical edge (≥ 1.0)
      'anisotropy'  — longest / middle physical edge (≥ 1.0)

    These are the per-candidate shape statistics node features of
    thesis §3.2.8.
    """
    ext = extent(b)
    edges = sorted(
        [ext[0] * spacing_mm[0], ext[1] * spacing_mm[1], ext[2] * spacing_mm[2]]
    )  # ascending
    shortest, mid, longest = edges[0], edges[1], edges[2]
    return {
        "volume_mm3": shortest * mid * longest,
        "elongation": longest / shortest if shortest > 0 else float("inf"),
        "anisotropy": longest / mid if mid > 0 else float("inf"),
    }


def clip(b: BBox, shape: tuple[int, int, int]) -> BBox:
    """Clip a BBox to [0, shape_di - 1] per axis (inclusive bounds).

    Parameters
    ----------
    b:
        The bounding box to clip.
    shape:
        The volume shape (d0, d1, d2) in voxels; each axis is clipped to
        [0, shape[i] - 1].

    Returns
    -------
    A new BBox with all coordinates in the valid range.
    """
    return BBox(
        min_d0=max(0, b.min_d0),
        min_d1=max(0, b.min_d1),
        min_d2=max(0, b.min_d2),
        max_d0=min(shape[0] - 1, b.max_d0),
        max_d1=min(shape[1] - 1, b.max_d1),
        max_d2=min(shape[2] - 1, b.max_d2),
    )


def contains(b: BBox, point_voxel: tuple[int, int, int]) -> bool:
    """Return True if the voxel index lies within the inclusive box.

    Parameters
    ----------
    b:
        The bounding box (inclusive endpoints).
    point_voxel:
        (d0, d1, d2) integer voxel index to test.
    """
    d0, d1, d2 = point_voxel
    return b.min_d0 <= d0 <= b.max_d0 and b.min_d1 <= d1 <= b.max_d1 and b.min_d2 <= d2 <= b.max_d2
