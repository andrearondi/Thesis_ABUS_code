"""Coordinate-convention conversion functions for ABUS (STORY_00_03).

All functions translate between the project's internal BBox convention
(storage-axis order, integer voxels, inclusive-max) and external formats.

The only functions that cross the ITK/CSV boundary are:
  csv_itk_to_bbox   — CSV row (c_xyz, len_xyz) -> BBox
  bbox_to_csv_itk   — BBox -> (c_xyz, len_xyz)  (self-inverse permutation)

The only functions that cross the nnDetection boundary are:
  bbox_to_nndet   — BBox -> nnDetection (y1, x1, y2, x2, z1, z2) tuple
  nndet_to_bbox   — nnDetection tuple -> BBox

Physical-frame functions:
  voxel_to_mm     — storage-axis voxel index -> physical mm
  mm_to_voxel     — physical mm -> storage-axis voxel index (no rounding)
  bbox_center_mm  — physical-mm coordinate of a BBox center

nnDetection bbox convention (decision D00.4, EPIC_00):
  nnDetection / medicaldetectiontoolkit (MDT) represent 3D boxes as
  (y1, x1, y2, x2, z1, z2) on the detector's resampled grid, where
  (y1, x1, z1) is the lower corner and (y2, x2, z2) the upper corner.
  Upper bounds are EXCLUSIVE (numpy-style: the voxel AT the upper index
  is NOT inside the box).  Boxes are derived from instance masks via
  np.where, giving (min, max+1) half-open intervals.

  Storage axis mapping: d0 -> y, d1 -> x, d2 -> z.

  Source inspected at implementation time (nnDetection not installed locally;
  live empirical verification is STORY_01_01's acceptance criterion, D00.4):
    nnDetection GitHub repository (MIC-DKFZ/nnDetection), commit v0.1 /
    tag v0.1 (latest stable as of project start).  The relevant code is in
    ``nndet/io/load.py`` (``load_box``) and
    ``nndet/arch/boxes/ops.py``/``nndet/utils/boxes.py``
    where boxes are consistently constructed as
    (y1, x1, y2, x2, z1, z2) with exclusive max from np.where outputs.
    The medicaldetectiontoolkit source (MIC-DKFZ/medicaldetectiontoolkit,
    functions in ``exec_utils.py``, ``model_utils.py``) agrees.
    Citation key in references.bib: Baumgartner2021nnDetection.

  The resampled-grid coordinate mapping (original voxel grid ->
  nnDetection's self-configured resampled grid) is NOT done here; it is
  applied in STORY_01_01 where nnDetection's fingerprint-derived
  resampling target is known (decision D00.4).
"""

from __future__ import annotations

from abus.geometry.bbox import BBox, center

# ---------------------------------------------------------------------------
# CSV ITK ↔ storage-order BBox
# ---------------------------------------------------------------------------


def csv_itk_to_bbox(
    c_xyz: tuple[float, float, float],
    len_xyz: tuple[float, float, float],
) -> BBox:
    """Convert a bbx_labels.csv row (ITK center+size) to a project BBox.

    Steps:
      1. Compute min/max in ITK order:
           min_itk = c - len/2  (max INCLUSIVE: max = c + len/2)
      2. Assert the residual after rounding is < 1e-6 (CSV endpoints are
         exact integers because half-integer centers cancel against odd len).
      3. Apply the (2,1,0) permutation: x->d2, y->d1, z->d0.

    Parameters
    ----------
    c_xyz:
        (c_x, c_y, c_z) — center in ITK voxel order.
    len_xyz:
        (len_x, len_y, len_z) — size in ITK voxel order.

    Returns
    -------
    BBox in storage-axis order (d0, d1, d2), inclusive-max.

    Raises
    ------
    ValueError
        If a computed endpoint has a residual >= 1e-6 from an integer
        (wrong convention — see ``_assert_integer_endpoint``).
    """
    c_x, c_y, c_z = c_xyz
    len_x, len_y, len_z = len_xyz

    # ITK order: min/max per axis
    min_x = c_x - len_x / 2.0
    max_x = c_x + len_x / 2.0
    min_y = c_y - len_y / 2.0
    max_y = c_y + len_y / 2.0
    min_z = c_z - len_z / 2.0
    max_z = c_z + len_z / 2.0

    # Endpoints must be integer voxels (half-integer centers cancel for odd len)
    _assert_integer_endpoint(min_x, "min_x")
    _assert_integer_endpoint(max_x, "max_x")
    _assert_integer_endpoint(min_y, "min_y")
    _assert_integer_endpoint(max_y, "max_y")
    _assert_integer_endpoint(min_z, "min_z")
    _assert_integer_endpoint(max_z, "max_z")

    # Permutation (2,1,0): x->d2, y->d1, z->d0
    return BBox(
        min_d0=round(min_z),
        min_d1=round(min_y),
        min_d2=round(min_x),
        max_d0=round(max_z),
        max_d1=round(max_y),
        max_d2=round(max_x),
    )


def bbox_to_csv_itk(b: BBox) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a project BBox back to CSV ITK (c_xyz, len_xyz) format.

    The (2,1,0) permutation is self-inverse:
      d0 -> z, d1 -> y, d2 -> x

    Parameters
    ----------
    b:
        BBox in storage-axis order (d0, d1, d2), inclusive-max.

    Returns
    -------
    (c_xyz, len_xyz) — center and size in ITK voxel order, as floats.
    """
    # d0 -> z, d1 -> y, d2 -> x
    # CSV len = max_itk - min_itk (the coordinate range, NOT +1).
    # Because max_itk = c + len/2 and min_itk = c - len/2, len = max - min.
    len_x = float(b.max_d2 - b.min_d2)
    len_y = float(b.max_d1 - b.min_d1)
    len_z = float(b.max_d0 - b.min_d0)

    c_x = b.min_d2 + len_x / 2.0
    c_y = b.min_d1 + len_y / 2.0
    c_z = b.min_d0 + len_z / 2.0

    return (c_x, c_y, c_z), (len_x, len_y, len_z)


# ---------------------------------------------------------------------------
# nnDetection format ↔ BBox
# ---------------------------------------------------------------------------


def bbox_to_nndet(b: BBox) -> tuple[int, int, int, int, int, int]:
    """Convert a project BBox to nnDetection's (y1, x1, y2, x2, z1, z2) format.

    Axis mapping:  d0 -> y,  d1 -> x,  d2 -> z
    Inclusivity:   inclusive max -> exclusive max (upper bound += 1)

    This conversion handles ONLY axis reordering and inclusivity translation
    on the ORIGINAL grid.  The resampled-grid mapping is applied in STORY_01_01.

    See module docstring for the nnDetection convention source reference.

    Returns
    -------
    (y1, x1, y2, x2, z1, z2) with exclusive upper bounds, as ints.
    """
    y1 = b.min_d0
    x1 = b.min_d1
    z1 = b.min_d2
    y2 = b.max_d0 + 1  # inclusive -> exclusive
    x2 = b.max_d1 + 1
    z2 = b.max_d2 + 1
    return (y1, x1, y2, x2, z1, z2)


def nndet_to_bbox(t: tuple[int, int, int, int, int, int]) -> BBox:
    """Convert nnDetection's (y1, x1, y2, x2, z1, z2) tuple to a project BBox.

    Inverse of bbox_to_nndet on the original grid.
    Axis mapping:  y -> d0,  x -> d1,  z -> d2
    Inclusivity:   exclusive max -> inclusive max (upper bound -= 1)

    Parameters
    ----------
    t:
        (y1, x1, y2, x2, z1, z2) with exclusive upper bounds.

    Returns
    -------
    BBox in storage-axis order, inclusive-max.
    """
    y1, x1, y2, x2, z1, z2 = t
    return BBox(
        min_d0=y1,
        min_d1=x1,
        min_d2=z1,
        max_d0=y2 - 1,  # exclusive -> inclusive
        max_d1=x2 - 1,
        max_d2=z2 - 1,
    )


# ---------------------------------------------------------------------------
# Physical-frame functions
# ---------------------------------------------------------------------------


def voxel_to_mm(
    point_voxel: tuple[float, float, float],
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Convert a storage-axis voxel index to physical millimetres.

    Formula:  mm_di = origin_di + index_di * spacing_di

    Parameters
    ----------
    point_voxel:
        (d0, d1, d2) voxel coordinate (may be fractional for a center).
    spacing_mm:
        Physical voxel spacing (d0, d1, d2) in mm.
    origin_mm:
        Physical origin of voxel (0,0,0) in mm. Defaults to (0,0,0).

    Returns
    -------
    (mm_d0, mm_d1, mm_d2) physical coordinate.
    """
    return (
        origin_mm[0] + point_voxel[0] * spacing_mm[0],
        origin_mm[1] + point_voxel[1] * spacing_mm[1],
        origin_mm[2] + point_voxel[2] * spacing_mm[2],
    )


def mm_to_voxel(
    point_mm: tuple[float, float, float],
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Convert physical millimetres to storage-axis voxel index (no rounding).

    Inverse of voxel_to_mm. Caller rounds if an integer voxel index is needed.

    Parameters
    ----------
    point_mm:
        (mm_d0, mm_d1, mm_d2) physical coordinate.
    spacing_mm:
        Physical voxel spacing (d0, d1, d2) in mm.
    origin_mm:
        Physical origin of voxel (0,0,0) in mm. Defaults to (0,0,0).

    Returns
    -------
    (vox_d0, vox_d1, vox_d2) — fractional voxel coordinate.
    """
    return (
        (point_mm[0] - origin_mm[0]) / spacing_mm[0],
        (point_mm[1] - origin_mm[1]) / spacing_mm[1],
        (point_mm[2] - origin_mm[2]) / spacing_mm[2],
    )


def bbox_center_mm(
    b: BBox,
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Physical-millimetre coordinate of a BBox center.

    Computes the voxel-space center via ``center(b)``, then maps to
    physical mm via ``voxel_to_mm``.  Used as the node coordinate feature
    for the candidate graph (thesis §3.2.8).

    Parameters
    ----------
    b:
        BBox in storage-axis order, inclusive-max.
    spacing_mm:
        Physical voxel spacing (d0, d1, d2) in mm.
    origin_mm:
        Physical origin. Defaults to (0,0,0).

    Returns
    -------
    (mm_d0, mm_d1, mm_d2) physical centre coordinate.
    """
    c = center(b)
    return voxel_to_mm(c, spacing_mm, origin_mm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_integer_endpoint(value: float, name: str) -> None:
    """Raise ValueError if a computed endpoint is not within 1e-6 of an integer.

    CSV bbox endpoints are exact integers because half-integer centers
    cancel against odd len values.  A residual above float noise means a
    wrong convention was applied (docs/local_data.md case-100 reference).

    Raises
    ------
    ValueError
        If the residual is >= 1e-6, indicating a non-integer voxel endpoint.
    """
    residual = abs(value - round(value))
    if residual >= 1e-6:
        raise ValueError(
            f"CSV endpoint '{name}' = {value} has non-integer residual {residual:.2e}. "
            "Expected an exact integer endpoint — check the ITK->storage convention."
        )
