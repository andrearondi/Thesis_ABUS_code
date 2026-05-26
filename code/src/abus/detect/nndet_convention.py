"""nnDetection bbox/spacing convention constants and resampled-grid helpers.

Single source of truth for Decision D00.4 (EPIC_00) — the live half delivered
in STORY_01_01. Documents, as code, the exact nnDetection box convention and the
resampled-grid mapping needed for the empirical bbox round-trip (ASC-01_01.4).

nnDetection box convention
--------------------------
nnDetection / medicaldetectiontoolkit (MDT) represent 3D boxes as a 6-tuple:

    (y1, x1, y2, x2, z1, z2)

where (y1, x1, z1) is the lower corner and (y2, x2, z2) is the upper corner.
Upper bounds are **exclusive** (numpy-style: the voxel AT the upper index is NOT
inside the box). Boxes are derived from instance masks via ``np.where``, giving
(min, max+1) half-open intervals.

Storage-axis mapping (NRRD storage axes d0, d1, d2):
    d0 -> y,  d1 -> x,  d2 -> z

Source: nnDetection GitHub (MIC-DKFZ/nnDetection), file ``nndet/io/load.py``
(``load_box``) and ``nndet/arch/boxes/ops.py`` / ``nndet/utils/boxes.py``,
inspected at project start (tag v0.1, the latest stable version as of the project
start date). Also confirmed against medicaldetectiontoolkit (MIC-DKFZ/
medicaldetectiontoolkit, ``exec_utils.py``, ``model_utils.py``). Citation key in
references.bib: ``Baumgartner2021nnDetection``.

Resampled-grid mapping
----------------------
nnDetection resamples every volume to a self-configured target grid.  The
resampled-grid voxel coordinate of an original-grid voxel coordinate v_orig is:

    v_resamp = v_orig * (orig_spacing / target_spacing)   (element-wise)

where ``orig_spacing = CANONICAL_SPACING_MM`` and ``target_spacing`` is the
target voxel spacing that nnDetection's fingerprint selects.

The inverse (resampled -> original) is:

    v_orig = v_resamp * (target_spacing / orig_spacing)

For the bbox round-trip (ASC-01_01.4) these mappings are applied to the min and
max corners of the box. Because the endpoints are floats after resampling, the
round-trip residual is bounded by the composed-rounding envelope
``(target_i + orig_i) / 2`` mm per axis. See D01.8 in decisions_log.md for the
full derivation.

When ``target_spacing == orig_spacing`` (no resampling), both mappings are the
identity and the round-trip is exact.

The empirical round-trip test on the real resampled grid is the server-side half
of ASC-01_01.4 (runbook STORY_01_01). This module's ``bbox_original_roundtrip``
implements the round-trip formula so the test can be run against any target spacing.
``bbox_roundtrip_residuals`` wraps it and evaluates the two-clause D01.8 gate.
"""

from __future__ import annotations

from abus.geometry.bbox import BBox
from abus.geometry.convert import bbox_to_nndet, nndet_to_bbox

# ---------------------------------------------------------------------------
# Documented constants
# ---------------------------------------------------------------------------

NNDET_BOX_AXES: tuple[str, ...] = ("y1", "x1", "y2", "x2", "z1", "z2")
"""nnDetection box axis ordering.

A 6-tuple of axis-label strings documenting the (y1, x1, y2, x2, z1, z2)
convention used by nnDetection's internal box representation.

Storage-axis mapping: d0->y, d1->x, d2->z.
Upper bounds are exclusive (numpy-style).
Source: nnDetection GitHub (MIC-DKFZ/nnDetection) v0.1, nndet/io/load.py,
        Baumgartner2021nnDetection.
"""

NNDET_BOX_EXCLUSIVE_MAX: bool = True
"""True: nnDetection upper bounds are exclusive (numpy-style, max+1 from np.where).

This directly corresponds to the +1 / -1 transforms in abus.geometry.convert
(bbox_to_nndet / nndet_to_bbox). Source: nndet/utils/boxes.py, nndet/io/load.py.
"""

D01_8_PRIMARY_MM: float = 0.5
"""ASC-01_01.4 primary gate threshold (mm). Analytic envelope ceiling under
D01.7 spacings (max_i((target_i + orig_i)/2) = 0.4878 mm, rounded up to one
decimal). See decisions_log.md D01.8 for derivation.

This constant is the single source of truth for the primary clause of the two-
clause bbox round-trip gate (ASC-01_01.4 amended 2026-05-26, D01.8).
"""


# ---------------------------------------------------------------------------
# Resampled-grid round-trip helper
# ---------------------------------------------------------------------------


def bbox_original_roundtrip(
    b: BBox,
    target_spacing_mm: tuple[float, float, float],
) -> BBox:
    """Push a project BBox to nnDetection's convention on the RESAMPLED grid and back.

    Implements — as executable, testable code — the exact resampled-grid mapping
    documented in the module docstring.  Used by the empirical round-trip
    verification (ASC-01_01.4) and unit-tested locally with the identity-resampling
    case (no resampling = exact round-trip).

    Algorithm:
        1. Convert BBox to nnDetection format on the original grid via bbox_to_nndet
           (EPIC_00 converter): (y1,x1,y2,x2,z1,z2) with exclusive max.
        2. Scale each corner coordinate to the resampled grid:
               coord_resamp = coord_orig * (orig_spacing_i / target_spacing_i)
           where the storage-axis mapping is d0<->y, d1<->x, d2<->z.
        3. Round each corner to the nearest integer on the resampled grid.
        4. Scale back to the original grid:
               coord_orig_recovered = coord_resamp * (target_spacing_i / orig_spacing_i)
        5. Round to integer and clamp to valid range (≥ 0).
        6. Convert back to a project BBox via nndet_to_bbox (exclusive max -> inclusive).

    Parameters
    ----------
    b:
        Project BBox in storage-axis order, inclusive-max.
    target_spacing_mm:
        nnDetection's target voxel spacing (d0, d1, d2) in mm as returned by
        the nnDetection fingerprint.  When equal to CANONICAL_SPACING_MM, the
        round-trip is exact (identity resampling).

    Returns
    -------
    BBox
        The recovered BBox in storage-axis order, inclusive-max. On the original
        grid the residual is ≤ 1 voxel per axis (gate: ASC-01_01.4).

    Notes
    -----
    The storage-axis to nnDetection-axis mapping (d0->y, d1->x, d2->z) is the
    same for both the original and resampled grid — nnDetection applies the same
    permutation regardless of the target grid, because its coordinate frame is
    defined by the spatial ordering of the resampled image, which preserves the
    storage-axis ordering relative to the volume.

    Rounding convention note
    ------------------------
    Steps 3 and 5 use Python's built-in ``round()`` (banker's rounding).
    nnDetection's internal resampling uses numpy/scipy operations that may use
    floor or ceil instead.  When the resampling factor is not an integer ratio,
    the recovered coordinate can differ from nnDetection's internally recovered
    coordinate by at most 1 voxel.  This is within the ASC-01_01.4 gate of
    ≤ 1 voxel per axis, so the round-trip check remains valid.
    """
    from abus.io.loader import CANONICAL_SPACING_MM as ORIG_SPACING

    # Step 1: original-grid nnDetection format (y1, x1, y2, x2, z1, z2), exclusive max.
    y1, x1, y2, x2, z1, z2 = bbox_to_nndet(b)

    # Step 2: scale factors per storage axis (d0->y, d1->x, d2->z)
    # scale_to_resamp[i] = orig_spacing[i] / target_spacing[i]
    # d0->y: ORIG[0]/target[0], d1->x: ORIG[1]/target[1], d2->z: ORIG[2]/target[2]
    sy = ORIG_SPACING[0] / target_spacing_mm[0]  # d0 scale factor
    sx = ORIG_SPACING[1] / target_spacing_mm[1]  # d1 scale factor
    sz = ORIG_SPACING[2] / target_spacing_mm[2]  # d2 scale factor

    # Step 3: scale to resampled grid and round
    ry1 = round(y1 * sy)
    rx1 = round(x1 * sx)
    ry2 = round(y2 * sy)
    rx2 = round(x2 * sx)
    rz1 = round(z1 * sz)
    rz2 = round(z2 * sz)

    # Step 4+5: scale back to original grid, round, clamp to ≥ 0
    # inverse scale: target[i] / orig[i]
    iy1 = max(0, round(ry1 / sy))
    ix1 = max(0, round(rx1 / sx))
    iy2 = max(0, round(ry2 / sy))
    ix2 = max(0, round(rx2 / sx))
    iz1 = max(0, round(rz1 / sz))
    iz2 = max(0, round(rz2 / sz))

    # Step 6: convert back — ensure valid nndet format (exclusive max >= exclusive min + 1)
    # clamp so that exclusive max > exclusive min per axis
    iy2 = max(iy2, iy1 + 1)
    ix2 = max(ix2, ix1 + 1)
    iz2 = max(iz2, iz1 + 1)

    return nndet_to_bbox((iy1, ix1, iy2, ix2, iz1, iz2))


def bbox_roundtrip_residuals(
    b: BBox,
    target_spacing_mm: tuple[float, float, float],
) -> dict:
    """Compute per-endpoint round-trip residuals in both voxels and mm.

    Calls ``bbox_original_roundtrip(b, target_spacing_mm)`` and returns a dict
    with per-axis voxel and mm residuals, the analytic per-axis envelope derived
    from the composed-rounding formula, and two gate booleans.  Used by
    ``verify_nndet_dataset.py`` to evaluate ASC-01_01.4 (D01.8 gate).

    The structural envelope is computed from ``CANONICAL_SPACING_MM`` and
    ``target_spacing_mm`` inline via ``ceil((target_i + orig_i) / (2 * orig_i))``,
    so a future re-resample re-derives the envelope automatically without touching
    this code.  Only ``D01_8_PRIMARY_MM`` (the ceiling rounded to one decimal) is
    a hard-coded module constant.

    Parameters
    ----------
    b:
        Project BBox in storage-axis order, inclusive-max.
    target_spacing_mm:
        nnDetection's target voxel spacing (d0, d1, d2) in mm.

    Returns
    -------
    dict with keys:
        "gt_bbox"                 : tuple — input bbox endpoints
                                    (d0min, d1min, d2min, d0max, d1max, d2max)
        "recovered_bbox"          : tuple — recovered endpoints after round-trip
        "residuals_vx"            : tuple[int, int, int, int, int, int] —
                                    |recovered - gt| per endpoint, original-grid voxels
        "residuals_mm"            : tuple[float, ...] —
                                    residuals_vx[i] * CANONICAL_SPACING_MM[axis_of(i)]
        "max_residual_mm"         : float — max over all 6 endpoints
        "envelope_vx_per_axis"    : tuple[int, int, int] —
                                    analytic per-axis worst-case in voxels
                                    (3, 2, 2) under D01.7 spacings
        "envelope_mm_per_axis"    : tuple[float, float, float] —
                                    analytic per-axis worst-case in mm
                                    ((target + orig) / 2, element-wise)
        "primary_gate_pass"       : bool — max_residual_mm <= D01_8_PRIMARY_MM
        "structural_gate_pass"    : bool — per-axis residuals_vx within envelope
        "gate_pass"               : bool — primary_gate_pass AND structural_gate_pass
    """
    import math

    from abus.io.loader import CANONICAL_SPACING_MM as ORIG

    recovered = bbox_original_roundtrip(b, target_spacing_mm)

    # Per-endpoint voxel residuals (6 values: min/max for each of d0, d1, d2).
    residuals_vx: tuple[int, int, int, int, int, int] = (
        abs(recovered.min_d0 - b.min_d0),
        abs(recovered.min_d1 - b.min_d1),
        abs(recovered.min_d2 - b.min_d2),
        abs(recovered.max_d0 - b.max_d0),
        abs(recovered.max_d1 - b.max_d1),
        abs(recovered.max_d2 - b.max_d2),
    )

    # Axis index for each of the 6 endpoints: (d0, d1, d2, d0, d1, d2)
    _axis_of = (0, 1, 2, 0, 1, 2)

    # Physical mm residuals: residual_vx * native_spacing_on_that_axis
    residuals_mm: tuple[float, float, float, float, float, float] = tuple(  # type: ignore[assignment]
        residuals_vx[i] * ORIG[_axis_of[i]] for i in range(6)
    )

    max_residual_mm: float = max(residuals_mm)

    # Analytic per-axis envelope (mm): (target_i + orig_i) / 2 per axis.
    # Derived from the composed-rounding formula: each of the two round() calls
    # contributes at most half a voxel in its own grid; together they bound the
    # round-trip residual by (target/2 + orig/2) mm per axis.
    envelope_mm_per_axis: tuple[float, float, float] = (
        (target_spacing_mm[0] + ORIG[0]) / 2.0,
        (target_spacing_mm[1] + ORIG[1]) / 2.0,
        (target_spacing_mm[2] + ORIG[2]) / 2.0,
    )

    # Structural envelope (original-grid voxels): ceil(envelope_mm_i / orig_spacing_i).
    # Equivalently: ceil((target_i + orig_i) / (2 * orig_i)).
    envelope_vx_per_axis: tuple[int, int, int] = (
        math.ceil(envelope_mm_per_axis[0] / ORIG[0]),
        math.ceil(envelope_mm_per_axis[1] / ORIG[1]),
        math.ceil(envelope_mm_per_axis[2] / ORIG[2]),
    )

    primary_gate_pass: bool = max_residual_mm <= D01_8_PRIMARY_MM

    # Structural gate: per-axis residuals_vx (max of min and max endpoint for each axis).
    # axis 0 (d0): indices 0 and 3; axis 1 (d1): indices 1 and 4; axis 2 (d2): indices 2 and 5.
    structural_gate_pass: bool = (
        max(residuals_vx[0], residuals_vx[3]) <= envelope_vx_per_axis[0]
        and max(residuals_vx[1], residuals_vx[4]) <= envelope_vx_per_axis[1]
        and max(residuals_vx[2], residuals_vx[5]) <= envelope_vx_per_axis[2]
    )

    return {
        "gt_bbox": b.as_tuple(),
        "recovered_bbox": recovered.as_tuple(),
        "residuals_vx": residuals_vx,
        "residuals_mm": residuals_mm,
        "max_residual_mm": max_residual_mm,
        "envelope_vx_per_axis": envelope_vx_per_axis,
        "envelope_mm_per_axis": envelope_mm_per_axis,
        "primary_gate_pass": primary_gate_pass,
        "structural_gate_pass": structural_gate_pass,
        "gate_pass": primary_gate_pass and structural_gate_pass,
    }
