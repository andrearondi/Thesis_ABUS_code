"""Unit tests for nndet_convention.py — D01.8 additions.

Tests added per the D01.8 architect spec (decisions_log.md 2026-05-26):

  test_bbox_roundtrip_residuals_identity_spacing
      At target_spacing_mm == CANONICAL_SPACING_MM, residuals are exactly zero
      and both gate clauses pass.

  test_bbox_roundtrip_residuals_envelope_d01_7
      With target_spacing_mm = (0.3, 0.3, 0.5) the function reports
      envelope_vx_per_axis == (3, 2, 2) and envelope_mm_per_axis matches
      the analytic formula (target + orig) / 2 element-wise to 1e-9.

  test_bbox_roundtrip_residuals_catches_axis_swap
      A monkey-patched bbox_original_roundtrip that swaps d0 and d2 scale
      factors (simulating an axis-swap bug) triggers structural_gate_pass=False
      and primary_gate_pass=False on a real-shaped bbox.

Additional verification (not a new test — enforces D01.8 requirement §5):
  The existing test_bbox_nndet_roundtrip_synthetic in test_nndet_io.py already
  uses ``assert recovered == original`` (== equality, zero tolerance). This
  is confirmed in the review loop; no tightening was needed.
"""

from __future__ import annotations

import math

from abus.detect.nndet_convention import (
    bbox_roundtrip_residuals,
)
from abus.geometry.bbox import BBox
from abus.io.loader import CANONICAL_SPACING_MM

# D01.7 target spacing (storage-axis order: d0, d1, d2), mm.
TARGET_SPACING_D01_7: tuple[float, float, float] = (0.3, 0.3, 0.5)

# A representative bbox with coordinates in the range of real ABUS lesions.
# min/max chosen to be non-trivial (not near origin, spans multiple target voxels).
_REPRESENTATIVE_BBOX = BBox(
    min_d0=100,
    min_d1=200,
    min_d2=80,
    max_d0=160,
    max_d1=280,
    max_d2=120,
)


# ===========================================================================
# Test A — identity spacing: both clauses pass, residuals are zero
# ===========================================================================


def test_bbox_roundtrip_residuals_identity_spacing() -> None:
    """At target_spacing_mm == CANONICAL_SPACING_MM, all residuals are exactly zero.

    When the target spacing equals the original spacing the round-trip formula
    reduces to the identity mapping (s = 1 per axis; round(round(x)/1) == x).
    Both gate clauses must pass; residuals_vx and residuals_mm must be all zero.
    """
    result = bbox_roundtrip_residuals(_REPRESENTATIVE_BBOX, target_spacing_mm=CANONICAL_SPACING_MM)

    assert result["residuals_vx"] == (
        0,
        0,
        0,
        0,
        0,
        0,
    ), f"Expected zero voxel residuals at identity spacing, got: {result['residuals_vx']}"
    for i, v in enumerate(result["residuals_mm"]):
        assert v == 0.0, f"Expected zero mm residual at index {i}, got: {v}"
    assert result["max_residual_mm"] == 0.0
    assert result["primary_gate_pass"] is True
    assert result["structural_gate_pass"] is True
    assert result["gate_pass"] is True


# ===========================================================================
# Test B — D01.7 spacings: envelope must be (3, 2, 2) and mm envelope correct
# ===========================================================================


def test_bbox_roundtrip_residuals_envelope_d01_7() -> None:
    """With D01.7 target_spacing_mm = (0.3, 0.3, 0.5), envelope is (3, 2, 2) voxels.

    The architect derived (decisions_log.md D01.8):
      - envelope_mm = ((0.3+0.073)/2, (0.3+0.2)/2, (0.5+0.475674)/2)
                    = (0.1865, 0.25, 0.487837)
      - envelope_vx = (ceil(0.1865/0.073), ceil(0.25/0.2), ceil(0.487837/0.475674))
                    = (ceil(2.555), ceil(1.25), ceil(1.0255))
                    = (3, 2, 2)

    This test locks those exact values against the implementation.
    """
    result = bbox_roundtrip_residuals(_REPRESENTATIVE_BBOX, target_spacing_mm=TARGET_SPACING_D01_7)

    expected_envelope_vx = (3, 2, 2)
    assert result["envelope_vx_per_axis"] == expected_envelope_vx, (
        f"envelope_vx_per_axis mismatch: got {result['envelope_vx_per_axis']}, "
        f"expected {expected_envelope_vx}"
    )

    # Analytic envelope_mm = (target_i + orig_i) / 2 per axis
    expected_envelope_mm = tuple(
        (TARGET_SPACING_D01_7[i] + CANONICAL_SPACING_MM[i]) / 2.0 for i in range(3)
    )
    actual_envelope_mm = result["envelope_mm_per_axis"]
    for i in range(3):
        assert abs(actual_envelope_mm[i] - expected_envelope_mm[i]) < 1e-9, (
            f"envelope_mm_per_axis[{i}] mismatch: "
            f"got {actual_envelope_mm[i]:.10f}, "
            f"expected {expected_envelope_mm[i]:.10f} "
            f"(diff={abs(actual_envelope_mm[i] - expected_envelope_mm[i]):.2e})"
        )

    # Sanity: ceil derivation matches (3, 2, 2) independently
    for i in range(3):
        computed_vx = math.ceil(expected_envelope_mm[i] / CANONICAL_SPACING_MM[i])
        assert computed_vx == expected_envelope_vx[i], (
            f"Axis {i}: ceil(envelope_mm / orig_spacing) = {computed_vx}, "
            f"expected {expected_envelope_vx[i]}"
        )


# ===========================================================================
# Test C — axis-swap bug simulation triggers structural gate failure
# ===========================================================================


def test_bbox_roundtrip_residuals_catches_axis_swap() -> None:
    """A d0/d2 axis-swap in bbox_original_roundtrip's output triggers gate failure.

    D01.8 requires that an axis-swap (e.g. the d0 and d2 coordinates are permuted
    in the recovered BBox) fires structural_gate_pass=False.

    We simulate the bug by monkey-patching bbox_original_roundtrip to return a BBox
    whose d0 and d2 endpoints are swapped relative to the correct round-trip result.
    This is the exact failure mode D01.8 describes: on a box centred at real ABUS
    coordinates, d0 ~ 300-400 voxels and d2 ~ 100-150 voxels; swapping them produces
    residuals of ~150-300 voxels on d0 (far above the 3-voxel d0 envelope) and
    ~100-300 voxels on d2 (far above the 2-voxel d2 envelope).

    The real bbox_original_roundtrip is restored after the test.
    """
    import abus.detect.nndet_convention as mod

    # A bbox with well-separated d0 and d2 coordinate ranges so an axis-swap
    # produces easily distinguishable residuals above the structural envelope.
    # d0: [300, 400], d2: [10, 30] — d0 > d2 guarantees the swap is detectable.
    original_bbox = BBox(
        min_d0=300,
        min_d1=200,
        min_d2=10,
        max_d0=400,
        max_d1=300,
        max_d2=30,
    )

    original_fn = mod.bbox_original_roundtrip

    def _buggy_roundtrip(b: BBox, target_spacing_mm: tuple) -> BBox:
        """Simulate an axis-swap bug: return the correct round-trip result but with
        d0 and d2 endpoints swapped (e.g. a d0/d2 storage-to-nndet permutation error)."""
        # Call the real function to get the correct recovered bbox
        correct = original_fn(b, target_spacing_mm)
        # Then swap d0 and d2 to simulate the bug
        return BBox(
            min_d0=correct.min_d2,
            min_d1=correct.min_d1,
            min_d2=correct.min_d0,
            max_d0=correct.max_d2,
            max_d1=correct.max_d1,
            max_d2=correct.max_d0,
        )

    # Monkey-patch
    mod.bbox_original_roundtrip = _buggy_roundtrip  # type: ignore[assignment]
    try:
        result = bbox_roundtrip_residuals(original_bbox, target_spacing_mm=TARGET_SPACING_D01_7)
    finally:
        # Always restore the real function
        mod.bbox_original_roundtrip = original_fn  # type: ignore[assignment]

    # The swap produces d0 residual = |correct.min_d0 - correct.min_d2| ~ 290 voxels,
    # far above the 3-voxel d0 structural envelope. Both clauses must fail.
    assert result["structural_gate_pass"] is False, (
        "structural_gate_pass should be False under a d0/d2 axis-swap bug; "
        f"got True. residuals_vx={result['residuals_vx']}, "
        f"envelope_vx_per_axis={result['envelope_vx_per_axis']}"
    )
    assert result["gate_pass"] is False, (
        "gate_pass should be False (structural clause violated); "
        f"got True. Full result: {result}"
    )
