"""Tests for src/abus/detect/grid_pool.py (STORY_01_02, D01.17).

D01.17 spec: `grid_pool` is the SINGLE operator both pooling modes (centroid and
roi_align) use. The axis/align_corners convention must be pinned here so a transpose
or half-voxel shift cannot silently corrupt every sample.

Design (from embedding_extraction.yaml + D01.17):
  - 5D input: feat_map (1, C, D0, D1, D2) for grid_sample
  - grid normalized coords in [-1, 1]
  - LAST grid component indexes FIRST-varying spatial axis (W=innermost=d2=x)
    i.e. grid ordered (x_norm, y_norm, z_norm) = (d2, d1, d0)
  - align_corners=False: voxel centre at -1+(2*i+1)/D
  - Coordinates: box/point in FEATURE space
    box axis (x1,y1,x2,y2,z1,z2): x=d2, y=d1, z=d0
    centroid: cx_d0=(z1+z2)/2, cx_d1=(y1+y2)/2, cx_d2=(x1+x2)/2

Backend: grid_pool uses torch when available (server), pure-numpy fallback when
torch is not installed (laptop). All tests run on both backends; no module-level skip.
The golden-impulse + differential tests pin the critical convention.

Tests:
  test_grid_pool_module_importable              - module exists and is importable
  test_grid_pool_function_signature             - grid_pool callable with correct signature
  test_grid_pool_golden_impulse_centroid        - 1.0 at known voxel pools 1.0 at that centroid
  test_grid_pool_golden_impulse_one_voxel_away  - pools <1.0 one voxel away from impulse
  test_grid_pool_align_corners_false_convention - align_corners=False pinning test
  test_grid_pool_differential_vs_point_pool_trilinear - centroid mode vs point_pool_trilinear
  test_grid_pool_roi_align_extent_mean          - roi_align on ramped map = closed-form mean
  test_grid_pool_output_shape                   - returns (C,) for any C
  test_grid_pool_invalid_mode_raises            - unknown mode raises ValueError
  test_grid_pool_centroid_matches_interior_voxel - centroid at exact integer voxel coords
  test_grid_pool_axis_discrimination            - d0/d2 axes are NOT interchangeable (S1 guard)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1: module importable
# ---------------------------------------------------------------------------


def test_grid_pool_module_importable() -> None:
    """grid_pool module must be importable from abus.detect.grid_pool."""
    import abus.detect.grid_pool as gp  # noqa: F401

    assert hasattr(gp, "grid_pool"), "grid_pool function must be defined in abus.detect.grid_pool"


# ---------------------------------------------------------------------------
# Test 2: function signature
# ---------------------------------------------------------------------------


def test_grid_pool_function_signature() -> None:
    """grid_pool must accept (feat_map, box_or_point, mode, *, align_corners, roi_grid)."""
    import inspect

    from abus.detect.grid_pool import grid_pool

    sig = inspect.signature(grid_pool)
    params = list(sig.parameters.keys())

    assert "feat_map" in params, f"feat_map param missing, got {params}"
    assert "box_or_point" in params, f"box_or_point param missing, got {params}"
    assert "mode" in params, f"mode param missing, got {params}"
    assert "align_corners" in params, f"align_corners param missing, got {params}"
    assert "roi_grid" in params, f"roi_grid param missing, got {params}"

    # align_corners and roi_grid must be keyword-only (after *)
    p_align = sig.parameters["align_corners"]
    p_roi = sig.parameters["roi_grid"]
    assert p_align.kind == inspect.Parameter.KEYWORD_ONLY, "align_corners must be keyword-only"
    assert p_roi.kind == inspect.Parameter.KEYWORD_ONLY, "roi_grid must be keyword-only"

    # Defaults
    assert p_align.default is False, f"align_corners default must be False, got {p_align.default}"
    assert p_roi.default == (3, 3, 3), f"roi_grid default must be (3,3,3), got {p_roi.default}"


# ---------------------------------------------------------------------------
# Test 3: golden-impulse — centroid at known voxel pools 1.0
# ---------------------------------------------------------------------------


def test_grid_pool_golden_impulse_centroid() -> None:
    """Golden-impulse test: feature map 1.0 at (d0=3, d1=2, d2=4), zero elsewhere.

    Pooling at box centroid cx_d0=3, cx_d1=2, cx_d2=4 (in feature space) must
    return 1.0 for all channels.

    D01.17: this test PINS the grid_sample axis ordering and align_corners=False
    convention. If the axes are transposed (e.g. grid ordered as (z,y,x) instead
    of (x,y,z)=(d2,d1,d0)) or align_corners=True is used, the test breaks.

    box_or_point for centroid mode: (cx, cy, cz) = (cx_d2, cx_d1, cx_d0)
    i.e. nnDetection notation (x, y, z) = (d2, d1, d0)
    """
    from abus.detect.grid_pool import grid_pool

    C = 4
    D0, D1, D2 = 8, 7, 9

    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    # Place 1.0 at (d0=3, d1=2, d2=4)
    feat_map[:, 3, 2, 4] = 1.0

    # Centroid at (d0=3, d1=2, d2=4) in feature space
    # In nnDetection notation: z=d0=3, y=d1=2, x=d2=4
    # box_or_point for centroid: (cx_d2, cx_d1, cx_d0) = (4, 2, 3)
    # BUT: the function takes (x1,y1,x2,y2,z1,z2) for roi_align
    # and for centroid mode the spec says (cx, cy, cz) = (cx_d2, cx_d1, cx_d0)
    # So point = (x, y, z) = (4, 2, 3) in nnDetection notation
    point = np.array([4.0, 2.0, 3.0], dtype=np.float32)  # (x=d2=4, y=d1=2, z=d0=3)

    result = grid_pool(feat_map, point, mode="centroid", align_corners=False)

    assert result.shape == (C,), f"Expected shape ({C},), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    np.testing.assert_allclose(
        result,
        1.0,
        atol=1e-5,
        err_msg=(
            "Golden-impulse FAIL: centroid at (d0=3,d1=2,d2=4) should pool 1.0 from impulse map. "
            "Likely axis ordering or align_corners convention is wrong (D01.17 inversion path #4)."
        ),
    )


# ---------------------------------------------------------------------------
# Test 4: golden-impulse — one voxel away pools < 1.0
# ---------------------------------------------------------------------------


def test_grid_pool_golden_impulse_one_voxel_away() -> None:
    """Golden-impulse: pooling half a voxel away from the impulse must give <1.0 and >0.0.

    This distinguishes correct trilinear interpolation from nearest-neighbour.
    The impulse is at d2=3; querying at d2=3.5 (half voxel away in d2) must
    give 0.5 under trilinear interpolation.

    Note: querying at an INTEGER position (d2=4) gives exactly 0.0 because
    the trilinear weight from d2=3 at that position is 0.0 — that is correct.
    The "partial interpolation" only occurs at fractional positions.
    """
    from abus.detect.grid_pool import grid_pool

    C = 1
    D0, D1, D2 = 8, 8, 8

    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    feat_map[:, 3, 3, 3] = 1.0

    # Query at (x=d2=3.5, y=d1=3, z=d0=3) — half a voxel away in d2
    # Trilinear: weight from d2=3 is 0.5, from d2=4 is 0.5 → result = 0.5
    point_half_away = np.array([3.5, 3.0, 3.0], dtype=np.float32)

    result = grid_pool(feat_map, point_half_away, mode="centroid", align_corners=False)

    assert result.shape == (C,)
    assert float(result[0]) < 1.0, (
        f"Half voxel away from impulse should give < 1.0 (trilinear), got {result[0]:.6f}. "
        "Nearest-neighbour or wrong axis is likely."
    )
    assert float(result[0]) > 0.0, (
        f"Half voxel away from impulse should give > 0.0 (trilinear), got {result[0]:.6f}. "
        "Expected ~0.5 at half-voxel offset. Check axis ordering convention."
    )
    np.testing.assert_allclose(
        float(result[0]),
        0.5,
        atol=0.01,
        err_msg=(
            "Impulse at d2=3, query at d2=3.5: expected 0.5 (trilinear half-voxel weight). "
            "Axis d2 corresponds to grid_pool x-coord (last input to box_or_point). "
            "If result is 0.0 the x↔z axis order is likely transposed."
        ),
    )


# ---------------------------------------------------------------------------
# Test 5: align_corners=False convention pinning
# ---------------------------------------------------------------------------


def test_grid_pool_align_corners_false_convention() -> None:
    """align_corners=False: voxel centre at -1+(2*i+1)/D.

    With align_corners=False, voxel i=0 centre is at -1+1/D and voxel i=D-1
    is at 1-1/D in the normalized grid. This means the normalized coordinate
    for feature voxel i in a dimension of size D is:
        norm_coord = -1 + (2*i + 1) / D

    Test: single channel, D0=D1=D2=4. Place 1.0 at voxel (d0=0, d1=0, d2=0).
    Query at the feature coordinate of that voxel.
    Using align_corners=False: norm for voxel 0 in dim 4 = -1 + 1/4 = -0.75.
    But we work in feature-pixel space (caller converts by dividing by stride),
    so the feature coord for d2=0 is just 0.0.

    Simplified pinning test: a constant map (all 1.0) should always pool 1.0
    regardless of query position (trivially verifies no crash / dtype issue).
    The true convention is pinned by the golden-impulse test above.
    """
    from abus.detect.grid_pool import grid_pool

    C = 3
    D0, D1, D2 = 4, 4, 4
    feat_map = np.ones((C, D0, D1, D2), dtype=np.float32)

    # Any point in the middle — constant map should give 1.0
    point = np.array([1.5, 1.5, 1.5], dtype=np.float32)  # (x=d2, y=d1, z=d0)
    result = grid_pool(feat_map, point, mode="centroid", align_corners=False)

    assert result.shape == (C,)
    np.testing.assert_allclose(result, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: differential test vs point_pool_trilinear (100 random maps + centroids)
# ---------------------------------------------------------------------------


def test_grid_pool_differential_vs_point_pool_trilinear() -> None:
    """grid_pool(centroid) must agree with point_pool_trilinear to 1e-5 on 100 random inputs.

    D01.17: the two operators are the SAME quantity (trilinear interpolation at the box
    centroid). This differential test locks the convention once it is established so that
    any future change that breaks agreement is immediately caught.

    point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2) takes feature-space coords.
    grid_pool centroid mode: box_or_point = (x, y, z) = (cx_d2, cx_d1, cx_d0).

    Both operate in the same feature-pixel coordinate system (caller divides by stride).
    """
    from abus.detect.grid_pool import grid_pool
    from abus.detect.nndet_inference import point_pool_trilinear

    rng = np.random.default_rng(2026_06_22)
    C = 128  # D_EMB
    D0, D1, D2 = 16, 14, 15

    n_trials = 100
    max_diff = 0.0

    for trial in range(n_trials):
        feat_map = rng.standard_normal((C, D0, D1, D2)).astype(np.float32)

        # Random centroid in interior (avoid boundary for cleaner comparison)
        cx_d0 = rng.uniform(1.0, D0 - 2.0)
        cx_d1 = rng.uniform(1.0, D1 - 2.0)
        cx_d2 = rng.uniform(1.0, D2 - 2.0)

        # point_pool_trilinear reference
        ref = point_pool_trilinear(feat_map, cx_d0, cx_d1, cx_d2)

        # grid_pool centroid: point = (x=d2, y=d1, z=d0)
        point = np.array([cx_d2, cx_d1, cx_d0], dtype=np.float32)
        gp = grid_pool(feat_map, point, mode="centroid", align_corners=False)

        diff = float(np.max(np.abs(ref - gp)))
        max_diff = max(max_diff, diff)

        if diff > 1e-5:
            pytest.fail(
                f"Trial {trial}: grid_pool(centroid) vs point_pool_trilinear disagree. "
                f"Max diff = {diff:.2e} (tolerance 1e-5). "
                f"cx=({cx_d2:.3f},{cx_d1:.3f},{cx_d0:.3f}). "
                "Check grid_sample axis ordering or align_corners convention (D01.17 inversion #4)."
            )

    # Report max diff for diagnostic purposes
    assert max_diff <= 1e-5, (
        f"Differential test: max diff over {n_trials} trials = {max_diff:.2e}, "
        f"expected <= 1e-5."
    )


# ---------------------------------------------------------------------------
# Test 7: roi_align extent mean — closed-form check
# ---------------------------------------------------------------------------


def test_grid_pool_roi_align_extent_mean() -> None:
    """roi_align on a ramped map equals the closed-form mean over the roi_grid.

    Test setup: single-channel map where feat[0, d0, d1, d2] = d2 (linear in x/d2).
    Box: x1=2, y1=1, x2=6, y2=3, z1=1, z2=5 in feature space.
    roi_grid=(3,3,3): sample at 3x3x3 = 27 equally-spaced points inside the box.

    For a linear map f(d2) = d2, the mean of samples at d2 positions
    [2.0 + (k + 0.5) * (6-2)/3 for k in 0,1,2] = [2.0+2/3, 2.0+2, 2.0+10/3] ≈ [2.667, 4.0, 5.333]
    Wait: roi_grid samples are evenly spaced INSIDE the box, so for roi_grid=3 in d2:
      d2 positions: x1 + (k+0.5)*(x2-x1)/roi_grid_x for k in 0..2
      = 2 + 0.5*(6-2)/3, 2 + 1.5*(6-2)/3, 2 + 2.5*(6-2)/3
      = 2 + 2/3, 2 + 2, 2 + 10/3
      = 2.667, 4.0, 5.333

    Mean over d2 positions: (2.667 + 4.0 + 5.333) / 3 = 12.0 / 3 = 4.0
    Since f = d2, mean f = 4.0 (the midpoint of [2, 6]).

    For the other axes (y=d1, z=d0) the map is constant = d2, so only d2 matters.
    Overall mean = 4.0.
    """
    from abus.detect.grid_pool import grid_pool

    C = 1
    D0, D1, D2 = 10, 10, 10

    # map: f[0, d0, d1, d2] = d2
    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    for d2 in range(D2):
        feat_map[0, :, :, d2] = float(d2)

    # Box in feature space: (x1, y1, x2, y2, z1, z2)
    # x=d2: [2, 6], y=d1: [1, 3], z=d0: [1, 5]
    box = np.array([2.0, 1.0, 6.0, 3.0, 1.0, 5.0], dtype=np.float32)

    result = grid_pool(feat_map, box, mode="roi_align", align_corners=False, roi_grid=(3, 3, 3))

    assert result.shape == (C,), f"Expected ({C},), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    # The mean of d2 values over roi_grid=3 samples in [2, 6]:
    # samples at d2 = 2 + (0.5/3)*(6-2), 2 + (1.5/3)*(6-2), 2 + (2.5/3)*(6-2)
    # = 2 + 2/3, 2 + 2, 2 + 10/3 ≈ 2.667, 4.0, 5.333
    # mean = (2.667 + 4.0 + 5.333) / 3 = 4.0
    expected = 4.0

    np.testing.assert_allclose(
        result[0],
        expected,
        atol=0.05,  # allow small trilinear interpolation error
        err_msg=(
            f"roi_align on d2-ramp map: expected mean~{expected:.3f}, got {result[0]:.4f}. "
            "The roi_align samples should span the box extent [x1,x2] = [2,6]."
        ),
    )


# ---------------------------------------------------------------------------
# Test 8: output shape for various C values
# ---------------------------------------------------------------------------


def test_grid_pool_output_shape() -> None:
    """grid_pool returns (C,) for any C, both modes."""
    from abus.detect.grid_pool import grid_pool

    for C in [1, 4, 16, 128]:
        feat_map = np.random.randn(C, 8, 8, 8).astype(np.float32)

        # centroid
        point = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        result_c = grid_pool(feat_map, point, mode="centroid")
        assert result_c.shape == (C,), f"C={C} centroid: expected ({C},), got {result_c.shape}"
        assert result_c.dtype == np.float32, f"C={C} centroid: expected float32"

        # roi_align
        box = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 5.0], dtype=np.float32)
        result_r = grid_pool(feat_map, box, mode="roi_align")
        assert result_r.shape == (C,), f"C={C} roi_align: expected ({C},), got {result_r.shape}"
        assert result_r.dtype == np.float32, f"C={C} roi_align: expected float32"


# ---------------------------------------------------------------------------
# Test 9: invalid mode raises ValueError
# ---------------------------------------------------------------------------


def test_grid_pool_invalid_mode_raises() -> None:
    """grid_pool raises ValueError for unknown mode strings."""
    from abus.detect.grid_pool import grid_pool

    feat_map = np.zeros((4, 8, 8, 8), dtype=np.float32)
    point = np.array([3.0, 3.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="mode"):
        grid_pool(feat_map, point, mode="invalid_mode")


# ---------------------------------------------------------------------------
# Test 10: centroid at exact integer voxel gives that voxel's value
# ---------------------------------------------------------------------------


def test_grid_pool_centroid_matches_interior_voxel() -> None:
    """Centroid at an interior integer voxel (d0=4, d1=3, d2=5) returns that voxel's value.

    For a feature map with a unique value at (d0=4, d1=3, d2=5) and zeros
    elsewhere, a centroid query at exactly that location must return the unique value.
    This verifies integer-coordinate trilinear interpolation is exact.
    """
    from abus.detect.grid_pool import grid_pool

    C = 8
    D0, D1, D2 = 10, 8, 12

    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    # Unique value at (d0=4, d1=3, d2=5)
    unique_val = 7.5
    feat_map[:, 4, 3, 5] = unique_val

    # Centroid at (x=d2=5, y=d1=3, z=d0=4) in nnDetection notation
    point = np.array([5.0, 3.0, 4.0], dtype=np.float32)

    result = grid_pool(feat_map, point, mode="centroid", align_corners=False)

    assert result.shape == (C,)
    np.testing.assert_allclose(
        result,
        unique_val,
        atol=1e-4,
        err_msg=(
            f"Centroid at integer voxel (d0=4,d1=3,d2=5) must return {unique_val}. "
            f"Got: {result[:3]}. Integer-coordinate trilinear must be exact."
        ),
    )


# ---------------------------------------------------------------------------
# Test 11: axis discrimination — d0 and d2 are NOT interchangeable (S1 guard)
# ---------------------------------------------------------------------------


def test_grid_pool_axis_discrimination() -> None:
    """Confirm d0 and d2 are routed to DIFFERENT feature-map axes (S1 review guard).

    This test pins the D01.17 M1 bug: in pool_embeddings_at_boxes (nndet_inference.py),
    the caller passes to grid_pool a centroid point = (cx_d2_feat, cx_d1_feat, cx_d0_feat)
    where box slot 0 -> d0 and box slot 4 -> d2.  If d0 and d2 are swapped when
    constructing the point, a map with an anisotropic gradient gives a WRONG result.

    Setup:
      - Feature map (C=1, D0=9, D1=9, D2=9)
      - feat_map[0, d0, d1, d2] = d0 * 10 + d2  (anisotropic: d0 and d2 differ)
      - Query A: (x=cx_d2, y=cx_d1, z=cx_d0) = (6.0, 4.0, 2.0)
          -> samples voxel d0=2, d1=4, d2=6 -> value = 2*10 + 6 = 26.0
      - Query B (swapped d0/d2): (x=cx_d2, y=cx_d1, z=cx_d0) = (2.0, 4.0, 6.0)
          -> samples voxel d0=6, d1=4, d2=2 -> value = 6*10 + 2 = 62.0

    The two results must differ. If d0 and d2 are swapped internally, both queries
    return the same (wrong) value. This catches the M1 inversion transparently.
    """
    from abus.detect.grid_pool import grid_pool

    C = 1
    D0, D1, D2 = 9, 9, 9

    # Anisotropic map: value at (d0, d1, d2) = d0 * 10 + d2
    feat_map = np.zeros((C, D0, D1, D2), dtype=np.float32)
    for d0 in range(D0):
        for d2 in range(D2):
            feat_map[0, d0, :, d2] = float(d0 * 10 + d2)

    # Query A: point = (x=cx_d2=6, y=cx_d1=4, z=cx_d0=2)
    # Expected: feat at d0=2, d1=4, d2=6 = 2*10 + 6 = 26.0
    point_a = np.array([6.0, 4.0, 2.0], dtype=np.float32)  # (x=d2, y=d1, z=d0)
    result_a = grid_pool(feat_map, point_a, mode="centroid", align_corners=False)
    np.testing.assert_allclose(
        float(result_a[0]),
        26.0,
        atol=1e-3,
        err_msg=(
            "Axis discrimination FAIL (query A): "
            "point (x=6,y=4,z=2) = (cx_d2=6, cx_d1=4, cx_d0=2) "
            "should sample voxel d0=2,d1=4,d2=6 = 26.0. "
            "d0 and d2 are likely swapped in grid_pool routing."
        ),
    )

    # Query B (d0 and d2 swapped): point = (x=cx_d2=2, y=cx_d1=4, z=cx_d0=6)
    # Expected: feat at d0=6, d1=4, d2=2 = 6*10 + 2 = 62.0
    point_b = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # (x=d2, y=d1, z=d0)
    result_b = grid_pool(feat_map, point_b, mode="centroid", align_corners=False)
    np.testing.assert_allclose(
        float(result_b[0]),
        62.0,
        atol=1e-3,
        err_msg=(
            "Axis discrimination FAIL (query B): "
            "point (x=2,y=4,z=6) = (cx_d2=2, cx_d1=4, cx_d0=6) "
            "should sample voxel d0=6,d1=4,d2=2 = 62.0. "
            "d0 and d2 are likely swapped in grid_pool routing."
        ),
    )

    assert result_a[0] != result_b[0], (
        f"Axis discrimination: query A ({result_a[0]}) == query B ({result_b[0]}). "
        "d0 and d2 are being treated as interchangeable — M1 axis inversion detected."
    )
