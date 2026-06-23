"""3D pooling operator for D01.17 decoupled embedding extraction (STORY_01_02).

OVERVIEW (D01.17)
-----------------
The decoupled design pools a 128-D embedding POST-HOC from the Retina U-Net's
FPN decoder feature map at native consensus boxes.  This module provides a single
operator `grid_pool` used by `pool_embeddings_at_boxes` (nndet_inference.py).

Two pooling modes:

  centroid  — single trilinear sample at the box centroid.
               Identical to the legacy point_pool_trilinear; the differential
               unit test asserts agreement to 1e-5 on 100 random inputs.
               Maximally faithful to the detector's firing location.

  roi_align — regular grid of N×N×N sample points inside the box extent,
               trilinear-sampled, mean-pooled to 1×1×1 → (C,).
               Extent-aware; pre-registered contingency that wins the audit if
               roi_align.embedding_auc > centroid.embedding_auc + δ_audit.

COORDINATE CONVENTION (D01.17, pinned by golden-impulse test)
--------------------------------------------------------------
  Feature map tensor: (C, D0, D1, D2)
  nnDetection box axis: (x1, y1, x2, y2, z1, z2)
    x = storage axis d2  (elevational)
    y = storage axis d1  (lateral)
    z = storage axis d0  (acoustic depth)

  Centroid → feature-voxel coords:
    cx_d0 = (z1 + z2) / 2   → tensor axis 1
    cx_d1 = (y1 + y2) / 2   → tensor axis 2
    cx_d2 = (x1 + x2) / 2   → tensor axis 3

  For `grid_pool`, the caller passes coordinates in FEATURE-PIXEL space (same
  space as the raw feature-map tensor, before dividing by any stride).

  centroid mode — box_or_point = (x, y, z) = (cx_d2, cx_d1, cx_d0)
  roi_align mode — box_or_point = (x1, y1, x2, y2, z1, z2)

  All in nnDetection's (x, y, z) = (d2, d1, d0) notation.

BACKEND SELECTION
-----------------
  Primary: torch.nn.functional.grid_sample (authoritative — matches the server
           forward pass; 5D input, align_corners=False, trilinear mode).
  Fallback: pure numpy trilinear interpolation.  Used when torch is not
            installed (laptop without GPU).  Produces identical results to 1e-5
            (pinned by the differential test in test_grid_pool.py).

The module is importable without torch or nnDetection.

SIGN-OFF POLICY
---------------
Any change to the coordinate convention or align_corners default MUST break
`test_grid_pool_golden_impulse_centroid` (the convention pin test) and requires
a new decision-log entry (D01.17).
"""

from __future__ import annotations

from typing import cast

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection: try torch (server), fall back to pure numpy (laptop)
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — torch always present on server
    _TORCH_AVAILABLE = False


def _to_norm(coord: float, size: int, align_corners: bool) -> float:
    """Convert a feature-pixel coordinate to the [-1, 1] grid_sample grid.

    align_corners=False (default, D01.17):
        voxel centre i is at norm = -1 + (2*i + 1) / size
        → norm(coord) = (2*coord + 1) / size - 1
    align_corners=True:
        voxel i is at norm = -1 + 2*i / (size - 1)
        → norm(coord) = 2*coord / (size - 1) - 1  (undefined for size=1)
    """
    if align_corners:
        return 2.0 * coord / max(size - 1, 1) - 1.0
    else:
        return (2.0 * coord + 1.0) / size - 1.0


# ---------------------------------------------------------------------------
# Numpy trilinear fallback (used when torch not installed)
# ---------------------------------------------------------------------------


def _trilinear_np(
    feat: np.ndarray,
    cx_d0: float,
    cx_d1: float,
    cx_d2: float,
) -> np.ndarray:
    """Pure-numpy trilinear interpolation at (cx_d0, cx_d1, cx_d2) in feat (C,D0,D1,D2).

    This is the reference implementation: identical to point_pool_trilinear in
    nndet_inference.py.  The differential test asserts agreement to 1e-5.
    """
    _, D0, D1, D2 = feat.shape
    cx_d0 = float(np.clip(cx_d0, 0.0, D0 - 1.0))
    cx_d1 = float(np.clip(cx_d1, 0.0, D1 - 1.0))
    cx_d2 = float(np.clip(cx_d2, 0.0, D2 - 1.0))

    f0, f1, f2 = int(cx_d0), int(cx_d1), int(cx_d2)
    c0 = min(f0 + 1, D0 - 1)
    c1 = min(f1 + 1, D1 - 1)
    c2 = min(f2 + 1, D2 - 1)

    fd0 = cx_d0 - f0
    fd1 = cx_d1 - f1
    fd2 = cx_d2 - f2

    result = (
        (1 - fd0) * (1 - fd1) * (1 - fd2) * feat[:, f0, f1, f2]
        + (1 - fd0) * (1 - fd1) * fd2 * feat[:, f0, f1, c2]
        + (1 - fd0) * fd1 * (1 - fd2) * feat[:, f0, c1, f2]
        + (1 - fd0) * fd1 * fd2 * feat[:, f0, c1, c2]
        + fd0 * (1 - fd1) * (1 - fd2) * feat[:, c0, f1, f2]
        + fd0 * (1 - fd1) * fd2 * feat[:, c0, f1, c2]
        + fd0 * fd1 * (1 - fd2) * feat[:, c0, c1, f2]
        + fd0 * fd1 * fd2 * feat[:, c0, c1, c2]
    )
    return cast(np.ndarray, result.astype(np.float32))


def _centroid_torch(
    feat: np.ndarray,
    cx_d2: float,
    cx_d1: float,
    cx_d0: float,
    align_corners: bool,
) -> np.ndarray:
    """Single trilinear sample using torch.nn.functional.grid_sample.

    feat: (C, D0, D1, D2) numpy float32
    coords in feature-pixel space; cx_d0=z, cx_d1=y, cx_d2=x (nndet notation)

    grid_sample 5D convention:
      input : (N=1, C, D, H, W) = (1, C, D0, D1, D2)
      grid  : (N=1, 1, 1, 1, 3) last dim = (x_norm, y_norm, z_norm) = (d2, d1, d0)
              i.e. LAST grid component indexes FIRST-varying spatial axis (D/d0)
    """
    _, D0, D1, D2 = feat.shape
    x_n = _to_norm(cx_d2, D2, align_corners)  # d2 → W axis → x component (last in grid)
    y_n = _to_norm(cx_d1, D1, align_corners)  # d1 → H axis → y component
    z_n = _to_norm(cx_d0, D0, align_corners)  # d0 → D axis → z component (first in grid)

    inp = torch.from_numpy(feat[np.newaxis])  # (1, C, D0, D1, D2)
    # grid shape: (1, 1, 1, 1, 3) — (x, y, z) ordering per grid_sample convention
    grid = torch.tensor([[[[[x_n, y_n, z_n]]]]])  # (1,1,1,1,3)

    out = F.grid_sample(
        inp,
        grid.to(inp.dtype),
        mode="bilinear",  # "bilinear" is trilinear for 5D input
        padding_mode="border",
        align_corners=align_corners,
    )
    # out: (1, C, 1, 1, 1) → (C,); use reshape not squeeze to handle C=1 correctly.
    C = feat.shape[0]
    return cast(np.ndarray, out.reshape(C).numpy().astype(np.float32))


def _roi_align_torch(
    feat: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    z1: float,
    z2: float,
    roi_grid: tuple,
    align_corners: bool,
) -> np.ndarray:
    """3D RoIAlign: regular grid of roi_grid[0]×[1]×[2] samples inside the box.

    Sample positions for axis a with box extent [a1, a2], grid_n samples:
        pos_k = a1 + (k + 0.5) * (a2 - a1) / grid_n   for k in 0..grid_n-1

    These are in feature-pixel space; each is converted to norm coords and then
    passed through grid_sample. The mean over all sample points is returned.
    """
    _, D0, D1, D2 = feat.shape
    gd2, gd1, gd0 = roi_grid  # grid samples per d2 (x), d1 (y), d0 (z)

    # Build sample points in feature-pixel space
    # z=d0 samples
    z_pts = [z1 + (k + 0.5) * (z2 - z1) / gd0 for k in range(gd0)]
    # y=d1 samples
    y_pts = [y1 + (k + 0.5) * (y2 - y1) / gd1 for k in range(gd1)]
    # x=d2 samples
    x_pts = [x1 + (k + 0.5) * (x2 - x1) / gd2 for k in range(gd2)]

    # Build grid: (1, gd0, gd1, gd2, 3) — (z, y, x) spatial order, last dim (x, y, z)
    grid_vals = []
    for z_coord in z_pts:
        for y_coord in y_pts:
            for x_coord in x_pts:
                x_n = _to_norm(x_coord, D2, align_corners)
                y_n = _to_norm(y_coord, D1, align_corners)
                z_n = _to_norm(z_coord, D0, align_corners)
                grid_vals.append([x_n, y_n, z_n])

    # grid shape: (1, gd0, gd1, gd2, 3)
    grid_arr = np.array(grid_vals, dtype=np.float32).reshape(1, gd0, gd1, gd2, 3)
    grid_t = torch.from_numpy(grid_arr)

    inp = torch.from_numpy(feat[np.newaxis])  # (1, C, D0, D1, D2)

    out = F.grid_sample(
        inp,
        grid_t.to(inp.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )
    # out: (1, C, gd0, gd1, gd2) → mean over spatial dims → (1, C) → (C,)
    # Use reshape not squeeze to handle C=1 correctly.
    C = feat.shape[0]
    return cast(np.ndarray, out.mean(dim=(2, 3, 4)).reshape(C).numpy().astype(np.float32))


def _roi_align_np(
    feat: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    z1: float,
    z2: float,
    roi_grid: tuple,
) -> np.ndarray:
    """Pure-numpy 3D RoIAlign fallback (laptop without torch)."""
    gd2, gd1, gd0 = roi_grid
    z_pts = [z1 + (k + 0.5) * (z2 - z1) / gd0 for k in range(gd0)]
    y_pts = [y1 + (k + 0.5) * (y2 - y1) / gd1 for k in range(gd1)]
    x_pts = [x1 + (k + 0.5) * (x2 - x1) / gd2 for k in range(gd2)]

    samples = []
    for z_coord in z_pts:
        for y_coord in y_pts:
            for x_coord in x_pts:
                # coord mapping: x=d2, y=d1, z=d0
                samples.append(_trilinear_np(feat, cx_d0=z_coord, cx_d1=y_coord, cx_d2=x_coord))

    return cast(np.ndarray, np.mean(samples, axis=0).astype(np.float32))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grid_pool(
    feat_map: np.ndarray,
    box_or_point: np.ndarray,
    mode: str,
    *,
    align_corners: bool = False,
    roi_grid: tuple = (3, 3, 3),
) -> np.ndarray:
    """3D pooling operator for D01.17 post-hoc embedding extraction.

    Parameters
    ----------
    feat_map : np.ndarray
        Shape (C, D0, D1, D2), float32. Feature map at the pinned FPN level
        (decoder_levels[0]) in preprocessed space.
    box_or_point : np.ndarray
        For mode="centroid": shape (3,), values (x, y, z) = (cx_d2, cx_d1, cx_d0)
            in feature-pixel space.
        For mode="roi_align": shape (6,), values (x1, y1, x2, y2, z1, z2)
            in feature-pixel space.  nnDetection box axis.
    mode : str
        "centroid" — single trilinear sample at the box centroid.
        "roi_align" — roi_grid×roi_grid×roi_grid regular grid samples, mean-pooled.
    align_corners : bool
        grid_sample align_corners convention.  Default False (D01.17 pin).
        MUST match the value used when feat_map was generated.
    roi_grid : tuple[int, int, int]
        (gd2, gd1, gd0) — number of samples per axis for roi_align.
        Default (3, 3, 3) = 27 samples (embedding_extraction.yaml).

    Returns
    -------
    np.ndarray
        Shape (C,), float32. Pooled embedding vector.

    Raises
    ------
    ValueError
        If mode is not "centroid" or "roi_align".

    Notes
    -----
    Coordinate convention (D01.17, pinned by test_grid_pool_golden_impulse_centroid):
      - feat_map axes: (C, d0, d1, d2) — nnDetection storage order
      - box axis: (x1, y1, x2, y2, z1, z2) where x=d2, y=d1, z=d0
      - grid_sample grid last dim: (x_norm, y_norm, z_norm) = (d2_norm, d1_norm, d0_norm)
        (LAST grid component → FIRST-varying spatial axis = W = innermost = d2)
      - align_corners=False: voxel i centre at norm = (2*i + 1) / size - 1
    """
    if mode not in ("centroid", "roi_align"):
        raise ValueError(f"grid_pool mode must be 'centroid' or 'roi_align', got {mode!r}")
    if len(roi_grid) != 3:
        raise ValueError(
            f"grid_pool roi_grid must be a 3-tuple (gd2, gd1, gd0), got length {len(roi_grid)}"
        )

    feat_map = np.asarray(feat_map, dtype=np.float32)
    box_or_point = np.asarray(box_or_point, dtype=np.float32)

    if mode == "centroid":
        # box_or_point = (x, y, z) = (cx_d2, cx_d1, cx_d0)
        cx_d2 = float(box_or_point[0])
        cx_d1 = float(box_or_point[1])
        cx_d0 = float(box_or_point[2])

        if _TORCH_AVAILABLE:
            return _centroid_torch(feat_map, cx_d2, cx_d1, cx_d0, align_corners)
        else:
            return _trilinear_np(feat_map, cx_d0=cx_d0, cx_d1=cx_d1, cx_d2=cx_d2)

    else:  # roi_align
        # box_or_point = (x1, y1, x2, y2, z1, z2)
        x1 = float(box_or_point[0])
        y1 = float(box_or_point[1])
        x2 = float(box_or_point[2])
        y2 = float(box_or_point[3])
        z1 = float(box_or_point[4])
        z2 = float(box_or_point[5])

        if _TORCH_AVAILABLE:
            return _roi_align_torch(feat_map, x1, y1, x2, y2, z1, z2, roi_grid, align_corners)
        else:
            return _roi_align_np(feat_map, x1, y1, x2, y2, z1, z2, roi_grid)
