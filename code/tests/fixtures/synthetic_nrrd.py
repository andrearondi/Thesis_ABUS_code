"""Helpers to write tiny synthetic NRRD files for loader unit tests.

Produces two variants:
  - identity-header: space directions = I(3), space origin = [0,0,0]  (the placeholder)
  - non-identity-header: space directions != I(3)  (triggers SpacingPlaceholderError)

No real data needed; tests using these helpers run on any machine.
"""

from __future__ import annotations

import os

import nrrd
import numpy as np


def write_identity_volume(
    path: str | os.PathLike[str], shape: tuple[int, int, int] = (4, 5, 6)
) -> None:
    """Write a minimal uint8 volume NRRD with identity space directions (placeholder)."""
    array = np.zeros(shape, dtype=np.uint8)
    array[1, 2, 3] = 42  # non-zero voxel so it is not entirely blank

    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(str(path), array, header)


def write_non_identity_spacing_volume(
    path: str | os.PathLike[str],
    shape: tuple[int, int, int] = (4, 5, 6),
    scale: float = 0.5,
) -> None:
    """Write a uint8 volume NRRD whose space directions is NOT the identity (scale along diagonal).

    This should trigger SpacingPlaceholderError on load.
    """
    array = np.ones(shape, dtype=np.uint8) * 10

    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(str(path), array, header)


def write_nonzero_origin_volume(
    path: str | os.PathLike[str],
    shape: tuple[int, int, int] = (4, 5, 6),
) -> None:
    """Write a uint8 volume NRRD with identity spacing but non-zero space origin.

    This should trigger SpacingPlaceholderError on load.
    """
    array = np.ones(shape, dtype=np.uint8) * 10

    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [1.0, 0.0, 0.0],
    }
    nrrd.write(str(path), array, header)


def write_identity_mask(
    path: str | os.PathLike[str],
    shape: tuple[int, int, int] = (4, 5, 6),
    binary: bool = True,
) -> None:
    """Write a minimal uint8 mask NRRD with identity space directions.

    If binary=False, writes a value of 2 (non-binary — triggers ValueError on load).
    """
    array = np.zeros(shape, dtype=np.uint8)
    if binary:
        array[1, 2, 3] = 1
    else:
        array[1, 2, 3] = 2  # invalid mask value

    header: dict = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "3D-right-handed",
        "sizes": list(shape),
        "space directions": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "kinds": ["space", "space", "space"],
        "encoding": "raw",
        "space origin": [0.0, 0.0, 0.0],
    }
    nrrd.write(str(path), array, header)
