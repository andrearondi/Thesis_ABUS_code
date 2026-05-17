"""NRRD volume and mask loader for TDSC-ABUS-2023.

Every NRRD file in the TDSC-ABUS-2023 dataset carries a placeholder identity
``space directions`` matrix and a placeholder ``[0,0,0]`` origin (see
docs/local_data.md "Quirks #1"). The true physical voxel spacing comes from
the official challenge description (thesis §3.2.4) and must be injected
programmatically.

This module is the **single entry point** for reading volumes and masks into
the project. No downstream code reads NRRD directly; it always calls
``load_volume`` / ``load_mask`` so that the physical frame is set once,
consistently, and with a loud guard against files that deviate from the
expected placeholder header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nrrd
import numpy as np

# ---------------------------------------------------------------------------
# Project-wide physical-frame constants
# ---------------------------------------------------------------------------

# Canonical physical voxel spacing, NRRD storage-axis order (d0, d1, d2), mm.
# Source: docs/local_data.md; official TDSC-ABUS-2023 challenge description;
# thesis §3.2.4. The header placeholder is exactly 1.0 mm/voxel — this is the
# true spacing derived from the inverse correlation between voxel count and
# physical size.
CANONICAL_SPACING_MM: tuple[float, float, float] = (0.073, 0.200, 0.475674)

# Canonical physical origin, storage-axis order, mm. The header origin is a
# placeholder [0,0,0]; the project adopts [0,0,0] as the volume-local frame
# origin. Candidate graphs never span volumes (thesis §3.2.8), so a
# volume-local frame is sufficient and the simplest correct choice
# (D00.1 in EPIC_00 decisions log, resolved 2026-05-16).
CANONICAL_ORIGIN_MM: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Tolerance for comparing header values to the expected placeholder.
_HEADER_ATOL: float = 1e-6

# Regex for parsing case IDs from canonical filenames.
_VOLUME_RE = re.compile(r"^DATA_(\d+)\.nrrd$")
_MASK_RE = re.compile(r"^MASK_(\d+)\.nrrd$")


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class SpacingPlaceholderError(RuntimeError):
    """Raised when an NRRD header's ``space directions`` or ``space origin`` is NOT
    the identity placeholder expected for TDSC-ABUS-2023 files.

    If this fires, the file may carry a real (non-placeholder) spacing, and blind
    injection of CANONICAL_SPACING_MM would silently corrupt all downstream
    physical-millimetre computations. Fail loud rather than corrupt geometry.
    """


# ---------------------------------------------------------------------------
# Public data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeRecord:
    """A loaded ABUS ultrasound volume with injected physical geometry.

    Attributes
    ----------
    array:
        Voxel array, shape ``(d0, d1, d2)``, dtype ``uint8``, NRRD storage-axis
        order. **Not resampled, not normalised** — returned as stored.
    spacing_mm:
        Injected canonical physical voxel spacing ``(d0, d1, d2)`` in mm.
        Always equal to ``CANONICAL_SPACING_MM``.
    origin_mm:
        Volume-local origin in mm. Always ``CANONICAL_ORIGIN_MM == (0,0,0)``.
    case_id:
        Integer parsed from the filename ``DATA_<NNN>.nrrd``.
    raw_header:
        Original pynrrd header dict — kept for provenance/audit. The header's
        ``space directions`` is the identity placeholder; the injected spacing is
        in ``spacing_mm``, NOT in this dict.
    source_path:
        Absolute path the record was loaded from.
    """

    array: np.ndarray
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    case_id: int
    raw_header: dict
    source_path: str

    # NOTE: `array` is a numpy ndarray held by reference. `frozen=True` prevents
    # field reassignment but does NOT prevent in-place mutation of the array
    # (e.g. `rec.array[0,0,0] = 255` will succeed). Callers must not mutate
    # the array; doing so will corrupt any other holder of the same VolumeRecord.


@dataclass(frozen=True)
class MaskRecord:
    """A loaded ABUS binary tumour mask with injected physical geometry.

    Attributes
    ----------
    array:
        Mask array, shape ``(d0, d1, d2)``, dtype ``uint8``, values in ``{0, 1}``.
    spacing_mm:
        Injected canonical physical voxel spacing, same as in ``VolumeRecord``.
    origin_mm:
        Volume-local origin, same as in ``VolumeRecord``.
    case_id:
        Integer parsed from ``MASK_<NNN>.nrrd``.
    raw_header:
        Original pynrrd header dict.
    source_path:
        Absolute path the record was loaded from.
    """

    array: np.ndarray
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    case_id: int
    raw_header: dict
    source_path: str

    # NOTE: see VolumeRecord — the array is not deep-copied; do not mutate it.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_volume_case_id(path: str) -> int:
    """Parse case ID from a DATA_<NNN>.nrrd filename.

    Raises
    ------
    ValueError
        If the filename does not match ``DATA_<NNN>.nrrd``.
    """
    name = Path(path).name
    m = _VOLUME_RE.match(name)
    if m is None:
        raise ValueError(
            f"Cannot parse case_id from filename {name!r}: " "expected pattern DATA_<NNN>.nrrd"
        )
    return int(m.group(1))


def _parse_mask_case_id(path: str) -> int:
    """Parse case ID from a MASK_<NNN>.nrrd filename.

    Raises
    ------
    ValueError
        If the filename does not match ``MASK_<NNN>.nrrd``.
    """
    name = Path(path).name
    m = _MASK_RE.match(name)
    if m is None:
        raise ValueError(
            f"Cannot parse case_id from filename {name!r}: " "expected pattern MASK_<NNN>.nrrd"
        )
    return int(m.group(1))


def _check_placeholder_header(path: str, header: dict, array: np.ndarray) -> None:
    """Verify that the NRRD header carries the expected identity placeholder geometry.

    The check covers four fields:
    - ``array.ndim``: must be 3 (rejects 4D or 2D arrays).
    - ``space directions``: must be the 3×3 identity matrix within ``atol = 1e-6``.
    - ``space origin``: must be ``[0, 0, 0]`` within ``atol = 1e-6``.
    - ``sizes``: must be present and must match ``array.shape``.

    Raises
    ------
    SpacingPlaceholderError
        On any deviation, naming the file and the offending header value.
    """
    name = Path(path).name

    # --- ndim ---
    if array.ndim != 3:
        raise SpacingPlaceholderError(
            f"{name}: expected a 3D array, got ndim={array.ndim}. "
            "TDSC-ABUS-2023 files are 3D volumes."
        )

    # --- space directions ---
    raw_dirs = header.get("space directions")
    if raw_dirs is None:
        raise SpacingPlaceholderError(f"{name}: missing 'space directions' in NRRD header")
    dirs = np.asarray(raw_dirs, dtype=float)
    if dirs.shape != (3, 3):
        raise SpacingPlaceholderError(
            f"{name}: 'space directions' has unexpected shape {dirs.shape}; " f"expected (3, 3)"
        )
    if not np.allclose(dirs, np.eye(3), atol=_HEADER_ATOL):
        raise SpacingPlaceholderError(
            f"{name}: 'space directions' is not the identity placeholder — "
            f"got {dirs.tolist()}. This file may carry a real spacing; "
            "injecting CANONICAL_SPACING_MM would silently corrupt geometry."
        )

    # --- space origin ---
    raw_origin = header.get("space origin")
    if raw_origin is None:
        raise SpacingPlaceholderError(f"{name}: missing 'space origin' in NRRD header")
    origin = np.asarray(raw_origin, dtype=float)
    if not np.allclose(origin, np.zeros(3), atol=_HEADER_ATOL):
        raise SpacingPlaceholderError(
            f"{name}: 'space origin' is not the zero placeholder — "
            f"got {origin.tolist()}. Injecting CANONICAL_ORIGIN_MM would "
            "silently override a non-trivial origin."
        )

    # --- sizes vs array shape ---
    raw_sizes = header.get("sizes")
    if raw_sizes is None:
        raise SpacingPlaceholderError(f"{name}: missing 'sizes' in NRRD header")
    sizes = tuple(raw_sizes)
    if sizes != array.shape:
        raise SpacingPlaceholderError(
            f"{name}: header 'sizes' {sizes} does not match array.shape {array.shape}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_volume(path: str) -> VolumeRecord:
    """Read a ``DATA_<NNN>.nrrd`` ABUS volume.

    Steps
    -----
    1. Parse ``case_id`` from the filename (raises ``ValueError`` on mismatch).
    2. Read the NRRD file via ``pynrrd`` (array in NRRD storage-axis order, dtype uint8).
    3. Verify the header geometry is the identity placeholder (raises
       ``SpacingPlaceholderError`` if not).
    4. Inject ``CANONICAL_SPACING_MM`` and ``CANONICAL_ORIGIN_MM``.
    5. Return a ``VolumeRecord``.

    Parameters
    ----------
    path:
        Absolute or relative path to the NRRD file. Filename must match
        ``DATA_<NNN>.nrrd``.

    Returns
    -------
    VolumeRecord

    Raises
    ------
    ValueError
        If the filename does not match ``DATA_<NNN>.nrrd``.
    SpacingPlaceholderError
        If the header ``space directions`` is not the identity or ``space origin``
        is non-zero.
    """
    case_id = _parse_volume_case_id(path)
    array, header = nrrd.read(path)
    _check_placeholder_header(path, header, array)
    if array.dtype != np.dtype("uint8"):
        raise ValueError(
            f"{Path(path).name}: expected dtype uint8, got {array.dtype}. "
            "TDSC-ABUS-2023 volumes are unsigned char (uint8)."
        )
    return VolumeRecord(
        array=array,
        spacing_mm=CANONICAL_SPACING_MM,
        origin_mm=CANONICAL_ORIGIN_MM,
        case_id=case_id,
        raw_header=header,
        source_path=str(Path(path).resolve()),
    )


def load_mask(path: str) -> MaskRecord:
    """Read a ``MASK_<NNN>.nrrd`` binary tumour mask.

    Same geometry guard as ``load_volume``. Additionally asserts that all voxel
    values are in ``{0, 1}``; raises ``ValueError`` if a non-binary value is
    found.

    Parameters
    ----------
    path:
        Path to the mask NRRD file. Filename must match ``MASK_<NNN>.nrrd``.

    Returns
    -------
    MaskRecord

    Raises
    ------
    ValueError
        If the filename does not match or the array contains non-binary values.
    SpacingPlaceholderError
        If the header geometry is not the identity placeholder.
    """
    case_id = _parse_mask_case_id(path)
    array, header = nrrd.read(path)
    _check_placeholder_header(path, header, array)
    if array.dtype != np.dtype("uint8"):
        raise ValueError(
            f"{Path(path).name}: expected dtype uint8, got {array.dtype}. "
            "TDSC-ABUS-2023 masks are unsigned char (uint8)."
        )

    unique_vals = set(np.unique(array).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"{Path(path).name}: non-binary mask values found: {unique_vals}. "
            "Expected values strictly in {0, 1}."
        )

    return MaskRecord(
        array=array,
        spacing_mm=CANONICAL_SPACING_MM,
        origin_mm=CANONICAL_ORIGIN_MM,
        case_id=case_id,
        raw_header=header,
        source_path=str(Path(path).resolve()),
    )


def assert_paired(volume: VolumeRecord, mask: MaskRecord) -> None:
    """Assert that a volume and mask share the same ``case_id`` and ``array.shape``.

    Used by the local-data-sanity check (STORY_00_03) and the nnDetection
    dataset builder (EPIC_01) to catch mis-matched file pairs before any
    geometry computation.

    Parameters
    ----------
    volume:
        A ``VolumeRecord`` from ``load_volume``.
    mask:
        A ``MaskRecord`` from ``load_mask``.

    Raises
    ------
    ValueError
        If ``case_id`` or ``array.shape`` do not match, with a descriptive message.
    """
    if volume.case_id != mask.case_id:
        raise ValueError(
            f"case_id mismatch: volume has case_id={volume.case_id}, "
            f"mask has case_id={mask.case_id}."
        )
    if volume.array.shape != mask.array.shape:
        raise ValueError(
            f"shape mismatch for case {volume.case_id}: "
            f"volume.shape={volume.array.shape}, mask.shape={mask.array.shape}."
        )
