"""Tests for abus.io.loader (STORY_00_02).

Real-data tests (marked with the local_data fixture) are skipped when the
local dataset root is absent so that CI on machines without data still runs
the synthetic-NRRD tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic_nrrd import (
    write_identity_mask,
    write_identity_volume,
    write_non_identity_spacing_volume,
    write_nonzero_origin_volume,
)

# ---------------------------------------------------------------------------
# Local dataset paths (from docs/local_data.md)
# ---------------------------------------------------------------------------
_DATA_ROOT = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
_VOLUME_PATH = _DATA_ROOT / "DATA" / "DATA_100.nrrd"
_MASK_PATH = _DATA_ROOT / "MASK" / "MASK_100.nrrd"

_LOCAL_DATA_AVAILABLE = _VOLUME_PATH.exists() and _MASK_PATH.exists()
_skip_if_no_data = pytest.mark.skipif(
    not _LOCAL_DATA_AVAILABLE,
    reason=(
        "Local dataset not available "
        "(expected at /Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation)"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_loader():  # type: ignore[no-untyped-def]
    """Import the loader module (deferred so tests can be collected even if import fails)."""
    from abus.io import loader  # noqa: PLC0415

    return loader


# ---------------------------------------------------------------------------
# Synthetic-NRRD tests (no real data needed)
# ---------------------------------------------------------------------------


class TestFailLoudOnNonIdentitySpacing:
    """load_volume must raise SpacingPlaceholderError when space directions != I(3)."""

    def test_non_identity_spacing_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "DATA_007.nrrd"
        write_non_identity_spacing_volume(p)
        with pytest.raises(loader.SpacingPlaceholderError):
            loader.load_volume(str(p))

    def test_error_message_names_file(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "DATA_007.nrrd"
        write_non_identity_spacing_volume(p)
        with pytest.raises(loader.SpacingPlaceholderError, match="DATA_007.nrrd"):
            loader.load_volume(str(p))


class TestFailLoudOnNonzeroOrigin:
    """load_volume must raise SpacingPlaceholderError when space origin != [0,0,0]."""

    def test_nonzero_origin_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "DATA_008.nrrd"
        write_nonzero_origin_volume(p)
        with pytest.raises(loader.SpacingPlaceholderError):
            loader.load_volume(str(p))


class TestMaskRejectsNonBinary:
    """load_mask must raise ValueError when the mask contains values outside {0, 1}."""

    def test_non_binary_mask_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "MASK_009.nrrd"
        write_identity_mask(p, binary=False)
        with pytest.raises(ValueError, match="non-binary"):
            loader.load_mask(str(p))


class TestIdentityVolumeLoads:
    """load_volume must succeed on a valid identity-header synthetic NRRD."""

    def test_load_synthetic_volume(self, tmp_path: Path) -> None:
        loader = _import_loader()
        shape = (4, 5, 6)
        p = tmp_path / "DATA_001.nrrd"
        write_identity_volume(p, shape=shape)
        rec = loader.load_volume(str(p))
        assert rec.array.shape == shape
        assert rec.array.dtype == np.dtype("uint8")
        assert rec.spacing_mm == loader.CANONICAL_SPACING_MM
        assert rec.origin_mm == loader.CANONICAL_ORIGIN_MM
        assert rec.case_id == 1
        # Use resolved paths: on macOS tmp_path may be a symlink (e.g. /var vs /private/var).
        assert Path(rec.source_path) == Path(str(p)).resolve()
        assert isinstance(rec.raw_header, dict)

    def test_load_synthetic_mask(self, tmp_path: Path) -> None:
        loader = _import_loader()
        shape = (4, 5, 6)
        p = tmp_path / "MASK_002.nrrd"
        write_identity_mask(p, shape=shape)
        rec = loader.load_mask(str(p))
        assert rec.array.shape == shape
        assert rec.array.dtype == np.dtype("uint8")
        assert set(rec.array.flat).issubset({0, 1})
        assert rec.spacing_mm == loader.CANONICAL_SPACING_MM
        assert rec.case_id == 2


class TestCaseIdParsing:
    """Case ID must be parsed from the filename; non-matching filenames raise ValueError."""

    def test_case_id_volume(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "DATA_042.nrrd"
        write_identity_volume(p)
        rec = loader.load_volume(str(p))
        assert rec.case_id == 42

    def test_case_id_mask(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "MASK_042.nrrd"
        write_identity_mask(p)
        rec = loader.load_mask(str(p))
        assert rec.case_id == 42

    def test_bad_volume_filename_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "volume.nrrd"
        write_identity_volume(p)
        with pytest.raises(ValueError, match="filename"):
            loader.load_volume(str(p))

    def test_bad_mask_filename_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        p = tmp_path / "mask.nrrd"
        write_identity_mask(p)
        with pytest.raises(ValueError, match="filename"):
            loader.load_mask(str(p))


class TestAssertPaired:
    """assert_paired must pass when volume and mask match; raise ValueError otherwise."""

    def test_paired_synthetic(self, tmp_path: Path) -> None:
        loader = _import_loader()
        vp = tmp_path / "DATA_005.nrrd"
        mp = tmp_path / "MASK_005.nrrd"
        write_identity_volume(vp, shape=(4, 5, 6))
        write_identity_mask(mp, shape=(4, 5, 6))
        vol = loader.load_volume(str(vp))
        mask = loader.load_mask(str(mp))
        loader.assert_paired(vol, mask)  # must not raise

    def test_unpaired_case_id_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        vp = tmp_path / "DATA_010.nrrd"
        mp = tmp_path / "MASK_011.nrrd"
        write_identity_volume(vp, shape=(4, 5, 6))
        write_identity_mask(mp, shape=(4, 5, 6))
        vol = loader.load_volume(str(vp))
        mask = loader.load_mask(str(mp))
        with pytest.raises(ValueError, match="case_id"):
            loader.assert_paired(vol, mask)

    def test_unpaired_shape_raises(self, tmp_path: Path) -> None:
        loader = _import_loader()
        vp = tmp_path / "DATA_012.nrrd"
        mp = tmp_path / "MASK_012.nrrd"
        write_identity_volume(vp, shape=(4, 5, 6))
        write_identity_mask(mp, shape=(4, 5, 7))  # different shape
        vol = loader.load_volume(str(vp))
        mask = loader.load_mask(str(mp))
        with pytest.raises(ValueError, match="shape"):
            loader.assert_paired(vol, mask)


class TestHeaderGuardEdgeCases:
    """Additional edge-case tests for _check_placeholder_header guard."""

    def test_missing_sizes_key_raises(self, tmp_path: Path) -> None:
        """A synthetic NRRD with no 'sizes' key must raise SpacingPlaceholderError.

        We cannot easily write a NRRD without 'sizes' via pynrrd, so we test the guard
        by calling _check_placeholder_header directly with a stripped header dict.
        """
        loader = _import_loader()
        p = tmp_path / "DATA_020.nrrd"
        array = np.zeros((4, 5, 6), dtype=np.uint8)
        header_no_sizes: dict = {
            "type": "unsigned char",
            "dimension": 3,
            "space": "3D-right-handed",
            "space directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "kinds": ["space", "space", "space"],
            "space origin": [0.0, 0.0, 0.0],
            # 'sizes' intentionally absent
        }
        with pytest.raises(loader.SpacingPlaceholderError, match="missing 'sizes'"):
            loader._check_placeholder_header(str(p), header_no_sizes, array)


# ---------------------------------------------------------------------------
# Real-data tests (skipped if dataset absent)
# ---------------------------------------------------------------------------


@_skip_if_no_data
class TestLoadRealVolumeCase100:
    """Load the real DATA_100.nrrd and verify geometry and content."""

    def test_shape(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert rec.array.shape == (865, 682, 353), f"Expected (865,682,353) got {rec.array.shape}"

    def test_dtype(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert rec.array.dtype == np.dtype("uint8")

    def test_spacing_injected(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert (
            rec.spacing_mm == loader.CANONICAL_SPACING_MM
        ), f"Expected {loader.CANONICAL_SPACING_MM} got {rec.spacing_mm}"

    def test_origin(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert rec.origin_mm == loader.CANONICAL_ORIGIN_MM

    def test_case_id(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert rec.case_id == 100

    def test_raw_header_present(self) -> None:
        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))
        assert "space directions" in rec.raw_header


@_skip_if_no_data
class TestLoadRealMaskCase100:
    """Load the real MASK_100.nrrd and verify binary content."""

    def test_binary_values(self) -> None:
        loader = _import_loader()
        rec = loader.load_mask(str(_MASK_PATH))
        assert set(np.unique(rec.array)).issubset(
            {0, 1}
        ), f"Expected values in {{0,1}}, got {np.unique(rec.array)}"

    def test_foreground_voxels(self) -> None:
        """Foreground count must equal the value from verification_report.json."""
        loader = _import_loader()
        rec = loader.load_mask(str(_MASK_PATH))
        fg = int(rec.array.sum())
        assert fg == 2049720, f"Expected 2049720 foreground voxels, got {fg}"

    def test_shape_parity_with_volume(self) -> None:
        loader = _import_loader()
        vol = loader.load_volume(str(_VOLUME_PATH))
        mask = loader.load_mask(str(_MASK_PATH))
        assert vol.array.shape == mask.array.shape


@_skip_if_no_data
class TestAssertPairedRealCase100:
    """assert_paired must pass for the real case 100 volume + mask pair."""

    def test_paired_case100(self) -> None:
        loader = _import_loader()
        vol = loader.load_volume(str(_VOLUME_PATH))
        mask = loader.load_mask(str(_MASK_PATH))
        loader.assert_paired(vol, mask)  # must not raise


@_skip_if_no_data
class TestSimpleITKCrossCheck:
    """SimpleITK and pynrrd must agree on shape and dtype for case 100.

    Defends against a silent axis-transpose in one library (risk #2 in the story).
    """

    def test_shape_dtype_agree(self) -> None:
        """SimpleITK and pynrrd must agree on the set of axis extents and dtype.

        pynrrd returns arrays in NRRD storage order (d0, d1, d2) = (865, 682, 353).
        SimpleITK's GetArrayFromImage reverses ITK axis order, yielding (d2, d1, d0)
        = (353, 682, 865). The sorted extents must be identical — if they differ, one
        library is reading a different number of voxels than the other, indicating a
        silent transpose or truncation.

        Dtype must also agree.
        """
        import SimpleITK as sitk  # noqa: PLC0415

        loader = _import_loader()
        rec = loader.load_volume(str(_VOLUME_PATH))

        sitk_img = sitk.ReadImage(str(_VOLUME_PATH))
        # GetArrayFromImage gives (d2, d1, d0) for NRRD — the reverse of pynrrd's
        # storage order. Sorting ensures we compare the same extents regardless of order.
        arr_sitk = sitk.GetArrayFromImage(sitk_img)

        assert sorted(arr_sitk.shape) == sorted(rec.array.shape), (
            f"Axis-extent set mismatch: SimpleITK sorted{sorted(arr_sitk.shape)} "
            f"vs pynrrd sorted{sorted(rec.array.shape)}. "
            "One library may be reading a different set of voxels."
        )
        assert (
            arr_sitk.dtype == rec.array.dtype
        ), f"Dtype mismatch: SimpleITK {arr_sitk.dtype} vs pynrrd {rec.array.dtype}"
