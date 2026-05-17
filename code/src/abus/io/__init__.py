"""abus.io — Volume and mask I/O subpackage.

Public interface (added in STORY_00_02):

  load_volume(path: str) -> VolumeRecord
  load_mask(path: str) -> MaskRecord
  assert_paired(volume: VolumeRecord, mask: MaskRecord) -> None
  VolumeRecord, MaskRecord    — frozen dataclasses
  SpacingPlaceholderError     — raised on non-placeholder header geometry
  CANONICAL_SPACING_MM        — (0.073, 0.200, 0.475674) mm, storage-axis order
  CANONICAL_ORIGIN_MM         — (0.0, 0.0, 0.0) mm
"""

from abus.io.loader import (
    CANONICAL_ORIGIN_MM,
    CANONICAL_SPACING_MM,
    MaskRecord,
    SpacingPlaceholderError,
    VolumeRecord,
    assert_paired,
    load_mask,
    load_volume,
)

__all__ = [
    "CANONICAL_ORIGIN_MM",
    "CANONICAL_SPACING_MM",
    "MaskRecord",
    "SpacingPlaceholderError",
    "VolumeRecord",
    "assert_paired",
    "load_mask",
    "load_volume",
]
