"""abus.detect — Candidate generation and nnDetection dataset I/O subpackage.

Added in STORY_01_01 (nnDetection integration + Retina U-Net dataset configuration).

Public interface re-exported here (STORY_01_01):
  export_dataset(tdsc_root, out_root, spec)  -> dict
  verify_nndet_dataset(nndet_dataset_root)   -> dict
  NndetDatasetSpec                           -- frozen dataclass
  NndetDatasetError                          -- raised on any dataset mismatch
"""

from abus.detect.nndet_io import (
    NndetDatasetError,
    NndetDatasetSpec,
    export_dataset,
    verify_nndet_dataset,
)

__all__ = [
    "NndetDatasetError",
    "NndetDatasetSpec",
    "export_dataset",
    "verify_nndet_dataset",
]
