"""abus.data — Patient-level fold split and ground-truth label readers.

Public API added in STORY_00_03:
  load_gt_bboxes(csv_path) -> dict[int, BBox]  (labels.py)

Public API added in STORY_00_04:
  FoldSplit                            frozen dataclass for the 5-fold split
  ManifestChecksumError                raised on tampered / inconsistent manifests
  make_fold_split(csv, n_folds, seed)  build the deterministic split
  manifest_sha256(split)               SHA256 over canonical split content
  write_manifest(split, path, ...)     serialize to JSON with embedded checksum
  load_split(path)                     read frozen manifest + verify checksum
  verify_manifest(csv, path)           re-derive and compare fold-for-fold
  SPLIT_SEED, N_FOLDS, SPLITTER_VERSION, MANIFEST_PATH  module constants
"""

from abus.data.labels import load_gt_bboxes
from abus.data.split import (
    MANIFEST_PATH,
    N_FOLDS,
    SPLIT_SEED,
    SPLITTER_VERSION,
    FoldSplit,
    ManifestChecksumError,
    load_split,
    make_fold_split,
    manifest_sha256,
    verify_manifest,
    write_manifest,
)

__all__ = [
    # STORY_00_03
    "load_gt_bboxes",
    # STORY_00_04
    "FoldSplit",
    "ManifestChecksumError",
    "make_fold_split",
    "manifest_sha256",
    "write_manifest",
    "load_split",
    "verify_manifest",
    "SPLIT_SEED",
    "N_FOLDS",
    "SPLITTER_VERSION",
    "MANIFEST_PATH",
]
