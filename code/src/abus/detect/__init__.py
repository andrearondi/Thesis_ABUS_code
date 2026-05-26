"""abus.detect — Candidate generation and nnDetection dataset I/O subpackage.

Added in STORY_01_01 (nnDetection integration + Retina U-Net dataset configuration).
Extended in STORY_01_02 (5-fold training + OOF/ensemble candidate generation).

Public interface re-exported here (STORY_01_01):
  export_dataset(tdsc_root, out_root, spec)  -> dict
  verify_nndet_dataset(nndet_dataset_root)   -> dict
  NndetDatasetSpec                           -- frozen dataclass
  NndetDatasetError                          -- raised on any dataset mismatch

Public interface added here (STORY_01_02):
  RawCandidate                               -- one raw detector proposal
  RawCandidateSet                            -- serialisable container for one split
  generate_oof_candidates(...)               -- OOF candidate generation for train split
  generate_ensemble_candidates(...)          -- ensemble candidate generation for val/test
  load_raw_candidates(split, dir)            -- load serialised RawCandidateSet
  provenance_check(cset)                     -- assert leakage-control invariants
  embedding_variance_gap(oof_set, ens_set)   -- OOF-vs-ensemble variance diagnostic
  ProvenanceError                            -- raised when provenance invariant violated
  DetectorRun                                -- metadata from a completed fold training run
  train_fold_detector(...)                   -- thin wrapper around nnDetection training CLI
  ensemble_combine(proposals, iou_threshold) -- union + WBC + score/embedding averaging
"""

from abus.detect.candidates import (
    ProvenanceError,
    RawCandidate,
    RawCandidateSet,
    embedding_variance_gap,
    generate_ensemble_candidates,
    generate_oof_candidates,
    load_raw_candidates,
    provenance_check,
)
from abus.detect.ensemble import ensemble_combine
from abus.detect.nndet_io import (
    NndetDatasetError,
    NndetDatasetSpec,
    export_dataset,
    verify_nndet_dataset,
)
from abus.detect.train import DetectorRun, train_fold_detector

__all__ = [
    # STORY_01_01
    "NndetDatasetError",
    "NndetDatasetSpec",
    "export_dataset",
    "verify_nndet_dataset",
    # STORY_01_02 — candidates
    "ProvenanceError",
    "RawCandidate",
    "RawCandidateSet",
    "embedding_variance_gap",
    "generate_ensemble_candidates",
    "generate_oof_candidates",
    "load_raw_candidates",
    "provenance_check",
    # STORY_01_02 — ensemble
    "ensemble_combine",
    # STORY_01_02 — training
    "DetectorRun",
    "train_fold_detector",
]
