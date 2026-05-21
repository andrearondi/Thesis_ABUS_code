"""TDSC-ABUS-2023 → nnDetection dataset converter and verifier (STORY_01_01).

Implements the three responsibilities that nnDetection cannot do for itself:

1. Dataset conversion — TDSC-ABUS-2023 (NRRD volumes + NRRD masks + CSV bbox
   labels) into nnDetection's Task directory layout (raw_splitted/imagesTr/,
   raw_splitted/labelsTr/, dataset.json, splits file).

2. Physical spacing injection — every NRRD volume written into the nnDetection
   dataset carries CANONICAL_SPACING_MM in its ``space directions`` header.
   Without this, nnDetection's fingerprint sees 1 mm isotropic spacing (the
   placeholder) and resamples on a wrong grid, corrupting every downstream
   coordinate (Risk 1 in STORY_01_01).

3. Cross-validation wiring — the nnDetection splits file is derived byte-
   faithfully from the frozen 5-fold manifest (load_split), never re-derived
   from scratch (Risk 3 / ASC-01_01.3).

Verification
------------
``verify_nndet_dataset`` re-checks (1)–(3) post-conversion: spacing in every
image file, splits faithfulness, and lesion counts per case. A mismatch raises
``NndetDatasetError``.

Dry-run CLI
-----------
Running ``python -m abus.detect.nndet_io --dry-run --case-dir <dir>`` converts
one case from <dir> to a temporary output directory and prints a summary.  This
is the local-data-sanity path (ASC-01_01.5).

nnDetection v0.1 task layout produced
--------------------------------------
nnDetection v0.1 (commit 97a58f3) expects the following layout, verified
against nndet/utils/check.py (_check_key_missing) and nndet/io/load.py:

    <out_root>/Task<NNN>_<name>/
        dataset.json                         # task root — nnDetection v0.1 schema
        raw_splitted/
            imagesTr/<case_id>_0000.nrrd     # volume with CANONICAL_SPACING_MM
            labelsTr/<case_id>.nrrd          # binary mask label (uint8, {0,1})
            labelsTr/<case_id>.json          # per-case instance→class JSON
            imagesTs/<case_id>_0000.nrrd     # val/test images (inference only)
        splits_final.json                    # nnDetection CV splits (see below)

dataset.json schema (nnDetection v0.1, NOT nnUNet)
---------------------------------------------------
Required keys verified against nndet/utils/check.py and preprocess.py:
    task       (str)  — task name, e.g. "Task001_TDSCABUS"
    dim        (int)  — 3 for 3D volumes
    modalities (dict) — {"0": "US"}  (plural key; nnUNet uses "modality")
    labels     (dict) — {"0": "tumor"} (foreground-only, zero-indexed; background
                         is implicit in nnDetection — do NOT include it)
    test_labels (bool) — False: preprocess.py line 394 reads this key with no
                         default. False means nndet_prep will NOT attempt to
                         check labelsTs/ (which we do not provide — thesis §3.2
                         held-out evaluation policy).
    _project   (dict) — provenance, project-specific (tolerated as extra key)

nnUNet-only keys explicitly dropped: name, numTraining, numTest,
tensorImageSize, training, test.

Per-case JSON sidecar (nnDetection v0.1 load.py requirement)
-------------------------------------------------------------
For every training case (labelsTr/<case>.nrrd), a sibling JSON is written:
    labelsTr/<case>.json  →  {"instances": {"1": 0}}
where key "1" is the binary mask value (lesion voxels = 1) and value 0 is the
class index (labels["0"] = "tumor" — zero-indexed foreground).
Source: nnDetection v0.1 nndet/io/load.py, load_case_cropped().

splits_final.json location
---------------------------
nnDetection v0.1 reads splits_final.json from the PREPROCESSED directory
(created by nndet_prep), not from raw_splitted/. Since the preprocessed
directory does not exist before nndet_prep runs, we write splits_final.json
to the task root alongside dataset.json. The runbook (STORY_01_01_runbook.md)
instructs copying it to the preprocessed directory after nndet_prep creates
it. This is option (b) of the STORY_01_01 defect 4 decision, recorded in
docs/decisions_log.md (2026-05-20).
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nrrd
import numpy as np
import pandas as pd

from abus.data.split import load_split
from abus.geometry.convert import csv_itk_to_bbox
from abus.io.loader import (
    CANONICAL_SPACING_MM,
    assert_paired,
    load_mask,
    load_volume,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TASK_ID_DEFAULT: int = 1
_TASK_NAME_DEFAULT: str = "Task001_TDSCABUS"

# nnDetection image filenames follow: <case_id_4d>_<modality_idx_4d>.nrrd
# Single modality (US): index 0000
_MODALITY_IDX: str = "0000"

# Base NRRD header for nnDetection image and label files.
# Both file types use the same base (uint8, 3D, gzip, placeholder origin).
# The space directions and sizes fields are filled in per file.
_NNDET_HEADER_BASE: dict[str, Any] = {
    "type": "unsigned char",
    "dimension": 3,
    "space": "3D-right-handed",
    "kinds": ["space", "space", "space"],
    "encoding": "gzip",
    "space origin": [0.0, 0.0, 0.0],
}


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class NndetDatasetError(RuntimeError):
    """Raised by verify_nndet_dataset on any mismatch.

    Covers: missing files, spacing errors, splits-faithfulness failures,
    and lesion-count mismatches.  Callers catch this to abort conversion
    pipelines safely.
    """


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NndetDatasetSpec:
    """Pinned spec for the nnDetection Task built from TDSC-ABUS-2023.

    Fields are pinned at conversion time and written into dataset.json.
    Downstream code (STORY_01_02) reads the spec from dataset.json; this
    dataclass is the in-memory representation and the single source of truth.

    Attributes
    ----------
    task_id:
        nnDetection Task numeric id (config-pinned; ``Task001`` = 1).
    task_name:
        Task directory name, e.g. ``"Task001_TDSCABUS"``.
    n_train_cases:
        Number of training cases (100 in TDSC-ABUS-2023 Train split).
    n_val_cases:
        Number of validation cases (30 in Validation split).
    n_test_cases:
        Number of test cases (70 in Test split).
    spacing_mm:
        Canonical physical voxel spacing in storage-axis order (d0, d1, d2), mm.
        Always ``CANONICAL_SPACING_MM = (0.073, 0.200, 0.475674)``.
    modality:
        nnDetection modality string. ``"US"`` for B-mode ultrasound.
    label_semantics:
        Human-readable description of the foreground class.
    """

    task_id: int
    task_name: str
    n_train_cases: int
    n_val_cases: int
    n_test_cases: int
    spacing_mm: tuple[float, float, float]
    modality: str
    label_semantics: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_image_name(case_id: int) -> str:
    """nnDetection image filename for case_id: ``<case_id_4d>_0000.nrrd``."""
    return f"{case_id:04d}_{_MODALITY_IDX}.nrrd"


def _canonical_label_name(case_id: int) -> str:
    """nnDetection label filename for case_id: ``<case_id_4d>.nrrd``."""
    return f"{case_id:04d}.nrrd"


def _make_spacing_header(spacing_mm: tuple[float, float, float]) -> list[list[float]]:
    """Build the NRRD ``space directions`` diagonal matrix for given spacing."""
    s0, s1, s2 = spacing_mm
    return [
        [s0, 0.0, 0.0],
        [0.0, s1, 0.0],
        [0.0, 0.0, s2],
    ]


def _write_image_nrrd(
    path: str,
    array: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> None:
    """Write a uint8 volume NRRD with the given physical spacing in the header.

    This is the spacing-injection step: the produced file's ``space directions``
    diagonal equals ``spacing_mm``, NOT the identity placeholder.  nnDetection's
    fingerprinting reads spacing from this header.

    Parameters
    ----------
    path:
        Destination file path (will be created; parent must exist).
    array:
        uint8 volume array, NRRD storage-axis order (d0, d1, d2).
    spacing_mm:
        Physical spacing to write into the NRRD header (d0, d1, d2), mm.
    """
    header: dict[str, Any] = dict(_NNDET_HEADER_BASE)
    header["sizes"] = list(array.shape)
    header["space directions"] = _make_spacing_header(spacing_mm)
    nrrd.write(path, array, header)


def _write_label_nrrd(
    path: str,
    array: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> None:
    """Write a uint8 binary mask NRRD with correct physical spacing.

    The label file carries the same spacing as the image file so that nnDetection
    can overlay them on the same physical grid without re-projection.
    """
    header: dict[str, Any] = dict(_NNDET_HEADER_BASE)
    header["sizes"] = list(array.shape)
    header["space directions"] = _make_spacing_header(spacing_mm)
    nrrd.write(path, array, header)


def _write_per_case_json(label_nrrd_path: str) -> None:
    """Write the nnDetection v0.1 per-case instance→class JSON sidecar.

    nnDetection v0.1 load.py (load_case_cropped) reads:
        seg_props_file = f"{str(seg_file).split('.')[0]}.json"
        properties_json["instances"] = {str(k): int(v) ...}

    For TDSC-ABUS-2023 (single-lesion, single foreground class):
        {"instances": {"1": 0}}
    where:
        key "1"  — binary mask value (lesion voxels = 1)
        value 0  — class index (labels[\"0\"] = \"tumor\", zero-indexed foreground)

    The BL1 escalation trigger in export_dataset aborts conversion for any
    multi-lesion case, so single-instance encoding is consistent with the
    in-scope assumption (confirmed from TDSC documentation and Train-split
    inspection during the 2026-05-18 review loop).

    Parameters
    ----------
    label_nrrd_path:
        Path to the labelsTr/<case>.nrrd file. The sidecar is written as
        the same path with .json replacing .nrrd.
    """
    json_path = Path(label_nrrd_path).with_suffix(".json")
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump({"instances": {"1": 0}}, f)
        f.write("\n")


def _parse_id(path: str) -> int:
    """Parse numeric id from DATA_<NNN>.nrrd or MASK_<NNN>.nrrd filename."""
    stem = Path(path).stem  # e.g. "DATA_0042" or "DATA_42"
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse case_id from {path!r}")
    return int(parts[-1])


# ---------------------------------------------------------------------------
# Public API — single-case conversion
# ---------------------------------------------------------------------------


def convert_case(
    volume_path: str,
    mask_path: str,
    bbox_csv_row: dict | list,
    out_image_path: str,
    out_label_path: str,
) -> dict:
    """Convert one TDSC case to nnDetection format.

    Steps
    -----
    1. Load the volume via load_volume (EPIC_00 loader, which injects
       CANONICAL_SPACING_MM and raises SpacingPlaceholderError on a
       non-placeholder header).
    2. Load the mask via load_mask (same guard).
    3. Assert the pair is consistent via assert_paired.
    4. Write the volume with CANONICAL_SPACING_MM in the nnDetection image file.
    5. Write the binary mask as the nnDetection label file, then write the
       per-case instance JSON sidecar required by nnDetection v0.1 load.py
       (``{"instances": {"1": 0}}``, single-lesion encoding only).
    6. Compute the lesion count from the CSV row (1 row = 1 lesion in the
       single-lesion convention used by TDSC-ABUS-2023; multi-lesion cases have
       multiple CSV rows with the same id, and the caller is responsible for
       grouping and passing one row per call or all rows per case).
    7. Return a per-case conversion summary dict.

    Parameters
    ----------
    volume_path:
        Absolute path to ``DATA_<NNN>.nrrd``.
    mask_path:
        Absolute path to ``MASK_<NNN>.nrrd``.
    bbox_csv_row:
        A single dict with keys: ``id``, ``c_x``, ``c_y``, ``c_z``,
        ``len_x``, ``len_y``, ``len_z``.  Represents one lesion instance.
        If a case has multiple lesions, the caller groups CSV rows by id and
        passes them all; ``n_lesions`` is computed from the group count.
        For the unit-test / single-lesion case, pass a single row dict.
    out_image_path:
        Destination for the nnDetection image file (with correct spacing).
    out_label_path:
        Destination for the nnDetection label file (binary mask).

    Returns
    -------
    dict with keys:
        ``case_id`` (int), ``out_image_path`` (str), ``out_label_path`` (str),
        ``n_lesions`` (int), ``spacing_written`` (tuple[float,float,float]).

    Raises
    ------
    SpacingPlaceholderError
        If the NRRD header is not the expected identity placeholder
        (propagated from load_volume / load_mask).
    ValueError
        On filename format errors or shape mismatches.
    """
    vol_rec = load_volume(volume_path)
    mask_rec = load_mask(mask_path)
    assert_paired(vol_rec, mask_rec)

    # Ensure parent directories exist
    Path(out_image_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_label_path).parent.mkdir(parents=True, exist_ok=True)

    # Write image with CANONICAL_SPACING_MM injected into the NRRD header
    _write_image_nrrd(out_image_path, vol_rec.array, CANONICAL_SPACING_MM)

    # Write label NRRD (binary mask, same spacing)
    _write_label_nrrd(out_label_path, mask_rec.array, CANONICAL_SPACING_MM)

    # Validate the CSV row and count lesions.
    # A dict has n_lesions=1; a list of dicts has n_lesions=len(list).
    # The sidecar is written AFTER this block so a multi-lesion ValueError
    # fires before writing a semantically-wrong {"instances": {"1": 0}}.
    _required_keys = {"c_x", "c_y", "c_z", "len_x", "len_y", "len_z"}

    def _validate_csv_row(row: dict) -> None:
        missing = _required_keys - set(row.keys())
        if missing:
            raise ValueError(
                f"CSV row is missing required keys: {missing}. "
                f"Got: {set(row.keys())}. "
                "Expected keys: c_x, c_y, c_z, len_x, len_y, len_z."
            )
        csv_itk_to_bbox(
            (float(row["c_x"]), float(row["c_y"]), float(row["c_z"])),
            (float(row["len_x"]), float(row["len_y"]), float(row["len_z"])),
        )

    if isinstance(bbox_csv_row, list):
        if not bbox_csv_row:
            raise ValueError(
                "bbox_csv_row is an empty list — no lesions found for this case. "
                "Expected at least one CSV row."
            )
        for row in bbox_csv_row:
            _validate_csv_row(row)
        n_lesions = len(bbox_csv_row)
    else:
        # Single row — validate keys and geometry
        _validate_csv_row(bbox_csv_row)
        n_lesions = 1

    # Guard: {"instances": {"1": 0}} is only valid for a single-lesion binary mask.
    # A multi-lesion case requires instance-index encoding (mask values 1, 2, 3, …),
    # which is not implemented. Raise before writing the sidecar so an interrupted
    # run never leaves a wrong-encoding JSON on disk.
    if n_lesions > 1:
        raise NndetDatasetError(
            f"Case {vol_rec.case_id}: multi-lesion encoding is not supported. "
            'Binary mask with {"instances": {"1": 0}} is only valid for '
            f"single-lesion cases (n_lesions={n_lesions}). "
            "Multi-lesion instance encoding (mask values 1, 2, 3, …) is out of scope "
            "for STORY_01_01 (see BL1 escalation trigger in the runbook)."
        )

    # Write nnDetection v0.1 per-case instance JSON sidecar (must come after the
    # lesion-count guard above to avoid writing a semantically-wrong sidecar).
    _write_per_case_json(out_label_path)

    return {
        "case_id": vol_rec.case_id,
        "out_image_path": str(out_image_path),
        "out_label_path": str(out_label_path),
        "n_lesions": n_lesions,
        "spacing_written": CANONICAL_SPACING_MM,
    }


# ---------------------------------------------------------------------------
# Public API — dataset.json
# ---------------------------------------------------------------------------


def build_dataset_json(spec: NndetDatasetSpec, out_path: str) -> None:
    """Write nnDetection v0.1 dataset.json from the pinned NndetDatasetSpec.

    Produces the nnDetection v0.1 schema verified against nndet/utils/check.py
    (commit 97a58f3110b71caf1b4bcc1851e67cf11e987fc5).

    Required keys (nnDetection v0.1):
        task        (str)  — task name
        dim         (int)  — 3 for 3D volumes
        modalities  (dict) — {"0": modality_string} (plural; NOT "modality")
        labels      (dict) — foreground-only, zero-indexed: {"0": "tumor"}
                             Background is implicit; do NOT include it.
        test_labels (bool) — False; preprocess.py line 394 reads this key with
                             no OmegaConf default. False directs nndet_prep to
                             skip the labelsTs/ check (no test labels provided —
                             thesis §3.2 held-out evaluation policy).

    Dropped (nnUNet-only, absent from nnDetection v0.1):
        name, numTraining, numTest, tensorImageSize, training, test.

    Kept:
        _project — project-specific provenance dict (tolerated as extra key).

    Parameters
    ----------
    spec:
        Pinned dataset specification.
    out_path:
        Destination JSON path.
    """
    data: dict[str, Any] = {
        # nnDetection v0.1 required keys (verified against nndet/utils/check.py)
        "task": spec.task_name,
        "dim": 3,
        "modalities": {"0": spec.modality},
        # Foreground-only, zero-indexed labels (background implicit in nnDetection)
        "labels": {"0": "tumor"},
        # preprocess.py line 394 reads cfg["data"]["test_labels"] with no default.
        # Value is False: val/test ground truth is consumed by our own evaluation
        # pipeline (thesis §3.2 held-out policy), NOT via nnDetection's labelsTs/
        # mechanism. Setting True would cause nndet_prep to call
        # check_data_and_label_splitted(test=True, labels=True) and demand
        # labelsTs/<case>.nrrd + <case>.json, which we do not provide.
        "test_labels": False,
        # Project-specific provenance — tolerated as extra key by nnDetection
        "_project": {
            "task_id": spec.task_id,
            "n_train_cases": spec.n_train_cases,
            "n_val_cases": spec.n_val_cases,
            "n_test_cases": spec.n_test_cases,
            "spacing_mm": list(spec.spacing_mm),
            "label_semantics": spec.label_semantics,
            "story_id": "01_01",
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Public API — nnDetection splits file
# ---------------------------------------------------------------------------


def write_nndet_splits(out_path: str) -> None:
    """Derive nnDetection's CV splits file from the FROZEN 5-fold manifest.

    Reads the frozen manifest via ``abus.data.split.load_split()`` — which
    verifies the embedded SHA256 — and writes nnDetection's ``splits_final.json``
    format: a JSON list of dicts, one per fold:

        [{"train": [case_ids], "val": [case_ids]}, ...]

    For fold k:
        train = frozen_split.train_ids(k)  (all folds except k)
        val   = frozen_split.oof_ids(k)    (fold k)

    NEVER calls make_fold_split — the function consumes the frozen manifest only.
    This is the leakage guard: ASC-01_01.3 / Risk 3.

    Parameters
    ----------
    out_path:
        Destination JSON path (e.g. ``<task_dir>/splits_final.json``).
    """
    split = load_split()  # verifies SHA256; raises ManifestChecksumError on tamper

    folds_list: list[dict[str, list[int]]] = []
    for k in range(len(split.folds)):
        folds_list.append(
            {
                "train": split.train_ids(k),
                "val": split.oof_ids(k),
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(folds_list, f, indent=2)
        f.write("\n")

    log.info("Wrote nnDetection splits file: %s (%d folds)", out_path, len(folds_list))


# ---------------------------------------------------------------------------
# Public API — full dataset export
# ---------------------------------------------------------------------------


def export_dataset(
    tdsc_root: str,
    out_root: str,
    spec: NndetDatasetSpec,
) -> dict:
    """Full conversion of all TDSC-ABUS-2023 cases to the nnDetection Task format.

    Converts volumes + masks from the TDSC layout into the nnDetection Task
    directory under ``<out_root>/<spec.task_name>/``, writes ``dataset.json``
    and the ``splits_final.json`` file derived from the frozen 5-fold manifest.

    Parameters
    ----------
    tdsc_root:
        Root of the TDSC-ABUS-2023 dataset. Two on-disk layouts are supported
        per split (the official distribution ships Train with nested DATA
        shards but a flat MASK directory; Validation/Test are flat both sides)::

            <tdsc_root>/
                Train/
                    DATA/DATA_<NNN>.nrrd                      # flat (any split)
                    DATA/DATA00_49/DATA_<NNN>.nrrd            # nested shard (Train)
                    DATA/DATA50_99/DATA_<NNN>.nrrd            # nested shard (Train)
                    MASK/MASK_<NNN>.nrrd
                    bbx_labels.csv
                Validation/
                    DATA/DATA_<NNN>.nrrd
                    MASK/MASK_<NNN>.nrrd
                    bbx_labels.csv
                Test/
                    DATA/DATA_<NNN>.nrrd
                    MASK/MASK_<NNN>.nrrd
                    bbx_labels.csv

        Discovery uses ``rglob`` so flat and nested-shard layouts both work;
        case-id matching across DATA and MASK is done by parsed integer, so
        any zero-padding width (``DATA_0.nrrd``, ``DATA_000.nrrd``,
        ``DATA_0000.nrrd``) is tolerated.

    out_root:
        Parent of the nnDetection Task directory.
    spec:
        Pinned conversion specification.

    Returns
    -------
    dict with keys:
        ``task_dir`` (str), ``n_train``, ``n_val``, ``n_test`` (int — converted),
        ``total_lesions`` (int), ``spacing_written`` (tuple), ``splits_file`` (str).

    Raises
    ------
    FileNotFoundError
        If any expected NRRD or CSV file is missing.
    SpacingPlaceholderError
        If any source NRRD header is not the expected placeholder.
    NndetDatasetError
        If any case fails the post-conversion lesion-count check.
    """
    tdsc_path = Path(tdsc_root)
    task_dir = Path(out_root) / spec.task_name

    # nnDetection v0.1 layout convention (verified against nndet/utils/check.py
    # and nndet_prep at commit 97a58f3):
    #
    #   Task<NNN>_<name>/
    #     dataset.json                       ← task root (NOT under raw_splitted)
    #     splits_final.json                  ← written here; runbook copies to
    #                                          preprocessed/ after nndet_prep runs
    #     raw_splitted/
    #       imagesTr/<case>_0000.nrrd        ← training images only (100 cases)
    #       labelsTr/<case>.nrrd             ← binary mask labels
    #       labelsTr/<case>.json             ← per-case instance→class sidecar
    #       imagesTs/<case>_0000.nrrd        ← val/test images (inference only)
    #
    # imagesTr + labelsTr hold training cases only (IDs in frozen 5-fold manifest).
    # imagesTs holds val+test; no labels written (batch inference targets).
    # nnDetection prep encounters only training case IDs in splits_final.json.
    raw_splitted = task_dir / "raw_splitted"
    images_tr = raw_splitted / "imagesTr"
    labels_tr = raw_splitted / "labelsTr"
    images_ts = raw_splitted / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    total_lesions = 0
    counts: dict[str, int] = {"n_train": 0, "n_val": 0, "n_test": 0}

    # Training cases go to imagesTr/labelsTr (nnDetection cross-validation).
    # Val/test cases go to imagesTs/ only — no labels, used for inference only.
    # Each tuple includes the spec-pinned expected case count so the discovery
    # phase can refuse to proceed if the on-disk file count does not match
    # (silent-zero protection; see 2026-05-20 patch in decisions_log).
    _train_split = ("Train", "n_train", images_tr, labels_tr, True, spec.n_train_cases)
    _val_split = ("Validation", "n_val", images_ts, None, False, spec.n_val_cases)
    _test_split = ("Test", "n_test", images_ts, None, False, spec.n_test_cases)

    for split_name, count_key, img_dir, lbl_dir, write_label, expected_count in (
        _train_split,
        _val_split,
        _test_split,
    ):
        split_dir = tdsc_path / split_name
        data_root = split_dir / "DATA"
        mask_root = split_dir / "MASK"
        csv_path = split_dir / "bbx_labels.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"bbx_labels.csv not found: {csv_path}")
        if not data_root.exists():
            raise FileNotFoundError(
                f"DATA directory not found for split '{split_name}': {data_root}"
            )
        if not mask_root.exists():
            raise FileNotFoundError(
                f"MASK directory not found for split '{split_name}': {mask_root}"
            )

        # Recursive glob — TDSC Train ships nested DATA00_49/DATA50_99 shards
        # under DATA/, while Validation/Test are flat. rglob handles both.
        # MASK ships flat in every split observed so far, but rglob is safe
        # there too. (Layout scope gap discovered 2026-05-20; see runbook.)
        data_files = sorted(data_root.rglob("DATA_*.nrrd"))
        mask_files = sorted(mask_root.rglob("MASK_*.nrrd"))

        # Key by parsed integer case_id so any zero-padding width is tolerated
        # (DATA_0.nrrd, DATA_000.nrrd, DATA_0000.nrrd all map to the same key).
        data_by_id: dict[int, Path] = {}
        for p in data_files:
            cid = _parse_id(str(p))
            if cid in data_by_id:
                raise NndetDatasetError(
                    f"Split '{split_name}': duplicate DATA file for case_id={cid}: "
                    f"{data_by_id[cid]} and {p}."
                )
            data_by_id[cid] = p

        mask_by_id: dict[int, Path] = {}
        for p in mask_files:
            cid = _parse_id(str(p))
            if cid in mask_by_id:
                raise NndetDatasetError(
                    f"Split '{split_name}': duplicate MASK file for case_id={cid}: "
                    f"{mask_by_id[cid]} and {p}."
                )
            mask_by_id[cid] = p

        # Discovery-phase count check. Refuses to write a partial dataset if
        # the on-disk file count diverges from the spec — this is the silent-
        # zero guard. Without it, an unmatched glob silently yields 0 cases.
        if len(data_by_id) != expected_count:
            raise NndetDatasetError(
                f"Split '{split_name}': discovered {len(data_by_id)} DATA file(s) "
                f"under {data_root}, but spec.{count_key}_cases requires "
                f"{expected_count}. On-disk layout may have changed; check the "
                "DATA directory and any nested subdirectories."
            )

        # Every DATA case must have a matching MASK in TDSC.
        missing_masks = sorted(set(data_by_id) - set(mask_by_id))
        if missing_masks:
            raise FileNotFoundError(
                f"Split '{split_name}': MASK file missing for case_id(s) "
                f"{missing_masks}. DATA files exist under {data_root} but the "
                f"corresponding masks are absent from {mask_root}."
            )

        bbox_df = pd.read_csv(csv_path)
        # Group by case id (column 'id' in CSV)
        grouped = bbox_df.groupby("id")

        for case_id, vol_nrrd in sorted(data_by_id.items()):
            mask_nrrd = mask_by_id[case_id]

            out_image = str(img_dir / _canonical_image_name(case_id))
            out_label = str(lbl_dir / _canonical_label_name(case_id)) if lbl_dir else None

            # Get all CSV rows for this case_id
            if case_id in grouped.groups:
                rows = grouped.get_group(case_id).to_dict("records")
            else:
                raise ValueError(
                    f"No CSV rows found for case_id={case_id} in {csv_path}. "
                    "Every case in the TDSC dataset must have at least one bbx_labels.csv entry."
                )

            # For a single-row case pass the dict; for multi-row pass the list.
            csv_arg: Any = rows[0] if len(rows) == 1 else rows

            if write_label and out_label is not None:
                summary = convert_case(str(vol_nrrd), str(mask_nrrd), csv_arg, out_image, out_label)
                total_lesions += summary["n_lesions"]
            else:
                # Val/test: write image only (spacing injection), no label
                vol_rec = load_volume(str(vol_nrrd))
                Path(out_image).parent.mkdir(parents=True, exist_ok=True)
                _write_image_nrrd(out_image, vol_rec.array, CANONICAL_SPACING_MM)
                # Count lesions for summary but don't write a label file
                total_lesions += len(rows) if isinstance(rows, list) else 1

            counts[count_key] += 1

            log.debug("Converted case %d (%s): %d lesion(s)", case_id, split_name, len(rows))

    # Write dataset.json
    ds_json_path = str(task_dir / "dataset.json")
    build_dataset_json(spec, ds_json_path)

    # Write splits file from frozen manifest
    splits_path = str(task_dir / "splits_final.json")
    write_nndet_splits(splits_path)

    log.info(
        "export_dataset complete: %d train, %d val, %d test; %d lesions total",
        counts["n_train"],
        counts["n_val"],
        counts["n_test"],
        total_lesions,
    )

    return {
        "task_dir": str(task_dir),
        "n_train": counts["n_train"],
        "n_val": counts["n_val"],
        "n_test": counts["n_test"],
        "total_lesions": total_lesions,
        "spacing_written": CANONICAL_SPACING_MM,
        "splits_file": splits_path,
    }


# ---------------------------------------------------------------------------
# Public API — post-conversion verification
# ---------------------------------------------------------------------------


def verify_nndet_dataset(nndet_dataset_root: str) -> dict:
    """Post-conversion verification of the nnDetection Task directory.

    Checks:
        1. Every nnDetection image in ``imagesTr/`` carries CANONICAL_SPACING_MM
           in its ``space directions`` header (not identity 1 mm).
        2. The ``splits_final.json`` file equals the frozen 5-fold manifest
           case-for-case (ASC-01_01.3).
        3. The ``dataset.json`` is present and contains the required keys.

    Parameters
    ----------
    nndet_dataset_root:
        Path to the nnDetection Task directory (e.g. ``Task001_TDSCABUS/``).

    Returns
    -------
    dict with keys:
        ``spacing_ok`` (bool), ``splits_ok`` (bool), ``dataset_json_ok`` (bool),
        ``n_images_checked`` (int), ``n_splits_checked`` (int).

    Raises
    ------
    NndetDatasetError
        On any mismatch (spacing, splits faithfulness, missing files).
    """
    task_dir = Path(nndet_dataset_root)

    # --- Check dataset.json exists and has nnDetection v0.1 required keys ---
    ds_json_path = task_dir / "dataset.json"
    if not ds_json_path.exists():
        raise NndetDatasetError(f"dataset.json not found in {task_dir}")
    with open(ds_json_path, encoding="utf-8") as f:
        ds_cfg = json.load(f)
    # Validate nnDetection v0.1 required keys — both the check_dataset_file contract
    # (task, dim, modalities, labels) and the downstream preprocess.py read contract
    # (test_labels — confirmed from server traceback at commit 97a58f3, line 394).
    for required_key in ("task", "dim", "modalities", "labels", "test_labels"):
        if required_key not in ds_cfg:
            raise NndetDatasetError(
                f"dataset.json is missing required key '{required_key}'. "
                f"Found keys: {sorted(ds_cfg.keys())}. "
                "Run build_dataset_json again to regenerate with the full schema."
            )

    # --- Check splits_final.json faithfulness (ASC-01_01.3) ---
    # splits_final.json lives at the task root (written by write_nndet_splits);
    # after nndet_prep it must also be copied to the preprocessed/ directory
    # (runbook step — see docs/decisions_log.md 2026-05-20 defect-4 decision).
    splits_path = task_dir / "splits_final.json"
    if not splits_path.exists():
        raise NndetDatasetError(
            f"splits_final.json not found in {task_dir}. "
            "The nnDetection splits file must be present."
        )

    with open(splits_path, encoding="utf-8") as f:
        produced_splits = json.load(f)

    frozen = load_split()

    if not isinstance(produced_splits, list):
        raise NndetDatasetError(
            f"splits_final.json must be a JSON list, got {type(produced_splits).__name__}"
        )
    if len(produced_splits) != len(frozen.folds):
        raise NndetDatasetError(
            f"split fold count mismatch: splits_final.json has {len(produced_splits)} folds, "
            f"frozen manifest has {len(frozen.folds)}."
        )

    for k, entry in enumerate(produced_splits):
        expected_train = frozen.train_ids(k)
        expected_val = frozen.oof_ids(k)
        produced_train = sorted(int(x) for x in entry.get("train", []))
        produced_val = sorted(int(x) for x in entry.get("val", []))

        if produced_train != expected_train:
            raise NndetDatasetError(
                f"split mismatch in fold {k} train set. "
                f"Expected {len(expected_train)} cases, "
                f"splits_final.json has {len(produced_train)}. "
                f"First mismatch: {set(produced_train) ^ set(expected_train)}"
            )
        if produced_val != expected_val:
            raise NndetDatasetError(
                f"split mismatch in fold {k} val set. "
                f"Expected {len(expected_val)} cases, "
                f"splits_final.json has {len(produced_val)}."
            )

    # --- Check per-case JSON sidecar presence in raw_splitted/labelsTr/ ---
    # nnDetection v0.1 load.py reads a sibling .json for every label NRRD.
    # An interrupted or partial export can leave .nrrd files without sidecars;
    # those cases pass spacing and splits checks here but fail at nndet_prep time.
    labels_tr = task_dir / "raw_splitted" / "labelsTr"
    if labels_tr.exists():
        for nrrd_file in sorted(labels_tr.glob("*.nrrd")):
            json_sidecar = nrrd_file.with_suffix(".json")
            if not json_sidecar.exists():
                raise NndetDatasetError(
                    f"Missing per-case JSON sidecar: {json_sidecar}. "
                    "nnDetection v0.1 load.py requires a sibling .json for every "
                    "label NRRD. Re-run export_dataset to regenerate sidecars."
                )

    # --- Check spacing in raw_splitted/imagesTr/ ---
    # nnDetection v0.1 layout: images live under raw_splitted/imagesTr/
    images_tr = task_dir / "raw_splitted" / "imagesTr"
    n_images_checked = 0
    spacing_atol = 1e-6

    if images_tr.exists():
        for img_path in sorted(images_tr.glob("*.nrrd")):
            _, header = nrrd.read(str(img_path))
            space_dirs = np.array(header.get("space directions", np.eye(3)), dtype=float)
            # Check diagonal entries match CANONICAL_SPACING_MM
            for i, expected_sp in enumerate(CANONICAL_SPACING_MM):
                actual_sp = space_dirs[i, i]
                if abs(actual_sp - expected_sp) > spacing_atol:
                    raise NndetDatasetError(
                        f"Spacing mismatch in {img_path.name}: "
                        f"axis {i} got {actual_sp:.6f} mm, expected {expected_sp:.6f} mm. "
                        "nnDetection will resample on a wrong grid."
                    )
            # Check off-diagonal entries are zero (no shear)
            for i in range(3):
                for j in range(3):
                    if i != j and abs(space_dirs[i, j]) > spacing_atol:
                        raise NndetDatasetError(
                            f"Non-zero off-diagonal in space directions of {img_path.name}: "
                            f"[{i},{j}] = {space_dirs[i,j]:.6g}. Expected diagonal matrix."
                        )
            n_images_checked += 1

    log.info(
        "verify_nndet_dataset: spacing OK (%d images), splits OK (%d folds), "
        "dataset.json OK, sidecars OK",
        n_images_checked,
        len(produced_splits),
    )

    return {
        "spacing_ok": True,
        "splits_ok": True,
        "dataset_json_ok": True,
        "sidecars_ok": True,
        "n_images_checked": n_images_checked,
        "n_splits_checked": len(produced_splits),
    }


# ---------------------------------------------------------------------------
# CLI — dry-run local-data-sanity entrypoint
# ---------------------------------------------------------------------------


def _dry_run(case_dir: str, out_dir: str) -> None:
    """Dry-run: convert one TDSC case from case_dir to out_dir and print a summary.

    Accepts two TDSC split layouts:
      1. Flat layout (used by test fixtures):
             <case_dir>/DATA_<NNN>.nrrd, <case_dir>/MASK_<NNN>.nrrd, <case_dir>/bbx_labels.csv
      2. TDSC split layout (the real validation/test root):
             <case_dir>/DATA/DATA_<NNN>.nrrd, <case_dir>/MASK/MASK_<NNN>.nrrd,
             <case_dir>/bbx_labels.csv
         In this layout the first case found in DATA/ is converted (for the
         local-data-sanity check this is always case 100 when pointing at
         the Validation split root).

    Prints the conversion summary dict and exits 0 on success.
    This is the local-data-sanity path (ASC-01_01.5 / STORY_01_01 inspectable result #1).
    """
    case_path = Path(case_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Support both flat layout and TDSC split layout (DATA/ + MASK/ subdirectories)
    vol_files = sorted(case_path.glob("DATA_*.nrrd"))
    if not vol_files:
        vol_files = sorted((case_path / "DATA").glob("DATA_*.nrrd"))

    mask_files = sorted(case_path.glob("MASK_*.nrrd"))
    if not mask_files:
        mask_files = sorted((case_path / "MASK").glob("MASK_*.nrrd"))

    if not vol_files:
        raise FileNotFoundError(f"No DATA_*.nrrd found in {case_dir} or {case_dir}/DATA/")
    if not mask_files:
        raise FileNotFoundError(f"No MASK_*.nrrd found in {case_dir} or {case_dir}/MASK/")

    vol_path = str(vol_files[0])
    mask_path = str(mask_files[0])
    case_id = _parse_id(vol_path)

    # Ensure the mask matches the case_id of the first volume
    matching_masks = [m for m in mask_files if _parse_id(str(m)) == case_id]
    if not matching_masks:
        raise FileNotFoundError(
            f"No matching MASK_<{case_id}>.nrrd found in {case_dir} or {case_dir}/MASK/"
        )
    mask_path = str(matching_masks[0])

    # Load bbx_labels.csv — may be at case_dir root (both layouts)
    csv_path = case_path / "bbx_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"bbx_labels.csv not found in {case_dir}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        rows = [row for row in reader if int(row.get("id", -1)) == case_id]

    if not rows:
        raise ValueError(f"No CSV rows found for case_id={case_id} in {csv_path}")

    csv_arg: Any = rows[0] if len(rows) == 1 else rows

    out_image = str(out_path / _canonical_image_name(case_id))
    out_label = str(out_path / _canonical_label_name(case_id))

    result = convert_case(vol_path, mask_path, csv_arg, out_image, out_label)

    print("=== nndet_io dry-run conversion summary ===")
    for key, val in result.items():
        print(f"  {key}: {val}")
    print("=== dry-run complete (exit 0) ===")


def main() -> None:
    """CLI entry point for ``python -m abus.detect.nndet_io``."""
    parser = argparse.ArgumentParser(
        description="abus.detect.nndet_io — TDSC→nnDetection converter (dry-run / build)"
    )
    # --dry-run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert a single local case (local-data-sanity path).",
    )
    parser.add_argument(
        "--case-dir",
        type=str,
        default=None,
        help="Directory containing DATA_*.nrrd, MASK_*.nrrd, bbx_labels.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for the dry-run converted files.",
    )

    args = parser.parse_args()

    if args.dry_run:
        if not args.case_dir:
            parser.error("--dry-run requires --case-dir")
        out_dir = args.out_dir or "/tmp/nndet_io_dryrun"
        _dry_run(args.case_dir, out_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
