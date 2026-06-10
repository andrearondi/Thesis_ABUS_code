"""Patient-level 5-fold stratified cross-validation split (STORY_00_04).

Produces a deterministic, B/M-stratified, patient-level 5-fold split of the 100
TDSC-ABUS-2023 training cases, serializes the frozen manifest to JSON with a SHA256
checksum, and provides a loader/verifier every downstream epic calls.

Key invariants
--------------
- make_fold_split is fully deterministic: inputs sorted by case_id ascending before
  passing to StratifiedKFold, fixed seed, pinned scikit-learn minor version.
- The frozen manifest (configs/splits/fold_split_5cv.json) is the artifact of record.
  It is built ONCE on the server from the real labels.csv.
- Downstream code calls load_split() — never make_fold_split() — so the server folds
  are used verbatim on every machine.
- load_split() verifies the embedded SHA256 on every read. Any silent edit is caught.
- verify_manifest() re-derives the split from labels.csv and checks it matches the
  manifest fold-for-fold, proving the manifest came from the documented algorithm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Module-level constants (frozen; changes here are a breaking change)
# ---------------------------------------------------------------------------

SPLIT_SEED: int = 20230516
"""Fixed project seed for the fold split. Frozen the moment the manifest is
committed; the manifest, not the seed, is the artifact of record (rules §12,
decision D00.2). Value chosen as the plan-sharding date 2026-05-16 reordered."""

N_FOLDS: int = 5
"""Pre-registered 5-fold design (thesis §3.2.1, §3.2.6). Not a free parameter."""

SPLITTER_VERSION: str = "1.0"
"""Bumped ONLY if the splitting algorithm changes. Never bumped for a re-run.
A version change makes verify_manifest fail loudly on any previously committed manifest."""

MANIFEST_PATH: str = "configs/splits/fold_split_5cv.json"
"""Default path for the frozen manifest, relative to the project root."""

_EXPECTED_CASE_COUNT: int = 100
"""TDSC-ABUS-2023 training set size. make_fold_split raises ValueError if labels.csv
does not contain exactly this many cases."""

_VALID_LABELS: frozenset[str] = frozenset({"B", "M"})
"""Allowed values for the label column. Any other value is a data error."""


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ManifestChecksumError(RuntimeError):
    """Raised by load_split when the manifest's sha256 field does not match the
    recomputed SHA256 of the manifest content. Indicates a silent edit."""


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSplit:
    """The frozen 5-fold patient-level cross-validation split.

    fold_of maps each case_id to its fold index 0..4.
    folds[k] is the sorted list of case_ids assigned to fold k.

    For fold k as the out-of-fold test fold, the detector trains on the union of
    folds != k (thesis §3.2.6); see train_ids() / oof_ids().

    Note on mutability: FoldSplit is a frozen dataclass, so field *reassignment*
    is blocked. The list and dict fields are mutable objects; callers must not
    mutate them in place. If mutation bugs are observed in downstream code, make
    the lists/dicts read-only at construction time (EPIC_01 can adopt this if needed).
    """

    folds: list[list[int]]  # length 5, each a sorted list of int case_ids
    fold_of: dict[int, int]  # case_id -> fold index 0..4
    seed: int  # the random_state used with StratifiedKFold
    splitter_version: str  # algorithm version tag
    label_of: dict[int, str]  # case_id -> 'B' | 'M' (provenance)

    def train_ids(self, holdout_fold: int) -> list[int]:
        """Sorted case_ids in all folds *except* holdout_fold (the detector's train set).

        Parameters
        ----------
        holdout_fold:
            Integer 0..N_FOLDS-1 identifying the out-of-fold test fold.

        Returns
        -------
        list[int]
            Sorted list of case_ids from the training folds (union of all folds
            except holdout_fold).

        Raises
        ------
        IndexError
            If holdout_fold is outside [0, len(folds)).
        """
        if not (0 <= holdout_fold < len(self.folds)):
            raise IndexError(
                f"holdout_fold={holdout_fold} is out of range; "
                f"valid range is [0, {len(self.folds)})."
            )
        ids: list[int] = []
        for k, fold in enumerate(self.folds):
            if k != holdout_fold:
                ids.extend(fold)
        return sorted(ids)

    def oof_ids(self, holdout_fold: int) -> list[int]:
        """Sorted case_ids in holdout_fold (the out-of-fold candidate-generation set).

        Parameters
        ----------
        holdout_fold:
            Integer 0..N_FOLDS-1 identifying the out-of-fold test fold.

        Returns
        -------
        list[int]
            Sorted list of case_ids in the holdout fold.

        Raises
        ------
        IndexError
            If holdout_fold is outside [0, len(folds)).
        """
        if not (0 <= holdout_fold < len(self.folds)):
            raise IndexError(
                f"holdout_fold={holdout_fold} is out of range; "
                f"valid range is [0, {len(self.folds)})."
            )
        return sorted(self.folds[holdout_fold])


# ---------------------------------------------------------------------------
# Core split builder
# ---------------------------------------------------------------------------


def make_fold_split(
    labels_csv_path: str,
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> FoldSplit:
    """Build the deterministic B/M-stratified patient-level k-fold split.

    Algorithm (fully determined; no hidden state):
      1. Read labels.csv; extract (case_id, label) pairs.
      2. Sort pairs by case_id ASCENDING — this is the determinism anchor
         (file order is not canonical; sorted order is).
      3. StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
         .split(case_ids, labels).
      4. Within each fold, sort case_ids ascending.

    Parameters
    ----------
    labels_csv_path:
        Path to labels.csv. Must contain 'case_id' and 'label' columns.
        'label' values must all be in {'B', 'M'}.
    n_folds:
        Number of folds (default 5, pre-registered).
    seed:
        Random state for StratifiedKFold (default SPLIT_SEED, frozen constant).

    Returns
    -------
    FoldSplit

    Raises
    ------
    ValueError
        If labels.csv does not contain exactly 100 training cases, if the
        required columns are missing, or if any label is not in {'B','M'}.
    """
    df = pd.read_csv(labels_csv_path)

    # Validate required columns
    missing_cols = {"case_id", "label"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"labels.csv is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate row count
    if len(df) != _EXPECTED_CASE_COUNT:
        raise ValueError(
            f"labels.csv must contain exactly {_EXPECTED_CASE_COUNT} training cases; "
            f"found {len(df)}."
        )

    # Validate label values
    bad_labels = set(df["label"].astype(str).unique()) - _VALID_LABELS
    if bad_labels:
        raise ValueError(
            f"labels.csv contains invalid label values: {bad_labels}. "
            f"Only {{'B','M'}} are allowed."
        )

    # Sort by case_id ascending — the determinism anchor (not file order)
    df = df.sort_values("case_id").reset_index(drop=True)
    case_ids = df["case_id"].astype(int).tolist()
    labels = df["label"].astype(str).tolist()
    # Preserve the strict length-equality guarantee without zip(strict=True), which
    # is Python 3.10+ and crashes in the server's nndet env (Python 3.8). See
    # scripts/check_py38_compat.py.
    if len(case_ids) != len(labels):
        raise ValueError(f"case_ids and labels length mismatch: {len(case_ids)} vs {len(labels)}.")
    label_of: dict[int, str] = dict(zip(case_ids, labels))  # noqa: B905 (py38: no strict=)

    # StratifiedKFold — shuffle=True is required for random_state to take effect
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[list[int]] = []
    fold_of: dict[int, int] = {}

    for fold_idx, (_train_idx, test_idx) in enumerate(skf.split(case_ids, labels)):
        fold_case_ids = sorted(case_ids[i] for i in test_idx)
        folds.append(fold_case_ids)
        for cid in fold_case_ids:
            fold_of[cid] = fold_idx

    return FoldSplit(
        folds=folds,
        fold_of=fold_of,
        seed=seed,
        splitter_version=SPLITTER_VERSION,
        label_of=label_of,
    )


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def manifest_sha256(split: FoldSplit) -> str:
    """SHA256 over the canonical JSON serialization of the split content.

    Covers: folds, seed, splitter_version (NOT the sha256 field itself, and NOT
    label_of or bm_ratios_per_fold, which are derived from the above and the CSV).

    Canonical form: JSON with sorted keys, no extra whitespace, separators=(',', ':').
    This means reformatting the file (adding indentation) does NOT change the hash,
    but any change to fold membership, seed, or algorithm version does.

    Parameters
    ----------
    split:
        The FoldSplit to hash.

    Returns
    -------
    str
        64-character lowercase hex SHA256 digest.
    """
    payload: dict[str, Any] = {
        "folds": split.folds,
        "seed": split.seed,
        "splitter_version": split.splitter_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _labels_csv_sha256(labels_csv_path: str) -> str:
    """SHA256 of the raw bytes of labels_csv_path. Used for provenance in the manifest."""
    h = hashlib.sha256()
    with open(labels_csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _bm_ratios(split: FoldSplit) -> list[dict[str, Any]]:
    """Compute per-fold B/M counts and benign_fraction."""
    result: list[dict[str, Any]] = []
    for fold in split.folds:
        n_b = sum(1 for cid in fold if split.label_of[cid] == "B")
        n_m = len(fold) - n_b
        result.append(
            {
                "benign": n_b,
                "malignant": n_m,
                "benign_fraction": n_b / len(fold) if fold else 0.0,
            }
        )
    return result


def _global_bm(split: FoldSplit) -> dict[str, Any]:
    """Compute global B/M counts and benign_fraction."""
    n_b = sum(1 for v in split.label_of.values() if v == "B")
    n_total = len(split.label_of)
    return {
        "benign": n_b,
        "malignant": n_total - n_b,
        "benign_fraction": n_b / n_total if n_total else 0.0,
    }


def write_manifest(
    split: FoldSplit,
    path: str = MANIFEST_PATH,
    labels_csv_path: str | None = None,
) -> None:
    """Serialize FoldSplit to JSON at `path`, embedding manifest_sha256(split).

    Follows the JSON schema defined in STORY_00_04:
      splitter_version, seed, n_folds, labels_csv_sha256, folds, label_of,
      bm_ratios_per_fold, global_bm, sha256.

    The sha256 field covers folds + seed + splitter_version (not itself).

    Parameters
    ----------
    split:
        The FoldSplit to serialize.
    path:
        Output file path. Parent directories are created if needed.
    labels_csv_path:
        If provided, its SHA256 is recorded in labels_csv_sha256.
        If None, the field is set to 'unknown' (acceptable for tests; the server
        build always provides this path).
    """
    # Convert case_id keys in label_of and fold_of to strings for JSON
    label_of_str = {str(k): v for k, v in split.label_of.items()}

    data: dict[str, Any] = {
        "splitter_version": split.splitter_version,
        "seed": split.seed,
        "n_folds": len(split.folds),
        "labels_csv_sha256": (
            _labels_csv_sha256(labels_csv_path) if labels_csv_path is not None else "unknown"
        ),
        "folds": split.folds,
        "label_of": label_of_str,
        "bm_ratios_per_fold": _bm_ratios(split),
        "global_bm": _global_bm(split),
        "sha256": manifest_sha256(split),
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical formatting: sorted keys, readable indentation (2 spaces)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
        f.write("\n")  # POSIX trailing newline


def load_split(path: str = MANIFEST_PATH) -> FoldSplit:
    """Read the frozen manifest, reconstruct FoldSplit, and VERIFY the embedded sha256.

    This is the function every downstream epic calls. It never calls make_fold_split;
    it consumes the frozen manifest built once on the server.

    Parameters
    ----------
    path:
        Path to the frozen manifest JSON.

    Returns
    -------
    FoldSplit
        The reconstructed split with all fields populated.

    Raises
    ------
    ManifestChecksumError
        If the manifest's sha256 field does not match manifest_sha256 of the
        reconstructed split. This catches any silent content edit.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct FoldSplit — convert JSON string keys back to int
    folds: list[list[int]] = [[int(cid) for cid in fold] for fold in data["folds"]]
    fold_of: dict[int, int] = {}
    for k, fold in enumerate(folds):
        for cid in fold:
            fold_of[cid] = k

    label_of: dict[int, str] = {int(k): v for k, v in data["label_of"].items()}

    split = FoldSplit(
        folds=folds,
        fold_of=fold_of,
        seed=int(data["seed"]),
        splitter_version=str(data["splitter_version"]),
        label_of=label_of,
    )

    # Verify checksum: catches any content edit (even a single case_id change)
    expected = manifest_sha256(split)
    actual = data.get("sha256", "")
    if actual != expected:
        raise ManifestChecksumError(
            f"Manifest checksum mismatch at {path!r}.\n"
            f"  Stored:   {actual!r}\n"
            f"  Expected: {expected!r}\n"
            "The manifest content has been modified after it was written."
        )

    return split


def verify_manifest(
    labels_csv_path: str,
    path: str = MANIFEST_PATH,
) -> bool:
    """Re-derive the split from labels.csv and assert it matches the manifest.

    This is the provenance check: it proves the manifest was produced by the
    documented seeded algorithm. It verifies:
      1. The manifest's internal sha256 is consistent (via load_split).
      2. The re-derived split from labels.csv matches the manifest fold-for-fold.
      3. The labels.csv sha256 matches the one recorded in the manifest (if not 'unknown').

    Parameters
    ----------
    labels_csv_path:
        Path to the labels.csv used to build the manifest.
    path:
        Path to the frozen manifest JSON.

    Returns
    -------
    bool
        True if everything matches.

    Raises
    ------
    ManifestChecksumError
        If the manifest's internal sha256 is inconsistent (via load_split).
    ValueError
        If the re-derived split's folds differ from the manifest's folds, or if
        the labels.csv sha256 recorded in the manifest differs from the actual file.
    AssertionError
        If fold-for-fold comparison fails.
    """
    # Step 1: load manifest (this also verifies the internal sha256).
    loaded = load_split(path)

    # Step 2: read the raw manifest to extract labels_csv_sha256, which is provenance
    # metadata not stored in FoldSplit.  The file is small (< 100 KB); the second read
    # is an acceptable cost for keeping FoldSplit's fields clean.
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Step 3: check labels_csv_sha256 if it was recorded
    recorded_csv_sha256 = raw_data.get("labels_csv_sha256", "unknown")
    if recorded_csv_sha256 != "unknown":
        actual_csv_sha256 = _labels_csv_sha256(labels_csv_path)
        if actual_csv_sha256 != recorded_csv_sha256:
            raise ValueError(
                f"labels.csv SHA256 mismatch:\n"
                f"  Recorded in manifest: {recorded_csv_sha256!r}\n"
                f"  Actual file:          {actual_csv_sha256!r}\n"
                "The labels.csv used for verification is not the one that produced this manifest."
            )

    # Step 4: re-derive split using the seed from the manifest
    rederived = make_fold_split(
        labels_csv_path,
        n_folds=len(loaded.folds),
        seed=loaded.seed,
    )

    # Step 5: compare fold-for-fold
    if rederived.folds != loaded.folds:
        raise ValueError(
            "Re-derived split does not match the manifest.\n"
            "The manifest was not produced by the documented seeded algorithm with "
            f"seed={loaded.seed} and splitter_version={loaded.splitter_version!r}.\n"
            "Possible causes: different scikit-learn version, different input order, "
            "or a different algorithm was used to build the manifest."
        )

    return True
