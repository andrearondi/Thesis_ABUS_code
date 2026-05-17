"""Tests for abus.data.split (STORY_00_04).

All tests use a synthetic 100-row labels.csv fixture with a realistic B/M mix.
Real labels.csv (100 training cases) lives only on the server; the committed manifest
(configs/splits/fold_split_5cv.json) is built there.

Covers:
  - test_reproducible:         two make_fold_split calls produce identical folds
  - test_coverage:             union of 5 folds == all 100 case_ids
  - test_disjoint:             5 folds are pairwise disjoint (no case appears twice)
  - test_stratified:           each fold's benign_fraction within +-0.10 of global
  - test_checksum_consistent:  manifest sha256 field equals manifest_sha256(split)
  - test_checksum_detects_tamper: mutating loaded split -> load_split raises ManifestChecksumError
  - test_train_oof_partition:  for each holdout_fold k, train_ids(k)+oof_ids(k) == 100 IDs, disjoint
  - test_rejects_wrong_case_count: 99-row CSV raises ValueError
  - test_rejects_invalid_label:    non-B/M label raises ValueError
  - test_folds_sorted:         case_ids within each fold are in ascending order
  - test_fold_of_consistent:   fold_of[id] is consistent with folds[k] membership
  - test_label_of_preserved:   label_of stores the original B/M values
  - test_verify_manifest_passes: verify_manifest returns True on a matching CSV+manifest
  - test_verify_manifest_detects_rederive_mismatch: different seed -> mismatch raises
  - test_load_split_verifies_checksum: load_split reads + verifies (round-trip through write)
  - test_manifest_sha256_stable: same split -> same hash (determinism)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Deferred import — collection succeeds even before split.py exists.
# ---------------------------------------------------------------------------


def _import_split():  # type: ignore[no-untyped-def]
    from abus.data import split as split_mod  # noqa: PLC0415

    return split_mod


# ---------------------------------------------------------------------------
# Synthetic-fixture helpers
# ---------------------------------------------------------------------------

_GLOBAL_B_FRACTION = 0.72  # 72/100 benign — typical for TDSC-ABUS-2023


def _make_labels_csv(
    tmp_path: Path,
    n_cases: int = 100,
    b_fraction: float = _GLOBAL_B_FRACTION,
    start_id: int = 1,
) -> str:
    """Write a synthetic labels.csv with `n_cases` rows.

    Case IDs run from start_id to start_id+n_cases-1.
    Labels cycle B/M to achieve approximately b_fraction benign cases.
    Returns the CSV path as str.
    """
    ids = list(range(start_id, start_id + n_cases))
    n_benign = round(n_cases * b_fraction)
    # Shuffle deterministically so all folds get a mix (not all B first).
    # Use a simple interleave so the first n_benign IDs are not all B.
    interleaved: list[str] = []
    bi, mi = 0, 0
    b_list = ["B"] * n_benign
    m_list = ["M"] * (n_cases - n_benign)
    for _ in range(n_cases):
        frac = bi / max(n_benign, 1)
        if bi < n_benign and (mi >= len(m_list) or frac <= b_fraction):
            interleaved.append(b_list[bi])
            bi += 1
        else:
            interleaved.append(m_list[mi])
            mi += 1

    # Realistic columns matching labels.csv schema (story spec + local_data.md)
    df = pd.DataFrame(
        {
            "case_id": ids,
            "label": interleaved,
            "data_path": [f"data/case_{idx:03d}.nrrd" for idx in ids],
            "mask_path": [f"masks/case_{idx:03d}.nrrd" for idx in ids],
        }
    )
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ---------------------------------------------------------------------------
# RED tests — will fail until split.py is implemented
# ---------------------------------------------------------------------------


class TestReproducible:
    """make_fold_split is deterministic: same input -> identical folds."""

    def test_reproducible(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split1 = split_mod.make_fold_split(csv_path)
        split2 = split_mod.make_fold_split(csv_path)
        assert split1.folds == split2.folds
        assert split1.fold_of == split2.fold_of
        assert split1.seed == split2.seed

    def test_different_seed_gives_different_folds(self, tmp_path: Path) -> None:
        """A different seed produces different folds (probability-near-certain)."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split1 = split_mod.make_fold_split(csv_path, seed=20230516)
        split2 = split_mod.make_fold_split(csv_path, seed=99999999)
        # With 100 cases / 5 folds it is astronomically unlikely two different seeds
        # produce identical fold membership lists.
        assert split1.folds != split2.folds


class TestCoverage:
    """Union of the 5 folds == exactly the set of case_ids in labels.csv."""

    def test_coverage(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        df = pd.read_csv(csv_path)
        expected_ids = set(df["case_id"].astype(int))

        all_ids: list[int] = []
        for fold in split.folds:
            all_ids.extend(fold)

        assert set(all_ids) == expected_ids

    def test_total_count_is_100(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        total = sum(len(f) for f in split.folds)
        assert total == 100


class TestDisjoint:
    """The 5 folds are pairwise disjoint (the leakage guard, EPIC_00 gate G00.7)."""

    def test_disjoint(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        seen: set[int] = set()
        for k, fold in enumerate(split.folds):
            fold_set = set(fold)
            overlap = seen & fold_set
            assert (
                not overlap
            ), f"Fold {k} shares case_ids {overlap} with a previous fold — leakage!"
            seen |= fold_set

    def test_each_id_appears_exactly_once_in_fold_of(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        # fold_of must have exactly 100 entries (one per case)
        assert len(split.fold_of) == 100


class TestStratified:
    """Each fold's benign_fraction is within +-0.10 of the global benign_fraction."""

    def test_stratified(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        # Compute global benign_fraction from label_of
        n_total = len(split.label_of)
        n_benign_global = sum(1 for v in split.label_of.values() if v == "B")
        global_b_frac = n_benign_global / n_total

        for k, fold in enumerate(split.folds):
            n_b = sum(1 for case_id in fold if split.label_of[case_id] == "B")
            fold_b_frac = n_b / len(fold)
            assert abs(fold_b_frac - global_b_frac) <= 0.10, (
                f"Fold {k}: benign_fraction={fold_b_frac:.3f} deviates from "
                f"global={global_b_frac:.3f} by more than 0.10"
            )


class TestChecksum:
    """SHA256 checksum field in the manifest is consistent and detects tampering."""

    def test_checksum_consistent(self, tmp_path: Path) -> None:
        """manifest_sha256(split) == the sha256 embedded in a freshly written manifest."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "test_manifest.json")
        split_mod.write_manifest(split, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        expected_hash = split_mod.manifest_sha256(split)
        assert data["sha256"] == expected_hash

    def test_checksum_detects_tamper(self, tmp_path: Path) -> None:
        """load_split raises ManifestChecksumError when a case_id is mutated."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "tampered_manifest.json")
        split_mod.write_manifest(split, manifest_path)

        # Tamper: replace one case_id in fold 0 with a non-existent id
        with open(manifest_path) as f:
            data = json.load(f)
        if data["folds"][0]:
            data["folds"][0][0] = 999999  # non-existent case id
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(split_mod.ManifestChecksumError):
            split_mod.load_split(manifest_path)

    def test_manifest_sha256_stable(self, tmp_path: Path) -> None:
        """manifest_sha256 is deterministic: same split object -> same hash."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        h1 = split_mod.manifest_sha256(split)
        h2 = split_mod.manifest_sha256(split)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest is 64 chars


class TestTrainOofPartition:
    """train_ids(k) and oof_ids(k) partition the full 100 IDs for every k."""

    def test_train_oof_partition(self, tmp_path: Path) -> None:
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        all_ids = set(split.fold_of.keys())

        for k in range(5):
            train = set(split.train_ids(k))
            oof = set(split.oof_ids(k))
            assert train & oof == set(), f"fold {k}: train and oof overlap"
            assert train | oof == all_ids, f"fold {k}: train+oof does not cover all IDs"

    def test_oof_size_approx_20(self, tmp_path: Path) -> None:
        """With 100 cases and 5 folds, each OOF fold should be ~20 cases."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        for k in range(5):
            n_oof = len(split.oof_ids(k))
            assert 18 <= n_oof <= 22, f"fold {k}: oof size={n_oof}, expected ~20"

    def test_train_oof_are_sorted(self, tmp_path: Path) -> None:
        """train_ids and oof_ids return sorted lists."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        for k in range(5):
            train = split.train_ids(k)
            oof = split.oof_ids(k)
            assert train == sorted(train), f"fold {k}: train_ids not sorted"
            assert oof == sorted(oof), f"fold {k}: oof_ids not sorted"

    def test_out_of_range_holdout_raises(self, tmp_path: Path) -> None:
        """train_ids and oof_ids raise IndexError for out-of-range holdout_fold."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        with pytest.raises(IndexError):
            split.train_ids(5)  # valid range is 0..4
        with pytest.raises(IndexError):
            split.oof_ids(-1)
        with pytest.raises(IndexError):
            split.train_ids(99)


class TestValidation:
    """make_fold_split raises ValueError for invalid inputs."""

    def test_rejects_wrong_case_count(self, tmp_path: Path) -> None:
        """A labels.csv with 99 rows raises ValueError."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path, n_cases=99)
        with pytest.raises(ValueError, match="100"):
            split_mod.make_fold_split(csv_path)

    def test_rejects_invalid_label(self, tmp_path: Path) -> None:
        """A label not in {'B','M'} raises ValueError."""
        split_mod = _import_split()
        # Build valid CSV then corrupt one label
        csv_path = _make_labels_csv(tmp_path)
        df = pd.read_csv(csv_path)
        df.loc[0, "label"] = "X"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="label"):
            split_mod.make_fold_split(csv_path)

    def test_rejects_missing_label_column(self, tmp_path: Path) -> None:
        """A CSV without a 'label' column raises ValueError."""
        split_mod = _import_split()
        df = pd.DataFrame({"case_id": list(range(1, 101))})
        csv_path = str(tmp_path / "no_label.csv")
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="label"):
            split_mod.make_fold_split(csv_path)


class TestFoldsStructure:
    """Internal structure invariants of FoldSplit."""

    def test_folds_sorted(self, tmp_path: Path) -> None:
        """case_ids within each fold are in ascending order."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        for k, fold in enumerate(split.folds):
            assert fold == sorted(fold), f"Fold {k} is not sorted ascending"

    def test_n_folds(self, tmp_path: Path) -> None:
        """split.folds has exactly 5 elements."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)
        assert len(split.folds) == 5

    def test_fold_of_consistent(self, tmp_path: Path) -> None:
        """fold_of[case_id] == k iff case_id in folds[k]."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        for k, fold in enumerate(split.folds):
            for case_id in fold:
                assert (
                    split.fold_of[case_id] == k
                ), f"fold_of[{case_id}]={split.fold_of[case_id]} but case is in folds[{k}]"

    def test_label_of_preserved(self, tmp_path: Path) -> None:
        """label_of stores 'B' or 'M' for every case_id in the split."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        assert len(split.label_of) == 100
        for v in split.label_of.values():
            assert v in ("B", "M"), f"label_of has unexpected value: {v!r}"

    def test_seed_and_version_stored(self, tmp_path: Path) -> None:
        """FoldSplit stores the seed and splitter_version used."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        assert split.seed == split_mod.SPLIT_SEED
        assert split.splitter_version == split_mod.SPLITTER_VERSION


class TestLoadSplit:
    """load_split reads a written manifest and verifies the checksum."""

    def test_load_split_verifies_checksum(self, tmp_path: Path) -> None:
        """Round-trip: write then load produces an equal FoldSplit."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        split_mod.write_manifest(split, manifest_path)
        loaded = split_mod.load_split(manifest_path)

        assert loaded.folds == split.folds
        assert loaded.fold_of == split.fold_of
        assert loaded.seed == split.seed
        assert loaded.splitter_version == split.splitter_version
        assert loaded.label_of == split.label_of

    def test_load_split_raises_on_bad_checksum(self, tmp_path: Path) -> None:
        """load_split raises ManifestChecksumError when sha256 field is wrong."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest_bad_hash.json")
        split_mod.write_manifest(split, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)
        data["sha256"] = "0" * 64  # corrupt the hash
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(split_mod.ManifestChecksumError):
            split_mod.load_split(manifest_path)


class TestVerifyManifest:
    """verify_manifest re-derives the split from labels.csv and checks the manifest."""

    def test_verify_manifest_passes(self, tmp_path: Path) -> None:
        """verify_manifest returns True when CSV matches the manifest."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        split_mod.write_manifest(split, manifest_path)

        result = split_mod.verify_manifest(csv_path, manifest_path)
        assert result is True

    def test_verify_manifest_detects_rederive_mismatch(self, tmp_path: Path) -> None:
        """verify_manifest raises if the manifest folds do not match re-derivation.

        Simulated by: building a valid manifest with the default seed, then patching
        the manifest's folds to be those from a DIFFERENT seed (while recomputing sha256
        so load_split does not fail on the internal checksum — the mismatch is at the
        fold comparison step, not the checksum step).
        """
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)

        # Build two splits with different seeds
        split_default = split_mod.make_fold_split(csv_path, seed=split_mod.SPLIT_SEED)
        split_alt = split_mod.make_fold_split(csv_path, seed=99999999)

        # If by an astronomically unlikely chance the folds are the same, skip
        if split_default.folds == split_alt.folds:
            pytest.skip("seeds produced identical folds (extremely unlikely)")

        # Write the default-seed manifest, then patch its folds to the alt-seed folds
        # and recompute sha256 so load_split passes — the mismatch will be caught by
        # the fold comparison in verify_manifest.
        manifest_path = str(tmp_path / "patched_manifest.json")
        split_mod.write_manifest(split_default, manifest_path, labels_csv_path=csv_path)

        with open(manifest_path) as f:
            data = json.load(f)

        # Patch folds to the alt-seed folds but keep the same seed value
        data["folds"] = split_alt.folds
        # Recompute sha256 over the patched content so load_split checksum passes
        patched_split = split_mod.FoldSplit(
            folds=split_alt.folds,
            fold_of={cid: k for k, fold in enumerate(split_alt.folds) for cid in fold},
            seed=int(data["seed"]),
            splitter_version=str(data["splitter_version"]),
            label_of=split_default.label_of,
        )
        data["sha256"] = split_mod.manifest_sha256(patched_split)
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        # verify_manifest re-derives with the manifest's seed=SPLIT_SEED but finds
        # that the re-derived folds (default-seed) differ from the patched folds
        # (alt-seed) → must raise ValueError.
        with pytest.raises((ValueError, AssertionError)):
            split_mod.verify_manifest(csv_path, manifest_path)

    def test_verify_manifest_detects_csv_change(self, tmp_path: Path) -> None:
        """verify_manifest raises when labels.csv has changed (different labels_csv_sha256).

        The manifest must be written WITH labels_csv_path so the SHA256 is recorded.
        Without it the field is 'unknown' and the check is skipped — this test
        specifically verifies the guaranteed-detection path via labels_csv_sha256.
        """
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        # Must pass labels_csv_path so labels_csv_sha256 is recorded in the manifest.
        split_mod.write_manifest(split, manifest_path, labels_csv_path=csv_path)

        # Modify the CSV (change a label) — this changes labels_csv_sha256
        df = pd.read_csv(csv_path)
        if df.loc[0, "label"] == "B":
            df.loc[0, "label"] = "M"
        else:
            df.loc[0, "label"] = "B"
        df.to_csv(csv_path, index=False)

        with pytest.raises((ValueError, AssertionError)):
            split_mod.verify_manifest(csv_path, manifest_path)


class TestWriteManifest:
    """write_manifest serializes all required fields to JSON."""

    def test_manifest_has_required_fields(self, tmp_path: Path) -> None:
        """The written manifest must contain all schema fields."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        split_mod.write_manifest(split, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        required_keys = {
            "splitter_version",
            "seed",
            "n_folds",
            "labels_csv_sha256",
            "folds",
            "label_of",
            "bm_ratios_per_fold",
            "global_bm",
            "sha256",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"Manifest missing keys: {missing}"

    def test_manifest_folds_length(self, tmp_path: Path) -> None:
        """Manifest folds list has exactly 5 sub-lists."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        split_mod.write_manifest(split, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        assert len(data["folds"]) == 5

    def test_manifest_bm_ratios_per_fold_structure(self, tmp_path: Path) -> None:
        """Each element of bm_ratios_per_fold has benign, malignant, benign_fraction."""
        split_mod = _import_split()
        csv_path = _make_labels_csv(tmp_path)
        split = split_mod.make_fold_split(csv_path)

        manifest_path = str(tmp_path / "manifest.json")
        split_mod.write_manifest(split, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        for i, ratio in enumerate(data["bm_ratios_per_fold"]):
            assert "benign" in ratio, f"bm_ratios_per_fold[{i}] missing 'benign'"
            assert "malignant" in ratio, f"bm_ratios_per_fold[{i}] missing 'malignant'"
            assert "benign_fraction" in ratio, f"bm_ratios_per_fold[{i}] missing 'benign_fraction'"
