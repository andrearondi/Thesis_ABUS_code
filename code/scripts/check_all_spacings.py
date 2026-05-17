#!/usr/bin/env python
"""CPU-only server script: verify that every ABUS NRRD file carries the identity
placeholder header expected by the loader.

Runs the ``SpacingPlaceholderError`` guard from ``abus.io.loader`` over every
``DATA_*.nrrd`` and ``MASK_*.nrrd`` found recursively under ``--root``.  Prints
one PASS or FAIL line per file, then a summary.

Usage
-----
    python scripts/check_all_spacings.py --root <dataset-root>

``<dataset-root>`` is the top of the TDSC-ABUS-2023 directory tree (may contain
Train/, Validation/, Test/ subdirectories — the script discovers all NRRD files
recursively).

Exit codes
----------
0  all volumes PASS
1  one or more volumes FAIL (or an unexpected read error occurred)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

# Ensure the package is importable when run from the repo root.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from abus.io.loader import SpacingPlaceholderError, load_mask, load_volume  # noqa: E402


def _iter_nrrd(root: Path) -> Iterator[Path]:
    """Yield all .nrrd files under root, sorted for deterministic output."""
    yield from sorted(root.rglob("DATA_*.nrrd"))
    yield from sorted(root.rglob("MASK_*.nrrd"))


def check_file(path: Path) -> tuple[bool, str]:
    """Check one NRRD file. Returns (passed, message)."""
    name = path.name
    try:
        if name.startswith("DATA_"):
            load_volume(str(path))
        elif name.startswith("MASK_"):
            load_mask(str(path))
        else:
            # Should not occur given the glob pattern, but be safe.
            return False, f"SKIP  {path}  (unrecognised pattern)"
    except SpacingPlaceholderError as exc:
        return False, f"FAIL  {path}  SpacingPlaceholderError: {exc}"
    except ValueError as exc:
        # Could be a non-binary mask or filename mismatch.
        return False, f"FAIL  {path}  ValueError: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"FAIL  {path}  UnexpectedError({type(exc).__name__}): {exc}"
    return True, f"PASS  {path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check all NRRD files under --root carry the identity placeholder header."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory of the TDSC-ABUS-2023 dataset (may contain Train/Val/Test sub-dirs).",
    )
    args = parser.parse_args()
    root: Path = args.root

    if not root.exists():
        print(f"ERROR: --root {root} does not exist.", file=sys.stderr)
        sys.exit(1)

    files = list(_iter_nrrd(root))
    if not files:
        print(f"WARNING: no DATA_*.nrrd or MASK_*.nrrd files found under {root}.", file=sys.stderr)
        sys.exit(1)

    n_pass = 0
    n_fail = 0
    for path in files:
        passed, msg = check_file(path)
        print(msg)
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    print()
    print(f"{'='*60}")
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL out of {len(files)} files.")
    print(f"{'='*60}")

    if n_fail > 0:
        print(
            "\nACTION REQUIRED: one or more files did not carry the identity placeholder.\n"
            "Blind injection of CANONICAL_SPACING_MM would corrupt their geometry.\n"
            "Consult the architect before proceeding to EPIC_01 training.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
