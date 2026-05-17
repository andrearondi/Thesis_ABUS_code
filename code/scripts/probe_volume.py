#!/usr/bin/env python
"""Probe a single ABUS NRRD file and print header vs injected geometry.

Usage
-----
    python scripts/probe_volume.py <path/to/DATA_NNN.nrrd>
    python scripts/probe_volume.py <path/to/MASK_NNN.nrrd>

Prints the array shape, dtype, intensity range, raw header spacing (which is
the identity placeholder for TDSC-ABUS-2023 files), and the injected
CANONICAL_SPACING_MM — so the user can eyeball the placeholder-vs-injected
distinction at a glance.

Exit codes
----------
0  success (file loaded and guards passed)
1  SpacingPlaceholderError (header is not the identity placeholder)
2  ValueError (filename pattern mismatch or non-binary mask)
3  FileNotFoundError
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the installed package (or src/ in dev mode) is importable when the
# script is run directly.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from abus.io.loader import (  # noqa: E402
    CANONICAL_ORIGIN_MM,
    CANONICAL_SPACING_MM,
    MaskRecord,
    SpacingPlaceholderError,
    VolumeRecord,
    load_mask,
    load_volume,
)


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def probe(path: str) -> int:  # noqa: PLR0912
    """Load and probe one NRRD file. Returns exit code."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 3

    name = p.name
    is_mask = name.startswith("MASK_")
    is_volume = name.startswith("DATA_")

    if not is_mask and not is_volume:
        print(
            f"WARNING: filename {name!r} does not match DATA_<NNN>.nrrd or "
            "MASK_<NNN>.nrrd — will attempt volume load anyway.",
            file=sys.stderr,
        )

    rec: VolumeRecord | MaskRecord
    try:
        if is_mask:
            rec = load_mask(path)
        else:
            rec = load_volume(path)
    except SpacingPlaceholderError as exc:
        print(f"SPACING-PLACEHOLDER-ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"VALUE-ERROR: {exc}", file=sys.stderr)
        return 2

    arr = rec.array

    _print_section(f"File:  {p.name}")
    print(f"  Source path  : {rec.source_path}")
    print(f"  Case ID      : {rec.case_id}")

    _print_section("Array properties")
    print(f"  Shape  (d0,d1,d2) : {arr.shape}")
    print(f"  Dtype             : {arr.dtype}")
    print(f"  Intensity min/max : {int(arr.min())} / {int(arr.max())}")
    print(f"  Intensity mean    : {float(arr.mean()):.4f}")
    if is_mask:
        unique = sorted(int(v) for v in np.unique(arr))
        print(f"  Unique values     : {unique}")
        print(f"  Foreground voxels : {int(arr.sum())}")

    _print_section("Raw NRRD header (placeholder geometry)")
    hdr = rec.raw_header
    print(f"  space directions  : {hdr.get('space directions')}  <- identity PLACEHOLDER")
    print(f"  space origin      : {hdr.get('space origin')}  <- placeholder [0,0,0]")
    print(f"  sizes             : {hdr.get('sizes')}")
    print(f"  encoding          : {hdr.get('encoding')}")
    print(f"  space             : {hdr.get('space')}")
    print(f"  kinds             : {hdr.get('kinds')}")

    _print_section("Injected physical frame")
    print(f"  CANONICAL_SPACING_MM  : {CANONICAL_SPACING_MM}  (d0, d1, d2), mm")
    print(f"  CANONICAL_ORIGIN_MM   : {CANONICAL_ORIGIN_MM}  (d0, d1, d2), mm")
    print(f"  rec.spacing_mm        : {rec.spacing_mm}")
    print(f"  rec.origin_mm         : {rec.origin_mm}")

    _print_section("Physical extent")
    s = CANONICAL_SPACING_MM
    for i, (n, sp) in enumerate(zip(arr.shape, s, strict=False)):
        extent_mm = n * sp
        print(f"  axis {i}: {n} voxels × {sp} mm/voxel = {extent_mm:.2f} mm")

    print()
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <path/to/DATA_NNN.nrrd or MASK_NNN.nrrd>")
        sys.exit(1)
    sys.exit(probe(sys.argv[1]))


if __name__ == "__main__":
    main()
