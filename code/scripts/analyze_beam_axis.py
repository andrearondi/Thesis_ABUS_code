#!/usr/bin/env python
"""Quantitative cross-check for the beam-axis investigation (supervisor C1).

For every storage axis of each requested ABUS volume, computes the 1-D mean-
intensity profile (mean over the other two axes) and samples it at seven
positions along the axis.  The acoustic-beam / depth axis is recognised
objectively by the ultrasound depth signature -- a dark near-field standoff,
a bright near-field, a long attenuation decline and a dark anechoic far field
(a strongly asymmetric profile) -- which the lateral and elevational (sweep)
axes do not carry.  No image rendering; pure NumPy, so it is unaffected by the
image-reading limit and can sweep the entire local validation split.

Usage:
  python scripts/analyze_beam_axis.py [CASE_ID ...]      # default: 100
"""

from __future__ import annotations

import sys
from pathlib import Path

_DATA_ROOT = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
_POS = (0.02, 0.15, 0.30, 0.50, 0.70, 0.85, 0.98)


def main() -> None:
    import numpy as np

    from abus.io.loader import load_volume

    case_ids = [int(a) for a in sys.argv[1:]] or [100]
    hdr = (
        f"{'case':>5} {'ax':>3} {'n':>5} {'mm/vox':>7} {'min':>6} {'max':>6}"
        + "".join(f"  p{int(q * 100):02d}" for q in _POS)
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cid in case_ids:
        vpath = _DATA_ROOT / "DATA" / f"DATA_{cid}.nrrd"
        if not vpath.exists():
            print(f"{cid:>5}  SKIP (missing)", flush=True)
            continue
        vol = load_volume(str(vpath))
        arr = vol.array
        sp = tuple(float(s) for s in vol.spacing_mm)
        for axis in range(3):
            other = tuple(a for a in range(3) if a != axis)
            p = arr.mean(axis=other)
            n = p.size
            samples = [p[min(n - 1, int(q * n))] for q in _POS]
            row = (
                f"{cid:>5} {axis:>3} {n:>5} {sp[axis]:>7.3f} "
                f"{p.min():>6.1f} {p.max():>6.1f}"
                + "".join(f" {s:>5.1f}" for s in samples)
            )
            print(row, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
