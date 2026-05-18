#!/usr/bin/env python
"""Beam-axis investigation renders (supervisor review issue C1).

Renders large single-plane slices of local TDSC-ABUS-2023 volumes plus an
axis intensity-profile plot, so the acoustic scan-beam axis can be identified
by visual inspection: the skin / transducer line, the depth attenuation
gradient, posterior acoustic shadowing behind the tumour, and Cooper's-ligament
shadow streaks all reveal the beam direction.

Outputs per case into ``docs/local_data_check/beam_axis/``:
  caseNNN_planeD1D2.png   slice along storage axis 0  -- plane (d1 vert, d2 horiz)
  caseNNN_planeD0D2.png   slice along storage axis 1  -- plane (d0 vert, d2 horiz)
  caseNNN_planeD0D1.png   slice along storage axis 2  -- plane (d0 vert, d1 horiz)
  caseNNN_profiles.png    mean intensity vs index along each storage axis

Slices are drawn at voxel-equal aspect (1 voxel -> 1 square pixel).  This is
assumption-free: the orientation of a shadow streak relative to a storage axis
is read off without relying on the (unconfirmed) physical-spacing assignment.

Usage:
  python scripts/inspect_beam_axis.py [CASE_ID ...]      # default: 100
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
_OUT_DIR = _REPO_ROOT / "docs" / "local_data_check" / "beam_axis"

# storage axis -> (row axis, col axis) of the 2-D slice obtained by indexing it
_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
_PLANE_NAME = {0: "D1D2", 1: "D0D2", 2: "D0D1"}


def _label_map() -> dict[int, str]:
    """case_id -> 'B'/'M' from labels.csv; empty dict if unavailable."""
    csv_path = _DATA_ROOT / "labels.csv"
    if not csv_path.exists():
        return {}
    out: dict[int, str] = {}
    try:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                vals = list(row.values())
                cid = "".join(c for c in str(vals[0]) if c.isdigit())
                if cid:
                    out[int(cid)] = str(vals[1]).strip()
    except Exception:
        return {}
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    from abus.io.loader import load_mask, load_volume

    case_ids = [int(a) for a in sys.argv[1:]] or [100]
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = _label_map()

    for cid in case_ids:
        vpath = _DATA_ROOT / "DATA" / f"DATA_{cid}.nrrd"
        mpath = _DATA_ROOT / "MASK" / f"MASK_{cid}.nrrd"
        if not vpath.exists() or not mpath.exists():
            print(f"SKIP case {cid}: volume or mask file missing")
            continue

        vol = load_volume(str(vpath))
        mask = load_mask(str(mpath))
        arr = vol.array
        marr = mask.array
        sp = tuple(float(s) for s in vol.spacing_mm)
        lab = labels.get(cid, "?")

        fg = np.argwhere(marr > 0)
        bb_min = [int(fg[:, i].min()) for i in range(3)]
        bb_max = [int(fg[:, i].max()) for i in range(3)]
        centroid = [int(round(float(fg[:, i].mean()))) for i in range(3)]

        print(f"\ncase {cid}  [label {lab}]  shape={tuple(marr.shape)}  centroid={centroid}")
        print(f"  tumour bbox  min {bb_min}  max {bb_max}")

        # ---- per-plane slices through the tumour centroid -------------------
        for axis in range(3):
            idx = centroid[axis]
            if axis == 0:
                img = arr[idx, :, :]
            elif axis == 1:
                img = arr[:, idx, :]
            else:
                img = arr[:, :, idx]
            ra, ca = _PLANE[axis]
            lo, hi = (float(x) for x in np.percentile(img, [2.0, 98.0]))

            h, w = img.shape
            long_in = 13.0
            figsize = (long_in * w / h, long_in) if h >= w else (long_in, long_in * h / w)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(
                img, cmap="gray", origin="upper", aspect="equal",
                vmin=lo, vmax=hi, interpolation="nearest",
            )
            ax.add_patch(patches.Rectangle(
                (bb_min[ca], bb_min[ra]),
                bb_max[ca] - bb_min[ca] + 1, bb_max[ra] - bb_min[ra] + 1,
                linewidth=1.4, edgecolor="red", facecolor="none",
            ))
            ax.plot([centroid[ca]], [centroid[ra]], "r+", markersize=13, markeredgewidth=1.6)
            ax.set_xlabel(f"d{ca}  -- storage axis {ca}  (n={marr.shape[ca]}, {sp[ca]:.3f} mm/voxel)")
            ax.set_ylabel(f"d{ra}  -- storage axis {ra}  (n={marr.shape[ra]}, {sp[ra]:.3f} mm/voxel)")
            ax.set_title(
                f"case {cid} [{lab}]  --  slice along axis {axis} @ idx {idx}\n"
                f"plane d{ra} (vertical) x d{ca} (horizontal)  --  voxel-equal aspect"
            )
            out = _OUT_DIR / f"case{cid:03d}_plane{_PLANE_NAME[axis]}.png"
            fig.savefig(str(out), dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out.relative_to(_REPO_ROOT)}")

        # ---- axis intensity profiles ---------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
        for axis in range(3):
            other = tuple(a for a in range(3) if a != axis)
            prof = arr.mean(axis=other)
            axes[axis].plot(np.arange(prof.size), prof, color="navy")
            axes[axis].axvline(centroid[axis], color="red", linestyle="--", linewidth=1)
            axes[axis].set_title(f"axis {axis}  (n={prof.size}, {sp[axis]:.3f} mm/vox)")
            axes[axis].set_xlabel(f"index along storage axis {axis}")
            axes[axis].set_ylabel("mean intensity")
            axes[axis].grid(alpha=0.3)
        fig.suptitle(
            f"case {cid} [{lab}]  --  mean intensity vs slice index along each storage axis\n"
            "(the acoustic-beam axis carries the depth attenuation / near-field "
            "signature; red dashed = tumour centroid)",
            fontsize=11,
        )
        out = _OUT_DIR / f"case{cid:03d}_profiles.png"
        fig.savefig(str(out), dpi=125, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.relative_to(_REPO_ROOT)}")

    print(f"\nDone. Figures in {_OUT_DIR.relative_to(_REPO_ROOT)}/")


if __name__ == "__main__":
    main()
