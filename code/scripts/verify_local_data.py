"""Verify TDSC-ABUS-2023 validation-split conventions on the local laptop copy.

Goal: empirically determine and document
  1. NRRD header contents (sizes, spacings, axis directions, data type).
  2. The axis convention used by bbx_labels.csv vs. the NRRD storage order.
  3. Whether `(c - len/2, c + len/2)` from the CSV recovers the mask's tight bbox.
  4. Inclusivity/exclusivity of bbox endpoints.

Usage:
    python scripts/verify_local_data.py
    python scripts/verify_local_data.py --dataset /path/to/Validation --ncases 5

Run with the thesis conda env's python:
    /Users/andrearondi/anaconda3/envs/thesis/bin/python scripts/verify_local_data.py
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import nrrd
import numpy as np
import pandas as pd

DEFAULT_DATASET = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")


def mask_tight_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (min_idx, max_idx_inclusive) along each axis of the binary mask.

    Coordinates are voxel indices in the array's storage order.
    """
    if mask.sum() == 0:
        raise ValueError("mask is empty")
    nonzero = np.where(mask > 0)
    mins = tuple(int(c.min()) for c in nonzero)
    maxs = tuple(int(c.max()) for c in nonzero)
    return mins, maxs


def csv_bbox_endpoints(row: pd.Series) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert (c_x, c_y, c_z, len_x, len_y, len_z) into (min_xyz, max_xyz)."""
    cx, cy, cz = row.c_x, row.c_y, row.c_z
    lx, ly, lz = row.len_x, row.len_y, row.len_z
    mins = (cx - lx / 2, cy - ly / 2, cz - lz / 2)
    maxs = (cx + lx / 2, cy + ly / 2, cz + lz / 2)
    return mins, maxs


def try_axis_permutations(
    csv_mins: tuple[float, float, float],
    csv_maxs: tuple[float, float, float],
    mask_mins: tuple[int, int, int],
    mask_maxs_inclusive: tuple[int, int, int],
) -> list[dict]:
    """Try all 6 permutations and ± inclusivity tweaks to find the one matching the mask."""
    results = []
    for perm in permutations(range(3)):
        # Apply permutation to the CSV bbox so that csv[perm[i]] aligns with mask axis i
        csv_min_perm = (csv_mins[perm[0]], csv_mins[perm[1]], csv_mins[perm[2]])
        csv_max_perm = (csv_maxs[perm[0]], csv_maxs[perm[1]], csv_maxs[perm[2]])

        # Compare against the mask bbox (mask_maxs is inclusive; CSV max could be inclusive or exclusive)
        for inclusivity_offset in (0, 1):
            # mask_maxs_inclusive vs. csv_max_perm - inclusivity_offset
            err_min = max(abs(csv_min_perm[i] - mask_mins[i]) for i in range(3))
            err_max = max(abs(csv_max_perm[i] - (mask_maxs_inclusive[i] + inclusivity_offset)) for i in range(3))
            err_total = max(err_min, err_max)
            results.append({
                "csv_axis_to_storage_axis": perm,
                "csv_max_is_exclusive": bool(inclusivity_offset),
                "min_residual": err_min,
                "max_residual": err_max,
                "total_residual": err_total,
            })
    results.sort(key=lambda d: d["total_residual"])
    return results


def inspect_case(case_id: int, dataset: Path, csv_row: pd.Series) -> dict:
    data_path = dataset / "DATA" / f"DATA_{case_id}.nrrd"
    mask_path = dataset / "MASK" / f"MASK_{case_id}.nrrd"

    data_arr, data_header = nrrd.read(str(data_path))
    mask_arr, mask_header = nrrd.read(str(mask_path))

    report: dict = {
        "case_id": case_id,
        "data": {
            "shape": list(data_arr.shape),
            "dtype": str(data_arr.dtype),
            "intensity_min": int(data_arr.min()),
            "intensity_max": int(data_arr.max()),
            "intensity_mean": float(data_arr.mean()),
            "header_sizes": list(map(int, data_header.get("sizes", []))),
            "header_space": data_header.get("space"),
            "header_space_directions": (
                np.asarray(data_header["space directions"], dtype=float).tolist()
                if "space directions" in data_header
                else None
            ),
            "header_space_origin": (
                list(map(float, data_header.get("space origin", [])))
                if "space origin" in data_header
                else None
            ),
            "header_kinds": data_header.get("kinds"),
            "header_endian": data_header.get("endian"),
            "header_encoding": data_header.get("encoding"),
            "header_type": data_header.get("type"),
            "header_keys": sorted(list(data_header.keys())),
        },
        "mask": {
            "shape": list(mask_arr.shape),
            "dtype": str(mask_arr.dtype),
            "unique_values": list(map(int, np.unique(mask_arr).tolist())),
            "foreground_voxels": int((mask_arr > 0).sum()),
            "header_space": mask_header.get("space"),
            "header_space_directions": (
                np.asarray(mask_header["space directions"], dtype=float).tolist()
                if "space directions" in mask_header
                else None
            ),
            "header_sizes": list(map(int, mask_header.get("sizes", []))),
        },
        "shape_match_data_vs_mask": list(data_arr.shape) == list(mask_arr.shape),
    }

    # Mask tight bbox
    mask_mins, mask_maxs_inclusive = mask_tight_bbox(mask_arr)
    report["mask_bbox_storage_order"] = {
        "min_voxel": list(mask_mins),
        "max_voxel_inclusive": list(mask_maxs_inclusive),
        "extent": [int(mask_maxs_inclusive[i] - mask_mins[i] + 1) for i in range(3)],
    }

    # CSV bbox endpoints
    csv_mins, csv_maxs = csv_bbox_endpoints(csv_row)
    report["csv_bbox"] = {
        "center": [csv_row.c_x, csv_row.c_y, csv_row.c_z],
        "size": [csv_row.len_x, csv_row.len_y, csv_row.len_z],
        "min_xyz": list(csv_mins),
        "max_xyz": list(csv_maxs),
    }

    # Permutation search
    perm_results = try_axis_permutations(csv_mins, csv_maxs, mask_mins, mask_maxs_inclusive)
    report["axis_permutation_best_matches"] = perm_results[:3]

    return report


def summarize_axis_mapping(reports: list[dict]) -> dict:
    """Across cases, check whether the best axis permutation and inclusivity are consistent."""
    best = [r["axis_permutation_best_matches"][0] for r in reports]
    perms = [tuple(b["csv_axis_to_storage_axis"]) for b in best]
    excls = [b["csv_max_is_exclusive"] for b in best]
    residuals = [b["total_residual"] for b in best]
    return {
        "n_cases": len(reports),
        "perm_unique": list({p for p in perms}),
        "perm_consistent_across_cases": len({p for p in perms}) == 1,
        "exclusivity_unique": list({e for e in excls}),
        "exclusivity_consistent_across_cases": len({e for e in excls}) == 1,
        "max_residual_voxels_across_cases": max(residuals),
        "mean_residual_voxels_across_cases": sum(residuals) / len(residuals),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify local TDSC-ABUS-2023 conventions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ncases", type=int, default=5, help="Number of cases to inspect.")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs/local_data_check/verification_report.json"
    )
    args = parser.parse_args()

    bbx_csv = args.dataset / "bbx_labels.csv"
    labels_csv = args.dataset / "labels.csv"
    assert bbx_csv.exists(), f"missing {bbx_csv}"
    assert labels_csv.exists(), f"missing {labels_csv}"

    bbx = pd.read_csv(bbx_csv)
    labels = pd.read_csv(labels_csv)

    # Sample first N cases
    case_ids = labels.case_id.head(args.ncases).tolist()

    reports = []
    for cid in case_ids:
        row = bbx[bbx.id == cid].iloc[0]
        rep = inspect_case(int(cid), args.dataset, row)
        # Attach label
        rep["label_BM"] = labels[labels.case_id == cid].iloc[0].label
        reports.append(rep)

    summary = summarize_axis_mapping(reports)

    # Print a concise human summary
    print("\n=== Per-case summary ===")
    for r in reports:
        d = r["data"]
        m = r["mask"]
        best = r["axis_permutation_best_matches"][0]
        print(f"\ncase {r['case_id']} ({r['label_BM']})")
        print(f"  data shape={d['shape']} dtype={d['dtype']} intensity=[{d['intensity_min']}, {d['intensity_max']}]")
        print(f"  data space={d['header_space']} kinds={d['header_kinds']}")
        if d["header_space_directions"] is not None:
            sd = np.array(d["header_space_directions"])
            print(f"  data space_directions (mm/voxel diag) = {np.diag(sd).tolist()}")
            print(f"  data space_directions matrix:\n{sd}")
        print(f"  data origin={d['header_space_origin']}")
        print(f"  mask shape={m['shape']} unique={m['unique_values']} fg_voxels={m['foreground_voxels']}")
        print(f"  mask tight bbox: storage min={r['mask_bbox_storage_order']['min_voxel']} max_incl={r['mask_bbox_storage_order']['max_voxel_inclusive']} extent={r['mask_bbox_storage_order']['extent']}")
        print(f"  csv center={r['csv_bbox']['center']} size={r['csv_bbox']['size']}")
        print(f"  csv min={r['csv_bbox']['min_xyz']} max={r['csv_bbox']['max_xyz']}")
        print(f"  BEST PERM: csv_axis_to_storage_axis={best['csv_axis_to_storage_axis']} max_exclusive={best['csv_max_is_exclusive']} residual_total_vox={best['total_residual']}")

    print("\n=== Across-case summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "reports": reports}
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote full JSON report to {args.out}")


if __name__ == "__main__":
    main()
