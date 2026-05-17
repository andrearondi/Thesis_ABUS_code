"""Ground-truth bounding box reader for TDSC-ABUS-2023 (STORY_00_03).

Reads ``bbx_labels.csv`` and applies the ITK->storage-axis conversion at
the I/O boundary so that no downstream caller ever handles ITK-ordered
coordinates.

CSV schema (per docs/local_data.md):
  id, c_x, c_y, c_z, len_x, len_y, len_z

Verified convention (docs/local_data_check/verification_report.json):
  - Axis order: ITK (x,y,z) — apply (2,1,0) permutation for storage order.
  - Endpoint: max is INCLUSIVE.
  - Round-trip residual: 0.0 voxels across all 6 sampled cases.
"""

from __future__ import annotations

import pandas as pd

from abus.geometry.bbox import BBox
from abus.geometry.convert import csv_itk_to_bbox


def load_gt_bboxes(csv_path: str) -> dict[int, BBox]:
    """Read bbx_labels.csv and return {case_id: BBox}.

    The ITK->storage-axis conversion is applied at this I/O boundary via
    ``csv_itk_to_bbox``.  Callers receive storage-axis-order, inclusive-max
    ``BBox`` objects — no ITK-ordered coordinates leak past this function.

    Parameters
    ----------
    csv_path:
        Path to ``bbx_labels.csv``.  Expected columns:
        ``id, c_x, c_y, c_z, len_x, len_y, len_z``.

    Returns
    -------
    dict[int, BBox]
        Maps integer case ID to the corresponding ground-truth BBox.

    Raises
    ------
    ValueError
        If required columns are missing from the CSV.
    AssertionError
        If a CSV endpoint is not an integer voxel (wrong convention guard
        in ``csv_itk_to_bbox``).
    """
    df = pd.read_csv(csv_path)
    required = {"id", "c_x", "c_y", "c_z", "len_x", "len_y", "len_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"bbx_labels.csv is missing required columns: {missing}. " f"Found: {list(df.columns)}"
        )

    bboxes: dict[int, BBox] = {}
    # Use itertuples (not iterrows) to avoid dtype-casting issues: iterrows can silently
    # convert int columns to float when any value is NaN, corrupting case_id.
    for row in df.itertuples(index=False):
        case_id = int(row.id)
        c_xyz = (float(row.c_x), float(row.c_y), float(row.c_z))
        len_xyz = (float(row.len_x), float(row.len_y), float(row.len_z))
        bboxes[case_id] = csv_itk_to_bbox(c_xyz, len_xyz)

    return bboxes
