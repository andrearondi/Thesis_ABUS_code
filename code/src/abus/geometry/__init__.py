"""abus.geometry — Bounding-box utilities and coordinate conversions (STORY_00_03).

Public API:
  BBox                — storage-axis order, voxel units, inclusive-max (bbox.py)
  volume, extent, center, iou_3d, shape_stats, clip, contains  (bbox.py)
  csv_itk_to_bbox, bbox_to_csv_itk  — ITK CSV <-> BBox  (convert.py)
  bbox_to_nndet, nndet_to_bbox       — nnDetection <-> BBox  (convert.py)
  voxel_to_mm, mm_to_voxel           — physical-frame helpers  (convert.py)
  bbox_center_mm                     — node coordinate feature  (convert.py)
"""

from abus.geometry.bbox import (
    BBox,
    center,
    clip,
    contains,
    extent,
    iou_3d,
    shape_stats,
    volume,
)
from abus.geometry.convert import (
    bbox_center_mm,
    bbox_to_csv_itk,
    bbox_to_nndet,
    csv_itk_to_bbox,
    mm_to_voxel,
    nndet_to_bbox,
    voxel_to_mm,
)

__all__ = [
    # bbox
    "BBox",
    "center",
    "clip",
    "contains",
    "extent",
    "iou_3d",
    "shape_stats",
    "volume",
    # convert
    "bbox_center_mm",
    "bbox_to_csv_itk",
    "bbox_to_nndet",
    "csv_itk_to_bbox",
    "mm_to_voxel",
    "nndet_to_bbox",
    "voxel_to_mm",
]
