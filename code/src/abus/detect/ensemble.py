"""Ensemble combination for the val/test candidate generation path (STORY_01_02).

Implements the union-then-WBC ensemble of the five fold detectors:
  1. Union: collect all raw proposals from all five detectors for a volume.
  2. WBC (Weighted Box Clustering): greedily cluster overlapping proposals.
  3. Per-cluster averaging: score = mean of contributing detectors' scores;
     embedding = mean of contributing detectors' embeddings; source_detectors
     = tuple of fold ids that contributed to the cluster.

This is the ensemble *combination* step described in thesis §3.2.1.
The WBC IoU threshold used here is provisional (0.5 default); the project-wide
post-processing WBC parameters are calibrated in STORY_01_03 and the val/test
ensemble combination is re-run there with the calibrated params so the val/test
candidate set uses the same frozen WBC as everything else.

No training happens here — only post-hoc combination of five frozen checkpoint
inference outputs.

Dependencies: abus.geometry.bbox (BBox, iou_3d), numpy.
No GPU, no nnDetection, no PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abus.geometry.bbox import BBox, iou_3d

if TYPE_CHECKING:
    from abus.detect.candidates import RawCandidate


def ensemble_combine(
    proposals: list[RawCandidate],
    iou_threshold: float = 0.5,
) -> list[RawCandidate]:
    """Combine multi-detector proposals for a single volume via union + WBC.

    Algorithm (Weighted Box Clustering — greedy, score-sorted):
      1. Sort proposals by score descending.
      2. For each unassigned proposal (cluster seed), create a new cluster.
      3. Assign all remaining unassigned proposals whose bbox IoU with the seed
         box exceeds ``iou_threshold`` to the same cluster.
      4. For each cluster:
         - score       = mean of all member scores
         - embedding   = mean of all member embeddings (float32)
         - bbox        = bbox of the highest-scoring member (seed box)
         - case_id     = case_id of the first member (all members share one volume)
         - split       = split of the first member
         - source_detectors = sorted tuple of distinct fold ids from all members

    Parameters
    ----------
    proposals:
        Raw proposals for ONE volume, from one or more fold detectors.
        All proposals must have the same ``case_id`` and ``split``.
    iou_threshold:
        IoU threshold for cluster membership. Proposals with IoU > threshold
        with the cluster seed are absorbed. Default 0.5 (provisional; superseded
        by calibrated WBCParams from STORY_01_03).

    Returns
    -------
    list[RawCandidate]
        Combined candidates, one per cluster, sorted by score descending.
        Each candidate's ``source_detectors`` lists all contributing fold ids.

    Raises
    ------
    ValueError
        If proposals is empty.
    """
    if not proposals:
        return []

    # Lazy import to avoid circular dependency at module load time.
    from abus.detect.candidates import RawCandidate

    # Sort by score descending (greedy WBC: highest-score proposal is cluster seed).
    sorted_props = sorted(proposals, key=lambda p: p.score, reverse=True)
    assigned = [False] * len(sorted_props)
    clusters: list[list[RawCandidate]] = []

    for i, seed in enumerate(sorted_props):
        if assigned[i]:
            continue
        cluster = [seed]
        assigned[i] = True
        seed_bbox = seed.bbox
        for j in range(i + 1, len(sorted_props)):
            if assigned[j]:
                continue
            # Strict > : proposals at exactly the threshold stay in separate clusters
            # (matches standard NMS convention; calibrated threshold in STORY_01_03).
            if iou_3d(seed_bbox, sorted_props[j].bbox) > iou_threshold:
                cluster.append(sorted_props[j])
                assigned[j] = True
        clusters.append(cluster)

    combined: list[RawCandidate] = []
    for cluster in clusters:
        mean_score = float(np.mean([c.score for c in cluster]))
        mean_emb = np.mean(np.stack([c.embedding for c in cluster], axis=0), axis=0).astype(
            np.float32
        )
        # Bbox: use the seed (highest-score) box
        seed_box: BBox = cluster[0].bbox
        case_id: int = cluster[0].case_id
        split: str = cluster[0].split
        # Collect all contributing fold ids (may include duplicates if same fold scored
        # multiple proposals in the cluster; keep unique, sorted).
        fold_ids: tuple[int, ...] = tuple(
            sorted({fd for c in cluster for fd in c.source_detectors})
        )
        combined.append(
            RawCandidate(
                case_id=case_id,
                split=split,
                bbox=seed_box,
                score=mean_score,
                embedding=mean_emb,
                source_detectors=fold_ids,
            )
        )

    # Return sorted by score descending
    return sorted(combined, key=lambda c: c.score, reverse=True)
