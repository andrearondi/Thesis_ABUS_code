#!/usr/bin/env python
"""STORY_01_02 Step 13 server-side embedding audit (D01.14).

Runs predict_with_embeddings on ONE training case and checks:
  [1] embeddings.shape == (N, 128)        — D_EMB pinned, hook captured right tensor
  [2] embeddings non-zero                 — dead hook returns zeros
  [3] embeddings non-constant across N     — constant/broadcast hook capture
  [4] box axis (x1,y1,x2,y2,z1,z2) max>=min
  [5] scores.shape == (N,)

This is the runbook Step 13 heredoc turned into a committed script so it can be
run as `python scripts/audit_embeddings_step13.py` with NO shell heredoc / paste.
GPU required. Must pass before Steps 14/16. Paths match the runbook exactly.
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from abus.detect.nndet_inference import D_EMB, predict_with_embeddings  # noqa: E402

FOLD_DIR = "/home/maia-user/nndet_data/Task001_TDSCABUS/RetinaUNetV001_D3V001_3d/fold0"
# D01.14b: a TRAINING case (0..19 are fold-0's OOF cases, in imagesTr). Val/test
# cases (0100+) live in imagesTs, which does not exist until Step 14a.
PREPROC_DIR = "/home/maia-user/nndet_data/Task001_TDSCABUS/preprocessed/D3V001_3d/imagesTr"
AUDIT_CASE_ID = 5


def main() -> int:
    print(f"Auditing case_id={AUDIT_CASE_ID} (training/imagesTr) with fold-0 model ...")
    raw = predict_with_embeddings(
        fold=0,
        case_ids=[AUDIT_CASE_ID],
        preprocessed_dir=PREPROC_DIR,
        fold_dir=FOLD_DIR,
    )

    if AUDIT_CASE_ID not in raw:
        print(
            f"case_id={AUDIT_CASE_ID} not in result dict. If N=0 or file missing, "
            "edit AUDIT_CASE_ID to another training case (0..99) from imagesTr."
        )
        return 1

    rd = raw[AUDIT_CASE_ID]
    n = rd.boxes.shape[0]
    print(f"\n--- Audit results for case {AUDIT_CASE_ID} ---")
    print(f"  N detections:      {n}")
    print(f"  boxes.shape:       {rd.boxes.shape}")
    print(f"  scores.shape:      {rd.scores.shape}")
    print(f"  embeddings.shape:  {rd.embeddings.shape}")

    shape_ok = rd.embeddings.shape == (n, D_EMB)
    print(f"\n  [1] embeddings.shape == ({n}, {D_EMB}): {'PASS' if shape_ok else 'FAIL — STOP'}")
    if not shape_ok:
        print(f"      Got: {rd.embeddings.shape}")

    if n == 0:
        print("  WARNING: N=0 detections for this case — try a different AUDIT_CASE_ID")
        print("\n  Overall: CHECKS FAILED — STOP before full run")
        return 1

    row_norms = np.linalg.norm(rd.embeddings, axis=1)
    nonzero_ok = bool(np.any(row_norms > 1e-6))
    v2 = "PASS" if nonzero_ok else "FAIL — dead hook?"
    rn = f"min={row_norms.min():.4f} max={row_norms.max():.4f} mean={row_norms.mean():.4f}"
    print(f"  [2] embeddings non-zero (any row_norm > 1e-6): {v2}")
    print(f"      row_norm: {rn}")

    if n > 1:
        pdv = rd.embeddings.var(axis=0)
        nonconstant_ok = bool(np.any(pdv > 1e-12))
        v3 = "PASS" if nonconstant_ok else "FAIL — constant embedding?"
        pv = f"min={pdv.min():.6f} max={pdv.max():.6f} mean={pdv.mean():.6f}"
        print(f"  [3] embeddings non-constant across N (any dim var > 1e-12): {v3}")
        print(f"      per_dim_var: {pv}")
    else:
        print("  [3] N=1 — cannot check cross-N variance; check row_norm above instead")
        nonconstant_ok = True

    x1, y1, x2, y2, z1, z2 = rd.boxes[0].tolist()
    axis_ok = x2 >= x1 and y2 >= y1 and z2 >= z1
    box_row = [round(float(v), 2) for v in rd.boxes[0].tolist()]
    print(f"  [4] first box row: {box_row}")
    print(f"      Axis (x1,y1,x2,y2,z1,z2): max>=min: {'PASS' if axis_ok else 'FAIL — STOP'}")

    scores_ok = rd.scores.shape == (n,)
    print(f"  [5] scores.shape == ({n},): {'PASS' if scores_ok else 'FAIL — STOP'}")

    all_pass = shape_ok and nonzero_ok and nonconstant_ok and axis_ok and scores_ok
    if all_pass:
        verdict = "ALL CHECKS PASS — proceed to Step 14"
    else:
        verdict = "CHECKS FAILED — STOP before full run"
    print(f"\n  Overall: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
