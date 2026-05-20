# nnDetection planning notes — STORY_01_01

Records the exact nnDetection version, the CLI invocations, and the documented
nnDetection bbox/spacing convention with source citations. This file satisfies
ASC-01_01.6 (version + commit hash) and ASC-01_01.4 (convention documentation).

---

## nnDetection version (filled in during /runbook 01.1 execution)

| Field | Value |
|---|---|
| nnDetection version | Name: nndet
Version: 0.1
Summary: 
Home-page: 
Author: Division of Medical Image Computing, German Cancer Research Center
Author-email: 
License: 
Location: /home/maia-user/Andre/nnDetection_src
Editable project location: /home/maia-user/Andre/nnDetection_src
Requires: batchgenerators, dicom2nifti, GitPython, hydra-core, loguru, matplotlib, medpy, mlflow, nevergrad, nnunet, pandas, python-gdcm, pytorch_lightning, PyYAML, scikit-image, scikit-learn, scipy, seaborn, SimpleITK, torchmetrics, tqdm
Required-by: 
 |
| nnDetection commit hash | 97a58f3110b71caf1b4bcc1851e67cf11e987fc5 |
| Install date | 18th May 2026 |
| conda env | `/home/maia-user/Andre/envs/thesis` |

**This must be filled in before STORY_01_02 trains anything.**
A blank entry here means the detector training is not reproducible (agent_rules §12).

---

## nnDetection CLI invocations

These are the exact commands the runbook runs on the server:

```bash
# Job 1 — dataset build (CPU-bound, ~10–30 min)
python scripts/build_nndet_dataset.py \
  --tdsc-root /home/maia-user/Andre/data \
  --out-root  /home/maia-user/Andre/outputs/nndet \
  --config    configs/detect/nndet_dataset.yaml

# Job 2 — nnDetection planning/preprocessing
nndet_prep \
  Task001 \
  /home/maia-user/Andre/outputs/nndet \
  --num_processes 8

nndet_unpack \
  Task001 \
  /home/maia-user/Andre/outputs/nndet \
  --num_processes 8

# Job 3 — verification
python scripts/verify_nndet_dataset.py \
  --task-dir /home/maia-user/Andre/outputs/nndet/Task001_TDSCABUS \
  --case-id  <any_train_case_id>
```

---

## nnDetection bbox/spacing convention

Source: MIC-DKFZ/nnDetection, tag v0.1 (latest stable as of project start).
Citation key: `Baumgartner2021nnDetection`.

### Box format

nnDetection represents 3D boxes as a 6-tuple:

    (y1, x1, y2, x2, z1, z2)

where:
- `(y1, x1, z1)` is the **lower corner** (inclusive).
- `(y2, x2, z2)` is the **upper corner** (**exclusive**, numpy-style).

Upper bounds are derived from `np.where` outputs: `(min, max+1)` — the voxel at
index `max` is inside the box, so the exclusive upper bound is `max + 1`.

### Storage-axis mapping

| nnDetection axis | Project storage axis |
|---|---|
| y | d0 (NRRD axis 0, CSV `_z`) |
| x | d1 (NRRD axis 1, CSV `_y`) |
| z | d2 (NRRD axis 2, CSV `_x`) |

This mapping is documented in `abus.geometry.convert` (D00.4, EPIC_00) and
implemented in `bbox_to_nndet` / `nndet_to_bbox`.

### Resampled-grid mapping

nnDetection resamples volumes to a self-configured target spacing. The target
spacing is determined by nnDetection's fingerprinting step (`nndet_prep`) and is
**not** overridden by this project (thesis §3.2.3/§3.2.4).

The resampled-grid coordinate of an original-grid voxel coordinate `v_orig` is:

    v_resamp = v_orig × (orig_spacing / target_spacing)   [element-wise]

where `orig_spacing = CANONICAL_SPACING_MM = (0.073, 0.200, 0.475674)` mm.

The inverse (resampled → original):

    v_orig_recovered = v_resamp × (target_spacing / orig_spacing)

The bbox round-trip residual on the original grid is bounded by the rounding
error of the resampling factor — in practice ≤ 1 voxel per axis. This is the
acceptance gate: **ASC-01_01.4** (measured on the server in the runbook).

### Code implementation

The resampled-grid mapping is implemented as executable, testable code in:

    src/abus/detect/nndet_convention.py :: bbox_original_roundtrip(b, target_spacing_mm)

The local unit test (`test_bbox_original_roundtrip_identity_spacing`) verifies
the identity case (target == original → exact round-trip). The server test runs
the real resampled grid and reports the actual residual.

---

## Fingerprint — expected output (paste server output here)

After `nndet_prep`, the nnDetection data fingerprint JSON should show:
- `"spacing_after_resampling"`: a target spacing derived from the dataset. With
  `CANONICAL_SPACING_MM = (0.073, 0.200, 0.475674)` mm written into the image
  files, the fingerprint must reflect values in the neighbourhood of these numbers
  (not `[1.0, 1.0, 1.0]` which would indicate the placeholder spacing was used).

**Paste the fingerprint JSON here after the runbook run:**

```
TBD
```

---

## Round-trip residual — expected result (paste server output here)

The `verify_nndet_dataset.py` script reports the bbox round-trip residual for a
named case. The acceptance criterion is ≤ 1 voxel per axis on the original grid.

**Paste the verification output here after the runbook run:**

```
TBD
```
