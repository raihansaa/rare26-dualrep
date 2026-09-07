# RARE26 — Barrett's neoplasia detection in a low-prevalence setting

Submission code for the RARE26 challenge: binary classification of endoscopy frames as
Barrett's neoplasia versus non-dysplastic Barrett's oesophagus (NDBE), scored by **positive
predictive value at 90% recall under a simulated 100:1 negative:positive prior**:

```
PPV@90R = 0.90 / (0.90 + 100 * FPR@90R)
```

The score is a strictly monotone function of one number — the false-positive rate at the
threshold achieving 90% recall — so it is decided entirely by where the bottom-decile positive
falls in the negative score distribution. Chance is 0.0099.

## The system

Two independent backbone arms, deliberately chosen to fail on different images:

| arm | backbone | adaptation | resolution |
|---|---|---|---|
| A | DINOv2 ViT-B/14 (reg4), self-supervised on GastroNet-5M | LoRA r16/α32 on `attn.qkv` + `attn.proj` of the last 6 of 12 blocks, plus a linear head | 336 |
| B | ResNet50, SWSL + GastroNet-5M | `layer4` + `fc` trainable, trunk frozen in `eval()` | 384 |

Five folds per arm. At inference each checkpoint is affine-calibrated on its own out-of-fold
predictions, the calibrated probabilities are averaged within each arm, and the two arm means
are averaged with equal weight. No prior shift and no noisy-OR: both were shipped in an earlier
submission and removed on 2026-08-26 after they failed to transfer externally (`EXPERIMENT_LOG.md` §3).

Why two arms rather than one backbone in several recipes: members that share a representation
fail on the same images. These two are Spearman 0.48–0.59 apart and share only ~50% of their
bottom-decile positives — the images that set the threshold. On leave-one-centre-out the fusion
is positive on both centres and all three metrics, which no single-backbone ensemble achieved.

### The submitted container

The Closed Testing Phase submission is **DualRep**, built from this source:

| | |
|---|---|
| image | `sha256:6571524e410b3e59a0e7af1b66b433495c2f7714c71439def0f74a8af88d376c` |
| tag | `rare26-twoarm-probavg` |
| archive | `twoarm_10model_probavg.tar.gz`, 6,357,006,153 bytes |

Verified under `--network none --gpus all`: both arms load their five checkpoints, calibration
executes, and the fixture scores carry no ties and no saturation. Checkpoints are not
redistributed here; `docs/MODEL_PROVENANCE.md` records the sha256 of every one of them.

## Results

Leave-one-centre-out, train on the other centres, test on a centre no model has seen. Paired
bootstrap clustered on near-duplicate groups.

| held-out centre | AUROC | AUPRC | FPR@90R |
|---|---|---|---|
| center_1 (2279 images, 61 positives, 2.68% prevalence) | 0.9870 | 0.9012 | 0.0140 |
| center_2 (816 images, 97 positives, 11.9% prevalence) | 0.9930 | 0.9754 | 0.0014 |

**Local numbers are a different regime from the leaderboard, not merely optimistic.** The same
family of model measured FPR@90R 17–100× worse on the challenge's held-out data than on a
held-out centre here. Treat the table above as valid for *ranking two candidates against each
other* and invalid as an absolute expectation. `EXPERIMENT_LOG.md` §1 documents the calibration.

## Reproduction

Requires Python 3.11+, a CUDA GPU, and the challenge data. Install with
`pip install -r requirements.txt`.

### 1. Data

Place the challenge training release under `RARE25-train-data/<centre>/<ndbe|neo>/<image_id>.png`,
and its accompanying inventory at `data_inventory_rare25.csv` — that file ships with the
challenge data and is not generated here. Fold grouping reads its `duplicate_group` column.

The third centre (`evc`, the EndoVis 2015 Barrett's set) is optional and is excluded from every
headline metric — both stock and domain models score ~1.0 AUROC on it without training on it,
so it cannot discriminate between candidates. If you want it, `data_split_scripts/ingest_evc.py`
builds `evc_inventory.csv` from that release.

Neither the data nor the pretrained weights are redistributed here.

### 2. Pretrained backbones

Download from [Theta Vision Cortex](https://cortex.thetavision.nl/dataset-provider/listing/2/)
into the project root. Verify checksums against `docs/MODEL_PROVENANCE.md` before use — the
loaders assert matched key counts, because `strict=False` silently yields a random backbone:

- `dinov2.pth` — GastroNet-5M DINOv2 ViT-B/14 reg4
- `RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth`

### 3. Folds

```bash
python data_split_scripts/make_folds.py --out folds_v1.csv
python data_split_scripts/build_combined.py --folds folds_v1.csv --out folds_v2.csv
```

`folds_v2.csv` is the split every shipped model uses. Grouping comes from the challenge's own
`duplicate_group` field; `make_folds.py` asserts no group spans two folds.

### 4. Train the shipped arms

```bash
for k in 0 1 2 3 4; do
  python training/train_dinov2.py --folds folds_v2.csv --fold $k \
      --init dinov2.pth --out runs/V2_dinogn_fold$k
  python training/train_resnet.py --folds folds_v2.csv --fold $k --arch resnet \
      --init 'RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth' \
      --freeze-until layer3 --out runs/V7_swsl_fold$k
done
```

All other hyperparameters are defaults and are recorded in each checkpoint's `args`:
336/384 px, batch 32, 30 epochs, lr 5e-4 (DINOv2) / 1e-4 (ResNet), weight decay 1e-4,
P0 preprocessing, no augmentation, seed 42, checkpoint selected on inner-split AUPRC.

**Check GPU memory before launching on new hardware.** An oversized batch does not OOM on
Windows — it spills to system RAM over PCIe and runs up to 32× slower in silence. See
`EXPERIMENT_LOG.md` §5.

### 5. Evaluate

```bash
# leave-one-centre-out, the protocol every decision in this project was made on
python training/train_cross_centre.py --folds folds_v2.csv --arch dinov2 \
    --holdout center_1 --init dinov2.pth --tag mytag

# paired A/B of two LOCO runs on the same held-out centre
python evaluation/compare_loco.py \
    --baseline runs/LOCO_dinov2_rest_to_center_1_domain \
    --candidate runs/LOCO_dinov2_rest_to_center_1_mytag --folds folds_v2.csv
```

Do **not** select on pooled or per-fold cross-validation. It has produced four documented
wrong conclusions on this project; `EXPERIMENT_LOG.md` §4 lists them.

### 6. Container

```bash
# refit the per-checkpoint calibration from the OOF predictions written by step 4
python evaluation/fit_platt.py     --arm dinov2=runs/V2_dinogn_fold{k}     --arm swsl=runs/V7_swsl_fold{k}     --out submission_template/platt.json

cd submission_template
# stage resources/dinov2_fold*.pt and resources/swsl_fold*.pt from the runs above
docker build -t rare26 .
docker save rare26 | gzip -c > rare26.tar.gz
```

`platt.json` holds `[a, b, prior_fit]` per checkpoint, frozen at build time. The copy committed
here is the one the submitted container shipped; `fit_platt.py` regenerates it exactly from the
out-of-fold predictions. `inference.py` raises if any checkpoint is missing its entry, so a
forgotten copy fails loudly rather than shipping uncalibrated scores.

## Repository layout

| path | contents |
|---|---|
| `preprocessing/` | preprocessing variants and augmentation pipelines |
| `training/` | fold trainers and the leave-one-centre-out trainer |
| `evaluation/` | paired A/B testing, feature banks, frozen probes, duplicate audit, shift stress |
| `data_split_scripts/` | fold construction |
| `submission_template/` | the container |
| `docs/MODEL_PROVENANCE.md` | every weight file's source, access date, checksum and licence status |
| `EXPERIMENT_LOG.md` | complete experimental record — every lever tested, with numbers |

## Experimental record

`EXPERIMENT_LOG.md` is the honest account: **twenty-seven levers tested, three worked.** It
records what failed and why, the protocol mistakes that produced wrong conclusions, and the
traps that cost time. It is written to be useful to someone continuing the work, including the
negative results — which are most of it.

## Licence and third-party assets

Code is MIT (see `LICENSE`). The licence covers this repository's source only. Pretrained
weights and challenge data are third-party, are not redistributed here, and remain subject to
their own terms — see `docs/MODEL_PROVENANCE.md`.

One carve-out inside that: `submission_template/` began as a clone of the organizers' own
submission template ([TUE-ARIA/RARE25-Submission](https://github.com/TUE-ARIA/RARE25-Submission)),
which is distributed under **CC BY-NC 4.0** and keeps its own `LICENSE` in that directory. The
MIT grant covers our additions and modifications there — the two-arm ensemble inference path,
`platt.json`, `_preflight.py`, and the Dockerfile and build-script changes — not the upstream
template itself. Checkpoints are excluded from the repository; stage them as described under
*Reproduction → Container* above.
