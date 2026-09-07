# Model provenance and archive record

Required by the challenge submission rules: every pretrained weight used, where it came
from, when it was obtained, its checksum, and its licence status. Nothing in this file is
inferred. Checksums were recomputed from the files on disk on 2026-08-19 and match the
`checkpoint_hash` values independently recorded in each run's out-of-fold prediction CSV.

**No weight file listed here is redistributed in this repository.** Each must be obtained
from its original source under that source's own terms.

## Pretrained backbones (third-party, not redistributed)

| file | size | sha256 | source | obtained |
|---|---|---|---|---|
| `dinov2.pth` | 436.5 MB | `20ab749ec4bfdd53a79fc82784d04d9480d87d9f122b877d0f20c6f40aabdc44` | Theta Vision Cortex | 2026-08-03 |
| `RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth` | 94.4 MB | `5688929fea4437031604001495fb77fb18cdc2ff92ae24120f93aeceaf5aa16d` | Theta Vision Cortex | 2026-08-03 |
| `RN50_GastroNet-5M_DINOv1.pth` | 94.4 MB | `5ae8e09cea634179e8caf95dbead1a86d7ce05d1e934c80c742b9ff5887f44dd` | Theta Vision Cortex | 2026-08-03 |
| `RN50_GastroNet-5M_MOCOv2.pth` | 94.4 MB | `695b892942b8862c03bf80d00da9934eeb7040511a27758c7adc93052db8f228` | Theta Vision Cortex | 2026-08-17 |
| `RN50_GastroNet-5M_SIMCLRv2.pth` | 94.4 MB | `8d520d19ea72ca3fcb41cf737fc6f9842d5c25c9d1e17f0e7936de8ba8fd2aac` | Theta Vision Cortex | 2026-08-17 |
| `RN50_GastroNet-1M_DINOv1.pth` | 94.4 MB | `ced56c7b86f76b267c46220d17c30615f92c9a8e1202e94f2e98f09bc73d0ea1` | Theta Vision Cortex | 2026-08-17 |
| `RN50_GastroNet-200K_DINOv1.pth` | 94.4 MB | `2dbf41d5123202c151467c361112ee4da8c18f96cf3e3ee6a7f09cddab1094ff` | Theta Vision Cortex | 2026-08-17 |
| `VITS_GastroNet-5M_DINOv1.pth` | 86.7 MB | `383c7a9a026cc7a8ecf0a201713714f91f51384fcacb0a7226df77212aa7da77` | Theta Vision Cortex | 2026-08-17 |

Source URL: <https://cortex.thetavision.nl/dataset-provider/listing/2/>

Only the first two are used by the shipped system. The remainder were screened and rejected
(see `EXPERIMENT_LOG.md` §3) and are listed because they were downloaded and evaluated.

**`dinov2.pth` identification.** A DINOv2 ViT-B/14 *register* variant (4 register tokens),
self-supervised on GastroNet-5M (~4.8M endoscopy images, 8 Dutch hospitals, 2012–2020). It is
not Meta's stock release: 0 of 173 tensors match stock reg4, and `blocks.11.mlp.fc2` has
cosine similarity 0.057 against it, so early layers sit close to stock and late layers are rewritten. The
HuggingFace repository `tgwboers/GastroNet-5M_Pretrained_Weights` contains only a README; the
weights come from the Theta portal above.

**Licensing status.** No separate licence text is published at either distribution point, and
this is recorded here as a finding rather than an omission. The Theta Vision Cortex listing page
does not expose licence terms in retrievable form (checked 2026-09-07), and the HuggingFace
repository `tgwboers/GastroNet-5M_Pretrained_Weights` carries no licence tag, containing only
`.gitattributes` and a README (checked against the HuggingFace API, last modified January 2026).

The weights are released publicly by their authors for scientific use, and that is the basis on
which they are used here. The RARE26 rules permit publicly available pretrained models and
prohibit only private external data, so this use is compliant. No terms have been agreed to
beyond public availability, and nothing here is redistributed.

Attribution is given to the originating work:

- *GastroNet-5M: a multicenter dataset for developing foundation models in gastrointestinal
  endoscopy*, Gastroenterology, 2025. PMID 40749857.
  <https://pubmed.ncbi.nlm.nih.gov/40749857/>
- *Foundation models in gastrointestinal endoscopic AI: impact of architecture, pre-training
  approach and data efficiency*, Medical Image Analysis, 2024.
  <https://www.sciencedirect.com/science/article/pii/S1361841524002238>

Anyone reusing these weights should obtain them from the Theta portal directly and satisfy
themselves as to the terms, which are not stated at the point of download.

## Trained checkpoints shipped in the container

Produced by this repository's code from the challenge training data. Staged at
`submission_template/resources/`.

| file | source run | size | sha256 |
|---|---|---|---|
| `dinov2_fold0.pt` | `V2_dinogn_fold0` | 345.7 MB | `227c1e5bc812f45a85dbd514d1e6985da72dd41cb26b99c83da59c0eff7af228` |
| `dinov2_fold1.pt` | `V2_dinogn_fold1` | 345.7 MB | `626ede068a223366480e7afd17af6a7d405764c8fa39a68036383514081e6426` |
| `dinov2_fold2.pt` | `V2_dinogn_fold2` | 345.7 MB | `ab866bd247db4876d8a818618971d878c773e186b9cfee64d13449a4f17a2b9d` |
| `dinov2_fold3.pt` | `V2_dinogn_fold3` | 345.7 MB | `bd7c44ba502593496d01e799f8a983cb45d45b451c36fba13c15623feb357877` |
| `dinov2_fold4.pt` | `V2_dinogn_fold4` | 345.7 MB | `abbb03b5de6707189035b17ed4e2507c50a85b08227a77a530c3b3619e9e60a8` |
| `swsl_fold0.pt` | `V7_swsl_fold0` | 94.3 MB | `3376ff96ba7ef64e0714c94762fa2af1988151e8b2de3f519e01d5cd8ddbccc8` |
| `swsl_fold1.pt` | `V7_swsl_fold1` | 94.3 MB | `ffcec8171c80722c1e9f0cb355df3f8611fda98142a7185e3bbd7e5d6fef2914` |
| `swsl_fold2.pt` | `V7_swsl_fold2` | 94.3 MB | `70bc2f40e74d368ed5bb00c80b6befa6518e2c15bfabffaf7f440ad049ed6329` |
| `swsl_fold3.pt` | `V7_swsl_fold3` | 94.3 MB | `3b618f3a2ca732ef438e6aa065f3da152337670803401d6774fa8e5870327fc2` |
| `swsl_fold4.pt` | `V7_swsl_fold4` | 94.3 MB | `d7cc0f5879bf12752fbed3352cff4fc2d6985b97cd6c5d2d00e739c818722222` |

## Container image

The submitted container, built 2026-08-26 and frozen. It was never pushed to a registry, so the
identifier below is the local image ID rather than a registry digest.

| | |
|---|---|
| image ID | `sha256:6571524e410b3e59a0e7af1b66b433495c2f7714c71439def0f74a8af88d376c` |
| tag | `rare26-twoarm-probavg` |
| archive | `twoarm_10model_probavg.tar.gz`, 6,357,006,153 bytes |
| base image | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |

Verified under `--network none --gpus all`: both arms load their five checkpoints, per-checkpoint
Platt calibration executes, and the fixture frames score across the full range with no ties and
no saturation. All state dicts load with `strict=True`.

## Data

- **EDD2020** (Endoscopy Disease Detection and Segmentation, EndoCV2020). 386 frames from 5
  institutions with per-class segmentation masks. Obtained 2026-08-19 from the public Kaggle
  mirror `orvile/edd2020-endoscopy-detection-and-segmentation`; original challenge site
  <https://edd2020.grand-challenge.org>. **Licence: CC BY-NC-SA 4.0, meaning attribution,
  NON-COMMERCIAL, share-alike.** Cite Ali et al., *Deep learning for detection and segmentation
  of artefact and disease instances in gastrointestinal endoscopy*, Medical Image Analysis, 2021
  (doi:10.1016/j.media.2021.102002) and Ali et al., *Endoscopy disease detection challenge 2020*,
  arXiv:2003.03376. Not redistributed.
  Used only to train the segmentation decoder in §3's localization experiment, **which measured
  negative and is not part of any shipped system.** If that ever changes, the non-commercial and
  share-alike clauses need checking against the challenge terms first.
  Checked for overlap against all 3,195 RARE images by pHash before use: zero matches at
  Hamming ≤ 4, closest 6.
- **RARE25 training release.** 3,095 images across two centres (`center_1`, `center_2`).
  Not redistributed.
- **EndoVis 2015 Barrett's set** (`evc`). 100 images (50 neoplasia / 50 non-dysplastic) from 39
  patients. **Included in the final five-fold training pool (`folds_v2.csv`) and excluded from
  all reported evaluation metrics.** Every shipped checkpoint therefore trains on these images.
  They are excluded from evaluation because the set is degenerate for it: both stock and
  domain-pretrained models reach roughly 1.0 AUROC on it without ever training on it, so it
  cannot discriminate between candidates. Not redistributed.

No private or non-public external data was used at any point.
