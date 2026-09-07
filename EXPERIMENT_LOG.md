# RARE26 — experiment log and handoff

State as of 2026-08-19. Read this before running anything.

---

## 1. Where the submission stands

**Submission 3 is uploaded and scored: PPV@90R 0.0151, rank 72.** The 15-model three-family
ensemble (§3). The score went *up* from submission 2's 0.0148; the rank fell ten places
because roughly ten new rows landed above us in a week. Nothing regressed — the field moved.
AUROC 0.820, AUPRC 0.3426, derived FPR@90R 58.7%. It ran genuinely (AUROC 0.816 → 0.8200
rules out a re-score of the same image), and changed almost nothing that mattered:
**in-distribution the ensemble cut FPR@90R 39%; on the leaderboard it cut 1.2 points.**

**Submission 4 is uploaded and scored: PPV@90R 0.0150, rank 104 (2026-08-25).** The two-arm
10-model fusion with per-model Platt, prior shift to 1:100 and noisy-OR — the system staged
since 2026-08-17, built and verified the same day. AUROC 0.8195, AUPRC 0.3187, derived
FPR@90R 59.1%, CI [0.0108, 0.0746].

Against submission 3 (0.0151 / 0.8200 / 0.3426 / 58.7%) this is **a null, not a regression**:
the PPV gap is ~4 false positives out of ~1,040 negatives, and AUROC is flat to 0.0005. Read
it as a null and nothing more.

**But the AUPRC moved the wrong way for our own story: −0.0239.** §3 predicted that a correct
tail intervention cuts FPR@90R *while* costing a little AUPRC. Here the AUPRC was paid and the
tail gain never arrived — the first external evidence that prior-shift + noisy-OR does not
transfer. It is confounded (ensemble and pooling both changed against submission 3), so it
does not convict the pooling on its own. **The clean test is one slot away and still worth
taking: the same ten checkpoints with prior shift off, plain probability averaging.**

**The plateau is now an external fact.** 0.0148 (Aug 10) → 0.0151 (Aug 17) → 0.0150 (Aug 25).
Two substantial changes, both scored clearly positive by LOCO, neither visible on the board.
This confirms §7's closure of the experimental search rather than merely arguing it, and it
extends the §1 calibration lesson: LOCO's *direction* was validated exactly once (submission
1 → 2) and has been null on every A/B since.

**Undocumented metric block, seen 2026-08-25.** The evaluation page also reports a second set
of scores labelled `Validation RARE25`: PPV@90R 0.0155 [0.0120, 0.1324], **AUROC 0.8889**
[0.7575, 0.9896], **AUPRC 0.5317** [0.2429, 0.7871] — far above the main set's 0.8195 / 0.3187.
No earlier entry in this log records it, so it is unknown whether prior submissions carry it.
**Retrieving that block from submission 3's evaluation page is free and is the highest-
information action currently available** — it converts a single read with a sevenfold CI into
a two-set comparison of the same A/B.

**Submission 2: PPV@90R 0.0148, rank 62** — up from 103rd. Five domain-pretrained
DINOv2-LoRA folds, logit-averaged, no ResNet arm, no fusion, no TTA, from
`runs/V2_dinogn_fold*/best.pt` (folds_v2, `init=dinov2.pth`).

Submission 1 placed 103rd with the **worst** configuration we later measured: stock
DINOv2 + 50/50 ResNet fusion. Its weights were deleted in the 2026-08-11 cleanup;
`runs/*/results.json` retains the record.

**Rank drift, read from the board 2026-08-25.** All three submissions, same board, same day:
0.0104 → **201st**, 0.0148 → **107th**, 0.0151 → **102nd**. Every score rose; every rank fell.
Two numbers worth keeping:

- **The entire 15-model ensemble is worth five places.** Submissions 2 and 3 sit 107th and
  102nd on one board — a 39% in-distribution FPR@90R cut bought +0.0003 PPV and five rows.
- **Standing still costs ~5 places a day.** The 0.0151 row was 72nd on 2026-08-19 and 102nd
  six days later with nothing submitted in between. The field fills in roughly six times
  faster than the best lever ever measured here climbs.

Density: 99 rows span 0.0104 → 0.0151, so **~21 rows per 0.001 PPV**, and nothing left in §7
moves 0.001. Note also the board ranks **submissions, not teams** — all three of ours hold
separate ranks — so far fewer distinct teams are ahead than the rank suggests; RARE25's
development cohort was 20 teams against a 201-row board. None of this changes the ranking that
counts (§8), but it does close the question of whether shipping a better dev score is worth
spending the remaining days on. It is not.

**What is staged right now is a different, unshipped system:** the two-arm 10-model fusion
(§3), re-staged 2026-08-17 19:52. `submission_template/resources/` holds `dinov2_fold0-4`
(copied from `runs/V2_dinogn_fold*`) and `swsl_fold0-4` (from `runs/V7_swsl_fold*`),
`platt.json` carries all ten (a, b) pairs, and `inference.py` implements P0 → two arms →
per-model Platt → equal-weight probability average. **Built and container-verified 2026-08-25, not yet uploaded.** Image
`sha256:904cbff64cdc8952f6982bd21cdcc70a02993477b4b3f57baa50269751533ed3`, 16.8 GB, saved to
`twoarm_10model.tar.gz` at the project root (gitignored via `*.tar.gz`) — 5.92 GiB (6,357,006,112 bytes). Verified under
`--network none --gpus all` on the RTX 5060: CUDA live, both arms loading 5 checkpoints each,
Platt calibration and the 1:100 prior shift both executing, 16 fixture frames scored
0.0007–0.99999987 with no ties and no saturation. Worth noting the fixture's negatives land
**lower** than the same 16 frames on 2026-08-11 (0.0007–0.0029 against 0.0015–0.0084) — the
direction prior shift is supposed to move them, though n=16 on a fixture proves nothing.

Measured inference cost: **6.1 ms per frame per model**, plus ~25s fixed (container start,
imports, CUDA init, checkpoint loads). The retired 15-model image ran 91 ms/frame — ~2 min
for 1,000 frames, ~16 min for 10,000 on an RTX 5060 Laptop. **Runtime is not a risk**, and
the two-arm tarball measures 5.92 GiB, matching the size known to upload successfully, so
**size is not a risk for two arms**. **Corrected 2026-08-25:** the base image
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` is **12.1 GB, not the 4.3 GB previously
recorded here** — the image is 12.1 base + 2.2 resources + 0.3 pip = 16.8 GB, compressing
2.64x overall. Checkpoints are high-entropy and barely compress, so a third arm's +1.7 GB
carries through nearly intact and would put the tarball near **8 GB, ~34% above the only size
ever known to upload**. A third arm is a size risk to test, not a free addition. Save
directly with `docker save <tag> | gzip -c > out.tar.gz`; do **not** use `do_save.sh`,
which rebuilds first and hits a Docker context lock (§5).

**Submission budget: one per week. The 2026-08-10 slot went to submission 2.** Slots do
not roll over — a week without a submission destroys a slot rather than banking one, so
any experiment that delays a submission past the week boundary costs one.

**Superseded 2026-08-19.** The best-vs-latest question above is moot in both directions: the
development leaderboard does not determine the ranking at all. Ranking and prizes come from
the Closed Testing Phase, 1–7 September, which takes **exactly one submission** on a much
larger test set (§8). Development submissions are therefore experiments, not placements —
their only value is as external measurements of changes local validation cannot resolve.
**How many remain is itself contested** (one per week in this log, ~twelve per the external
review) and needs checking against the rules page.

### The calibration point (2026-08-10) — read this before trusting any number below

Submission 2's leaderboard result against the same model's local LOCO:

| | local LOCO (c1 / c2) | leaderboard |
|---|---|---|
| AUPRC | 0.871 / 0.975 | **0.342** [0.100, 0.622] |
| AUROC | 0.984 / 0.992 | **0.816** [0.671, 0.942] |
| PPV@90R (Mode-A, 100:1) | 0.205 / 0.685 | **0.0148** [0.0105, 0.0641] |
| FPR@90R | 3.6% / 0.6% | **60%** |

Chance PPV at 100:1 is 0.0099, so the scored metric sits at 1.5× chance. The field is
compressed near the floor: 1st = 0.0332 (FPR@90R 26%), 10th = 0.0228 (39%). Reaching top
10 means cutting FPR@90R from 60% to 39%.

Two conclusions, pointing in opposite directions. Hold both:

- **LOCO's direction is validated.** A change LOCO called large and positive moved us
  103 → 62. It is a sound basis for A/B decisions.
- **LOCO's absolute numbers are meaningless.** FPR@90R is 17–100× worse in the wild than
  locally. Never quote a local number as an expected leaderboard number.

And note the interval: [0.0105, 0.0641] **contains the entire top 10**. Placement between
ranks 10 and 62 is largely not resolvable by this test set, so a modest genuine improvement
can move many places — and so can nothing at all.

---

## 2. The one thing that worked

Swapping the frozen backbone from stock DINOv2 (LVD-142M) to the endoscopy-pretrained
`dinov2.pth` — DINOv2 ViT-B/14 **register variant**, self-supervised on GastroNet-5M
(~4.8M images, 8 Dutch hospitals, 2012–2020).

| leave-one-centre-out | stock | domain |
|---|---|---|
| center_1 AUPRC | 0.634 | **0.873** |
| center_2 AUPRC | 0.863 | **0.975** |

Non-overlapping confidence intervals on both centres. Independently confirmed with a
linear probe on frozen features (no LoRA at all): AUPRC 0.521 → 0.695 and 0.763 → 0.938.

Provenance: 0 of 173 tensors match Meta's stock reg4 release; `blocks.11.mlp.fc2` cosine
similarity 0.057 — early layers close to stock, late layers rewritten. Source
[Theta Vision Cortex](https://cortex.thetavision.nl/dataset-provider/listing/2/) (the
HuggingFace repo `tgwboers/GastroNet-5M_Pretrained_Weights` holds only a README).
sha256 prefix `20ab749ec4bfdd53`.

---

## 3. Everything measured

All via **leave-one-centre-out** (train on two centres, test on the third), paired
group-clustered bootstrap. Baseline throughout: LoRA-6 domain DINOv2, center_1
AUPRC 0.8710 / center_2 0.9735.

| lever | verdict | numbers |
|---|---|---|
| **domain backbone** | **large win** | AUPRC 0.634→0.873, 0.863→0.975, CIs disjoint |
| GastroNet vs ImageNet ResNet50 | real win | AUPRC +0.085, P(gain)=1.000 (arm later dropped) |
| LoRA blocks 1/3/6/12 | **6 is a tested optimum** | 0.812 / 0.856 / **0.871** / 0.798 on center_1 |
| ResNet fusion (fine-tuned arm) | harmful | center_2 PPV 0.685→0.167, AUPRC delta P=0.012 |
| seed-varied ensembling (5 members) | null | center_1 0.221→0.211, center_2 0.685→0.623 |
| TTA (hflip + centre crop) | null on PPV | P(gain) 0.54 / 0.66 |
| threshold calibration | irrelevant | container submits likelihoods; organizers sweep |
| augmentation (`--aug acq`) | null / slightly negative | AUPRC −0.004 / −0.006, all six deltas negative |
| tail ranking loss (λ=0.1, 0.3) | one centre only | center_2 AUPRC +0.008 P=0.989; center_1 null; **null on PPV** |
| frozen-probe fusion (3 backbones) | works among equals | center_1 0.695→0.744, center_2 0.938→0.967 |
| frozen probes on top of LoRA model | no gain | center_1 −0.033 P=0.097, center_2 +0.003 P=0.646 |
| **restricted-adaptation ResNet** | **loses to DINOv2; closed** | AUPRC −0.069 P(gain)=0.015 (c1), −0.029 P(gain)=0.006 (c2); both CIs exclude 0 |

Seven of nine peripheral levers null or harmful. **The frozen backbone is nearly the
whole system**, but LoRA is not inert — it adds +0.176 AUPRC on center_1 over a linear
probe (0.695 → 0.871). What is null is *changing* the trainable part, not having it.

### Backbone screening (linear probe on frozen features, LOCO)

| backbone | center_1 | center_2 | evc |
|---|---|---|---|
| stock DINOv2 | 0.5213 | 0.7628 | 0.9016 |
| RN50 SWSL+GastroNet DINOv1 | 0.6593 | 0.9183 | 0.9711 |
| RN50 GastroNet-5M DINOv1 | 0.6857 | 0.9334 | 0.9772 |
| domain DINOv2 (shipped) | **0.6950** | **0.9378** | 0.9714 |

A ResNet50 matches the ViT-B as a frozen extractor. The plain `GastroNet-5M` variant
beats `SWSL+GastroNet` — the opposite of what was predicted when they were downloaded.

### Restricted adaptation on the ResNet arm (LOCO, `--freeze-until`, 2026-08-10)

Held-out AUPRC, `RN50_GastroNet-5M_DINOv1.pth`, folds_v2:

| trainable | center_1 | center_2 |
|---|---|---|
| everything (full fine-tune) | — | 0.586 |
| **layer4 + fc (15.0M of 23.5M)** | **0.805** | **0.944** |
| fc only (~2K) | 0.469 | 0.853 |
| DINOv2 LoRA-6 (shipped) | 0.871 | 0.975 |

**The curve is an inverted U, not monotonic in less capacity.** `layer4+fc` beats both
the full fine-tune and the linear head. This is the one place where §4's "restricting
trainable capacity has helped every single time" needs qualifying: it has an optimum,
and going past it hurts as much as stopping short.

Freezing the trunk rescues center_2 from 0.586 to 0.944 — a +0.36 swing that confirms
§4's claim that the arm was retired on a training mistake, not on its merits. **It still
loses to DINOv2 on both centres** under the paired test, so the predeclared parity gate
for a fusion test failed and the arm is closed. Closed correctly this time.

Beyond the failed gate, there is a structural reason not to pursue fusion anyway: with
EVC degenerate, only two usable centres exist, so any fusion weight is selected on the
same data that would validate it. That is exactly the mechanism that put +0.12 of pure
optimism into `fuse.py`. `fuse_loco.py` needs a predeclared weight and a third centre;
we have neither.

Caveat on the `fc only` row: it is undertrained, not a linear-probe measurement. The same
frozen trunk scores 0.686 / 0.933 under `probe_features.py`, which uses sklearn logistic
regression; 30 epochs of AdamW at 1e-4 on one linear layer does not converge comparably.
Do not cite that row as the probe number.

### Acquisition shift is NOT the failure mode — hypothesis refuted (2026-08-11)

The obvious explanation for the leaderboard gap was device/acquisition shift: twelve unseen
centres use different scopes, processors and compression. Tested with
`evaluation/shift_stress.py` — score the held-out-centre model under 36 graded corruptions
(gamma 0.45–2.10, white balance ±22%, brightness, contrast, sharpness, JPEG q25–60,
blur σ3.5, downsample to 25%, noise σ0.09, vignette 0.7).

| | clean FPR@90R | worst of 36 corruptions | leaderboard |
|---|---|---|---|
| center_2 | 0.56% | **1.67%** (gamma 2.10) | 60% |
| center_1 | 3.56% | **4.82%** (gamma 0.45) | 60% |

AUPRC never fell below 0.83 and several corruptions *improved* the score. **The model is
robust to acquisition shift; it explains essentially none of the gap.** Matching the
leaderboard from center_1 would need ~17×; the worst corruption manages 1.35×.

Consequences:

- Augmentation targets a failure mode we do not have. The earlier "augmentation is null"
  result was **correct**, and the argument that LOCO was structurally blind to it was wrong.
- Holding out an acquisition stratum instead (scope model, modality) is not possible:
  `imaging_mode`, `patient_id`, `video_id`, `examination_id` and `fov_geometry` are all
  **0/3095 populated** in `data_inventory_rare25.csv`. Native resolution varies but is
  perfectly centre-correlated, so holding it out merely repeats LOCO.

**Best remaining explanation: semantic case-mix shift in the positive tail.** Unseen centres
likely carry subtler neoplasia, post-treatment cases, inflammation and mimics; a few
atypical positives score low, drag the 90%-recall threshold down, and admit masses of
negatives. That produces FPR@90R 60% while AUPRC stays a respectable 0.342. Pretraining
familiarity — our centres are probably inside GastroNet's SSL corpus — contributes to local
optimism but is likely not the whole gap. **Neither is testable with the data we hold:
local validation cannot be repaired.**

### What the metric actually is — one order statistic (2026-08-17)

PPV = 0.90 / (0.90 + 100·FPR@90R), so the score is decided entirely by **where the
bottom-decile positive falls in the negative distribution**. Locally 4 images per fold set
the threshold; lifting the worst 1/2/3/5 positives to the positive median takes mean
per-fold FPR@90R 0.0110 → 0.0060 / 0.0030 / 0.0023 / 0.0017.

| | FPR@90R | bottom-decile positive outranks |
|---|---|---|
| our folds (in-distribution) | 1.1% | 98.9% of negatives |
| LOCO center_1 | 3.6% | 96.4% |
| **us, leaderboard** | **58.7%** | **41.3%** — below the median negative |
| 10th place | 38.6% | 61.4% |
| 1st place | 26.0% | 74.0% |

Two things follow. **Nobody in this challenge is finding the hard positives** — even 1st
place is outscored by a quarter of all negatives at its threshold. And the sensitivity is
tiny: **dPPV/dFPR ≈ −0.0253, so one point of FPR is worth +0.00025 PPV.** Top 10 needs ~20
points. The 15-model ensemble delivered 1.2. No incremental lever reaches top 10; that is
arithmetic, not pessimism.

### Cross-centre score-scale drift — refuted (2026-08-17)

The leaderboard pools 12 centres, so a low positive from centre A sets a global threshold
that admits negatives from every other centre. Tested per fold, so cross-fold scales never
mix: pooled AUROC **0.9819** vs mean within-centre **0.9777** (pooling *helps*), and a
90%-recall threshold calibrated on one centre gives FPR 0.0691 on its own negatives vs
0.0686 on the other centre's — **inflation 0.99×**. Not the mechanism. Weak evidence (two
Dutch centres, both probably inside GastroNet's SSL corpus) but it points away from drift,
and it closes "make the scores centre-invariant" as a lever.

### The diagnosis, revised: it is not only the tail (2026-08-17)

Leaderboard **AUROC 0.820 against 0.98 locally.** If a handful of atypical positives were
dragging the threshold, AUROC would barely move — a few positives cannot shift a rank
statistic over thousands of images. AUROC at 0.82 means the **whole ranking degrades** on
unseen centres. This supersedes the pure "semantic case-mix shift in the positive tail"
reading of §3: the tail is where the metric reads the damage, but the damage is broad.

It also explains why every tail-targeted lever came back null — they were aimed at the wrong
mechanism. **The representation does not transfer**, and changing the representation is the
only thing that has ever worked here (§2).

### Closed on predeclared LOCO tests (2026-08-17)

Three levers, all judged on AUPRC against `LOCO_dinov2_rest_to_center_{1,2}_domain`:

| lever | center_1 AUPRC Δ | center_2 AUPRC Δ | verdict |
|---|---|---|---|
| positive-enriched batches + capped tail loss (`posenr`) | −0.0042 P(gain) 0.326 | +0.0018 P(gain) 0.695 | **null** |
| the same with AUPRC selection (`posenr_ap`) | −0.0134 P(gain) 0.113 | +0.0016 P(gain) 0.680 | **null / worse**; AUROC −0.0079 [−0.0220, −0.0009] |
| **224px** (`r224`) | **−0.0286 [−0.0610, −0.0028] P(gain) 0.017** | +0.0042 P(gain) 0.717 | **worse; 336 is a tested optimum** |

Two real pipeline defects were found and *fixed*, and fixing them changed nothing:

- **Checkpoint selection was on a saturated statistic.** Inner AUPRC sat at 0.952–0.992 with
  a mean of **4.7 epochs tied within 0.005**, and it early-stopped every shipped run at
  **epoch 2–7 of a 30-epoch budget**. The replacement (`--select spec`, mean specificity over
  recall 0.80–0.95) saturated *harder* — 0.997–1.000, selecting epoch 0 in three of five
  folds. `--ckpt-avg 2` partly rescued it by averaging epochs 0–2. Note the diagnostic
  result: the degenerate early selection was the **better** of the two (−0.0042 vs −0.0134 on
  center_1), i.e. it was accidentally regularising. Restricting capacity again.
- **The tail loss had ~2 positives per batch**, as its own docstring conceded, so it was
  hard-pair mining rather than a quantile objective. Given 8 positives per batch and a skip
  band that stops it mining the mislabelled worst, it is still null.

**Per-fold CV said this was a large win: V6_pos alone reached FPR@90R 0.0067, matching the
entire 15-model ensemble at a third the size, and adding it to the shipped three cut another
25% to 0.0050. LOCO says null. That is the fourth time pooled/per-fold CV has misled.**

### The two-arm fusion — the second thing that ever worked (2026-08-17)

Domain DINOv2 LoRA-6 **+ GastroNet+SWSL ResNet50 (layer4+fc)**, equal weight, per-model
Platt calibration, probabilities averaged. Predeclared LOCO gate, judged on AUPRC:

| held-out | AUPRC | Δ | P(gain) | AUROC Δ | PPV@90R |
|---|---|---|---|---|---|
| center_1 | 0.8710 → 0.9012 | **+0.0290** | 0.945 | +0.0029 | 0.2049 → 0.3921 |
| center_2 | 0.9735 → 0.9754 | +0.0018 | 0.703 | +0.0012 | 0.6850 → 0.8684 |

**All six deltas positive on both centres** — the mirror image of how augmentation was
rejected. Members are Spearman 0.48–0.59 apart and share only ~50% of bottom-decile
positives; the retired 3-family ensemble shared ~75% at Spearman 0.69–0.91. **Diversity of
representation, not of recipe, is what pays.**

Two mechanical requirements, both measured, not assumed:
- **Probabilities, not logits.** Fitted Platt slopes range 0.46–1.19 and intercepts −3.25 to
  +1.98 across the ten checkpoints. The **raw** probability average measured *negative* on
  both centres (center_2 AUPRC −0.0187, P(gain) 0.039).
- **Equal weight by predeclaration.** With two usable centres any fitted weight is selected
  on the data that would validate it.

Also note **SWSL adapted (0.859/0.967) beats plain GastroNet-5M adapted (0.805/0.944)**,
reversing the frozen-probe screen that ranked SWSL lower (0.659/0.918 vs 0.686/0.933).

### Closed on predeclared LOCO tests (2026-08-17/18) — that day's search exhausted

| lever | center_1 AUPRC Δ | center_2 AUPRC Δ | verdict |
|---|---|---|---|
| third arm: plain GastroNet RN50 | +0.0177 (vs +0.0290) | +0.0007 | **dilutes** |
| third arm: stock DINOv2 | **−0.0263 [−0.0567, −0.0007]** | −0.0029 | **worse** |
| MoCoV2 / SimCLRv2 GastroNet variants | frozen screen: −11.8% / +10.7% FPR | **+175% / +62.5%** | **worse** |
| GastroNet-1M / 200K / ViT-S | frozen screen: none helps both centres | | **no** |
| crop aggregation, top-2 of 5 native-res windows | **−0.0505 [−0.105, −0.006]** | +0.0104 | **worse** |
| crop aggregation, blended with whole frame | −0.0085 | +0.0137 | **fails gate** (AUROC +0.0038/+0.0060, P 0.906/0.966) |
| **P3 preprocessing (border masking)** | **−0.0210 [−0.0485, −0.0003]**, AUROC −0.0063 [−0.0126, −0.0016] | +0.0006 | **worse; P0 stands** |

**P0 had never been validated on the shipping system.** It was chosen on the *retired ResNet
arm*, on *folds_v1*, by *per-fold CV*, on a 0.8242 vs 0.8098 margin — inside noise. It is now
tested properly and holds.

**Crop aggregation is instructive, not merely null.** It helps center_2 and significantly hurts
center_1 — the 2.68%-prevalence centre with 2279 images. Five windows give every negative five
chances to fire. Under a 100:1 metric that is the dominant effect, so the low-prevalence centre
is the one to trust.

### The diagnosis, corrected again by adversarial review (2026-08-17)

Two independent external red-team passes rejected "the representation does not transfer" as **a
description, not a diagnosis**, and both independently derived the same alternative:

If a fraction **f** of hidden positives are frames where no lesion is reliably visible
(exam-level labelling, frame selection, post-treatment), then reaching 90% recall forces
**FPR@90R = (f − 0.1)/f**. Our 58.7% implies **f ≈ 0.24**; 1st place's 26% implies **f ≈ 0.135**.
The same mixture reproduces the observed hidden AUROC: 0.98·(1−f) + 0.50·f ≈ 0.82.

One model accounts for AUROC 0.820, AUPRC 0.343, FPR@90R 58.7%, **and** why fourteen levers
came back null: none of them made a previously-invisible lesion visible. **The gap between rank
72 and rank 1 is finding the lesion in roughly 10% more of the hard positives.**

**Superseded 2026-08-19 — do not reason from this section without reading §8.** The mixture
above is over-determined and its two constraints disagree: f = 0.242 predicts AUROC 0.864,
while AUROC 0.820 requires f = 0.333 and predicts FPR@90R 0.700. Solved jointly it gives
f = 0.242 with visible-positive AUROC 0.922, not 0.98 — and the ceiling implied by treating
f as a property of the data (0.879) is already exceeded on the board (0.8949). The hard
positives carry usable signal. Roughly half the gap is an ordinary cross-centre
generalisation gap, which is exactly the half that backbones and augmentation act on.

**The standing criticism of our own protocol:** two repeatedly-reused Dutch centres have become
"a two-point development set, not external validation". Roughly a dozen predeclared LOCO tests
were run against them on 2026-08-17 alone. Each spends validity even when no weights see the
data. Treat further LOCO A/Bs as close to exhausted.

**External data is a trap here, and specifically so.** RARE26 is the direct sequel to RARE25;
our training set is almost certainly RARE25's public release (2937 NDBE + 158 neoplasia,
2 Dutch centres) and the hidden test is 12 BONS-AI consortium centres. Barrett's images taken
from BONS-AI-affiliated publications risk **literal overlap with the hidden test set**.
Realistic net-new positive yield from public sources is ~50–150 images with a domain gap.

### The Aug 18 battery — seven levers from two external reviews (2026-08-18)

Two outside reviews (`suggestion_gp.txt`, `suggestion_2025 winner.txt`, both since rewritten
with a second round) supplied a fresh candidate list, largely reconstructed from the RARE25
winner's public code. All judged on the standard predeclared LOCO test against
`LOCO_dinov2_rest_to_center_{1,2}_domain` (baseline AUPRC 0.8710 / 0.9735).

| lever | center_1 ΔAUPRC (P gain) | center_2 ΔAUPRC (P gain) | verdict |
|---|---|---|---|
| stock DINOv3 ViT-L/16 + LoRA (`dinov3l`) | 0.622 absolute vs 0.871 | 0.921 vs 0.974 | **far worse — but not bankable, see below** |
| 448px (`r448`) | −0.0234 (0.129) | −0.0025 (0.371) | **worse; 336 remains the optimum** |
| patch-token MIL, top-k pooling (`mil`) | −0.0127 (0.184) | +0.0073 (0.889) | **fails the gate** |
| PPV@90R surrogate loss (`ppv`) | −0.0000 (0.488) | +0.0008 (0.822) | **null** |
| prior-shift + noisy-OR pooling (`priorshift`) | −0.0072 (0.190) | +0.0000 (0.495) | **not a clean null — see below** |
| jigsaw augmentation, standalone (`jig`) | +0.0189 (0.857) | −0.0051 (0.161) | **fails as a replacement arm** |
| **jigsaw as a third fusion arm** | **+0.0158 (0.899)**, AUROC +0.0055 (0.917) | +0.0016 (0.686), AUROC −0.0002 (0.422) | **the only survivor** |

**The DINOv3 result is not bankable as a backbone finding.** Our LoRA config — r16 on the
last 6 attention blocks — was tuned for a 12-block ViT-B. On a 24-block ViT-L it covers the
top quarter, attention only, 0.591M of 303.7M parameters (0.19%); the winner used all linear
layers at r32/α64/dropout 0.1. Line the backbones up on center_1 — stock DINOv2 0.634, stock
DINOv3-L 0.622, GastroNet DINOv2 0.871 — and a 0.25 AUPRC step is invariant to a 3.5×
capacity increase and a full pretraining generation. Either the config does not transfer, or
the only thing center_1 measures is whether a checkpoint saw GastroNet. **Both are live**,
and the second would invalidate the arm-selection logic behind the whole system.

**The three-arm fusion is equal thirds**, verified numerically against the stored predictions
(max |diff| 1.0e-05 on center_1, 3.9e-04 on center_2; 0.5/0.25/0.25 is off by 0.14). Each
member is Platt-scaled on its own within-rest holdout, then probabilities averaged.
`V8_jig_fold0-4` is the matching full-data family (`--aug jigsaw`, 336px, LoRA-6, P0,
confirmed from the checkpoints' own args), trained 2026-08-18, complete with OOF predictions.

Read on FPR@90R, the statistic the scored metric is actually a function of:

| held-out | 1-arm (shipped) | 2-arm | 3-arm +jig |
|---|---|---|---|
| center_1 | 0.0356 | 0.0140 | **0.0113** |
| center_2 | 0.0042 | **0.0014** | 0.0028 |

The two centres disagree about the third arm even though AUPRC prefers it on both. center_1
is the 2.68%-prevalence centre and the one to trust under a 100:1 metric (§3, crop
aggregation). Note also that two of the three arms share the domain DINOv2 backbone, so an
equal-thirds fusion moves that backbone from half the weight to two thirds — this is recipe
diversity partly displacing the representation diversity that §3 says is what pays.

### Calibration and pooling, and the protocol bug it exposed (2026-08-18)

Five poolings of the same underlying logits, on the two-member LOCO fusion and on the
10-member 5-fold OOF (`calib_pool.py`):

| variant | c1 FPR@90R | vs shipped | c1 AUPRC | c2 FPR@90R | OOF FPR@90R |
|---|---|---|---|---|---|
| `logit_mean` (what the 15-model system did) | 0.0176 | +25.8% | 0.8975 | 0.0028 | 0.0084 |
| `prob_nat` (shipped) | 0.0140 | — | 0.9012 | 0.0014 | 0.0137 |
| `prob_prior` (shifted to 1:100) | **0.0108** | **−22.6%** | 0.8932 | 0.0014 | 0.0144 |
| `noisyor_nat` | 0.0117 | −16.1% | 0.8952 | 0.0014 | 0.0137 |
| `noisyor_prior` | **0.0108** | **−22.6%** | 0.8856 | 0.0014 | 0.0144 |

Scored on AUPRC this is a null (−0.0072, P(gain) 0.190) and was recorded as one. Both
external reviews independently called that a **protocol bug**: the ranking metric is PPV@90R,
a strictly monotone function of FPR@90R, and AUROC/AUPRC are supplementary diagnostics that
do not affect ranking. A change that cuts FPR@90R 22.6% while costing 0.008 AUPRC is what a
*correct* tail intervention looks like — AUPRC integrates across all recalls and is nearly
blind to the extreme tail. See §4.

Do not over-read it either: center_2 is unchanged, the 5-fold OOF reverses the ordering, and
0.0140 → 0.0108 at 61 positives is about seven fewer false positives. A relative cut measured
at FPR 0.014 says little about behaviour at 0.587. Even transferring intact it lands near
0.0194, short of 10th place at 0.0228.

### The near-duplicate audit — the validator is clean (2026-08-19)

Both external reviews made this P0. `group_id` comes from the challenge's own
`duplicate_group` field (3078 groups over 3095 images) and patient/video/exam IDs are
0/3095 populated, so the "patient-clustered bootstrap" behind every A/B decision in this log
was effectively image-level. If near-duplicate frames crossed folds, every local number was
inflated and the null results were never powered.

`evaluation/dup_audit.py` runs three independent detectors over all 3195 images — md5,
64-bit dHash/pHash, and cosine similarity in `feats_domain.npz`:

| cosine ≥ | pairs | cross-fold | pos–pos pairs | independent positives c1 / c2 / evc |
|---|---|---|---|---|
| 0.95 | 13 | 3 | 0 | **61 / 97** / 50 |
| 0.90 | 14 | 3 | 0 | **61 / 97** / 50 |
| 0.85 | 20 | 4 | 5 | **61 / 97** / 46 |
| 0.80 | 40 | 12 | 12 | **61 / 97** / 41 |
| 0.75 | 123 | 64 | 33 | **61 / 97** / 31 |

Nominal is 61 / 97 / 50. **center_1 and center_2 positives stay fully independent down to
0.75**, a threshold loose enough to merge visibly different images. Every lever test was
powered by exactly the positive count it claimed.

**`evc` is the built-in positive control**, and it is what makes this a credible null rather
than a failed search: it is the one centre known to carry multiple images per patient (100
images, 39 patients), and it is the one centre that clusters — 50 → 46 → 41 → 31. The method
detects same-patient multi-frame structure where it exists, and finds none at the two centres
every decision was made on.

Supporting numbers: nearest-neighbour cosine has median 0.652, 90th 0.724, 99th 0.829, so
0.95 is a genuine outlier cut rather than an arbitrary one. The strict union (md5 | pHash ≤ 4
| cosine ≥ 0.95) gives 3168 components over 3195 images — 23 non-singleton groups, 9
cross-fold pairs, 1 cross-centre, 2 label-mixing. All 11 anomalies were rendered to
`dup_pairs_review.png` and inspected: they are real same-examination frames, several sharing
identical anonymisation redaction bars. The challenge's own `duplicate_group` is *more*
aggressive than this detector (3117 groups vs 3168) and `make_folds.py` keeps its groups
within a fold, so the 9 cross-fold pairs are precisely what its numbering missed.

**One pair deserves a human look:** center_1 `20d84cef…` (fold 3, NDBE) and `06997e08…`
(fold 4, neoplasia) are the same examination by appearance, with opposite labels, straddling
a fold boundary. One instance — either label noise or a before/after pair.

**What it does not settle.** Two frames from one examination minutes apart score in the bulk
of the distribution (0.65–0.75) and are indistinguishable from unrelated images by any method
available here, so exam-level grouping is not *proven* absent — only that it is not
near-duplicate-shaped. And it says nothing about **GastroNet pretraining overlap**, a
different mechanism entirely (pretraining exposure, not fold leakage), which remains the
leading explanation for why center_1 performance tracks only whether a checkpoint saw
GastroNet.

### The Aug 19 battery — the winner's recipe, transplanted and rejected (2026-08-19)

IMSY's RARE25 winning code was read directly (see §8 for the confirmed configuration), and
the three transplantable pieces were tested here. All three lost.

| lever | center_1 | center_2 | verdict |
|---|---|---|---|
| all-linear LoRA r32/α64/dropout 0.1 vs our attention-only r16 | AUPRC +0.0225 (P 0.868), **FPR@90R 0.0965 → 0.1686** | +0.0058 (P 0.664), FPR@90R 0.0515 → 0.0584 | **no** |
| the same, against the GastroNet arm | **−0.2232 [−0.334, −0.134]** P 0.000, AUROC −0.0477 [−0.089, −0.015] | **−0.0451 [−0.082, −0.018]** P 0.000, AUROC −0.0127 | **far worse** |
| heavy augmentation floor (`--aug heavy`) vs the jigsaw arm | **AUPRC −0.0414 [−0.081, −0.013] P 0.002** | −0.0058 (P 0.316) | **worse** |
| MaxViT-Tiny 384, ImageNet, full fine-tune | AUPRC 0.5604, FPR@90R 0.3088 | 0.8419, 0.1043 | **far worse** |

**The under-adaptation confound is refuted.** Our LoRA config (r16, attention only, last 6 of
24 blocks — 0.591M of 303.7M, 0.19%) was the leading explanation for stock DINOv3 losing by
0.25 AUPRC. Giving it the winner's adapter — all linear layers in those blocks, r32/α64,
dropout 0.1, **3.147M trainable, a 5.3× increase** — recovers +0.0225 AUPRC on center_1
against a 0.223 gap, and makes the tail *worse*. Roughly a tenth of the gap, none of it where
the metric reads.

The winner's exact config is all 24 blocks (12.584M trainable). Measured on this GPU: peak
19.62 GB at batch 32, 10.81 GB at 16, **6.42 GB at 8**. It only fits at batch 8, which would
confound the adapter change with a batch change, and at 4.7% prevalence batch 8 averages 0.37
positives per batch. Given the last-6 step bought +0.0225 against a 0.223 gap, all-24 would
have to be ~10× more effective; **not worth the GPU hour.**

**The learning rate was a real defect and cost nothing.** IMSY train at lr 1e-4 for 50 epochs;
we use 5e-4 for 30 and early-stop at epoch 2–7 on a saturated inner statistic. Retrained at
1e-4 / 50 epochs / patience 10, the run trained to **epoch 18** before stopping — confirming the
premature stop is a learning-rate artifact — and measured **null to mildly negative**: center_1
AUPRC −0.0082 (P 0.213) and AUROC −0.0077 [−0.0191, +0.0008] (P 0.041), center_2 AUPRC +0.0014
(P 0.565). Note center_1's FPR@90R *improved* (0.0356 → 0.0298) while the stabler band mean
worsened (0.0385 → 0.0652) — a clean illustration of why §4 now requires both. This closes a
standing suspicion that the whole training regime was capped by its schedule. It was not.

**Heavy augmentation is not undertrained.** IMSY apply a floor to every one of their 40 models
— RandomResizedCrop(0.8–1.0), H/V flip, rotate ±15° p 0.8, ColorJitter p 0.8 — and vary only
the preset above it. Transplanted here (with jigsaw on top, so the single variable is the
floor) it ran to epoch 21 with best at 14, where our runs normally stop at 2–7. It trained
longer, as it should, and still lost. Our earlier "augmentation is null" verdict was measured
in a different regime and the objection was fair; the answer is the same.

### MaxViT, closed on the whole capacity curve (2026-08-19)

The RARE25 runner-up (UT) reached **2nd place** with MaxViT-Tiny at 384px on challenge data
alone, no large-scale endoscopy pretraining, dev AUROC 0.7709. Our `W1_maxvit` is logged as
the worst arm we ever trained — but on folds_v1, with our recipe. §4 records rejecting the
ResNet arm on exactly this mistake, so MaxViT was measured at three points instead of one:

| trainable capacity | center_1 AUPRC | center_2 AUPRC |
|---|---|---|
| frozen probe (0) | 0.1770 | 0.3208 |
| stages.3 + norm + head (17.8M of 30.5M) | 0.4125 | 0.7625 |
| full fine-tune (30.5M) | **0.5604** | **0.8419** |

**Monotonic in capacity — no inverted U, and the best point is the full fine-tune.** The
ResNet precedent did not repeat, and even the best point sits 0.31 AUPRC behind the GastroNet
arm on center_1. Closed, and closed fairly this time.

### Six representations, one variable (2026-08-19)

Held-out center_1 AUPRC, every backbone and adaptation regime measured on this project:

| representation | c1 AUPRC | c1 FPR@90R |
|---|---|---|
| ImageNet MaxViT-Tiny, frozen | 0.1770 | — |
| ImageNet MaxViT-Tiny, stages.3+head | 0.4125 | — |
| ImageNet MaxViT-Tiny, full fine-tune | 0.5604 | 0.3088 |
| stock DINOv3 ViT-L/16, attention-only LoRA | 0.6204 | 0.0965 |
| stock DINOv2 ViT-B/14 | 0.6311 | 0.2214 |
| stock DINOv3 ViT-L/16, all-linear LoRA r32 | 0.6429 | 0.1686 |
| **GastroNet DINOv2 ViT-B/14** | **0.8710** | **0.0356** |

The gap is invariant to architecture family, pretraining paradigm (supervised vs SSL),
pretraining corpus size (1.3M → 142M → 1689M images), adapter type and trainable capacity.
**Exactly one variable predicts center_1 performance: whether the checkpoint saw GastroNet.**

Two readings survive, and after six experiments aimed at separating them the conclusion is
that **no local experiment can**: either GastroNet carries domain knowledge nothing else has,
or this validator rewards having seen these centres and every arm selected on it was selected
on an artifact. Both predict this table exactly. Only an external measurement separates them
(§7 item 3).

### Jigsaw re-tested on the metric that ranks (2026-08-19)

§4's revised rule says AUPRC must not veto a metric-aligned change, so the jigsaw arm — which
we rejected as a standalone on AUPRC — was re-scored on the tail. Point estimates favour it
over the shipped arm on both centres (c1 FPR@90R 0.0356 → 0.0144, c2 0.0042 → 0.0000), but the
paired group-clustered bootstrap does not confirm it:

| | center_1 | center_2 |
|---|---|---|
| FPR@90R | P(better) 0.701, CI [−0.098, +0.125] | P(better) 0.691, CI [−0.021, +0.025] |
| mean FPR 0.80–0.95 | P(better) 0.626 | P(better) 0.494 |

Consistently favourable, never decisive, every CI spanning zero. **Not a win — jigsaw stays a
third fusion arm, not a replacement.** Recorded because the point estimates are seductive and
someone will re-derive them.

### The localization route — the last podium approach, and it also fails (2026-08-19)

Jmees (RARE25 3rd) reframed the task as segmentation, on the argument that early neoplasia
against NDBE is barely decidable without spatial cues. This is **not** the `--mil` head that
measured null in §3: MIL infers location from image-level labels and never learns what a lesion
looks like. Here the decoder is trained on **real masks from an external dataset first**, so
lesion appearance is learned before it ever meets RARE data.

Built as: frozen GastroNet DINOv2 → last four blocks concatenated (4 × 768 at 24×24) → a 1.38M
parameter conv decoder → per-pixel logits, BCE + soft Dice, trained on **EDD2020** (386 frames,
5 institutions, CC BY-NC-SA 4.0). Lesion = suspicious + HGD + cancer; BE excluded, since
non-dysplastic Barrett's is RARE's *negative* class and marking it lesion would train the
decoder to fire on exactly what it must ignore. The 188 BE-only and polyp-only images are kept
with all-zero masks — they teach the discrimination that matters. Image score is the mean of the
top-8 patch logits, never a spatial mean, which would let a large normal field wash out a small
lesion.

**The decoder works**: validation Dice **0.667** on held-out EDD2020 frames. This is a
functioning segmentation model, not a degenerate one.

**Zero-shot on RARE it splits by centre**, which is the interesting part:

| | median score, positives | median, negatives | AUROC | AUPRC |
|---|---|---|---|---|
| center_1 | **0.0020** | 0.0000 | 0.8230 | 0.2985 |
| center_2 | **0.9636** | 0.0000 | 0.8859 | 0.7662 |

It transfers to center_2 and evc and essentially fails on center_1 — the 2.68%-prevalence centre
that §3 says to trust under a 100:1 metric.

**As a third fusion arm it is worse on both centres:**

| | 2-arm | + segmentation |
|---|---|---|
| center_1 AUPRC / FPR@90R | 0.9012 / 0.0140 | 0.8860 / **0.0198** |
| center_2 AUPRC / FPR@90R | 0.9754 / 0.0014 | 0.9644 / **0.0028** |

The arm was Platt-calibrated on the centres it was *not* being tested on, so the fusion carries
no leakage. And the exchange rate repeats the DINOv3 trap exactly: on center_1 the cheapest
rescue of a core bottom-decile positive admits **129 negatives against a 31-negative budget** —
four times the entire false-positive allowance for one image. center_2 is nearer break-even
(1 and 2 negatives against a budget of 1 for the top two) and the fusion still measures worse.

**Overlap was checked before training and is clean.** EDD2020 against all 3195 RARE images by
pHash: **zero matches at Hamming ≤ 4**, the strict cut from the near-duplicate audit; closest is
6, which that audit showed already yields cross-centre false positives. This cannot speak to the
hidden test set, but it is the strongest check available and it is the risk §8 flags for
external Barrett's data.

**This was the last substantial untried idea** — the only podium approach never attempted here.
It was built properly, the segmentation model demonstrably works, and it still does not move the
metric.

### Acquisition-conditional tail normalisation — rejected, with a general lesson (2026-08-19)

Proposed as the one remaining use of the unlabeled release that is neither SSL nor
pseudo-labelling: if an unfamiliar scope or style inflates every score it produces, its
negatives outrank subtle positives from a lower-scoring style, and a metric that is one order
statistic over a pooled set pays for exactly that. Map each arm's score to its quantile *within
the acquisition stratum* of the image, shrunk toward the global CDF, before fusing.

Built acquisition-only features — geometry, exposure, colour balance, focus, specularity,
border extent — deliberately with **no learned embedding**, since GastroNet features are
lesion-sensitive and clustering on them would stratify partly by pathology. Six k-means strata.
Transductive: the reference CDF is built from the held-out centre's own images without labels,
frozen, and only then are labels revealed. Shrinkage predeclared at n/(n+100), untuned.

**The precondition passed on the centre that matters.** Strata cut across centres (purity
0.865; stratum 4 is exactly `evc`). Predeclared stop condition was <0.25 SD spread AND <2x
top-10% exceedance ratio:

| arm / centre | spread | exceedance ratio | |
|---|---|---|---|
| DINOv2, center_1 | 0.26 SD | 2.6x | proceed |
| DINOv2, center_2 | 0.13 SD | 1.5x | stop |
| **SWSL ResNet, center_1** | **1.08 SD** | **15.1x** | proceed |
| SWSL ResNet, center_2 | 0.18 SD | 5.4x | stop |

And the direction fit the hypothesis exactly: on the center_1 run — trained on center_2 — the
stratum dominated by center_2-*looking* images scored **+0.73 SD** with a 0.321 exceedance rate
against another stratum's −0.35 and 0.021. Acquisition familiarity inflating scores, visible
directly.

**The transform still failed, badly, on both centres:**

| held-out | current fusion | acq-normalised | gate |
|---|---|---|---|
| center_1 | FPR@90R 0.0140 | **0.0383** | −2.43 points (needs +2.00); non-worsening 26.4% (needs 90%) |
| center_2 | FPR@90R 0.0014 | **0.0042** | −0.28 points; non-worsening 45.5% |

**Why, and this is the part worth keeping: prevalence varies 15.4x across acquisition strata**
(3.2% to 50.0%; still 3.3x excluding `evc`). The stratum running +0.73 SD has 10.7% prevalence
against another's 3.2%. So the offsets were never pure acquisition nuisance — they were
substantially real signal, and forcing every stratum onto a common score distribution deletes it.

> **A within-stratum normalisation is only safe when the stratum is conditionally independent
> of the label.** Here acquisition style and pathology are strongly dependent, plausibly because
> clinicians image suspicious areas differently — closer, and more often under NBI or
> chromoendoscopy. The transform removes the confound and the signal together, and the signal is
> larger. This also retro-explains "cross-centre score-scale drift — refuted" above rather than
> contradicting it, and it applies to any future proposal of this shape.

### Evaluator parity audit — clean, and one thing worth knowing (2026-08-19)

Not a lever; pure downside protection, and it does not change the lever count. The question was
whether we are losing points to an implementation detail rather than to the model: the container
runs its forward pass under fp16 autocast, which quantises logits to roughly three decimal
digits and could in principle produce tie blocks at the operating point. Checked on real
container output (816 images through all ten checkpoints) and on both held-out LOCO fusions:

| check | container | center_1 | center_2 |
|---|---|---|---|
| scores exactly 0.0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 |
| distinct values | 99.1% | 99.9% | 97.4% |
| largest tie block | 3 | 2 | 4 |
| images within 1e-6 of the threshold | — | **1** | **1** |
| FPR@90R under tie-breaking jitter (20 draws) | — | identical to 6 dp | identical to 6 dp |
| JSON round-trip error | **0**, lossless | — | — |

Nothing to fix. `float(p)` through `json.dumps` preserves full double precision, and the fp16
forward pass does not bunch scores at the threshold.

**The thing worth knowing: recall quantisation in the official protocol.** It resamples
positives down to n_neg/100 per iteration — **22 on center_1, 7 on center_2** — so "recall
>= 0.90" actually binds at 0.909 and **1.000** respectively. On center_2 the official estimator
is effectively demanding *perfect* recall, a stricter operating point than the metric's stated
90%. That is why per-centre PPV has been so unstable here (weighted 0.8684 against resampled
0.8750 with CI [0.0268, 1.0000]), and it is a further reason local absolute PPV was never
comparable to a leaderboard number. It does not bind this way on the hidden set, which is ~20x
larger. `evaluation/metric.py` already carries both estimators and documents the quantisation;
this is the measured instance of it.

### The three-family ensemble (2026-08-11)

Three full-data fold families sharing one backbone and architecture, differing only in
training recipe. Evaluated **per fold**, so the cross-fold score-scale artifact cannot
contaminate it.

| model | mean per-fold FPR@90R | mean per-fold AUPRC |
|---|---|---|
| V2_dinogn alone (the 62nd-place system) | 0.0110 | 0.9446 |
| V3_aug alone | 0.0087 | 0.9516 |
| V4_tail alone | 0.0134 | 0.9404 |
| **ensemble 0.45 / 0.30 / 0.25** | **0.0067** | **0.9543** |

The ensemble cuts FPR@90R by 39% against the shipped system and beats every individual
member, including two that are individually worse than V2_dinogn. Not unanimous: three
folds improve, fold 0 worsens (0.0067 → 0.0117), fold 4 ties at zero and is uninformative.
All of this is in-distribution, where FPR@90R is ~1% and the leaderboard's is 60%.

Construction decisions worth preserving:

- **Weighted logit average, NOT percentile-rank fusion.** Ranks would be computed within
  whatever stack the container is handed, while the metric is pooled over the entire test
  set — so per-stack normalisation makes every prediction depend on its batch's composition.
  Keeping one backbone family makes logit scales comparable and removes the need entirely.
- **0.45/0.30/0.25 kept even though equal weighting scored marginally better** (0.0064 vs
  0.0067, AUPRC 0.9546 vs 0.9543). That gap is noise across five folds, and adopting
  whatever won on the validation data is precisely what put +0.12 of fiction into `fuse.py`.
  V2_dinogn leads because it holds the only external leaderboard evidence — a reason that
  exists independently of this table.
- The families agree on **~75% of the lowest-decile positives** (the images that set the
  threshold), so the gain comes from the 25% that differ. Real, but bounded.

**Closed on a predeclared test: V5_cb** (`--loss-balance centre`, intended to remove the
centre-prevalence shortcut — centre alone predicts the label at AUROC 0.685). Added as a
fourth family at 0.36/0.24/0.20/0.20, mean per-fold FPR@90R was **identical at 0.0067**;
AUPRC moved +0.0013, noise. `cb` alone is the weakest family (FPR@90R 0.0177, AUPRC 0.9391).
There is a plausible unfalsifiable story that it removes a shortcut which only helps
in-distribution and would pay off on unseen centres — that story has the same shape as the
reasoning that kept the harmful ResNet fusion alive for weeks. Question asked, answer no,
arm closed.

---

### Why the validator never could have worked (2026-08-26)

Four measurements made after submission 4 came back. Together they explain the plateau
better than "we ran out of levers", and they change how every FPR@90R number above should be
read.

**1. AUROC and the scored metric are decoupled.** Submission 4 was scored on two external sets
at once — the same container, so this is controlled:

| set | AUROC | PPV@90R |
|---|---|---|
| RARE26 main | 0.8195 | 0.0150 |
| Validation RARE25 | **0.8889** | **0.0155** |

**+0.069 AUROC bought +0.0005 PPV.** Every remaining lever in §7 is an average-discrimination
lever. This says average discrimination no longer converts.

**2. The ROC shape differs between local and leaderboard, and the sign flips.** Against an
equal-variance binormal model, FPR@90R implied by AUROC versus actual:

| | AUROC | implied | actual | gap |
|---|---|---|---|---|
| LOCO center_1 (domain) | 0.9840 | 4.0% | 3.6% | −0.4 pp |
| LOCO center_2 (domain) | 0.9916 | 1.8% | 0.4% | −1.4 pp |
| LOCO, jigsaw arm | 0.982–0.987 | 3.0–4.5% | 0.0–1.4% | −3.0 pp |
| EVC | **1.0000** | — | — | — |
| leaderboard main | 0.8195 | 49.6% | 59.1% | **+9.5 pp** |
| Validation RARE25 | 0.8889 | 32.8% | 57.2% | **+24.3 pp** |

Locally these models *beat* the FPR@90R their own AUROC implies; externally they lose 9.5 to
24.3 points to it. **The entire `shift_stress` acquisition sweep bottoms out at AUROC 0.9669**
— every gamma, white-balance and vignette severity — against the board's 0.8195. No local
configuration available reaches within 0.15 AUROC of the regime being scored, and none
reproduces the external ROC shape at all. Most of the levers in §3 were therefore A/B'd on a
validator in which the failure mode does not exist. They are untested, not refuted.
*Caveat (external review, and it is correct): AUROC plus one ROC point proves deviation from the
equal-variance binormal, not specifically a heavy positive tail. A multimodal negative
population of hard mimics fits the same numbers, and since only ordering matters, demoting
mimic negatives is exactly as valid a route as rescuing tail positives.*

**3. There is no irreducible hard-positive core, locally.** Bottom-decile positives across
seven genuinely different representations (domain DINOv2, stock DINOv2, stock DINOv3, MaxViT,
jigsaw, SWSL RN50, GastroNet RN50):

| | center_1 | center_2 |
|---|---|---|
| in **every** arm's bottom decile | 1 of 61 (1.6%) | 0 of 97 (0.0%) |
| in at least one arm's bottom decile | 18 (29.5%) | 27 (27.8%) |
| buried by exactly one arm | 8 | 9 |

Different representations bury different positives — though ~2x more concordantly than chance
(union 18 against ~32 expected under independence; 27 against ~52). Diversity is real but
half-spent.

**4. FPR@90R is one image, and most tail deltas in this log are noise.** Mean versus noisy-OR
pooling of the same two LOCO arms:

| held-out | Spearman | FPR@90R mean | FPR@90R noisy-OR | images moving >1% of list |
|---|---|---|---|---|
| center_1 | **0.99997** | 0.0221 | **0.0090** | 27 of 2,279 |
| center_2 | 0.99977 | 0.0014 | 0.0014 | 33 of 816 |

**A rank correlation of 0.99997 produces a 2.5x difference in FPR@90R.** At 90% recall the
threshold is set by a single image — the 10th-percentile positive, one image at n+=61, about
the tenth on the dev board. Move it and most of the negative distribution crosses with it.

Consequences, and they are large:

- **The 22.6% FPR@90R cut that justified prior shift (§3) was a handful of images**, not an
  effect. So is most of the FPR@90R column throughout this log. Deltas on this metric at
  n+ = 61–97 should be treated as uninformative unless they survive a group bootstrap.
- **A single development submission cannot distinguish two configurations.** 0.0148 / 0.0151 /
  0.0150 are three draws from one noisy process, exactly as the CI [0.0108, 0.0746] implies.
  Do not spend a slot to A/B anything at this resolution.
- **The closed phase is a different measurement, not just a bigger one.** With ~3,200 positives
  the threshold is set by roughly the 320th positive rather than the 10th — about 5–6x less
  noise. Differences invisible here can appear there, which is the argument for choosing the
  final container on mechanism and predeclaration rather than on any development number.

### Container reverted to plain probability averaging (2026-08-26)

`inference.py` now ships `prob_nat`: per-model Platt, mean within arm, equal-weight mean across
arms. Prior shift and noisy-OR were removed — externally they cost 0.024 AUPRC with no tail
gain, and finding 4 above shows the local evidence that justified them was an order statistic
twitching. `prob_nat` is the exact configuration that passed the predeclared two-arm LOCO gate
(center_1 AUPRC 0.9012, all six deltas positive) before any tail machinery was layered on it.
Independently recommended by an external review, which advised the simplest proven representation-diverse
stack with plain fixed fusion and explicitly no noisy-OR, conditional normalisation or learned
gating, selected on cross-centre stability rather than best local score.

Built and verified 2026-08-26 under `--network none --gpus all`: image
`sha256:6571524e410b3e59a0e7af1b66b433495c2f7714c71439def0f74a8af88d376c`, tag
`rare26-twoarm-probavg`, tarball `twoarm_10model_probavg.tar.gz` at 6,357,006,153 bytes.
`platt.json` is unchanged and still carries the third (prior_fit) element, now unread.

**Unresolved and worth stating plainly:** submission 3 (15 models, one backbone, three recipes)
scored equal or better than submission 4 on all three external metrics — 0.0151/0.8200/0.3426
against 0.0150/0.8195/0.3187 — which points the opposite way from LOCO, where two-arm fusion
was positive on all six deltas. None of it is significant, and the pooling change contributes
almost nothing to ordering, so the AUPRC difference is attributable to ensemble composition
rather than to pooling. Two-arm remains the pick on mechanism; the external evidence mildly
disagrees. That tension is not resolvable with the instruments available.

---

## 4. Protocol rules, learned the hard way

**Use LOCO, not pooled OOF.** Pooled or per-fold cross-validation has now misled **four**
times: it argued for keeping the harmful ResNet fusion, it hid the centre-prevalence
shortcut, its PPV estimate is noise-dominated (per-fold mean 0.36 vs pooled 0.12 for the same
model — a cross-fold score-scale artifact that does not exist at deployment), and on
2026-08-17 it called positive-enriched batching a 39% FPR cut that LOCO scored at zero.
Per-fold CV is *not* a weaker version of LOCO; it answers a different question.

**A wrong protocol can be wrong by 0.33 AUPRC.** The ResNet arm was rejected on a
full-fine-tune LOCO result of 0.586 on center_2; a linear probe on the *same frozen
backbone* gets 0.918. We rejected the arm on a training mistake.

**Judge on AUPRC, not PPV@90R — but never *veto* on AUPRC alone.** At 61 and 97 positives
PPV straddles zero even for real improvements, which is why AUPRC is the decidable signal.
**Revised 2026-08-18:** using AUPRC as a veto was wrong and cost a real result. The ranking
metric is PPV@90R, a strictly monotone function of FPR@90R; AUROC and AUPRC are supplementary
diagnostics that do not affect ranking. A tail intervention will often cut FPR@90R while
costing a little AUPRC, because AUPRC integrates across all recalls and is nearly blind to
the extreme tail — that is the expected signature, not a contradiction (prior-shift, §3).
Report exact FPR@90R and mean FPR over recalls 0.80–0.95 (the stabler statistic) alongside
AUPRC, and accept a wide CI on the right metric over a tight CI on the wrong one. Exception
unchanged: a change targeting the operating point directly is judged on the operating point.

**One predeclared A/B at a time.** Selecting a hyperparameter on pooled PPV and reporting
it there produced +0.12 of pure optimism in `fuse.py`. A broad trend across many values
(the LoRA sweep, the fusion-weight sweep) is evidence; a single value that wins is not.

**Restricting trainable capacity helps — but it has an optimum.** LoRA-6 beats LoRA-12;
augmentation hurts a frozen encoder; full fine-tuning is catastrophic. The rule used to
read "helped every single time"; the ResNet freeze sweep (§3) qualified it. That curve is
an inverted U — `layer4+fc` beats both the full fine-tune *and* the linear head — so less
is not monotonically better. Treat a proposal that adds trainable capacity as unlikely to
survive centre transfer, and one that strips almost all of it as equally suspect.

**Revised again 2026-08-19: the curve's shape is architecture-specific, so measure it.** MaxViT
is *monotonic* in capacity here — frozen 0.177, stages.3+head 0.413, full fine-tune 0.560 —
the opposite of the ResNet's inverted U, and stock DINOv3 gained almost nothing from a 5.3×
adapter increase. There is no general rule to apply; there is only the measurement. What does
generalise is the procedural lesson: **never close an arm on a single point of that curve.**
The ResNet arm was rejected on its full fine-tune and was worth +0.36 AUPRC when re-measured;
MaxViT was re-measured the same way and genuinely lost at all three points. Both took one
extra run to establish, and in one of the two cases it changed the answer completely.

---

## 5. Traps that cost time

- **`folds_v1.csv` and `folds_v2.csv` are different splits, not nested.** v1 = 3095 imgs /
  158 pos / 2 centres (the official RARE25 training set). v2 = 3195 / 208 / adds `evc`.
  **2524 of 3095 shared images sit in a different fold.** Nothing transfers across them.
  The whole W1 wave is on v1; everything from V2_* onward is on v2.
- **`--batch-size` defaults to 32; W1's resnet512 used 16.** Omitting it doubled activation
  memory, spilled 1.6 GB to system RAM over PCIe, and ran 2.3–12× slower with no OOM.
  Check `(Get-Counter '\GPU Process Memory(*)\Non Local Usage')` — it must read 0 MB.
  **Measured again on MaxViT-384, 2026-08-19, and the penalty is far worse than 12×:**
  batch 32 peaks at 15.43 GB on an 8.5 GB card and runs at **1.5 img/s**; batch 16 at 7.95 GB
  and 13.2 img/s; batch 8 at 4.23 GB and **47.3 img/s**. That is a **32× spread**, silent, with
  no OOM — the difference between a 20-minute run and a 13-hour one. Measure peak VRAM with a
  three-step forward/backward probe before launching any run on a new architecture; it costs a
  minute and this trap has now fired twice.
- **EVC is degenerate as an evaluation centre.** 50% prevalence, and *both* stock and
  domain models score ~1.0 AUROC on it without training on it. It cannot discriminate
  anything; it also raised the centre-prevalence shortcut from AUROC 0.685 to 0.750.
  Keep it out of headline metrics.
- **Windows DataLoader needs `if __name__ == "__main__"`.** Without it, `num_workers>0`
  re-imports the module in every worker and the job hangs silently.
- **PowerShell:** paste loops on one line with `;` between statements, match your braces,
  and note `Tee-Object` buffers Python stdout — use `python -u` to see live progress.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is Linux-only.** On Windows it emits a
  stderr warning that PowerShell renders as a red `NativeCommandError`; it is harmless.
- **`torch.cuda.is_available()` is not proof the GPU works.** `pytorch/pytorch:latest` is
  stale at torch 2.2.1, whose kernels stop at sm_90; this machine's RTX 5060 is Blackwell
  (**sm_120**). `is_available()` still returns **True** — the failure appears only on the
  first real allocation, as `no kernel image is available for execution on the device`.
  Any GPU check that does not run an actual tensor op is a false negative waiting to
  happen. Container base is pinned to `2.11.0-cuda12.8-cudnn9-runtime`: matches the torch
  that wrote the checkpoints, carries sm_70–sm_120, 4.3 GB against `latest`'s 11.4 GB.
- **Never list `torch`, `torchvision` or `numpy` in the container `requirements.txt`.**
  They ship in the base image; pip resolves them against PyPI and installs a `--user` copy
  that shadows the base image's CUDA build, silently disabling the GPU. Verified: with
  them removed, pip installs `timm` and touches neither torch nor numpy.
- **The challenge template's `do_*.sh` scripts arrive with CRLF line endings** and die
  under bash with `$'\r': command not found`. Converted to LF on 2026-08-10.
- **Git Bash mangles container paths, so `do_test_run.sh` cannot be used.** It fails with
  `exec: "C:/Program Files/Git/usr/bin/sh": no such file or directory` — MSYS rewrites
  `--entrypoint /bin/sh` into a Windows path, and does the same to `--volume ...:/tmp` and
  `:/input:ro`. Patching one line only moves the failure. **`MSYS_NO_PATHCONV=1 ./do_test_run.sh` is not the
  fix and fails differently** (tried 2026-08-25): it also suppresses conversion of the script's
  own `pwd`-derived build context, so `docker build` is handed a `/c/Users/...` path and reports
  `unable to prepare context: path not found`. The env var is needed for *container-side* paths
  and harmful for *host-side* ones, and the script mixes both. What works from Git Bash is to
  skip the script and run the forward pass directly, env var set and host paths written
  Windows-style so nothing needs converting:
  `R="C:/path/to/repo/submission_template"   # Windows-style, forward slashes; MSYS_NO_PATHCONV=1 docker run --rm
  --platform=linux/amd64 --network none --gpus all -v "$R/test/input/interface_0:/input:ro"
  -v "$R/test/output/interface_0:/output" -v "<tag>-volume:/tmp" <tag>`.
  `./do_build.sh` on its own is fine and needs no env var. PowerShell also works.
- **The pinned base uses Debian system Python carrying a PEP 668 marker**, so
  `pip install --user` fails at build time with `externally-managed-environment`. The
  Dockerfile passes `--break-system-packages`; nothing is at risk because requirements.txt
  deliberately lists no package the base image already provides.
- **Windows dataloader workers pickle the Dataset**, so every transform it holds must be a
  module-level class. Closures and lambdas die with
  `Can't get local object '<fn>.<locals>.<lambda>'`. This is why `preprocessing/pipelines.py`
  uses classes, and why `evaluation/shift_stress.py` had to be rewritten to match.
- **dHash is degenerate on endoscopy frames.** In the near-duplicate audit, 64-bit dHash at
  any threshold loose enough to find anything also fired *across centres* (25 pairs at ≤2,
  2495 at ≤6), which a genuine duplicate cannot do, and single-linkage then chained 1618 of
  3195 images into one component. pHash, md5 and embedding cosine all agree with each other;
  dHash agrees with none of them. Use pHash, and always report cross-centre pair counts as
  the sanity check — they should be ~0 for any honest duplicate detector.
- **Checkpoint args predate later flags.** Runs written before `--lora-blocks` existed have
  no such key and `cfg["lora_blocks"]` raises KeyError; use `cfg.get("lora_blocks", 6)`,
  which was the behaviour then. A wrong guess cannot pass silently — the LoRA tensor count
  would differ and `strict=True` rejects it.

---

## 6. Tools built this session

| file | purpose |
|---|---|
| `evaluation/compare_loco.py` | paired group-clustered A/B of two LOCO runs — **the standard test** |
| `evaluation/feature_bank.py` | extract frozen features from any backbone (auto-detects ViT vs RN50) |
| `evaluation/probe_features.py` | centre identifiability + LOCO label probe from a feature bank |
| `evaluation/ensemble_loco.py` | ensemble + Platt calibration + fixed-threshold transfer |
| `evaluation/fuse_loco.py` | validate a **predeclared** fusion weight on a held-out centre |
| `evaluation/shift_stress.py` | score a LOCO model under 36 graded acquisition corruptions, reporting FPR@90R per family — the probe that refuted the acquisition-shift hypothesis |
| `evaluation/dup_audit.py` | md5 + pHash/dHash + embedding near-duplicate audit; caches hashes to `dup_hashes.npz`, writes `dup_pairs.csv`. The probe that cleared the validator (§3) |
| `training/train_seg_decoder.py` | segmentation decoder on frozen DINOv2 patch tokens, trained from an external mask manifest (§3, the localization route) |
| `evaluation/seg_score.py` | applies that decoder to RARE zero-shot, top-k patch pooling, writes per-centre predictions the paired-comparison tooling reads unchanged |
| `data_split_scripts/build_edd_manifest.py` | composes EDD2020 per-class masks into a lesion manifest; the lesion class set is an explicit argument because it *is* the experiment |

Flags added 2026-08-17 to **both** `train_dinov2.py` and `train_cross_centre.py`, all
defaulting to previous behaviour, all now measured null (§3): `--pos-per-batch` (fixed-
composition batches via `PosEnrichedBatches`; also switches BCE to `class_balanced_bce`,
since `pos_weight` would double-count the enrichment), `--select {auprc,spec}`
(`mean_specificity` over recall 0.80–0.95), `--ckpt-avg` (averages trainable weights over
best epoch ± N; only LoRA + head, ~2 MB per snapshot), `--tail-pos-skip` and `--tail-warmup`.
`--tail-pos-skip 0.0` is bit-identical to what the V4_tail runs ran. One real bug fixed: a
NaN selection statistic left no checkpoint on disk and the run died at the end with a
`FileNotFoundError`; epoch 0 now always writes one.

Earlier flags, all defaulting to previous behaviour: `--init` (`train_resnet.py`,
`train_dinov2.py`, `train_cross_centre.py`), `--lora-blocks` and `--tail-lambda`
(`train_cross_centre.py`, `train_dinov2.py`), `--split-seed` (`train_cross_centre.py`),
`--freeze-until` (`train_cross_centre.py`, helpers in `train_resnet.py`). The last freezes
the ResNet stem and stages up to a named cut, holding those stages in `eval()` so their
BatchNorm running statistics stop tracking the training centres — a trunk whose BN still
drifts is not frozen. Re-applied after every `model.train()`, like `freeze_batchnorm`.
All loaders assert matched key counts — `strict=False` silently yields a random backbone.

Flags added 2026-08-18, all defaulting to previous behaviour, all measured in §3:
`--aug jigsaw` (`Jigsaw` in `preprocessing/pipelines.py`, 3×3 tile shuffle — the one that
survived), `--mil {topk,lse}` with `--mil-k`, and `--ppv-lambda` / `--ppv-q` /
`--ppv-margin` for the PPV@90R surrogate.

Added 2026-08-19, all defaulting to previous behaviour, all measured in §3. The default path
was verified to still build exactly 0.591M trainable parameters on DINOv3-L, so nothing
already measured is disturbed:

- `--lora-all-linear` and `--lora-dropout` (both trainers) — adapts `mlp.fc1/fc2` as well as
  attention, with dropout on the adapter input only. `--lora-rank` / `--lora-alpha` are now
  wired through `train_cross_centre.py` as well; they were previously hardcoded to 16/32 there.
- `--aug heavy` — IMSY's augmentation floor plus jigsaw. Optical distortion and elastic are
  omitted: they need `albumentations`, which is not installed, and adding a container
  dependency is a separate risk.
- `train_cross_centre.py --arch` now accepts `maxvit`, `maxvit384` and `resnet512`; input size
  and lr come from `train_resnet.ARCH` rather than a shared default, because MaxViT window
  sizes are tied to the input resolution.
- `freeze_until` handles timm stage-style trunks (`stem`, `stages.0..3`), and
  `apply_frozen_eval` now matches nested module names so a frozen MaxViT trunk keeps its
  normalisation statistics fixed.
- `feature_bank.py --init timm:<model_name>` screens any stock timm checkpoint as a frozen
  extractor — the confound-free way to separate a weak representation from a bad training
  regime (§4).

Feature banks on disk: `feats_stock.npz`, `feats_domain.npz`, `feats_rn50_5m.npz`,
`feats_rn50_swsl.npz` (3195 × 1536 or 2048, fp16), plus the five GastroNet-variant banks
from the 2026-08-17 screen. A probe takes seconds.

**Checkpoint cleanup, 2026-08-11.** 98 `.pt` files (21.0 GB) were deleted from closed arms.
**Only 17 checkpoints remain**: `V2_dinogn_fold0-4`, `V3_aug_fold0-4`, `V4_tail_fold0-4`
(the shipping ensemble, also staged in `submission_template/resources/`) plus
`LOCO_dinov2_rest_to_center_{1,2}_domain` (needed by `shift_stress.py`). Every
`results.json`, `log.csv` and prediction CSV was preserved — 11 MB covering all 133 runs —
so no conclusion in this log has lost its evidence, but any retired arm must be retrained
before it can be run again.

**As of 2026-08-19 that has grown back to 54 checkpoints / 18 GB**, and a second cleanup is
due. Worth keeping: the six full-data fold families `V2_dinogn`, `V3_aug`, `V4_tail`,
`V6_pos`, `V7_swsl`, `V8_jig` (5 each), and the LOCO pair
`LOCO_dinov2_rest_to_center_{1,2}_domain` plus `LOCO_rest_to_center_{1,2}_rn50swsl_layer3`,
which are the two arms every fusion result in §3 is computed from. The `dinov3l` pair is
also worth keeping until the overlap probe (§7) is decided — it is the probe's payload.
Everything else (`r224`, `r448`, `p3_p3`, `posenr*`, `mil`, `ppv`, the two smokes) is a
closed arm holding ~10 GB.

### Standard workflow

```powershell
# one predeclared change, LOCO, both centres
foreach ($h in @("center_1","center_2")) { .\.venv\Scripts\python.exe -u training\train_cross_centre.py --folds folds_v2.csv --arch dinov2 --holdout $h --init dinov2.pth <CHANGE> --tag mytag 2>&1 | Tee-Object -Append logs\run.log }

foreach ($h in @("center_1","center_2")) { .\.venv\Scripts\python.exe evaluation\compare_loco.py --baseline "runs/LOCO_dinov2_rest_to_${h}_domain" --candidate "runs/LOCO_dinov2_rest_to_${h}_mytag" --folds folds_v2.csv }
```

---

## 7. Open items, ranked

1. **The Aug 27 deliverables — now the highest-value work left.** Missing these scores zero
   regardless of rank, **and the closed phase — the one that decides the ranking (§8) — cannot
   be entered without them.** In the predecessor 20 teams entered development and only 11
   completed the closed phase, so this is where roughly half the field eliminates itself; it
   is worth more expected rank than any remaining modelling lever.
   Done 2026-08-19: `LICENSE` (MIT, covering code only), `docs/MODEL_PROVENANCE.md` with full
   sha256 for all eight backbone weights and all ten shipped checkpoints — independently
   matching the `checkpoint_hash` recorded in each run's OOF CSV — and a `.gitignore` audited
   to exclude data, `runs/`, `*.pth`, `*.npz` and the staged container weights.
   Done 2026-08-25: README with full reproduction instructions; the `LICENSE` copyright holder
   filled in as `Raihan (FAU_PRLab)`; the container source published — `submission_template/`
   was a clone of the organizers' template (`TUE-ARIA/RARE25-Submission`) carrying one upstream
   commit with every one of our changes uncommitted inside it, so git could only ever track it
   as a pointer to a repo we do not control; the nested clone was detached to
   `.git-upstream-template-clone/` and the source is now tracked directly, weights excluded.
   **Initial commit `136c3dc`** — 51 files, 333 KB, audited for absolute paths, secrets, model
   weights and challenge data before committing.
   **Flagged 2026-08-25:** that upstream template is **CC BY-NC 4.0**, not MIT. Our additions
   are MIT and the subdirectory retains its own `LICENSE`; the README states the carve-out
   explicitly. Confirm against the rules page that a CC BY-NC subdirectory inside an otherwise
   MIT repository satisfies the code deliverable — it is the organizers' own template, so it
   ought to, but that is their call and not one to discover late.
   **Still outstanding:** the GastroNet licence text (the Theta listing page does not render
   for automated fetching — it must be copied by hand); **publishing the repository** (no
   remote is configured and `gh` is not installed, so the organizers currently cannot reach
   it); the final container digest; and the 2–3 page paper — `report/RARE26_report.pdf` (2026-08-18)
   already names the two-arm fusion as the final system and carries a *pending* row for it
   dated Aug 24, so it needs that row filled with the real result, the rank figures refreshed
   (§1), and the container digest added; it is a revision, not a rewrite.
2. **Decide and ship a container.** The two-arm system is staged and unbuilt (§1). The
   three-arm variant needs `V8_jig` staged as `jig_fold*.pt`, `platt.json` extended to
   fifteen entries fitted on that family's OOF, and one more tuple in `ARMS`. Either way:
   build, verify under `--network none --gpus all`, `docker save | gzip`, upload. This is the
   only item with a hard external deadline — the final container must be dry-run before
   1 September (§8).
3. **The overlap probe — the highest-information experiment available.** Submit the stock
   DINOv3 model already trained (`LOCO_dinov2_rest_to_center_*_dinov3l`) and read its **dev
   AUROC** against the GastroNet system's 0.820. If a deliberately under-configured stock
   backbone that LOCO scores 0.25 AUPRC worse lands within ~0.02, the local validator is
   measuring GastroNet exposure rather than generalisation and the arm-selection logic behind
   the entire system is unsafe. Either answer is decisive. Read AUROC, not PPV@90R — AUROC
   uses every image, PPV@90R is a tail order statistic whose CI spans sixfold.
4. **Prior-shift + noisy-OR into the inference path** (§3). Free at inference, no retraining,
   and metric-aligned. The clean version is a controlled external A/B: the same checkpoints
   submitted twice, once with the raw equal-weight probability average and once with
   per-member prior shift plus noisy-OR. Gated on the submission-budget question (§8).
5. **Augmentation arms, screened as fusion members rather than replacements.** Jigsaw is the
   only lever that has moved since the backbone and its value is error diversity, so the
   remaining winner presets — optical/elastic distortion, hide-and-seek / coarse dropout —
   should be judged by what they add to the fusion, not by their standalone LOCO score. A
   candidate may be worse alone and still earn its place.
6. ~~**GastroNet-5M DINOv3-B/16.**~~ **CLOSED 2026-08-19 — it is not obtainable.** The
   HuggingFace repo `tgwboers/GastroNet-5M_Pretrained_Weights` contains exactly two files,
   `.gitattributes` and `README.md` (verified against the HF API; last modified January 2026),
   no public DINOv3 endoscopy checkpoint turns up in search, and the Theta portal does not list
   one. REVEAL's reported 0.835 vs 0.818 Barrett's AUROC is not actionable without the weights.
   **Stop waiting on this**; it had been carried as blocked across several sessions.
   This empties the shelf: "change the frozen representation" is the only class of change that
   has ever worked here, and there is no longer a downloadable member of it. Anything further
   in that class has to be built.
7. ~~**All-linear LoRA at r32/α64/dropout 0.1.**~~ **Done 2026-08-19, negative (§3).**
8. **Hard-example reweighting.** Unchanged, still blocked on human inspection.
   `hard_positives.csv` has 20 positives controlling the operating point, `hard_negatives.csv`
   39 persistent false positives; montages at `hard_positives_top20.png` /
   `hard_negatives_top20.png`. Five positives score <0.005 from *both* families — likely
   label or visibility problems, and upweighting those is how E6 was destroyed previously.
   Mark `selected_for_reweighting`, then `evaluation/make_weights.py` → `--weights`, keeping
   weights at 2.0–3.0 and never stacking with `--loss-balance centre`.
9. **Global hard-tail mining** — a memory bank of the top 2–5% negatives and bottom ~20%
   positives scored over the whole fold, with a pairwise logistic loss mixed into BCE rather
   than replacing it. This is the non-batch-local version of the tail loss that was null; both
   external reviews rank it below everything above.
10. **Anomaly / one-class detection on the RARE26 unlabeled release.** RARE25 organizers noted
    *no* team attempted it and then released a large unlabeled corpus — a telegraphed hint,
    and the one resource that distinguishes RARE26 from RARE25. Genuinely unexplored, a
    research bet rather than a ranking fix, and not a two-week project on top of the above.

**Closed since the last revision:** the near-duplicate / group audit (§3), which both external
reviews ranked P0. It came back clean, so no earlier conclusion in this log needs revisiting
on power grounds.

**Do not revisit:** seed ensembling, TTA, threshold calibration, deeper LoRA, **resolution in
either direction** (224px measured worse on center_1 with the CI excluding zero, §3; 336 is
the tested optimum, and higher adds capacity against the one pattern that holds), fusing a
fine-tuned ResNet, **the ResNet arm in any form** (restricted adaptation was its best case and
still lost on both centres; frozen-probe complementarity among *weak* learners does not
survive once LoRA has added +0.176 AUPRC), **augmentation as a lever in itself** (mechanism
refuted, §3 — though V3_aug remains valuable as an *ensemble member*), **`--loss-balance
centre`**, **positive-enriched batching and the capped tail loss**, **metric-aligned
checkpoint selection**, **making scores centre-invariant** (drift refuted, §3), **448px**,
**patch-token MIL**, **the PPV@90R surrogate loss**, and **model count for its own sake** —
the winner's 40 models are diversity of preset and architecture, not of count.

Added to the list 2026-08-19: **stock DINOv3 in any adapter configuration**, **MaxViT in any
adaptation regime**, **the heavy augmentation floor**, **lr 1e-4 / longer schedules**, and
**segmentation pretraining as a fusion arm**, and **acquisition-conditional score
normalisation** — the last of which fails for a reason that generalises to any within-stratum
rescaling here: the strata are not conditionally independent of the label (§3). The caveat that
previously kept stock DINOv3
open — that our adapter config was tuned for a different architecture — was tested and refuted
(§3). The localization route is closed on a *working* decoder (val Dice 0.667), not a broken
one, which is the strongest form of that verdict available.

**Twenty-seven levers tested. Three have ever worked**: the domain backbone (§2), two-arm
representation-diverse fusion (§3), and jigsaw as a third arm (§3). All three change what the
model represents; nothing that changed only how it is trained has ever survived. The most
recent of the three is 2026-08-18, and the eight experiments since have all failed to beat it —
three transplanted from the RARE25 winner's own code, one from the third-place team's.

**The experimental search is finished.** This is not fatigue: the levers were tried, the
external recipes were transplanted, the last unexplored podium approach was built and measured,
and the only class of change that has ever worked — a different frozen representation — has no
remaining member that can be obtained (GastroNet DINOv3-B does not exist publicly, §7) or built
in the time available. Anyone continuing should read §7 item 3 first: the single open question
is external, not local, and no further local experiment can resolve it.

---

## 8. Honest caveats

- **The development leaderboard does not decide the ranking.** Ranking and prizes come from
  the Closed Testing Phase — **1–7 September, one submission**, on a far larger set. In
  RARE25 the development cohort was 1,040 NDBE + 103 neoplasia against a test cohort of
  23,176 + 3,232 across 12 BONS-AI centres, and the organizers state plainly that the open
  validation set is small and not advised for final model selection. **Every rank quoted in
  §1 is a rank on the set that decides nothing.**
- **Rank 72 is not a verdict.** RARE25's second-place team reported development AUROC 0.7709
  and PPV@90R 0.0112 — worse than our 0.820 / 0.0151 — and finished 2nd overall with the best
  AUPRC on the hidden test. Another team posted development numbers essentially identical to ours
  and finished mid-pack. The field also thins hard: 20 teams in development, 11 completing the
  closed phase, which requires a 2–3 page paper and MIT-licensed code (§7 item 1).
- **The invisible-positive model is refuted (2026-08-19).** If a fraction f = 0.242 of
  positives were intrinsically invisible, no model could exceed AUROC 0.758·1 + 0.242·0.5 =
  **0.879**. The current board holds **0.8949**, and an eligible external team 0.8928, so
  those positives contain usable signal — they are hard *for our representation*, not
  invisible. Solving the two constraints jointly rather than assuming 0.98 gives f = 0.242
  with visible-positive AUROC **0.922**, so a substantial share of the gap is an ordinary
  cross-centre generalisation gap — which backbones, augmentation and loss design do act on.
  This supersedes the §3 reading that fourteen null levers were all aimed at the wrong
  mechanism, and it is consistent with jigsaw being the one thing that has moved since.
- **The submission budget is contradicted and unresolved.** This log has recorded one
  submission per week with no roll-over; the 2026-08-18 external review states roughly twelve
  submissions remain before 31 August. Those imply opposite strategies — scarce slots argue
  for banking, plentiful slots make submissions cheap external experiments (§7 items 3 and 4
  both assume the latter). **Check the rules page before planning around either.**
- **Dates.** Methodology lock 20 August, report and MIT code 27 August, container freeze
  30 August, closed phase 1–7 September taking one submission. The ICDAR presentation in
  Vienna is 3 September, so the final container must be built and dry-run **before**
  1 September, not during it.
- **Local numbers are not merely optimistic — they are a different regime, and this is now
  measured rather than suspected.** See §1's calibration table: FPR@90R is 17–100× worse on
  the leaderboard than on a held-out centre. This bullet previously read "local numbers are
  optimistic", inferred from the RARE25 winner's 0.822; the true gap is far larger than that
  comparison implied. One held-out centre is easier than twelve heterogeneous ones, and
  pretraining overlap is still not excluded — GastroNet's corpus is 8 Dutch hospitals and
  RARE's centres are Dutch centres from the same consortium (TU/e ARIA lab + Amsterdam UMC
  organise both challenges). A centre-identifiability probe was **inconclusive**: stock
  DINOv2 also fingerprints centres at 99.4%, so near-perfect centre recovery indicates
  acquisition shift, not contamination. Acquisition shift as the *explanation* for the
  leaderboard gap is now refuted outright (§3), leaving semantic case-mix shift in the
  positive tail as the leading account.
- **Score scales differ.** RARE25's published range 0.015–0.320 is at the test set's
  natural ~12.2% prevalence, **not** at simulated 100:1. Do not compare it to our
  Mode-A numbers.
- **RARE25 podium, read from source (2026-08-19).** Previously approximate; these are now
  verified against the published code at `IMSY-DKFZ/rare2025-challenge`.
  **1st, IMSY (0.320 at the test set's natural ~12% prevalence — NOT comparable to our
  Mode-A 100:1 numbers):** LoRA on *every* `nn.Linear` in a stock DINOv3 ViT-L/16 by recursive
  traversal, **r32 / α64 / dropout 0.1**, head trainable and base frozen; paired with a
  GastroNet ResNet50. **40 checkpoints** = 20 ResNet + 20 DINOv3, each 4 augmentation presets ×
  5 folds. Pooling is **noisy-OR**, `1 - prod(1 - p)`. Calibration is `psrcal`
  `AffineCalLogLoss` fitted **per model per fold on out-of-fold predictions** with
  `priors=[100/101, 1/101]`, applied as `(t*preds + b) + log(prior)` then log-softmax, *before*
  the noisy-OR. **batch 32, 50 epochs, lr 1e-4, wd 1e-4.** Resolution 224 ("regular") or 448
  ("large"). Losses: `CE_PPVAtRecallLoss` (λ 0.5, margin 0.1, EMA β 0.99 on the positive
  quantile) and `DifferentiableSurrogateLoss` (−precision + λ·ReLU(recall_target − recall),
  λ 0.1). Their `cv_type` options include `center1`/`center2`, so they ran LOCO-style splits too.
  Every preset sits on a shared floor: RandomResizedCrop(0.8–1.0, ratio 0.75–1.33), H/V flip,
  rotate ±15° p 0.8, ColorJitter(0.2/0.2/0.2, hue 0.1) p 0.8. Preset `top1` adds jigsaw, blur,
  colour and optical distortion.
  **2nd, UT:** MaxViT-Tiny at 384px, 5-fold CV, PyTorch + timm, CV AUC 0.9457, **dev AUC
  0.7709 — worse than our 0.820 — and they finished 2nd.** Code still withheld at time of
  reading, so the claim that they used no large-scale external pretraining is unverified.
  **3rd, Jmees:** segmentation-driven. Staged pretraining from large polyp segmentation sets →
  Barrett's segmentation on EDD2020 and EVC → cancer segmentation and classification; final
  system pairs a DINOv3 ViT-B with a DPT decoder and a ViT-L pretrained by masked image
  modelling. AUROC 0.9896, AUPRC 0.7408, PPV 0.235.
  **Three of the winner's ingredients were transplanted here and all three lost** (§3). What
  has not been tried is Jmees' localization route, which is the only podium approach we have
  not touched.
- **Rules permit** publicly available datasets and publicly pretrained models; private
  external data is banned. GastroNet weights are public, so their use is compliant.
- **Plan §1 archive record** still needs licence text and access date. Hashes:
  `5688929fea443703` (RN50 SWSL+GastroNet), `20ab749ec4bfdd53` (DINOv2 domain),
  `5ae8e09cea634179` (RN50 GastroNet-5M).
- **Plan deviations to reconcile before the Aug 20 freeze:** MaxViT was trained despite
  being on the exclusion list (it is the worst arm; dropping it resolves this), and
  GastroNet was obtained Aug 3, one day past the plan's Aug 2 gate.
