# Third-party notices

This repository's original code is released under the MIT License (see `LICENSE`). That grant
covers the code written for this project and nothing else. The items below are third-party and
are governed by their own terms.

Nothing listed on this page is redistributed in this repository. Each item must be obtained from
its original source, under that source's own terms.

## Organizer-provided template code

The `submission_template/` directory is derived from the organizers' RARE submission template,
[TUE-ARIA/RARE25-Submission](https://github.com/TUE-ARIA/RARE25-Submission). It retains its
original licence in `submission_template/LICENSE` (CC BY-NC 4.0), which applies to that directory
regardless of anything stated elsewhere in this repository.

## Pretrained weights

The GastroNet family of pretrained backbones, obtained from
[Theta Vision Cortex](https://cortex.thetavision.nl/dataset-provider/listing/2/), are not
redistributed here and retain their original terms.

No separate licence text is published at either distribution point. The Theta listing page does
not expose licence terms in retrievable form (checked 2026-09-07), and the HuggingFace repository
`tgwboers/GastroNet-5M_Pretrained_Weights` carries no licence tag. The weights are released
publicly by their authors for scientific use, which is the basis on which they are used here.
Anyone reusing them should obtain them from the Theta portal directly and satisfy themselves as
to the terms.

Attribution to the originating work:

- *GastroNet-5M: a multicenter dataset for developing foundation models in gastrointestinal
  endoscopy*, Gastroenterology, 2025. PMID 40749857.
  <https://pubmed.ncbi.nlm.nih.gov/40749857/>
- *Foundation models in gastrointestinal endoscopic AI: impact of architecture, pre-training
  approach and data efficiency*, Medical Image Analysis, 2024.
  <https://www.sciencedirect.com/science/article/pii/S1361841524002238>

Per-file sources, access dates and sha256 checksums are recorded in `docs/MODEL_PROVENANCE.md`.

## Datasets

- **RARE25 / RARE26 challenge data.** Not redistributed here. Remains subject to the challenge
  organisers' terms.
- **EndoVis 2015 Barrett's set** (`evc`). Not redistributed here. Retains its original terms.
- **EDD2020** (Endoscopy Disease Detection and Segmentation, EndoCV2020). Not redistributed here.
  Licensed **CC BY-NC-SA 4.0**: attribution, non-commercial, share-alike. Used only to train the
  segmentation decoder in an experiment that measured negative and forms no part of any shipped
  system. Cite Ali et al., *Deep learning for detection and segmentation of artefact and disease
  instances in gastrointestinal endoscopy*, Medical Image Analysis, 2021
  (doi:10.1016/j.media.2021.102002), and Ali et al., *Endoscopy disease detection challenge 2020*,
  arXiv:2003.03376.

No private or non-public external data was used at any point.
