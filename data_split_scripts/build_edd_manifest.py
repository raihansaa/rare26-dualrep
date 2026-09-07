"""Compose EDD2020 per-class masks into a lesion manifest for the segmentation decoder.

EDD2020 ships one mask per class per image under `masks/`, named `<stem>_<class>.tif`,
gray 255 inside the region; `masksPerClass/<CLASS>/` holds the same files split by class.
The five classes are BE, suspicious, HGD, cancer and polyp. 386 frames, 5 institutions.

LICENCE: CC BY-NC-SA 4.0 -- attribution, NON-COMMERCIAL, share-alike. Its use must be
disclosed, and the non-commercial clause is a real constraint on anything downstream.
Record it in docs/MODEL_PROVENANCE.md before any model trained on it is shipped.

WHICH CLASSES COUNT AS LESION IS THE WHOLE EXPERIMENT, so it is an explicit argument
rather than a default buried in code:

  suspicious, HGD, cancer   -> lesion            (the default here)
  BE                        -> NOT lesion        (non-dysplastic Barrett's is RARE's
                                                  negative class; marking it as lesion
                                                  would train the decoder to fire on
                                                  exactly what we need it to ignore)
  polyp                     -> NOT lesion        (different pathology, different organ
                                                  context; 228 annotations that would
                                                  teach an irrelevant appearance)

Images with no lesion mask are KEPT, with an all-zero mask. They are not useless
negatives -- most of them are non-dysplastic Barrett's, which is precisely the
discrimination the decoder has to learn. Dropping them would train a model that has
only ever seen disease.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ("BE", "suspicious", "HGD", "cancer", "polyp")
# `_mask` is optional: the GitHub README documents <stem>_<class>_mask.tif, the released
# archive actually ships <stem>_<class>.tif. Accept both rather than trusting either.
MASK_RE = re.compile(r"^(?P<stem>.+?)_(?P<cls>%s)(_mask)?\.(tif|tiff|png)$" % "|".join(CLASSES),
                     re.IGNORECASE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="EDD2020 directory containing img/ and masks/")
    p.add_argument("--images", default="originalImages")
    p.add_argument("--masks", default="masks")
    p.add_argument("--lesion", nargs="+", default=["suspicious", "HGD", "cancer"],
                   choices=list(CLASSES), help="classes treated as lesion")
    p.add_argument("--out-masks", default="edd_lesion_masks")
    p.add_argument("--out", default="edd_manifest.csv")
    a = p.parse_args()

    img_dir = os.path.join(a.root, a.images)
    msk_dir = os.path.join(a.root, a.masks)
    for d in (img_dir, msk_dir):
        assert os.path.isdir(d), "not a directory: %s" % d
    os.makedirs(a.out_masks, exist_ok=True)

    lesion = {c.lower() for c in a.lesion}
    by_stem = {}
    unmatched = []
    for f in sorted(os.listdir(msk_dir)):
        m = MASK_RE.match(f)
        if not m:
            unmatched.append(f)
            continue
        by_stem.setdefault(m.group("stem"), []).append((m.group("cls").lower(), f))
    if unmatched:
        print("WARNING: %d files in %s did not match <stem>_<class>_mask.<ext>, e.g. %s"
              % (len(unmatched), msk_dir, unmatched[:3]))

    rows, counts, empty = [], {c: 0 for c in CLASSES}, 0
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        entries = by_stem.get(stem, [])
        if not entries:
            print("  no masks for %s, skipping" % f)
            continue
        acc = None
        for cls, mf in entries:
            for k in CLASSES:
                if k.lower() == cls:
                    counts[k] += 1
            if cls not in lesion:
                continue
            m = np.asarray(Image.open(os.path.join(msk_dir, mf)).convert("L")) > 0
            acc = m if acc is None else (acc | m)
        if acc is None:                       # every annotation was a non-lesion class
            probe = np.asarray(Image.open(os.path.join(img_dir, f)).convert("L"))
            acc = np.zeros(probe.shape, dtype=bool)
            empty += 1
        out_mask = os.path.join(a.out_masks, stem + "_lesion.png")
        Image.fromarray((acc.astype(np.uint8) * 255)).save(out_mask)
        rows.append((os.path.join(img_dir, f), out_mask, int(acc.sum())))

    df = pd.DataFrame(rows, columns=["image_path", "mask_path", "lesion_pixels"])
    df.to_csv(a.out, index=False)
    print("\nclass annotation counts: %s" % counts)
    print("lesion classes: %s" % sorted(lesion))
    print("%d images | %d with lesion pixels | %d all-negative (kept deliberately)"
          % (len(df), int((df.lesion_pixels > 0).sum()), empty))
    print("wrote %s and %d masks under %s/" % (a.out, len(df), a.out_masks))
    print("\nBEFORE TRAINING: check these images against the RARE data for overlap --")
    print("  the hidden test set is 12 BONS-AI centres and this is a 4-centre GI dataset.")
    print("  evaluation/dup_audit.py has the pHash and embedding machinery to do it.")


if __name__ == "__main__":
    main()
