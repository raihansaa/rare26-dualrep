"""Acquisition-only strata, and whether their score tails differ enough to be worth exploiting.

The proposal under test: if an unfamiliar scope or style lifts every score it produces, that
style's negatives sit above subtle positives from a lower-scoring style, and the metric --
one order statistic over a pooled test set -- pays for it. Mapping each score to its
within-stratum quantile would undo that offset.

This script is the precondition, not the transform. It answers two questions cheaply, and if
either answer is unfavourable the transform cannot help and should not be built:

  1. Do acquisition strata collapse onto CENTRES? If they do, a within-stratum rank transform
     is monotone inside each centre, so on a leave-one-centre-out evaluation it cannot change
     FPR@90R at all -- the mechanism would be real but unmeasurable on the data we hold.
  2. Do the strata actually differ in their score tails? The predeclared stop condition is
     < 0.25 SD separation in mean score AND < 2x difference in the top-10% exceedance rate.

Features are ACQUISITION ONLY -- geometry, exposure, colour balance, focus, specularity. No
learned embedding is used anywhere here: GastroNet features are lesion-sensitive, so
clustering on them would stratify partly by pathology and the normalisation would then
subtract the signal it is supposed to protect.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace

PROBE = 256


def acquisition_features(path):
    """Geometry, exposure, colour balance, focus, specularity, border extent."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    a = np.asarray(im.resize((PROBE, PROBE), Image.BILINEAR), dtype=np.float32) / 255.0
    grey = a.mean(-1)
    lap = laplace(grey)
    dark = grey < 0.08
    # border extent: dark fraction in the outer eighth, which is where letterboxing and
    # circular field-of-view masks live
    edge = np.ones_like(dark, dtype=bool)
    edge[PROBE // 8:-PROBE // 8, PROBE // 8:-PROBE // 8] = False
    return {
        "width": w, "height": h, "aspect": w / max(h, 1),
        "brightness": float(grey.mean()), "brightness_sd": float(grey.std()),
        "r": float(a[..., 0].mean()), "g": float(a[..., 1].mean()), "b": float(a[..., 2].mean()),
        "rg_ratio": float(a[..., 0].mean() / max(a[..., 1].mean(), 1e-6)),
        "blur": float(lap.var()),
        "specular": float((a.min(-1) > 0.94).mean()),
        "dark_frac": float(dark.mean()),
        "border_dark": float(dark[edge].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--scores", nargs="+", required=True,
                   help="run dirs whose pred_cross_*.csv supply the scores to stratify")
    p.add_argument("--k", type=int, default=6, help="number of strata, fixed before looking")
    p.add_argument("--cache", default="acq_features.csv")
    a = p.parse_args()

    df = pd.read_csv(a.folds)
    if os.path.exists(a.cache):
        feats = pd.read_csv(a.cache)
        assert list(feats.image_id) == list(df.image_id), "stale cache -- delete %s" % a.cache
        print("loaded %s" % a.cache)
    else:
        paths = {}
        for c in sorted(os.listdir(a.data_root)):
            for k in sorted(os.listdir(os.path.join(a.data_root, c))):
                d = os.path.join(a.data_root, c, k)
                for f in os.listdir(d):
                    paths[os.path.splitext(f)[0]] = os.path.join(d, f)
        rows = []
        for n, iid in enumerate(df.image_id):
            rows.append(dict(image_id=iid, **acquisition_features(paths[iid])))
            if (n + 1) % 500 == 0:
                print("  featurised %d/%d" % (n + 1, len(df)), flush=True)
        feats = pd.DataFrame(rows)
        feats.to_csv(a.cache, index=False)
        print("wrote %s" % a.cache)

    cols = [c for c in feats.columns if c != "image_id"]
    X = feats[cols].to_numpy(dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(X)
    df = df.assign(stratum=km.labels_)

    print("\n1. DO STRATA COLLAPSE ONTO CENTRES?")
    ct = pd.crosstab(df.stratum, df.centre_id)
    print(ct.to_string())
    purity = ct.max(axis=1).sum() / len(df)
    print("  stratum purity w.r.t. centre: %.3f  (1.000 = strata ARE centres)" % purity)
    print("  -> %s" % ("COLLAPSED: a within-stratum transform is monotone inside each centre, "
                       "so LOCO cannot measure it" if purity > 0.95 else
                       "strata cut across centres, so the transform is measurable here"))

    print("\n2. DO THE STRATA DIFFER IN THEIR SCORE TAILS?")
    import glob
    for run in a.scores:
        f = glob.glob(os.path.join(run, "pred_cross_*.csv"))
        if not f:
            print("  %s: no predictions, skipped" % run)
            continue
        pr = pd.read_csv(f[0])
        m = df.merge(pr[["image_id", "score"]], on="image_id")
        neg = m[m.label == 0]
        if len(neg) < 50:
            continue
        z = (neg.score - neg.score.mean()) / (neg.score.std() + 1e-9)
        g = z.groupby(neg.stratum)
        spread = g.mean().max() - g.mean().min()
        thr = neg.score.quantile(0.90)
        exc = neg.groupby(neg.stratum).score.apply(lambda s: (s >= thr).mean())
        exc = exc[exc.index.isin(neg.stratum.value_counts()[lambda v: v >= 25].index)]
        ratio = exc.max() / max(exc.min(), 1e-9)
        print("\n  %s  (%d negatives)" % (os.path.basename(run), len(neg)))
        print("    mean standardised score by stratum: %s"
              % ", ".join("%d:%+.2f" % (k, v) for k, v in g.mean().items()))
        print("    top-10%% exceedance by stratum:      %s"
              % ", ".join("%d:%.3f" % (k, v) for k, v in exc.items()))
        print("    spread %.2f SD | exceedance ratio %.1fx" % (spread, ratio))
        verdict = "PROCEED" if (spread >= 0.25 and ratio >= 2.0) else "STOP (predeclared)"
        print("    -> %s" % verdict)


if __name__ == "__main__":
    main()
