"""Acquisition-conditional score normalisation, tested transductively on a held-out centre.

Maps each arm's score to its quantile WITHIN the acquisition stratum of the image, then to a
normal deviate, before fusing. If an unfamiliar scope or style inflates every score it
produces, its negatives outrank subtle positives from a lower-scoring style; the metric is one
order statistic over a pooled set, so it pays for exactly that. Within-stratum quantiles remove
the offset without touching within-stratum ordering.

Transductive by construction: the reference CDF is built from the held-out centre's own images
WITHOUT their labels, frozen, and only then are labels revealed for scoring. That is the
deployment analogue -- a container handed a batch can compute the same reference from the batch
alone. Nothing here is fitted on labels, so there is no leakage to argue about.

Shrinkage is predeclared as lambda_k = n_k / (n_k + 100): large strata are trusted, small ones
fall back to the global CDF. It is not tuned, because with two usable centres any tuned
constant would be selected on the data that validates it -- the error that put +0.12 of pure
optimism into an earlier fusion.

Adoption gate, also predeclared (from the consultation that proposed this):
  - at least 2 absolute FPR@90R points better on BOTH centres
  - non-worsening in at least 90% of group-clustered bootstrap replicates on each
Anything less and this is noise dressed as a mechanism.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SHRINK_N = 100.0


def logit(s):
    q = np.clip(np.asarray(s, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def platt(run, h):
    """Each arm calibrated on its own within-rest holdout, exactly as the container does."""
    ins = pd.read_csv(os.path.join(run, "pred_within_rest.csv"))
    out = pd.read_csv(glob.glob(os.path.join(run, "pred_cross_center_%s.csv" % h))[0])
    out = out.sort_values("image_id").reset_index(drop=True)
    m = LogisticRegression(max_iter=1000).fit(logit(ins.score).reshape(-1, 1), ins.label.to_numpy())
    return out, m.predict_proba(logit(out.score).reshape(-1, 1))[:, 1]


def quantile_normalise(s, strata):
    """Within-stratum empirical quantile, shrunk toward the global one, then Phi^-1.

    Uses every image in the stratum, labels unseen. Ranks are computed with 'average' ties and
    mapped to (0,1) by (r - 0.5)/n so the extremes stay finite under the normal quantile.
    """
    s = np.asarray(s, dtype=float)
    glob_u = (pd.Series(s).rank(method="average").to_numpy() - 0.5) / len(s)
    out = np.empty_like(s)
    for k in np.unique(strata):
        m = strata == k
        n = int(m.sum())
        u_k = (pd.Series(s[m]).rank(method="average").to_numpy() - 0.5) / n
        lam = n / (n + SHRINK_N)
        out[m] = norm.ppf(np.clip(lam * u_k + (1 - lam) * glob_u[m], 1e-6, 1 - 1e-6))
    return out


def fpr90(y, s):
    p = np.sort(s[y == 1])
    return float((s[y == 0] >= p[len(p) - int(np.ceil(0.9 * len(p)))]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--features", default="acq_features.csv")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()

    folds = pd.read_csv(a.folds)
    feats = pd.read_csv(a.features)
    assert list(feats.image_id) == list(folds.image_id), "feature cache does not match folds"
    cols = [c for c in feats.columns if c != "image_id"]
    X = feats[cols].to_numpy(dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    folds = folds.assign(stratum=KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(X).labels_)
    strata_of = folds.set_index("image_id").stratum

    rng = np.random.default_rng(0)
    for h in ("1", "2"):
        o, g = platt("runs/LOCO_dinov2_rest_to_center_%s_domain" % h, h)
        _, r = platt("runs/LOCO_rest_to_center_%s_rn50swsl_layer3" % h, h)
        y = o.label.to_numpy()
        st = o.image_id.map(strata_of).to_numpy()

        base = (g + r) / 2.0
        cand = (quantile_normalise(g, st) + quantile_normalise(r, st)) / 2.0

        print("\n--- held-out center_%s  (%d images, %d positives, %d strata present) ---"
              % (h, len(y), int(y.sum()), len(np.unique(st))))
        for nm, s in (("current fusion", base), ("acq-normalised", cand)):
            print("  %-16s FPR@90R %.4f   AUPRC %.4f   AUROC %.4f"
                  % (nm, fpr90(y, s), average_precision_score(y, s), roc_auc_score(y, s)))

        grp = o.image_id.map(folds.set_index("image_id").group_id).to_numpy()
        uniq = np.unique(grp)
        idx = {u: np.flatnonzero(grp == u) for u in uniq}
        d = []
        for _ in range(a.n_boot):
            take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), True)])
            yy = y[take]
            if yy.sum() < 5 or (yy == 0).sum() < 5:
                continue
            d.append(fpr90(yy, cand[take]) - fpr90(yy, base[take]))
        d = np.array(d)
        pts = 100 * (fpr90(y, base) - fpr90(y, cand))
        print("  paired delta FPR@90R %+.4f  [%+.4f, %+.4f]  P(not worse) = %.3f"
              % (np.median(d), *np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())))
        print("  GATE: %+.2f absolute points (need >= +2.00) | non-worsening %.1f%% (need >= 90%%)"
              % (pts, 100 * float((d <= 0).mean())))


if __name__ == "__main__":
    main()
