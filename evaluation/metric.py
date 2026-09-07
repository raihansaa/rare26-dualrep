"""Mode-A PPV@90%Recall at simulated 100:1 prevalence (plan S6).

Two estimators of the same quantity:

  mode_a_resampled  mirrors the plan text -- keep all negatives, resample
                    positives with replacement down to n_neg/ratio. This is
                    what the leaderboard-style protocol does, and it is what
                    should be reported.

  mode_a_weighted   keeps ALL positives and weights each negative by
                    (n_pos*ratio)/n_neg. Algebraically the same target
                    (PPV = TPR / (TPR + ratio*FPR)) but far lower variance,
                    and it does not quantise recall. Use for model selection.

Why the quantisation matters: with 2937 negatives the resampled protocol draws
only 29 positives, so achievable recall is k/29 and "recall >= 0.90" actually
binds at 27/29 = 0.931 -- a stricter operating point that depresses PPV. With
all 158 positives it binds at 143/158 = 0.905.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


def ppv_at_recall(y, s, target=0.90, sample_weight=None):
    """Max precision among operating points with recall >= target."""
    prec, rec, thr = precision_recall_curve(y, s, sample_weight=sample_weight)
    ok = rec >= target
    if not ok.any():
        return np.nan, np.nan
    i = np.flatnonzero(ok)[np.argmax(prec[ok])]
    return float(prec[i]), float(thr[i]) if i < len(thr) else np.inf


def mode_a_resampled(y, s, ratio=100, n_boot=1000, target=0.90, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_draw = int(round(len(neg) / ratio))
    rng = np.random.default_rng(seed)
    ppvs, thrs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([neg, rng.choice(pos, n_draw, replace=True)])
        yy = np.concatenate([np.zeros(len(neg), int), np.ones(n_draw, int)])
        ppvs[b], thrs[b] = ppv_at_recall(yy, s[idx], target)
    return dict(
        median=float(np.nanmedian(ppvs)),
        lo=float(np.nanpercentile(ppvs, 2.5)),
        hi=float(np.nanpercentile(ppvs, 97.5)),
        n_pos_per_iter=n_draw,
        effective_recall=np.ceil(target * n_draw) / n_draw,
        thr_median=float(np.nanmedian(thrs)),
    )


def mode_a_weighted(y, s, ratio=100, target=0.90):
    y, s = np.asarray(y), np.asarray(s)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:          # degenerate subset -- nothing to rank
        return dict(ppv=np.nan, threshold=np.nan, weighted_fp=np.nan,
                    effective_recall=np.nan, auprc=np.nan, auroc=np.nan)
    w = np.where(y == 1, 1.0, (n_pos * ratio) / n_neg)
    ppv, thr = ppv_at_recall(y, s, target, sample_weight=w)
    fp = float((w * ((s >= thr) & (y == 0))).sum())
    return dict(
        ppv=ppv, threshold=thr, weighted_fp=fp,
        effective_recall=np.ceil(target * n_pos) / n_pos,
        auprc=float(average_precision_score(y, s)),
        auroc=float(roc_auc_score(y, s)),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_neg, n_pos, ratio = 2937, 158, 100
    y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]
    chance = 1.0 / (1 + ratio)

    def show(name, s):
        r = mode_a_resampled(y, s, ratio, n_boot=400)
        w = mode_a_weighted(y, s, ratio)
        print("%-22s resampled %.4f [%.4f, %.4f] | weighted %.4f | auroc %.3f"
              % (name, r["median"], r["lo"], r["hi"], w["ppv"], w["auroc"]))
        return r, w

    print("chance PPV at %d:1 = %.4f\n" % (ratio, chance))
    r, _ = show("random", rng.normal(size=n_neg + n_pos))
    show("perfect", y.astype(float))
    show("inverted", -y.astype(float))
    show("realistic (auroc~.85)", np.r_[rng.normal(0, 1, n_neg), rng.normal(1.5, 1, n_pos)])

    print("\nrecall quantisation:")
    print("  resampled draws %d positives -> recall binds at %.3f"
          % (r["n_pos_per_iter"], r["effective_recall"]))
    print("  weighted keeps  %d positives -> recall binds at %.3f"
          % (n_pos, np.ceil(0.90 * n_pos) / n_pos))

    rnd = mode_a_resampled(y, rng.normal(size=n_neg + n_pos), ratio, n_boot=400)
    assert abs(rnd["median"] - chance) < 0.004, "random should sit at chance"
    assert mode_a_weighted(y, y.astype(float), ratio)["ppv"] == 1.0
    print("\nself-tests OK")
