"""Ensemble + calibrated-threshold LOCO evaluation (plan S5B, S11).

Measures the object that actually ships: N models averaged, ONE threshold picked
on data none of them trained on, applied once to an unseen centre. Every other
number in this project measures single models with a threshold re-optimised on
the same labels it is reported against, which is not what deployment does.

Members must share a fit/holdout split -- train with --split-seed fixed and
--seed varying -- so the within-training slice is a common calibration set that
no member trained on. The script asserts that.

Reports the fixed-threshold result (what ships) next to the oracle re-optimised
threshold (what the OOF tables have been quoting). The gap between them is the
threshold-transfer cost.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from metric import ppv_at_recall

RATIO = 100


def mode_a_weights(y):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return np.where(y == 1, 1.0, (n_pos * RATIO) / n_neg)


def at_threshold(y, s, t):
    """Weighted PPV and achieved recall at a FIXED threshold."""
    w, pred = mode_a_weights(y), s >= t
    tp, fp = w[(y == 1) & pred].sum(), w[(y == 0) & pred].sum()
    return (tp / (tp + fp) if tp + fp > 0 else np.nan,
            (y[pred] == 1).sum() / max((y == 1).sum(), 1))


def pick_threshold(y, s, target=0.90):
    return ppv_at_recall(y, s, target, sample_weight=mode_a_weights(y))[1]


def load(run, kind):
    f = glob.glob(os.path.join(run, "pred_%s_*.csv" % kind))
    assert len(f) == 1, "expected one pred_%s_*.csv in %s, found %d" % (kind, run, len(f))
    d = pd.read_csv(f[0]).sort_values("image_id").reset_index(drop=True)
    assert "logit" in d.columns, "%s predates the logit column -- retrain this member" % f[0]
    return d


p = argparse.ArgumentParser()
p.add_argument("--runs", required=True, help="glob matching the ensemble member run dirs")
p.add_argument("--folds", default="folds_v2.csv")
p.add_argument("--n-boot", type=int, default=1000)
a = p.parse_args()

runs = sorted(glob.glob(a.runs))
assert len(runs) > 1, "need >1 member, matched %s" % runs
wit = [load(r, "within") for r in runs]
crs = [load(r, "cross") for r in runs]
for d in wit[1:]:
    assert (d.image_id.to_numpy() == wit[0].image_id.to_numpy()).all(), \
        "members do not share a calibration slice -- retrain with --split-seed fixed"
for d in crs[1:]:
    assert (d.image_id.to_numpy() == crs[0].image_id.to_numpy()).all()

yw, yc = wit[0].label.to_numpy(), crs[0].label.to_numpy()
print("%d members | calibration slice %d (%d pos) | held-out centre %s %d (%d pos)\n"
      % (len(runs), len(yw), yw.sum(), crs[0].centre_id.iloc[0], len(yc), yc.sum()))

# Platt per member, fit on the slice it did not train on
cal_w, cal_c = [], []
for w, c in zip(wit, crs):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(w[["logit"]].to_numpy(), w.label.to_numpy())
    cal_w.append(lr.predict_proba(w[["logit"]].to_numpy())[:, 1])
    cal_c.append(lr.predict_proba(c[["logit"]].to_numpy())[:, 1])
ens_w, ens_c = np.mean(cal_w, axis=0), np.mean(cal_c, axis=0)

t_ens = pick_threshold(yw, ens_w)
ppv_fixed, rec_fixed = at_threshold(yc, ens_c, t_ens)
ppv_oracle = ppv_at_recall(yc, ens_c, 0.90, sample_weight=mode_a_weights(yc))[0]

print("%-34s %9s %9s" % ("", "PPV@90R", "recall"))
print("%-34s %9.4f %9s" % ("ensemble, threshold from slice", ppv_fixed, "%.3f" % rec_fixed))
print("%-34s %9.4f %9s   <- what the OOF tables quote" % ("ensemble, oracle threshold", ppv_oracle, "0.900"))

single = [at_threshold(yc, s, pick_threshold(yw, sw)) for sw, s in zip(cal_w, cal_c)]
sp, sr = np.array([x[0] for x in single]), np.array([x[1] for x in single])
print("%-34s %9.4f %9s   (range %.4f-%.4f)"
      % ("single members, fixed threshold", np.nanmean(sp), "%.3f" % sr.mean(), np.nanmin(sp), np.nanmax(sp)))

# group-level cluster bootstrap on the fixed-threshold ensemble result
g = pd.read_csv(a.folds).set_index("image_id").group_id
groups = crs[0].image_id.map(g).to_numpy()
uniq = np.unique(groups)
rng = np.random.default_rng(0)
boot = []
for _ in range(a.n_boot):
    take = np.concatenate([np.flatnonzero(groups == u) for u in rng.choice(uniq, len(uniq), True)])
    if len(np.unique(yc[take])) < 2:
        continue
    boot.append(at_threshold(yc[take], ens_c[take], t_ens)[0])
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
print("\nfixed-threshold PPV, group-cluster bootstrap: %.4f [%.4f, %.4f]" % (ppv_fixed, lo, hi))
print("threshold %.6f picked on the calibration slice, applied unchanged" % t_ens)
if rec_fixed < 0.90:
    print("WARNING: transferred threshold achieved %.1f%% recall, below the 90%% the metric requires"
          % (100 * rec_fixed))
