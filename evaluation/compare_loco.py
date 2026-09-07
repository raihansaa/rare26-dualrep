"""Paired comparison of two LOCO runs on the same held-out centre.

One predeclared A/B at a time, on the protocol that has survived scrutiny here:
train on the other centres, test on a centre no model saw. Both runs score the
identical images, so the bootstrap is paired and clustered by near-duplicate group -- far more
powerful than comparing two marginal confidence intervals, which overlap heavily
at this positive count even when the paired delta is consistent.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from metric import mode_a_weighted

p = argparse.ArgumentParser()
p.add_argument("--baseline", required=True, help="run dir for the current system")
p.add_argument("--candidate", required=True, help="run dir for the change under test")
p.add_argument("--folds", default="folds_v2.csv")
p.add_argument("--n-boot", type=int, default=2000)
a = p.parse_args()


def cross(run):
    f = glob.glob(os.path.join(run, "pred_cross_*.csv"))
    assert len(f) == 1, "expected one pred_cross_*.csv in %s, found %d" % (run, len(f))
    return pd.read_csv(f[0]).sort_values("image_id").reset_index(drop=True)


A, B = cross(a.baseline), cross(a.candidate)
assert (A.image_id.to_numpy() == B.image_id.to_numpy()).all(), \
    "runs cover different images -- are they the same holdout centre?"
y = A.label.to_numpy()
sa, sb = A.score.to_numpy(), B.score.to_numpy()
print("held-out %s: %d images, %d positives\n" % (A.centre_id.iloc[0], len(y), y.sum()))

print("%-12s %8s %8s %9s" % ("", "AUROC", "AUPRC", "PPV@90R"))
for name, s in (("baseline", sa), ("candidate", sb)):
    m = mode_a_weighted(y, s)
    print("%-12s %8.4f %8.4f %9.4f" % (name, m["auroc"], m["auprc"], m["ppv"]))

groups = A.image_id.map(pd.read_csv(a.folds).set_index("image_id").group_id).to_numpy()
uniq = np.unique(groups)
idx = {u: np.flatnonzero(groups == u) for u in uniq}
rng = np.random.default_rng(0)
dp, da, dr = [], [], []
for _ in range(a.n_boot):
    take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), True)])
    yy = y[take]
    if len(np.unique(yy)) < 2:
        continue
    mb, ma = mode_a_weighted(yy, sb[take]), mode_a_weighted(yy, sa[take])
    dp.append(mb["ppv"] - ma["ppv"])
    da.append(mb["auprc"] - ma["auprc"])
    dr.append(mb["auroc"] - ma["auroc"])
f = lambda d: (np.nanmedian(d), *np.nanpercentile(d, [2.5, 97.5]), np.nanmean(np.array(d) > 0))
print("\npaired delta (candidate - baseline), %d group-clustered draws:" % len(dp))
for name, d in (("AUPRC", da), ("AUROC", dr), ("PPV@90R", dp)):
    print("  %-8s %+.4f  [%+.4f, %+.4f]   P(gain) = %.3f" % ((name,) + f(d)))
print("\nAUPRC/AUROC are the decidable signals at this sample size; PPV@90R will")
print("usually straddle zero even for a real improvement -- do not judge on it alone.")
