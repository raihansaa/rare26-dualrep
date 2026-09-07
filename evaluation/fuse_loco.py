"""Validate a FIXED fusion weight on a held-out centre (plan S11 + S5B).

fuse.py selects the weight per fold on pooled PPV@90R, which the weight sweep
showed to be noise -- the per-fold picks were unstable and the nested estimate
came out negative for that reason. Here the weight is PREDECLARED, so there is
no selection and no optimism to subtract; the only question is whether it holds
on a centre no model trained on.

Paired patient-clustered bootstrap, because both arms score the identical images.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from metric import mode_a_weighted

p = argparse.ArgumentParser()
p.add_argument("--dinov2", required=True, help="LOCO run dir for the DINOv2 arm")
p.add_argument("--resnet", required=True, help="LOCO run dir for the ResNet arm")
p.add_argument("--w-resnet", type=float, required=True, help="PREDECLARED weight, not tuned here")
p.add_argument("--folds", default="folds_v2.csv")
p.add_argument("--n-boot", type=int, default=2000)
a = p.parse_args()


def cross(run):
    f = glob.glob(os.path.join(run, "pred_cross_*.csv"))
    assert len(f) == 1, "expected one pred_cross_*.csv in %s" % run
    return pd.read_csv(f[0]).sort_values("image_id").reset_index(drop=True)


D, R = cross(a.dinov2), cross(a.resnet)
assert (D.image_id.to_numpy() == R.image_id.to_numpy()).all(), "arms cover different images"
y = D.label.to_numpy()
sd, sr = D.score.to_numpy(), R.score.to_numpy()
fused = a.w_resnet * sr + (1 - a.w_resnet) * sd
print("held-out centre %s: %d images, %d positives | fixed w_R = %.2f\n"
      % (D.centre_id.iloc[0], len(y), y.sum(), a.w_resnet))

print("%-22s %8s %8s %9s" % ("", "AUROC", "AUPRC", "PPV@90R"))
for name, s in (("ResNet alone", sr), ("DINOv2 alone", sd), ("fusion (fixed w)", fused)):
    m = mode_a_weighted(y, s)
    print("%-22s %8.4f %8.4f %9.4f" % (name, m["auroc"], m["auprc"], m["ppv"]))

groups = D.image_id.map(pd.read_csv(a.folds).set_index("image_id").group_id).to_numpy()
uniq = np.unique(groups)
idx = {u: np.flatnonzero(groups == u) for u in uniq}
rng = np.random.default_rng(0)
dp, da = [], []
for _ in range(a.n_boot):
    take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), True)])
    yy = y[take]
    if len(np.unique(yy)) < 2:
        continue
    f_, d_ = mode_a_weighted(yy, fused[take]), mode_a_weighted(yy, sd[take])
    dp.append(f_["ppv"] - d_["ppv"])
    da.append(f_["auprc"] - d_["auprc"])
dp, da = np.array(dp), np.array(da)
print("\npaired delta (fusion - DINOv2 alone), %d patient-clustered draws:" % len(dp))
print("  PPV@90R  %+.4f  [%+.4f, %+.4f]   P(gain) = %.3f"
      % (np.nanmedian(dp), *np.nanpercentile(dp, [2.5, 97.5]), np.nanmean(dp > 0)))
print("  AUPRC    %+.4f  [%+.4f, %+.4f]   P(gain) = %.3f"
      % (np.nanmedian(da), *np.nanpercentile(da, [2.5, 97.5]), np.nanmean(da > 0)))
