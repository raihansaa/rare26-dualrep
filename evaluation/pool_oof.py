"""Pool per-fold OOF predictions and evaluate (plan S6, S7).

Reports both Mode-A estimators plus the per-centre breakdown, because pooled
numbers are inflated by the centre-prevalence shortcut (centre alone predicts
the label at AUROC 0.685 on RARE25).
"""
import argparse, glob
import numpy as np, pandas as pd
from metric import mode_a_resampled, mode_a_weighted

p = argparse.ArgumentParser()
p.add_argument("--glob", default="runs/E1_resnet_p0_fold*/oof_fold*.csv")
p.add_argument("--n-boot", type=int, default=1000)
a = p.parse_args()

files = sorted(glob.glob(a.glob))
if not files:
    raise SystemExit("no OOF files matched %s" % a.glob)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
assert df.image_id.is_unique, "duplicate image_id across folds -- OOF is not disjoint"
print("pooled %d folds, %d images, %d positives\n" % (df.fold.nunique(), len(df), df.label.sum()))

y, s = df.label.to_numpy(), df.score.to_numpy()
w, r = mode_a_weighted(y, s), mode_a_resampled(y, s, n_boot=a.n_boot)
print("POOLED OOF")
print("  AUROC              %.4f" % w["auroc"])
print("  AUPRC              %.4f" % w["auprc"])
print("  PPV@90R weighted   %.4f   <- decide on this" % w["ppv"])
print("  PPV@90R resampled  %.4f  [%.4f, %.4f]   <- report this" % (r["median"], r["lo"], r["hi"]))
print("  chance             %.4f" % (1 / 101))

print("\nPER CENTRE (pooled hides the shortcut)")
for c, g in df.groupby("centre_id"):
    if g.label.nunique() < 2:
        continue
    gw = mode_a_weighted(g.label.to_numpy(), g.score.to_numpy())
    print("  %-9s n=%4d pos=%3d  AUROC %.4f  AUPRC %.4f  PPV@90R %.4f"
          % (c, len(g), g.label.sum(), gw["auroc"], gw["auprc"], gw["ppv"]))

print("\nPER FOLD (spread shows single-fold noise)")
for f, g in df.groupby("fold"):
    gw = mode_a_weighted(g.label.to_numpy(), g.score.to_numpy())
    print("  fold %d  pos=%2d  AUROC %.4f  AUPRC %.4f  PPV@90R %.4f"
          % (f, g.label.sum(), gw["auroc"], gw["auprc"], gw["ppv"]))

print("\nAUDIT (plan S6)")
print("  score orientation  : mean pos %.4f vs mean neg %.4f  %s"
      % (s[y == 1].mean(), s[y == 0].mean(),
         "OK" if s[y == 1].mean() > s[y == 0].mean() else "INVERTED"))
if w["ppv"] > 0.08:
    print("  PPV %.4f EXCEEDS 0.08 -- audit for leakage before believing it" % w["ppv"])
else:
    print("  PPV below the 0.08 audit trigger")
print("  S8 gate (>=0.015)  : %s" % ("PASS" if w["ppv"] >= 0.015 else "FAIL"))
