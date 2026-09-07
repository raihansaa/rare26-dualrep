"""E10 -- nested fusion of the two supervised arms (plan S11).

s = w_R*s_R + w_D*s_D,  w_R,w_D >= 0,  w_R + w_D = 1,  grid step 0.05.

Nested: for held-out fold k the weight is chosen on the other four folds, then
applied to fold k. Concatenating the five held-out folds gives an estimate that
never selected on the data it reports.

Naive: the weight is chosen on the pooled OOF and reported on the same pooled
OOF. The naive-minus-nested difference is the model-selection optimism S11 asks
us to quantify.
"""
import argparse
import numpy as np, pandas as pd
from metric import mode_a_weighted, mode_a_resampled
from sklearn.metrics import average_precision_score, roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--resnet", default="runs/E1n_resnet_p0_fold{f}/oof_fold{f}.csv")
p.add_argument("--dinov2", default="runs/E4n_dinov2_lora_fold{f}/oof_fold{f}.csv")
p.add_argument("--select-on", default="ppv90", choices=["ppv90", "auprc"])
p.add_argument("--n-boot", type=int, default=1000)
a = p.parse_args()

load = lambda t: pd.concat([pd.read_csv(t.format(f=f)) for f in range(5)], ignore_index=True)
R, D = load(a.resnet), load(a.dinov2)
df = R[["image_id", "label", "fold", "centre_id", "score"]].merge(
    D[["image_id", "score"]], on="image_id", suffixes=("_r", "_d"))
assert len(df) == len(R) == len(D), "OOF sets do not align on image_id"
print("fused %d images, %d positives\n" % (len(df), df.label.sum()))

GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
y = df.label.to_numpy()
sr, sd = df.score_r.to_numpy(), df.score_d.to_numpy()


def score(y, s, how):
    return average_precision_score(y, s) if how == "auprc" else mode_a_weighted(y, s)["ppv"]


def best_w(y, sr, sd, how):
    vals = [score(y, w * sr + (1 - w) * sd, how) for w in GRID]
    return GRID[int(np.argmax(vals))]


# --- nested -------------------------------------------------------------
nested = np.empty(len(df))
picks = []
for k in range(5):
    inner, outer = (df.fold != k).to_numpy(), (df.fold == k).to_numpy()
    w = best_w(y[inner], sr[inner], sd[inner], a.select_on)
    picks.append(w)
    nested[outer] = w * sr[outer] + (1 - w) * sd[outer]

# --- naive --------------------------------------------------------------
w_naive = best_w(y, sr, sd, a.select_on)
naive = w_naive * sr + (1 - w_naive) * sd


def row(name, s):
    w, r = mode_a_weighted(y, s), mode_a_resampled(y, s, n_boot=a.n_boot)
    print("  %-28s %.4f   %.4f   %.4f   %.4f [%.4f, %.4f]"
          % (name, w["auroc"], w["auprc"], w["ppv"], r["median"], r["lo"], r["hi"]))
    return w["ppv"]


print("  %-28s %-8s %-8s %-8s %s" % ("", "AUROC", "AUPRC", "PPV(w)", "PPV resampled [95%]"))
ppv_r = row("ResNet alone", sr)
ppv_d = row("DINOv2 alone", sd)
ppv_na = row("naive fusion (w_R=%.2f)" % w_naive, naive)
ppv_ne = row("nested fusion", nested)

print("\nper-fold weights chosen (w_R): %s" % picks)
print("selection metric: %s" % a.select_on)
best_single = max(ppv_r, ppv_d)
print("\nnested minus best single arm : %+.4f" % (ppv_ne - best_single))
print("naive minus nested (optimism): %+.4f" % (ppv_na - ppv_ne))

print("\nPER CENTRE (nested fusion)")
for c, g in df.groupby("centre_id"):
    m = (df.centre_id == c).to_numpy()
    w = mode_a_weighted(y[m], nested[m])
    print("  %-9s n=%4d pos=%3d  AUROC %.4f  AUPRC %.4f  PPV@90R %.4f"
          % (c, m.sum(), y[m].sum(), w["auroc"], w["auprc"], w["ppv"]))
