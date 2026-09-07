"""E6/E7 -- cross-fitted hard-example mining (plan S9).

Uses only nested OOF predictions, so no image is judged by a model that
selected on it.

The 90%-recall operating point is re-derived on each bootstrap iteration; an
example is "hard" by how often it lands on the wrong side of that moving
threshold, not by a single point estimate.

Columns the plan requires but a script cannot fill -- quality_issue,
possible_label_issue, manual_notes, selected_for_reweighting -- are emitted
empty. S9 requires every repeatedly influential positive to be inspected
before its weight changes.
"""
import argparse
import numpy as np, pandas as pd
from metric import ppv_at_recall

p = argparse.ArgumentParser()
p.add_argument("--resnet", default="runs/E1n_resnet_p0_fold{f}/oof_fold{f}.csv")
p.add_argument("--dinov2", default="runs/E4n_dinov2_lora_fold{f}/oof_fold{f}.csv")
p.add_argument("--tta-resnet", default="tta_resnet.csv")
p.add_argument("--tta-dinov2", default="tta_dinov2.csv")
p.add_argument("--fov", default="fov_crops.csv")
p.add_argument("--w-resnet", type=float, default=0.5)
p.add_argument("--n-boot", type=int, default=1000)
p.add_argument("--target", type=float, default=0.90)
a = p.parse_args()

load = lambda t: pd.concat([pd.read_csv(t.format(f=f)) for f in range(5)], ignore_index=True)
R, D = load(a.resnet), load(a.dinov2)
df = R[["image_id", "label", "fold", "centre_id", "score"]].merge(
    D[["image_id", "score"]], on="image_id", suffixes=("_r", "_d"))
df["score_fused"] = a.w_resnet * df.score_r + (1 - a.w_resnet) * df.score_d

y = df.label.to_numpy()
s = df.score_fused.to_numpy()
n_pos, n_neg = int(y.sum()), int((y == 0).sum())
w = np.where(y == 1, 1.0, (n_pos * 100) / n_neg)
_, t_point = ppv_at_recall(y, s, a.target, sample_weight=w)
print("operating point: threshold %.4f at recall >= %.2f" % (t_point, a.target))

# bootstrap the threshold
rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
ts = []
for _ in range(a.n_boot):
    i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
    yy, ss = y[i], s[i]
    ww = np.where(yy == 1, 1.0, (yy.sum() * 100) / (yy == 0).sum())
    _, t = ppv_at_recall(yy, ss, a.target, sample_weight=ww)
    ts.append(t)
ts = np.array([t for t in ts if np.isfinite(t)])
print("bootstrap threshold: median %.4f  [%.4f, %.4f]  over %d draws"
      % (np.median(ts), *np.percentile(ts, [2.5, 97.5]), len(ts)))

band = 0.05 * (np.percentile(s, 99) - np.percentile(s, 1))
df["times_below_threshold"] = [(ts > v).mean() for v in s]
df["times_near_threshold"] = [(np.abs(ts - v) < band).mean() for v in s]

# per-arm thresholds, for "low/high from BOTH families"
tr = ppv_at_recall(y, df.score_r.to_numpy(), a.target, sample_weight=w)[1]
td = ppv_at_recall(y, df.score_d.to_numpy(), a.target, sample_weight=w)[1]
df["below_both"] = (df.score_r < tr) & (df.score_d < td)
df["above_both"] = (df.score_r >= tr) & (df.score_d >= td)

# persistence across the four TTA views (S9: "across folds or augmentations").
# Pass --tta-resnet none when no TTA views exist for THESE models; views from a
# different model would score persistence for a model that is not under test.
sig = lambda z: 1 / (1 + np.exp(-z))
V = ["T0", "T1", "T2", "T3"]
has_tta = a.tta_resnet.lower() != "none" and a.tta_dinov2.lower() != "none"
if has_tta:
    TR, TD = pd.read_csv(a.tta_resnet), pd.read_csv(a.tta_dinov2)
    tta = TR[["image_id"]].copy()
    for v in V:
        tta["v_" + v] = a.w_resnet * sig(TR["logit_" + v]) + (1 - a.w_resnet) * sig(
            TD.set_index("image_id").loc[TR.image_id, "logit_" + v].to_numpy())
    df = df.merge(tta, on="image_id")
    vv = df[["v_" + v for v in V]].to_numpy()
    df["views_above_thr"] = (vv >= t_point).sum(1)      # 0..4
    df["views_below_thr"] = (vv < t_point).sum(1)
else:
    df["views_above_thr"] = df["views_below_thr"] = -1  # not measured

has_fov = a.fov.lower() != "none"
if has_fov:
    fov = pd.read_csv(a.fov)[["image_id", "status", "kept"]]
    df = df.merge(fov, on="image_id", how="left")
    df["crop_failure"] = df.status.ne("ok")
else:
    df["crop_failure"] = False

BLANK = ["quality_issue", "possible_label_issue", "manual_notes", "selected_for_reweighting"]
COLS = (["image_id", "fold", "centre_id", "score_r", "score_d", "score_fused",
         "times_below_threshold", "times_near_threshold", "crop_failure"] + BLANK)

hp = df[df.label == 1].copy()
hp = hp[(hp.times_below_threshold > 0.10) | (hp.times_near_threshold > 0.10) | hp.below_both]
for c in BLANK:
    hp[c] = ""
hp = hp.sort_values("times_below_threshold", ascending=False)
hp[COLS + ["below_both", "views_below_thr"]].to_csv("hard_positives.csv", index=False)

hn = df[df.label == 0].copy()
hn = hn[(hn.times_below_threshold < 0.90) | hn.above_both]      # scores at/above the threshold
for c in BLANK:
    hn[c] = ""
hn["frac_above_threshold"] = 1 - hn.times_below_threshold
hn = hn.sort_values("frac_above_threshold", ascending=False)
hn[COLS + ["frac_above_threshold", "above_both", "views_above_thr"]].to_csv("hard_negatives.csv", index=False)

print("\nhard_positives.csv : %d of %d positives (%.1f%%)" % (len(hp), n_pos, 100 * len(hp) / n_pos))
print("   below threshold in >50%% of bootstraps : %d" % (hp.times_below_threshold > .5).sum())
print("   low from BOTH families                : %d" % hp.below_both.sum())
print("   below threshold in all 4 TTA views    : %s"
      % ((hp.views_below_thr == 4).sum() if has_tta else "not measured"))
print("   by centre: %s" % hp.centre_id.value_counts().to_dict())
print("\nhard_negatives.csv : %d of %d negatives (%.1f%%)" % (len(hn), n_neg, 100 * len(hn) / n_neg))
print("   above threshold in >50%% of bootstraps : %d" % (hn.frac_above_threshold > .5).sum())
print("   high from BOTH families               : %d" % hn.above_both.sum())
print("   above threshold in all 4 TTA views    : %s"
      % ((hn.views_above_thr == 4).sum() if has_tta else "not measured"))
print("   by centre: %s" % hn.centre_id.value_counts().to_dict())
print("\ncrop failures among flagged: %s"
      % (int(pd.concat([hp, hn]).crop_failure.sum()) if has_fov else "not measured"))
