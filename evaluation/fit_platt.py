"""Fit the per-checkpoint Platt calibration that the container ships as platt.json.

Each entry is [a, b, prior_fit]. a and b are the affine calibration of that
checkpoint's out-of-fold logits -- data the checkpoint never trained on -- fitted
by plain logistic regression at sklearn's default regularisation. prior_fit is the
prevalence of the slice they were fitted on; inference.py retains it for provenance
and no longer reads it, the prior shift having been removed (EXPERIMENT_LOG.md 3).

Calibration matters here because the two arms are different architectures on
different pretraining corpora and their logit scales are not comparable: averaging
raw scores measured negative on both held-out centres.

Regenerates the shipped file exactly:

  python evaluation/fit_platt.py \
      --arm dinov2=runs/V2_dinogn_fold{k} \
      --arm swsl=runs/V7_swsl_fold{k} \
      --out submission_template/platt.json
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

p = argparse.ArgumentParser()
p.add_argument("--arm", action="append", required=True,
               help="NAME=PATTERN, where PATTERN contains {k} for the fold index "
                    "(e.g. dinov2=runs/V2_dinogn_fold{k})")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--out", default="submission_template/platt.json")
a = p.parse_args()

platt = {}
for spec in a.arm:
    if "=" not in spec:
        raise SystemExit("--arm needs NAME=PATTERN, got %s" % spec)
    name, pattern = spec.split("=", 1)
    for k in range(a.folds):
        run = pattern.format(k=k)
        csv = os.path.join(run, "oof_fold%d.csv" % k)
        if not os.path.exists(csv):
            raise SystemExit("missing OOF predictions: %s" % csv)
        d = pd.read_csv(csv)

        # Scores are stored as probabilities; Platt fits an affine map in logit space.
        prob = np.clip(d["score"].to_numpy(float), 1e-7, 1 - 1e-7)
        logit = np.log(prob / (1 - prob)).reshape(-1, 1)
        y = d["label"].to_numpy(int)
        assert y.min() == 0 and y.max() == 1, "%s has one class only -- cannot calibrate" % csv

        m = LogisticRegression().fit(logit, y)
        key = "%s_fold%d.pt" % (name, k)
        platt[key] = [round(float(m.coef_[0][0]), 6),
                      round(float(m.intercept_[0]), 6),
                      round(float(y.mean()), 6)]
        print("%-18s n=%4d pos=%3d  a=%9.6f b=%9.6f prior=%.6f"
              % (key, len(y), y.sum(), *platt[key]))

json.dump(platt, open(a.out, "w"), indent=1)
print("\nwrote %d entries to %s" % (len(platt), a.out))
