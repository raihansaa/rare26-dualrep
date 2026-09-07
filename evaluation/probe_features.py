"""Two questions of a frozen feature bank, both linear, both seconds.

CENTRE IDENTIFIABILITY -- can a linear probe tell which centre an image came
from? If yes, the encoder carries acquisition fingerprints. That matters here
because our two informative validation centres are Dutch centres from the same
consortium that produced the backbone's pretraining corpus: a backbone that
fingerprints centres it was pretrained on will look better on those centres than
it will on twelve it has never seen.

LABEL, LOCO -- train a linear probe on two centres, test on the third. No LoRA,
no fine-tuning. Tells us how much of the system's performance is already sitting
in the frozen features, and lets backbones be compared without training runs.
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from metric import mode_a_weighted

p = argparse.ArgumentParser()
p.add_argument("--banks", nargs="+", required=True)
p.add_argument("--C", type=float, default=1.0)
a = p.parse_args()

for path in a.banks:
    d = np.load(path, allow_pickle=True)
    X, y = d["features"].astype(np.float32), d["label"]
    centre, group = d["centre_id"], d["group_id"]
    print("=" * 66)
    print("%s   %d images x %d dims" % (path, *X.shape))
    print("=" * 66)

    print("\nCENTRE IDENTIFIABILITY (patient-grouped 5-fold)")
    cls = np.unique(centre)
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, centre, groups=group):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
        clf.fit(X[tr], centre[tr])
        pred[te] = clf.predict(X[te])
    acc = (pred == centre).mean()
    base = max((centre == c).mean() for c in cls)
    print("  accuracy %.4f   (majority-class baseline %.4f)" % (acc, base))
    for c in cls:
        m = centre == c
        print("    %-9s n=%4d  recalled %.4f" % (c, m.sum(), (pred[m] == c).mean()))
    print("  -> %s" % ("features FINGERPRINT the centre" if acc > 0.95 else
                       "centre only partly recoverable"))

    print("\nLABEL, LEAVE-ONE-CENTRE-OUT linear probe")
    print("  %-9s %6s %5s %9s %9s %9s" % ("held-out", "n", "pos", "AUROC", "AUPRC", "PPV@90R"))
    for c in cls:
        te = centre == c
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=a.C, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        m = mode_a_weighted(y[te], s)
        print("  %-9s %6d %5d %9.4f %9.4f %9.4f"
              % (c, te.sum(), y[te].sum(), m["auroc"], m["auprc"], m["ppv"]))
    print()
