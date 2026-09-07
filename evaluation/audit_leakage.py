"""Is the OOF result explained by same-lesion frames straddling folds? (plan S6)

folds_v1 groups only near-identical images (pHash <= 4). Frames of the same
lesion at a different angle sit further apart and were free to split across
folds. If a validation image scores well *because* a close twin sat in its
training folds, the OOF number is memorisation, not detection.

Test: for every image, distance to its nearest neighbour in its TRAINING
folds. Then re-evaluate on progressively more novel subsets.
"""
import os
import numpy as np, pandas as pd
from PIL import Image
import imagehash
from metric import mode_a_weighted

folds = pd.read_csv("folds_v1.csv")
oof = pd.concat([pd.read_csv("runs/E1_resnet_p0_fold%d/oof_fold%d.csv" % (f, f))
                 for f in range(5)], ignore_index=True)
df = folds.merge(oof[["image_id", "score"]], on="image_id")
assert len(df) == len(folds)

path = lambda r: os.path.join("RARE25-train-data", r.centre_id,
                              "neo" if r.label else "ndbe", r.image_id + ".png")
bits = np.array([imagehash.phash(Image.open(path(r)).convert("RGB")).hash.flatten()
                 for r in df.itertuples()], dtype=np.uint8)
packed = np.packbits(bits, axis=1).view(np.uint64).ravel()
ham = np.bitwise_count(packed[:, None] ^ packed[None, :]).astype(np.int16)
np.fill_diagonal(ham, 99)

fold = df.fold.to_numpy()
y, s = df.label.to_numpy(), df.score.to_numpy()
# nearest neighbour that lived in this image's TRAINING folds
d_train = np.array([ham[i][fold != fold[i]].min() for i in range(len(df))])
# ...and restricted to the same class, the memorisation-relevant one
d_same = np.array([ham[i][(fold != fold[i]) & (y == y[i])].min() for i in range(len(df))])
df["d_train"], df["d_same"] = d_train, d_same

print("distance to nearest training-fold image")
print(pd.cut(d_train, [-1, 4, 8, 12, 16, 20, 99]).value_counts().sort_index().to_string())

print("\npositives: mean OOF score by distance to nearest TRAINING POSITIVE")
pos = df[df.label == 1]
b = pd.cut(pos.d_same, [-1, 8, 12, 16, 20, 99])
print(pos.groupby(b, observed=True).agg(n=("score", "size"), mean_score=("score", "mean"),
                                        median=("score", "median")).round(3).to_string())

print("\nnegatives: mean OOF score by distance to nearest TRAINING NEGATIVE")
neg = df[df.label == 0]
b = pd.cut(neg.d_same, [-1, 8, 12, 16, 20, 99])
print(neg.groupby(b, observed=True).agg(n=("score", "size"), mean_score=("score", "mean")).round(3).to_string())

print("\nre-evaluated on progressively more novel subsets")
print("  min_dist      n   pos    AUROC    AUPRC   PPV@90R")
for D in (0, 4, 8, 12, 16):
    m = d_train > D
    if len(np.unique(y[m])) < 2:
        continue
    w = mode_a_weighted(y[m], s[m])
    print("  > %-6d %5d  %4d   %.4f   %.4f    %.4f"
          % (D, m.sum(), y[m].sum(), w["auroc"], w["auprc"], w["ppv"]))
