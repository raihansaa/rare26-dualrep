"""Grouped 5-fold split (plan S5A) -> folds_v1.csv

Grouping: duplicate_group from the inventory. Patient/exam/video IDs do not
exist in this dataset, so near-duplicate clusters are the only grouping
available -- it prevents near-identical frames straddling folds, nothing more.

Stratification is on label x centre jointly, not label alone: the centres
differ 4.4x in positive rate, so label-only stratification would let fold
centre-mix drift and add avoidable variance to pooled OOF.
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

p = argparse.ArgumentParser()
p.add_argument("--inventory", default="data_inventory_rare25.csv")
p.add_argument("--out", default="folds_v1.csv")
p.add_argument("--n-splits", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

df = pd.read_csv(a.inventory)
strata = df.label.astype(str) + "_" + df.centre_id.astype(str)

cv = StratifiedGroupKFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed)
df["fold"] = -1
for k, (_, va) in enumerate(cv.split(df, strata, groups=df.duplicate_group)):
    df.loc[df.index[va], "fold"] = k
assert (df.fold >= 0).all(), "some rows never assigned a fold"

out = df[["image_id", "label", "duplicate_group", "centre_id", "fold"]].rename(
    columns={"duplicate_group": "group_id"})
out.to_csv(a.out, index=False)

# --- verification ---------------------------------------------------------
leaked = df.groupby("duplicate_group").fold.nunique()
assert (leaked == 1).all(), "group split across folds: %s" % leaked[leaked > 1].index.tolist()

print("wrote %s (%d rows)\n" % (a.out, len(out)))
tab = df.pivot_table(index="fold", columns=["centre_id", "label"],
                     values="image_id", aggfunc="size", fill_value=0)
print(tab)
print("\npositives per fold :", df[df.label == 1].fold.value_counts().sort_index().tolist())
print("images per fold    :", df.fold.value_counts().sort_index().tolist())
print("pos rate per fold  :", df.groupby("fold").label.mean().round(4).tolist())
print("\nno duplicate_group spans folds: OK")
