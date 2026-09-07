"""Combine the two challenge centres with EVC into folds_v2.csv.

Grouping differs by source and that is deliberate:
  center_1/center_2 -- perceptual-hash near-duplicate clusters (no IDs exist)
  evc               -- real patient IDs, so grouping is exact

Fold assignment is stratified on centre x label jointly, so every fold carries
the same mix of all three domains. EVC group ids are offset to avoid colliding
with the challenge data's duplicate_group numbering.
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

p = argparse.ArgumentParser()
p.add_argument("--folds", default="folds_v1.csv")
p.add_argument("--evc", default="evc_inventory.csv")
p.add_argument("--out", default="folds_v2.csv")
p.add_argument("--n-splits", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

ch = pd.read_csv(a.folds)[["image_id", "label", "group_id", "centre_id"]]
ev = pd.read_csv(a.evc)
offset = int(ch.group_id.max()) + 1
ev = pd.DataFrame({
    "image_id": ev.image_id,
    "label": ev.label,
    "group_id": ev.patient_id.factorize()[0] + offset,   # real patients
    "centre_id": "evc",
})
df = pd.concat([ch, ev], ignore_index=True)

strata = df.centre_id.astype(str) + "_" + df.label.astype(str)
cv = StratifiedGroupKFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed)
df["fold"] = -1
for k, (_, va) in enumerate(cv.split(df, strata, groups=df.group_id)):
    df.loc[df.index[va], "fold"] = k
assert (df.fold >= 0).all()
assert (df.groupby("group_id").fold.nunique() == 1).all(), "a group straddles folds"
assert df.image_id.is_unique, "image_id collision between sources"
df.to_csv(a.out, index=False)

print("wrote %s (%d rows)\n" % (a.out, len(df)))
print(df.pivot_table(index="centre_id", columns="label", values="image_id",
                     aggfunc="size", fill_value=0).to_string())
print("\npositive rate by centre:")
print(df.groupby("centre_id").label.mean().round(4).to_string())
print("\nfold composition:")
print(df.pivot_table(index="fold", columns=["centre_id", "label"], values="image_id",
                     aggfunc="size", fill_value=0).to_string())
print("\ntotal: %d images, %d positives (was %d / %d)"
      % (len(df), df.label.sum(), len(ch), ch.label.sum()))
