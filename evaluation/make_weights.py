"""Build the S9 sample-weight file from inspected hard-example lists.

Reads hard_positives.csv / hard_negatives.csv and keeps rows where
selected_for_reweighting is truthy (1, true, yes, y). S9 caps weights at
2.0-3.0; anything outside that range is rejected rather than silently clipped.
"""
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--positives", default="hard_positives.csv")
p.add_argument("--negatives", default="hard_negatives.csv")
p.add_argument("--weight", type=float, default=2.5)
p.add_argument("--out", default="weights.csv")
a = p.parse_args()

if not 2.0 <= a.weight <= 3.0:
    raise SystemExit("S9 specifies 2.0-3.0 for selected hard examples; got %.2f" % a.weight)

TRUE = {"1", "true", "yes", "y", "t"}
sel = lambda d: d[d.selected_for_reweighting.astype(str).str.strip().str.lower().isin(TRUE)]

parts = []
for path, kind in ((a.positives, "positive"), (a.negatives, "negative")):
    d = sel(pd.read_csv(path))
    print("%-9s selected: %d" % (kind, len(d)))
    parts.append(pd.DataFrame({"image_id": d.image_id, "weight": a.weight}))

w = pd.concat(parts, ignore_index=True).drop_duplicates("image_id")
if w.empty:
    raise SystemExit("nothing selected -- fill selected_for_reweighting first (plan S9)")
w.to_csv(a.out, index=False)
print("wrote %s: %d images at weight %.1f" % (a.out, len(w), a.weight))
