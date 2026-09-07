"""Contact sheets for the S9 manual inspection step.

S9 requires every repeatedly influential example to be looked at before its
weight changes. This lays them out with their scores so that is a few minutes
of work rather than hundreds of file opens.
"""
import argparse, os
import pandas as pd
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--root", default="RARE25-train-data")
p.add_argument("--label", type=int, required=True, help="1 for positives, 0 for negatives")
p.add_argument("--limit", type=int, default=60)
p.add_argument("--cols", type=int, default=5)
p.add_argument("--cell", type=int, default=200)
a = p.parse_args()

df = pd.read_csv(a.csv).head(a.limit)
cls = "neo" if a.label else "ndbe"
STRIP = 34
rows = (len(df) + a.cols - 1) // a.cols
sheet = Image.new("RGB", (a.cols * a.cell, rows * (a.cell + STRIP)), "black")
d = ImageDraw.Draw(sheet)

for i, r in enumerate(df.itertuples()):
    path = os.path.join(a.root, r.centre_id, cls, r.image_id + ".png")
    im = Image.open(path).convert("RGB").resize((a.cell, a.cell))
    x, y = (i % a.cols) * a.cell, (i // a.cols) * (a.cell + STRIP)
    sheet.paste(im, (x, y))
    key = r.times_below_threshold if a.label else r.frac_above_threshold
    d.text((x + 3, y + a.cell + 2),
           "%s  c%s" % (r.image_id[:10], r.centre_id[-1]), fill="white")
    d.text((x + 3, y + a.cell + 14),
           "R %.3f  D %.3f  hard %.2f" % (r.score_r, r.score_d, key), fill="yellow")
    d.rectangle([x, y, x + a.cell - 1, y + a.cell + STRIP - 1], outline="gray")

sheet.save(a.out)
print("wrote %s -- %d images, %dx%d grid" % (a.out, len(df), a.cols, rows))
