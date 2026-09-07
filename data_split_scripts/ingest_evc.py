"""Ingest the TU/e Early Barrett's (EVC) set as a third acquisition domain.

Our two challenge centres turned out to be one effective domain (within-centre
0.8784 +/- 0.062 vs cross-centre 0.8767 +/- 0.023, indistinguishable), so they
cannot validate robustness. EVC is the only genuinely different acquisition
domain available, which makes leave-one-centre-out possible.

Labels: NDBT (non-dysplastic Barrett tissue) -> 0, ACHD -> 1.
Unlike the challenge data, EVC carries real patient IDs, so grouping here is
exact rather than perceptual-hash guesswork.

Extracts into RARE25-train-data/evc/{ndbe,neo}/ so the existing Frames dataset
picks it up with centre_id='evc' and no code change.
"""
import argparse, os, re, zipfile
import pandas as pd
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--zip", default="EVC_Barretts_FullSet.zip")
p.add_argument("--root", default="RARE25-train-data")
p.add_argument("--out", default="evc_inventory.csv")
a = p.parse_args()

z = zipfile.ZipFile(a.zip)
members = [i for i in z.infolist()
           if i.filename.startswith("images/") and i.filename.lower().endswith(".png")]
print("found %d images in %s" % (len(members), a.zip))

rows = []
for m in members:
    stem = os.path.splitext(os.path.basename(m.filename))[0]
    mt = re.match(r"^(pat\d+)_(im\d+)_(\w+)$", stem)
    if not mt:
        raise ValueError("unexpected filename: %s" % stem)
    patient, imno, path_label = mt.groups()
    label = {"NDBT": 0, "ACHD": 1}[path_label]
    cls = "neo" if label else "ndbe"
    dest_dir = os.path.join(a.root, "evc", cls)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, stem + ".png")
    with z.open(m) as src, open(dest, "wb") as dst:
        dst.write(src.read())
    with Image.open(dest) as im:
        w, h = im.size
        mode = im.mode
    rows.append(dict(image_id=stem, label=label, centre_id="evc",
                     patient_id=patient, image_no=imno, pathology=path_label,
                     original_width=w, original_height=h, mode=mode,
                     source_split="evc"))

df = pd.DataFrame(rows).sort_values("image_id")
df.to_csv(a.out, index=False)

print("\nwrote %s" % a.out)
print(df.groupby(["label", "pathology"]).agg(n=("image_id", "size"),
                                             patients=("patient_id", "nunique")).to_string())
print("\npatients: %d | images per patient: %.1f mean, %d max"
      % (df.patient_id.nunique(), len(df) / df.patient_id.nunique(),
         df.patient_id.value_counts().max()))
print("\nimage sizes:")
print(df.groupby(["original_width", "original_height"]).size().sort_values(ascending=False).head(8).to_string())
print("\ncolour modes:", df["mode"].value_counts().to_dict())
