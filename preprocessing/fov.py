"""P1 field-of-view detection (plan S4) + a measurement of whether it does anything.

S4-P1: near-black border pixels -> largest central non-black component ->
bounding box -> 2-5% padding -> preserve aspect ratio -> resize. Falls back to
the full frame when detection looks unreliable.
"""
import argparse, glob, os
import numpy as np, pandas as pd, cv2


def detect_fov(bgr, thresh=16, pad=0.03, min_area=0.25):
    """Return (x0, y0, x1, y1) of the endoscopic field, or full frame on failure."""
    h, w = bgr.shape[:2]
    full = (0, 0, w, h)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (g >= thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return full, "no_component"
    cid = lab[h // 2, w // 2]                       # component containing the centre
    if cid == 0:                                    # centre is itself dark -> take largest
        cid = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, cw, ch, area = stats[cid]
    if area < min_area * h * w:                     # implausibly small -> unreliable
        return full, "too_small"
    px, py = int(cw * pad), int(ch * pad)
    return (max(x - px, 0), max(y - py, 0), min(x + cw + px, w), min(y + ch + py, h)), "ok"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="RARE25-train-data")
    p.add_argument("--inventory", default="data_inventory_rare25.csv")
    a = p.parse_args()

    inv = pd.read_csv(a.inventory)
    rows = []
    for r in inv.itertuples():
        f = os.path.join(a.root, r.centre_id, "neo" if r.label else "ndbe", r.image_id + ".png")
        im = cv2.imread(f)
        h, w = im.shape[:2]
        (x0, y0, x1, y1), why = detect_fov(im)
        rows.append(dict(image_id=r.image_id, centre_id=r.centre_id, label=r.label,
                         w=w, h=h, x0=x0, y0=y0, x1=x1, y1=y1, status=why,
                         kept=((x1 - x0) * (y1 - y0)) / (w * h)))
    d = pd.DataFrame(rows)
    d.to_csv("fov_crops.csv", index=False)

    print("detector status:", d.status.value_counts().to_dict())
    print("\nfraction of image area retained by the P1 crop")
    print(d.kept.describe(percentiles=[.01, .05, .25, .5, .75, .95]).round(4).to_string())
    print("\nimages where P1 removes more than...")
    for t in (0.01, 0.02, 0.05, 0.10, 0.20):
        n = (d.kept < 1 - t).sum()
        print("   %4.0f%% of the frame : %4d / %d  (%.1f%%)" % (100 * t, n, len(d), 100 * n / len(d)))
    print("\nby centre (median kept):")
    print(d.groupby("centre_id").kept.median().round(4).to_string())
    print("\nwrote fov_crops.csv")
