"""Near-duplicate audit of the training set (RARE26 group-validation audit).

Every A/B decision in this project was scored with a bootstrap clustered on
`group_id`, which comes from the challenge's own `duplicate_group` field --
exact-duplicate numbering, 3078 groups over 3095 images. If endoscopy frames from
the same examination survive as near-duplicates under different image_ids, they
cross fold boundaries freely, every local number is inflated, and the effective
positive count behind each test is smaller than the nominal one.

Three independent detectors, deliberately not combined into one score:

  md5        byte-identical files (catches renamed copies)
  dhash/phash  64-bit perceptual hashes, Hamming distance
  embedding  cosine similarity in the frozen domain-DINOv2 feature bank

Hashes are cached to --cache so thresholds can be re-swept without re-reading
3195 PNGs. Read-only: writes nothing but the cache and its report.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

HASH_BITS = 64
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(gray: Image.Image) -> np.uint64:
    """Horizontal gradient hash: 9x8 -> 64 bits of 'is each pixel > its right neighbour'."""
    a = np.asarray(gray.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return bits_to_uint64(a[:, 1:] > a[:, :-1])


def phash(gray: Image.Image) -> np.uint64:
    """DCT hash: low-frequency 8x8 block (DC dropped) against its median."""
    a = np.asarray(gray.resize((32, 32), Image.BILINEAR), dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    return bits_to_uint64(flat > np.median(flat[1:]))


def bits_to_uint64(bits) -> np.uint64:
    out = np.uint64(0)
    for b in np.asarray(bits).flatten()[:HASH_BITS]:
        out = np.uint64(out << np.uint64(1)) | np.uint64(bool(b))
    return out


def build_cache(folds: pd.DataFrame, root: str, cache: str) -> None:
    paths = {}
    for centre in sorted(os.listdir(root)):
        for cls in sorted(os.listdir(os.path.join(root, centre))):
            d = os.path.join(root, centre, cls)
            for f in os.listdir(d):
                paths[os.path.splitext(f)[0]] = os.path.join(d, f)

    missing = [i for i in folds.image_id if i not in paths]
    assert not missing, "%d image_ids have no file, e.g. %s" % (len(missing), missing[:3])

    md5, dh, ph, wh = [], [], [], []
    for n, iid in enumerate(folds.image_id):
        p = paths[iid]
        md5.append(hashlib.md5(open(p, "rb").read()).hexdigest())
        im = Image.open(p)
        wh.append(im.size)
        gray = im.convert("L")
        dh.append(dhash(gray))
        ph.append(phash(gray))
        if (n + 1) % 500 == 0:
            print("  hashed %d/%d" % (n + 1, len(folds)), flush=True)

    np.savez(cache, image_id=folds.image_id.to_numpy(), md5=np.array(md5),
             dhash=np.array(dh, dtype=np.uint64), phash=np.array(ph, dtype=np.uint64),
             size=np.array(wh))
    print("cached -> %s" % cache)


def hamming_pairs(codes: np.ndarray, thr: int):
    """All i<j with Hamming(codes[i], codes[j]) <= thr. 3195^2 uint64 xors, chunked."""
    n = len(codes)
    out = []
    for i in range(0, n, 256):
        blk = codes[i:i + 256, None] ^ codes[None, :]
        d = POPCOUNT[np.frombuffer(blk.tobytes(), dtype=np.uint8).reshape(blk.shape + (8,))].sum(-1)
        r, c = np.nonzero(d <= thr)
        r += i
        keep = r < c
        out.append(np.stack([r[keep], c[keep]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)


def cosine_pairs(feats: np.ndarray, thr: float):
    x = feats.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    out = []
    for i in range(0, len(x), 256):
        s = x[i:i + 256] @ x.T
        r, c = np.nonzero(s >= thr)
        r += i
        keep = r < c
        out.append(np.stack([r[keep], c[keep]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)


def components(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def describe(name, edges, f):
    """Every reported pair count is split by whether it crosses a fold or a centre."""
    if not len(edges):
        print("  %-28s 0 pairs" % name)
        return
    fold, centre, label = f.fold.to_numpy(), f.centre_id.to_numpy(), f.label.to_numpy()
    a, b = edges[:, 0], edges[:, 1]
    xfold = fold[a] != fold[b]
    xcentre = centre[a] != centre[b]
    pos = (label[a] == 1) | (label[b] == 1)
    print("  %-28s %6d pairs | %5d cross-fold | %4d cross-centre | %5d involve a positive"
          % (name, len(edges), xfold.sum(), xcentre.sum(), pos.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--root", default="RARE25-train-data")
    p.add_argument("--feats", default="feats_domain.npz")
    p.add_argument("--cache", default="dup_hashes.npz")
    p.add_argument("--phash-thr", type=int, default=4, help="Hamming cut for the union")
    p.add_argument("--cos-thr", type=float, default=0.95, help="cosine cut for the union")
    p.add_argument("--pairs-out", default="dup_pairs.csv", help="union pairs, for eyeballing")
    a = p.parse_args()

    f = pd.read_csv(a.folds)
    if not os.path.exists(a.cache):
        print("hashing %d images (no cache at %s)" % (len(f), a.cache))
        build_cache(f, a.root, a.cache)
    h = np.load(a.cache, allow_pickle=True)
    assert (h["image_id"] == f.image_id.to_numpy()).all(), "cache is stale -- delete it"

    bank = np.load(a.feats, allow_pickle=True)
    order = pd.Series(np.arange(len(bank["image_id"])), index=bank["image_id"]).loc[f.image_id]
    feats = bank["features"][order.to_numpy()]

    n = len(f)
    print("\n%d images, %d positives, %d shipped group_ids\n" % (n, f.label.sum(), f.group_id.nunique()))

    print("PAIR COUNTS BY DETECTOR AND THRESHOLD")
    md5_eq = pd.Series(np.arange(n)).groupby(h["md5"]).apply(list)
    md5_edges = np.array([[g[i], g[j]] for g in md5_eq if len(g) > 1
                          for i in range(len(g)) for j in range(i + 1, len(g))]).reshape(-1, 2)
    describe("md5 identical", md5_edges, f)
    for t in (0, 2, 4, 6, 10):
        describe("dhash <= %d" % t, hamming_pairs(h["dhash"], t), f)
    for t in (0, 2, 4, 6, 10):
        describe("phash <= %d" % t, hamming_pairs(h["phash"], t), f)
    for t in (0.999, 0.995, 0.99, 0.98, 0.95):
        describe("domain cosine >= %.3f" % t, cosine_pairs(feats, t), f)

    x = feats.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    nn = []
    for i in range(0, n, 256):
        s = x[i:i + 256] @ x.T
        s[np.arange(len(s)), np.arange(i, i + len(s))] = -1.0   # drop self-match
        nn.append(s.max(1))
    nn = np.concatenate(nn)
    print("\nNEAREST-NEIGHBOUR COSINE (is 0.95 an outlier cut or the bulk of the data?)")
    print("  percentiles  50%% %.3f  90%% %.3f  99%% %.3f  max %.3f"
          % tuple(np.percentile(nn, [50, 90, 99]).tolist() + [nn.max()]))

    # dhash is excluded from the union: at every threshold loose enough to find
    # anything it also fires across centres, which a genuine duplicate cannot do,
    # and single-linkage then chains the whole set into one component.
    edges = np.concatenate([md5_edges,
                            hamming_pairs(h["phash"], a.phash_thr),
                            cosine_pairs(feats, a.cos_thr)])
    comp = components(n, edges)
    f = f.assign(comp=comp)
    pairs = pd.DataFrame(edges, columns=["i", "j"]).drop_duplicates()
    pd.DataFrame({
        "image_a": f.image_id.to_numpy()[pairs.i], "image_b": f.image_id.to_numpy()[pairs.j],
        "fold_a": f.fold.to_numpy()[pairs.i], "fold_b": f.fold.to_numpy()[pairs.j],
        "centre_a": f.centre_id.to_numpy()[pairs.i], "centre_b": f.centre_id.to_numpy()[pairs.j],
        "label_a": f.label.to_numpy()[pairs.i], "label_b": f.label.to_numpy()[pairs.j],
    }).to_csv(a.pairs_out, index=False)
    print("  %d union pairs -> %s" % (len(pairs), a.pairs_out))

    print("\nUNION (md5 | phash <= %d | cosine >= %.2f)" % (a.phash_thr, a.cos_thr))
    sizes = f.groupby("comp").size()
    print("  %d components over %d images (shipped grouping: %d)"
          % (len(sizes), n, f.group_id.nunique()))
    print("  size distribution: %s" % sizes.value_counts().sort_index().to_dict())
    print("  components straddling a fold:   %d" % (f.groupby("comp").fold.nunique() > 1).sum())
    print("  components straddling a centre: %d" % (f.groupby("comp").centre_id.nunique() > 1).sum())
    print("  components mixing labels:       %d" % (f.groupby("comp").label.nunique() > 1).sum())

    print("\nEFFECTIVE INDEPENDENT POSITIVES (what every A/B test was actually powered by)")
    print("  %-10s %8s %8s %8s" % ("centre", "images", "positives", "pos groups"))
    for c, g in f.groupby("centre_id"):
        print("  %-10s %8d %8d %8d"
              % (c, len(g), g.label.sum(), g[g.label == 1].comp.nunique()))
    print("  %-10s %8d %8d %8d"
          % ("ALL", len(f), f.label.sum(), f[f.label == 1].comp.nunique()))


if __name__ == "__main__":
    main()
