"""Acquisition-shift stress test -- does the LOCO model break the way the leaderboard says?

LOCO holds out one centre, but all three of our centres are Dutch and from the
consortium that produced the GastroNet pretraining corpus. The leaderboard tests
twelve heterogeneous centres, where FPR@90R is 60% against LOCO's 0.4-3.2%. LOCO
cannot see that gap, so this substitutes synthetic acquisition shift for the real
thing: score the held-out centre under graded device-style corruptions and watch
where the operating point collapses.

This does NOT estimate leaderboard performance -- absolute numbers here mean
nothing. It is a *relative* probe, for two purposes:

  1. confirm or refute the diagnosis. If mild colour or compression shift moves
     FPR@90R from 3% toward 40%, acquisition shift explains the leaderboard gap.
     If the model barely degrades, the diagnosis is wrong and something else is.
  2. give us a selection signal LOCO cannot provide -- worst-domain FPR@90R --
     for choosing an augmentation recipe.

Severities are graded past the clinically-safe range used for augmentation in
preprocessing/pipelines.py. That is deliberate: augmentation must preserve the
lesion, a stress test must find the breaking point.

Usage:
  python evaluation/shift_stress.py --run runs/LOCO_dinov2_rest_to_center_1_domain
"""
import argparse, io, os, sys

import numpy as np, pandas as pd, torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from metric import mode_a_weighted
from pipelines import build as build_preproc


# --- corruptions: deterministic, applied at native resolution before resize,
# --- because acquisition shift happens at capture time.

# Classes, not closures: Windows spawns dataloader workers, which pickles the
# dataset and everything it holds. A lambda or a nested function cannot be
# pickled and the workers die on the first corruption.

class Gamma:
    def __init__(self, g):
        self.tbl = [min(255, int(round(255.0 * (i / 255.0) ** g))) for i in range(256)] * 3

    def __call__(self, im):
        return im.point(self.tbl)


class ChannelGain:
    """Per-channel white-balance drift -- the most likely difference between
    endoscope processors from different manufacturers."""

    def __init__(self, r, g, b):
        self.gain = np.array([r, g, b], dtype=np.float32)

    def __call__(self, im):
        a = np.asarray(im, dtype=np.float32) * self.gain
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class Enhance:
    ENHANCERS = {"brightness": ImageEnhance.Brightness,
                 "contrast": ImageEnhance.Contrast,
                 "sharpness": ImageEnhance.Sharpness}

    def __init__(self, kind, factor):
        self.kind, self.factor = kind, factor

    def __call__(self, im):
        return self.ENHANCERS[self.kind](im).enhance(self.factor)


class Jpeg:
    def __init__(self, q):
        self.q = q

    def __call__(self, im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Blur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, im):
        return im.filter(ImageFilter.GaussianBlur(self.sigma))


class Downsample:
    def __init__(self, frac):
        self.frac = frac

    def __call__(self, im):
        w, h = im.size
        small = im.resize((max(int(w * self.frac), 8), max(int(h * self.frac), 8)), Image.LANCZOS)
        return small.resize((w, h), Image.BILINEAR)


class Noise:
    def __init__(self, sigma, seed=0):
        self.sigma, self.seed = sigma, seed

    def __call__(self, im):
        rng = np.random.default_rng(self.seed)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = a + rng.standard_normal(a.shape).astype(np.float32) * self.sigma
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


class Vignette:
    """Light-guide falloff: scopes differ in how far illumination reaches the rim."""

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, im):
        w, h = im.size
        y, x = np.mgrid[0:h, 0:w]
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
        m = np.clip(1.0 - self.strength * np.clip(r, 0, 1) ** 2, 0, 1)[..., None]
        return Image.fromarray(np.clip(np.asarray(im, np.float32) * m, 0, 255).astype(np.uint8))


def battery():
    """(family, label, fn) at three severities each, mild -> severe."""
    out = [("clean", "clean", None)]
    for g in [0.75, 0.60, 0.45]:
        out.append(("gamma_dark", "gamma %.2f" % g, Gamma(g)))
    for g in [1.35, 1.70, 2.10]:
        out.append(("gamma_bright", "gamma %.2f" % g, Gamma(g)))
    for r, b in [(1.06, 0.94), (1.14, 0.87), (1.22, 0.80)]:
        out.append(("warm_shift", "wb r%.2f b%.2f" % (r, b), ChannelGain(r, 1.0, b)))
    for r, b in [(0.94, 1.06), (0.87, 1.14), (0.80, 1.22)]:
        out.append(("cool_shift", "wb r%.2f b%.2f" % (r, b), ChannelGain(r, 1.0, b)))
    for f in [0.85, 0.72, 0.60]:
        out.append(("dim", "brightness %.2f" % f, Enhance("brightness", f)))
    for f in [0.85, 0.70, 0.55]:
        out.append(("low_contrast", "contrast %.2f" % f, Enhance("contrast", f)))
    for f in [1.4, 1.8, 2.4]:
        out.append(("oversharp", "sharpness %.1f" % f, Enhance("sharpness", f)))
    for q in [60, 40, 25]:
        out.append(("jpeg", "quality %d" % q, Jpeg(q)))
    for s in [1.0, 2.0, 3.5]:
        out.append(("blur", "sigma %.1f" % s, Blur(s)))
    for f in [0.60, 0.40, 0.25]:
        out.append(("downsample", "scale %.2f" % f, Downsample(f)))
    for s in [0.02, 0.05, 0.09]:
        out.append(("noise", "sigma %.3f" % s, Noise(s)))
    for s in [0.30, 0.50, 0.70]:
        out.append(("vignette", "strength %.2f" % s, Vignette(s)))
    return out


class CorruptFrames(Dataset):
    def __init__(self, df, root, size, corrupt, preproc="P0"):
        self.df, self.root, self.corrupt = df.reset_index(drop=True), root, corrupt
        self.tf = build_preproc(preproc, size, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.root, r.centre_id, "neo" if r.label else "ndbe", r.image_id + ".png")
        im = Image.open(p).convert("RGB")
        if self.corrupt is not None:
            im = self.corrupt(im)
        return self.tf(im), np.float32(r.label)


@torch.no_grad()
def score(model, loader, dev):
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(dev, non_blocking=True)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def fpr_at(y, s, thr):
    neg = y == 0
    return float((s[neg] >= thr).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="LOCO run dir holding best.pt")
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ck = torch.load(os.path.join(a.run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["args"]
    holdout = cfg.get("holdout") or cfg["test_centre"]
    size = cfg["size"]
    print("run %s | holdout %s | arch %s | size %d" % (a.run, holdout, cfg["arch"], size))

    if cfg["arch"] != "dinov2":
        raise SystemExit("stress test currently assumes the dinov2 arm")
    from train_dinov2 import build_model
    # runs predating --lora-blocks carry no such key; 6 was the behaviour then and
    # remains the default. A wrong guess cannot pass silently -- the state dict
    # would carry a different number of LoRA tensors and strict=True would reject it.
    model = build_model(size, 16, 32, cfg.get("lora_blocks", 6), cfg["init"])
    model.load_state_dict(ck["model"], strict=True)
    dev = torch.device("cuda")
    model.to(dev).eval()

    df = pd.read_csv(a.folds)
    tst = df[df.centre_id == holdout].reset_index(drop=True)
    y = tst.label.to_numpy()
    print("held-out %s: %d images, %d positives\n" % (holdout, len(tst), y.sum()))

    rows = []
    print("%-14s %-18s %8s %8s %8s %8s" % ("family", "severity", "PPV@90R", "FPR@90R", "AUPRC", "AUROC"))
    for family, label, fn in battery():
        dl = DataLoader(CorruptFrames(tst, a.data_root, size, fn, cfg["preproc"]),
                        batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
        s = 1 / (1 + np.exp(-score(model, dl, dev)))
        m = mode_a_weighted(y, s)
        f = fpr_at(y, s, m["threshold"])
        rows.append(dict(family=family, severity=label, ppv90=m["ppv"], fpr90=f,
                         auprc=m["auprc"], auroc=m["auroc"]))
        print("%-14s %-18s %8.4f %8.4f %8.4f %8.4f"
              % (family, label, m["ppv"], f, m["auprc"], m["auroc"]))

    out = a.out or os.path.join(a.run, "shift_stress.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    base = rows[0]
    worst = min(rows[1:], key=lambda r: r["ppv90"])
    print("\nclean          PPV@90R %.4f  FPR@90R %.4f" % (base["ppv90"], base["fpr90"]))
    print("worst (%s %s)  PPV@90R %.4f  FPR@90R %.4f"
          % (worst["family"], worst["severity"], worst["ppv90"], worst["fpr90"]))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
