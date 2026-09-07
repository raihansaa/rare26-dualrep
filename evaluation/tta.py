"""E12 -- lightweight test-time augmentation (plan S12).

Re-scores the existing OOF folds with the already-trained checkpoints under
four views, saving per-view logits so any subset can be evaluated afterwards
without re-running the network.

  T0  full frame, resized            (identical to P0 inference)
  T1  horizontal flip
  T2  92% centre crop, resized       (see below)
  T3  horizontal flip of T2

S12 specifies T2 as a "slightly wider context crop". That is not reachable
here: P0 already uses the whole frame (P1 was measured to be an identity
transform), so "wider" would mean padding artificial borders in. A mild
centre crop is used instead -- same intent, opposite direction.

S12 requires averaging LOGITS within a model family, which is what happens
here; the sigmoid is applied only after averaging.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet50

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from pipelines import IMAGENET

VIEWS = ("T0", "T1", "T2", "T3")
CROP = 0.92


class CentreCropFrac:
    """Picklable -- a lambda here breaks Windows spawn-based DataLoader workers."""

    def __init__(self, frac):
        self.frac = frac

    def __call__(self, im):
        w, h = im.size
        return T.functional.center_crop(im, [int(h * self.frac), int(w * self.frac)])


def view_transform(name, size):
    pre = [CentreCropFrac(CROP)] if name in ("T2", "T3") else []
    post = [T.RandomHorizontalFlip(p=1.0)] if name in ("T1", "T3") else []
    return T.Compose(pre + [T.Resize((size, size))] + post + [T.ToTensor(), T.Normalize(*IMAGENET)])


class ViewFrames(Dataset):
    def __init__(self, df, root, size, view):
        self.df, self.root = df.reset_index(drop=True), root
        self.tf = view_transform(view, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.root, r.centre_id, "neo" if r.label else "ndbe", r.image_id + ".png")
        return self.tf(Image.open(p).convert("RGB")), np.float32(r.label)


def build(arch, ckpt, size, dev, init=None):
    """init must match the checkpoint's architecture: the domain DINOv2 is the
    reg4 variant and carries reg_token, which the stock model has no slot for."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    if arch == "resnet":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        from train_dinov2 import build_model
        m = build_model(size, 16, 32, 6, init or "dinov2")
    m.load_state_dict(state, strict=True)
    return m.to(dev).eval()


@torch.no_grad()
def logits(model, loader, dev):
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(dev, non_blocking=True)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["resnet", "dinov2"])
    p.add_argument("--runs", default=None, help="template with {f}, e.g. runs/E1n_..._fold{f}")
    p.add_argument("--run", default=None,
                   help="single LOCO checkpoint dir; with --holdout, scores that centre instead "
                        "of looping folds. Pooled OOF has misled repeatedly; prefer this.")
    p.add_argument("--holdout", default=None, help="centre to score in --run mode")
    p.add_argument("--init", default=None,
                   help="backbone the checkpoint was trained from; required for the domain DINOv2")
    p.add_argument("--folds", default="folds_v1.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    size = a.size or (384 if a.arch == "resnet" else 336)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(a.folds)
    assert bool(a.runs) != bool(a.run), "pass either --runs (fold template) or --run (single LOCO dir)"
    if a.run:
        assert a.holdout, "--run needs --holdout to say which centre to score"
        jobs = [(a.holdout, df[df.centre_id == a.holdout].reset_index(drop=True), a.run)]
    else:
        jobs = [(k, df[df.fold == k].reset_index(drop=True), a.runs.format(f=k)) for k in range(5)]

    rows = []
    for tag, va, run in jobs:
        model = build(a.arch, os.path.join(run, "best.pt"), size, dev, a.init)
        rec = va[["image_id", "label", "fold", "centre_id"]].copy()
        for v in VIEWS:
            dl = DataLoader(ViewFrames(va, a.data_root, size, v), batch_size=a.batch_size,
                            shuffle=False, num_workers=a.workers, pin_memory=True)
            rec["logit_" + v] = logits(model, dl, dev)
        rows.append(rec)
        print("%s: %d images x %d views" % (tag, len(va), len(VIEWS)))
        del model
        torch.cuda.empty_cache()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(a.out, index=False)
    print("wrote %s (%d rows)" % (a.out, len(out)))


if __name__ == "__main__":
    main()
