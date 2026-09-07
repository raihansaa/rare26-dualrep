"""Lesion-segmentation decoder on frozen GastroNet DINOv2 patch tokens.

Why this and not `--mil`. The MIL head (EXPERIMENT_LOG.md section 3) inferred lesion
location from image-level labels alone and measured null. It never saw what a lesion
looks like -- it only knew that *somewhere* in a positive image there should be
evidence. This decoder is trained on real masks from an external dataset first, so
lesion appearance is learned before the model ever meets RARE data. That is the
mechanism the RARE25 third-place team used (staged segmentation pretraining, then
classification), and it is a change to the representation, which is the only class of
change that has ever worked on this project.

The backbone stays frozen. Every result here says the frozen GastroNet representation
is nearly the whole system and that adding trainable capacity loses on centre transfer,
so only the decoder learns -- about 2.5M parameters against the backbone's 86M.

Input is a manifest CSV with `image_path,mask_path`, so any mask dataset can be used
without a loader per source. Masks are read as non-zero = lesion; point the manifest at
class-specific mask files if the source separates lesion grades.

Geometric augmentation is applied to image and mask together and is limited to
horizontal flips: the heavy-augmentation floor measured worse here (section 3), and a
mask makes silent train/label desynchronisation easy to introduce and hard to notice.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_dinov2 import load_dinov2_backbone

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
N_BLOCKS = 4          # last four blocks, concatenated -- the DINOv2 linear-probe recipe


class SegFrames(Dataset):
    """Paired image/mask loading. Both are resized to `size`; the mask uses NEAREST so
    label values are never interpolated into non-existent classes."""

    def __init__(self, df, size, train):
        self.df, self.size, self.train = df.reset_index(drop=True), size, train
        self.mean = torch.tensor(IMAGENET[0]).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET[1]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r.image_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mk = Image.open(r.mask_path).resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy((np.asarray(mk) > 0).astype(np.float32).copy())
        if y.ndim == 3:                      # RGB mask -> any channel firing is lesion
            y = y.amax(-1)
        if self.train and torch.rand(1).item() < 0.5:
            x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        return (x - self.mean) / self.std, y.unsqueeze(0)


class Decoder(nn.Module):
    """1x1 mix over the concatenated blocks, one 3x3 for spatial context, 1x1 to logits.

    Deliberately small. A DPT decoder is the fuller version of this and is what the
    third-place team used, but it adds trainable capacity against the one pattern that
    has held throughout this project, and it cannot be validated on the data we hold.
    """

    def __init__(self, in_ch, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 1), nn.GroupNorm(32, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(32, width), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, feats, out_hw):
        z = self.net(feats)
        return F.interpolate(z, size=out_hw, mode="bilinear", align_corners=False)


def dice_loss(logits, y, eps=1.0):
    """Soft Dice. Paired with BCE because lesion pixels are a small minority and BCE
    alone converges to predicting background everywhere."""
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def features(backbone, x):
    outs = backbone.get_intermediate_layers(x, n=N_BLOCKS, reshape=True)
    return torch.cat(outs, dim=1)


@torch.no_grad()
def evaluate(backbone, dec, loader, dev):
    dec.eval()
    inter = union = 0.0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            z = dec(features(backbone, x), y.shape[-2:])
        p = (torch.sigmoid(z.float()) > 0.5).float()
        inter += (p * y).sum().item()
        union += p.sum().item() + y.sum().item()
    dec.train()
    return 2 * inter / max(union, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV with image_path,mask_path")
    p.add_argument("--init", default="dinov2.pth", help="frozen backbone checkpoint")
    p.add_argument("--size", type=int, default=336)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/SEG_decoder")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")

    df = pd.read_csv(a.manifest)
    for c in ("image_path", "mask_path"):
        assert c in df.columns, "manifest needs an %s column, found %s" % (c, list(df.columns))
    missing = [q for q in list(df.image_path) + list(df.mask_path) if not os.path.exists(q)]
    assert not missing, "%d manifest paths do not exist, e.g. %s" % (len(missing), missing[:3])

    idx = np.random.permutation(len(df))
    n_val = max(1, int(round(a.val_frac * len(df))))
    va, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
    print("manifest %d images | train %d | val %d" % (len(df), len(tr), len(va)))

    import timm
    backbone = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                                 num_classes=0, img_size=a.size)
    print("init %s | %d backbone tensors" % (os.path.basename(a.init),
                                             load_dinov2_backbone(backbone, a.init)))
    backbone.eval().to(dev)
    for q in backbone.parameters():
        q.requires_grad_(False)

    dec = Decoder(N_BLOCKS * backbone.embed_dim).to(dev)
    print("decoder %.2fM trainable / backbone %.1fM frozen"
          % (sum(q.numel() for q in dec.parameters()) / 1e6,
             sum(q.numel() for q in backbone.parameters()) / 1e6))

    dl = lambda d, s: DataLoader(SegFrames(d, a.size, s), batch_size=a.batch_size, shuffle=s,
                                 num_workers=a.workers, pin_memory=True, drop_last=s)
    tl, vl = dl(tr, True), dl(va, False)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    for ep in range(a.epochs):
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = dec(features(backbone, x), y.shape[-2:])
            loss = F.binary_cross_entropy_with_logits(z.float(), y) + dice_loss(z.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        d = evaluate(backbone, dec, vl, dev)
        flag = ""
        if d > best:
            best, flag = d, "  *best"
            torch.save({"decoder": dec.state_dict(), "args": vars(a),
                        "in_ch": N_BLOCKS * backbone.embed_dim, "epoch": ep},
                       os.path.join(a.out, "best.pt"))
        print("ep %2d  loss %.4f  val_dice %.4f%s" % (ep, tot / max(len(tr), 1), d, flag), flush=True)

    print("best val Dice %.4f -> %s" % (best, a.out))


if __name__ == "__main__":
    main()
