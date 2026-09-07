"""Score RARE images with a trained segmentation decoder, as a classification arm.

The decoder never sees RARE data during training -- it learns lesion appearance from an
external mask dataset -- so this is zero-shot transfer, not a fold model. There is no
"train on the rest, test on the holdout" structure to respect: one fixed model scores
every image, and the per-centre files exist only so the existing paired-comparison and
fusion tooling can consume them unchanged.

Image score = mean of the top-k patch logits over the 24x24 token grid. The metric is
decided by whether *any* region of a frame carries evidence, so a top-k pool is the
right reduction and a spatial mean is not: averaging over 576 patches would let a large
normal field wash out a small lesion, which is the exact failure the localization route
is meant to fix. k=8 matches the MIL head's default so the two are comparable.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from train_dinov2 import load_dinov2_backbone            # noqa: E402
from train_seg_decoder import Decoder, N_BLOCKS, features  # noqa: E402
import pipelines                                          # noqa: E402


def image_paths(root):
    out = {}
    for centre in sorted(os.listdir(root)):
        for cls in sorted(os.listdir(os.path.join(root, centre))):
            d = os.path.join(root, centre, cls)
            for f in os.listdir(d):
                out[os.path.splitext(f)[0]] = os.path.join(d, f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", default="runs/SEG_decoder/best.pt")
    p.add_argument("--init", default="dinov2.pth")
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--size", type=int, default=336)
    p.add_argument("--topk", type=int, default=8, help="patches pooled per image")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tag", default="seg")
    a = p.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(a.decoder, map_location="cpu", weights_only=False)

    import timm
    backbone = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                                 num_classes=0, img_size=a.size)
    load_dinov2_backbone(backbone, a.init)
    backbone.eval().to(dev)

    dec = Decoder(ck["in_ch"])
    dec.load_state_dict(ck["decoder"], strict=True)
    dec.eval().to(dev)
    print("decoder from %s (epoch %s) | backbone %s"
          % (a.decoder, ck.get("epoch"), os.path.basename(a.init)))

    df = pd.read_csv(a.folds)
    paths = image_paths(a.data_root)
    missing = [i for i in df.image_id if i not in paths]
    assert not missing, "%d images have no file, e.g. %s" % (len(missing), missing[:3])

    tf = pipelines.build("P0", a.size, train=False)
    from PIL import Image
    grid = a.size // backbone.patch_embed.patch_size[0]

    scores = np.empty(len(df), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(df), a.batch_size):
            chunk = df.image_id.iloc[s:s + a.batch_size]
            x = torch.stack([tf(Image.open(paths[i]).convert("RGB")) for i in chunk]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                z = dec(features(backbone, x), (grid, grid)).float()
            flat = z.flatten(1)
            k = min(a.topk, flat.shape[1])
            scores[s:s + len(chunk)] = flat.topk(k, dim=1).values.mean(1).cpu().numpy()
            if (s // a.batch_size) % 50 == 0:
                print("  scored %d/%d" % (s + len(chunk), len(df)), flush=True)

    df = df.assign(score=1.0 / (1.0 + np.exp(-scores)))
    for centre, g in df.groupby("centre_id"):
        out = "runs/SEG_zeroshot_%s_%s" % (centre, a.tag)
        os.makedirs(out, exist_ok=True)
        cols = ["image_id", "label", "centre_id", "fold", "score"]
        g[cols].to_csv(os.path.join(out, "pred_cross_%s.csv" % centre), index=False)
        pos, neg = g.score[g.label == 1], g.score[g.label == 0]
        print("%-10s n=%4d pos=%3d | median score pos %.4f neg %.4f -> %s"
              % (centre, len(g), int(g.label.sum()), pos.median(), neg.median(), out))

    print("\nCompare against any LOCO arm on the same centre, e.g.:")
    print("  python evaluation/compare_loco.py --baseline runs/LOCO_dinov2_rest_to_center_1_domain \\")
    print("      --candidate runs/SEG_zeroshot_center_1_%s --folds %s" % (a.tag, a.folds))


if __name__ == "__main__":
    main()
