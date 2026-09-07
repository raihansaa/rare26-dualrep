"""Extract frozen backbone features once, so probes cost seconds instead of minutes.

The backbone is frozen in training anyway, so nothing about a linear/attention head
requires re-running the encoder. Caching CLS + mean-patch descriptors turns every
downstream question -- feature quality, centre identifiability, head choice --
into a few seconds of scikit-learn.

Also the instrument for the leakage question: if these features identify the
CENTRE almost perfectly, the encoder is carrying acquisition fingerprints, and
validation on centres inside its pretraining corpus is optimistic.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from pipelines import IMAGENET


class Frames(Dataset):
    def __init__(self, df, root, size):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(*IMAGENET)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.root, r.centre_id, "neo" if r.label else "ndbe", r.image_id + ".png")
        return self.tf(Image.open(p).convert("RGB"))


def build(init, size=None):
    """Returns (model, kind, native_size). Architecture is detected from the
    checkpoint so any GastroNet variant can be screened without a flag: the RN50
    releases are flat torchvision state dicts, the DINOv2 one wraps its backbone
    in an EMA teacher, and the ViT-S release is a plain patch16 state dict.

    native_size is what that release was trained to produce; `size` overrides it."""
    import timm
    if init == "dinov2":
        return timm.create_model("vit_base_patch14_dinov2", pretrained=True,
                                 num_classes=0, img_size=size or 336), "vit", 336
    if init.startswith("timm:"):
        # Any stock timm checkpoint by name, e.g. timm:maxvit_tiny_tf_384.in1k. Used
        # to screen a representation WITHOUT the full-fine-tune confound: our own
        # protocol says full fine-tuning is catastrophic here, so a fine-tuned score
        # cannot tell a weak representation apart from a bad training regime.
        # num_classes=0 returns the pooled vector, so this is the "cnn" path.
        name = init.split(":", 1)[1]
        m = timm.create_model(name, pretrained=True, num_classes=0)
        native = m.pretrained_cfg.get("input_size", (3, 384, 384))[-1]
        return m, "cnn", size or native
    sd = torch.load(init, map_location="cpu", weights_only=True)
    if "conv1.weight" in sd:
        from torchvision.models import resnet50
        m = resnet50(weights=None)
        m.fc = torch.nn.Identity()                 # 2048-d global-pooled features
        r = m.load_state_dict(sd, strict=False)
        assert not r.unexpected_keys, "unexpected keys: %s" % r.unexpected_keys[:5]
        assert not r.missing_keys, "missing keys: %s" % r.missing_keys[:5]
        return m, "cnn", 512
    pe = sd.get("patch_embed.proj.weight")
    if pe is not None and pe.shape[0] == 384 and pe.shape[-1] == 16:
        # GastroNet ViT-S/16: 384-d, 12 blocks, no register tokens. Its pos_embed
        # is a 14x14 grid, so 224 is native -- extracting at the ViT-B/14 default
        # of 336 would interpolate pos_embed and measure a representation this
        # release never produces.
        m = timm.create_model("vit_small_patch16_224", pretrained=False,
                              num_classes=0, img_size=size or 224)
        r = m.load_state_dict(sd, strict=False)
        assert not r.unexpected_keys, "unexpected keys: %s" % r.unexpected_keys[:5]
        assert not [k for k in r.missing_keys if not k.startswith("head")], \
            "backbone tensors missing: %s" % r.missing_keys[:5]
        return m, "vit", 224
    from train_dinov2 import load_dinov2_backbone
    m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                          num_classes=0, img_size=size or 336)
    load_dinov2_backbone(m, init)
    return m, "vit", 336


@torch.no_grad()
def extract(m, loader, dev, kind):
    """ViT: CLS and mean-patch concatenated (registers are prefix tokens, not
    patches). CNN: the global-pooled vector, which is already the analogue."""
    npf = m.num_prefix_tokens if kind == "vit" else 0
    out = []
    for x in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if kind == "vit":
                t = m.forward_features(x)
                f = torch.cat([t[:, 0], t[:, npf:].mean(1)], dim=1)
            else:
                f = m(x)
        out.append(f.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="dinov2", help="'dinov2' for stock, or path to a domain .pth")
    p.add_argument("--folds", default="folds_v2.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--size", type=int, default=None, help="default: 336 ViT / 512 CNN")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    df = pd.read_csv(a.folds)
    dev = torch.device("cuda")
    m, kind, native = build(a.init, a.size)   # size only constrains the ViT's pos_embed
    size = a.size or native
    m = m.to(dev).eval()
    dl = DataLoader(Frames(df, a.data_root, size), batch_size=a.batch_size,
                    shuffle=False, num_workers=a.workers, pin_memory=True)
    F = extract(m, dl, dev, kind)
    print("  %s backbone at %dpx" % (kind, size))
    np.savez_compressed(a.out, features=F, image_id=df.image_id.to_numpy(),
                        label=df.label.to_numpy(), centre_id=df.centre_id.to_numpy(),
                        group_id=df.group_id.to_numpy(), fold=df.fold.to_numpy())
    print("wrote %s: %d images x %d dims (init=%s)" % (a.out, *F.shape, a.init))


if __name__ == "__main__":
    # Windows spawns DataLoader workers by re-importing this module; without the
    # guard each worker re-runs extraction and spawns more.
    main()
