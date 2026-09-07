"""RARE26 overlap probe -- a single-model container, built to be read on AUROC.

This is NOT a ranking attempt. It answers one question: does our local validator
measure generalisation, or does it measure GastroNet pretraining exposure?

Line the backbones up on held-out center_1 (EXPERIMENT_LOG.md section 3): stock
DINOv2 ViT-B 0.634 AUPRC, stock DINOv3 ViT-L 0.622, GastroNet DINOv2 ViT-B 0.871.
A 0.25 step that is invariant to a 3.5x capacity increase and a full pretraining
generation is not a representation-quality effect. Either our LoRA config fails to
transfer to a 24-block ViT-L, or the only thing center_1 rewards is having seen
GastroNet -- in which case every arm we selected locally was selected on an
artifact. The dev leaderboard is the one signal not contaminated by that exposure.

The two checkpoints this ships are a MATCHED PAIR. Both are trained on
center_1 + evc (2379 images, 111 positives), same recipe, same folds, same
schedule; they differ only in the pretrained backbone. Submitting both turns a
one-sided test into a controlled A/B, so a negative result is as interpretable as
a positive one.

  probe_dinov3.pt   stock DINOv3 ViT-L/16 (LVD-1689M), LoRA on blocks 18-23
  probe_dinov2.pt   GastroNet-5M DINOv2 ViT-B/14 reg4, LoRA on blocks 6-11

Read AUROC, not PPV@90R. AUROC uses every image in the dev set; PPV@90R is a tail
order statistic over ~100 positives passed through a convex transform, and its CI
here spans sixfold. If the stock backbone lands within ~0.02 AUROC of the
GastroNet arm's 0.816, the local ranking is measuring exposure.

No Platt calibration, deliberately. Calibration is a strictly monotone transform
of a single model's score, so it cannot change AUROC by even a rounding error --
carrying it would add a build step that cannot affect the number being read.

Offline by construction: timm builds with pretrained=False and every weight comes
from the checkpoint. Training used pretrained=True, which reaches for HuggingFace
and would fail under --network none; that asymmetry is the one real trap here, and
it is why the strict load below is worth keeping.
"""

from pathlib import Path
from glob import glob
import json
import os

import numpy as np
import SimpleITK
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

SIZE = 336                          # both arms trained at 336; see pipelines.py
BATCH = 8                           # ViT-L at 336 is 3.5x the ViT-B activation cost
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
LORA_BLOCKS = 6


class LoRALinear(nn.Module):
    """Must match training/train_dinov2.py so state dicts load strictly."""

    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def build(timm_name):
    import timm
    m = timm.create_model(timm_name, pretrained=False, num_classes=1, img_size=SIZE)
    for blk in m.blocks[-LORA_BLOCKS:]:
        blk.attn.qkv = LoRALinear(blk.attn.qkv)
        blk.attn.proj = LoRALinear(blk.attn.proj)
    return m


ARCH = {
    "probe_dinov3.pt": "vit_large_patch16_dinov3",
    "probe_dinov2.pt": "vit_base_patch14_reg4_dinov2",
}


def run():
    interface_key = get_interface_key()
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]
    return handler()


def interface_0_handler():
    images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    if images.shape[-1] != 3:
        raise ValueError("expected channel-last RGB, got %s" % (images.shape,))

    found = [f for f in sorted(os.listdir(RESOURCE_PATH)) if f in ARCH]
    if len(found) != 1:
        raise FileNotFoundError(
            "expected exactly one of %s in %s, found %s"
            % (sorted(ARCH), RESOURCE_PATH, found))
    name = found[0]
    print("probe arm: %s -> %s" % (name, ARCH[name]))

    model = build(ARCH[name])
    ckpt = torch.load(RESOURCE_PATH / name, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tf = T.Compose([T.ToPILImage(), T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(*IMAGENET)])
    probs = 1.0 / (1.0 + np.exp(-batched_logits(model, images, tf, device)))

    print("scored %d frames | min %.4f max %.4f mean %.4f"
          % (len(probs), probs.min(), probs.max(), probs.mean()))
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=[float(p) for p in probs],
    )
    return 0


@torch.no_grad()
def batched_logits(model, images, tf, device):
    out = np.empty(len(images), dtype=np.float64)
    for s in range(0, len(images), BATCH):
        chunk = images[s:s + BATCH]
        x = torch.stack([tf(np.ascontiguousarray(f)) for f in chunk]).to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x).float().squeeze(1)
        else:
            z = model(x).squeeze(1)
        out[s:s + BATCH] = z.cpu().numpy()
    return out


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
