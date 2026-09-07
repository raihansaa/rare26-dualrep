"""RARE26 container inference -- drop-in replacement for the template's inference.py.

Final system: P0 preprocessing -> two independent backbone arms -> per-model affine
calibration -> mean within arm -> equal-weight mean across arms.

  arm A  five domain-pretrained DINOv2-Base-LoRA folds  (336px, ViT-B/14 reg4)
  arm B  five GastroNet+SWSL ResNet50 folds, layer4+fc  (384px, torchvision RN50)

Why two arms rather than one backbone in triplicate. The previous submission
shipped three training recipes over a single backbone; it cut in-distribution
FPR@90R by 39% and moved the leaderboard by 1.2 points, because members sharing a
representation fail on the same images. These two arms have Spearman 0.48-0.59
against each other and share only ~50% of the bottom-decile positives -- the
images that set the 90%-recall threshold. On leave-one-centre-out the fusion is
positive on both centres and on all three metrics (center_1 AUPRC +0.0290,
P(gain) 0.945; center_2 +0.0018, P(gain) 0.703), which no previous ensemble
change achieved. See EXPERIMENT_LOG.md section 3.

Why probabilities and not logits. The arms are different architectures on
different pretraining corpora, so their logit scales are not comparable -- fitted
Platt slopes range 0.46 to 1.19 and intercepts -3.25 to +1.98. Averaging raw
scores without calibration measured *negative* on both held-out centres
(center_2 AUPRC -0.0187, P(gain) 0.039). Each checkpoint therefore carries
(a, b, prior_fit) in platt.json, frozen at build time: the affine calibration
fitted on its own out-of-fold predictions -- data that model never trained on.
The third element is the prevalence of that slice; it is retained in the file but
is no longer read, the prior shift having been removed (below).

Deliberately NOT percentile-rank fusion. Ranks would be computed within whatever
stack the container is handed, while the metric is pooled over the entire test
set, so per-stack normalisation would make every prediction depend on its batch's
composition. Frozen offline calibration has no such dependence.

Weights are equal by predeclaration, not by search. With two usable centres any
fitted fusion weight is selected on the data that would validate it; that is what
put +0.12 of pure optimism into an earlier fusion.

Why the prior shift and noisy-OR were REMOVED (2026-08-26). They were added
2026-08-19 on a leave-one-centre-out measurement (center_1 FPR@90R 0.0140 -> 0.0108,
a 22.6% cut) and shipped as submission 4. Externally they did not transfer. Against
submission 3: PPV@90R 0.0151 -> 0.0150, unchanged within noise, while AUPRC fell
0.3426 -> 0.3187. That is the AUPRC cost a correct tail intervention is supposed to
pay, with none of the tail gain it is supposed to buy.

The same container also scored AUROC 0.8195 / PPV 0.0150 on the main set and AUROC
0.8889 / PPV 0.0155 on the RARE25 validation set -- +0.069 AUROC for +0.0005 PPV.
Ranking quality and the scored metric are decoupled in this regime, and local tail
measurements do not carry into it: on held-out centres these models beat the FPR@90R
their own AUROC implies (gap -0.4 to -3.1 pp) while on the leaderboard they lose 9.5
pp to it. No local perturbation available reaches within 0.15 AUROC of the scored
regime, so LOCO cannot measure interventions of this kind at all.

This file therefore returns to plain `prob_nat`: per-model Platt, mean within arm,
equal-weight mean across arms. That is the exact configuration that passed the
predeclared two-arm LOCO gate (center_1 AUPRC 0.9012, all six deltas positive),
before any tail machinery was added on top of it.

Offline by construction. Every model is built with pretrained/weights disabled and
loaded from checkpoints baked into resources/, so nothing touches the network
under --network=none. Models are loaded one at a time and released, so peak memory
is one model rather than ten -- the platform GPU is unknown and may be smaller
than the machine these were trained on.
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
from torchvision.models import resnet50

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
PLATT_PATH = Path("platt.json")

DINOV2_SIZE = 336                   # 336/14 = 24 patches, the checkpoint's native grid
RESNET_SIZE = 384                   # train_resnet.py ARCH["resnet"] native size
BATCH = 16
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


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


def build_dinov2():
    """reg4 variant: the domain checkpoints carry 4 register tokens, which the
    stock vit_base_patch14_dinov2 has no slot for and would refuse to load."""
    import timm
    m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                          num_classes=1, img_size=DINOV2_SIZE)
    for blk in m.blocks[-6:]:
        blk.attn.qkv = LoRALinear(blk.attn.qkv)
        blk.attn.proj = LoRALinear(blk.attn.proj)
    return m


def build_resnet50():
    """weights=None: the GastroNet+SWSL trunk comes from the checkpoint, and the
    container must not reach for torchvision's ImageNet download."""
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


ARMS = (
    ("dinov2_fold*.pt", build_dinov2, DINOV2_SIZE),
    ("swsl_fold*.pt", build_resnet50, RESNET_SIZE),
)


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

    platt = load_json_file(location=PLATT_PATH)
    arms = [arm_probs(images, pattern, builder, size, device, platt)
            for pattern, builder, size in ARMS]
    # Equal-weight mean across arms, by predeclaration rather than search. Averaged
    # rather than OR'd: noisy-OR was measured externally and did not transfer, and it
    # saturates with member count, making the output depend on the ensemble's size.
    probs = np.mean(arms, axis=0)

    print("scored %d frames | min %.4f max %.4f mean %.4f"
          % (len(probs), probs.min(), probs.max(), probs.mean()))
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=[float(p) for p in probs],
    )
    return 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def arm_probs(images, pattern, builder, size, device, platt):
    """Mean calibrated probability over one arm's fold checkpoints."""
    paths = sorted(glob(str(RESOURCE_PATH / pattern)))
    if not paths:
        raise FileNotFoundError("no %s under %s" % (pattern, RESOURCE_PATH))
    tf = T.Compose([T.ToPILImage(), T.Resize((size, size)), T.ToTensor(), T.Normalize(*IMAGENET)])
    acc = np.zeros(len(images), dtype=np.float64)
    for p in paths:
        name = os.path.basename(p)
        if name not in platt:
            raise KeyError("no calibration for %s in %s" % (name, PLATT_PATH))
        a, b = platt[name][:2]
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = builder()
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        acc += sigmoid(a * batched_logits(model, images, tf, device) + b)
        del model, state, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("  %-18s %d checkpoints, calibrated, averaged" % (pattern, len(paths)))
    return acc / len(paths)


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
