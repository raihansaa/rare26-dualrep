"""Throwaway pre-flight: does the real inference path work in the base image?

Not part of the submission -- the Dockerfile COPYs named files only, so this is
never baked into the image. Delete it once the container is verified.

Checks, in order of what would sink the submission:
  1. pip resolution left torch/numpy intact (the numpy 1.x -> 2.x trap)
  2. timm builds vit_base_patch14_reg4_dinov2 at img_size=336
  3. checkpoints written by torch 2.11 load strict=True into this torch
  4. a real forward pass runs on the GPU under autocast
  5. SimpleITK reads the test tiff
"""

import sys
from glob import glob
from pathlib import Path

import numpy
import torch

print("torch       %s  (cuda %s, available=%s)"
      % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
print("numpy       %s" % numpy.__version__)

import timm
print("timm        %s" % timm.__version__)

import SimpleITK
print("SimpleITK   %s" % SimpleITK.__version__)
print("-" * 60)

# numpy 1.x/2.x mismatch does not raise on import -- it raises on the first op
# that crosses the C boundary, so force one.
_ = torch.from_numpy(numpy.zeros(4, dtype=numpy.float32)).sum()
print("[ok] torch<->numpy bridge intact")

sys.path.insert(0, "/w")
import inference  # module-level only; run() is guarded by __main__

model = inference.build_dinov2()
n_params = sum(p.numel() for p in model.parameters())
print("[ok] built vit_base_patch14_reg4_dinov2 @ %d px, %.1fM params"
      % (inference.DINOV2_SIZE, n_params / 1e6))

paths = sorted(glob("/w/resources/dinov2_fold*.pt"))
if len(paths) != 5:
    raise SystemExit("expected 5 checkpoints, found %d" % len(paths))

for p in paths:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print("[ok] strict load %s" % Path(p).name)
    del ckpt, state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
x = torch.randn(2, 3, inference.DINOV2_SIZE, inference.DINOV2_SIZE, device=device)
with torch.no_grad():
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float().squeeze(1)
    else:
        z = model(x).squeeze(1)
print("[ok] forward pass on %s -> logits %s" % (device.type, z.tolist()))

tif = glob("/w/test/input/interface_0/images/stacked-barretts-esophagus-endoscopy/*.tif*")
arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(tif[0]))
print("[ok] SimpleITK read %s -> %s %s" % (Path(tif[0]).name, arr.shape, arr.dtype))

print("-" * 60)
print("PREFLIGHT PASSED")
