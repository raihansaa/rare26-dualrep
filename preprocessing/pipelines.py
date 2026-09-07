"""Preprocessing arms for E2 (plan S4).

P0  direct resize + model normalisation (the E1 baseline)
P3  P0 + mask the border region that is dark in >50% of either centre,
    normalising the artificial border geometry across centres
P4  P0 + per-image grey-world white balance

P1 (FOV crop) and P2 (FOV mask) are omitted deliberately: measured on all 3095
RARE25 images, the S4 detector retains 100.0% of every frame (std 0.0), so both
are exact identity transforms of P0. See preprocessing/fov.py.

P4 is per-image on purpose. Centre identity is unknown at inference, so any
per-centre correction would be untrainable in the container.
"""
import random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
DARKMAP = "fov_darkmap.npy"   # (2, 64, 64) per-centre P(pixel < 8)


class MaskBorder:
    """Zero the pixels that are near-black in >50% of images at either centre."""

    def __init__(self, size, path=DARKMAP, thresh=0.5):
        m = np.load(path)
        union = ((m[0] > thresh) | (m[1] > thresh)).astype(np.float32)
        t = torch.from_numpy(union)[None, None]
        self.keep = 1.0 - F.interpolate(t, size=(size, size), mode="nearest")[0]

    def __call__(self, x):
        return x * self.keep


class GreyWorld:
    """Per-image white balance: equalise channel means, preserving overall level."""

    def __call__(self, x):
        m = x.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (x * m.mean() / m).clamp(0, 1)


class Gamma:
    def __init__(self, lo=0.90, hi=1.10):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import random
        return T.functional.adjust_gamma(im, random.uniform(self.lo, self.hi))


class Jpeg:
    """Re-encode at a random quality. Quality stays >=80: below that, blocking
    artifacts counterfeit the irregular mucosal texture that defines the
    positive class."""

    def __init__(self, lo=80, hi=100):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import io, random
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=random.randint(self.lo, self.hi))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Downsample:
    """Mild resolution loss. Floor of 85%: vessels and pits span only a few
    pixels at 384, and heavier downsampling removes them irreversibly."""

    def __init__(self, lo=0.85, hi=1.00):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import random
        w, h = im.size
        f = random.uniform(self.lo, self.hi)
        small = im.resize((max(int(w * f), 8), max(int(h * f), 8)), Image.LANCZOS)
        return small.resize((w, h), Image.BILINEAR)


class PilNoise:
    def __init__(self, lo=0.002, hi=0.010):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import random
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = a + np.random.randn(*a.shape).astype(np.float32) * random.uniform(self.lo, self.hi)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


class Sharpen:
    """Endoscopy processors apply edge enhancement; its strength differs by
    centre and by operator setting."""

    def __init__(self, lo=0.90, hi=1.15):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import random
        from PIL import ImageEnhance
        return ImageEnhance.Sharpness(im).enhance(random.uniform(self.lo, self.hi))


class Illumination:
    """Smooth spatial gain -- light-guide falloff and vignetting vary by scope."""

    def __init__(self, lo=0.90, hi=1.10):
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        import random
        _, h, w = x.shape
        gy = torch.linspace(-1, 1, h).view(h, 1)
        gx = torch.linspace(-1, 1, w).view(1, w)
        ang = random.uniform(0, 6.283)
        ramp = gx * float(np.cos(ang)) + gy * float(np.sin(ang))
        amp = random.uniform(self.lo, self.hi) - 1.0
        return (x * (1.0 + amp * ramp)).clamp(0, 1)


class ChannelGain:
    """Mild per-channel white-balance drift, the honest form of the colour
    invariance we want: the lesion stays redder than its surround under a
    global device shift. P4 removed colour outright and lost cross-centre
    transfer; this perturbs it slightly instead."""

    def __init__(self, lo=0.97, hi=1.03):
        self.lo, self.hi = lo, hi

    def __call__(self, im):
        import random
        a = np.asarray(im, dtype=np.float32)
        g = np.array([random.uniform(self.lo, self.hi) for _ in range(3)], dtype=np.float32)
        return Image.fromarray(np.clip(a * g, 0, 255).astype(np.uint8))


class Jigsaw:
    """Permute a grid of tiles: destroys global layout, preserves local texture.

    A different mechanism from the photometric `acq` family, which was measured
    null and whose target (acquisition shift) was later refuted outright. This one
    attacks a shortcut that is documented to exist here: centre identity is
    recoverable from frozen features at 99.4%, and centre alone predicts the label
    at AUROC 0.685, so global anatomical composition is an available and misleading
    cue. Early neoplasia is read from local vascular and pit-pattern irregularity,
    which survives a tile permutation; the layout that identifies the centre does not.

    Unlike cutout or hide-and-seek -- excluded on clinical grounds in
    `acquisition_aug` -- no pixels are removed, so a positive image still contains
    its lesion and the label stays valid.
    """

    def __init__(self, grid=3, p=0.5):
        self.grid, self.p = grid, p

    def __call__(self, im):
        if random.random() > self.p:
            return im
        g = self.grid
        w, h = im.size
        tw, th = w // g, h // g
        cells = [(r, c) for r in range(g) for c in range(g)]
        tiles = [im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r, c in cells]
        random.shuffle(tiles)
        out = im.copy()
        for t, (r, c) in zip(tiles, cells):
            out.paste(t, (c * tw, r * th))
        return out


def acquisition_aug(size):
    """Pathology-preserving acquisition randomisation (clinically reviewed).

    Early Barrett neoplasia is often a flat lesion read from subtle vascular
    and pit-pattern irregularity plus focal reddish discolouration. Severities
    are therefore deliberately mild, and the families are mutually exclusive
    (OneOf) rather than stacked: stacking blur, JPEG and downsampling can
    remove the lesion outright, which turns augmentation into label noise.

    Excluded on clinical grounds: grayscale, inversion, solarisation, large hue
    shifts, cutout/erasing, elastic warping (fold loss and subtle elevation are
    themselves diagnostic), and synthetic occlusions such as mucus, bubbles or
    specular highlights -- without lesion masks those can cover the finding and
    invalidate the positive label.
    """
    photometric = T.RandomChoice([
        T.ColorJitter(brightness=(0.85, 1.15)),
        T.ColorJitter(contrast=(0.90, 1.10)),
        Gamma(),
        T.ColorJitter(saturation=(0.90, 1.10), hue=(-0.008, 0.008)),
        ChannelGain(),
    ])
    degradation = T.RandomChoice([
        T.GaussianBlur(5, sigma=(0.2, 0.6)),
        Jpeg(),
        Downsample(),
        PilNoise(),
        Sharpen(),
    ])
    body = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.97, 1.03))], p=0.35),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.015, p=1.0)], p=0.06),
        T.RandomApply([photometric], p=0.55),
        T.RandomApply([degradation], p=0.20),
    ])
    # Square stretch is kept deliberately: the challenge itself stretches to
    # 512x512 before we see the data (verified on the example TIFF -- content
    # fills the frame, aspect 1.000, no letterbox). Aspect-preserving padding
    # here would create a train/test mismatch.
    return [T.Resize((size, size)), T.RandomApply([body], p=0.75)]   # 25% left clean


def build(preproc, size, train, aug="none"):
    if preproc not in ("P0", "P3", "P4"):
        raise ValueError("unknown preprocessing %r (P1/P2 are identity -- see module docstring)" % preproc)
    if aug not in ("none", "acq", "jigsaw", "heavy"):
        raise ValueError("unknown aug %r" % aug)
    post = {"P0": [], "P3": [MaskBorder(size)], "P4": [GreyWorld()]}[preproc]
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor()] + post + [T.Normalize(*IMAGENET)])
    if aug == "acq":
        pre = acquisition_aug(size)
        post = post + [T.RandomApply([Illumination()], p=0.12)]
    elif aug == "jigsaw":
        pre = [T.Resize((size, size)), T.RandomHorizontalFlip(), Jigsaw()]
    elif aug == "heavy":
        # The RARE25 winner applies this geometric+photometric floor to EVERY one of
        # their 40 models and varies only the preset on top of it. Our "augmentation
        # is null" result (EXPERIMENT_LOG.md 3) was measured by adding one preset to a
        # no-augmentation baseline, which is a different regime and does not test this.
        # Jigsaw is retained because it is the one preset element already measured to
        # pay here, so the single variable against the `jigsaw` arm is the floor itself.
        # Optical distortion and elastic are omitted: they need albumentations, which
        # is not installed, and adding a dependency to the container is a separate risk.
        pre = [T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
               T.RandomHorizontalFlip(0.5),
               T.RandomVerticalFlip(0.5),
               T.RandomApply([T.RandomRotation(15)], p=0.8),
               T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
               Jigsaw()]
    else:
        pre = [T.Resize((size, size)), T.RandomHorizontalFlip()]
    return T.Compose(pre + [T.ToTensor()] + post + [T.Normalize(*IMAGENET)])
