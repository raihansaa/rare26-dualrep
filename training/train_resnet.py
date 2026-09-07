"""E1 -- ResNet50 P0 baseline, single fold (plan S13 E1, config S1, preproc S4-P0).

P0 = read image, keep RGB, resize to input_size, apply model normalisation.
Writes a checkpoint plus OOF predictions in the S7 column format.
"""
import argparse, hashlib, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from metric import mode_a_weighted
from pipelines import build as build_preproc, IMAGENET


class Frames(Dataset):
    def __init__(self, df, root, size, train, preproc="P0", weights=None, aug="none"):
        self.df, self.root = df.reset_index(drop=True), root
        self.tf = build_preproc(preproc, size, train, aug)
        # per-sample loss weight (plan S9 hard-example reweighting); 1.0 by default
        self.w = (self.df.image_id.map(weights).fillna(1.0).to_numpy(np.float32)
                  if weights is not None else np.ones(len(self.df), np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.root, r.centre_id, "neo" if r.label else "ndbe", r.image_id + ".png")
        return self.tf(Image.open(p).convert("RGB")), np.float32(r.label), self.w[i]


def group_balanced_weights(df):
    """Equal aggregate loss per (centre x class) cell.

    A global pos_weight rescales classes but leaves centre predictive of label
    inside the weighted distribution -- our centres differ 4.4x in positive rate
    (and EVC is 50%), so centre alone predicts the label at AUROC 0.685. This
    equalises every cell instead, removing that route. Must NOT be stacked with
    a pos_weight: doubling the positive multiplier is what broke E6.
    """
    key = df.centre_id.astype(str) + "|" + df.label.astype(str)
    n = key.map(key.value_counts())
    w = (len(df) / (key.nunique() * n)).astype("float32")
    return pd.Series(w.to_numpy(), index=df.image_id.to_numpy())


ARCH = {
    # name -> (timm/torchvision id, native input size)
    "resnet":      (None, 384),                        # torchvision ResNet50
    "resnet512":   (None, 512),
    "maxvit":      ("maxvit_tiny_tf_512.in1k", 512),   # RARE25 runner-up used MaxViT-Tiny
    "maxvit384":   ("maxvit_tiny_tf_384.in1k", 384),
}


def load_backbone(m, path):
    """Load a backbone-only ResNet50 state dict (GastroNet DINOv1 checkpoints).

    strict=False tolerates a total key mismatch in silence and would leave the
    backbone randomly initialised, so the match is asserted rather than trusted.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    r = m.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys, "unexpected keys: %s" % r.unexpected_keys[:5]
    assert set(r.missing_keys) <= {"fc.weight", "fc.bias"}, \
        "backbone tensors missing: %s" % [k for k in r.missing_keys if not k.startswith("fc.")][:5]
    return len(sd)


def build_arch(arch, num_classes=1, init="imagenet"):
    """MaxViT window sizes are tied to input resolution, so the variant must
    match the size -- hence the separate 384/512 entries rather than a resize."""
    if arch.startswith("resnet"):
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if init == "imagenet" else None)
        if init != "imagenet":
            print("init %s | %d backbone tensors | sha256 %s"
                  % (os.path.basename(init), load_backbone(m, init),
                     hashlib.sha256(open(init, "rb").read()).hexdigest()[:16]))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    import timm
    return timm.create_model(ARCH[arch][0], pretrained=True, num_classes=num_classes)


def balanced_bce(logits, y, pos_weight_unused=None):
    """0.5*mean(positive loss) + 0.5*mean(negative loss).

    Equalises the two classes without a pos_weight multiplier, so nothing can
    stack on top of it. Stacking is what turned S9's nominal 2.5x into an
    effective ~50x and destroyed E6.
    """
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pos, neg = y > 0.5, y <= 0.5
    terms = []
    if pos.any():
        terms.append(l[pos].mean())
    if neg.any():
        terms.append(l[neg].mean())
    return sum(terms) / len(terms)


def freeze_batchnorm(model, affine=True):
    """Keep BN in eval mode so running stats stay at ImageNet values.

    BN running statistics adapt to the training centres' image statistics and
    become an acquisition-domain encoder; on unseen centres they are simply
    wrong. Must be re-applied after every model.train() call.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            n += 1
            if affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
    return n


RESNET_STAGES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
MAXVIT_STAGES = ("stem", "stages.0", "stages.1", "stages.2", "stages.3")


def freeze_until(model, stage):
    """Freeze the stem and every stage up to and including `stage`.

    The CNN analogue of restricting LoRA to the trailing blocks. Every reduction
    in trainable capacity has survived centre transfer better than the larger one
    on this project, and a full fine-tune of this arch was catastrophic.

    Returns the frozen stage names so the caller can re-assert eval() on them
    after each model.train() -- a trunk whose BatchNorm running statistics still
    track the training centres is not frozen, it is just slower to adapt.
    """
    if hasattr(model, "stages"):
        # timm MaxViT-style: stem + stages.0..3. `stage3` here is the analogue of
        # ResNet's layer3 cut -- train the last stage, the norm and the head only.
        order = MAXVIT_STAGES
        frozen = set(order[:order.index(stage) + 1])
        n = 0
        for name, p in model.named_parameters():
            if any(name == f or name.startswith(f + ".") for f in frozen):
                p.requires_grad_(False)
                n += p.numel()
        return frozen, n
    frozen = set(RESNET_STAGES[:RESNET_STAGES.index(stage) + 1])
    n = 0
    for name, p in model.named_parameters():
        if name.split(".")[0] in frozen:
            p.requires_grad_(False)
            n += p.numel()
    return frozen, n


def apply_frozen_eval(model, frozen):
    """Re-assert eval() on frozen stages; model.train() undoes it every epoch.

    Matches nested names too ("stages.0"), so a MaxViT trunk frozen stage-by-stage
    keeps its normalisation statistics fixed rather than tracking the training
    centres -- the same requirement that made the ResNet freeze work.
    """
    for name, m in model.named_modules():
        if name in frozen:
            m.eval()


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for x, *_ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(dev, non_blocking=True)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v1.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--preproc", default="P0", choices=["P0", "P3", "P4"])
    p.add_argument("--inner-splits", type=int, default=5)  # 1/5 of train folds -> early stopping
    p.add_argument("--weights", default=None, help="CSV with image_id,weight (plan S9); absent -> 1.0")
    p.add_argument("--arch", default="resnet", choices=sorted(ARCH))
    p.add_argument("--init", default="imagenet",
                   help="'imagenet' or path to a backbone-only ResNet50 .pth (GastroNet)")
    p.add_argument("--aug", default="none", choices=["none", "acq"])
    p.add_argument("--loss", default="posweight", choices=["posweight", "balanced"])
    p.add_argument("--loss-balance", default="global", choices=["global", "centre"])
    p.add_argument("--freeze-bn", action="store_true")
    p.add_argument("--freeze-until", default="none",
                   choices=["none", "layer1", "layer2", "layer3", "layer4"],
                   help="resnet only: freeze the stem and stages up to and including this "
                        "one, leaving the rest trainable; layer3 = train layer4+fc, the "
                        "CNN analogue of LoRA-6")
    p.add_argument("--size", type=int, default=None)   # arch native size
    p.add_argument("--batch-size", type=int, default=32)   # 64 spills to shared RAM on 8GB
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.out = 2, a.out or "runs/smoke"
    if a.size is None:
        a.size = ARCH[a.arch][1]
    out = a.out or ("runs/E1_%s_p0_fold%d" % (a.arch, a.fold) if a.preproc == "P0"
                    else "runs/E2_%s_%s_fold%d" % (a.arch, a.preproc.lower(), a.fold))
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    df = pd.read_csv(a.folds)
    tr, va = df[df.fold != a.fold].reset_index(drop=True), df[df.fold == a.fold]
    # Early stopping must not see the OOF fold, or the reported predictions come
    # from a checkpoint selected using those same predictions (plan S6 audit).
    icv = StratifiedGroupKFold(a.inner_splits, shuffle=True, random_state=a.seed)
    fit_i, inner_i = next(icv.split(tr, tr.label, groups=tr.group_id))
    fit, inner = tr.iloc[fit_i], tr.iloc[inner_i]
    if a.smoke:
        fit, inner, va = fit.sample(128, random_state=0), inner.sample(64, random_state=0), va.sample(128, random_state=0)
    print("fit %d (%d pos) | inner-val %d (%d pos) | OOF fold %d: %d (%d pos)"
          % (len(fit), fit.label.sum(), len(inner), inner.label.sum(), a.fold, len(va), va.label.sum()))

    wmap = None
    if a.weights:
        wdf = pd.read_csv(a.weights)
        wmap = wdf.set_index("image_id").weight
        n_up = int((fit.image_id.map(wmap).fillna(1.0) != 1.0).sum())
        print("reweighting: %d of %d fit images carry a non-unit weight" % (n_up, len(fit)))
    if a.loss_balance == "centre":
        assert wmap is None, "centre balancing must not stack with S9 hard-example weights"
        wmap = group_balanced_weights(fit)
        cells = fit.groupby(["centre_id", "label"]).size()
        print("centre-balanced loss over %d cells:\n%s" % (len(cells), cells.to_string()))

    dl = lambda d, t, w=None: DataLoader(Frames(d, a.data_root, a.size, t, a.preproc, w, a.aug if t else "none"),
                                         batch_size=a.batch_size,
                                         shuffle=t, num_workers=a.workers, pin_memory=True,
                                         drop_last=t, persistent_workers=a.workers > 0)
    tl, il, vl = dl(fit, True, wmap), dl(inner, False), dl(va, False)

    model = build_arch(a.arch, init=a.init).to(dev)
    print("arch %s @ %dpx | %.1fM params" % (a.arch, a.size, sum(p.numel() for p in model.parameters()) / 1e6))

    if a.freeze_bn:
        print("froze %d BatchNorm layers (running stats + affine)" % freeze_batchnorm(model))
    frozen = set()
    if a.freeze_until != "none":
        assert a.arch == "resnet", "--freeze-until is resnet-only"
        frozen, n_frozen = freeze_until(model, a.freeze_until)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("froze through %s: %.1fM frozen | %.1fM trainable"
              % (a.freeze_until, n_frozen / 1e6, n_train / 1e6))
    # Centre balancing already equalises the classes; stacking pos_weight on top
    # would double the positive multiplier, which is what broke E6.
    pwv = 1.0 if a.loss_balance in ("centre",) or a.loss == "balanced" \
        else (fit.label == 0).sum() / max((fit.label == 1).sum(), 1)
    pw = torch.tensor([float(pwv)], device=dev)
    print("loss %s | pos_weight %.2f | sample balance %s | aug %s"
          % (a.loss, pw.item(), a.loss_balance, a.aug))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best, best_ep, hist = -1.0, -1, []
    for ep in range(a.epochs):
        model.train()
        if a.freeze_bn:
            freeze_batchnorm(model)          # model.train() re-enables BN updates
        if frozen:
            apply_frozen_eval(model, frozen)   # ditto for the frozen stages' BN
        t0, tot = time.time(), 0.0
        for x, y, sw in tl:
            x, y, sw = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), sw.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x).squeeze(1)
                loss = balanced_bce(z, y) if a.loss == "balanced" else (crit(z, y) * sw).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()

        s = predict(model, il, dev)                      # inner split, never the OOF fold
        y = inner.label.to_numpy()
        auprc, auroc = average_precision_score(y, s), roc_auc_score(y, s)
        hist.append(dict(epoch=ep, loss=tot / len(fit), inner_auprc=auprc, inner_auroc=auroc,
                         lr=sched.get_last_lr()[0], secs=round(time.time() - t0, 1)))
        flag = ""
        if auprc > best:
            best, best_ep, flag = auprc, ep, "  *best"
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(a)},
                       os.path.join(out, "best.pt"))
        print("ep %2d  loss %.4f  inner_auprc %.4f  inner_auroc %.4f  %.0fs%s"
              % (ep, tot / len(fit), auprc, auroc, time.time() - t0, flag))
        if ep - best_ep >= a.patience:
            print("early stop (no inner AUPRC gain in %d epochs)" % a.patience)
            break

    ckpt = os.path.join(out, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    s = predict(model, vl, dev)                          # OOF fold scored once, after selection
    h = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    y = va.label.to_numpy()
    pd.DataFrame(dict(image_id=va.image_id.to_numpy(), label=y,
                      fold=a.fold, centre_id=va.centre_id.to_numpy(),
                      score=1 / (1 + np.exp(-s)), checkpoint_hash=h[:16],
                      preprocessing_version=a.preproc)).to_csv(
        os.path.join(out, "oof_fold%d.csv" % a.fold), index=False)
    pd.DataFrame(hist).to_csv(os.path.join(out, "log.csv"), index=False)

    w = mode_a_weighted(y, s)
    print("\nselected epoch %d on inner AUPRC %.4f" % (best_ep, best))
    print("OOF fold %d: AUPRC %.4f AUROC %.4f" % (a.fold, w["auprc"], w["auroc"]))
    print("weighted PPV@90R %.4f (chance 0.0099)" % w["ppv"])
    print("checkpoint sha256 %s -> %s" % (h[:16], out))


if __name__ == "__main__":
    main()
