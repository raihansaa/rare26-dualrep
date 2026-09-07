"""E5 -- cross-centre evaluation (plan S13 E5, protocol S5B).

Train on one centre, test on the other. A held-out slice of the TRAINING
centre drives early stopping, so the test centre is never used for model
selection -- and that same slice is the within-centre reference, without which
a cross-centre drop cannot be told apart from "the other centre is harder".

Immune to within-centre patient leakage: no patient can appear on both sides.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
from metric import mode_a_weighted
from train_resnet import (ARCH, Frames, predict, group_balanced_weights, freeze_batchnorm,
                          build_arch, freeze_until, apply_frozen_eval)
from train_dinov2 import (tail_ranking_loss, class_balanced_bce, mean_specificity,
                          PosEnrichedBatches, PPVAtRecallLoss)


def boot(y, s, n=1000, seed=0):
    """Stratified bootstrap -- resample each class to its own size."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {"auroc": [], "auprc": [], "ppv90": []}
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        yy, ss = y[i], s[i]
        out["auroc"].append(roc_auc_score(yy, ss))
        out["auprc"].append(average_precision_score(yy, ss))
        out["ppv90"].append(mode_a_weighted(yy, ss)["ppv"])
    return {k: (float(np.nanmedian(v)), float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v1.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--arch", default="resnet",
                   choices=["resnet", "resnet512", "dinov2", "maxvit", "maxvit384"])
    p.add_argument("--init", default=None,
                   help="path to a pretrained backbone; arch default: imagenet / dinov2")
    p.add_argument("--lora-blocks", type=int, default=6,
                   help="dinov2 only: how many trailing blocks carry LoRA adapters (of 12)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-all-linear", action="store_true",
                   help="also adapt mlp.fc1/fc2, not attention alone (the RARE25 winner's config)")
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="dropout on the adapter input; 0.0 reproduces every earlier run")
    p.add_argument("--mil", default=None, choices=["topk", "lse"],
                   help="pool a per-patch head alongside the CLS head; None = CLS only")
    p.add_argument("--mil-k", type=int, default=8, help="patches pooled when --mil topk")
    p.add_argument("--ppv-lambda", type=float, default=0.0,
                   help="weight on PPVAtRecallLoss (FIFO bank of positive logits, hinge on "
                        "negatives above the tracked quantile); 0 disables it")
    p.add_argument("--ppv-q", type=float, default=0.10,
                   help="quantile of positive logits tracked as the threshold")
    p.add_argument("--ppv-margin", type=float, default=0.5, help="hinge margin")
    p.add_argument("--tail-pos-skip", type=float, default=0.0,
                   help="fraction of the very lowest positives the tail term skips before mining")
    p.add_argument("--tail-warmup", type=int, default=0,
                   help="epochs of plain BCE before the tail term switches on")
    p.add_argument("--pos-per-batch", type=int, default=0,
                   help="positives per batch; 0 keeps natural prevalence. Also switches BCE to "
                        "class-balanced, since pos_weight would double-count the enrichment")
    p.add_argument("--select", default="auprc", choices=["auprc", "spec"],
                   help="checkpoint selection on the inner holdout: AUPRC, or mean specificity "
                        "over recall 0.80-0.95")
    p.add_argument("--ckpt-avg", type=int, default=0,
                   help="average trainable weights over the best epoch +/- this many epochs")
    p.add_argument("--tail-lambda", type=float, default=0.0,
                   help="weight on the tail ranking term; 0 disables it (plain weighted BCE)")
    p.add_argument("--train-centre", default="center_1")
    p.add_argument("--test-centre", default=None)
    p.add_argument("--holdout", default=None,
                   help="leave-one-centre-out: test on this centre, train on all others")
    p.add_argument("--aug", default="none", choices=["none", "acq", "jigsaw", "heavy"])
    p.add_argument("--loss-balance", default="global", choices=["global", "centre"])
    p.add_argument("--freeze-bn", action="store_true")
    p.add_argument("--freeze-until", default="none",
                   choices=["none", "layer1", "layer2", "layer3", "layer4",
                            "stages.0", "stages.1", "stages.2"],
                   help="freeze the stem and stages up to and including this one, leaving "
                        "the rest trainable. resnet: layer3 = train layer4+fc, the CNN "
                        "analogue of LoRA-6. maxvit: stages.2 = train stages.3+norm+head")
    p.add_argument("--tag", default="")
    p.add_argument("--holdout-splits", type=int, default=6)   # 1/6 held out
    p.add_argument("--preproc", default="P0", choices=["P0", "P3", "P4"])
    p.add_argument("--size", type=int, default=None)   # arch default: 384 / 336
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=None)   # arch default: 1e-4 / 5e-4
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="fix the fit/holdout split while --seed varies the model, so "
                        "ensemble members share one calibration slice; default: --seed")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    df = pd.read_csv(a.folds)
    centres = sorted(df.centre_id.unique())
    if a.holdout:
        assert a.holdout in centres, "%s not among %s" % (a.holdout, centres)
        a.test_centre, a.train_centre = a.holdout, "rest"
    elif a.test_centre is None:
        other = [c for c in centres if c != a.train_centre]
        assert len(other) == 1, "ambiguous test centre among %s -- pass --test-centre" % centres
        a.test_centre = other[0]
    if a.size is None:
        # MaxViT window sizes are tied to the input resolution, so the size must come
        # from the arch table rather than a shared default (see train_resnet.ARCH).
        a.size = 336 if a.arch == "dinov2" else ARCH[a.arch][1]
    if a.lr is None:
        a.lr = 5e-4 if a.arch == "dinov2" else 1e-4
    if a.init is None:
        a.init = "dinov2" if a.arch == "dinov2" else "imagenet"
    out = a.out or "runs/%s_%s%s_to_%s%s%s" % (
        "LOCO" if a.holdout else "E5_cross",
        "" if a.arch == "resnet" else a.arch + "_",
        a.train_centre, a.test_centre,
        "" if a.preproc == "P0" else "_" + a.preproc.lower(),
        ("_" + a.tag) if a.tag else "")
    if a.smoke:
        a.epochs, a.n_boot, out = 2, 50, "runs/smoke_cross"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    src = df[df.centre_id != a.test_centre if a.holdout else df.centre_id == a.train_centre].reset_index(drop=True)
    tst = df[df.centre_id == a.test_centre].reset_index(drop=True)
    cv = StratifiedGroupKFold(a.holdout_splits, shuffle=True,
                              random_state=a.seed if a.split_seed is None else a.split_seed)
    fit_i, hold_i = next(cv.split(src, src.label, groups=src.group_id))
    fit, hold = src.iloc[fit_i], src.iloc[hold_i]
    if a.smoke:
        cap = lambda d, n: d.sample(min(n, len(d)), random_state=0)
        fit, hold, tst = cap(fit, 128), cap(hold, 64), cap(tst, 128)

    print("train  %s: fit %d (%d pos) | holdout %d (%d pos)"
          % (a.train_centre, len(fit), fit.label.sum(), len(hold), hold.label.sum()))
    print("test   %s: %d (%d pos)" % (a.test_centre, len(tst), tst.label.sum()))

    wmap = group_balanced_weights(fit) if a.loss_balance == "centre" else None
    if wmap is not None:
        print("centre-balanced cells:\n%s" % fit.groupby(["centre_id", "label"]).size().to_string())

    mk = lambda d, t, w=None: Frames(d, a.data_root, a.size, t, a.preproc, w, a.aug if t else "none")
    dl = lambda d, t, w=None: DataLoader(mk(d, t, w), batch_size=a.batch_size,
                                         shuffle=t, num_workers=a.workers, pin_memory=True,
                                         drop_last=t, persistent_workers=a.workers > 0)
    if a.pos_per_batch:
        bs = PosEnrichedBatches(fit.label.to_numpy(), a.batch_size, a.pos_per_batch, a.seed)
        tl = DataLoader(mk(fit, True, wmap), batch_sampler=bs, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)
        print("positive-enriched batches: %d pos + %d neg, %d batches/epoch "
              "(each positive seen ~%.1fx per epoch, each negative ~%.2fx)"
              % (a.pos_per_batch, a.batch_size - a.pos_per_batch, len(bs),
                 len(bs) * a.pos_per_batch / max((fit.label == 1).sum(), 1),
                 len(bs) * (a.batch_size - a.pos_per_batch) / max((fit.label == 0).sum(), 1)))
    else:
        tl = dl(fit, True, wmap)
    hl, xl = dl(hold, False), dl(tst, False)

    if a.arch != "dinov2":
        model = build_arch(a.arch, 1, a.init)
    else:
        from train_dinov2 import build_model
        model = build_model(a.size, a.lora_rank, a.lora_alpha, a.lora_blocks, a.init,
                            a.mil, a.mil_k, a.lora_all_linear, a.lora_dropout)
    model.to(dev)
    if a.freeze_bn and a.arch == "resnet":
        print("froze %d BatchNorm layers" % freeze_batchnorm(model))
    frozen = set()
    if a.freeze_until != "none":
        assert a.arch != "dinov2", "--freeze-until is for CNN-style archs (dinov2 uses --lora-blocks)"
        frozen, n_frozen = freeze_until(model, a.freeze_until)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("froze through %s: %.1fM frozen | %.1fM trainable"
              % (a.freeze_until, n_frozen / 1e6, n_train / 1e6))
    pwv = 1.0 if a.loss_balance == "centre" else (fit.label == 0).sum() / max((fit.label == 1).sum(), 1)
    pw = torch.tensor([float(pwv)], device=dev)
    if a.pos_per_batch:
        assert a.loss_balance == "global", \
            "class-balanced BCE already balances the classes; --loss-balance centre would stack"
        print("loss: class-balanced BCE -- pos_weight %.2f bypassed, the sampler carries the "
              "balance (aug: %s)" % (pw.item(), a.aug))
    else:
        print("pos_weight %.2f (loss balance: %s | aug: %s)" % (pw.item(), a.loss_balance, a.aug))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    ppv_crit = PPVAtRecallLoss(a.ppv_q, margin=a.ppv_margin).to(dev) if a.ppv_lambda else None
    if ppv_crit is not None:
        print("PPV@%d-recall surrogate: lambda %.3f, margin %.2f, FIFO bank of positive logits"
              % (round(100 * (1 - a.ppv_q)), a.ppv_lambda, a.ppv_margin))
    trainable = {n for n, q in model.named_parameters() if q.requires_grad}
    snaps = {}
    n_seen = len(tl) * a.batch_size if a.pos_per_batch else len(fit)
    best, best_ep, hist = -1.0, -1, []
    for ep in range(a.epochs):
        model.train()
        if a.freeze_bn and a.arch == "resnet":
            freeze_batchnorm(model)          # model.train() re-enables BN updates
        if frozen:
            apply_frozen_eval(model, frozen)   # ditto for the frozen stages' BN
        t0, tot = time.time(), 0.0
        for x, y, sw in tl:
            x, y, sw = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), sw.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x).squeeze(1)
                loss = class_balanced_bce(z, y, sw) if a.pos_per_batch else (crit(z, y) * sw).mean()
                if a.tail_lambda and ep >= a.tail_warmup:
                    loss = loss + a.tail_lambda * tail_ranking_loss(z, y, a.tail_pos_skip)
                if ppv_crit is not None and ep >= a.tail_warmup:
                    loss = loss + a.ppv_lambda * ppv_crit(z, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        sh = predict(model, hl, dev)
        yh = hold.label.to_numpy()
        ap = average_precision_score(yh, sh)
        spec = mean_specificity(yh, sh)
        sel = spec if a.select == "spec" else ap
        hist.append(dict(epoch=ep, loss=tot / n_seen, holdout_auprc=ap, holdout_spec8095=spec))
        if a.ckpt_avg:
            snaps[ep] = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items() if k in trainable}
        flag = ""
        if sel > best or best_ep < 0:        # always leave a checkpoint, even if sel is nan
            best, best_ep, flag = sel, ep, "  *best"
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(a)},
                       os.path.join(out, "best.pt"))
        print("ep %2d  loss %.4f  holdout_auprc %.4f  spec8095 %.4f  %.0fs%s"
              % (ep, tot / n_seen, ap, spec, time.time() - t0, flag))
        if ep - best_ep >= a.patience:
            print("early stop")
            break

    if a.ckpt_avg:
        eps = [e for e in range(best_ep - a.ckpt_avg, best_ep + a.ckpt_avg + 1) if e in snaps]
        sd = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
        for k in snaps[best_ep]:
            sd["model"][k] = torch.stack([snaps[e][k].float() for e in eps]).mean(0) \
                                  .to(sd["model"][k].dtype)
        sd["averaged_epochs"] = eps
        torch.save(sd, os.path.join(out, "best.pt"))
        print("averaged trainable weights over epochs %s" % eps)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"))["model"])
    h = hashlib.sha256(open(os.path.join(out, "best.pt"), "rb").read()).hexdigest()[:16]

    res = {}
    for name, d, loader in (("within_%s" % a.train_centre, hold, hl),
                            ("cross_%s" % a.test_centre, tst, xl)):
        # keep the raw logit too: sigmoid saturates at 1.0 in float32, and Platt
        # calibration downstream cannot recover a logit from a saturated score
        z = predict(model, loader, dev)
        s = 1 / (1 + np.exp(-z))
        y = d.label.to_numpy()
        pd.DataFrame(dict(image_id=d.image_id.to_numpy(), label=y, fold=-1,
                          centre_id=d.centre_id.to_numpy(), score=s, logit=z,
                          checkpoint_hash=h, preprocessing_version="P0")).to_csv(
            os.path.join(out, "pred_%s.csv" % name), index=False)
        res[name] = dict(n=len(d), pos=int(y.sum()), **boot(y, s, a.n_boot))

    pd.DataFrame(hist).to_csv(os.path.join(out, "log.csv"), index=False)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)

    print("\n%-22s %5s %4s  %-22s %-22s %-22s" % ("set", "n", "pos", "AUROC", "AUPRC", "PPV@90R"))
    for k, v in res.items():
        f = lambda m: "%.4f [%.4f,%.4f]" % v[m]
        print("%-22s %5d %4d  %-22s %-22s %-22s" % (k, v["n"], v["pos"], f("auroc"), f("auprc"), f("ppv90")))
    print("\ncheckpoint %s -> %s" % (h, out))


if __name__ == "__main__":
    main()
