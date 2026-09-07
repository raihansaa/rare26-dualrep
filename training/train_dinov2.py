"""E4 -- DINOv2-Base + LoRA, one fold (plan S13 E4, config S1).

Backbone frozen; LoRA adapters (rank 16) on attn.qkv and attn.proj of the last
six blocks, plus the linear head. Same folds_v1.csv and same P0 preprocessing
as the ResNet arm, as S4 requires once preprocessing is selected.

S1 fixes rank and block count but not which projections carry the adapters, nor
the learning rate; qkv+proj and 5e-4 are chosen here and should be frozen with
the rest of the config on Aug 20.
"""
import argparse, hashlib, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
from metric import mode_a_weighted
from train_resnet import Frames, predict, group_balanced_weights


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update, zero-initialised on B."""

    def __init__(self, base, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)   # B stays zero -> starts as identity
        self.scale = alpha / r
        # Dropout on the adapter input only, never on the frozen path, so dropout=0.0
        # is bit-identical to every checkpoint written before this argument existed.
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


TAIL_POS_FRAC, TAIL_NEG_FRAC = 0.25, 0.10


def tail_ranking_loss(z, y, pos_skip=0.0):
    """Soft pairwise ranking: worst positives against the negatives that outrank them.

    PPV@90R is decided entirely by the operating point that captures 90% of
    positives, so the only images that matter are the lowest-scoring positives
    and whichever negatives sit above them. BCE spends most of its gradient on
    examples that are already comfortably right; this term spends all of it on
    that boundary.

    Selection is within-batch. At batch 32 and ~6.5% prevalence a batch holds
    about two positives, so in practice this is hard-pair mining per step rather
    than a true bottom-quartile over the epoch -- use --pos-per-batch to enrich
    the batch and give this term a quantile to work on.

    pos_skip drops that fraction of the very lowest positives before mining. The
    metric permits missing 10% of positives outright, so the images that set the
    threshold sit *around* the 10th percentile, not below it; the ones below are
    disproportionately label or visibility problems and spending gradient on them
    is how E6 was destroyed. 0.0 mines the worst, as the V4_tail runs did.
    """
    z = z.float()                       # softplus on fp16 logit gaps can overflow
    pos, neg = z[y > 0.5], z[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return z.sum() * 0.0            # keep the graph, contribute nothing
    kp = max(1, int(round(TAIL_POS_FRAC * pos.numel())))
    kn = max(1, int(round(TAIL_NEG_FRAC * neg.numel())))
    skip = min(int(round(pos_skip * pos.numel())), pos.numel() - kp)
    skip = max(skip, 0)
    hard_pos = torch.topk(pos, skip + kp, largest=False).values[skip:]  # nearly-missed band
    hard_neg = torch.topk(neg, kn, largest=True).values    # negatives ranked above them
    return F.softplus(hard_neg[None, :] - hard_pos[:, None]).mean()


class PPVAtRecallLoss(nn.Module):
    """Hinge on negatives that outrank a running estimate of the 90%-recall threshold.

    PPV@90R is the false-positive rate at the threshold capturing 90% of positives
    -- i.e. at the 10th percentile of the positive logits. The earlier
    `tail_ranking_loss` estimated that threshold *inside each batch*, where 6.5%
    prevalence leaves about two positives; its own docstring conceded that made it
    hard-pair mining rather than a quantile objective, and enriching the batch to
    eight positives still measured null on LOCO.

    Positives are pooled in a FIFO bank and the quantile taken over the bank, so the
    threshold is estimated from hundreds of positives while the gradient stays local
    to the current batch. The threshold is detached: it is a measurement, not a
    parameter, and back-propagating through a quantile of the positives would let the
    model lower the bar instead of raising the positives.

    An EMA of each batch's quantile does NOT work here, and was measured failing
    before this version was written: at ~2-3 positives per batch
    `quantile(pos, 0.10)` is effectively the batch minimum, and the mean of batch
    minima is a biased estimate of the pooled quantile -- 2.355 against a true 1.762
    on synthetic data. The threshold would sit too high and let most negatives off.

    Intended as an addition to BCE, which supplies the pressure on positives; this
    term supplies only the pressure pushing negatives below them.
    """

    def __init__(self, q=0.10, bank=256, margin=0.5):
        super().__init__()
        self.q, self.margin = q, margin
        self.register_buffer("bank", torch.zeros(bank))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        self.register_buffer("head", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _push(self, pos):
        n, cap = pos.numel(), self.bank.numel()
        idx = (self.head + torch.arange(n, device=self.bank.device)) % cap
        self.bank[idx] = pos.detach().float()
        self.head = (self.head + n) % cap
        self.filled = torch.clamp(self.filled + n, max=cap)

    def forward(self, z, y):
        z = z.float()
        pos, neg = z[y > 0.5], z[y <= 0.5]
        if pos.numel():
            self._push(pos)
        if neg.numel() == 0 or int(self.filled) < 16:      # too few to estimate a quantile
            return z.sum() * 0.0                           # keep the graph, contribute nothing
        tau = torch.quantile(self.bank[:int(self.filled)], self.q)
        return F.relu(neg - tau + self.margin).mean()


def class_balanced_bce(z, y, sw):
    """Mean over positives plus mean over negatives, halved.

    Independent of batch composition, so unlike pos_weight it does not
    double-count the enrichment when --pos-per-batch is on.
    """
    l = F.binary_cross_entropy_with_logits(z.float(), y.float(), reduction="none") * sw
    terms = [l[m].mean() for m in (y > 0.5, y <= 0.5) if m.any()]
    return sum(terms) / len(terms)


def mean_specificity(y, s, lo=0.80, hi=0.95):
    """Mean specificity over the recall band [lo, hi].

    The scored metric is one order statistic of the positive scores, so selecting
    a checkpoint on PPV@90R directly is decided by a single image. Averaging
    specificity across the recall band either side of 0.90 uses the ~7
    neighbouring order statistics instead: nearly the same alignment, far less
    variance. Inner-split AUPRC, by contrast, saturates above 0.95 and leaves
    several epochs tied.
    """
    pos = np.sort(s[y == 1])[::-1]                     # descending
    neg = s[y == 0]
    n = len(pos)
    if n == 0 or len(neg) == 0:
        return float("nan")
    ks = [k for k in range(1, n + 1) if lo <= k / n <= hi]
    if not ks:                                        # too few positives to span the band
        ks = [max(1, int(round(0.90 * n)))]
    return float(np.mean([(neg < pos[k - 1]).mean() for k in ks]))


class PosEnrichedBatches(torch.utils.data.Sampler):
    """Fixed-composition batches: n_pos positives and batch_size - n_pos negatives.

    Positives are drawn without replacement within a batch and repeat across
    batches; negatives stream through a permutation. The number of batches per
    epoch matches the shuffled loader's, so the cosine LR schedule is unchanged.
    """

    def __init__(self, labels, batch_size, n_pos, seed=0):
        self.pos = np.flatnonzero(np.asarray(labels) == 1)
        self.neg = np.flatnonzero(np.asarray(labels) == 0)
        assert len(self.pos) and len(self.neg), "need both classes"
        self.n_pos, self.n_neg = n_pos, batch_size - n_pos
        self.n_batches = len(labels) // batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order, at = rng.permutation(self.neg), 0
        for _ in range(self.n_batches):
            if at + self.n_neg > len(order):
                order, at = rng.permutation(self.neg), 0
            neg = order[at:at + self.n_neg]
            at += self.n_neg
            pos = rng.choice(self.pos, self.n_pos, replace=len(self.pos) < self.n_pos)
            yield np.concatenate([pos, neg]).tolist()


def load_dinov2_backbone(m, path):
    """Load a Meta-format DINOv2 checkpoint into a timm reg4 model.

    The release wraps the backbone in an EMA teacher and keeps the DINO head.
    timm's converter does the rest -- register_tokens -> reg_token, and folding
    the class position embedding into cls_token because timm builds DINOv2 with
    no_embed_class (577 -> 576 positions). strict=False would leave the backbone
    randomly initialised in silence, so the match is asserted rather than trusted.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)["teacher"]
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    sd = checkpoint_filter_fn(sd, m)
    r = m.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys, "unexpected keys: %s" % r.unexpected_keys[:5]
    assert all(k.startswith("head.") for k in r.missing_keys), \
        "backbone tensors missing: %s" % [k for k in r.missing_keys if not k.startswith("head.")][:5]
    return len(sd)


class MILHead(nn.Module):
    """Image logit = the ViT's own CLS head plus a pooled per-patch head.

    Whole-image pooling gives one vector for a frame in which early neoplasia may
    occupy a few percent of the pixels -- at 336px on a /14 grid a 2% lesion is
    roughly 12 of 576 tokens, and the CLS vector averages it against everything
    else. This adds a shared linear head over the patch tokens and pools the top-k
    patch logits, so a small number of strongly positive locations can carry the
    image on their own. That is the multiple-instance formulation: a positive image
    needs one positive patch, a negative image needs all patches low.

    Distinct from the five-crop test, which aggregated whole-image *scores* from a
    model that never learned where to look; here the patch head is trained.

    The patch head is zero-initialised, so at step 0 this model is numerically
    identical to the CLS-only baseline and any divergence is learned rather than an
    artefact of re-initialisation.
    """

    def __init__(self, vit, k=8, mode="topk", tau=1.0):
        super().__init__()
        self.vit = vit
        self.patch = nn.Linear(vit.embed_dim, 1)
        nn.init.zeros_(self.patch.weight)
        nn.init.zeros_(self.patch.bias)
        self.k, self.mode, self.tau = k, mode, tau

    def forward(self, x):
        t = self.vit.forward_features(x)                  # [B, N, D], already normed
        cls = self.vit.forward_head(t)                    # [B, 1] -- timm's own head path
        p = self.patch(t[:, self.vit.num_prefix_tokens:]).squeeze(-1)   # [B, P]
        if self.mode == "lse":
            pool = self.tau * torch.logsumexp(p / self.tau, dim=1) - self.tau * np.log(p.shape[1])
        else:
            pool = torch.topk(p, min(self.k, p.shape[1]), dim=1).values.mean(1)
        return cls + pool[:, None]


DINOV3 = {"dinov3-l": "vit_large_patch16_dinov3", "dinov3-b": "vit_base_patch16_dinov3"}


def build_model(size, rank, alpha, n_blocks, init="dinov2", mil=None, mil_k=8,
                all_linear=False, lora_dropout=0.0):
    if init in DINOV3:
        # Stock DINOv3 (LVD-1689M), the RARE25 winner's transformer arm. Same
        # attn.qkv / attn.proj structure as DINOv2, so LoRALinear wraps it
        # unchanged; ViT-L carries 24 blocks against ViT-B's 12, so --lora-blocks
        # is a different fraction of depth here and is not comparable across them.
        m = timm.create_model(DINOV3[init], pretrained=True, num_classes=1, img_size=size)
        print("init %s | timm %s | %d blocks, %d prefix tokens"
              % (init, DINOV3[init], len(m.blocks), m.num_prefix_tokens))
    elif init == "dinov2":
        m = timm.create_model("vit_base_patch14_dinov2", pretrained=True,
                              num_classes=1, img_size=size)
    else:
        m = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False,
                              num_classes=1, img_size=size)
        print("init %s | %d backbone tensors | sha256 %s"
              % (os.path.basename(init), load_dinov2_backbone(m, init),
                 hashlib.sha256(open(init, "rb").read()).hexdigest()[:16]))
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks[-n_blocks:]:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank, alpha, lora_dropout)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank, alpha, lora_dropout)
        if all_linear:
            # The RARE25 winner adapted every linear layer, not attention alone. On a
            # 24-block ViT-L our attention-only config reaches 0.19% of parameters --
            # a config tuned for a 12-block ViT-B, and the confound that makes the
            # "stock DINOv3 is far worse" result unbankable (EXPERIMENT_LOG.md 3).
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank, alpha, lora_dropout)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank, alpha, lora_dropout)
    for p in m.head.parameters():
        p.requires_grad = True
    if mil:
        m = MILHead(m, k=mil_k, mode=mil)
        print("MIL head: %s pooling over patch tokens (k=%d), zero-initialised"
              % (mil, mil_k))
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    print("trainable %.3fM / %.1fM total (%.2f%%)" % (tr / 1e6, tot / 1e6, 100 * tr / tot))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="folds_v1.csv")
    p.add_argument("--data-root", default="RARE25-train-data")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--preproc", default="P0", choices=["P0", "P3", "P4"])
    p.add_argument("--inner-splits", type=int, default=5)  # 1/5 of train folds -> early stopping
    p.add_argument("--weights", default=None, help="CSV with image_id,weight (plan S9); absent -> 1.0")
    p.add_argument("--aug", default="none", choices=["none", "acq", "jigsaw", "heavy"])
    p.add_argument("--loss-balance", default="global", choices=["global", "centre"])
    p.add_argument("--size", type=int, default=336)      # 336/14 = 24 patches
    p.add_argument("--batch-size", type=int, default=32)  # measured 4.14 GB at 336px
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--init", default="dinov2",
                   help="'dinov2' (timm LVD-142M) or path to a Meta-format DINOv2 .pth")
    p.add_argument("--tail-lambda", type=float, default=0.0,
                   help="weight on the tail ranking term; 0 disables it (plain weighted BCE)")
    p.add_argument("--mil", default=None, choices=["topk", "lse"],
                   help="pool a per-patch head alongside the CLS head; None = CLS only")
    p.add_argument("--mil-k", type=int, default=8, help="patches pooled when --mil topk")
    p.add_argument("--ppv-lambda", type=float, default=0.0,
                   help="weight on PPVAtRecallLoss (EMA of the 90%%-recall threshold); "
                        "0 disables it")
    p.add_argument("--ppv-q", type=float, default=0.10,
                   help="quantile of positive logits tracked as the threshold")
    p.add_argument("--ppv-margin", type=float, default=0.5,
                   help="hinge margin above the tracked threshold")
    p.add_argument("--tail-pos-skip", type=float, default=0.0,
                   help="fraction of the very lowest positives the tail term skips before "
                        "mining; 0 mines the worst, as the V4_tail runs did")
    p.add_argument("--tail-warmup", type=int, default=0,
                   help="epochs of plain BCE before the tail term switches on")
    p.add_argument("--pos-per-batch", type=int, default=0,
                   help="positives per batch; 0 keeps natural prevalence (~2 of 32). Switching "
                        "this on also switches BCE to class-balanced, since pos_weight would "
                        "double-count the enrichment")
    p.add_argument("--select", default="auprc", choices=["auprc", "spec"],
                   help="checkpoint selection statistic on the inner split: inner AUPRC, or "
                        "mean specificity over recall 0.80-0.95")
    p.add_argument("--ckpt-avg", type=int, default=0,
                   help="average the trainable weights of the best epoch +/- this many epochs; "
                        "0 saves the single best epoch")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-blocks", type=int, default=6)
    p.add_argument("--lora-all-linear", action="store_true",
                   help="also adapt mlp.fc1/fc2, not attention alone (the RARE25 winner's config)")
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="dropout on the adapter input; 0.0 reproduces every earlier run")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.out = 2, a.out or "runs/smoke_dino"
    out = a.out or "runs/E4_dinov2_lora_fold%d" % a.fold
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    df = pd.read_csv(a.folds)
    tr, va = df[df.fold != a.fold].reset_index(drop=True), df[df.fold == a.fold]
    # Early stopping must not see the OOF fold (plan S6 audit).
    icv = StratifiedGroupKFold(a.inner_splits, shuffle=True, random_state=a.seed)
    fit_i, inner_i = next(icv.split(tr, tr.label, groups=tr.group_id))
    fit, inner = tr.iloc[fit_i], tr.iloc[inner_i]
    if a.smoke:
        # keep the inner split's positives -- a positive-free inner split makes every
        # selection statistic nan and the smoke run stops exercising selection at all
        inner = pd.concat([inner[inner.label == 1],
                           inner[inner.label == 0].sample(32, random_state=0)])
        fit, va = fit.sample(128, random_state=0), va.sample(128, random_state=0)
    print("fit %d (%d pos) | inner-val %d (%d pos) | OOF fold %d: %d (%d pos)"
          % (len(fit), fit.label.sum(), len(inner), inner.label.sum(), a.fold, len(va), va.label.sum()))

    wmap = None
    if a.weights:
        wmap = pd.read_csv(a.weights).set_index("image_id").weight
        n_up = int((fit.image_id.map(wmap).fillna(1.0) != 1.0).sum())
        print("reweighting: %d of %d fit images carry a non-unit weight" % (n_up, len(fit)))
    if a.loss_balance == "centre":
        assert wmap is None, "centre balancing must not stack with S9 hard-example weights"
        wmap = group_balanced_weights(fit)
        print("centre-balanced loss over %d cells:\n%s"
              % (fit.groupby(['centre_id', 'label']).ngroups,
                 fit.groupby(['centre_id', 'label']).size().to_string()))

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
    il, vl = dl(inner, False), dl(va, False)

    model = build_model(a.size, a.lora_rank, a.lora_alpha, a.lora_blocks, a.init,
                        a.mil, a.mil_k, a.lora_all_linear, a.lora_dropout).to(dev)
    pwv = 1.0 if a.loss_balance == "centre" else (fit.label == 0).sum() / max((fit.label == 1).sum(), 1)
    pw = torch.tensor([float(pwv)], device=dev)
    if a.pos_per_batch:
        assert a.loss_balance == "global", \
            "class-balanced BCE already balances the classes; --loss-balance centre would stack"
        print("loss: class-balanced BCE -- pos_weight %.2f bypassed, the sampler carries the balance"
              % pw.item())
    else:
        print("pos_weight %.2f (loss balance: %s)" % (pw.item(), a.loss_balance))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    ppv_crit = PPVAtRecallLoss(a.ppv_q, margin=a.ppv_margin).to(dev) if a.ppv_lambda else None
    if ppv_crit is not None:
        print("PPV@%d-recall surrogate: lambda %.3f, margin %.2f, EMA across batches"
              % (round(100 * (1 - a.ppv_q)), a.ppv_lambda, a.ppv_margin))
    trainable = {n for n, q in model.named_parameters() if q.requires_grad}
    snaps = {}
    best, best_ep, hist = -1.0, -1, []
    n_seen = len(fit) if not a.pos_per_batch else len(tl) * a.batch_size
    for ep in range(a.epochs):
        model.train()
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

        s = predict(model, il, dev)                      # inner split, never the OOF fold
        y = inner.label.to_numpy()
        auprc, auroc = average_precision_score(y, s), roc_auc_score(y, s)
        spec = mean_specificity(y, s)
        sel = spec if a.select == "spec" else auprc
        hist.append(dict(epoch=ep, loss=tot / n_seen, inner_auprc=auprc, inner_auroc=auroc,
                         inner_spec8095=spec, lr=sched.get_last_lr()[0],
                         secs=round(time.time() - t0, 1)))
        if a.ckpt_avg:
            snaps[ep] = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items() if k in trainable}
        flag = ""
        if sel > best or best_ep < 0:        # always leave a checkpoint, even if sel is nan
            best, best_ep, flag = sel, ep, "  *best"
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(a)},
                       os.path.join(out, "best.pt"))
        print("ep %2d  loss %.4f  inner_auprc %.4f  spec8095 %.4f  %.0fs%s"
              % (ep, tot / n_seen, auprc, spec, time.time() - t0, flag))
        if ep - best_ep >= a.patience:
            print("early stop (no inner %s gain in %d epochs)" % (a.select, a.patience))
            break

    ckpt = os.path.join(out, "best.pt")
    if a.ckpt_avg:
        eps = [e for e in range(best_ep - a.ckpt_avg, best_ep + a.ckpt_avg + 1) if e in snaps]
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in snaps[best_ep]:
            sd["model"][k] = torch.stack([snaps[e][k].float() for e in eps]).mean(0) \
                                  .to(sd["model"][k].dtype)
        sd["averaged_epochs"] = eps
        torch.save(sd, ckpt)
        print("averaged trainable weights over epochs %s" % eps)
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
    print("\nselected epoch %d on inner %s %.4f" % (best_ep, a.select, best))
    print("OOF fold %d: AUPRC %.4f AUROC %.4f" % (a.fold, w["auprc"], w["auroc"]))
    print("weighted PPV@90R %.4f (chance 0.0099)" % w["ppv"])
    print("checkpoint sha256 %s -> %s" % (h[:16], out))


if __name__ == "__main__":
    main()
