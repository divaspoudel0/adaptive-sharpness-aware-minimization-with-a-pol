#!/usr/bin/env python3
"""Run Claims 4-6 (arXiv 2606.01827) on free molab/marimo tier.

Self-contained standalone script (no private repo needed). Protocol matches the
paper's Sec. 4.2-4.3: ResNet-32, CIFAR-100, 100 epochs, batch 128, cross-entropy,
SGD momentum 0, weight decay 5e-4, random-crop + hflip + per-channel norm, best
test acc over epochs, mean +/- std over seeds 0,1,2.

Grids:
  claim-4  USAM (lambd=0)  rho=0.2 : polyak / constant(0.1) / cosine(0.1->1e-3)
  claim-5  SAM  (lambd=1)  rho=0.2 : polyak / constant(0.1) / cosine(0.1->1e-3)
  claim-6  N-SAM(lambd=1) rho-sweep: polyak at rho 0.1,0.2,0.3,0.4 + cosine at 0.4

Incremental + resumable: every completed (claim,label,seed) is written to the
results JSON; on re-run completed items are skipped. Also safe to run the three
claims in separate invocations.

Usage:
  python claims_4_6.py --claims 4,5,6            # run all
  python claims_4_6.py --claims 6 --epochs 100   # only claim 6
  python claims_4_6.py --epochs 2 --seeds 0      # smoke test
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

def _ensure_deps():
    try:
        import torch  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
    for mod in ("pyarrow", "huggingface_hub", "PIL", "scipy"):
        try:
            __import__(mod)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", mod])
    import torch, torchvision, pyarrow, huggingface_hub  # noqa: F401,E402


_ensure_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------
# CIFAR data (HF parquet mirror, falls back to torchvision)
# ----------------------------------------------------------------------------
def _decode(img_struct):
    from PIL import Image
    import io

    b = img_struct["bytes"]
    if isinstance(b, memoryview):
        b = b.tobytes()
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))


def load_cifar(dataset="cifar100", cache_dir=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if cache_dir is None:
        cache_dir = os.environ.get("CIFAR_CACHE", os.path.expanduser("~/.cache/cifar_hf"))
    hub = {"cifar10": "uoft-cs/cifar10", "cifar100": "uoft-cs/cifar100"}
    files = {
        "cifar10": ("plain_text/train-00000-of-00001.parquet", "plain_text/test-00000-of-00001.parquet"),
        "cifar100": ("cifar100/train-00000-of-00001.parquet", "cifar100/test-00000-of-00001.parquet"),
    }
    ds = hub[dataset]
    tr_file, te_file = files[dataset]
    out = {}
    for split, fname in (("tr", tr_file), ("te", te_file)):
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            os.makedirs(cache_dir, exist_ok=True)
            local = hf_hub_download(repo_id=ds, filename=fname, repo_type="dataset",
                                    cache_dir=cache_dir, local_dir=cache_dir)
        t = pq.read_table(local)
        imgs = np.stack([_decode(x) for x in t.column("img").to_pylist()])
        labels = t.column("fine_label").to_numpy() if dataset == "cifar100" else t.column("label").to_numpy()
        out[split] = (imgs, labels)
    return out["tr"][0], out["tr"][1], out["te"][0], out["te"][1]


CIFAR_MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class RandCropHFlip:
    def __init__(self, size=32, pad=4):
        self.size, self.pad = size, pad

    def __call__(self, x):
        n = x.shape[0]
        xp = np.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
        H = xp.shape[1]
        out = np.empty_like(x)
        for i in range(n):
            r = np.random.randint(0, 2 * self.pad + 1)
            c = np.random.randint(0, 2 * self.pad + 1)
            out[i] = xp[i, r:r + self.size, c:c + self.size]
        flip = np.random.rand(n) < 0.5
        out[flip] = out[flip, :, ::-1, :]
        return out


class NormAug(nn.Module):
    def __init__(self, mean, std, size=32, pad=4):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.size, self.pad = size, pad

    def __call__(self, batch):
        x, y = batch
        x = x.numpy().astype(np.float32) / 255.0
        x = RandCropHFlip(self.size, self.pad)(x)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)
        return (x - self.mean) / self.std, y


def _worker_seed(_):
    np.random.seed()


# ----------------------------------------------------------------------------
# ResNet-20/32 (He et al. 2016)
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


# ----------------------------------------------------------------------------
# Optimizers: SAM/USAM with Stochastic Polyak Scheduler (official port)
# ----------------------------------------------------------------------------
class _SAMBase(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0):
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        defaults = dict(lr=0.1, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rho, self.lambd = rho, lambd
        self.grad_norm = torch.tensor(0.0)

    @torch.no_grad()
    def _sgd_update(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad
            if wd != 0.0:
                d_p = d_p.add(p, alpha=wd)
            p.add_(d_p, alpha=-lr)

    def step(self, closure):
        eps = 1e-12
        closure = torch.enable_grad()(closure)
        self.zero_grad(set_to_none=True)
        loss_x = closure()

        gns = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gx = p.grad.detach().to(torch.float32)
                    gns = gx.pow(2).sum() if gns is None else gns + gx.pow(2).sum()
        self.grad_norm = gns.sqrt().clamp(min=eps)

        scale = self.rho * ((1 - self.lambd) + self.lambd / self.grad_norm)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    rest = (p.grad * scale).to(dtype=p.dtype)
                    p.add_(rest)
                    self.state[p]["e_t_minus_x"] = rest

        self.zero_grad(set_to_none=True)
        loss_e = closure()
        self._set_gamma(loss_e)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    rest = self.state.get(p, {}).get("e_t_minus_x", None)
                    if rest is not None:
                        p.sub_(rest)
                        self.state[p].pop("e_t_minus_x", None)

        with torch.no_grad():
            for group in self.param_groups:
                self._sgd_update(group)
        return loss_e

    def _set_gamma(self, loss_e):
        raise NotImplementedError


class SAM_SPS(_SAMBase):
    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0, f_star=0.0, gamma_b=1.0):
        super().__init__(params, weight_decay=weight_decay, rho=rho, lambd=lambd)
        self.f_star, self.gamma_b = f_star, gamma_b
        for g in self.param_groups:
            g["lr"] = gamma_b

    def _set_gamma(self, loss_e):
        eps, dot, gne = 1e-12, None, None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                rest = self.state.get(p, {}).get("e_t_minus_x", None)
                if rest is None:
                    continue
                rest = rest.detach().to(torch.float32)
                ge = p.grad.detach().to(torch.float32)
                dot = torch.sum(ge * rest) if dot is None else dot + torch.sum(ge * rest)
                gne = ge.pow(2).sum() if gne is None else gne + ge.pow(2).sum()
        num = loss_e.detach().to(torch.float32) - self.f_star - dot
        denom = gne.clamp(min=eps)
        gamma = max(0.0, min((num / denom).item(), self.gamma_b))
        for group in self.param_groups:
            group["lr"] = gamma


class SAM_Fixed(_SAMBase):
    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0, lr=0.1):
        super().__init__(params, weight_decay=weight_decay, rho=rho, lambd=lambd)
        for g in self.param_groups:
            g["lr"] = lr

    def _set_gamma(self, loss_e):
        pass


def cosine_gamma(epoch, T, gamma_min, gamma_max):
    return gamma_min + (gamma_max - gamma_min) * (1 + torch.cos(torch.tensor(float(epoch) * 3.141592653589793 / T))) / 2


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def make_loaders(dataset, batch_size, num_workers=2, cache_dir=None):
    mean, std = CIFAR_MEAN_STD[dataset]
    norm = NormAug(mean, std)
    x_tr, y_tr, x_te, y_te = load_cifar(dataset, cache_dir=cache_dir)
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, worker_init_fn=_worker_seed)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader, norm, (10 if dataset == "cifar10" else 100)


def evaluate(model, loader, norm, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = norm((x, y))
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_seed(cfg, seed, device, cache_dir=None, num_workers=2, log_freq=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, norm, n_classes = make_loaders(cfg["dataset"], cfg["batch_size"],
                                                              num_workers=num_workers, cache_dir=cache_dir)
    model = resnet32(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    lambd = 1.0 if cfg.get("sam_normalized", False) else 0.0
    wd, rho = cfg.get("weight_decay", 5e-4), cfg.get("rho", 0.2)
    if cfg["scheduler"] == "polyak":
        optimizer = SAM_SPS(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd,
                            f_star=cfg.get("f_star", 0.0), gamma_b=cfg.get("gamma_b", 1.0))
    else:
        optimizer = SAM_Fixed(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd,
                              lr=cfg.get("lr", 0.1))
    epochs, T = cfg.get("epochs", 100), cfg.get("epochs", 100)
    best, best_epoch, losses, accs = 0.0, -1, [], []
    t0 = time.time()
    for epoch in range(epochs):
        if cfg["scheduler"] == "cosine":
            lr = float(cosine_gamma(epoch, T, cfg.get("lr_min", 1e-3), cfg.get("lr", 0.1)))
            for g in optimizer.param_groups:
                g["lr"] = lr
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = norm((x, y))
            x, y = x.to(device), y.to(device)

            def closure():
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                return loss

            tot += optimizer.step(closure).item()
            nb += 1
        losses.append(tot / nb)
        acc = evaluate(model, test_loader, norm, device)
        accs.append(acc)
        if acc > best:
            best, best_epoch = acc, epoch
        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(f"[train] seed={seed} epoch={epoch+1}/{epochs} train_loss={losses[-1]:.4f} "
                  f"test_acc={acc:.2f} best={best:.2f} (ep {best_epoch+1})", flush=True)
    print(f"[train] seed={seed} DONE best_acc={best:.4f} best_epoch={best_epoch+1} "
          f"final_acc={accs[-1]:.4f} wall_sec={time.time()-t0:.1f}", flush=True)
    return {"best_acc": best, "final_acc": accs[-1], "best_epoch": best_epoch}


# ----------------------------------------------------------------------------
# Grids (Claims 4-6)
# ----------------------------------------------------------------------------
def build_claims(epochs=100, seeds=(0, 1, 2), rho=0.2, batch_size=128, weight_decay=5e-4):
    base = dict(mode="cifar", dataset="cifar100", arch="resnet32", epochs=epochs,
                batch_size=batch_size, weight_decay=weight_decay, rho=rho, seeds=list(seeds))
    return {
        "claim-4": {
            "title": "Claim 4: USAM (lambd=0) Polyak scheduler vs constant/cosine",
            "sam_normalized": False,
            "grid": [
                {"label": "usam-polyak", "scheduler": "polyak"},
                {"label": "usam-constant", "scheduler": "constant", "lr": 0.1},
                {"label": "usam-cosine", "scheduler": "cosine", "lr": 0.1, "lr_min": 0.001},
            ],
        },
        "claim-5": {
            "title": "Claim 5: SAM (lambd=1) Polyak scheduler vs constant/cosine",
            "sam_normalized": True,
            "grid": [
                {"label": "sam-polyak", "scheduler": "polyak"},
                {"label": "sam-constant", "scheduler": "constant", "lr": 0.1},
                {"label": "sam-cosine", "scheduler": "cosine", "lr": 0.1, "lr_min": 0.001},
            ],
        },
        "claim-6": {
            "title": "Claim 6: N-SAM (lambd=1) rho-sweep",
            "sam_normalized": True,
            "grid": [
                {"label": "nsam-polyak-rho0.1", "scheduler": "polyak", "rho": 0.1},
                {"label": "nsam-polyak-rho0.2", "scheduler": "polyak", "rho": 0.2},
                {"label": "nsam-polyak-rho0.3", "scheduler": "polyak", "rho": 0.3},
                {"label": "nsam-polyak-rho0.4", "scheduler": "polyak", "rho": 0.4},
                {"label": "nsam-cosine-rho0.4", "scheduler": "cosine", "lr": 0.1, "lr_min": 0.001, "rho": 0.4},
            ],
        },
    }


def _load_done(results_file):
    if os.path.exists(results_file):
        try:
            return json.load(open(results_file))
        except Exception:
            return {}
    return {}


def _save_done(done, results_file):
    tmp = results_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done, f, indent=2)
    os.replace(tmp, results_file)


def run_claim(claim_id, claims, device, results_file, cache_dir=None, num_workers=2):
    spec = claims[claim_id]
    done = _load_done(results_file)
    cfg = dict(claims.get("_base", {}))
    cfg.update({"sam_normalized": spec["sam_normalized"]})
    cfg["epochs"] = claims.get("_epochs", cfg["epochs"])
    cfg["seeds"] = claims.get("_seeds", cfg["seeds"])
    results = {}
    for item in spec["grid"]:
        item_cfg = dict(cfg)
        item_cfg.update(item)
        label = item_cfg["label"]
        print(f"[grid] {claim_id} label={label} cfg={json.dumps({k: item_cfg[k] for k in ('scheduler','lr','lr_min','rho','sam_normalized') if k in item_cfg})}", flush=True)
        per_seed = {}
        for seed in cfg["seeds"]:
            skey = f"{claim_id}::{label}::seed{seed}"
            if skey in done:
                per_seed[seed] = done[skey]
                print(f"[resume] {label} seed={seed} cached", flush=True)
                continue
            try:
                r = train_seed(item_cfg, seed, device, cache_dir=cache_dir, num_workers=num_workers)
            except Exception as exc:
                r = {"error": repr(exc)}
                print(f"[FAILED] {label} seed={seed}: {exc!r}", flush=True)
                traceback.print_exc()
            done[skey] = r
            _save_done(done, results_file)
            per_seed[seed] = r
        bests = [r["best_acc"] for r in per_seed.values() if "best_acc" in r]
        if bests:
            summary = {
                "label": label,
                "per_seed_best": bests,
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "mean_final": float(np.mean([r["final_acc"] for r in per_seed.values() if "final_acc" in r])),
                "epochs": item_cfg.get("epochs", 100),
                "config": {k: item_cfg[k] for k in ("scheduler", "lr", "lr_min", "rho", "sam_normalized") if k in item_cfg},
            }
        else:
            summary = {"label": label, "error": "no completed seeds"}
        results[label] = summary
        print(f"[RESULT] {claim_id} {label}: {summary}", flush=True)
    done[f"{claim_id}::summary"] = results
    done[f"_complete_{claim_id}"] = True
    _save_done(done, results_file)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="4,5,6", help="comma-separated claim ids")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "claims_4_6_results.json"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--log-freq", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} torch={torch.__version__}", flush=True)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    claims = build_claims(epochs=args.epochs, seeds=seeds)
    claims["_epochs"] = args.epochs
    claims["_seeds"] = seeds
    claims["_base"] = dict(mode="cifar", dataset="cifar100", arch="resnet32", epochs=args.epochs,
                           batch_size=128, weight_decay=5e-4, rho=0.2, seeds=list(seeds))
    ids = ["claim-" + c.strip() for c in args.claims.split(",") if c.strip()]

    results = {}
    for claim_id in ids:
        print(f"\n===== {claim_id}: {claims[claim_id]['title']} =====", flush=True)
        t0 = time.time()
        try:
            results[claim_id] = run_claim(claim_id, claims, device, args.out,
                                          cache_dir=args.cache_dir, num_workers=args.num_workers)
        except Exception as exc:
            results[claim_id] = {"error": repr(exc)}
            print(f"[FAILED] {claim_id}: {exc!r}", flush=True)
            traceback.print_exc()
        print(f"[CLAIM-DONE] {claim_id} wall_sec={time.time()-t0:.1f}", flush=True)
    print("\n[ALL-DONE]", flush=True)
    print(f"results file: {args.out}", flush=True)


if __name__ == "__main__":
    main()
