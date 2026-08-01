"""DNN training for Claims 4-6: ResNet-20/32 on CIFAR-10/100 with SAM/USAM.

Protocol follows the paper (arXiv 2606.01827, Sec. 4.2-4.3):
100 epochs, batch 128, cross-entropy, SGD with momentum 0, weight decay 5e-4,
random-crop + horizontal-flip augmentation + normalization, best test accuracy
over epochs, mean +/- std over 3 seeds. Schedulers: Stochastic Polyak Scheduler
(l*=0, gamma_b=1.0), tuned constant LR, and cosine annealing.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from repro.data import load_cifar
from repro.models import resnet20, resnet32
from repro.optimizers import SAM_Fixed, SAM_SPS, cosine_gamma

CIFAR_MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class RandCropHFlip:
    """Random crop with padding + random horizontal flip on (N,H,W,C) uint8 array (batched)."""

    def __init__(self, size=32, pad=4):
        self.size = size
        self.pad = pad

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


def make_loaders(dataset, batch_size, num_workers=2, data_root="/data", cache_dir=None):
    mean, std = CIFAR_MEAN_STD[dataset]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    x_tr, y_tr, x_te, y_te = load_cifar(dataset, cache_dir=cache_dir)
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=lambda _: np.random.seed())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    aug = RandCropHFlip(32, 4)

    def norm(batch):
        x, y = batch
        x = x.float().div_(255.0)
        x = aug(x)
        x = x.permute(0, 3, 1, 2)
        x = (x - mean_t) / std_t
        return x, y

    return train_loader, test_loader, norm, (10 if dataset == "cifar10" else 100)


def build_model(arch, n_classes, device):
    if arch == "resnet20":
        model = resnet20(num_classes=n_classes)
    elif arch == "resnet32":
        model = resnet32(num_classes=n_classes)
    else:
        raise ValueError(arch)
    return model.to(device)


def evaluate(model, loader, norm, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = norm((x, y))
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_seed(cfg, seed, device, data_root):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, norm, n_classes = make_loaders(
        cfg["dataset"], cfg["batch_size"], data_root=data_root,
        cache_dir=cfg.get("cache_dir", None))
    model = build_model(cfg["arch"], n_classes, device)
    criterion = nn.CrossEntropyLoss()

    lambd = 1.0 if cfg.get("sam_normalized", False) else 0.0
    wd = cfg.get("weight_decay", 5e-4)
    rho = cfg.get("rho", 0.2)
    scheduler = cfg["scheduler"]

    if scheduler == "polyak":
        optimizer = SAM_SPS(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd,
                            f_star=cfg.get("f_star", 0.0), gamma_b=cfg.get("gamma_b", 1.0))
    else:
        lr = cfg.get("lr", 0.1)
        optimizer = SAM_Fixed(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd, lr=lr)

    epochs = cfg.get("epochs", 100)
    T = epochs
    best = 0.0
    best_epoch = -1
    train_losses = []
    test_accs = []
    t0 = time.time()
    for epoch in range(epochs):
        if scheduler == "cosine":
            lr = float(cosine_gamma(epoch, T, cfg.get("lr_min", 1e-3), cfg.get("lr", 0.1)))
            for g in optimizer.param_groups:
                g["lr"] = lr
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = norm((x, y))
            x, y = x.to(device), y.to(device)

            def closure():
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                return loss

            loss_e = optimizer.step(closure)
            tot_loss += loss_e.item()
            n_batches += 1
        train_losses.append(tot_loss / n_batches)
        acc = evaluate(model, test_loader, norm, device)
        test_accs.append(acc)
        if acc > best:
            best = acc
            best_epoch = epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[train] seed={seed} epoch={epoch+1}/{epochs} train_loss={train_losses[-1]:.4f} "
                  f"test_acc={acc:.2f} best={best:.2f} (ep {best_epoch+1})", flush=True)
    dt = time.time() - t0
    print(f"[train] seed={seed} DONE best_acc={best:.4f} best_epoch={best_epoch+1} "
          f"final_acc={test_accs[-1]:.4f} wall_sec={dt:.1f}", flush=True)
    return {"best_acc": best, "final_acc": test_accs[-1], "best_epoch": best_epoch,
            "train_losses": train_losses, "test_accs": test_accs}


def main(cfg, device=None, data_root="/data"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[cfg] " + json.dumps(cfg))
    seeds = cfg.get("seeds", [0])
    results = []
    for seed in seeds:
        r = train_seed(cfg, seed, device, data_root)
        results.append(r)
    bests = [r["best_acc"] for r in results]
    mean = float(np.mean(bests))
    std = float(np.std(bests))
    summary = {
        "per_seed_best": bests,
        "mean_best": mean,
        "std_best": std,
        "mean_final": float(np.mean([r["final_acc"] for r in results])),
        "epochs": cfg.get("epochs", 100),
    }
    print("[RESULT] " + json.dumps(summary), flush=True)
    # save results for the record
    outdir = os.environ.get("REPRO_OUT", "/data")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{cfg['dataset']}_{cfg['arch']}_{cfg['scheduler']}_rho{cfg.get('rho',0)}"
    with open(os.path.join(outdir, f"result_{tag}.json"), "w") as f:
        json.dump({"cfg": cfg, "results": results, "summary": summary}, f, indent=2)
    print(f"[save] wrote {os.path.join(outdir, f'result_{tag}.json')}", flush=True)
    return summary
