"""Quick smoke/timing run: trains a few epochs on the DNN config and reports throughput."""
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from repro.cifar import CIFAR_MEAN_STD, build_model, make_loaders
from repro.optimizers import SAM_Fixed, SAM_SPS, cosine_gamma


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}", flush=True)
    torch.manual_seed(cfg.get("seed", 0))
    train_loader, test_loader, norm, n_classes = make_loaders(
        cfg["dataset"], cfg["batch_size"], data_root=cfg.get("data_root", "/data"),
        cache_dir=cfg.get("cache_dir", None))
    model = build_model(cfg["arch"], n_classes, device)
    criterion = nn.CrossEntropyLoss()
    lambd = 1.0 if cfg.get("sam_normalized", False) else 0.0
    wd = cfg.get("weight_decay", 5e-4)
    rho = cfg.get("rho", 0.2)
    if cfg["scheduler"] == "polyak":
        optimizer = SAM_SPS(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd,
                            gamma_b=cfg.get("gamma_b", 1.0))
    else:
        optimizer = SAM_Fixed(model.parameters(), weight_decay=wd, rho=rho, lambd=lambd, lr=cfg.get("lr", 0.1))
    n_epochs = cfg.get("n_epochs", 2)
    t0 = time.time()
    n_steps = 0
    for epoch in range(n_epochs):
        if cfg["scheduler"] == "cosine":
            lr = float(cosine_gamma(epoch, cfg.get("epochs", 100), cfg.get("lr_min", 1e-3), cfg.get("lr", 0.1)))
            for g in optimizer.param_groups:
                g["lr"] = lr
        model.train()
        for x, y in train_loader:
            x, y = norm((x, y))
            x, y = x.to(device), y.to(device)

            def closure():
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                return loss

            optimizer.step(closure)
            n_steps += 1
        acc = 0.0  # skip eval in smoke mode
    dt = time.time() - t0
    print(f"[smoke] {n_epochs} epochs, {n_steps} steps in {dt:.1f}s -> {n_steps/dt:.2f} steps/s "
          f"({dt/n_epochs:.1f} s/epoch)", flush=True)
    print("[RESULT] " + json.dumps({"steps": n_steps, "sec": dt, "steps_per_s": n_steps / dt}))
