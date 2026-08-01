"""SAM/USAM optimizers for the reproduction.

`SAM_SPS` is a self-contained port of the official optimizer from
https://github.com/dimitris-oik/sam_sps (MIT License, commit 82d83c0), adapted to
share a base class with the fixed-LR SAM/USAM baselines. `lambd=0` -> USAM
(un-normalized perturbation), `lambd=1` -> SAM (normalized perturbation).
"""
import torch
from torch import nn


class _SAMBase(torch.optim.Optimizer):
    """Two-pass (perturbed-grad) SAM/USAM update.

    e^t = x^t + rho * (1 - lambd + lambd/||grad f(x^t)||) * grad f(x^t)
    x^{t+1} = x^t - gamma_t * grad f(e^t)   (with L2 weight decay in the SGD update)
    """

    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0):
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        defaults = dict(lr=0.1, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rho = rho
        self.lambd = lambd
        self.grad_norm = torch.tensor(0.0)

    @torch.no_grad()
    def _sgd_update(self, group):
        lr = group["lr"]
        wd = group["weight_decay"]
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

        grad_norm_sq_x = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gx = p.grad.detach().to(torch.float32)
                    grad_norm_sq_x = gx.pow(2).sum() if grad_norm_sq_x is None else grad_norm_sq_x + gx.pow(2).sum()
        self.grad_norm = grad_norm_sq_x.sqrt().clamp(min=eps)

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
    """USAM (lambd=0) / SAM (lambd=1) with the Stochastic Polyak Scheduler.

    gamma_t = min{ [ f(e^t) - l* - <grad f(e^t), e^t - x^t> ]_+ / ||grad f(e^t)||^2 , gamma_b }
    """

    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0, f_star=0.0, gamma_b=1.0):
        super().__init__(params, weight_decay=weight_decay, rho=rho, lambd=lambd)
        self.f_star = f_star
        self.gamma_b = gamma_b
        for g in self.param_groups:
            g["lr"] = gamma_b

    def _set_gamma(self, loss_e):
        eps = 1e-12
        dot = None
        grad_norm_sq_e = None
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
                grad_norm_sq_e = ge.pow(2).sum() if grad_norm_sq_e is None else grad_norm_sq_e + ge.pow(2).sum()
        num = loss_e.detach().to(torch.float32) - self.f_star - dot
        denom = grad_norm_sq_e.clamp(min=eps)
        gamma = max(0.0, min((num / denom).item(), self.gamma_b))
        for group in self.param_groups:
            group["lr"] = gamma


class SAM_Fixed(_SAMBase):
    """USAM/SAM with a fixed learning rate (constant or externally scheduled)."""

    def __init__(self, params, weight_decay=5e-4, rho=0.1, lambd=1.0, lr=0.1):
        super().__init__(params, weight_decay=weight_decay, rho=rho, lambd=lambd)
        for g in self.param_groups:
            g["lr"] = lr

    def _set_gamma(self, loss_e):
        pass  # lr already set per epoch by the scheduler


def cosine_gamma(epoch, T, gamma_min, gamma_max):
    """PyTorch CosineAnnealingLR-equivalent per-epoch learning rate (paper eq. 8)."""
    return gamma_min + (gamma_max - gamma_min) * (1 + torch.cos(torch.tensor(float(epoch) * 3.141592653589793 / T))) / 2
