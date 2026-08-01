"""Generate figures for the reproduction logbook from the theory audits."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro.ridge import RidgeRegression, ConvexRidge
from repro import audit


def _figdir():
    d = os.environ.get("REPRO_FIGS", "figures")
    os.makedirs(d, exist_ok=True)
    return d


def claim1_figures(outdir):
    rng = np.random.default_rng(1234)
    sc = RidgeRegression(n=100, d=100, lambd=0.0, cond=10.0, consistent=True, x0_seed=7, data_seed=3)
    x0 = rng.standard_normal(sc.d)
    L, mu = sc.L, sc.mu
    rho = 0.1 / L

    n_its = 3000
    xs, gammas, _ = audit.usam_det(sc, x0, n_its, rho)
    d2 = np.array([np.sum((x - sc.x_star) ** 2) for x in xs])
    rate_pred = 1.0 - mu * (1 - L * rho) ** 2 / (4 * L)
    env = rate_pred ** np.arange(n_its + 1) * d2[0]
    t = np.arange(n_its + 1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(t, d2, lw=1.5, label=r"$\|x_t-x_*\||^2$ (USAM+Polyak)")
    ax[0].semilogy(t, env, lw=1.2, ls="--", color="C3",
                   label=r"predicted $(1-\frac{\mu(1-L\rho)^2}{4L})^t$")
    ax[0].set_xlabel("iteration t")
    ax[0].set_ylabel(r"$\|x_t-x_*\|^2$")
    ax[0].set_title("Thm 3.1: linear convergence (log-scale)")
    ax[0].legend()

    rhoLs = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    preds, emps = [], []
    for rhoL in rhoLs:
        rr = rhoL / L
        rate_pred_r, emp, holds = audit.audit_claim1(sc, x0, rr, 0, n_its=1500)
        preds.append(rate_pred_r)
        emps.append(emp)
    ax[1].plot(rhoLs, preds, "o-", label="predicted rate")
    ax[1].plot(rhoLs, emps, "s--", label="empirical max contraction")
    ax[1].set_xlabel(r"$\rho L$")
    ax[1].set_ylabel("contraction factor")
    ax[1].set_title("Rate vs " + r"$\rho L$ (Thm 3.1)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "claim1_linear_convergence.png"), dpi=150)
    plt.close(fig)


def claim2_figure(outdir):
    rng = np.random.default_rng(1234)
    x0 = rng.standard_normal(100)
    cv = ConvexRidge(n=100, d=100, cond=10.0, seed=11, dropped_cols=1)
    rho = 0.1 / cv.L
    n_its = 3000
    xs, _, _ = audit.usam_det(cv, x0, n_its, rho)
    full = np.arange(cv.n)
    fs = np.array([cv.func(x, full) - cv.f_star for x in xs])
    Ts = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000]
    fbars = []
    bounds = []
    for T in Ts:
        xbar = np.mean(xs[:T], axis=0)
        fbars.append(cv.func(xbar, full) - cv.f_star)
        bounds.append(2 * cv.L * np.sum((x0 - cv.x_star) ** 2) / (T * (1 - cv.L * rho) ** 2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(Ts, fbars, "o-", label=r"$f(\bar{x}_T)-f_*$ (empirical)")
    ax.loglog(Ts, bounds, "s--", color="C3",
              label=r"$2L\|x_0-x_*\|^2/[T(1-L\rho)^2]$ (bound)")
    ax.set_xlabel("T")
    ax.set_ylabel("suboptimality")
    ax.set_title("Thm 3.2: convex O(1/T) rate")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "claim2_convex_rate.png"), dpi=150)
    plt.close(fig)


def claim3_figure(outdir):
    rng = np.random.default_rng(1234)
    sc = RidgeRegression(n=100, d=100, lambd=0.0, cond=10.0, consistent=True, x0_seed=7, data_seed=3)
    x0 = rng.standard_normal(sc.d)
    L = sc.L

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rhoL, style in [(0.0, "C0-"), (0.5, "C1-"), (0.9, "C2-")]:
        rho = rhoL / L
        T = 200
        xs, gammas, nums = audit.usam_det(sc, x0, T, rho)
        ax.semilogy(np.arange(T), np.abs(np.array(nums)) + 1e-300, style, label=f"rhoL={rhoL}")
    for rhoL, style in [(1.5, "C3--"), (2.0, "C4--")]:
        rho = rhoL / L
        T = 200
        xs, gammas, nums = audit.usam_det(sc, x0, T, rho)
        ax.semilogy(np.arange(T), np.abs(np.array(nums)) + 1e-300, style, label=f"rhoL={rhoL} (diverge)")
    ax.set_xlabel("iteration t")
    ax.set_ylabel(r"$|\nu_t|$ (Polyak numerator), log-scale")
    ax.set_title("Prop 2.1: non-negative step size for rho*L <= 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "claim3_nonnegativity.png"), dpi=150)
    plt.close(fig)


def main():
    outdir = _figdir()
    claim1_figures(outdir)
    claim2_figure(outdir)
    claim3_figure(outdir)
    print(f"figures written to {outdir}:")
    for f in sorted(os.listdir(outdir)):
        print("  ", f)


if __name__ == "__main__":
    main()
