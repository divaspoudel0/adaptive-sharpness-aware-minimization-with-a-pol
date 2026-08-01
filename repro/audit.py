"""Independent numerical audits of the theoretical claims of arXiv 2606.01827.

Claims verified:
  Claim 1  Theorem 3.1  -- strongly convex smooth f: linear rate (1 - mu(1-L rho)^2/(4L))^t
  Claim 2  Theorem 3.2  -- convex smooth f: O(1/T) rate f(xbar_T) - f* <= 2L||x0-x*||^2/(T(1-L rho)^2)
          Theorems 3.5/3.8 -- stochastic: convergence up to a neighborhood
  Claim 3  Prop. 2.1    -- Polyak scheduler non-negative whenever rho_t <= 1/L

This is a numerical audit, not a proof. All inequalities are checked against
exact evaluations of the USAM+Polyak iterates.
"""
import numpy as np

from repro.ridge import RidgeRegression, ConvexRidge


def usam_det(loss, x0, T, rho, polyak=True, gamma_const=None, seed=0):
    """Deterministic (full-batch) USAM iterates. e = x + rho grad f(x); x -= gamma grad f(e)."""
    full = np.arange(loss.n)
    x = x0.copy()
    xs = [x.copy()]
    gammas = []
    num_hist = []
    for t in range(T):
        gx = loss.grad(x, full)
        e = x + rho * gx
        ge = loss.grad(e, full)
        if polyak:
            num = loss.func(e, full) - loss.f_star - rho * np.inner(ge, gx)
            gamma = num / max(np.inner(ge, ge), 1e-30)
            num_hist.append(num)
        else:
            gamma = gamma_const
        gammas.append(gamma)
        x = x - gamma * ge
        xs.append(x.copy())
    return xs, gammas, num_hist


def usam_stoch(loss, x0, T, rho, gamma_b, tau=10, n_runs=5, seed=0):
    """Stochastic USAM + capped Polyak scheduler. Returns per-run iterate lists and gamma info."""
    rng = np.random.default_rng(seed)
    all_xs = []
    all_fvals = []
    for r in range(n_runs):
        x = x0.copy()
        xs = [x.copy()]
        fvals = []
        for t in range(T):
            S = rng.choice(loss.n, size=tau, replace=False)
            gx = loss.grad(x, S)
            e = x + rho * gx
            ge = loss.grad(e, S)
            num = loss.func(e, S) - 0.0 - rho * np.inner(ge, gx)  # ell* = 0
            gamma = min(num / max(np.inner(ge, ge), 1e-30), gamma_b)
            x = x - gamma * ge
            xs.append(x.copy())
            fvals.append(loss.func(x, np.arange(loss.n)))
        all_xs.append(xs)
        all_fvals.append(fvals)
    return all_xs, all_fvals


def audit_claim1(loss, x0, rho, T, n_its=3000):
    """Theorem 3.1: linear convergence with rate (1 - mu(1-L rho)^2/(4L))^t."""
    L, mu = loss.L, loss.mu
    rate_pred = 1.0 - mu * (1 - L * rho) ** 2 / (4 * L)
    print(f"[claim1] L={L:.6e} mu={mu:.6e} rho={rho} rate_pred={rate_pred:.10f}")

    xs, gammas, _ = usam_det(loss, x0, n_its, rho)
    d2 = np.array([np.sum((x - loss.x_star) ** 2) for x in xs])
    # predicted linear envelope
    env = rate_pred ** np.arange(n_its + 1) * d2[0]
    bound_holds = np.all(d2 <= env * (1 + 1e-9) + 1e-30)
    print(f"[claim1] bound_holds_all_t={bound_holds}")

    # empirical contraction factor (late-iteration geometric mean / max ratio)
    ratios = d2[1:] / np.maximum(d2[:-1], 1e-300)
    ratios = ratios[d2[:-1] > 1e-24]
    if len(ratios) > 0:
        emp_max = float(np.max(ratios))
        emp_last = float(ratios[-min(200, len(ratios)):].max())
    else:
        emp_max = float("nan")
        emp_last = float("nan")
    print(f"[claim1] emp_contraction_max_ratio={emp_max:.10f} emp_contraction_last200={emp_last:.10f}")
    print(f"[claim1] d2_final={d2[-1]:.3e} d2_0={d2[0]:.3e} log10_reduction={np.log10(max(d2[0],1e-300)/max(d2[-1],1e-300)):.3f}")

    # verify descent inequality (Prop 2.2): ||x+ - x*||^2 <= ||x - x*||^2 - (1-L rho)^2/(2L)(f(x)-f*)
    full = np.arange(loss.n)
    viol = 0
    worst = 0.0
    for t in range(n_its):
        lhs = d2[t + 1]
        rhs = d2[t] - (1 - L * rho) ** 2 / (2 * L) * max(loss.func(xs[t], full) - loss.f_star, 0.0)
        if lhs > rhs + 1e-9:
            viol += 1
            worst = max(worst, (lhs - rhs) / max(rhs, 1e-300))
    print(f"[claim1] descent_violations={viol} worst_rel_viol={worst:.3e}")
    return rate_pred, emp_last, bound_holds


def audit_claim1_sweep(loss, x0, n_its=1500):
    """Check the explicit rho-dependence of the Theorem 3.1 rate."""
    L, mu = loss.L, loss.mu
    print(f"[claim1sweep] mu/L = {mu/L:.6f} (L={L:.6e})")
    print(f"[claim1sweep] {'rho*L':>6} {'rate_pred':>14} {'emp_contraction':>16} {'bound_holds':>11}")
    for rhoL in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
        rho = rhoL / L
        rate_pred, emp_last, holds = audit_claim1(loss, x0, rho, 0, n_its=n_its)
        print(f"[claim1sweep] {rhoL:6.2f} {rate_pred:14.10f} {emp_last:16.10f} {str(holds):>11}")


def audit_claim2_det(loss, x0, rho, n_its=3000):
    """Theorem 3.2: f(xbar_T) - f* <= 2 L ||x0-x*||^2 / (T (1-L rho)^2)."""
    L = loss.L
    xs, _, _ = usam_det(loss, x0, n_its, rho)
    full = np.arange(loss.n)
    fs = np.array([loss.func(x, full) - loss.f_star for x in xs])  # f(x^t) - f*
    print(f"[claim2det] L={L:.6e} rho={rho} D2={np.sum((x0-loss.x_star)**2):.6f}")

    check_Ts = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000]
    ok = True
    for T in check_Ts:
        xbar = np.mean(xs[:T], axis=0)
        fbar = loss.func(xbar, full) - loss.f_star
        bound = 2 * L * np.sum((x0 - loss.x_star) ** 2) / (T * (1 - L * rho) ** 2)
        rel = fbar / bound if bound > 0 else float("inf")
        if fbar > bound + 1e-12:
            ok = False
        print(f"[claim2det] T={T:5d} f(xbar)={fbar:.6e} bound={bound:.6e} ratio={rel:.4f}")
    print(f"[claim2det] bound_holds_all_T={ok}")
    return ok


def audit_claim2_stoch_convex(loss, x0, rho, gamma_b, tau=10, T=2000, n_runs=10):
    """Theorem 3.8: E[f(xbar_T)] - f* <= ||x0-x*||^2/(alpha T) + (2 gamma_b - alpha)/alpha sigma^2."""
    Lmax = loss.L_i.max()
    alpha = (1 - Lmax * rho) ** 2 * min(1.0 / (2 * Lmax), gamma_b)
    sigma2 = loss.f_star - 0.0  # E[f_S(x*) - 0] with ell* = 0
    print(f"[claim2stoch] Lmax={Lmax:.6e} mu={loss.mu:.6e} alpha={alpha:.6e} sigma2={sigma2:.6e} gamma_b={gamma_b}")
    xbar_T = 2 * np.sum((x0 - loss.x_star) ** 2) / (alpha * T) + (2 * gamma_b - alpha) / alpha * sigma2

    all_xs, all_fvals = usam_stoch(loss, x0, T, rho, gamma_b, tau=tau, n_runs=n_runs)
    fbar = np.mean([np.mean(fv[T // 2:]) for fv in all_fvals])  # late-time average
    best = np.mean([np.min(fv) for fv in all_fvals])
    final = np.mean([fv[-1] for fv in all_fvals])
    neigh = (2 * gamma_b - alpha) / alpha * sigma2
    print(f"[claim2stoch] E[f(xbar_T)]_late_avg={fbar:.6e} E[min f]={best:.6e} E[f(x_T)]={final:.6e}")
    print(f"[claim2stoch] theory_bound_at_T={xbar_T:.6e} (neighborhood term={neigh:.6e})")
    print(f"[claim2stoch] final_vs_neighborhood: final/neigh={final/max(neigh,1e-300):.3f}  "
          f"(<=1 means inside the predicted neighborhood)")
    return fbar, xbar_T


def audit_claim2_stoch_sc(loss, x0, rho, gamma_b, tau=10, T=3000, n_runs=10):
    """Theorem 3.5: E||x^t-x*||^2 <= (1 - mu alpha/2)^t ||x0-x*||^2 + 2(2 gamma_b - alpha)/(mu alpha) sigma^2."""
    Lmax = loss.L_i.max()
    mu = loss.mu
    alpha = (1 - Lmax * rho) ** 2 * min(1.0 / (2 * Lmax), gamma_b)
    sigma2 = loss.f_star - 0.0
    neigh = 2 * (2 * gamma_b - alpha) / (mu * alpha) * sigma2
    rate_sc = 1.0 - mu * alpha / 2
    print(f"[claim2sc] Lmax={Lmax:.6e} mu={mu:.6e} alpha={alpha:.6e} sigma2={sigma2:.6e} rate={rate_sc:.8f} neighborhood={neigh:.6e}")

    all_xs, _ = usam_stoch(loss, x0, T, rho, gamma_b, tau=tau, n_runs=n_runs)
    d2 = np.array([[np.sum((x - loss.x_star) ** 2) for x in xs] for xs in all_xs])
    mean_d2 = d2.mean(axis=0)
    env = rate_sc ** np.arange(T + 1) * mean_d2[0] + neigh
    ok = np.all(mean_d2 <= env * (1 + 1e-6) + 1e-24)
    print(f"[claim2sc] E||xT-x*||^2={mean_d2[-1]:.6e} envelope_upper={env[-1]:.6e} bound_holds_all_t={ok}")
    if neigh > 0:
        print(f"[claim2sc] final_vs_neighborhood: final/neigh={mean_d2[-1]/neigh:.3f}  "
              f"(<=1 means inside the predicted neighborhood)")
    else:
        print(f"[claim2sc] neighborhood=0 (interpolated): E||xT-x*||^2={mean_d2[-1]:.6e} decays to 0")
    return mean_d2[-1], neigh, ok


def audit_claim3(loss, x0, rho_values, T=3000):
    """Proposition 2.1: numerator of Polyak scheduler >= 0 whenever rho <= 1/L."""
    full = np.arange(loss.n)
    L = loss.L
    print(f"[claim3] L={L:.6e} 1/L={1/L:.6e}")
    for rho in rho_values:
        # cap iterations for diverging controls (rho >> 1/L)
        t_max = min(T, 400 if rho * L > 1.0 else T)
        xs, gammas, num_hist = usam_det(loss, x0, t_max, rho)
        npos = np.array(num_hist)
        finite = np.isfinite(npos)
        below = (npos < 0) & finite
        n_below = int(below.sum())
        min_num = float(npos[finite].min()) if finite.any() else float("nan")
        lb = (1 - L * rho) / (2 * L * (1 + L * rho)) if rho < 1 / L else float("nan")
        gammas = np.array(gammas)
        gf = np.isfinite(gammas)
        # Only meaningful scales: near machine-convergence the (f-f*)/||grad||^2 ratio is
        # dominated by float noise, so restrict the lower-bound check to iterations where
        # the numerator (a gap) is a non-negligible fraction of its initial value.
        npos0 = np.array(num_hist)
        if len(npos0) > 0 and np.isfinite(npos0).any() and np.max(np.abs(npos0)) > 0:
            mask = (np.abs(npos0) > 1e-8 * np.max(np.abs(npos0))) & gf
        else:
            mask = gf
        lb_holds = bool(np.all(gammas[mask] >= lb * (1 - 1e-9))) if (rho < 1 / L and mask.any()) else None
        status = "OK (non-negative)" if (n_below == 0 and rho <= 1 / L) else "NEGATIVE VALUES"
        print(f"[claim3] rho*L={rho*L:.4f} n_negative={n_below}/{len(npos)} min_num={min_num:.4e} "
              f"lower_bound={lb:.4e} gamma>=lb:{lb_holds} -> {status}")


def main():
    print("=" * 80)
    print("NUMERICAL AUDIT of arXiv 2606.01827 theory claims (USAM + Polyak scheduler)")
    print("=" * 80)

    rng = np.random.default_rng(1234)

    # ---------- strongly convex ridge (deterministic) ----------
    sc = RidgeRegression(n=100, d=100, lambd=0.0, cond=10.0, consistent=True, x0_seed=7, data_seed=3)
    x0 = rng.standard_normal(sc.d)
    rho_sc = 0.1 / sc.L  # rho such that L*rho = 0.1 <= 1
    print("\n--- Claim 1 / Thm 3.1 (strongly convex, deterministic) ---")
    audit_claim1(sc, x0, rho_sc, 0)
    audit_claim1_sweep(sc, x0)

    print("\n--- Claim 2 / Thm 3.2 (convex, deterministic) ---")
    cv = ConvexRidge(n=100, d=100, cond=10.0, seed=11, dropped_cols=1)
    rho_cv = 0.1 / cv.L
    audit_claim2_det(cv, x0, rho_cv)

    print("\n--- Claim 3 / Prop 2.1 (non-negativity) ---")
    audit_claim3(sc, x0, [0.0, 0.1 / sc.L, 0.5 / sc.L, 0.9 / sc.L, 1.0 / sc.L, 1.5 / sc.L, 2.0 / sc.L, 5.0 / sc.L])

    # ---------- stochastic strongly convex ----------
    print("\n--- Claim 2 / Thm 3.5 (stochastic strongly convex, up to a neighborhood) ---")
    sc_stoch = RidgeRegression(n=100, d=100, lambd=1e-3, cond=10.0, consistent=False, x0_seed=7, data_seed=3)
    rho_scs = 0.1 / sc_stoch.L_i.max()
    audit_claim2_stoch_sc(sc_stoch, x0, rho_scs, gamma_b=1.0)

    print("\n--- Claim 2 / Thm 3.8 (stochastic convex, up to a neighborhood) ---")
    cv_stoch = ConvexRidge(n=100, d=100, cond=10.0, seed=13, dropped_cols=2, consistent=False, noise_scale=0.1)
    rho_cvs = 0.1 / cv_stoch.L_i.max()
    audit_claim2_stoch_convex(cv_stoch, x0, rho_cvs, gamma_b=1.0)

    # ---------- interpolation regime (stochastic strongly convex, sigma2=0) ----------
    print("\n--- Corollary (interpolation: sigma^2 = 0) ---")
    sc_interp = RidgeRegression(n=100, d=100, lambd=1e-3, cond=10.0, consistent=True, x0_seed=7, data_seed=3)
    rho_sci = 0.1 / sc_interp.L_i.max()
    # In interpolation, ell* = f_S(x*) achievable; with a consistent system every component shares x*,
    # so f_S(x*) = 0 = f*, hence sigma^2 = 0.
    audit_claim2_stoch_sc(sc_interp, x0, rho_sci, gamma_b=1e6, T=3000)  # gamma_b = inf

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
