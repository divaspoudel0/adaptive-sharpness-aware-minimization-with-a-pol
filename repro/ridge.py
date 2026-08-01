import numpy as np


def sq_matrix_with_given_cond(n, cond, seed=0):
    """Symmetric PSD matrix with given condition number (Lenard-style construction)."""
    rng = np.random.default_rng(seed)
    log_cond = np.log(cond)
    exp_vec = np.arange(-log_cond / 4.0, log_cond * (n + 1) / (4.0 * (n - 1)), log_cond / (2.0 * (n - 1)))
    exp_vec = exp_vec[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((rng.random((n, n)) - 5.0) * 200)
    V, _ = np.linalg.qr((rng.random((n, n)) - 5.0) * 200)
    P = U @ S @ V.T
    P = P @ P.T
    return P


class RidgeRegression:
    """
    Ridge regression objective f(x) = (1/2n) sum_i (A_i x - b_i)^2 + (lambda_r/2)||x||^2,
    matching the paper's synthetic setup (n=d=100, cond(A)=10).
    """

    def __init__(self, n=100, d=100, lambd=0.0, cond=10.0, consistent=False, x0_seed=0, data_seed=0):
        self.n = n
        self.d = d
        self.lambd = lambd
        assert n == d and cond is not None
        self.A = sq_matrix_with_given_cond(n, cond, seed=data_seed)
        # f(x) = (1/2n)||Ax-b||^2 + (lambd/2)||x||^2  =>  Hessian = A^T A / n + lambd I
        w = np.linalg.eigvalsh(self.A @ self.A)
        self.L = w.max() / n + lambd
        self.mu = w.min() / n + lambd
        self.kappa = self.L / self.mu if self.mu > 0 else np.inf

        if consistent:
            self.f_star = 0.0
            rng = np.random.default_rng(x0_seed)
            self.x_star = rng.standard_normal(self.d)
            self.b = self.A @ self.x_star
        else:
            self.f_star = (lambd / 2.0) * 0.0  # placeholder; f* computed analytically below
            rng = np.random.default_rng(x0_seed)
            self.x_star = np.zeros(self.d)
            self.b = rng.standard_normal(self.n)
            # f* for non-consistent ridge: min over x of (1/2n)||Ax-b||^2 + (lambd/2)||x||^2
            M = self.A / np.sqrt(self.n)
            bb = self.b / np.sqrt(self.n)
            if lambd > 0:
                self.x_star = np.linalg.solve(M.T @ M + lambd * np.eye(self.d), M.T @ bb)
            else:
                self.x_star = np.linalg.lstsq(M, bb, rcond=None)[0]
            self.f_star = float(0.5 * np.sum((M @ self.x_star - bb) ** 2) + (lambd / 2.0) * np.sum(self.x_star ** 2))

        self.L_i = (np.linalg.norm(self.A, axis=1) ** 2) + lambd  # ||A_i||^2 + lambd = L_i for each component

    def func(self, x, batch):
        r = self.A[batch] @ x - self.b[batch]
        return float(0.5 * np.sum(r ** 2) / batch.shape[0] + (self.lambd / 2.0) * np.sum(x ** 2))

    def grad(self, x, batch):
        r = self.A[batch] @ x - self.b[batch]
        return (self.A[batch].T @ r) / batch.shape[0] + self.lambd * x


def make_rank_deficient(n=100, d=100, cond=10.0, seed=0, dropped_cols=1):
    """Return an A (n x d) that is PSD-symmetric with one zero column => rank d-1 (mu=0)."""
    A = sq_matrix_with_given_cond(n, cond, seed=seed)
    A = A.copy()
    for c in range(dropped_cols):
        A[:, d - 1 - c] = 0.0
        A[d - 1 - c, :] = 0.0
    return A


class ConvexRidge(RidgeRegression):
    """Convex (not strongly convex) ridge: same objective but with a rank-deficient A."""

    def __init__(self, n=100, d=100, cond=10.0, seed=0, dropped_cols=1, consistent=True, noise_scale=0.0):
        super().__init__(n=n, d=d, lambd=0.0, cond=cond, consistent=True, x0_seed=seed, data_seed=seed)
        self.A = make_rank_deficient(n, d, cond, seed=seed, dropped_cols=dropped_cols)
        w = np.linalg.eigvalsh(self.A @ self.A)
        self.L = w.max() / n
        self.mu = 0.0
        self.kappa = np.inf
        self.b = self.A @ self.x_star  # consistent with the rank-deficient A
        if not consistent:
            rng = np.random.default_rng(seed + 1000)
            self.b = self.b + noise_scale * rng.standard_normal(n)
        # closed-form f* for the (possibly non-consistent) objective
        M = self.A / np.sqrt(n)
        bb = self.b / np.sqrt(n)
        self.x_star = np.linalg.lstsq(M, bb, rcond=None)[0]
        self.f_star = float(0.5 * np.sum((M @ self.x_star - bb) ** 2))
        self.L_i = (np.linalg.norm(self.A, axis=1) ** 2)  # ||A_i||^2 = L_i
