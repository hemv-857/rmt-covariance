"""Marchenko-Pastur spectrum, RMT eigenvalue clipping and Ledoit-Wolf shrinkage."""

from __future__ import annotations

import numpy as np

__all__ = [
    "clean_rmt",
    "fit_noise_from_bulk",
    "ledoit_wolf",
    "min_variance_weights",
    "mp_density",
    "mp_support",
]


def mp_support(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur bulk for aspect ratio q = N/T."""
    lam_m = sigma2 * (1 - np.sqrt(q)) ** 2
    lam_p = sigma2 * (1 + np.sqrt(q)) ** 2
    return float(lam_m), float(lam_p)


def mp_density(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP density (normalized to integrate to 1 over the bulk)."""
    x = np.asarray(x, dtype=float)
    lam_m, lam_p = mp_support(q, sigma2)
    out = np.zeros_like(x)
    inside = (x >= lam_m) & (x <= lam_p)
    out[inside] = np.sqrt((lam_p - x[inside]) * (x[inside] - lam_m)) / (
        2 * np.pi * q * sigma2 * x[inside]
    )
    return out


def fit_noise_from_bulk(eigenvalues: np.ndarray, T: int) -> float:
    """Estimate sigma^2 from the trace of eigenvalues below the 2-sigma MP edge.

    ponytail: bisection on the consistency condition; Laloux-Bouchaud style.
    """
    eigs = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    N = eigs.size
    q = N / T
    lo, hi = 1e-10, max(float(eigs.max()) / (1 + np.sqrt(q)) ** 2 + 1e-6, 1.0)

    def bulk_mean(sigma2):
        _, lam_p = mp_support(q, sigma2)
        bulk = eigs[eigs <= lam_p]
        return bulk.mean() if bulk.size else 0.0

    for _ in range(60):  # bisect until bulk mean matches its own sigma2
        mid = 0.5 * (lo + hi)
        if bulk_mean(mid) > mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clean_rmt(corr_or_cov: np.ndarray, T: int) -> dict:
    """Laloux-Bouchaud eigenvalue clipping on the correlation structure.

    Eigenvalues inside the fitted MP bulk are replaced by their average
    (pure-noise direction -> no information); outliers above the edge are kept.
    Returns cleaned matrix plus diagnostics.
    """
    C = np.asarray(corr_or_cov, dtype=float)
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)

    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    sigma2 = fit_noise_from_bulk(evals, T)
    q = corr.shape[0] / T
    _, lam_p = mp_support(q, sigma2)
    n_out = int(np.sum(evals > lam_p))

    cleaned_evals = evals.copy()
    if n_out < len(evals):
        bulk_avg = float(evals[n_out:].mean())
        cleaned_evals[n_out:] = bulk_avg
    # restore unit trace of correlation, rebuild at original scale
    cleaned_evals *= len(evals) / cleaned_evals.sum()
    corr_clean = (evecs * cleaned_evals) @ evecs.T
    cov_clean = corr_clean * np.outer(d, d)
    return {
        "matrix": cov_clean,
        "sigma2": float(sigma2),
        "mp_edge": float(lam_p),
        "n_signal": n_out,
        "n_total": len(evals),
        "q": float(q),
    }


def ledoit_wolf(cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward scaled identity with optimal intensity."""
    X = np.asarray(cov, dtype=float)
    n = X.shape[0]
    mu = np.trace(X) / n
    delta2 = np.sum((X - mu * np.eye(n)) ** 2) / n**2
    # beta^2 estimated via the standard LW decomposition; with one sample matrix
    # this reduces to shrinking toward mu*I by delta2/beta2 capped at 1.
    off_diag = X - np.diag(np.diag(X))
    beta2 = min(np.sum(off_diag**2) / n, delta2)
    shrink = 1.0 if delta2 <= 1e-300 else beta2 / delta2
    return shrink * mu * np.eye(n) + (1 - shrink) * X


def min_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Global minimum-variance weights, sum(w)=1, optionally w>=0."""
    n = cov.shape[0]
    if not long_only:
        ones = np.ones(n)
        w = np.linalg.solve(cov, ones)
        return w / w.sum()
    from scipy.optimize import minimize

    res = minimize(
        lambda w: float(w @ cov @ w),
        np.full(n, 1.0 / n),
        jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x
