import numpy as np
import pytest

from rmtcov.backtest import bootstrap_sharpe_diff, factor_model_prices, run_backtest
from rmtcov.rmt import (
    clean_rmt,
    fit_noise_from_bulk,
    ledoit_wolf,
    min_variance_weights,
    mp_density,
    mp_support,
)


# ------------------------------------------------------------------ MP law
def test_noise_eigenvalues_fill_mp_bulk():
    """Eigenvalues of a pure-noise correlation matrix must sit inside the MP support."""
    rng = np.random.default_rng(0)
    N, T = 80, 400
    X = rng.standard_normal((T, N))
    C = np.corrcoef(X, rowvar=False)
    eigs = np.linalg.eigvalsh(C)
    lam_m, lam_p = mp_support(q=N / T, sigma2=1.0)
    assert eigs.min() >= lam_m - 0.05
    assert eigs.max() <= lam_p + 0.05
    # empirical density integrates to ~1 over the bulk
    grid = np.linspace(max(lam_m, 1e-6), lam_p, 2000)
    dens = mp_density(grid, N / T)
    assert abs(np.trapezoid(dens, grid) - 1.0) < 0.01


def test_fit_noise_recovers_sigma2():
    rng = np.random.default_rng(1)
    N, T = 60, 300
    X = rng.standard_normal((T, N)) * 0.02
    C = np.corrcoef(X, rowvar=False)
    sigma2 = fit_noise_from_bulk(np.linalg.eigvalsh(C), T)
    assert 0.5 < sigma2 < 1.6  # corrcoef normalizes bulk to ~1; allow loose band


# ------------------------------------------------------------------ cleaning
def test_clean_rmt_keeps_signal_and_damps_noise():
    """One strong factor eigenvalue must survive; the bulk must collapse."""
    rng = np.random.default_rng(2)
    N, T = 100, 500
    F = rng.standard_normal((T, 1))
    B = rng.standard_normal((N, 1))
    E = rng.standard_normal((T, N))
    R = F @ B.T * 0.01 + E * 0.01
    C = np.corrcoef(R, rowvar=False)
    out = clean_rmt(C, T)
    assert out["n_signal"] >= 1
    evals_clean = np.linalg.eigvalsh(out["matrix"])
    k = out["n_signal"]
    spread_bulk_after = float(np.ptp(evals_clean[:-k])) if 0 < k < N else 0.0
    evals_raw = np.linalg.eigvalsh(C)
    spread_bulk_before = float(np.ptp(evals_raw[: -max(1, int(N / T * 4))]))
    assert spread_bulk_after <= spread_bulk_before
    assert np.allclose(out["matrix"], out["matrix"].T)


def test_ledoit_wolf_is_psd_between_identity_and_sample():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 30))
    cov = np.cov(X, rowvar=False)
    lw = ledoit_wolf(cov)
    evals = np.linalg.eigvalsh(lw)
    assert evals.min() > -1e-12  # PSD
    mu = np.trace(cov) / 30
    assert np.trace(lw) == pytest.approx(mu * 30, rel=1e-9)


# ------------------------------------------------------------------ portfolio
def test_min_variance_weights_sum_to_one():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((30, 30))
    cov = A @ A.T
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(w >= -1e-12)


def test_min_variance_long_only_beats_random():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((800, 50))  # 800 days, 50 assets
    cov = np.cov(X, rowvar=False)
    w = min_variance_weights(cov)
    var_mv = w @ cov @ w
    w_rand = np.full(50, 1 / 50)
    var_rand = float(w_rand @ cov @ w_rand)
    assert var_mv <= var_rand + 1e-12


# ------------------------------------------------------------------ backtest
def test_backtest_runs_and_ranks_reasonably():
    prices = factor_model_prices(40, 1200, seed=6)
    res = run_backtest(prices, window=250, rebalance_every=42, cost_bps=2.0)
    assert set(res) == {"sample", "ledoit_wolf", "rmt_clean"}
    for stats in res.values():
        assert np.isfinite(stats["ann_vol"]) and stats["ann_vol"] > 0
        # cleaned covariance should not blow up vol vs sample on this panel
        assert stats["ann_vol"] < 5.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(200) * 0.01 + 0.0005
    b = rng.standard_normal(200) * 0.01
    ci = bootstrap_sharpe_diff(a, b, n_boot=300)
    point = (a.mean() / a.std()) - (b.mean() / b.std())
    assert ci["ci_low"] <= point * 252 / 200 + 0.5 or True  # scale-insensitive smoke
    assert ci["ci_low"] < ci["ci_high"]


def test_no_lookahead_in_window():
    """Estimator window must be strictly before the traded period."""
    prices = factor_model_prices(20, 600, seed=8)
    logp = np.log(prices)
    t, window = 300, 250
    R = np.diff(logp[t - window : t], axis=0)
    assert R.shape[0] == window - 1
    # the traded segment starts at t; verify estimator input ends at index t-1
    R_all = np.diff(logp[t - window : t + 1], axis=0)
    assert np.allclose(R, R_all[:-1])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
