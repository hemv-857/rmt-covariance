"""Walk-forward min-variance backtest: sample vs Ledoit-Wolf vs RMT-cleaned."""

from __future__ import annotations

import numpy as np

from .rmt import clean_rmt, ledoit_wolf, min_variance_weights

__all__ = ["bootstrap_sharpe_diff", "factor_model_prices", "run_backtest"]


def factor_model_prices(
    n_assets: int, T_days: int, k_factors: int = 3, seed: int = 0
) -> np.ndarray:
    """Correlated daily price panel from a factor model (offline stand-in for data).

    ponytail: synthetic factor prices make the pipeline testable offline; swap
    in `--csv` real closes for the report.
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T_days, k_factors)) * 0.01
    beta = rng.standard_normal((n_assets, k_factors))
    idio = rng.standard_normal((T_days, n_assets)) * 0.015
    rets = F @ beta.T + idio
    return 100 * np.cumprod(1 + rets, axis=0)


def run_backtest(
    prices: np.ndarray,
    window: int = 250,
    rebalance_every: int = 21,
    cost_bps: float = 5.0,
) -> dict[str, dict[str, float]]:
    """Monthly-rebalanced min-variance, three covariance estimators.

    Strictly trailing window: estimator at time t uses rows [t-window, t).
    Returns per-method {ann_return, ann_vol, sharpe, turnover_cost_share}.
    """
    logp = np.log(prices)
    methods = {"sample": None, "ledoit_wolf": ledoit_wolf, "rmt_clean": "rmt"}
    results = {m: {"ret": [], "turnover": [], "w_prev": np.zeros(prices.shape[1])} for m in methods}
    dates: list[int] = []

    for t in range(window, len(prices) - 1, rebalance_every):
        R = np.diff(logp[t - window : t], axis=0)  # strictly past returns
        cov_raw = R.T @ R / (window - 1)
        dates.append(t)
        for name, fn in methods.items():
            if fn is None:
                cov = cov_raw
            elif fn == "rmt":
                cov = clean_rmt(cov_raw, window)["matrix"]
            else:
                cov = fn(cov_raw)
            w = min_variance_weights(cov)
            # realized return over next rebalance period
            seg = np.diff(logp[t : t + rebalance_every], axis=0)
            port = float(np.sum(seg @ w))
            prev_w = results[name]["w_prev"]
            results[name]["turnover"].append(float(np.abs(w - prev_w).sum()))
            results[name]["w_prev"] = w
            results[name]["ret"].append(port)

    out = {}
    periods_per_year = 252 / rebalance_every
    for name, r in results.items():
        rets = np.asarray(r["ret"])
        mean_p = rets.mean()
        vol_p = rets.std(ddof=1)
        costs = np.asarray(r["turnover"]) * (cost_bps / 1e4)
        net = rets - costs
        out[name] = {
            "ann_return": float(mean_p * periods_per_year),
            "ann_vol": float(vol_p * np.sqrt(periods_per_year)),
            "sharpe_gross": float(mean_p / vol_p * np.sqrt(periods_per_year)) if vol_p > 0 else 0.0,
            "sharpe_net": (
                float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year))
                if net.std(ddof=1) > 0
                else 0.0
            ),
            "avg_turnover": float(r["turnover"][0] and np.mean(r["turnover"])),
        }
    return out


def bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap CI for Sharpe difference of two return streams."""
    rng = np.random.default_rng(0)
    a, b = np.asarray(a), np.asarray(b)

    def sr(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(252 / max(len(x) // max(1, 12), 1)) if s > 0 else 0.0

    diffs = []
    idx = np.arange(a.size)
    for _ in range(n_boot):
        take = rng.choice(idx, size=idx.size, replace=True)
        diffs.append(sr(a[take]) - sr(b[take]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi)}
