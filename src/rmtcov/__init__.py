"""rmt-covariance: random-matrix-theory covariance cleaning for portfolio choice."""

from .backtest import bootstrap_sharpe_diff, factor_model_prices, run_backtest
from .rmt import (
    clean_rmt,
    fit_noise_from_bulk,
    ledoit_wolf,
    min_variance_weights,
    mp_density,
    mp_support,
)

__version__ = "0.1.0"
__all__ = [
    "bootstrap_sharpe_diff",
    "clean_rmt",
    "factor_model_prices",
    "fit_noise_from_bulk",
    "ledoit_wolf",
    "min_variance_weights",
    "mp_density",
    "mp_support",
    "run_backtest",
]
