# rmt-covariance

Random-matrix-theory cleaning of covariance matrices for minimum-variance
portfolios: **Marchenko–Pastur eigenvalue clipping** (Laloux–Bouchaud–Potters)
vs **Ledoit–Wolf shrinkage** vs the raw sample covariance — compared on a
walk-forward backtest with honest costs.

```
bulk of noise eigenvalues  ->  Marchenko-Pastur support (sigma^2 (1±√q)^2, q = N/T)
eigenvalues inside bulk    ->  replaced by bulk average (noise carries no signal)
outliers above the edge    ->  kept (genuine factors)
```

## Quickstart

```bash
pip install -e ".[dev]"
make test
python -m rmtcov.cli demo --assets 60 --days 1500
python -m rmtcov.cli backtest --csv closes.csv --window 250 --cost 5
```

## Correctness gates (enforced in tests)

- Pure-noise correlation spectra sit inside the MP support; density integrates to 1
- Noise-sigma fit recovers scale within a loose band
- Clipping keeps ≥1 signal eigenvector and collapses the bulk spread; output symmetric
- Ledoit–Wolf output is PSD with preserved trace
- GMV weights: sum-to-one, long-only feasible, variance ≤ equal-weight
- Backtest window is strictly trailing (`t-window, t`), never touching traded period

## Study design (for the report)

1. S&P 100 constituents as-of-date (survivorship!), daily closes.
2. Grid over `(N/T)` ratios and rebalance frequencies.
3. Paired bootstrap CIs on net-Sharpe differences between cleaners.
4. Report turnover and cost sensitivity — cleaning often wins *before* costs.

## Honest scope

- `ponytail:` identity-target LW and clipping only. Nonlinear shrinkage
  (Donoho–Gavish–Johnstone) and factor-targets are the next rungs.
- Synthetic demo uses a 3-factor DGP so results are diagnostic, not evidence;
  real-data runs via `--csv` are the deliverable.
