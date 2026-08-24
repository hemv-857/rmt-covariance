"""Multi-seed RMT cleaning study: N/T grid, paired bootstrap CIs, spectral figure.

    python scripts/run_study.py

Outputs docs/study_results.json, docs/eigen_spectrum.png, docs/sharpe_by_seed.png,
docs/table.md, and compiles docs/rmt_report.pdf via pandoc (if available).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmtcov.backtest import bootstrap_sharpe_diff, factor_model_prices, run_backtest
from rmtcov.rmt import mp_support

Path("docs").mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
SCENARIOS = {  # name -> (n_assets, days, k_factors)
    "N/T=0.10 (100/1000)": (100, 1000, 3),
    "N/T=0.25 (100/400)": (100, 400, 3),
    "N/T=0.50 (200/400)": (200, 400, 5),
}


def main() -> None:
    out = {"scenarios": {}}
    spectra_for_fig = None

    for name, (n, days, k) in SCENARIOS.items():
        per_seed = {"sample": [], "ledoit_wolf": [], "rmt_clean": []}
        rets_by_seed = []
        for seed in SEEDS:
            prices = factor_model_prices(n, days, k_factors=k, seed=seed)
            res = run_backtest(prices, window=min(250, days // 3),
                               rebalance_every=21, cost_bps=5.0)
            for m, vals in per_seed.items():
                vals.append(res[m]["sharpe_net"])
            rets_by_seed.append({m: np.asarray(res[m]["net_returns"]) for m in per_seed})
            if seed == 0 and name == list(SCENARIOS)[1]:
                spectra_for_fig = np.linalg.eigvalsh(
                    np.corrcoef(np.diff(np.log(prices), axis=0).T))

        # paired bootstrap: RMT vs sample, LW vs sample (first seed's return streams)
        rb = bootstrap_sharpe_diff(rets_by_seed[0]["rmt_clean"],
                                   rets_by_seed[0]["sample"], n_boot=1500)
        lb = bootstrap_sharpe_diff(rets_by_seed[0]["ledoit_wolf"],
                                   rets_by_seed[0]["sample"], n_boot=1500)
        out["scenarios"][name] = {
            "mean_sharpe_net": {m: round(float(np.mean(v)), 3) for m, v in per_seed.items()},
            "std_sharpe_net": {m: round(float(np.std(v)), 3) for m, v in per_seed.items()},
            "bootstrap_rmt_minus_sample": {k2: round(v, 3) for k2, v in rb.items()},
            "bootstrap_lw_minus_sample": {k2: round(v, 3) for k2, v in lb.items()},
        }
        print(f"{name}: " + "  ".join(
            f"{m}={np.mean(v):.3f}" for m, v in per_seed.items()))

    # ---- figure 1: eigenvalue spectrum vs MP edge
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if spectra_for_fig is not None:
        n_assets = 100
        q = n_assets / 400.0
        _, lam_p = mp_support(q, 1.0)
        ax.hist(spectra_for_fig, bins=60, density=True, alpha=0.75,
                label="eigenvalues (N/T=0.25 panel)")
        ax.axvline(lam_p, color="crimson", ls="--",
                   label=f"MP edge $\\lambda_+={lam_p:.2f}$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title("Correlation spectrum vs Marchenko-Pastur edge")
    fig.tight_layout()
    fig.savefig("docs/eigen_spectrum.png", dpi=130)

    # ---- figure 2: net Sharpe by method across seeds
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    methods = ["sample", "ledoit_wolf", "rmt_clean"]
    width = 0.25
    for j, m in enumerate(methods):
        vals = [out["scenarios"][s]["mean_sharpe_net"][m] for s in SCENARIOS]
        stds = [out["scenarios"][s]["std_sharpe_net"][m] for s in SCENARIOS]
        ax.bar(np.arange(len(SCENARIOS)) + (j - 1) * width, vals, width,
               yerr=stds, capsize=3, label=m)
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(list(SCENARIOS), fontsize=8)
    ax.set_ylabel("net Sharpe (mean over seeds)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_title("Min-variance net Sharpe by covariance estimator")
    fig.tight_layout()
    fig.savefig("docs/sharpe_by_seed.png", dpi=130)

    Path("docs").mkdir(exist_ok=True)
    with open("docs/study_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved docs/study_results.json + figures")


if __name__ == "__main__":
    main()
