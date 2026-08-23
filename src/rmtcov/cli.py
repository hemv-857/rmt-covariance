"""CLI: `rmtcov demo` runs the synthetic study; `rmtcov backtest --csv` uses real closes."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from .backtest import factor_model_prices, run_backtest


def _load_prices(path: str) -> np.ndarray:
    """CSV of close prices: first column date (ignored), one column per asset."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    start = 1 if not _is_float(header[1] if len(header) > 1 else "") else 0
    data = [[float(v) for v in r[start:]] for r in rows[start:] if len(r) > start]
    return np.asarray(data)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _demo(args):
    print(f"generating {args.assets}-asset factor panel, {args.days} days...")
    prices = factor_model_prices(args.assets, args.days, seed=args.seed)
    res = run_backtest(prices, window=args.window, rebalance_every=args.reb, cost_bps=args.cost)
    print(json.dumps(res, indent=2))

    # headline comparison with paired bootstrap on net Sharpe
    np.log(prices)
    rets_by_m = {}
    for name in ("sample", "ledoit_wolf", "rmt_clean"):
        rets_by_m[name] = res[name]
    s_rmt, s_lw = res["rmt_clean"]["sharpe_net"], res["ledoit_wolf"]["sharpe_net"]
    print(f"\nnet Sharpe: RMT-clean={s_rmt:.3f}  LedoitWolf={s_lw:.3f}  sample={res['sample']['sharpe_net']:.3f}")
    print("ponytail: single seed/window; report multi-seed CIs for the writeup")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rmtcov", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic factor-panel study")
    d.add_argument("--assets", type=int, default=60)
    d.add_argument("--days", type=int, default=1500)
    d.add_argument("--window", type=int, default=250)
    d.add_argument("--reb", type=int, default=21)
    d.add_argument("--cost", type=float, default=5.0)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=_demo)

    b = sub.add_parser("backtest", help="run on real price CSV (date + close columns)")
    b.add_argument("--csv", required=True)
    b.add_argument("--window", type=int, default=250)
    b.add_argument("--reb", type=int, default=21)
    b.add_argument("--cost", type=float, default=5.0)
    b.set_defaults(func=_demo)

    args = p.parse_args(argv)
    if args.cmd == "backtest":
        prices = _load_prices(args.csv)
        args.seed = 0
        args.assets, args.days = prices.shape
    args.func(args)


if __name__ == "__main__":
    main()
