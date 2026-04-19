"""Small simulation harness for MCNNM baseline projection experiments.

Run from the repository root with:

    PYTHONPATH=src poetry run python experiments/nonnegative/mcnnm_eval.py

The projected outputs are post-estimation companions, so this script reports raw
and projected tau side by side.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from causaltensor.cauest.MCNNM import MCNNMPanelSolver


def make_low_rank(rng, n_rows, n_cols, rank):
    U = rng.normal(size=(n_rows, rank))
    V = rng.normal(size=(n_cols, rank))
    return U @ V.T


def make_baseline(rng, n_rows, n_cols, rank, margin, fixed_effects=True):
    L = make_low_rank(rng, n_rows, n_cols, rank)
    if fixed_effects:
        a = rng.normal(scale=0.5, size=(n_rows, 1))
        b = rng.normal(scale=0.5, size=(1, n_cols))
        baseline = L + a + b
    else:
        baseline = L
    return baseline - np.min(baseline) + margin


def make_block_treatment(shape):
    Z = np.zeros(shape)
    Z[shape[0] // 2:, shape[1] // 2:] = 1
    return Z


def summarize_projection(res):
    diagnostics = res.projection_diagnostics
    return {
        "tau_raw": float(res.tau),
        "tau_projected": float(res.tau_projected),
        "tau_shift": float(res.tau_projected - res.tau),
        "baseline_min_raw": diagnostics["baseline_min_raw"],
        "baseline_min_projected": diagnostics["baseline_min_projected"],
        "clipped_fraction": diagnostics["clipped_fraction"],
        "clipped_mass": diagnostics["clipped_mass"],
    }


def run_experiment(
    reps=20,
    n_rows=30,
    n_cols=30,
    rank=2,
    noise=0.1,
    margin=0.05,
    tau=0.2,
    l=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        baseline_true = make_baseline(rng, n_rows, n_cols, rank, margin=margin)
        Z = make_block_treatment(baseline_true.shape)
        O = baseline_true + tau * Z + rng.normal(scale=noise, size=baseline_true.shape)

        solver = MCNNMPanelSolver(Z=Z)
        res = solver.solve_with_regularizer(
            O=O,
            l=l,
            max_iter=500,
            baseline_projection="clip_nonnegative",
        )
        baseline_rmse_raw = np.sqrt(np.mean((res.baseline - baseline_true) ** 2))
        baseline_rmse_projected = np.sqrt(np.mean((res.baseline_projected - baseline_true) ** 2))
        rows.append(
            {
                "rep": rep,
                "tau_true": tau,
                "tau_error_raw": float(res.tau - tau),
                "tau_error_projected": float(res.tau_projected - tau),
                "baseline_rmse_raw": float(baseline_rmse_raw),
                "baseline_rmse_projected": float(baseline_rmse_projected),
                "noise": noise,
                "margin": margin,
                **summarize_projection(res),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_experiment(reps=args.reps)
    summary = results[["tau_error_raw", "tau_error_projected", "clipped_fraction"]].agg(["mean", "std"])
    print(summary.to_string())
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
