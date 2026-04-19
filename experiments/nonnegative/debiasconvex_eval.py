"""Small simulation harness for DebiasConvex non-negativity experiments.

Run from the repository root with:

    PYTHONPATH=src poetry run python experiments/nonnegative/debiasconvex_eval.py

The script is intentionally lightweight. It is meant for local exploration, not
for pytest or CI.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from causaltensor.cauest.DebiasConvex import DCPanelSolver


def make_low_rank(rng, n_rows, n_cols, rank, margin):
    U = rng.normal(size=(n_rows, rank))
    V = rng.normal(size=(n_cols, rank))
    M = U @ V.T
    return M - np.min(M) + margin


def make_block_treatment(shape):
    Z = np.zeros(shape)
    Z[shape[0] // 2:, shape[1] // 2:] = 1
    return Z


def summarize_baseline(M):
    negative_part = np.minimum(M, 0)
    s = np.linalg.svd(M, compute_uv=False)
    stable_rank = 0.0 if s[0] == 0 else float(np.sum(s**2) / (s[0]**2))
    return {
        "baseline_min": float(np.min(M)),
        "negative_fraction": float(np.mean(M < 0)),
        "negative_mass": float(np.sum(-negative_part)),
        "rank": int(np.linalg.matrix_rank(M)),
        "stable_rank": stable_rank,
    }


def fit_raw(O, Z, rank):
    solver = DCPanelSolver(Z=Z, O=O)
    res = solver.fit(suggest_r=rank, method="non-convex")
    return {
        "method": "raw",
        "tau": float(res.tau),
        "std": float(res.std) if res.std is not None else np.nan,
        "inference_valid": res.inference_valid,
        **summarize_baseline(res.baseline),
    }


def fit_nonnegative_svd(O, Z, rank):
    solver = DCPanelSolver(Z=Z, O=O)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solver.fit(suggest_r=rank, method="non-convex", method_non_neg="svd")
    return {
        "method": "nonnegative_svd",
        "tau": float(res.tau),
        "std": np.nan,
        "inference_valid": res.inference_valid,
        **summarize_baseline(res.baseline),
        **res.diagnostics,
    }


def fit_final_clip(O, Z, rank):
    solver = DCPanelSolver(Z=Z, O=O)
    res = solver.fit(suggest_r=rank, method="non-convex")
    baseline_projected = np.maximum(res.baseline, 0)
    tau_projected = np.sum((O - baseline_projected) * Z) / np.sum(Z)
    return {
        "method": "final_clip",
        "tau": float(tau_projected),
        "std": np.nan,
        "inference_valid": False,
        "tau_raw": float(res.tau),
        "tau_shift": float(tau_projected - res.tau),
        **summarize_baseline(baseline_projected),
    }


def run_experiment(reps=20, n_rows=30, n_cols=30, rank=2, noise=0.1, margin=0.05, tau=0.2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        M0 = make_low_rank(rng, n_rows, n_cols, rank, margin=margin)
        Z = make_block_treatment(M0.shape)
        O = M0 + tau * Z + rng.normal(scale=noise, size=M0.shape)

        for fit_fn in (fit_raw, fit_nonnegative_svd, fit_final_clip):
            row = fit_fn(O, Z, rank)
            row.update(
                {
                    "rep": rep,
                    "tau_true": tau,
                    "tau_error": row["tau"] - tau,
                    "noise": noise,
                    "margin": margin,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_experiment(reps=args.reps)
    summary = results.groupby("method")["tau_error"].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
