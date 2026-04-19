"""Run a local scenario grid for non-negativity experiments.

Example:

    PYTHONPATH=src poetry run python experiments/nonnegative/run_grid.py --reps 3

The output CSVs are written to ``experiments/nonnegative/results`` by default.
The grid is intentionally modest so it can be run on a laptop while still
covering near-zero outcomes, larger positive margins, low/high noise, positive
and negative treatment effects, and two treatment patterns.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from causaltensor.cauest.DebiasConvex import DCPanelSolver
from causaltensor.cauest.MCNNM import MCNNMPanelSolver


def stable_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    if len(s) == 0 or s[0] == 0:
        return 0.0
    return float(np.sum(s**2) / (s[0]**2))


def low_rank(rng, shape, rank):
    U = rng.normal(size=(shape[0], rank))
    V = rng.normal(size=(shape[1], rank))
    return U @ V.T


def positive_low_rank(rng, shape, rank, margin):
    M = low_rank(rng, shape, rank)
    return M - np.min(M) + margin


def mcnnm_baseline(rng, shape, rank, margin, fixed_effects=True):
    M = low_rank(rng, shape, rank)
    if fixed_effects:
        a = rng.normal(scale=0.5, size=(shape[0], 1))
        b = rng.normal(scale=0.5, size=(1, shape[1]))
        M = M + a + b
    return M - np.min(M) + margin


def treatment_pattern(rng, shape, pattern):
    Z = np.zeros(shape)
    if pattern == "block":
        Z[shape[0] // 2:, shape[1] // 2:] = 1
    elif pattern == "row":
        Z[1::2, shape[1] // 2:] = 1
    elif pattern == "staggered":
        for row in range(shape[0] // 2, shape[0]):
            start = shape[1] // 3 + (row - shape[0] // 2) % max(1, shape[1] // 3)
            Z[row, start:] = 1
    elif pattern == "sparse":
        Z = (rng.random(shape) < 0.15).astype(float)
        while np.sum(Z) == 0:
            Z = (rng.random(shape) < 0.15).astype(float)
    else:
        raise ValueError(f"Unknown treatment pattern: {pattern}")
    return Z


def apply_outcome_mode(M, outcome_mode):
    if outcome_mode == "dense":
        return M
    if outcome_mode == "intermittent":
        threshold = np.quantile(M, 0.35)
        return np.maximum(M - threshold, 0)
    raise ValueError(f"Unknown outcome mode: {outcome_mode}")


def baseline_metrics(M):
    negative_part = np.minimum(M, 0)
    return {
        "baseline_min": float(np.min(M)),
        "negative_fraction": float(np.mean(M < 0)),
        "negative_mass": float(np.sum(-negative_part)),
        "rank": int(np.linalg.matrix_rank(M)),
        "stable_rank": stable_rank(M),
    }


def fit_debiasconvex(O, Z, rank, method_name, dc_mode):
    if method_name == "raw":
        solver = DCPanelSolver(Z=Z, O=O)
        res = solver.fit(suggest_r=rank, method=dc_mode)
        row = {
            "method": "raw",
            "dc_mode": dc_mode,
            "tau": float(res.tau),
            "std": float(res.std) if res.std is not None else np.nan,
            "inference_valid": res.inference_valid,
            **baseline_metrics(res.baseline),
        }
    elif method_name in {"nonnegative_svd", "nonnegative_nnmf"}:
        nonneg_method = "svd" if method_name == "nonnegative_svd" else "nnmf"
        solver = DCPanelSolver(Z=Z, O=O)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = solver.fit(suggest_r=rank, method=dc_mode, method_non_neg=nonneg_method)
        row = {
            "method": method_name,
            "dc_mode": dc_mode,
            "tau": float(res.tau),
            "std": np.nan,
            "inference_valid": res.inference_valid,
            **baseline_metrics(res.baseline),
        }
        row.update(res.diagnostics)
    elif method_name == "final_clip":
        solver = DCPanelSolver(Z=Z, O=O)
        res = solver.fit(suggest_r=rank, method=dc_mode)
        baseline_projected = np.maximum(res.baseline, 0)
        tau_projected = np.sum((O - baseline_projected) * Z) / np.sum(Z)
        row = {
            "method": "final_clip",
            "dc_mode": dc_mode,
            "tau": float(tau_projected),
            "std": np.nan,
            "inference_valid": False,
            "tau_raw": float(res.tau),
            "tau_shift": float(tau_projected - res.tau),
            **baseline_metrics(baseline_projected),
        }
    else:
        raise ValueError(f"Unknown DebiasConvex method: {method_name}")
    return row


def run_debiasconvex_grid(rng, reps, sizes, rank, margins, noises, taus, patterns, outcome_modes, dc_modes, dc_methods):
    rows = []
    for shape in sizes:
        for margin in margins:
            for noise in noises:
                for tau_true in taus:
                    for pattern in patterns:
                        for outcome_mode in outcome_modes:
                            for rep in range(reps):
                                M0 = positive_low_rank(rng, shape, rank, margin)
                                M0 = apply_outcome_mode(M0, outcome_mode)
                                Z = treatment_pattern(rng, shape, pattern)
                                O = M0 + tau_true * Z + rng.normal(scale=noise, size=shape)
                                for dc_mode in dc_modes:
                                    for method in dc_methods:
                                        row = fit_debiasconvex(O, Z, rank, method, dc_mode)
                                        row.update(
                                            {
                                                "solver": "DebiasConvex",
                                                "rep": rep,
                                                "n_rows": shape[0],
                                                "n_cols": shape[1],
                                                "margin": margin,
                                                "noise": noise,
                                                "tau_true": tau_true,
                                                "tau_error": row["tau"] - tau_true,
                                                "pattern": pattern,
                                                "outcome_mode": outcome_mode,
                                            }
                                        )
                                        rows.append(row)
    return pd.DataFrame(rows)


def run_mcnnm_grid(rng, reps, sizes, rank, margins, noises, taus, patterns, outcome_modes, l):
    rows = []
    for shape in sizes:
        for margin in margins:
            for noise in noises:
                for tau_true in taus:
                    for pattern in patterns:
                        for outcome_mode in outcome_modes:
                            for rep in range(reps):
                                baseline_true = mcnnm_baseline(rng, shape, rank, margin)
                                baseline_true = apply_outcome_mode(baseline_true, outcome_mode)
                                Z = treatment_pattern(rng, shape, pattern)
                                O = baseline_true + tau_true * Z + rng.normal(scale=noise, size=shape)
                                solver = MCNNMPanelSolver(Z=Z)
                                res = solver.fit(
                                    O=O,
                                    l=l,
                                    max_iter=500,
                                    baseline_projection="clip_nonnegative",
                                )
                                raw_rmse = np.sqrt(np.mean((res.baseline - baseline_true) ** 2))
                                projected_rmse = np.sqrt(np.mean((res.baseline_projected - baseline_true) ** 2))
                                rows.append(
                                    {
                                        "solver": "MCNNM",
                                        "method": "raw_vs_projected",
                                        "rep": rep,
                                        "n_rows": shape[0],
                                        "n_cols": shape[1],
                                        "margin": margin,
                                        "noise": noise,
                                        "tau_true": tau_true,
                                        "tau_raw": float(res.tau),
                                        "tau_projected": float(res.tau_projected),
                                        "tau_error_raw": float(res.tau - tau_true),
                                        "tau_error_projected": float(res.tau_projected - tau_true),
                                        "tau_shift": float(res.tau_projected - res.tau),
                                        "baseline_rmse_raw": float(raw_rmse),
                                        "baseline_rmse_projected": float(projected_rmse),
                                        "pattern": pattern,
                                        "outcome_mode": outcome_mode,
                                        **res.projection_diagnostics,
                                    }
                                )
    return pd.DataFrame(rows)


def parse_size(value):
    n_rows, n_cols = value.lower().split("x", 1)
    return int(n_rows), int(n_cols)


def write_outputs(df_dc, df_mc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dc_path = os.path.join(output_dir, "debiasconvex_grid.csv")
    mc_path = os.path.join(output_dir, "mcnnm_grid.csv")
    dc_summary_path = os.path.join(output_dir, "debiasconvex_summary.csv")
    mc_summary_path = os.path.join(output_dir, "mcnnm_summary.csv")

    df_dc.to_csv(dc_path, index=False)
    df_mc.to_csv(mc_path, index=False)

    dc_summary = (
        df_dc.groupby(["method", "dc_mode", "outcome_mode", "margin", "noise", "pattern"])["tau_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mc_summary = (
        df_mc.groupby(["outcome_mode", "margin", "noise", "pattern"])[
            ["tau_error_raw", "tau_error_projected", "tau_shift", "clipped_fraction"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    mc_summary.columns = [
        "_".join([str(part) for part in column if part])
        if isinstance(column, tuple)
        else column
        for column in mc_summary.columns
    ]
    dc_summary.to_csv(dc_summary_path, index=False)
    mc_summary.to_csv(mc_summary_path, index=False)
    return dc_summary, mc_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", default=["12x12", "20x20"])
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.05, 1.0])
    parser.add_argument("--noises", nargs="+", type=float, default=[0.05, 0.2])
    parser.add_argument("--taus", nargs="+", type=float, default=[-0.2, 0.2])
    parser.add_argument("--patterns", nargs="+", default=["block", "sparse"])
    parser.add_argument("--outcome-modes", nargs="+", default=["dense", "intermittent"])
    parser.add_argument("--dc-modes", nargs="+", default=["non-convex"])
    parser.add_argument(
        "--dc-methods",
        nargs="+",
        default=["raw", "nonnegative_svd", "nonnegative_nnmf", "final_clip"],
    )
    parser.add_argument("--mcnnm-l", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="experiments/nonnegative/results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = [parse_size(size) for size in args.sizes]
    df_dc = run_debiasconvex_grid(
        rng=rng,
        reps=args.reps,
        sizes=sizes,
        rank=args.rank,
        margins=args.margins,
        noises=args.noises,
        taus=args.taus,
        patterns=args.patterns,
        outcome_modes=args.outcome_modes,
        dc_modes=args.dc_modes,
        dc_methods=args.dc_methods,
    )
    df_mc = run_mcnnm_grid(
        rng=rng,
        reps=args.reps,
        sizes=sizes,
        rank=args.rank,
        margins=args.margins,
        noises=args.noises,
        taus=args.taus,
        patterns=args.patterns,
        outcome_modes=args.outcome_modes,
        l=args.mcnnm_l,
    )
    dc_summary, mc_summary = write_outputs(df_dc, df_mc, args.output_dir)
    print("DebiasConvex summary")
    print(dc_summary.head(20).to_string(index=False))
    print("\nMCNNM summary")
    print(mc_summary.head(20).to_string(index=False))
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
