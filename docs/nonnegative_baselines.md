# Non-Negative Baselines

This note summarizes the current non-negativity support and the limitations
discussed in https://github.com/TianyiPeng/causaltensor/issues/12.

## Local Branch Changelog

These changes are currently local to the `feature/nonnegative-baselines` branch
and have not been proposed for an upstream package release yet.

- Marked `DebiasConvex` non-negative modes as experimental point-estimation
  heuristics, disabled standard errors for those modes, and added diagnostics
  for inference validity, baseline negativity, rank, conditioning, and residuals.
- Added `baseline_projection="clip_nonnegative"` to `MCNNMPanelSolver` solve
  methods and `fit`. Raw baseline/tau remain unchanged while projected
  companion outputs and projection diagnostics are attached to the result.
- Added non-negativity tests, this documentation page, an executable tutorial
  notebook, and local experiment scripts/results for DebiasConvex and MCNNM
  scenarios.

## DebiasConvex

`DebiasConvex` supports `method_non_neg` as an experimental point-estimation
heuristic. These modes modify the low-rank estimate used by the debiasing and
standard-error formulas, so standard errors and confidence intervals should not
be treated as theoretically validated.

When `method_non_neg` is used, the solver emits a warning, marks the result as
`inference_valid=False`, returns `std=None`, and attaches diagnostics to
`res.diagnostics`.

```python
from causaltensor.cauest.DebiasConvex import DCPanelSolver

solver = DCPanelSolver(Z=Z, O=O)
res = solver.fit(suggest_r=2, method="non-convex", method_non_neg="svd")

print(res.tau)
print(res.std)                 # None
print(res.inference_valid)     # False
print(res.diagnostics)
```

Hard clipping of the low-rank matrix inside the iteration should not be treated
as a safe default. In experiments discussed in issue #12, this could make the
estimated standard errors extremely large.

## MCNNM

For `MCNNMPanelSolver`, the non-negativity target is the full baseline:

```python
baseline = fitted_value + M
```

The low-rank component `M` is a residual correction around fixed effects and
covariates, so `M >= 0` is not generally the right target. The supported option
is a transparent post-estimation projection:

```python
from causaltensor.cauest.MCNNM import MCNNMPanelSolver

solver = MCNNMPanelSolver(Z=Z)
res = solver.solve_with_regularizer(
    O=O,
    l=1.0,
    baseline_projection="clip_nonnegative",
)

print(res.baseline)             # raw fitted baseline
print(res.tau)                  # raw tau
print(res.baseline_projected)   # np.maximum(res.baseline, 0)
print(res.tau_projected)        # tau recalculated from projected baseline
print(res.projection_diagnostics)
```

The projected outputs are companion outputs. They do not replace the raw
baseline, raw tau, or the fitted low-rank/fixed-effect decomposition.

## Local Evaluation

The local scenario grid can be run from the repository root:

```bash
PYTHONPATH=src poetry run python experiments/nonnegative/run_grid.py --reps 2
```

The script writes detailed and summarized CSVs to
`experiments/nonnegative/results/`.

The current tutorial notebook is:

```text
tutorials/nonnegative_baselines.ipynb
```
