import numpy as np
import pytest

from causaltensor.cauest.DebiasConvex import DCPanelSolver, DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MCNNMPanelSolver


def test_debiasconvex_nonnegative_fit_warns_and_disables_std():
    M0 = np.outer(np.linspace(1.0, 2.0, 6), np.linspace(0.5, 1.5, 6))
    Z = np.zeros_like(M0)
    Z[3:, 3:] = 1
    O = M0 + 0.25 * Z

    solver = DCPanelSolver(Z=Z, O=O)
    with pytest.warns(RuntimeWarning, match="experimental"):
        res = solver.fit(suggest_r=1, method="non-convex", method_non_neg="svd")

    assert res.std is None
    assert res.inference_valid is False
    assert res.non_negative_method == "svd"
    assert res.diagnostics["std_available"] is False
    assert res.diagnostics["inference_valid"] is False
    assert "projected_design_condition_number" in res.diagnostics


def test_debiasconvex_nonnegative_wrapper_returns_none_std():
    M0 = np.outer(np.linspace(1.0, 2.0, 6), np.linspace(0.5, 1.5, 6))
    Z = np.zeros_like(M0)
    Z[3:, 3:] = 1
    O = M0 + 0.25 * Z

    with pytest.warns(RuntimeWarning, match="experimental"):
        M, tau, std = DC_PR_with_suggested_rank(
            O,
            Z,
            suggest_r=1,
            method="non-convex",
            method_non_neg="svd",
        )

    assert M.shape == O.shape
    assert np.isfinite(tau)
    assert std is None


def test_mcnnm_baseline_projection_keeps_raw_outputs_and_adds_projected_outputs():
    O = -np.ones((5, 5))
    Z = np.zeros_like(O)
    Z[-1, -1] = 1

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_regularizer(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    assert res.baseline_projected is not None
    assert np.min(res.baseline) < 0
    assert np.min(res.baseline_projected) >= 0
    assert res.tau == res.tau_raw
    assert res.baseline is res.baseline_raw
    assert res.tau_projected <= res.tau
    assert res.projection_diagnostics["clipped_fraction"] > 0
    assert res.projection_diagnostics["baseline_min_projected"] == 0


def test_mcnnm_fit_dispatches_regularizer_with_baseline_projection():
    O = -np.ones((5, 5))
    Z = np.zeros_like(O)
    Z[-1, -1] = 1

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.fit(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    assert res.baseline_projection == "clip_nonnegative"
    assert np.min(res.baseline_projected) >= 0
    assert res.projection_diagnostics["clipped_fraction"] > 0


def test_mcnnm_fit_requires_observations():
    Z = np.zeros((5, 5))
    Z[-1, -1] = 1

    solver = MCNNMPanelSolver(Z=Z)
    with pytest.raises(ValueError, match="O must be provided"):
        solver.fit(l=0.1)


def test_mcnnm_baseline_projection_is_noop_when_raw_baseline_is_nonnegative():
    O = np.ones((5, 5))
    Z = np.zeros_like(O)
    Z[-1, -1] = 1

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_regularizer(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    np.testing.assert_allclose(res.baseline_projected, res.baseline)
    assert res.tau_projected == pytest.approx(res.tau)
    assert res.projection_diagnostics["clipped_fraction"] == 0
    assert res.projection_diagnostics["clipped_mass"] == 0


def test_mcnnm_rejects_unknown_baseline_projection():
    O = np.ones((5, 5))
    Z = np.zeros_like(O)
    Z[-1, -1] = 1

    solver = MCNNMPanelSolver(Z=Z)
    with pytest.raises(ValueError, match="baseline_projection"):
        solver.solve_with_regularizer(O=O, l=0.1, max_iter=1, baseline_projection="clip")
