.. :changelog:

Changelog
=========

Unreleased
----------
- Marked DebiasConvex non-negative modes as experimental point-estimation
  heuristics, disabled standard errors for those modes, and added diagnostics
  for inference validity, baseline negativity, rank, conditioning, and residuals.
- Added ``baseline_projection="clip_nonnegative"`` to ``MCNNMPanelSolver``
  solve methods and ``fit``. Raw baseline/tau remain unchanged while projected
  companion outputs and projection diagnostics are attached to the result.
- Added non-negativity tests, documentation, an executable tutorial notebook,
  and local experiment scripts/results for DebiasConvex and MCNNM scenarios.

0.1.12 (2025-03-12)
------------------
- Added CVXPY package for SDID method

0.1.11 (2025-03-12)
------------------
- Fix a bug in the DC method: suggest_r was ignored due to the priority of auto_rank and now it will be prioritized over auto_rank

0.1.10 (2025-02-08)
------------------
- Added Covariate support for SDID method

0.1.9 (2025-02-07)
------------------
- Added Panel Solver Interface
- Added more test cases
- Added covariate support for synthetic control 

0.1.8 (2023-11-05)
------------------
- Enhanced MC-NNM functionality with covariate integration and improved handling of missing data.

0.1.7 (2023-08-24)
------------------
- Introduced support for synthetic control methodology.

0.1.5 (2023-05-16)
------------------
- Expanded capabilities to address multiple-treatment problems using panel regression methods with debiasing features.
