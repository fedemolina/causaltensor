Below is a technical memo. One small note first: the issue body and PR were accessible, but the GitHub interface here did not reliably expose every issue comment. I therefore treated your quoted latest experiment about hard clipping inside `DebiasConvex` and exploding standard errors as direct evidence in the analysis.

## 1. Executive summary

Issue #12 explicitly raised the lack of non-negativity constraints and the appearance of negative counterfactual predictions. PR #16 responded by adding `method_non_neg` to `DebiasConvex` and a `non_negative_decomposition` helper in `util.py`, with two families of heuristics: SVD-based clipping and sklearn NMF. In the current code, though, `DebiasConvex` still performs debiasing and standard-error calculations from the tangent space of the estimated low-rank matrix, while `MCNNM` defines the baseline as `res.fitted_value + M`. Those two facts make the two “non-negativity” problems fundamentally different. ([GitHub][1])

My bottom line is:

* **`DebiasConvex`: do not add in-loop hard clipping or advertise current non-negative modes as inference-valid.** The existing `method_non_neg` implementation is at best experimental for point estimation, and in convex/auto modes it is especially hard to defend mathematically. Standard errors should not be reported as if they were valid for those variants. ([GitHub][2])
* **`MCNNMPanelSolver`: if you add anything, add only a transparent post-estimation baseline projection.** The correct target is the full baseline, not the low-rank component `M`. A final projection such as `clip_nonnegative` is reasonable as a pragmatic post-processing option, provided the raw baseline and raw tau are preserved and exposed. ([GitHub][3])
* **Recommended order:** first ship MCNNM post-processing plus transparency/diagnostics; separately add warnings/docs around experimental non-negative `DebiasConvex`; leave true constrained `DebiasConvex` as a research branch, not a routine feature. ([GitHub][4])

## 2. Analysis of `DebiasConvex`

### 2.1 What the current solver is doing

`DebiasConvex` prepares an OLS design from the nonzero treatment cells, then alternates between estimating a baseline matrix `M` from `O - Z tau` and re-estimating `tau` from `O - M`. In convex mode the `M` step is soft-thresholded SVD; in non-convex mode it is hard rank truncation. The code path is literally “low-rank update, then OLS update.” ([GitHub][2])

The convex estimator follows the paper’s two-step logic: first compute a rough low-rank-plus-treatment estimate by minimizing squared error plus nuclear norm, then debias `tau` using the Gram matrix built from the treatment designs projected onto the orthogonal complement of the tangent space of the estimated low-rank matrix. The paper defines
[
(\hat M,\hat\tau)\in\arg\min_{M,\tau}\frac12\left|O-M-\sum_m \tau_m Z_m\right|*F^2+\lambda|M|**,
]
then
[
\tau^d=\hat\tau-D^{-1}\Delta,
]
with (D_{lm}=\langle P_{\hat T}^\perp(Z_l),P_{\hat T}^\perp(Z_m)\rangle) and (\Delta_l=\lambda\langle Z_l,\hat U\hat V^\top\rangle). The package code mirrors that structure in `debias()`. ([arXiv][5])

The standard errors are then computed in `panel_regression_CI()` from the same tangent-space geometry. The code forms a design matrix whose columns are `vec(remove_tangent_space_component(u, vh, Z_k))`, then computes
[
A=(X^\top X)^{-1}X^\top,\qquad
\widehat{\mathrm{Cov}}(\hat\tau)=A,\mathrm{diag}(\operatorname{vec}(E^2)),A^\top,
]
which is essentially a heteroskedastic sandwich on the projected treatment design. So the estimated singular vectors of `M` are not a side detail; they are the object the variance formula is built on. ([GitHub][2])

### 2.2 Why non-negativity is hard here

The key difficulty is that the current inference theory is **low-rank tangent-space theory**, not “low-rank plus arbitrary support projection” theory. The paper’s debiasing formula and its error decomposition both depend on (P_{\hat T}^\perp(Z)). In the single-treatment case the paper even writes the leading error term as proportional to (|P_T^\perp(Z)|_F^{-2}), so if that orthogonal component gets small, instability is exactly what you should expect. ([arXiv][5])

Hard clipping (M_{ij}\leftarrow \max(M_{ij},0)) is problematic for at least five separate reasons.

First, **it is not the proximal step of the constrained convex problem**
[
\min_{M\ge 0,\tau}\frac12|O-M-Z\tau|*F^2+\lambda|M|**.
]
Soft-thresholding followed by clipping is a composition of two projections/prox steps, but the prox of a sum is not generally the composition of the two prox operators. So even as point estimation, `SVD_soft` then clip is not “the convex non-negative estimator”; it is a heuristic. The same is true of hard-rank SVD then clip. The code confirms that `SVD_soft_non_negative` is literally `SVD_soft` followed by entrywise clipping, and `SVD_non_negative` is `SVD` followed by entrywise clipping. ([GitHub][6])

Second, **entrywise clipping usually destroys exact low-rank structure**. A rank-`r` matrix can become much higher rank after entrywise truncation at zero. That means the object whose tangent space is used in `debias()` and `panel_regression_CI()` is no longer the clean low-rank estimator envisioned by the paper. The code then re-estimates effective rank from the clipped matrix’s singular values, so the tangent space itself can change drastically. ([GitHub][2])

Third, **the continuation logic in convex mode becomes unreliable**. `DC_PR_with_suggested_rank()` decreases `l` until `np.linalg.matrix_rank(M) > suggest_r`. But if `M` is produced by soft-thresholding and then clipping, `matrix_rank(M)` is no longer tracking the nuclear-norm path in a meaningful way. The same issue appears again at the final convex step, where the code uses `SVD_non_negative(M_debias, suggest_r)`; after clipping, the returned matrix need not actually have rank `suggest_r`. ([GitHub][2])

Fourth, **the projected treatment design can collapse**. In the CI code, if clipping enlarges the row/column space of `M`, then (P_{\hat T}^\perp(Z)) shrinks, making `X.T @ X` ill-conditioned. Since the code uses a plain inverse, not a stabilized inverse, this shows up as huge standard errors or numerical instability. In `debias()` the analogous matrix `D` is only pseudo-inverted, which avoids crashes but not instability. ([GitHub][2])

Fifth, **clipping is nonsmooth at zero**. When many entries are near zero, tiny perturbations in `M` can change the active set of clipped cells. That is exactly the kind of nonsmoothness that breaks first-order linearizations based on a smooth low-rank manifold.

As a local sanity check, I ran toy examples on random (20\times 20) rank-2 matrices. Entrywise clipping typically raised numerical rank from 2 to about 16, and for a block treatment pattern it reduced (|P_T^\perp(Z)|) to about 16% of its original size. With three sparse treatment patterns, the median condition number of the projected design roughly doubled and often became singular. That is exactly the direction that produces the huge standard errors you observed.

### 2.3 Assessment of the current `method_non_neg` implementation

PR #16 added `method_non_neg` and routed it through essentially every `DebiasConvex` path, including convex, non-convex, and auto modes. It also added `non_negative_decomposition()` to `util.py`, with `method='svd'` and `method='nnmf'`. The PR conversation itself already showed a warning sign: on the promotions-style data, convex + SVD(non-negative) gave a strongly negative tau, while non-convex + SVD(non-negative) looked much closer to unconstrained non-convex. ([GitHub][4])

My classification is:

* **`method_non_neg="svd"` in `DebiasConvex`: experimental heuristic only.** It does not solve the constrained convex problem, it does not preserve exact rank in the non-convex mode, and it changes the geometry used for debiasing and CI. In convex/auto modes it is potentially misleading if presented as a theoretically grounded estimator. ([GitHub][4])
* **`method_non_neg="nnmf"` in convex mode: not defensible as “non-negative DebiasConvex.”** The helper first clips the residual matrix to `max(M,0)`, then calls sklearn NMF. When `l` is passed without `r`, it sets `n_components = M.shape[1]`, i.e. a full-component factorization, and uses the NMF objective with L1/L2 penalties on `W` and `H`, not a nuclear norm on `M`. That is simply a different estimator. ([GitHub][6])
* **`method_non_neg="nnmf"` in non-convex fixed-rank mode: mathematically coherent only as a different model family.** A rank-`r` nonnegative factor model can make sense, but it is no longer the DC-PR estimator from the paper, and the current DC-PR debiasing and standard-error code should not be reused for it. ([GitHub][6])

So the honest label is: **experimental point-estimation heuristic, not inference-valid DC-PR**.

### 2.4 What is coherent, pragmatic, or not worth doing

For `DebiasConvex`, I would separate approaches into three buckets.

**Mathematically coherent but larger project**

A true constrained convex estimator
[
\min_{M\ge 0,\tau}\frac12|O-M-Z\tau|*F^2+\lambda|M|**
]
is coherent as a point estimator. It would need a new solver for the `M`-subproblem, not `soft-threshold then clip`. ADMM, PDHG, or Dykstra/splitting methods are plausible. But I would not attach the current debiasing or CI formulas to it without new theory.

**Pragmatic and transparent**

A post-estimation diagnostic/projection is fine, but only if it is labeled as post-processing. For example:

* keep the raw estimator as official `M`, `tau`, `std`;
* optionally compute `baseline_projected = np.maximum(M, 0)`;
* optionally compute `tau_projected` from that projected baseline;
* report **no** standard error for `tau_projected`.

This can be useful for users who want support-respecting counterfactuals as a derived object, but it should not overwrite the inference-bearing estimator.

**Should not be implemented**

* hard clipping inside the ALS loop while keeping current `debias()`/`panel_regression_CI()`;
* presenting current `method_non_neg` convex output as “the non-negative DC-PR estimator”;
* returning standard errors for any non-negative variant that changes the matrix used to define the tangent space.

### 2.5 Should non-negative `DebiasConvex` support standard errors?

My answer is **no**, not in the current architecture.

If the non-negativity step changes the matrix whose singular vectors define (\hat T), then the current CI code is no longer aligned with the estimator. The only safe exception is a purely diagnostic projection that leaves the raw `tau`/`std` untouched and stores projected outputs separately. In that case the raw standard errors remain attached to the raw estimator only.

A conservative implementation policy would be:

* if `method_non_neg is None`: current behavior;
* if `method_non_neg is not None`: either

  1. set `std=None` and warn, or
  2. for backward compatibility, keep computing `std` for one release but add a very explicit warning and a flag like `res.inference_valid = False`.

I prefer option 1, but option 2 is a gentler migration path.

### 2.6 Diagnostics worth reporting

If any non-negativity-related option is used in `DebiasConvex`, I would report at least:

* `baseline_min_raw`, `baseline_min_projected`;
* fraction of negative entries in the raw baseline;
* total clipped mass ( \sum |\min(M_{ij},0)| ) and max clipping magnitude;
* `tau_raw`, `tau_projected`, and `tau_shift`;
* numerical rank and stable rank before/after projection;
* smallest eigenvalue / condition number of `X.T @ X` in `panel_regression_CI`;
* smallest eigenvalue / condition number of `D` in `debias`;
* residual Frobenius norm on untreated cells before/after projection;
* number of iterations and convergence status.

Those diagnostics directly tell the user whether the projection was negligible, moderate, or severe.

## 3. Analysis of `MCNNMPanelSolver`

### 3.1 Why `M >= 0` is the wrong target

`MCNNMPanelSolver` does something structurally different from `DebiasConvex`. Inside `solve_with_regularizer()` it first fits fixed effects and optional covariates via `FixedEffectPanelSolver`, whose `fitted_value` is (a + b^\top) or (a + b^\top + X\beta). It then estimates the low-rank component `M` by soft-imputing the residual around that fitted value, and finally sets
`baseline = res.fitted_value + M`. That is the object used to compute tau. ([GitHub][3])

That matches the MC-NNM paper’s formulation, where the estimator is a low-rank component plus explicit fixed effects, and only the low-rank part is regularized. The paper is very clear that the full untreated-outcome estimator is ( \hat L + \hat\Gamma 1^\top + 1 \hat\Delta^\top ), not just ( \hat L ). ([arXiv][7])

So if outcomes must be non-negative, the natural target is **the final baseline**, not `M`. A negative `M` can coexist with a positive baseline because the fixed effects are positive enough; a non-negative `M` can still yield a negative baseline if the fixed effects are negative enough. Therefore `M >= 0` is a decomposition-dependent restriction, not a support restriction on the counterfactual outcome itself. ([GitHub][3])

### 3.2 Is final clipping reasonable?

This version is reasonable as a **post-processing correction**:

```python
baseline_raw = res.fitted_value + M
baseline = np.maximum(baseline_raw, 0)
tau = np.sum((O - baseline) * Z) / np.sum(Z)
```

It is not the constrained MC-NNM estimator, but it is a transparent support correction on the actual target object. ([GitHub][3])

Because MCNNM currently does not provide standard errors, the downside is smaller than in `DebiasConvex`. You are not breaking an existing tangent-space CI formula. You are changing only the returned counterfactual baseline and the derived treatment effect.

There is one very important transparency point, though: after clipping, the returned baseline is **no longer equal** to `row_fixed_effects + column_fixed_effects + X beta + M`. That means the raw decomposition and the projected baseline must both be stored.

### 3.3 How to classify that projection

I would classify it as:

* **not** a constrained estimator;
* **yes** a post-processing correction;
* **yes** a diagnostic option;
* **possibly** a useful default in some nonnegative-outcome applications, but only if raw results are preserved alongside it.

I would not call it `non_negative_baseline=True`, because that sounds like the optimization itself imposed the constraint. I would call it something like `baseline_projection="clip_nonnegative"` or `postprocess_baseline="clip_nonnegative"`.

### 3.4 Should the projection overwrite `baseline`?

My recommendation is **no**. Keep the raw objects canonical.

Concretely:

* `res.baseline` stays raw;
* `res.tau` stays raw;
* if post-processing is requested, add
  `res.baseline_projected`, `res.tau_projected`, and `res.projection_diagnostics`.

That is the least misleading design because:

1. it preserves backward compatibility;
2. it preserves the decomposition identity between `M`, fixed effects, covariates, and `baseline`;
3. it makes it obvious that the projection is an extra layer, not the fitted model itself.

If maintainers strongly prefer changing the main return values, then `baseline_raw` and `tau_raw` become mandatory. But I still think “raw remains canonical, projected is additional” is the cleanest API.

### 3.5 Can the non-negativity constraint be imposed during iteration?

In principle, yes. The joint problem
[
\min_{M,a,b,\beta}\frac12|\Omega\odot(O-M-a-b-X\beta)|*F^2+\lambda|M|**
\quad\text{s.t.}\quad M+a+b+X\beta\ge 0
]
is convex in the variables used by MC-NNM, because the loss is convex, the nuclear norm is convex, and the baseline non-negativity constraint is affine. But the current algorithm is not solving that problem. It alternates a fixed-effects fit with a soft-impute step on the residual, and nothing in that routine corresponds to a projection onto the affine cone (M+\text{FE}+X\beta\ge 0). ([GitHub][3])

So a naive in-loop rule like “clip `M + fitted_value` to zero and continue” would be another heuristic, and it would also break the low-rank decomposition because the projected baseline generally cannot be written back as the same fixed effects plus the same low-rank `M`.

My recommendation is: **do not do this in the current solver**. If you want a true constrained estimator later, prototype it separately with CVXPY or a dedicated ADMM/PDHG solver.

### 3.6 Should NNMF or SVD clipping be avoided in MCNNM?

Yes.

The low-rank component in MCNNM is a residual correction around fixed effects and covariates. There is no reason it should itself be non-negative. Applying NNMF or entrywise clipping to `M` would force a decomposition-specific sign restriction that is unrelated to the support of the final untreated outcome.

So for MCNNM:

* avoid `M >= 0`;
* avoid NMF on the residual low-rank component as a “non-negativity fix”;
* target only the final baseline if you do anything at all.

## 4. Recommendation: what to implement first

Implement **only two things** first.

**First:** add an MCNNM post-processing option that computes non-negative projected companion outputs:

* `baseline_projection="clip_nonnegative"`;
* keep raw `baseline`/`tau`;
* add `baseline_projected`, `tau_projected`, `projection_diagnostics`.

**Second:** add warnings and documentation around `DebiasConvex` non-negative modes:

* mark them experimental;
* do not claim inference validity;
* ideally suppress or deprecate standard errors when `method_non_neg` is used.

I would **not** implement in-loop non-negativity for either solver as the next PR.

## 5. What not to implement

I would explicitly avoid the following.

* `DebiasConvex`: `M = max(M, 0)` inside ALS while continuing to run the current `debias()` and `panel_regression_CI()`.
* `DebiasConvex`: presenting convex `method_non_neg="svd"` as a proper constrained nuclear-norm estimator.
* `DebiasConvex`: presenting convex `method_non_neg="nnmf"` as a variant of the paper’s estimator.
* `DebiasConvex`: any non-negative variant with reported standard errors unless a new inference theory is derived.
* `MCNNM`: `M >= 0` as the target restriction.
* `MCNNM`: NNMF or SVD-clipping on the low-rank residual component.
* `MCNNM`: in-iteration clipping of `M + fitted_value` inside the current alternating solver.

## 6. Empirical evaluation design

Before implementation, I would run a dedicated evaluation branch.

### 6.1 Simulation design

Use two DGP families.

For `DebiasConvex`, generate
[
O = M^\star + \sum_k \tau_k^\star Z_k + E,
]
with (M^\star) exactly low rank or approximately low rank. Control non-negativity by adding a global offset so that (M^\star) is:

* strongly positive;
* near zero;
* zero-inflated / sparse positive.

For `MCNNM`, generate
[
B^\star = a^\star 1^\top + 1 {b^\star}^\top + X\beta^\star + L^\star,\qquad
O = B^\star + \tau^\star Z + E,
]
and choose the offset so that the final baseline (B^\star) has:

* large positive margin;
* many entries close to zero;
* structural zeros / intermittent demand.

### 6.2 Scenario grid

Use a grid over:

* panel sizes: `20x20`, `50x50`, `100x100`;
* noise: low vs high;
* structure: exact low rank vs approximate low rank;
* outcomes: dense positive vs sparse/intermittent;
* treatment effect: positive and negative;
* treatment patterns: contiguous block, row-specific, staggered adoption, sparse treated cells;
* `DebiasConvex` modes: convex, non-convex, auto;
* `MCNNM` structure: FE absent (`a=b=0`) vs FE present; covariates absent vs present.

### 6.3 Methods to compare

For `DebiasConvex`:

* unconstrained baseline;
* current `method_non_neg="svd"` as experimental baseline;
* current `method_non_neg="nnmf"` as experimental baseline;
* optional final-only clipping as post-processing;
* if prototyped, true constrained convex point estimator with **no** SE.

For `MCNNM`:

* raw baseline;
* post-processed `baseline_projection="clip_nonnegative"`.

### 6.4 Metrics

For both solvers:

* bias and RMSE of `tau`;
* baseline RMSE;
* fraction of negative baseline entries;
* total negative mass and maximum negative magnitude;
* fraction clipped and clipping magnitude;
* `tau_projected - tau_raw`;
* convergence failures / iterations;
* numerical rank and stable rank.

For `DebiasConvex` specifically:

* condition number and minimum eigenvalue of `D`;
* condition number and minimum eigenvalue of `X.T @ X` in CI;
* empirical SD of tau versus mean estimated SE;
* 90% / 95% CI coverage;
* width of intervals;
* rank chosen along the convex path.

For `MCNNM` specifically:

* untreated-cell validation MSE of the raw baseline;
* treated-cell counterfactual error if simulated truth is available;
* decomposition inconsistency warning rate if projected outputs are used.

### 6.5 Unit tests vs notebooks/scripts

The repository’s current `tests/` directory is mostly notebooks (`MCNNM_test.ipynb`, `test_DC_PR.ipynb`, `semi_synthetic.ipynb`, etc.), with only limited pytest coverage. Large Monte Carlo studies should therefore live in a new experiment script or notebook, not in pytest. Pytest should only cover deterministic invariants and API contracts. ([GitHub][8])

So I would split evaluation into:

* **unit tests**

  * projection never returns negative values;
  * identity when the raw baseline is already non-negative;
  * raw outputs are preserved;
  * diagnostics are populated;
  * `DebiasConvex` non-negative modes warn and mark inference invalid.

* **notebooks / scripts**

  * Monte Carlo calibration;
  * semi-synthetic masking experiments;
  * comparisons across treatment patterns and noise levels.

## 7. Proposed implementation plan

### 7.1 Recommended branches / PRs

I would not do this as one large PR.

**Evaluation branch first**

* `research/nonnegative-baseline-eval`

This branch is for experiments only:

* `experiments/nonnegative/debiasconvex_eval.py`
* `experiments/nonnegative/mcnnm_eval.py`
* maybe `tutorials/nonnegative_baseline_eval.ipynb`

No merge pressure; use it to decide whether MCNNM projection is actually helpful.

**PR A: DebiasConvex safety and documentation**

* branch: `warn/debiasconvex-nonneg-experimental`

**PR B: MCNNM baseline projection**

* branch: `feat/mcnnm-baseline-projection`

If only one feature PR is desired, make it PR B.

### 7.2 PR A file-by-file plan

`src/causaltensor/cauest/DebiasConvex.py`

* emit a warning whenever `method_non_neg` is used;
* add `res.inference_valid = False` when `method_non_neg` is not `None`;
* preferred behavior: set `std=None` for non-negative modes;
* fallback behavior for backward compatibility: keep `std` for one release but warn strongly.

`src/causaltensor/cauest/result.py` or `DCResult`

* add optional fields such as `inference_valid`, `non_negative_method`, `diagnostics`.

`tests/test_debiasconvex_nonneg.py`

* verify warnings;
* verify diagnostics are produced;
* verify `std` handling policy.

`docs` / tutorials

* explicitly label `method_non_neg` as experimental point-estimation only.

Small but important separate note: I would keep **one more housekeeping fix in its own PR**, not in the non-negativity PR. In `als()`, on convergence the code currently returns `tau`, not `tau_new`, which can confound experiments. That is orthogonal enough that I would patch it separately. ([GitHub][2])

### 7.3 PR B file-by-file plan

`src/causaltensor/cauest/MCNNM.py`

* add `baseline_projection=None` to the public solve methods;
* after the raw fit converges, compute:

  * `baseline_raw = res.fitted_value + M`
  * `tau_raw = ...`
* if `baseline_projection == "clip_nonnegative"`:

  * compute `baseline_projected = np.maximum(baseline_raw, 0)`
  * compute `tau_projected`
  * compute projection diagnostics
* keep `res.baseline` and `res.tau` raw;
* attach projected companion fields.

`src/causaltensor/cauest/result.py` or `MCNNMResult`

* add optional fields:

  * `baseline_projected`
  * `tau_projected`
  * `projection_diagnostics`
  * maybe `baseline_projection`

`tests/test_mcnnm_baseline_projection.py`

* projected baseline is non-negative;
* raw baseline/tau remain present;
* no-op when raw baseline already non-negative;
* `tau_projected <= tau_raw` for binary treatment and clip-to-zero projection.

`tutorials/MCNNM_test.ipynb` or a new notebook

* show raw vs projected baseline;
* show diagnostics and interpretation.

### 7.4 Backward compatibility

* Keep all existing raw outputs unchanged by default.
* Do not change the old wrapper functions to return extra objects; leave them raw and document that projected outputs are available via the solver result object.
* Any change to `DebiasConvex` standard errors under `method_non_neg` should be called out clearly in release notes.

### 7.5 Warnings / documentation language

For `DebiasConvex`:

> Non-negative modes are experimental. They modify the low-rank estimate used by the debiasing and variance formulas, so standard errors and confidence intervals are not theoretically validated.

For `MCNNM`:

> `baseline_projection="clip_nonnegative"` is a post-estimation support correction applied to the final baseline. It is not the constrained MC-NNM estimator. Raw baseline and tau are preserved and returned alongside the projected outputs.

## 8. Open theoretical questions

1. For constrained `DebiasConvex`, what is the correct debiasing formula when the estimator lies on the boundary (M_{ij}=0) for many entries? The current tangent-space argument is not enough; the relevant object is a combination of a low-rank tangent space and a non-negativity normal cone.

2. Under what conditions is the non-negativity constraint asymptotically inactive? If the true baseline is uniformly bounded away from zero, then the constraint should eventually stop mattering, which means the practical value of the feature is mainly finite-sample and near-boundary.

3. Is any resampling method reliable for post-processed clipped estimators here? A naive bootstrap may inherit the same nonsmoothness and tuning-parameter instability.

4. For MCNNM with FE/covariates, can a scalable ADMM or PDHG solver for `baseline >= 0` outperform simple post-processing enough to justify the extra complexity?

## 9. Suggested wording for a GitHub issue / PR description

Here is wording I would actually use:

> I reviewed issue #12, PR #16, and the current code paths for `DebiasConvex` and `MCNNMPanelSolver`. My conclusion is that the two non-negativity problems should be treated differently.
>
> For `DebiasConvex`, the current debiasing and standard-error formulas depend on the tangent space of the estimated low-rank matrix. Entrywise clipping or NNMF changes that geometry, so the current non-negative modes should be treated as experimental point-estimation heuristics, not inference-valid estimators. I do not recommend adding in-loop clipping or expanding non-negative `DebiasConvex` without a separate constrained-estimation and inference derivation.
>
> For `MCNNMPanelSolver`, the correct target is the full baseline `fitted_value + M`, not `M` itself. A pragmatic and transparent first feature is a post-estimation option such as `baseline_projection="clip_nonnegative"` that computes projected companion outputs (`baseline_projected`, `tau_projected`) while preserving the raw baseline and raw tau.
>
> I propose two separate PRs:
>
> 1. mark `DebiasConvex` non-negative modes as experimental and add warnings/diagnostics;
> 2. add MCNNM baseline post-processing with raw/projected outputs and diagnostics.

If you want, the next step I’d recommend is drafting the exact API signatures and test cases for PR B first, because that is the one feature that looks both honest and useful.

[1]: https://github.com/TianyiPeng/causaltensor/issues/12 "https://github.com/TianyiPeng/causaltensor/issues/12"
[2]: https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/cauest/DebiasConvex.py "https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/cauest/DebiasConvex.py"
[3]: https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/cauest/MCNNM.py "https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/cauest/MCNNM.py"
[4]: https://github.com/TianyiPeng/causaltensor/pull/16/files "https://github.com/TianyiPeng/causaltensor/pull/16/files"
[5]: https://arxiv.org/pdf/2106.02780 "https://arxiv.org/pdf/2106.02780"
[6]: https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/matlib/util.py "https://github.com/TianyiPeng/causaltensor/raw/refs/heads/main/src/causaltensor/matlib/util.py"
[7]: https://arxiv.org/pdf/1710.10251 "https://arxiv.org/pdf/1710.10251"
[8]: https://github.com/TianyiPeng/causaltensor/tree/main/tests "https://github.com/TianyiPeng/causaltensor/tree/main/tests"
