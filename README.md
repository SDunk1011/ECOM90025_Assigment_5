# ECOM90025 — Assignment 5

Factors, for the ADA Kaggle case: does the common factor in X1–X50 improve the Assignment 4
quadratic LASSO, and should it be added to the regressors or replace them?

## What this repository contains

| File | Description |
|---|---|
| `assignment05.py` | The complete, standalone analysis. Downloads the data, runs PCA on the 50 regressors, cross-validates five specifications (plus two checks), refits the winner and writes the submission files. |
| `Assignment05.ipynb` | The same code, one section per cell. |
| `data/fig1_scree.png` | Scree plot of the 50 standardised regressors: one dominant component, the rest small and similar. |
| `data/fig2_fold_differences.png` | Each specification's fold MSE minus the Assignment 4 baseline, fold by fold. |
| `data/scree.csv`, `data/pc1_loadings.csv` | Explained variance ratios and the loadings of the factor. |
| `data/model_scores.csv`, `data/fold_mse.csv` | Cross-validated MSE, R², fold sd and model size of every specification. |
| `data/final_coefficients.csv` | Nonzero coefficients of the submitted model. |
| `data/submission.csv` | The Kaggle prediction file: the cross-validation winner. |
| `data/submission_control.csv` | Control file: the Assignment 4 quadratic LASSO, unchanged. |
| `kaggle_screenshot_circled.png` | Public leaderboard after submission (6.13947), group accounts circled. |

## Data access

The data is fetched at runtime from the Kaggle competition

```
ecom-90025-2026-sm-2-ada-assignment-and-practice
```

through the Kaggle API, exactly as in the Week 1 lecture notebook. Competition data cannot be
redistributed, so **to run the script you need your own Kaggle credentials in `~/.kaggle/` and
must have joined the competition** through its invitation link.

## Running it

```bash
pip install pandas numpy scikit-learn patsy matplotlib kaggle
python assignment05.py
```

Runtime is about 14 minutes on a laptop. Flags at the top of the script:

| Flag | Default | Effect |
|---|---|---|
| `SUBMIT_TO_KAGGLE` | `False` | Uploads `submission.csv`. |
| `SUBMIT_CONTROL` | `False` | Also uploads `submission_control.csv`, so the value of the factor is measured on the test sample. |

## Method

1. **Is there a factor? (Weeks 9–10).** PCA on the 50 standardised regressors and a scree plot.
   The first component explains 21.3% of the variance; each of the other 49 explains 1.2–2.0%,
   so there is no second common pattern (together they hold each variable's own 78.7%). So k = 1.
   Its score F correlates 0.62 with Y.
2. **Factor split (Algorithm 18, W9).** Each standardised observation is split as
   `x_i = F_i phi + e_i`, where `phi` are the PC1 loadings and `e_i` is the data Algorithm 18 is left
   with after removing the first component. (`phi_j` is also the OLS slope of X_j on F, so this is
   the usual common part plus own part.) PCA is fitted on the training rows only.
3. **Five specifications**, all scored on the same outer folds. Degree-2 terms are written as
   `patsy` formulas, `(X1 + ... + X50)**2 + I(X1**2) + ... - 1`, the way the lectures write
   interactions; a square is a variable interacted with itself.
   - **Baseline (control):** the Assignment 4 model, LASSO on the 1,325 degree-2 terms of X.
   - **(a) add:** the same terms plus F and F².
   - **(a) replace:** the degree-2 terms of (F, e_1, …, e_50) instead of X.
   - **(b) Algorithm 19:** PCA of the standardised 1,325 terms; LASSO on all PCs and the terms together.
   - **(c) Algorithm 21:** PLS on the 1,325 terms, the number of components (1–15) chosen by an
     inner 5-fold CV inside each training fold. This is the W10 rule for choosing k, applied to
     the training rows only so the fold that scores the model never helps choose it (W3).

   Two checks are scored but not tabled: the (a) add model with F × X_j interactions, and LASSO
   on the PCs alone (principal component regression).
4. **Penalty.** Every LASSO is standardised and refitted at each penalty on the Week 4 grid
   `np.logspace(0.5, -2.5, 60)`; AICc picks the penalty, exactly as in W4 cell 25 and
   Assignments 2 and 4. Only the regressors change between rows.
5. **Scoring.** Five outer folds, `KFold(5, shuffle=True, random_state=1)`, identical to
   Assignments 2–4. The rule, fixed before any result was seen: the lowest mean MSE is refitted
   on all 2,400 rows and submitted; the baseline is written as a control file.

## Results

Out-of-fold MSE on five folds. Var(Y) = 21.755, and a CV MSE is on the same scale as a Kaggle score.

| Specification | CV MSE | sd over folds | Folds better than baseline | Size |
|---|---|---|---|---|
| Assignment 4 quadratic LASSO (control) | 6.563 | 0.521 | — | 166 terms |
| **(a) add F, F² (submitted)** | **6.522** | 0.467 | 3 / 5 | 173 terms |
| (a) replace X by F and e | 7.081 | 0.581 | 0 / 5 | 218 terms |
| (b) LASSO on PCs + terms | 6.960 | 0.609 | 0 / 5 | 194 terms |
| (c) PLS on terms | 9.133 | 0.529 | 0 / 5 | 5–6 components |
| check: add F, F², F × X_j | 6.522 | 0.467 | 3 / 5 | 173 terms |
| check: LASSO on PCs only | 11.168 | 0.579 | 0 / 5 | 371 PCs |

F and the e_j span exactly the same space as X, so no rotation can add information. What it
changes is which terms LASSO finds cheap. The fitted models rely on a few individual variables
(X48, X34, …). Written in F and e, a square of one becomes three terms and a product four, and
the replacement model keeps 218 terms against 166; it loses in every fold. Adding the PCs of the
1,325 terms doubles the candidate set and costs 0.40.

Adding F and F² lowers CV MSE by 0.041 (0.6%), in three folds of five, almost all from fold 5
(folds 1–4 average 0.003). The paired fold differences have sd 0.093, so the gain is small and
unstable. The F × X_j model matches it in every fold, in MSE and in terms kept: no F × X_j term
is ever selected.

On the public leaderboard the submitted model scores **6.139**, against 6.127 for Assignment 4's
quadratic LASSO. The small CV gain does not carry over: the public score gives no evidence of
improvement over the control.

## Where each method comes from

| Method | Lecture |
|---|---|
| Kaggle API download and submission | W1 |
| Interactions as formulas (`a*b`, `(a + b)**2`) through `patsy` | W3 cell 33, W4 cell 97, W10 cell 74 |
| `KFold` cross-validation, out-of-sample MSE | W3 cells 83–85 |
| `StandardScaler`, `Lasso` refitted along the grid, AICc, `np.logspace` | W4 cells 24–25, 40 |
| PCA on standardised data, scree plot, choosing k by the elbow | W9 cells 28, 38; W10 cells 15–17 |
| Removing the first component from the data (Algorithm 18) | W9 cell 32 |
| Principal component (LASSO) regression, PCs and x together (Algorithm 19) | W10 cells 42, 79–86 |
| `PLSRegression`, number of components by cross-validation (Algorithm 21) | W10 cells 101–114 |

Two steps combine lecture methods rather than copy one cell: choosing the PLS components inside
each training fold (W10's CV over k, kept away from the scoring fold as W3 requires), and
treating `e` as a set of regressors (W9's Algorithm 18 residual).
