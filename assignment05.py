"""
ECOM90025 Advanced Data Analysis - Assignment 5
Factors: does the common factor in X1-X50 improve the Assignment 4 quadratic LASSO?

Competition: ecom-90025-2026-sm-2-ada-assignment-and-practice

This script:
  1. Downloads train/test data from the Kaggle competition (same API route as Week 1).
  2. Runs PCA on the 50 standardised regressors and draws the scree plot (Week 9-10).
     One component stands out; the other 49 explain about 2% each, like noise.
  3. Scores five specifications on the same five outer folds as Assignments 2-4:
       baseline    A4 model: LASSO on the 1,325 degree-2 terms of X (control)
       add_F       (a) the same terms plus the factor F = PC1 and F^2
       replace_Fe  (a) degree-2 terms of F and the idiosyncratic parts e = X - F phi',
                   the residual of Algorithm 18 after the first component is removed
       pca_terms   (b) Algorithm 19: PCA of the 1,325 terms, LASSO on PCs and terms together
       pls         (c) Algorithm 21: PLS on the 1,325 terms, components chosen by inner CV
     Two checks are scored but not tabled: F x X_j interactions (add_FX) and
     LASSO on the PCs alone (pcr_only).
  4. Every LASSO penalty is chosen by AICc on np.logspace(0.5, -2.5, 60), as in A2 and A4.
     All scaling, PCA and PLS are fitted on the training part of each fold only.
  5. Refits the lowest-CV-MSE specification on all 2,400 rows and writes the submission,
     plus the A4 baseline as a control file.

Every method is from the lectures: interactions written as patsy formulas and KFold
out-of-sample MSE (W3, W4, W10), StandardScaler, Lasso and AICc (W4), PCA, Algorithm 18,
scree plot and principal component (LASSO) regression (W9-10), PLSRegression with CV over
the number of components (W10).
"""

import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write figures to file rather than opening windows
import matplotlib.pyplot as plt
import patsy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore


# ## Configuration

COMPETITION_ID = "ecom-90025-2026-sm-2-ada-assignment-and-practice"
DATA_DIR = "./data"
FEATURE_COLS = [f"X{i}" for i in range(1, 51)]

N_FOLDS = 5                                # outer folds, identical to Assignments 2-4
FOLD_SEED = 1
ALPHA_GRID = np.logspace(0.5, -2.5, 60)    # Week 4 grid, searched from large to small
MAX_ITER = 20_000
PLS_MAX_COMPONENTS = 15                    # PLS components searched by inner CV
PLS_INNER_SEED = 0

MAIN_SPECS = ["baseline", "add_F", "replace_Fe", "pca_terms", "pls"]
CHECK_SPECS = ["add_FX", "pcr_only"]
LABELS = {
    "baseline": "A4 quadratic LASSO (control)",
    "add_F": "(a) add F, F²",
    "replace_Fe": "(a) replace X by F and e",
    "pca_terms": "(b) LASSO on PCs + terms",
    "pls": "(c) PLS on terms",
    "add_FX": "check: add F, F², F × X",
    "pcr_only": "check: LASSO on PCs only",
}

SUBMIT_TO_KAGGLE = False   # uploads submission.csv; flip only when told to
SUBMIT_CONTROL = False     # also uploads the A4 baseline, to measure the factors on the test set

INK, MUTED, GRID, SERIES = "#1f1f1e", "#6b6a64", "#dddcd5", "#2a78d6"

os.makedirs(DATA_DIR, exist_ok=True)


# ## 1. Kaggle API: authenticate & download (Week 1)

def download_competition_data():
    """Authenticate with the key in ~/.kaggle/ and extract the competition files."""
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition=COMPETITION_ID, path=DATA_DIR)
    zip_filepath = os.path.join(DATA_DIR, f"{COMPETITION_ID}.zip")
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_filepath)
    return api


api = download_competition_data()
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
X = train_df[FEATURE_COLS].to_numpy()
y = train_df["Y"].to_numpy()
X_test = test_df[FEATURE_COLS].to_numpy()
print(f"train {train_df.shape}, test {test_df.shape}, Var(Y) = {y.var():.3f}")


# ## 2. Is there a factor? PCA and scree plot (Weeks 9-10)

X_scaled = StandardScaler().fit_transform(X)
pca_x = PCA().fit(X_scaled)
explained = pca_x.explained_variance_ratio_
pc1 = pca_x.transform(X_scaled)[:, 0]
pairwise = np.corrcoef(X.T)[np.triu_indices(50, 1)]

pd.DataFrame({"component": np.arange(1, 51), "explained_variance_ratio": explained}).to_csv(
    os.path.join(DATA_DIR, "scree.csv"), index=False)
pd.DataFrame({"variable": FEATURE_COLS, "pc1_loading": pca_x.components_[0]}).to_csv(
    os.path.join(DATA_DIR, "pc1_loadings.csv"), index=False)
print(f"PC1 explains {explained[0]:.1%}; PCs 2-50 explain {explained[1:].min():.1%}-"
      f"{explained[1:].max():.1%} each")
print(f"mean pairwise corr among X = {pairwise.mean():.3f}; corr(PC1, Y) = "
      f"{abs(np.corrcoef(pc1, y)[0, 1]):.3f}; PC1 loadings share one sign: "
      f"{np.all(np.sign(pca_x.components_[0]) == np.sign(pca_x.components_[0][0]))}")

shown = 20
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=300)
ax.bar(np.arange(1, shown + 1), explained[:shown], width=0.7, color=SERIES)
ax.axhline(1 / 50, color=MUTED, linestyle="--", linewidth=1)
ax.text(shown + 0.4, 1 / 50 + 0.004, "equal share, 1/50", color=MUTED, fontsize=8, ha="right")
ax.text(1.5, explained[0], f"PC1: {explained[0]:.1%}", color=INK, fontsize=9, va="center")
ax.set_xlabel("Principal component", color=INK)
ax.set_ylabel("Explained variance ratio", color=INK)
ax.set_xticks([1, 5, 10, 15, 20])
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(DATA_DIR, "fig1_scree.png"))
plt.close(fig)


# ## 3. Building blocks

def quadratic(names, extra=""):
    """Formula for levels, pairwise interactions and squares: 1,325 terms for 50 inputs.

    (a + b + ...)**2 gives the levels and every pairwise interaction (W3, W4); a square is
    a variable interacted with itself, written I(x**2). -1 drops the intercept, which the
    LASSO fits separately."""
    terms = "(" + " + ".join(names) + ")**2 + " + " + ".join(f"I({v}**2)" for v in names)
    return terms + extra + " - 1"


X_NAMES = FEATURE_COLS
E_NAMES = [f"e{i}" for i in range(1, 51)]
FORMULAS = {
    "baseline": quadratic(X_NAMES),
    "add_F": quadratic(X_NAMES, " + F + I(F**2)"),
    "add_FX": quadratic(X_NAMES, " + F + I(F**2) + F:(" + " + ".join(X_NAMES) + ")"),
    "replace_Fe": quadratic(["F"] + E_NAMES),
}


def factor_frames(X_train, X_test):
    """Standardised X, the factor F = PC1 score, and e = X - F phi' (Algorithm 18, W9).

    Algorithm 18 removes each component from the data before finding the next one; e is the
    data left after the first. Scaling and PCA are fitted on the training rows only."""
    scaler = StandardScaler().fit(X_train)
    S_train, S_test = scaler.transform(X_train), scaler.transform(X_test)
    pca = PCA(n_components=1).fit(S_train)
    phi = pca.components_                     # 1 x 50 loadings
    frames = []
    for S in (S_train, S_test):
        F = pca.transform(S)
        frame = pd.DataFrame(S, columns=X_NAMES)
        frame["F"] = F[:, 0]
        frame[E_NAMES] = S - F @ phi
        frames.append(frame)
    return frames


def build_design(spec, X_train, X_test):
    """Training and test design matrices for one LASSO specification."""
    train, test = factor_frames(X_train, X_test)
    if spec in FORMULAS:
        D_train = patsy.dmatrix(FORMULAS[spec], train, return_type="dataframe")
        D_test = patsy.build_design_matrices([D_train.design_info], test, return_type="dataframe")[0]
        return D_train, D_test
    if spec in ("pca_terms", "pcr_only"):
        # Algorithm 19: PCA of the standardised 1,325-term design, all components kept
        Q_train, Q_test = build_design("baseline", X_train, X_test)
        q_scaler = StandardScaler().fit(Q_train)
        pca = PCA().fit(q_scaler.transform(Q_train))
        V_train = pd.DataFrame(pca.transform(q_scaler.transform(Q_train)), index=Q_train.index)
        V_test = pd.DataFrame(pca.transform(q_scaler.transform(Q_test)), index=Q_test.index)
        V_train.columns = V_test.columns = [f"PC{k + 1}" for k in range(V_train.shape[1])]
        if spec == "pcr_only":
            return V_train, V_test
        return pd.concat([V_train, Q_train], axis=1), pd.concat([V_test, Q_test], axis=1)
    raise ValueError(spec)


def aicc_lasso(D_train, y_train, D_test):
    """Standardise, fit the LASSO at every penalty on the grid, keep the AICc minimum (W4 cell 25)."""
    scaler = StandardScaler().fit(D_train)
    A_train, A_test = scaler.transform(D_train), scaler.transform(D_test)
    n = len(y_train)
    best = (np.inf, None, None)
    for alpha in ALPHA_GRID:
        model = Lasso(alpha=alpha, max_iter=MAX_ITER).fit(A_train, y_train)
        k = int((model.coef_ != 0).sum()) + 1
        if k >= n - 1:
            continue  # AICc is undefined once the fit is saturated
        mse = np.mean((y_train - model.predict(A_train)) ** 2)
        aicc = n * np.log(mse) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        if aicc < best[0]:
            best = (aicc, model.coef_.copy(), model.intercept_)
    _, coef, intercept = best
    return A_test @ coef + intercept, int((coef != 0).sum()), coef


def pls_fit_predict(X_train, y_train, X_test):
    """Algorithm 21 on the 1,325 terms; components chosen by inner 5-fold CV (Week 10)."""
    Q_train, Q_test = build_design("baseline", X_train, X_test)
    inner = KFold(n_splits=5, shuffle=True, random_state=PLS_INNER_SEED)
    inner_mse = []
    for ncomp in range(1, PLS_MAX_COMPONENTS + 1):
        y_cv = cross_val_predict(PLSRegression(n_components=ncomp), Q_train, y_train, cv=inner)
        inner_mse.append(np.mean((y_train - y_cv.ravel()) ** 2))
    best_ncomp = int(np.argmin(inner_mse)) + 1
    pls = PLSRegression(n_components=best_ncomp).fit(Q_train, y_train)
    return pls.predict(Q_test).ravel(), best_ncomp


def fit_predict(spec, X_train, y_train, X_test):
    """Prediction and size (terms kept, or PLS components) for one specification."""
    if spec == "pls":
        return pls_fit_predict(X_train, y_train, X_test)
    D_train, D_test = build_design(spec, X_train, X_test)
    pred, n_terms, _ = aicc_lasso(D_train, y_train, D_test)
    return pred, n_terms


# ## 4. Cross-validation: five specifications on identical folds (Week 3)

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
rows = []
for fold, (tr, va) in enumerate(kfold.split(X), start=1):
    for spec in MAIN_SPECS + CHECK_SPECS:
        pred, size = fit_predict(spec, X[tr], y[tr], X[va])
        mse = np.mean((y[va] - pred) ** 2)
        rows.append({"fold": fold, "spec": spec, "mse": mse, "size": size})
        print(f"fold {fold} {spec:>10}: MSE {mse:.3f}, size {size}")

fold_mse = pd.DataFrame(rows)
fold_mse.to_csv(os.path.join(DATA_DIR, "fold_mse.csv"), index=False)

wide = fold_mse.pivot(index="fold", columns="spec", values="mse")
scores = pd.DataFrame({
    "label": [LABELS[s] for s in MAIN_SPECS + CHECK_SPECS],
    "cv_mse": wide.mean()[MAIN_SPECS + CHECK_SPECS].to_numpy(),
    "sd_over_folds": wide.std(ddof=1)[MAIN_SPECS + CHECK_SPECS].to_numpy(),
    "change_vs_baseline": (wide.mean() - wide.mean()["baseline"])[MAIN_SPECS + CHECK_SPECS].to_numpy(),
    "folds_better_than_baseline": [(wide[s] < wide["baseline"]).sum() for s in MAIN_SPECS + CHECK_SPECS],
    "mean_size": fold_mse.groupby("spec")["size"].mean()[MAIN_SPECS + CHECK_SPECS].to_numpy(),
}, index=MAIN_SPECS + CHECK_SPECS)
scores["cv_r2"] = 1 - scores["cv_mse"] / y.var()
scores.to_csv(os.path.join(DATA_DIR, "model_scores.csv"))
print(scores.round(3).to_string())

# Paired fold differences against the baseline: one dot per fold, one row per specification
fig, ax = plt.subplots(figsize=(7, 3.4), dpi=300)
contenders = MAIN_SPECS[1:]
for i, spec in enumerate(contenders):
    diff = wide[spec] - wide["baseline"]
    ax.scatter(diff, np.full(N_FOLDS, i), s=28, color=SERIES, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(diff.max() + 0.08, i, f"mean {diff.mean():+.2f}", color=INK, fontsize=8, va="center")
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_yticks(range(len(contenders)), [LABELS[s] for s in contenders], color=INK)
ax.invert_yaxis()
ax.set_xlabel("Fold MSE minus A4 baseline (negative = better)", color=INK)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlim(right=ax.get_xlim()[1] + 0.5)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(DATA_DIR, "fig2_fold_differences.png"))
plt.close(fig)


# ## 5. Refit on all 2,400 rows and write submissions

winner = scores.loc[MAIN_SPECS, "cv_mse"].idxmin()  # rule fixed in advance: lowest CV MSE
print(f"winner: {LABELS[winner]}")

for spec, filename in [(winner, "submission.csv"), ("baseline", "submission_control.csv")]:
    pred, size = fit_predict(spec, X, y, X_test)
    pd.DataFrame({"ID": test_df["ID"], "Y": pred}).to_csv(os.path.join(DATA_DIR, filename), index=False)
    print(f"{filename}: {LABELS[spec]}, size {size}")

if winner != "pls":
    D_train, D_test = build_design(winner, X, X_test)
    _, _, coef = aicc_lasso(D_train, y, D_test)
    kept = pd.DataFrame({"term": D_train.columns, "coef": coef})
    kept.loc[kept["coef"] != 0].to_csv(os.path.join(DATA_DIR, "final_coefficients.csv"), index=False)


# ## 6. Kaggle submission (Week 1)

if SUBMIT_TO_KAGGLE:
    api.competition_submit(file_name=os.path.join(DATA_DIR, "submission.csv"),
                           message=f"A5: {LABELS[winner]}", competition=COMPETITION_ID)
if SUBMIT_CONTROL:
    api.competition_submit(file_name=os.path.join(DATA_DIR, "submission_control.csv"),
                           message="A5 control: A4 quadratic LASSO", competition=COMPETITION_ID)
