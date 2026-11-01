"""
stage7_hypervolume_comparison.py
-----------------------------------
Stage 7 of XMOFS-Deep: fair, rigorous comparison of THREE methods using
hypervolume and cross-hypervolume (Part C.1/C.2 of the project guide):

  1. Baseline #1 (Stage 3): classical filter-based FS + RF/SVM,
     evaluated at fixed feature counts (not a real multi-objective search,
     but still produces a set of (accuracy, feature-count) points we can
     treat as a "front" for comparison).

  2. Baseline #3: a NEW shallow-inner-model NSGA-II search, built in this
     script. Same multi-objective machinery as Stage 6, but the inner
     model is Logistic Regression with L1 regularization (a classical
     embedded sparse method) instead of a deep gated network. This
     isolates exactly ONE variable: deep vs shallow inner model, with
     everything else (NSGA-II, objectives, data splits) held constant -
     this is THE key comparison for your core research claim.

  3. XMOFS-Deep (Stage 6): the deep gated NSGA-II front, loaded from
     Stage 6's saved results.

OUTPUT: a comparison table, a hypervolume score per method, and
cross-hypervolume contributions showing exactly where each method wins.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
STAGE3_RESULTS = "stage3_results/baseline_results.csv"
STAGE6_PARETO_FRONT = "stage6_results/pareto_front.csv"
OUTPUT_DIR = "stage7_results"

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Shallow NSGA-II search settings (fast model -> can afford a bigger search)
SHALLOW_POP_SIZE = 12
SHALLOW_N_GENERATIONS = 10
LOG10_C_RANGE = (-3.0, 1.0)   # sklearn's C = 1/lambda; smaller C = sparser

HV_REF_POINT = [1.0, 1.0]   # worst case: 0 F1 (loss=1), all genes used (fraction=1)
# ===========================================================================


def log(msg):
    print(f"[stage7_comparison] {msg}")


def load_data():
    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"]
    y = y.loc[X.index]
    if X.isna().sum().sum() > 0:
        X = X.apply(lambda col: col.fillna(col.median()), axis=0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.values, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_trainval)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_genes = X.shape[1]
    return X_train, y_train, X_val, y_val, X_test, y_test, n_genes


# --------------------------------------------------------------------
# Shallow NSGA-II: Logistic Regression with L1 penalty as inner model
# --------------------------------------------------------------------
def train_and_evaluate_shallow(log10_C, X_train, y_train, X_val, y_val, class_weight_dict):
    C = 10 ** log10_C
    # liblinear is much faster than saga for L1 penalty on data of this size,
    # but doesn't natively support multinomial multi-class - wrap in
    # One-vs-Rest (a standard, acceptable simplification for a classical
    # baseline; note this explicitly in your report).
    base_clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                   class_weight=class_weight_dict, max_iter=200,
                                   random_state=RANDOM_STATE)
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)

    # A gene is "selected" if ANY of the one-vs-rest classifiers has a
    # non-zero coefficient for it.
    coefs = np.vstack([est.coef_ for est in clf.estimators_])  # (n_classes, n_features)
    n_selected = int((np.abs(coefs).max(axis=0) > 1e-6).sum())

    val_pred = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    return val_f1, n_selected, clf


class ShallowNSGAProblem(Problem):
    def __init__(self, X_train, y_train, X_val, y_val, n_total_genes, class_weight_dict):
        super().__init__(n_var=1, n_obj=2, n_constr=0,
                          xl=np.array([LOG10_C_RANGE[0]]),
                          xu=np.array([LOG10_C_RANGE[1]]))
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.n_total_genes = n_total_genes
        self.class_weight_dict = class_weight_dict
        self.eval_log = []

    def _evaluate(self, X, out, *args, **kwargs):
        objs = []
        for row in X:
            log10_C = row[0]
            val_f1, n_selected, _ = train_and_evaluate_shallow(
                log10_C, self.X_train, self.y_train, self.X_val, self.y_val,
                self.class_weight_dict)
            obj1 = -val_f1
            obj2 = n_selected / self.n_total_genes
            objs.append([obj1, obj2])
            self.eval_log.append({"log10_C": log10_C, "C": 10**log10_C,
                                   "val_f1_macro": val_f1, "n_selected_genes": n_selected})
            log(f"  shallow individual: C={10**log10_C:.4f} -> "
                f"val_f1={val_f1:.3f}, n_genes={n_selected}")
        out["F"] = np.array(objs)


def run_shallow_nsga2(X_train, y_train, X_val, y_val, X_test, y_test, n_genes):
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    problem = ShallowNSGAProblem(X_train, y_train, X_val, y_val, n_genes, class_weight_dict)
    algorithm = NSGA2(pop_size=SHALLOW_POP_SIZE)

    log(f"Starting SHALLOW NSGA-II (Logistic Regression inner model): "
        f"pop_size={SHALLOW_POP_SIZE}, generations={SHALLOW_N_GENERATIONS}")
    result = minimize(problem, algorithm, ("n_gen", SHALLOW_N_GENERATIONS),
                       seed=RANDOM_STATE, verbose=True)

    front_X = result.X
    if front_X.ndim == 1:
        front_X = front_X.reshape(1, -1)

    front_records = []
    for row in front_X:
        log10_C = row[0]
        val_f1, n_selected, clf = train_and_evaluate_shallow(
            log10_C, X_train, y_train, X_val, y_val, class_weight_dict)
        test_pred = clf.predict(X_test)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_acc = accuracy_score(y_test, test_pred)
        front_records.append({
            "C": 10**log10_C, "val_f1_macro": val_f1, "n_selected_genes": n_selected,
            "test_f1_macro": test_f1, "test_accuracy": test_acc,
        })

    front_df = pd.DataFrame(front_records).sort_values("n_selected_genes")
    return front_df, pd.DataFrame(problem.eval_log)


# --------------------------------------------------------------------
# Hypervolume + cross-hypervolume
# --------------------------------------------------------------------
def to_hv_points(df, n_total_genes, f1_col="test_f1_macro", n_col="n_selected_genes"):
    """Convert a front's (f1, n_genes) into normalized (loss, gene_fraction)
    points for hypervolume computation. Both minimized, both in [0,1]."""
    loss = 1 - df[f1_col].values
    frac = df[n_col].values / n_total_genes
    return np.column_stack([loss, frac])


def to_hv_points_log(df, n_total_genes, f1_col="test_f1_macro", n_col="n_selected_genes"):
    """Same as to_hv_points, but the gene-count objective is log10(max(n, 1)) / log10(total genes).
    With the plain fraction n / total, every panel of a few hundred genes looks almost as 'cheap' as a
    20-gene panel (all fractions are tiny), so the hypervolume mostly measures the best F1 reached at
    modest sizes. The log version rewards compact panels. We report both."""
    loss = 1 - df[f1_col].values
    g = np.log10(np.maximum(df[n_col].values, 1)) / np.log10(n_total_genes)
    return np.column_stack([loss, g])


def compute_hv(points, ref_point):
    hv_indicator = HV(ref_point=np.array(ref_point))
    return hv_indicator(points)


def non_dominated_filter(points):
    """Keep only non-dominated points (both objectives minimized)."""
    is_nd = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and is_nd[i]:
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    is_nd[i] = False
                    break
    return points[is_nd]


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    log("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, n_genes = load_data()
    log(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}, "
        f"Total genes: {n_genes}")

    # -----------------------------------------------------------------
    # Method 1: Stage 3 filter-based baseline (load existing results)
    # -----------------------------------------------------------------
    stage3_df = pd.read_csv(STAGE3_RESULTS)
    # Take the best model per feature-count (RF or SVM, whichever is better)
    stage3_best = stage3_df.loc[stage3_df.groupby("n_features")["test_f1_macro"].idxmax()]
    stage3_best = stage3_best.rename(columns={"n_features": "n_selected_genes"})
    log(f"\nLoaded Stage 3 (filter-based) front: {len(stage3_best)} points")
    log(stage3_best[["n_selected_genes", "model", "test_f1_macro", "test_accuracy"]].to_string(index=False))

    # -----------------------------------------------------------------
    # Method 2: Shallow NSGA-II (NEW - run now)
    # -----------------------------------------------------------------
    shallow_front, shallow_all_evals = run_shallow_nsga2(
        X_train, y_train, X_val, y_val, X_test, y_test, n_genes)
    shallow_front.to_csv(out_dir / "shallow_nsga2_pareto_front.csv", index=False)
    shallow_all_evals.to_csv(out_dir / "shallow_nsga2_all_individuals.csv", index=False)
    log(f"\nShallow NSGA-II front: {len(shallow_front)} points")
    log(shallow_front.to_string(index=False))

    # -----------------------------------------------------------------
    # Method 3: Deep gated NSGA-II (Stage 6 - already computed)
    # -----------------------------------------------------------------
    deep_front = pd.read_csv(STAGE6_PARETO_FRONT)
    log(f"\nLoaded Stage 6 (deep gated NSGA-II) front: {len(deep_front)} points")
    log(deep_front[["n_selected_genes", "test_f1_macro", "test_accuracy"]].to_string(index=False))

    # -----------------------------------------------------------------
    # Hypervolume for each method (using TEST set metrics, normalized by
    # total gene count for a fair, consistent comparison)
    # -----------------------------------------------------------------
    stage3_points = non_dominated_filter(to_hv_points(stage3_best, n_genes))
    shallow_points = non_dominated_filter(to_hv_points(shallow_front, n_genes))
    deep_points = non_dominated_filter(to_hv_points(deep_front, n_genes))

    hv_stage3 = compute_hv(stage3_points, HV_REF_POINT)
    hv_shallow = compute_hv(shallow_points, HV_REF_POINT)
    hv_deep = compute_hv(deep_points, HV_REF_POINT)

    log(f"\n{'='*60}")
    log(f"HYPERVOLUME COMPARISON (ref point {HV_REF_POINT})")
    log(f"{'='*60}")
    log(f"  Stage 3  (filter-based, fixed k)      : HV = {hv_stage3:.4f}")
    log(f"  Baseline #3 (shallow NSGA-II, LogReg)  : HV = {hv_shallow:.4f}")
    log(f"  XMOFS-Deep (deep gated NSGA-II)        : HV = {hv_deep:.4f}")

    # -----------------------------------------------------------------
    # Cross-hypervolume: unique contribution of each method over another
    # -----------------------------------------------------------------
    def cross_contribution(points_a, points_b, ref_point):
        union = np.vstack([points_a, points_b])
        union_nd = non_dominated_filter(union)
        hv_union = compute_hv(union_nd, ref_point)
        hv_b = compute_hv(points_b, ref_point)
        return hv_union - hv_b

    cross_deep_over_shallow = cross_contribution(deep_points, shallow_points, HV_REF_POINT)
    cross_shallow_over_deep = cross_contribution(shallow_points, deep_points, HV_REF_POINT)
    cross_deep_over_stage3 = cross_contribution(deep_points, stage3_points, HV_REF_POINT)
    cross_stage3_over_deep = cross_contribution(stage3_points, deep_points, HV_REF_POINT)

    log(f"\n{'='*60}")
    log(f"CROSS-HYPERVOLUME (unique contribution of A beyond B)")
    log(f"{'='*60}")
    log(f"  Deep gated NSGA-II unique contribution over Shallow NSGA-II : {cross_deep_over_shallow:.4f}")
    log(f"  Shallow NSGA-II unique contribution over Deep gated NSGA-II : {cross_shallow_over_deep:.4f}")
    log(f"  Deep gated NSGA-II unique contribution over Stage 3 filter  : {cross_deep_over_stage3:.4f}")
    log(f"  Stage 3 filter unique contribution over Deep gated NSGA-II  : {cross_stage3_over_deep:.4f}")

    if cross_deep_over_shallow > cross_shallow_over_deep:
        log(f"\n  -> XMOFS-Deep's front dominates MORE unique territory than "
            f"the shallow NSGA-II baseline. This SUPPORTS the core research claim.")
    else:
        log(f"\n  -> The shallow NSGA-II baseline dominates more unique "
            f"territory than XMOFS-Deep in this run. Report this honestly - "
            f"it may motivate increasing the inner network's training budget "
            f"(see Stage 6 comments) before final conclusions.")

    # -----------------------------------------------------------------
    # Robustness check: the same comparison with a log-scaled gene-count objective
    # -----------------------------------------------------------------
    s3_log = non_dominated_filter(to_hv_points_log(stage3_best, n_genes))
    sh_log = non_dominated_filter(to_hv_points_log(shallow_front, n_genes))
    dp_log = non_dominated_filter(to_hv_points_log(deep_front, n_genes))
    hv_s3_log, hv_sh_log, hv_dp_log = (compute_hv(x, HV_REF_POINT) for x in (s3_log, sh_log, dp_log))
    log_cross = {
        "Deep over Shallow": cross_contribution(dp_log, sh_log, HV_REF_POINT),
        "Shallow over Deep": cross_contribution(sh_log, dp_log, HV_REF_POINT),
        "Deep over Stage3": cross_contribution(dp_log, s3_log, HV_REF_POINT),
        "Stage3 over Deep": cross_contribution(s3_log, dp_log, HV_REF_POINT),
    }
    log(f"\n{'=' * 60}\nROBUSTNESS: same comparison with log10(gene count) as objective 2\n{'=' * 60}")
    log(f"  Stage 3 filter : HV = {hv_s3_log:.4f}")
    log(f"  Shallow NSGA-II: HV = {hv_sh_log:.4f}")
    log(f"  XMOFS-Deep     : HV = {hv_dp_log:.4f}")
    for k_, v_ in log_cross.items():
        log(f"  {k_}: {v_:.4f}")
    log("  If a ranking differs between the linear and the log version, report BOTH and say "
        "that the conclusion depends on how compactness is scaled.")
    pd.DataFrame([
        {"method": "Stage 3: Filter-based (fixed k)", "hypervolume_log": hv_s3_log},
        {"method": "Baseline #3: Shallow NSGA-II (LogReg)", "hypervolume_log": hv_sh_log},
        {"method": "XMOFS-Deep: Deep gated NSGA-II", "hypervolume_log": hv_dp_log},
    ]).to_csv(out_dir / "hypervolume_log_scale.csv", index=False)

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------
    summary = pd.DataFrame([
        {"method": "Stage 3: Filter-based (fixed k)", "hypervolume": hv_stage3,
         "n_front_points": len(stage3_points)},
        {"method": "Baseline #3: Shallow NSGA-II (LogReg)", "hypervolume": hv_shallow,
         "n_front_points": len(shallow_points)},
        {"method": "XMOFS-Deep: Deep gated NSGA-II", "hypervolume": hv_deep,
         "n_front_points": len(deep_points)},
    ])
    summary.to_csv(out_dir / "hypervolume_summary.csv", index=False)

    with open(out_dir / "cross_hypervolume_summary.txt", "w") as f:
        f.write(f"Deep over Shallow: {cross_deep_over_shallow:.4f}\n")
        f.write(f"Shallow over Deep: {cross_shallow_over_deep:.4f}\n")
        f.write(f"Deep over Stage3:  {cross_deep_over_stage3:.4f}\n")
        f.write(f"Stage3 over Deep:  {cross_stage3_over_deep:.4f}\n")
        f.write("\n-- same comparison with log10(gene count) as objective 2 --\n")
        for k_, v_ in log_cross.items():
            f.write(f"{k_} (log): {v_:.4f}\n")

    log(f"\nSaved all results to '{out_dir}/'")
    log(f"\nNext (Stage 8): SHAP explainability on the best XMOFS-Deep front point.")


if __name__ == "__main__":
    main()
