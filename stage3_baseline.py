"""
stage3_baseline.py
--------------------
Stage 3 of XMOFS-Deep: stratified train/test split + first baseline
(filter-based feature selection + Random Forest / SVM classifier).

This gives a performance "floor" - XMOFS-Deep must beat this to justify
the extra complexity of deep gating + multi-objective optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage3_results"

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_FOLDS = 5

# Feature-count settings to test - mirrors what your Pareto front will later
# search over, so these numbers are directly comparable later
TOP_K_GENES_OPTIONS = [20, 50, 100, 200, 500]
# ===========================================================================


def log(msg):
    print(f"[stage3_baseline] {msg}")


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    log("Loading data...")
    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"]
    y = y.loc[X.index]  # ensure alignment

    log(f"X: {X.shape}, y: {y.shape}")

    # Safety net: if the uploaded/transferred file picked up any NaNs
    # (e.g. an interrupted upload), warn loudly and fix rather than crash.
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        log(f"WARNING: found {n_missing} missing values in X - this should "
            f"be 0 if the file transferred correctly. Check your upload "
            f"(file size, re-upload if needed). Imputing with column "
            f"median as a stopgap so the script can still run.")
        X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    else:
        log("Confirmed: 0 missing values in X.")
    log(f"Class distribution:\n{y.value_counts()}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    log(f"Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # -----------------------------------------------------------------
    # Stratified train/test split (held out for final, honest evaluation)
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    log(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    log(f"Train class proportions: {train_dist.round(3).tolist()}")
    log(f"Test class proportions:  {test_dist.round(3).tolist()}")

    # -----------------------------------------------------------------
    # Baseline: for each feature count, select top-K genes by ANOVA
    # F-score (filter method), then train RF and SVM, evaluate on
    # stratified 5-fold CV (on training set) AND the held-out test set
    # -----------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for k in TOP_K_GENES_OPTIONS:
        log(f"\n--- Feature count = {k} ---")

        selector = SelectKBest(score_func=f_classif, k=k)

        for model_name, model in [
            ("RandomForest", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
            ("SVM_RBF", SVC(kernel="rbf", C=1.0, class_weight="balanced",
                             random_state=RANDOM_STATE)),
        ]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", selector),
                ("clf", model),
            ])

            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf,
                                         scoring="f1_macro", n_jobs=-1)

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1_macro = f1_score(y_test, y_pred, average="macro")

            log(f"  {model_name}: CV f1_macro = {cv_scores.mean():.3f} "
                f"(+/- {cv_scores.std():.3f}) | "
                f"Test acc = {test_acc:.3f}, Test f1_macro = {test_f1_macro:.3f}")

            results.append({
                "n_features": k,
                "model": model_name,
                "cv_f1_macro_mean": cv_scores.mean(),
                "cv_f1_macro_std": cv_scores.std(),
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1_macro,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "baseline_results.csv", index=False)
    log(f"\nSaved all results to {out_dir / 'baseline_results.csv'}")

    # Best single baseline config -> detailed report
    best_row = results_df.loc[results_df["test_f1_macro"].idxmax()]
    log(f"\nBest baseline config: {best_row['model']} with "
        f"{best_row['n_features']} genes "
        f"(test f1_macro = {best_row['test_f1_macro']:.3f})")

    best_k = int(best_row["n_features"])
    best_model_name = best_row["model"]
    best_model = (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                          class_weight="balanced")
                  if best_model_name == "RandomForest"
                  else SVC(kernel="rbf", C=1.0, class_weight="balanced",
                           random_state=RANDOM_STATE))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=best_k)),
        ("clf", best_model),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    log(f"\nDetailed classification report (best config):\n{report}")

    with open(out_dir / "best_config_report.txt", "w") as f:
        f.write(f"Best config: {best_model_name}, k={best_k} genes\n\n")
        f.write(report)

    log(f"\nDone. This is your baseline #1 (classical filter-based FS) "
        f"performance floor - XMOFS-Deep should be compared against this.")


if __name__ == "__main__":
    main()
