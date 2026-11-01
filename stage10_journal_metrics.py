"""
stage10_journal_metrics.py
--------------------------
Stage 10 of XMOFS-Deep: the extra evidence a journal paper needs.

Adds (on top of Stages 3-9):
  1. Full metric set on the test set: accuracy, precision, recall, F1-macro,
     MCC and AUC-ROC (one-vs-rest, macro) for XMOFS-Deep and the baselines.
  2. Confusion matrix and ROC curves of the XMOFS-Deep reduced network.
  3. Learning curves (training-set size vs cross-validated F1).
  4. LOCAL explanations: SHAP values for individual test patients, and LIME
     if the 'lime' package is installed (pip install lime).
  5. Friedman test over the 10 repeated splits of Stage 9 (+ mean ranks).

It re-uses the data split, network and training code of Stage 8, so the
stage8_shap_explainability.py file must be next to this one.

USAGE (from the folder that holds the stage scripts and the data):
  BRCA: python stage10_journal_metrics.py --x X_expression_clean.csv --y y_pam50_labels.csv \
          --results results_brca
  OV  : python stage10_journal_metrics.py --x cleaned_data_ov/ov_X_expression_clean.csv \
          --y cleaned_data_ov/ov_y_subtype_labels.csv --results results_ov
lambda / gate_lr are read from <results>/chosen_point.json (written by the runner).
Output goes to <results>/stage10_results/.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
sys.path.insert(0, ".")
import stage8_shap_explainability as s8


def log(m):
    print(f"[stage10] {m}")


def metric_row(name, n_genes, y_true, y_pred, proba):
    return {
        "method": name, "n_genes": n_genes,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_roc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
    }


def svm_fit_predict(Xtr, ytr, Xte, cols, seed):
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=seed)
    clf.fit(Xtr[:, cols], ytr)
    return clf.predict(Xte[:, cols]), clf.predict_proba(Xte[:, cols])


def friedman_part(results_dir, out_dir):
    f = Path(results_dir) / "stage9_results" / "repeat_results.csv"
    if not f.exists():
        log(f"{f} not found - skipping Friedman test")
        return
    df = pd.read_csv(f)
    wide = df.pivot_table(index="repeat", columns="method", values="test_f1_macro").dropna()
    names = dict(zip(df["method"], df["method_name"]))
    stat, p = friedmanchisquare(*[wide[c].values for c in wide.columns])
    ranks = np.vstack([rankdata(-row) for row in wide.values]).mean(axis=0)   # rank 1 = best
    lines = [f"Friedman test on test F1-macro over {len(wide)} repeated splits, "
             f"{wide.shape[1]} methods", f"chi-square = {stat:.4f}, p = {p:.6f}", "",
             "Mean rank (1 = best):"]
    for c, r in sorted(zip(wide.columns, ranks), key=lambda t: t[1]):
        lines.append(f"  {c}: {names.get(c, c)}  mean rank {r:.2f}")
    (out_dir / "friedman.txt").write_text("\n".join(lines))
    log("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    out = Path(a.results) / "stage10_results"
    out.mkdir(parents=True, exist_ok=True)

    cpf = Path(a.results) / "chosen_point.json"
    if cpf.exists():
        cp = json.load(open(cpf))
        s8.LAMBDA, s8.GATE_LR = cp["lambda"], cp["gate_lr"]
    else:
        log("chosen_point.json not found - using the lambda / gate_lr set inside stage8")
    log(f"chosen point: lambda={s8.LAMBDA}, gate_lr={s8.GATE_LR}")

    friedman_part(a.results, out)

    seed = s8.RANDOM_STATE
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data: same split as Stage 8 ----------------
    X = pd.read_csv(a.x, index_col=0)
    y = pd.read_csv(a.y, index_col=0)["label"].loc[X.index]
    if X.isna().sum().sum() > 0:
        X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    genes = np.array(X.columns)
    le = LabelEncoder(); y_enc = le.fit_transform(y); classes = list(le.classes_)
    nc = len(classes)
    X_tv, X_te, y_tv, y_te = train_test_split(X.values, y_enc, test_size=s8.TEST_SIZE,
                                              random_state=seed, stratify=y_enc)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=s8.VAL_SIZE,
                                              random_state=seed, stratify=y_tv)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_va = sc.transform(X_va); X_te = sc.transform(X_te)
    cw = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr),
                      dtype=torch.float32).to(device)

    # ---------------- XMOFS-Deep chosen model ----------------
    model = s8.train_gated_model(X_tr, y_tr, X_va, y_va, X.shape[1], nc, device, cw)
    gates = model.gate.get_final_gates().cpu().numpy()
    sel = np.where(gates > s8.GATE_SELECT_THRESHOLD)[0]
    k = len(sel)
    if k == 0:
        raise RuntimeError("No genes selected; rerun stage 8/9 or pick another point.")
    sel_genes = genes[sel]
    reduced = s8.ReducedNet(model.cpu(), sel).eval()
    with torch.no_grad():
        logits = reduced(torch.tensor(X_te[:, sel], dtype=torch.float32))
        proba_x = torch.softmax(logits, dim=1).numpy()
    pred_x = proba_x.argmax(1)

    # ---------------- baselines at matched k ----------------
    rows = [metric_row("XMOFS-Deep (reduced net)", k, y_te, pred_x, proba_x)]
    p, pr = svm_fit_predict(X_tr, y_tr, X_te, sel, seed)
    rows.append(metric_row("SVM on XMOFS genes", k, y_te, p, pr))
    F, _ = f_classif(X_tr, y_tr); F = np.nan_to_num(F)
    top_k = np.argsort(-F)[:k]
    p, pr = svm_fit_predict(X_tr, y_tr, X_te, top_k, seed)
    rows.append(metric_row("SVM on top-k ANOVA genes", k, y_te, p, pr))
    rng = np.random.RandomState(seed)
    rnd = rng.choice(X.shape[1], size=k, replace=False)
    p, pr = svm_fit_predict(X_tr, y_tr, X_te, rnd, seed)
    rows.append(metric_row("SVM on k random genes", k, y_te, p, pr))
    top500 = np.argsort(-F)[:500]
    p, pr = svm_fit_predict(X_tr, y_tr, X_te, top500, seed)
    rows.append(metric_row("SVM on top-500 ANOVA genes", 500, y_te, p, pr))
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics_table.csv", index=False)
    log("\n" + table.round(4).to_string(index=False))
    (out / "classification_report_xmofs.txt").write_text(
        classification_report(y_te, pred_x, target_names=classes, zero_division=0))

    # ---------------- confusion matrix + ROC ----------------
    cm = confusion_matrix(y_te, pred_x)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False,
                                                            xticks_rotation=30)
    ax.set_title("XMOFS-Deep: confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=200); plt.close(fig)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")

    Yb = label_binarize(y_te, classes=list(range(nc)))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Yb[:, i], proba_x[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("XMOFS-Deep: one-vs-rest ROC (test set)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=200); plt.close(fig)

    # ---------------- learning curves (training set only) ----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sizes = np.linspace(0.3, 1.0, 6)
    curves = {
        "SVM on XMOFS genes": (SVC(kernel="rbf", C=1.0, class_weight="balanced"), X_tr[:, sel]),
        f"SVM on top-{k} ANOVA (inside CV)": (Pipeline([("s", SelectKBest(f_classif, k=k)),
                      ("m", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]), X_tr),
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.2)); rows_lc = []
    for (name, (est, Xc)), col in zip(curves.items(), ["#0F766E", "#B45309"]):
        ts, trs, vas = learning_curve(est, Xc, y_tr, cv=cv, train_sizes=sizes,
                                      scoring="f1_macro", n_jobs=1)
        ax.plot(ts, trs.mean(1), "--", color=col, label=f"{name}: train")
        ax.plot(ts, vas.mean(1), "-", color=col, label=f"{name}: CV")
        for t, a_, b_ in zip(ts, trs.mean(1), vas.mean(1)):
            rows_lc.append({"curve": name, "train_size": int(t), "train_f1": a_, "cv_f1": b_})
    ax.set_xlabel("Training samples"); ax.set_ylabel("F1-macro"); ax.grid(alpha=0.25)
    ax.legend(fontsize=7); ax.set_title("Learning curves (training split only)")
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=200); plt.close(fig)
    pd.DataFrame(rows_lc).to_csv(out / "learning_curves.csv", index=False)

    # ---------------- local explanations ----------------
    import shap
    rs = np.random.RandomState(seed)
    bg = torch.tensor(X_tr[rs.choice(len(X_tr), size=min(100, len(X_tr)), replace=False)][:, sel],
                      dtype=torch.float32)
    Xt = torch.tensor(X_te[:, sel], dtype=torch.float32)
    sv = s8.shap_to_array(shap.GradientExplainer(reduced, bg).shap_values(Xt), nc)  # (cls, n, genes)
    picks = []
    for c in range(nc):                      # most confident correct patient of each subtype
        ok = np.where((y_te == c) & (pred_x == c))[0]
        if len(ok):
            picks.append(int(ok[np.argmax(proba_x[ok, c])]))
    ncol = len(picks)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.8), sharex=False)
    axes = np.atleast_1d(axes); rows_loc = []
    for ax, i in zip(axes, picks):
        c = int(pred_x[i]); v = sv[c, i, :]
        order = np.argsort(-np.abs(v))[:8][::-1]
        ax.barh([sel_genes[j] for j in order], v[order],
                color=["#B33A3A" if v[j] > 0 else "#3A66B5" for j in order])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"Patient {i}\npredicted {classes[c]}\n(p = {proba_x[i, c]:.2f})", fontsize=8)
        ax.tick_params(labelsize=7); ax.set_xlabel("SHAP value", fontsize=7)
        for j in range(len(sel)):
            rows_loc.append({"patient_index": i, "predicted": classes[c], "gene": sel_genes[j],
                             "shap": float(sv[c, i, j]), "z_expression": float(X_te[i, sel[j]])})
    fig.suptitle("Local SHAP explanations (red raises, blue lowers the predicted subtype score)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / "local_shap_patients.png", dpi=200); plt.close(fig)
    pd.DataFrame(rows_loc).to_csv(out / "local_shap_patients.csv", index=False)

    try:
        from lime.lime_tabular import LimeTabularExplainer
        expl = LimeTabularExplainer(X_tr[:, sel], feature_names=list(sel_genes),
                                    class_names=classes, mode="classification",
                                    discretize_continuous=True, random_state=seed)
        def fn(arr):
            with torch.no_grad():
                return torch.softmax(reduced(torch.tensor(arr, dtype=torch.float32)), 1).numpy()
        rows_l = []; fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.8))
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, picks):
            c = int(pred_x[i])
            e = expl.explain_instance(X_te[i, sel], fn, labels=(c,), num_features=8, num_samples=2000)
            items = e.as_list(label=c)[::-1]
            ax.barh([t[0] for t in items], [t[1] for t in items],
                    color=["#B33A3A" if t[1] > 0 else "#3A66B5" for t in items])
            ax.set_title(f"Patient {i}: {classes[c]}", fontsize=8); ax.tick_params(labelsize=6)
            rows_l += [{"patient_index": i, "predicted": classes[c], "rule": t[0], "weight": t[1]}
                       for t in items]
        fig.suptitle("Local LIME explanations", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "local_lime_patients.png", dpi=200); plt.close(fig)
        pd.DataFrame(rows_l).to_csv(out / "local_lime_patients.csv", index=False)
        log("LIME done")
    except ImportError:
        log("lime not installed - run  pip install lime  and rerun to add LIME (optional)")

    (out / "run_summary.txt").write_text(
        f"genes selected: {k}\nselected genes: {', '.join(sel_genes)}\n"
        f"test patients explained locally: {picks}\n")
    log("DONE - results in " + str(out))


if __name__ == "__main__":
    main()
