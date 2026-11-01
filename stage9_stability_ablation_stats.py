"""
stage9_stability_ablation_stats.py
-------------------------------------
Stage 9 of XMOFS-Deep. Answers three questions that a single run cannot:

  Q1 (STABILITY)  Which genes does XMOFS-Deep pick again and again, and
                  how much does the gene list change between runs?
  Q2 (ABLATION)   At the SAME number of genes, what really drives the
                  result: the gene choice, or the deep classifier?
  Q3 (STATISTICS) Are the differences real, or just random variation?

HOW IT WORKS
  We repeat the whole experiment N_REPEATS times. Every repeat uses a NEW
  stratified train/val/test split (different split seed) and a new network
  initialisation. In every repeat we:
    1. Train XMOFS-Deep's gated network (the "balanced" Pareto solution from
       Stage 6). The genes with gate > 0.5 are the selected panel S, k=|S|.
    2. Compare, at the same k, on the SAME test set:
         A. XMOFS gated deep net                      (the proposed model)
         B. SVM trained on XMOFS genes S              (same genes, shallow classifier)
         C. SVM trained on the top-k ANOVA genes      (classical filter genes)
         D. SVM trained on k random genes             (sanity floor)
         E. Gated net with NO sparsity penalty        (ablation: lambda = 0)
  Then we test the differences with paired Wilcoxon signed-rank tests,
  Holm-corrected for multiple comparisons.

HOW TO READ THE ABLATION
  B vs C (same classifier = SVM)  -> are XMOFS genes better than filter genes?
  A vs B (same genes)             -> does the deep classifier add anything?
  B vs D                          -> are the selected genes better than chance?
  A vs E                          -> what does the sparsity penalty change?

HONEST LIMITS (also write these in your report)
  * Repeated random splits overlap, so the repeats are not fully independent
    and p-values are somewhat optimistic. Treat them as supporting evidence,
    not proof. Nested cross-validation would be stricter.
  * Each repeat also changes the training data, so gene-list differences mix
    "model randomness" and "data sampling" - this is realistic.
  * k differs per repeat (XMOFS decides it), so comparisons are matched
    PER REPEAT, not to one fixed k.
"""

import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, hypergeom

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.utils.class_weight import compute_class_weight

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage9_results"

N_REPEATS = 10            # more repeats = stronger statistics (20 is better, 2x slower)
BASE_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15

# The "balanced" solution from your Stage 6 Pareto front
LAMBDA = 0.003564
GATE_LR = 0.121347

# Same network/training settings as Stage 6 and Stage 8
HIDDEN_SIZES = [256, 64]
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
NETWORK_LR = 1e-3
GATE_SIGMA = 0.5
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 8
GATE_SELECT_THRESHOLD = 0.5

RUN_NO_SPARSITY_ABLATION = True   # method E (adds one more training per repeat)
TOP_N_PLOT = 20
# ===========================================================================

PAM50_GENES = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
    "TYMS", "UBE2C", "UBE2T", "ORC6L", "CDCA1", "KNTC2", "HER2",
}

METHODS = {
    "A": "XMOFS-Deep gated net",
    "B": "SVM on XMOFS genes",
    "C": "SVM on top-k ANOVA genes",
    "D": "SVM on k random genes",
    "E": "Gated net, no sparsity (lambda=0)",
}


def log(msg):
    print(f"[stage9] {msg}")


# --------------------------------------------------------------------
# Model (same as Stage 5/6/8)
# --------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class StochasticGate(nn.Module):
    def __init__(self, n_features, sigma=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((n_features,), 0.5))
        self.sigma = sigma

    def forward(self, x):
        z = self.mu + self.sigma * torch.randn_like(self.mu) if self.training else self.mu
        gate = torch.clamp(z, 0.0, 1.0)
        return x * gate, gate

    def regularization(self):
        return Normal(0.0, 1.0).cdf(self.mu / self.sigma).sum()

    def get_final_gates(self):
        return torch.clamp(self.mu, 0.0, 1.0).detach()


class GatedDeepNet(nn.Module):
    def __init__(self, n_features, hidden_sizes, n_classes, dropout, gate_sigma):
        super().__init__()
        self.gate = StochasticGate(n_features, sigma=gate_sigma)
        layers, in_size = [], n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        gx, g = self.gate(x)
        return self.net(gx), g


def train_gated(X_train, y_train, X_val, y_val, n_genes, n_classes, device, cw, lam):
    tl = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    model = GatedDeepNet(n_genes, HIDDEN_SIZES, n_classes, DROPOUT, GATE_SIGMA).to(device)
    opt = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": NETWORK_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.gate.parameters(), "lr": GATE_LR, "weight_decay": 0.0},
    ])
    crit = nn.CrossEntropyLoss(weight=cw)

    best, state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(xb)
            loss = crit(out, yb) + lam * model.gate.regularization()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = []
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                out, _ = model(xb)
                v.append((crit(out, yb) + lam * model.gate.regularization()).item())
            vloss = float(np.mean(v))
        if vloss < best:
            best, state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_gated(model, X, device):
    out, _ = model(torch.tensor(X, dtype=torch.float32).to(device))
    return out.argmax(1).cpu().numpy()


def svm_eval(X_tr, y_tr, X_te, y_te, idx, seed):
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X_tr[:, idx], y_tr)
    pred = clf.predict(X_te[:, idx])
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")


# --------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------
def holm_adjust(pvals):
    """Holm step-down correction for multiple comparisons."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | repeats: {N_REPEATS}")

    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"].loc[X.index]
    if X.isna().sum().sum() > 0:
        X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    gene_names = np.array(X.columns)
    n_genes = X.shape[1]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    pam_in_data = len(set(gene_names) & PAM50_GENES)
    log(f"Samples {X.shape[0]}, genes {n_genes}, PAM50 genes in data: {pam_in_data}")

    rows = []            # per repeat, per method results
    selected_sets = []   # selected gene index arrays
    pam_counts, ks = [], []
    collapsed = 0

    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        X_tv, X_te, y_tv, y_te = train_test_split(
            X.values, y_enc, test_size=TEST_SIZE, random_state=seed, stratify=y_enc)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=VAL_SIZE, random_state=seed, stratify=y_tv)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_va = sc.transform(X_va)
        X_te = sc.transform(X_te)
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(cw, dtype=torch.float32).to(device)

        # ---- A: XMOFS-Deep gated net ----
        model = train_gated(X_tr, y_tr, X_va, y_va, n_genes, n_classes, device, cw, LAMBDA)
        pred = predict_gated(model, X_te, device)
        accA, f1A = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
        gates = model.gate.get_final_gates().cpu().numpy()
        S = np.where(gates > GATE_SELECT_THRESHOLD)[0]
        k = len(S)
        if k == 0:
            collapsed += 1
            log(f"repeat {r+1}/{N_REPEATS}: gates collapsed to 0 genes - skipped")
            continue

        # ---- B, C, D: SVM at matched k ----
        accB, f1B = svm_eval(X_tr, y_tr, X_te, y_te, S, seed)
        F, _ = f_classif(X_tr, y_tr)
        F = np.nan_to_num(F)
        topk = np.argsort(-F)[:k]
        accC, f1C = svm_eval(X_tr, y_tr, X_te, y_te, topk, seed)
        rand_idx = rng.choice(n_genes, size=k, replace=False)
        accD, f1D = svm_eval(X_tr, y_tr, X_te, y_te, rand_idx, seed)

        res = {"A": (accA, f1A, k), "B": (accB, f1B, k), "C": (accC, f1C, k), "D": (accD, f1D, k)}

        # ---- E: no sparsity penalty ----
        if RUN_NO_SPARSITY_ABLATION:
            modelE = train_gated(X_tr, y_tr, X_va, y_va, n_genes, n_classes, device, cw, 0.0)
            predE = predict_gated(modelE, X_te, device)
            kE = int((modelE.gate.get_final_gates() > GATE_SELECT_THRESHOLD).sum().item())
            res["E"] = (accuracy_score(y_te, predE), f1_score(y_te, predE, average="macro"), kE)

        for m, (a, f, kk) in res.items():
            rows.append({"repeat": r, "seed": seed, "method": m, "method_name": METHODS[m],
                         "test_accuracy": a, "test_f1_macro": f, "n_genes": kk})

        selected_sets.append(S)
        pam_hits = int(np.isin(gene_names[S], list(PAM50_GENES)).sum())
        pam_counts.append(pam_hits)
        ks.append(k)
        log(f"repeat {r+1}/{N_REPEATS}: k={k} genes | F1  A={f1A:.3f}  B={f1B:.3f}  "
            f"C={f1C:.3f}  D={f1D:.3f}" + (f"  E={res['E'][1]:.3f} (k={res['E'][2]})"
                                           if "E" in res else "") + f" | PAM50 hits={pam_hits}")

    if len(selected_sets) < 2:
        raise RuntimeError("Fewer than 2 usable repeats - check LAMBDA/epochs.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "repeat_results.csv", index=False)
    n_ok = len(selected_sets)
    log(f"\nUsable repeats: {n_ok}/{N_REPEATS} (collapsed to 0 genes: {collapsed})")

    # ------------------------- summary table --------------------------
    summ = (df.groupby(["method", "method_name"])
              .agg(f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
                   acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", "std"),
                   genes_mean=("n_genes", "mean")).reset_index())
    summ.to_csv(out_dir / "summary_table.csv", index=False)
    log("\n=== SUMMARY (mean +/- std over repeats) ===")
    for _, rr in summ.iterrows():
        log(f"  {rr['method']}. {rr['method_name']:<36} F1 = {rr['f1_mean']:.3f} +/- {rr['f1_std']:.3f} | "
            f"acc = {rr['acc_mean']:.3f} +/- {rr['acc_std']:.3f} | genes = {rr['genes_mean']:.1f}")

    # ------------------------- statistical tests ----------------------
    piv = df.pivot(index="repeat", columns="method", values="test_f1_macro")
    comparisons = [
        ("A", "B", "Deep net vs SVM (same XMOFS genes): does the deep classifier help?"),
        ("B", "C", "XMOFS genes vs ANOVA genes (same SVM): are XMOFS genes better?"),
        ("B", "D", "XMOFS genes vs random genes (same SVM): better than chance?"),
        ("A", "C", "XMOFS-Deep vs classical filter+SVM at the same k (overall)"),
    ]
    if RUN_NO_SPARSITY_ABLATION:
        comparisons.append(("A", "E", "With vs without sparsity penalty (note: gene counts differ)"))
    trows = []
    for a, b, desc in comparisons:
        sub = piv[[a, b]].dropna()
        p = paired_wilcoxon(sub[a].values, sub[b].values)
        diff = sub[a].values - sub[b].values
        trows.append({"comparison": f"{a} vs {b}", "description": desc, "n_pairs": len(sub),
                      "mean_F1_diff (first - second)": diff.mean(),
                      "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                      "losses": int((diff < 0).sum()), "p_value": p})
    tdf = pd.DataFrame(trows)
    tdf["p_holm"] = holm_adjust(tdf["p_value"].values)
    tdf["significant_at_0.05"] = tdf["p_holm"] < 0.05
    tdf.to_csv(out_dir / "stat_tests.csv", index=False)
    log("\n=== PAIRED WILCOXON TESTS on test F1-macro (Holm-corrected) ===")
    for _, rr in tdf.iterrows():
        log(f"  {rr['comparison']}: diff={rr['mean_F1_diff (first - second)']:+.3f} "
            f"(W/T/L = {rr['wins']}/{rr['ties']}/{rr['losses']}), "
            f"p={rr['p_value']:.4f}, Holm p={rr['p_holm']:.4f} -> "
            f"{'significant' if rr['significant_at_0.05'] else 'not significant'}")
        log(f"      {rr['description']}")
    log(f"  (With only {n_ok} repeats the smallest possible p-value is limited; "
        f"more repeats = more power.)")

    # ------------------------- stability ------------------------------
    counts = np.zeros(n_genes, dtype=int)
    for S in selected_sets:
        counts[S] += 1
    freq_df = pd.DataFrame({"gene": gene_names, "times_selected": counts,
                            "selection_frequency": counts / n_ok,
                            "in_PAM50_signature": np.isin(gene_names, list(PAM50_GENES))})
    freq_df = freq_df[freq_df["times_selected"] > 0].sort_values(
        ["times_selected", "gene"], ascending=[False, True]).reset_index(drop=True)
    freq_df.to_csv(out_dir / "gene_selection_frequency.csv", index=False)

    jac = [len(set(a) & set(b)) / len(set(a) | set(b))
           for a, b in itertools.combinations(selected_sets, 2)]
    union_size = len(freq_df)
    stable50 = int((freq_df["selection_frequency"] >= 0.5).sum())

    log("\n=== STABILITY ===")
    log(f"  Genes selected per run: mean {np.mean(ks):.1f} (min {min(ks)}, max {max(ks)})")
    log(f"  Mean pairwise Jaccard similarity between runs: {np.mean(jac):.3f} "
        f"(1.0 = identical lists, 0 = no overlap)")
    log(f"  Distinct genes ever selected: {union_size} | selected in >=50% of runs: {stable50}")
    log("  Most frequently selected genes:")
    log(freq_df.head(TOP_N_PLOT).to_string(index=False))

    # PAM50 enrichment vs chance
    exp_per_run = np.mean([k_ * pam_in_data / n_genes for k_ in ks])
    obs = np.mean(pam_counts)
    pvals = [hypergeom.sf(x - 1, n_genes, pam_in_data, k_) if x > 0 else 1.0
             for x, k_ in zip(pam_counts, ks)]
    log("\n=== PAM50 OVERLAP vs CHANCE ===")
    log(f"  PAM50 genes per selected panel: observed mean {obs:.2f} vs expected by "
        f"chance {exp_per_run:.3f}  (~{obs / max(exp_per_run, 1e-9):.0f}x chance)")
    log(f"  Median per-run hypergeometric p-value: {np.median(pvals):.4f}")

    with open(out_dir / "stability_summary.txt", "w") as f:
        f.write(f"Usable repeats: {n_ok}/{N_REPEATS}\n")
        f.write(f"Genes per run: mean {np.mean(ks):.1f}, min {min(ks)}, max {max(ks)}\n")
        f.write(f"Mean pairwise Jaccard: {np.mean(jac):.3f}\n")
        f.write(f"Distinct genes ever selected: {union_size}; in >=50% of runs: {stable50}\n")
        f.write(f"PAM50 per panel: observed {obs:.2f} vs chance {exp_per_run:.3f}; "
                f"median per-run p={np.median(pvals):.4f}\n")

    # ------------------------- plots ----------------------------------
    order = ["A", "B", "C", "D"] + (["E"] if RUN_NO_SPARSITY_ABLATION else [])
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[df["method"] == m]["test_f1_macro"].values for m in order]
    labels = [f"{m}\n{METHODS[m]}" for m in order]
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)   # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)        # older matplotlib
    for i, d in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, alpha=0.5, s=18, color="k")
    ax.set_ylabel("Test F1-macro")
    ax.set_title(f"Matched-budget comparison over {n_ok} repeated splits")
    plt.xticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_f1_boxplot.png", dpi=150)
    plt.show()

    top = freq_df.head(TOP_N_PLOT)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 2))
    colors = ["tab:red" if p else "tab:blue" for p in top["in_PAM50_signature"]]
    ax.barh(range(len(top)), top["selection_frequency"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of runs in which the gene was selected")
    ax.set_title("Gene selection stability (red = PAM50 signature gene)")
    plt.tight_layout()
    plt.savefig(out_dir / "gene_stability.png", dpi=150)
    plt.show()

    log(f"\nSaved everything to '{out_dir}/'.")
    log("Next: repeat the pipeline on TCGA-OV, then write the report.")


if __name__ == "__main__":
    main()
