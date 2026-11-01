"""
stage8_shap_explainability.py
--------------------------------
Stage 8 of XMOFS-Deep: explain WHICH genes the deep gated model relies on,
and for WHICH PAM50 subtype (SHAP), then check whether they make
biological sense (overlap with the PAM50 signature, optional pathway
enrichment).

WHAT THIS SCRIPT DOES
  1. Re-trains ONE chosen solution from your Stage 6 Pareto front
     (default: the 32-gene "balanced" solution, lambda=0.003564,
     gate_lr=0.121347) using the same settings as Stage 6.
     NOTE: training is stochastic, so the exact gene count / list can differ
     a little from your Stage 6 run. That is expected - report the numbers
     THIS script prints, not the old ones.
  2. Takes the genes whose gate > 0.5 as "selected genes".
  3. Builds a "reduced" network that only sees those genes (mathematically
     identical to the gated model when the other genes' gates are ~0) and
     CHECKS that its test performance matches the full gated model.
  4. Runs SHAP (GradientExplainer) on the reduced network over the TEST set.
  5. Saves: per-gene SHAP importance (overall and per subtype), plots,
     PAM50-signature overlap, per-subtype mean expression, and (optionally)
     pathway enrichment via Enrichr.

HOW TO READ SHAP HERE
  SHAP is computed on the network's output scores (logits) for each
  subtype. For a gene g and subtype k, |SHAP| tells how strongly g moves
  the score of subtype k, averaged over test patients. Large = important.
  SHAP shows what the MODEL uses; it does not prove biological causation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import matplotlib
if os.environ.get("MPLBACKEND") is None:
    pass
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage8_results"

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

# --- Which Pareto-front solution to explain (from Stage 6 pareto_front.csv) ---
# "Balanced" 32-gene solution. To explain another one, copy its lambda and
# gate_lr from your pareto_front.csv (e.g. the 423-gene one: 0.001209 / 0.216565)
LAMBDA = 0.003564
GATE_LR = 0.121347

# --- Same settings as Stage 6, so we reproduce that solution faithfully ---
HIDDEN_SIZES = [256, 64]
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
NETWORK_LR = 1e-3
GATE_SIGMA = 0.5
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 8
GATE_SELECT_THRESHOLD = 0.5

N_BACKGROUND = 100      # background samples for SHAP (taken from TRAIN set)
TOP_N_PLOT = 20         # how many genes to show in plots
RUN_ENRICHR = True      # optional pathway enrichment (needs internet + gseapy)
# ===========================================================================

# The 50 genes of the PAM50 signature (with a few common older aliases that
# may appear in TCGA RSEM files)
PAM50_GENES = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
    "TYMS", "UBE2C", "UBE2T",
    # aliases
    "ORC6L", "CDCA1", "KNTC2", "HER2",
}


def log(msg):
    print(f"[stage8_shap] {msg}")


# --------------------------------------------------------------------
# Model definitions (same as Stage 5/6)
# --------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StochasticGate(nn.Module):
    def __init__(self, n_features, sigma=0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((n_features,), 0.5))
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            z = self.mu + self.sigma * torch.randn_like(self.mu)
        else:
            z = self.mu
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


class ReducedNet(nn.Module):
    """Sees ONLY the selected genes. Uses the learned gate values and the
    matching columns of the first layer, so it reproduces the full gated
    model (assuming non-selected genes have ~0 gate). This is the model
    SHAP explains - a 32-input network is tiny and fast to explain."""

    def __init__(self, gated_model, selected_idx):
        super().__init__()
        idx = torch.tensor(selected_idx, dtype=torch.long)
        gates = gated_model.gate.get_final_gates()[idx].clone()
        self.register_buffer("gates", gates)
        first = gated_model.net[0]
        new_first = nn.Linear(len(selected_idx), first.out_features)
        with torch.no_grad():
            new_first.weight.copy_(first.weight[:, idx])
            new_first.bias.copy_(first.bias)
        rest = list(gated_model.net.children())[1:]
        self.net = nn.Sequential(new_first, *rest)

    def forward(self, x):
        return self.net(x * self.gates)


def train_gated_model(X_train, y_train, X_val, y_val, n_genes, n_classes, device, cw):
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GatedDeepNet(n_genes, HIDDEN_SIZES, n_classes, DROPOUT, GATE_SIGMA).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": NETWORK_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.gate.parameters(), "lr": GATE_LR, "weight_decay": 0.0},
    ])
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, _ = model(xb)
            loss = criterion(out, yb) + LAMBDA * model.gate.regularization()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, _ = model(xb)
                vl.append((criterion(out, yb) + LAMBDA * model.gate.regularization()).item())
            val_loss = float(np.mean(vl))

        n_active = int((model.gate.get_final_gates() > GATE_SELECT_THRESHOLD).sum().item())
        if epoch == 1 or epoch % 5 == 0:
            log(f"  epoch {epoch:3d} | val_loss={val_loss:.4f} | active_genes={n_active}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                log(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, device, gated=True):
    xb = torch.tensor(X, dtype=torch.float32).to(device)
    out = model(xb)
    if gated:
        out = out[0]
    return out.argmax(dim=1).cpu().numpy()


def shap_to_array(sv, n_classes):
    """Normalise SHAP output to shape (n_classes, n_samples, n_features),
    whatever the installed shap version returns."""
    if isinstance(sv, list):
        return np.stack([np.asarray(s) for s in sv], axis=0)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[-1] == n_classes:
        return np.transpose(sv, (2, 0, 1))
    if sv.ndim == 3 and sv.shape[0] == n_classes:
        return sv
    raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # ----------------------------- data ------------------------------
    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"].loc[X.index]
    if X.isna().sum().sum() > 0:
        X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    gene_names = np.array(X.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X.values, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tv)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    log(f"Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}, genes {X.shape[1]}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = torch.tensor(cw, dtype=torch.float32).to(device)

    # --------------------- 1) re-train chosen solution ----------------
    log(f"Re-training chosen Pareto solution: lambda={LAMBDA}, gate_lr={GATE_LR}")
    model = train_gated_model(X_train, y_train, X_val, y_val, X.shape[1], n_classes, device, cw)

    gates = model.gate.get_final_gates().cpu().numpy()
    sel_idx = np.where(gates > GATE_SELECT_THRESHOLD)[0]
    n_sel = len(sel_idx)
    log(f"Selected genes (gate > {GATE_SELECT_THRESHOLD}): {n_sel}")
    if n_sel == 0:
        raise RuntimeError("No genes selected - try a smaller LAMBDA or more epochs.")
    sel_genes = gene_names[sel_idx]

    # --------------- 2) check full gated vs reduced network ------------
    full_pred = predict(model, X_test, device, gated=True)
    full_acc = accuracy_score(y_test, full_pred)
    full_f1 = f1_score(y_test, full_pred, average="macro")

    reduced = ReducedNet(model.cpu(), sel_idx).eval()
    with torch.no_grad():
        red_pred = reduced(torch.tensor(X_test[:, sel_idx], dtype=torch.float32)).argmax(1).numpy()
    red_acc = accuracy_score(y_test, red_pred)
    red_f1 = f1_score(y_test, red_pred, average="macro")

    log(f"Full gated model   : test acc={full_acc:.3f}, F1-macro={full_f1:.3f}")
    log(f"Reduced ({n_sel}-gene) net: test acc={red_acc:.3f}, F1-macro={red_f1:.3f}")
    if abs(full_acc - red_acc) > 0.03:
        log("NOTE: the reduced net differs from the full model by >3% accuracy. "
            "Some genes with small non-zero gates (<0.5) were dropped. SHAP "
            "below explains the REDUCED net, so report its numbers.")
    log("\nClassification report (reduced network, test set):\n" +
        classification_report(y_test, red_pred, target_names=class_names, zero_division=0))

    # ------------------------------ 3) SHAP ---------------------------
    log("Running SHAP (GradientExplainer) on the test set...")
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
    background = torch.tensor(X_train[bg_idx][:, sel_idx], dtype=torch.float32)
    X_explain = torch.tensor(X_test[:, sel_idx], dtype=torch.float32)

    explainer = shap.GradientExplainer(reduced, background)
    sv = explainer.shap_values(X_explain)
    sv = shap_to_array(sv, n_classes)          # (classes, samples, genes)
    log(f"SHAP array shape (classes, samples, genes): {sv.shape}")

    mean_abs_per_class = np.abs(sv).mean(axis=1)        # (classes, genes)
    overall = mean_abs_per_class.sum(axis=0)             # (genes,)

    # --------------- 4) per-subtype mean expression (z-scored) --------
    Xsel_all = np.vstack([X_train, X_val, X_test])[:, sel_idx]
    y_all = np.concatenate([y_train, y_val, y_test])
    mean_expr = np.vstack([Xsel_all[y_all == k].mean(axis=0) for k in range(n_classes)])  # (classes, genes)

    # ------------------------- 5) results table -----------------------
    df = pd.DataFrame({"gene": sel_genes, "gate_value": gates[sel_idx],
                       "mean_abs_shap_overall": overall})
    for k, cname in enumerate(class_names):
        df[f"shap_{cname}"] = mean_abs_per_class[k]
    for k, cname in enumerate(class_names):
        df[f"meanZ_{cname}"] = mean_expr[k]
    df["in_PAM50_signature"] = df["gene"].isin(PAM50_GENES)
    df["top_subtype_by_shap"] = [class_names[i] for i in mean_abs_per_class.argmax(axis=0)]
    df = df.sort_values("mean_abs_shap_overall", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "selected_genes_shap.csv", index=False)

    log("\nTop genes by overall SHAP importance:")
    log(df[["gene", "gate_value", "mean_abs_shap_overall", "top_subtype_by_shap",
            "in_PAM50_signature"]].head(TOP_N_PLOT).to_string(index=False))

    # ----------------------- 6) PAM50 overlap -------------------------
    pam_hits = df[df["in_PAM50_signature"]]
    n_pam_total = len(PAM50_GENES - {"ORC6L", "CDCA1", "KNTC2", "HER2"})
    in_data_pam = len(set(gene_names) & PAM50_GENES)
    log(f"\nPAM50 signature genes found among the {n_sel} selected genes: {len(pam_hits)}")
    log(f"(PAM50 genes present anywhere in your cleaned data: {in_data_pam} of ~{n_pam_total})")
    with open(out_dir / "pam50_overlap.txt", "w") as f:
        f.write(f"Selected genes: {n_sel}\n")
        f.write(f"PAM50 genes among selected: {len(pam_hits)}\n")
        f.write(", ".join(pam_hits["gene"].tolist()) + "\n")
        f.write(f"PAM50 genes present in cleaned data: {in_data_pam}\n")

    # Top-5 genes per subtype
    with open(out_dir / "top_genes_per_subtype.txt", "w") as f:
        for k, cname in enumerate(class_names):
            order = np.argsort(-mean_abs_per_class[k])[:5]
            line = f"{cname}: " + ", ".join(f"{sel_genes[i]} ({mean_abs_per_class[k][i]:.3f})" for i in order)
            log(line)
            f.write(line + "\n")

    # ---------------------------- 7) plots ----------------------------
    top = df.head(TOP_N_PLOT)
    order_genes = top["gene"].tolist()
    gene_pos = {g: i for i, g in enumerate(sel_genes)}
    cmap = plt.get_cmap("tab10")

    # (a) stacked bar: overall importance split by subtype
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(order_genes) + 2))
    left = np.zeros(len(order_genes))
    for k, cname in enumerate(class_names):
        vals = np.array([mean_abs_per_class[k][gene_pos[g]] for g in order_genes])
        ax.barh(range(len(order_genes)), vals, left=left, color=cmap(k), label=cname)
        left += vals
    ax.set_yticks(range(len(order_genes)))
    ax.set_yticklabels(order_genes)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (sum over subtypes)")
    ax.set_title(f"XMOFS-Deep: top {len(order_genes)} genes by SHAP importance")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_importance.png", dpi=150)
    plt.show()

    # (b) heatmap: SHAP importance gene x subtype
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(order_genes) + 2))
    M = np.array([[mean_abs_per_class[k][gene_pos[g]] for k in range(n_classes)] for g in order_genes])
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(order_genes))); ax.set_yticklabels(order_genes)
    ax.set_title("Mean |SHAP| per gene and subtype")
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_heatmap_gene_by_subtype.png", dpi=150)
    plt.show()

    # (c) heatmap: mean standardized expression per subtype (direction of effect)
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(order_genes) + 2))
    E = np.array([[mean_expr[k][gene_pos[g]] for k in range(n_classes)] for g in order_genes])
    lim = np.abs(E).max()
    im = ax.imshow(E, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(order_genes))); ax.set_yticklabels(order_genes)
    ax.set_title("Mean standardized expression per subtype\n(red = higher, blue = lower than average)")
    plt.colorbar(im, ax=ax, label="Mean z-score")
    plt.tight_layout()
    plt.savefig(out_dir / "expression_heatmap_gene_by_subtype.png", dpi=150)
    plt.show()

    # ------------------- 8) optional pathway enrichment ---------------
    if RUN_ENRICHR:
        try:
            import gseapy as gp
            libs = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2021"]
            rows = []
            for lib in libs:
                try:
                    enr = gp.enrichr(gene_list=list(sel_genes), gene_sets=lib,
                                     organism="human", outdir=None, cutoff=0.5)
                    r = enr.results.head(10).copy()
                    r["library"] = lib
                    rows.append(r)
                except Exception as e:
                    log(f"  Enrichr library {lib} skipped: {e}")
            if rows:
                res = pd.concat(rows)[["library", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
                res.to_csv(out_dir / "pathway_enrichment.csv", index=False)
                log("\nTop enriched pathways (check Adjusted P-value!):")
                log(res.head(10).to_string(index=False))
        except ImportError:
            log("gseapy not installed - skipping pathway enrichment. "
                "(Optional: run '!pip install gseapy' and re-run.)")
        except Exception as e:
            log(f"Pathway enrichment skipped ({e}).")

    with open(out_dir / "run_summary.txt", "w") as f:
        f.write(f"lambda={LAMBDA}, gate_lr={GATE_LR}\n")
        f.write(f"Selected genes: {n_sel}\n")
        f.write(f"Full gated model test acc={full_acc:.4f}, F1-macro={full_f1:.4f}\n")
        f.write(f"Reduced net test acc={red_acc:.4f}, F1-macro={red_f1:.4f}\n")
        f.write(f"PAM50 genes among selected: {len(pam_hits)}\n")

    log(f"\nSaved everything to '{out_dir}/'.")
    log("Next (Stage 9): ablations + statistical tests, then repeat on TCGA-OV.")


if __name__ == "__main__":
    main()
