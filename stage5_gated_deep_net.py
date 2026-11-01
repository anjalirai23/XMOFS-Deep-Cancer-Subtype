"""
stage5_gated_deep_net.py
---------------------------
Stage 5 of XMOFS-Deep: deep neural network WITH a feature-gating layer,
using a single FIXED sparsity penalty (lambda). This is baseline #5 from
your project plan: "deep feature selection without multi-objective
optimization" - the direct predecessor to the full NSGA-II search you'll
add in Stage 6.

GATING METHOD: Stochastic Gates (STG), based on Yamada et al.,
"Feature Selection using Stochastic Gates" (ICML 2020). Each feature i
gets a learnable parameter mu_i. During training, Gaussian noise is
added and the result is clamped to [0,1] to form a "soft" gate - this
approximates discrete 0/1 selection while staying differentiable.
At evaluation time, the gate is just clamp(mu_i, 0, 1) (no noise).

The gate's contribution to the loss (the regularization term) is the
probability that the gate is "active" (> 0), computed via the Gaussian
CDF - summed across all features and scaled by LAMBDA_SPARSITY. A
larger LAMBDA_SPARSITY pushes more genes toward a gate of 0 (i.e. fewer
selected genes) - this exact knob is what Stage 6's NSGA-II will search
over automatically instead of you picking one fixed value by hand.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage5_results"

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

HIDDEN_SIZES = [512, 128]
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3      # for the classifier network weights
GATE_LEARNING_RATE = 0.05  # gates need a MUCH higher LR to converge in
                            # reasonable time when there are thousands of
                            # features - standard practice for STG-style
                            # gating (each gate only gets a tiny gradient
                            # signal per feature, so a small shared LR
                            # would take far too many epochs to separate
                            # useful genes from useless ones)
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 20

GATE_SIGMA = 0.5          # noise std-dev for stochastic gates (STG default)
LAMBDA_SPARSITY = 0.01    # <-- THE key knob: higher = fewer genes selected
                           # Stage 6 (NSGA-II) will search a RANGE of these
GATE_SELECT_THRESHOLD = 0.5   # a gene is "selected" if its final gate > this
# ===========================================================================


def log(msg):
    print(f"[stage5_gated_deep_net] {msg}")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StochasticGate(nn.Module):
    """One learnable gate per input feature. mu_i near 1 -> feature kept,
    mu_i near 0 -> feature dropped. Training adds noise (stochastic,
    for a smooth gradient signal); evaluation is deterministic."""

    def __init__(self, n_features, sigma=0.5):
        super().__init__()
        # Initialize near 0.5 so every feature starts "half open" -
        # unbiased starting point, let training decide.
        self.mu = nn.Parameter(torch.full((n_features,), 0.5))
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            eps = torch.randn_like(self.mu)
            z = self.mu + self.sigma * eps
        else:
            z = self.mu
        gate = torch.clamp(z, 0.0, 1.0)
        return x * gate, gate

    def regularization(self):
        # P(gate > 0) via Gaussian CDF - this is the "expected number of
        # active features" that gradient descent tries to shrink.
        normal = Normal(0.0, 1.0)
        return normal.cdf(self.mu / self.sigma).sum()

    def get_final_gates(self):
        # Deterministic gate values, for reporting which genes were selected.
        return torch.clamp(self.mu, 0.0, 1.0).detach()


class GatedDeepNet(nn.Module):
    def __init__(self, n_features, hidden_sizes, n_classes, dropout, gate_sigma):
        super().__init__()
        self.gate = StochasticGate(n_features, sigma=gate_sigma)

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        gated_x, gate_values = self.gate(x)
        out = self.net(gated_x)
        return out, gate_values


def train_one_epoch(model, loader, optimizer, criterion, lam, device):
    model.train()
    total_loss, total_ce, total_reg = 0.0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out, _ = model(xb)
        ce_loss = criterion(out, yb)
        reg_loss = model.gate.regularization()
        loss = ce_loss + lam * reg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_ce += ce_loss.item() * xb.size(0)
        total_reg += reg_loss.item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_ce / n, total_reg / n


@torch.no_grad()
def evaluate(model, loader, criterion, lam, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out, _ = model(xb)
        ce_loss = criterion(out, yb)
        reg_loss = model.gate.regularization()
        loss = ce_loss + lam * reg_loss
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_true), np.array(all_preds)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    log("Loading data...")
    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"]
    y = y.loc[X.index]
    gene_names = X.columns.tolist()

    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        log(f"WARNING: {n_missing} missing values - imputing with median.")
        X = X.apply(lambda col: col.fillna(col.median()), axis=0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.values, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_trainval)
    log(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = GatedDeepNet(n_features=X.shape[1], hidden_sizes=HIDDEN_SIZES,
                          n_classes=n_classes, dropout=DROPOUT,
                          gate_sigma=GATE_SIGMA).to(device)
    gate_params = list(model.gate.parameters())
    network_params = list(model.net.parameters())
    optimizer = torch.optim.Adam([
        {"params": network_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": gate_params, "lr": GATE_LEARNING_RATE, "weight_decay": 0.0},
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log(f"LAMBDA_SPARSITY = {LAMBDA_SPARSITY}")
    log(f"Total genes available for gating: {X.shape[1]}")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_ce, train_reg = train_one_epoch(
            model, train_loader, optimizer, criterion, LAMBDA_SPARSITY, device)
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, LAMBDA_SPARSITY, device)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        n_active_genes = (model.gate.get_final_gates() > GATE_SELECT_THRESHOLD).sum().item()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_ce": train_ce,
                         "train_reg": train_reg, "val_loss": val_loss,
                         "val_f1_macro": val_f1, "n_active_genes": n_active_genes})

        if epoch % 10 == 0 or epoch == 1:
            log(f"Epoch {epoch:3d} | train_ce={train_ce:.4f} | reg={train_reg:.1f} | "
                f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                f"active_genes={n_active_genes}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # -----------------------------------------------------------------
    # Final evaluation + which genes got selected
    # -----------------------------------------------------------------
    _, test_true, test_pred = evaluate(model, test_loader, criterion, LAMBDA_SPARSITY, device)
    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro")
    report = classification_report(test_true, test_pred, target_names=le.classes_)

    final_gates = model.gate.get_final_gates().cpu().numpy()
    gate_df = pd.DataFrame({"gene": gene_names, "gate_value": final_gates})
    gate_df = gate_df.sort_values("gate_value", ascending=False)
    selected_genes = gate_df[gate_df["gate_value"] > GATE_SELECT_THRESHOLD]

    log(f"\n{'='*60}")
    log(f"FINAL TEST RESULTS - Gated Deep Net (lambda={LAMBDA_SPARSITY})")
    log(f"{'='*60}")
    log(f"Test accuracy: {test_acc:.4f}")
    log(f"Test F1-macro: {test_f1_macro:.4f}")
    log(f"Genes selected (gate > {GATE_SELECT_THRESHOLD}): {len(selected_genes)} "
        f"out of {X.shape[1]}")
    log(f"\nTop 15 selected genes by gate value:\n{selected_genes.head(15).to_string(index=False)}")
    log(f"\n{report}")

    log(f"\nCompare to:")
    log(f"  Stage 3 (SVM + 500 genes, filter-based): accuracy=0.924, f1_macro=0.918")
    log(f"  Stage 4 (Plain deep net, all 17677 genes): accuracy=0.888, f1_macro=0.858")
    log(f"  Stage 5 (Gated deep net, {len(selected_genes)} genes): "
        f"accuracy={test_acc:.3f}, f1_macro={test_f1_macro:.3f}")

    gate_df.to_csv(out_dir / "gene_gate_values.csv", index=False)
    selected_genes.to_csv(out_dir / "selected_genes.csv", index=False)
    with open(out_dir / "test_report.txt", "w") as f:
        f.write(f"lambda_sparsity={LAMBDA_SPARSITY}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\nTest F1-macro: {test_f1_macro:.4f}\n")
        f.write(f"Genes selected: {len(selected_genes)} / {X.shape[1]}\n\n")
        f.write(report)
    torch.save(model.state_dict(), out_dir / "gated_deep_net.pt")

    log(f"\nSaved model, gate values, and report to '{out_dir}/'")
    log(f"\nNext (Stage 6): sweep/search LAMBDA_SPARSITY with NSGA-II instead "
        f"of this one fixed value, to build the full Pareto front.")


if __name__ == "__main__":
    main()
