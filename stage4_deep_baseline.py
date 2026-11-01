"""
stage4_deep_baseline.py
--------------------------
Stage 4 of XMOFS-Deep: plain deep neural network on ALL genes
(no feature gating, no feature selection at all).

This is baseline #4 from your project plan: "deep learning without
feature selection" - it isolates whether feature selection itself adds
value, separate from whether "going deep" helps.

Compare this script's test accuracy/F1 against the best Stage 3 result
(filter-based selection + SVM/RF, in stage3_results/baseline_results.csv) -
that comparison is what goes in your report/viva under
"baseline #1 vs baseline #4".
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage4_results"
STAGE3_RESULTS = ""   # optional: path to Stage 3 baseline_results.csv, used only for the comparison message at the end

TEST_SIZE = 0.20
VAL_SIZE = 0.15          # taken out of the training set, for early stopping
RANDOM_STATE = 42

HIDDEN_SIZES = [512, 128]   # two hidden layers
DROPOUT = 0.5              # aggressive dropout - needed given p >> n
WEIGHT_DECAY = 1e-3         # L2 regularization
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15    # stop if val loss doesn't improve for N epochs
# ===========================================================================


def log(msg):
    print(f"[stage4_deep_baseline] {msg}")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PlainDeepNet(nn.Module):
    """A standard feed-forward network - NO gating layer, uses every
    input feature. This is intentionally simple: the point of this
    baseline is to isolate "does going deep help" from "does feature
    selection help", so no fancy architecture tricks here."""

    def __init__(self, n_features, hidden_sizes, n_classes, dropout):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_preds)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # -----------------------------------------------------------------
    log("Loading data...")
    X = pd.read_csv(X_FILE, index_col=0)
    y = pd.read_csv(Y_FILE, index_col=0)["label"]
    y = y.loc[X.index]

    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        log(f"WARNING: {n_missing} missing values found - imputing with median.")
        X = X.apply(lambda col: col.fillna(col.median()), axis=0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    log(f"Classes: {list(le.classes_)}")

    # -----------------------------------------------------------------
    # Same train/test split as Stage 3 (same random_state, test_size)
    # so the comparison between baselines is apples-to-apples.
    # -----------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.values, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_trainval
    )
    log(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Scale using train-set statistics only (avoid leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(TabularDataset(X_train, y_train),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------------------
    # Class weights - Normal-like has very few samples, so without this
    # the model will likely just ignore that class entirely.
    # -----------------------------------------------------------------
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train),
                                          y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log(f"Class weights: {dict(zip(le.classes_, class_weights.cpu().numpy().round(2)))}")

    model = PlainDeepNet(n_features=X.shape[1], hidden_sizes=HIDDEN_SIZES,
                          n_classes=n_classes, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log(f"Model architecture:\n{model}")
    log(f"Total trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # -----------------------------------------------------------------
    # Training loop with early stopping
    # -----------------------------------------------------------------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_f1_macro": val_f1})

        if epoch % 10 == 0 or epoch == 1:
            log(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1_macro={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"Early stopping at epoch {epoch} "
                    f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # -----------------------------------------------------------------
    # Final evaluation on held-out test set
    # -----------------------------------------------------------------
    _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro")

    report = classification_report(test_true, test_pred, target_names=le.classes_)

    log(f"\n{'='*60}")
    log(f"FINAL TEST RESULTS - Plain Deep Net (no feature selection)")
    log(f"{'='*60}")
    log(f"Test accuracy: {test_acc:.4f}")
    log(f"Test F1-macro: {test_f1_macro:.4f}")
    log(f"\n{report}")

    best_s3 = None
    if STAGE3_RESULTS and Path(STAGE3_RESULTS).exists():
        s3 = pd.read_csv(STAGE3_RESULTS)
        best_s3 = float(s3["test_f1_macro"].max())
    if best_s3 is None:
        log("\n(No Stage 3 results file given - compare this F1 with your Stage 3 "
            "baseline manually.)")
    else:
        log(f"\nBest Stage 3 (filter-based, shallow) test F1-macro: {best_s3:.3f}")
        if test_f1_macro > best_s3:
            log("-> Plain deep net scores HIGHER than the best shallow filter-based baseline.")
        else:
            log("-> Plain deep net does NOT beat the best shallow filter-based baseline. "
                "Report this as it is: on small-sample, high-dimensional data, going deep "
                "alone is not enough, which motivates feature selection.")

    with open(out_dir / "test_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest F1-macro: {test_f1_macro:.4f}\n\n")
        f.write(report)

    torch.save(model.state_dict(), out_dir / "plain_deep_net.pt")
    log(f"\nSaved model, history, and report to '{out_dir}/'")


if __name__ == "__main__":
    main()
