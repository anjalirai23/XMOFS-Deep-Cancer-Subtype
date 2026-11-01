"""
stage6_nsga2_search.py
-------------------------
Stage 6 of XMOFS-Deep: multi-objective search (NSGA-II) over the gated
deep network from Stage 5, to build a Pareto front instead of picking
one fixed lambda by hand.

WHAT NSGA-II SEARCHES OVER (the "genes" of each candidate solution, not
to be confused with the biological genes in the data!):
    1. log10(lambda_sparsity) - how strongly gates are pushed toward 0
    2. gate_lr_multiplier     - how fast the gate parameters learn
       (both explained in Stage 5's comments)

WHY NOT SEARCH OVER FULL NETWORK WEIGHTS:
    Evolving millions of network weights directly with a genetic
    algorithm is computationally infeasible. Instead (as flagged as a
    design choice in the project guide, Part B.3), each "individual" in
    the NSGA-II population is just a pair of hyperparameters; a full
    gated network is trained FROM SCRATCH with those hyperparameters
    (for a reduced number of epochs, for speed) and then evaluated. This
    is the standard, tractable way to combine deep learning with
    evolutionary multi-objective search on a student-project budget.

THE TWO OBJECTIVES (mirrors accuracy vs parsimony from the guide):
    Objective 1: maximize validation F1-macro   (implemented as minimize -F1)
    Objective 2: minimize number of selected genes (gate > 0.5)

OUTPUT: a Pareto front CSV (lambda, gate_lr, n_genes, val_f1, test_f1,
test_accuracy for each non-dominated solution) + a hypervolume score.
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# ============================== CONFIG ===================================
X_FILE = "cleaned_data/X_expression_clean.csv"
Y_FILE = "cleaned_data/y_pam50_labels.csv"
OUTPUT_DIR = "stage6_results"

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

HIDDEN_SIZES = [256, 64]     # smaller than Stage 5 - speed, since NSGA-II
                              # trains MANY networks, not just one
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
NETWORK_LR = 1e-3
GATE_SIGMA = 0.5
BATCH_SIZE = 32
GATE_SELECT_THRESHOLD = 0.5

# --- Per-individual training budget (kept small for tractability) ---
# On Colab GPU, you can raise these for a better final result.
INNER_MAX_EPOCHS = 40
INNER_PATIENCE = 8

# --- NSGA-II settings ---
# Start SMALL to make sure everything runs end-to-end without errors,
# then increase POP_SIZE / N_GENERATIONS on Colab GPU for a real result.
# Total networks trained = POP_SIZE * (N_GENERATIONS + 1) roughly.
POP_SIZE = 8
N_GENERATIONS = 6

# Search ranges for the two decision variables
LOG10_LAMBDA_RANGE = (-4.0, -0.5)   # lambda from 0.0001 to ~0.316
GATE_LR_MULT_RANGE = (0.01, 0.5)    # gate_lr = this * 1.0 (i.e. 0.01-0.5)

# Reference point for hypervolume: worst case = 0 F1, ALL genes selected
HV_REF_POINT = [1.0, 1.0]   # after normalizing both objectives to [0,1]
# ===========================================================================


def log(msg):
    print(f"[stage6_nsga2] {msg}")


# --------------------------------------------------------------------
# Data (loaded once, reused by every individual - only the model is
# retrained each time, not the data split)
# --------------------------------------------------------------------
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
    gene_names = X.columns.tolist()
    n_classes = len(le.classes_)
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            n_genes, n_classes, gene_names, le)


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
            eps = torch.randn_like(self.mu)
            z = self.mu + self.sigma * eps
        else:
            z = self.mu
        gate = torch.clamp(z, 0.0, 1.0)
        return x * gate, gate

    def regularization(self):
        normal = Normal(0.0, 1.0)
        return normal.cdf(self.mu / self.sigma).sum()

    def get_final_gates(self):
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
        return self.net(gated_x), gate_values


def train_and_evaluate_one_individual(lam, gate_lr, data, device, class_weights_tensor):
    """Trains ONE gated network with the given (lambda, gate_lr) and
    returns (val_f1_macro, n_selected_genes, trained_model)."""
    (X_train, y_train, X_val, y_val, X_test, y_test,
     n_genes, n_classes, gene_names, le) = data

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GatedDeepNet(n_genes, HIDDEN_SIZES, n_classes, DROPOUT, GATE_SIGMA).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": NETWORK_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.gate.parameters(), "lr": gate_lr, "weight_decay": 0.0},
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(INNER_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, _ = model(xb)
            loss = criterion(out, yb) + lam * model.gate.regularization()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, _ = model(xb)
                val_losses.append((criterion(out, yb) + lam * model.gate.regularization()).item())
            val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= INNER_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_val, dtype=torch.float32).to(device)
        out, _ = model(xb)
        val_pred = out.argmax(dim=1).cpu().numpy()
    val_f1 = f1_score(y_val, val_pred, average="macro")
    n_selected = (model.gate.get_final_gates() > GATE_SELECT_THRESHOLD).sum().item()

    return val_f1, n_selected, model


class GatedNetProblem(Problem):
    """pymoo Problem wrapper: each 'individual' = (log10_lambda, gate_lr_mult).
    Objectives (both minimized by pymoo convention):
        obj1 = -val_f1_macro   (so minimizing this = maximizing F1)
        obj2 = n_selected_genes / n_total_genes   (normalized feature count)
    """

    def __init__(self, data, device, class_weights_tensor, n_total_genes):
        super().__init__(n_var=2, n_obj=2, n_constr=0,
                          xl=np.array([LOG10_LAMBDA_RANGE[0], GATE_LR_MULT_RANGE[0]]),
                          xu=np.array([LOG10_LAMBDA_RANGE[1], GATE_LR_MULT_RANGE[1]]))
        self.data = data
        self.device = device
        self.class_weights_tensor = class_weights_tensor
        self.n_total_genes = n_total_genes
        self.eval_log = []  # keep a record of every individual ever evaluated

    def _evaluate(self, X, out, *args, **kwargs):
        objs = []
        for row in X:
            log10_lambda, gate_lr = row[0], row[1]
            lam = 10 ** log10_lambda
            val_f1, n_selected, _ = train_and_evaluate_one_individual(
                lam, gate_lr, self.data, self.device, self.class_weights_tensor)

            obj1 = -val_f1
            obj2 = n_selected / self.n_total_genes
            objs.append([obj1, obj2])

            self.eval_log.append({
                "lambda": lam, "gate_lr": gate_lr,
                "val_f1_macro": val_f1, "n_selected_genes": n_selected,
            })
            log(f"  individual: lambda={lam:.5f}, gate_lr={gate_lr:.3f} "
                f"-> val_f1={val_f1:.3f}, n_genes={n_selected}")

        out["F"] = np.array(objs)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    log("Loading data...")
    data = load_data()
    (X_train, y_train, X_val, y_val, X_test, y_test,
     n_genes, n_classes, gene_names, le) = data
    log(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}, "
        f"Total genes: {n_genes}")

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    problem = GatedNetProblem(data, device, class_weights_tensor, n_genes)

    algorithm = NSGA2(pop_size=POP_SIZE)

    log(f"Starting NSGA-II: pop_size={POP_SIZE}, generations={N_GENERATIONS} "
        f"(~{POP_SIZE * (N_GENERATIONS + 1)} networks will be trained - this "
        f"will take a while, especially on CPU)")

    result = minimize(problem, algorithm, ("n_gen", N_GENERATIONS),
                       seed=RANDOM_STATE, verbose=True)

    # -----------------------------------------------------------------
    # Save every individual ever evaluated (full search history)
    # -----------------------------------------------------------------
    all_evals_df = pd.DataFrame(problem.eval_log)
    all_evals_df.to_csv(out_dir / "all_individuals_evaluated.csv", index=False)
    log(f"\nSaved {len(all_evals_df)} total evaluated individuals to "
        f"'all_individuals_evaluated.csv'")

    # -----------------------------------------------------------------
    # Final Pareto front (non-dominated solutions from the last population)
    # Re-train each front solution once more and evaluate on the TEST set
    # for an honest final report (val set was used during search).
    # -----------------------------------------------------------------
    front_X = result.X
    if front_X.ndim == 1:
        front_X = front_X.reshape(1, -1)

    log(f"\nFinal Pareto front has {len(front_X)} solutions. "
        f"Re-training each on train set, evaluating on TEST set...")

    front_records = []
    for row in front_X:
        log10_lambda, gate_lr = row[0], row[1]
        lam = 10 ** log10_lambda
        val_f1, n_selected, model = train_and_evaluate_one_individual(
            lam, gate_lr, data, device, class_weights_tensor)

        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_test, dtype=torch.float32).to(device)
            out, _ = model(xb)
            test_pred = out.argmax(dim=1).cpu().numpy()
        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_acc = accuracy_score(y_test, test_pred)

        front_records.append({
            "lambda": lam, "gate_lr": gate_lr,
            "val_f1_macro": val_f1, "n_selected_genes": n_selected,
            "test_f1_macro": test_f1, "test_accuracy": test_acc,
        })
        log(f"  Front point: lambda={lam:.5f} -> {n_selected} genes, "
            f"test_f1={test_f1:.3f}, test_acc={test_acc:.3f}")

    front_df = pd.DataFrame(front_records).sort_values("n_selected_genes")
    front_df.to_csv(out_dir / "pareto_front.csv", index=False)

    # -----------------------------------------------------------------
    # Hypervolume of the front (using VALIDATION objectives, since
    # that's what NSGA-II actually optimized against)
    # -----------------------------------------------------------------
    F_normalized = result.F.copy()
    F_normalized[:, 0] = -F_normalized[:, 0]          # back to +F1 (higher=better)
    F_normalized[:, 0] = 1 - F_normalized[:, 0]        # convert to "loss" for HV (lower=better)
    # F_normalized[:,1] already in [0,1] (fraction of genes used)
    hv_indicator = HV(ref_point=np.array(HV_REF_POINT))
    hv_score = hv_indicator(F_normalized)

    log(f"\n{'='*60}")
    log(f"PARETO FRONT SUMMARY")
    log(f"{'='*60}")
    log(f"\n{front_df.to_string(index=False)}")
    log(f"\nHypervolume of this front (ref point {HV_REF_POINT}): {hv_score:.4f}")
    log(f"(Higher hypervolume = better front. Compare this number against "
        f"a shallow-model NSGA-II baseline's hypervolume to test your "
        f"core research claim - see Part C.1/C.2 of the project guide.)")

    with open(out_dir / "hypervolume.txt", "w") as f:
        f.write(f"Hypervolume: {hv_score:.4f}\n")
        f.write(f"Reference point: {HV_REF_POINT}\n")
        f.write(f"Number of Pareto-front solutions: {len(front_df)}\n")

    log(f"\nSaved Pareto front to 'pareto_front.csv', hypervolume to "
        f"'hypervolume.txt'")
    log(f"\nNext (Stage 7): compare this front's hypervolume against "
        f"Stage 3's shallow filter-based baseline evaluated at multiple "
        f"feature counts, and against a shallow-inner-model NSGA-II run "
        f"(swap GatedDeepNet's inner network for logistic regression/SVM).")


if __name__ == "__main__":
    main()
