"""
clean_cbioportal_generic.py
-----------------------------
One cleaning script for BOTH TCGA-BRCA (PAM50) and TCGA-OV (transcriptional
subtypes), starting from files downloaded from cBioPortal.

WHAT IT NEEDS
  EXPRESSION_FILE : a cBioPortal expression matrix (rows = genes with a
                    Hugo_Symbol column, columns = samples like TCGA-XX-XXXX-01),
                    e.g. data_mrna_seq_v2_rsem.txt
  CLINICAL_FILES  : one or more cBioPortal clinical files
                    (data_clinical_patient.txt, data_clinical_sample.txt, ...).
                    The script SCANS ALL their columns and picks the one whose
                    values look like cancer-subtype names. For OV it looks for
                    Immunoreactive / Differentiated / Proliferative / Mesenchymal
                    (also short codes IMR, DIF, PRO, MES); for BRCA it looks for
                    PAM50 names (LumA, LumB, Her2, Basal, Normal).

IF NO LABEL COLUMN IS FOUND, the script stops and lists every column it
looked at. It never guesses a label from unrelated columns (ER/PR/HER2
status, stage, grade, survival are NOT subtypes).

WHAT IT DOES AUTOMATICALLY
  * matches samples to labels by the 12-character TCGA patient ID, and keeps
    one primary-tumour sample (barcode ending -01) per patient
  * detects the scale of the expression values and only applies log2(x+1)
    to raw/RSEM-like data (so z-scores or already-logged data are NOT
    transformed twice)
  * removes genes with many missing values / almost never expressed /
    lowest variance, and duplicate gene symbols
  * prints the class balance and warns about very small classes

OUTPUT (same format the stage scripts expect):
  <OUTPUT_PREFIX>X_expression_clean.csv   samples x genes
  <OUTPUT_PREFIX>y_subtype_labels.csv     index = sample ID, column "label"
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ============================== CONFIG ===================================
EXPRESSION_FILE = "data_mrna_seq_v2_rsem.txt"
CLINICAL_FILES = ["data_clinical_patient.txt", "data_clinical_sample.txt"]
OUTPUT_DIR = "cleaned_data_ov"
OUTPUT_PREFIX = "ov_"          # files become ov_X_expression_clean.csv, ov_y_subtype_labels.csv

# If you already know the exact label column, put it here, e.g. "SUBTYPE".
# None = detect automatically.
LABEL_COLUMN_OVERRIDE = None

MIN_FRACTION_EXPRESSED = 0.20      # keep genes > 0 in at least 20% of samples
VARIANCE_PERCENTILE_CUTOFF = 5     # drop the lowest 5% variance genes
MAX_MISSING_FRACTION = 0.20
MIN_CLASS_SIZE_WARNING = 15
# ===========================================================================


def norm(v):
    """lowercase and strip everything except letters/digits."""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


# normalised value -> canonical class name
LABEL_MAP = {}
for canon, aliases in {
    # ---- BRCA / PAM50 ----
    "Luminal A": ["luma", "brcaluma", "luminala"],
    "Luminal B": ["lumb", "brcalumb", "luminalb"],
    "HER2-enriched": ["her2", "brcaher2", "her2enriched", "her2e"],
    "Basal-like": ["basal", "brcabasal", "basallike"],
    "Normal-like": ["normal", "brcanormal", "normallike"],
    # ---- OV / TCGA transcriptional subtypes ----
    "Immunoreactive": ["immunoreactive", "imr", "imm", "c2imm", "ovimmunoreactive"],
    "Differentiated": ["differentiated", "dif", "c4dif", "ovdifferentiated"],
    "Proliferative": ["proliferative", "pro", "c5pro", "ovproliferative"],
    "Mesenchymal": ["mesenchymal", "mes", "c1mes", "ovmesenchymal"],
}.items():
    for a in aliases:
        LABEL_MAP[a] = canon

ID_LIKE = {"PATIENT_ID", "SAMPLE_ID", "OTHER_PATIENT_ID", "OTHER_SAMPLE_ID"}


def log(msg):
    print(f"[clean_generic] {msg}")


def read_cbioportal_txt(path):
    """cBioPortal clinical files start with '#' metadata rows - skip them."""
    skip = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False)


def patient_id(x):
    """TCGA-XX-XXXX-01 -> TCGA-XX-XXXX"""
    return str(x).strip()[:12]


def find_label_table(clinical_dfs):
    """Return a DataFrame [patient, label] built from the best label column."""
    best = None   # (score, file_idx, column)
    scanned = []
    for fi, df in enumerate(clinical_dfs):
        for col in df.columns:
            if col in ID_LIKE:
                continue
            if LABEL_COLUMN_OVERRIDE and col != LABEL_COLUMN_OVERRIDE:
                continue
            vals = df[col].dropna().astype(str)
            if len(vals) == 0:
                continue
            hit = vals.map(lambda v: norm(v) in LABEL_MAP)
            frac = float(hit.mean())
            n_classes = vals[hit].map(lambda v: LABEL_MAP[norm(v)]).nunique()
            scanned.append((fi, col, round(frac, 2), n_classes))
            # a real subtype column: most values recognised AND >= 3 classes
            if frac >= 0.5 and n_classes >= 3:
                score = frac * n_classes
                if best is None or score > best[0]:
                    best = (score, fi, col)

    if best is None:
        log("No subtype column found. Columns scanned (file#, column, "
            "fraction recognised, #classes):")
        for row in sorted(scanned, key=lambda r: -r[2])[:15]:
            log(f"   {row}")
        raise ValueError(
            "Could not find a subtype label column in the clinical files you gave. "
            "Do NOT substitute stage/grade/receptor status. Options: (1) download "
            "the clinical data of another cBioPortal study of the same cancer and "
            "add it to CLINICAL_FILES, (2) if you know the column name set "
            "LABEL_COLUMN_OVERRIDE, (3) compute subtypes with a published "
            "classifier (see the guide).")

    _, fi, col = best
    df = clinical_dfs[fi]
    id_col = "PATIENT_ID" if "PATIENT_ID" in df.columns else "SAMPLE_ID"
    lab = df[[id_col, col]].dropna()
    lab.columns = ["id", "raw"]
    lab = lab[lab["raw"].astype(str).map(lambda v: norm(v) in LABEL_MAP)]
    lab["label"] = lab["raw"].astype(str).map(lambda v: LABEL_MAP[norm(v)])
    lab["patient"] = lab["id"].map(patient_id)
    lab = lab.drop_duplicates(subset="patient")
    log(f"Label column: '{col}' (clinical file #{fi + 1}); {len(lab)} patients with a recognised label")
    log("Raw values found -> classes: " +
        ", ".join(f"{r} -> {LABEL_MAP[norm(r)]}" for r in sorted(lab['raw'].astype(str).unique())))
    return lab[["patient", "label"]]


def load_expression(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    df = df.dropna(subset=[gene_col])
    df[gene_col] = df[gene_col].astype(str).str.strip()
    df = df[df[gene_col] != ""]
    df = df.drop_duplicates(subset=gene_col).set_index(gene_col)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id", "Entrez_Gene_ID") if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"Expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def pick_primary_samples(sample_ids):
    """One sample per patient; prefer barcodes ending in -01 (primary tumour)."""
    s = pd.DataFrame({"sample": list(sample_ids)})
    s["patient"] = s["sample"].map(patient_id)
    s["primary"] = s["sample"].astype(str).str.endswith("-01")
    s = s.sort_values(["patient", "primary"], ascending=[True, False])
    return s.drop_duplicates(subset="patient", keep="first")


def clean_expression(df):
    n0 = df.shape[0]
    df = df.dropna(axis=0, how="all")
    df = df.loc[df.isna().mean(axis=1) <= MAX_MISSING_FRACTION]
    df = df.apply(lambda r: r.fillna(r.median()), axis=1)

    vmin, vmax = float(np.nanmin(df.values)), float(np.nanmax(df.values))
    if vmin < 0:
        mode = "centered"          # z-scores / log-ratios: leave untouched
    elif vmax > 50:
        mode = "raw"               # RSEM / counts-like: log2(x+1)
    else:
        mode = "logscale"          # already log-transformed
    log(f"Expression range [{vmin:.2f}, {vmax:.2f}] -> treated as '{mode}' data")

    if mode != "centered":
        df = df.loc[(df > 0).mean(axis=1) >= MIN_FRACTION_EXPRESSED]
    if mode == "raw":
        df = np.log2(df.clip(lower=0) + 1)
        log("Applied log2(x + 1)")
    else:
        log("No log transform applied (data are already on a log/centered scale)")

    var = df.var(axis=1)
    df = df.loc[var > np.percentile(var, VARIANCE_PERCENTILE_CUTOFF)]
    log(f"Gene filtering: {n0} -> {df.shape[0]} genes")
    return df


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    clinical = [read_cbioportal_txt(p) for p in CLINICAL_FILES]
    labels = find_label_table(clinical)

    expr = load_expression(EXPRESSION_FILE)
    prim = pick_primary_samples(expr.columns)
    merged = prim.merge(labels, on="patient", how="inner")
    log(f"Patients with BOTH expression and a label: {len(merged)} "
        f"(expression patients: {len(prim)}, labelled patients: {len(labels)})")
    if len(merged) < 50:
        raise ValueError("Fewer than 50 matched samples - check that the clinical "
                         "files and the expression file belong to the same cancer type.")

    expr = expr[merged["sample"].tolist()]
    expr = clean_expression(expr)
    y = pd.Series(merged["label"].values, index=merged["sample"].values, name="label")
    y = y.loc[expr.columns]

    counts = y.value_counts()
    log("Class balance after cleaning:")
    for k, v in counts.items():
        log(f"    {k}: {v}")
    if counts.min() < MIN_CLASS_SIZE_WARNING:
        log(f"WARNING: a class has fewer than {MIN_CLASS_SIZE_WARNING} samples - "
            "metrics for it will be unstable; report this as a limitation.")

    X = expr.T
    x_path = out_dir / f"{OUTPUT_PREFIX}X_expression_clean.csv"
    y_path = out_dir / f"{OUTPUT_PREFIX}y_subtype_labels.csv"
    X.to_csv(x_path)
    y.to_csv(y_path, header=["label"])
    log(f"Saved {x_path.name}: {X.shape[0]} samples x {X.shape[1]} genes")
    log(f"Saved {y_path.name}: {len(y)} labels")
    log("Done. Next: run xmofs_runner.py with DATASET = 'ov'.")


if __name__ == "__main__":
    main()
