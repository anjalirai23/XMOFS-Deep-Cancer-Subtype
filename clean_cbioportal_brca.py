"""
clean_cbioportal_brca.py
--------------------------
Cleans TCGA-BRCA data downloaded from cBioPortal's
"Breast Invasive Carcinoma (TCGA, PanCancer Atlas)" study
(brca_tcga_pan_can_atlas_2018), producing a PAM50-labeled,
model-ready expression matrix for XMOFS-Deep.

REQUIRED INPUT FILES (from the extracted tarball, unmodified):
    data_clinical_patient.txt   - has PATIENT_ID + SUBTYPE (PAM50)
    data_clinical_sample.txt    - maps SAMPLE_ID -> PATIENT_ID
    data_mrna_seq_v2_rsem.txt   - genes x samples RSEM expression matrix

Run: python clean_cbioportal_brca.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================== CONFIG ===================================
CLINICAL_PATIENT_FILE = "data_clinical_patient.txt"
CLINICAL_SAMPLE_FILE  = "data_clinical_sample.txt"
EXPRESSION_FILE       = "data_mrna_seq_v2_rsem.txt"
OUTPUT_DIR            = "cleaned_data"

VARIANCE_PERCENTILE_CUTOFF = 5     # drop bottom 5% lowest-variance genes
MIN_SAMPLES_EXPRESSED_FRACTION = 0.20

# cBioPortal's SUBTYPE values are prefixed with the study code, e.g.
# "BRCA_LumA". This map strips the prefix and standardizes naming.
SUBTYPE_CANONICAL_MAP = {
    "BRCA_LumA": "Luminal A", "BRCA_LumB": "Luminal B",
    "BRCA_Her2": "HER2-enriched", "BRCA_Basal": "Basal-like",
    "BRCA_Normal": "Normal-like",
    # also handle unprefixed variants just in case
    "LumA": "Luminal A", "LumB": "Luminal B", "Her2": "HER2-enriched",
    "Basal": "Basal-like", "Normal": "Normal-like",
}
# ===========================================================================


def log(msg):
    print(f"[clean_cbioportal_brca] {msg}")


def read_cbioportal_txt(path):
    """cBioPortal clinical files start with '#'-commented metadata rows
    before the real header - skip them automatically."""
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False)


def load_pam50_labels():
    patient_df = read_cbioportal_txt(CLINICAL_PATIENT_FILE)
    sample_df = read_cbioportal_txt(CLINICAL_SAMPLE_FILE)

    if "SUBTYPE" not in patient_df.columns:
        raise ValueError(
            f"No 'SUBTYPE' column found in {CLINICAL_PATIENT_FILE}. "
            f"Available columns: {list(patient_df.columns)}. "
            "Open the file and check the exact column name."
        )

    patient_df = patient_df[["PATIENT_ID", "SUBTYPE"]].dropna(subset=["SUBTYPE"])
    patient_df["SUBTYPE"] = patient_df["SUBTYPE"].astype(str).str.strip()

    before = len(patient_df)
    patient_df = patient_df[patient_df["SUBTYPE"].isin(SUBTYPE_CANONICAL_MAP)]
    log(f"Patients with a usable PAM50 SUBTYPE: {len(patient_df)} "
        f"(dropped {before - len(patient_df)} with missing/other values)")

    patient_df["label"] = patient_df["SUBTYPE"].map(SUBTYPE_CANONICAL_MAP)

    # Map PATIENT_ID -> SAMPLE_ID (a patient can have >1 sample; keep primary
    # tumor samples, which in TCGA barcodes end in "-01")
    sample_df = sample_df[["PATIENT_ID", "SAMPLE_ID"]]
    merged = patient_df.merge(sample_df, on="PATIENT_ID", how="left")
    merged = merged.dropna(subset=["SAMPLE_ID"])

    # Prefer primary tumor samples (barcode ends in -01) if a patient has
    # multiple samples listed
    merged["is_primary_tumor"] = merged["SAMPLE_ID"].str.endswith("-01")
    merged = merged.sort_values("is_primary_tumor", ascending=False)
    merged = merged.drop_duplicates(subset="PATIENT_ID", keep="first")

    log(f"Final patient->sample->label table: {len(merged)} rows")
    return merged[["SAMPLE_ID", "PATIENT_ID", "label"]]


def load_expression():
    df = pd.read_csv(EXPRESSION_FILE, sep="\t", low_memory=False)
    # First columns are typically Hugo_Symbol, Entrez_Gene_Id; rest are samples
    id_cols = [c for c in df.columns if c in ("Hugo_Symbol", "Entrez_Gene_Id")]
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else id_cols[0]

    df = df.dropna(subset=[gene_col])
    df = df.drop_duplicates(subset=gene_col)
    df = df.set_index(gene_col)
    df = df.drop(columns=[c for c in id_cols if c != gene_col], errors="ignore")

    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"Expression matrix loaded: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def clean_expression_matrix(expr_df):
    n_genes_before = expr_df.shape[0]

    missing_frac = expr_df.isna().mean(axis=1)
    expr_df = expr_df.loc[missing_frac <= 0.20]
    expr_df = expr_df.apply(lambda row: row.fillna(row.median()), axis=1)

    expressed_frac = (expr_df > 0).mean(axis=1)
    expr_df = expr_df.loc[expressed_frac >= MIN_SAMPLES_EXPRESSED_FRACTION]

    # RSEM values are raw-ish counts here (unlike the earlier dataset) ->
    # log2(x+1) transform is appropriate
    expr_df = np.log2(expr_df.clip(lower=0) + 1)

    variances = expr_df.var(axis=1)
    cutoff = np.percentile(variances, VARIANCE_PERCENTILE_CUTOFF)
    expr_df = expr_df.loc[variances > cutoff]

    log(f"Gene filtering: {n_genes_before} -> {expr_df.shape[0]} genes")
    return expr_df


def check_class_balance(labels_series):
    counts = labels_series.value_counts()
    log("PAM50 class balance after cleaning:")
    for cls, n in counts.items():
        log(f"    {cls}: {n} samples")
    if counts.min() < 10:
        log("WARNING: at least one subtype has fewer than 10 samples - "
            "use stratified CV carefully and report this as a limitation.")


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    labels_df = load_pam50_labels()
    expr_df = load_expression()

    common_samples = sorted(set(labels_df["SAMPLE_ID"]) & set(expr_df.columns))
    log(f"Samples present in BOTH expression matrix and PAM50 labels: {len(common_samples)}")
    if not common_samples:
        raise ValueError(
            "No overlap between expression sample IDs and clinical sample IDs. "
            "Print a few examples from both sides and compare formats manually."
        )

    expr_aligned = expr_df[common_samples]
    labels_aligned = labels_df.set_index("SAMPLE_ID").loc[common_samples, "label"]

    expr_clean = clean_expression_matrix(expr_aligned)
    labels_final = labels_aligned.loc[expr_clean.columns]

    check_class_balance(labels_final)

    X = expr_clean.T  # samples as rows, genes as columns
    X.to_csv(out_dir / "X_expression_clean.csv")
    labels_final.to_csv(out_dir / "y_pam50_labels.csv", header=["label"])

    log(f"Saved to '{out_dir}/': X = {X.shape[0]} samples x {X.shape[1]} genes, "
        f"y = {labels_final.shape[0]} PAM50 labels")
    log("Done. Next: stratified CV split + baseline models (Stage 3).")


if __name__ == "__main__":
    main()
