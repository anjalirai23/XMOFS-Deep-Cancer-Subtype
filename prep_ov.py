"""
prep_ov.py  -  prepare TCGA-OV data for the XMOFS-Deep runner
--------------------------------------------------------------
INPUT  (downloaded in the guide):
  EXPRESSION_FILE : cBioPortal ov_tcga_pan_can_atlas_2018 / data_mrna_seq_v2_rsem.txt
  LABEL_FILE      : TCGA_489_UE.k4.txt  (GDC, Verhaak et al. JCI 2013 page): the published
                    TCGA four-cluster assignment; columns "Sample ID" and "k4"
OUTPUT (the paths xmofs_runner.py expects for --dataset ov):
  cleaned_data_ov/ov_X_expression_clean.csv     samples x genes
  cleaned_data_ov/ov_y_subtype_labels.csv       index = sample ID, column "label"

THE CLEANING FILTERS ARE THE SAME AS THE BRCA CLEANER (clean_cbioportal_brca.py):
  genes with > 20% missing removed, median imputation, genes expressed (> 0) in < 20% of
  samples removed, log2(x + 1), lowest 5% variance removed, first of duplicate symbols kept.
  They are computed on the labelled samples only, exactly as for BRCA, so the two
  datasets are processed identically.

OPTIONAL DIAGNOSTIC (never used to drop samples or change labels):
  POS_SILHOUETTE_FILE lists the 413 tumours whose cluster assignment fitted well. The script
  only reports how many of your patients are on that list, per subtype.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ============================== CONFIG ===================================
EXPRESSION_FILE = "ov_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt"
LABEL_FILE = "TCGA_489_UE.k4.txt"
LABEL_ID_COL = "Sample ID"
LABEL_COL = "k4"
POS_SILHOUETTE_FILE = "TCGA_489_UE.k4.posSilhouette.txt"   # optional; "" to skip
OUTPUT_DIR = "cleaned_data_ov"
OUTPUT_PREFIX = "ov_"

MAX_MISSING_FRACTION = 0.20        # same as BRCA
MIN_FRACTION_EXPRESSED = 0.20      # same as BRCA
VARIANCE_PERCENTILE_CUTOFF = 5     # same as BRCA
MIN_PATIENTS = 100
# ===========================================================================


def log(msg):
    print(f"[prep_ov] {msg}")


def patient_id(x):
    """'TCGA.04.1331.01A.01R' or 'TCGA-04-1331-01' -> 'TCGA-04-1331'"""
    return str(x).strip().replace(".", "-")[:12]


def load_labels():
    lab = pd.read_csv(LABEL_FILE, sep="\t")
    for c in (LABEL_ID_COL, LABEL_COL):
        if c not in lab.columns:
            raise ValueError(f"Column '{c}' not found in {LABEL_FILE}. Columns: {list(lab.columns)}")
    lab = lab[[LABEL_ID_COL, LABEL_COL]].dropna()
    lab["patient"] = lab[LABEL_ID_COL].map(patient_id)
    lab = lab.rename(columns={LABEL_COL: "label"}).drop_duplicates(subset="patient")
    log(f"Labels: {len(lab)} patients; classes: {lab['label'].value_counts().to_dict()}")
    return lab[["patient", "label"]]


def load_expression():
    df = pd.read_csv(EXPRESSION_FILE, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    df = df.dropna(subset=[gene_col]).drop_duplicates(subset=gene_col).set_index(gene_col)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id", "Entrez_Gene_ID") if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"Expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def pick_primary_samples(sample_ids):
    """one sample per patient, preferring barcodes ending in -01 (primary tumour)"""
    s = pd.DataFrame({"sample": list(sample_ids)})
    s["patient"] = s["sample"].map(patient_id)
    s["primary"] = s["sample"].astype(str).str.endswith("-01")
    return s.sort_values(["patient", "primary"], ascending=[True, False]).drop_duplicates("patient")


def clean_expression(df):
    n0 = df.shape[0]
    df = df.loc[df.isna().mean(axis=1) <= MAX_MISSING_FRACTION]
    df = df.where(df.notna(), df.median(axis=1), axis=0)            # median imputation per gene
    df = df.loc[(df > 0).mean(axis=1) >= MIN_FRACTION_EXPRESSED]
    vmin, vmax = float(df.values.min()), float(df.values.max())
    if vmax > 50 or vmin < 0:
        df = np.log2(df.clip(lower=0) + 1)
        log(f"Applied log2(x + 1) (value range before: {vmin:.2f} to {vmax:.1f})")
    else:
        log(f"WARNING: values already look log-scaled (max {vmax:.1f}); log2 NOT applied")
    var = df.var(axis=1)
    df = df.loc[var > np.percentile(var, VARIANCE_PERCENTILE_CUTOFF)]
    log(f"Gene filtering: {n0} -> {df.shape[0]} genes")
    return df


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    expr = load_expression()
    prim = pick_primary_samples(expr.columns)
    merged = prim.merge(labels, on="patient", how="inner").sort_values("sample")
    log(f"Expression patients: {len(prim)} | labelled patients: {len(labels)} | in both: {len(merged)}")
    if len(merged) < MIN_PATIENTS:
        raise ValueError(f"Only {len(merged)} matched patients - check that the label file and the "
                         f"expression file are the right ones (expected a few hundred).")

    expr = clean_expression(expr[merged["sample"].tolist()])
    y = pd.Series(merged["label"].values, index=merged["sample"].values, name="label").loc[expr.columns]

    counts = y.value_counts()
    log("Class balance after cleaning:")
    for k, v in counts.items():
        log(f"    {k}: {v}")
    if counts.min() < 15:
        log("WARNING: a class has fewer than 15 samples; its metrics will be unstable.")

    X = expr.T
    X.to_csv(out_dir / f"{OUTPUT_PREFIX}X_expression_clean.csv")
    y.to_csv(out_dir / f"{OUTPUT_PREFIX}y_subtype_labels.csv", header=["label"])
    log(f"Saved {OUTPUT_PREFIX}X_expression_clean.csv: {X.shape[0]} samples x {X.shape[1]} genes")
    log(f"Saved {OUTPUT_PREFIX}y_subtype_labels.csv: {len(y)} labels")

    if POS_SILHOUETTE_FILE and Path(POS_SILHOUETTE_FILE).exists():
        pos = pd.read_csv(POS_SILHOUETTE_FILE, sep="\t")
        pos_ids = set(pos[LABEL_ID_COL].map(patient_id))
        flag = pd.Series([patient_id(s) in pos_ids for s in y.index], index=y.index, name="in_posSilhouette")
        flag.to_csv(out_dir / f"{OUTPUT_PREFIX}label_quality.csv", header=True)
        log("Diagnostic only (labels were NOT changed): patients on the well-fitting (posSilhouette) list:")
        log("\n" + pd.crosstab(y, flag).to_string())
    log("Done. Next: python xmofs_runner.py --dataset ov --profile quick")


if __name__ == "__main__":
    main()
