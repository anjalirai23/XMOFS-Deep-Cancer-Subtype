# XMOFS-Deep

**Explainable Multi-Objective Deep Feature Selection for Ovarian and Breast Cancer Subtype Classification**

Anjali Rai (UID 25MCI10038), MCA (AI & ML), Chandigarh University.

XMOFS-Deep learns a small gene panel inside a deep network with stochastic gates, searches the sparsity penalty and gate learning rate with NSGA-II to obtain a Pareto front of accuracy versus number of genes, and explains the selected genes with SHAP, LIME and pathway enrichment. It is applied to TCGA breast cancer (981 patients, 5 PAM50 subtypes) and ovarian cancer (291 patients, 4 TCGA subtypes) and compared with an ANOVA-filter + SVM/Random Forest, a plain deep network and a shallow NSGA-II (L1 logistic regression).

> Research prototype. Not a clinical diagnostic tool.

## Folder structure

| Folder | Content |
|---|---|
| `code/` | all pipeline scripts (stages 3 to 10), data preparation and the runner |
| `notebooks/` | Google Colab notebooks for BRCA and OV (GPU runtime) |
| `results/brca_run2/` | result tables, figures and `FINAL_SUMMARY.md` of the final breast cancer run |
| `results/ov_run3/` | the same for the final ovarian cancer run |

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1-2 | `clean_cbioportal_generic.py`, `prep_ov.py` | clean expression data and prepare subtype labels |
| 3 | `stage3_baseline.py` | ANOVA top-k + SVM / Random Forest |
| 4 | `stage4_deep_baseline.py` | plain deep network on all genes |
| 5 | `stage5_gated_deep_net.py` | gated deep network (stochastic gates) |
| 6 | `stage6_nsga2_search.py` | NSGA-II search, Pareto front |
| 7 | `stage7_hypervolume_comparison.py` | shallow NSGA-II baseline, hypervolume |
| 8 | `stage8_shap_explainability.py`, `stage8_enrichment_only.py` | SHAP and pathway enrichment |
| 9 | `stage9_stability_ablation_stats.py` | 10 repeated splits, ablation, Wilcoxon + Holm, stability |
| 10 | `stage10_journal_metrics.py` | precision/recall/F1/MCC/AUC, confusion matrix, ROC, learning curves, local SHAP and LIME |
| all | `xmofs_runner.py` | runs stages 3 to 9 for `brca` or `ov` |

## Data

The expression data are public and are **not included** in this repository.

- Breast cancer: TCGA PanCancer Atlas BRCA from cBioPortal; PAM50 labels from the clinical table.
- Ovarian cancer: TCGA PanCancer Atlas OV from cBioPortal; the four TCGA subtype labels from the Genomic Data Commons (TCGA 2011 clustering file).

## How to run (Google Colab, GPU)

1. Open a notebook from `notebooks/` in Colab (Runtime: T4 GPU) and upload the files of `code/` and your cleaned data.
2. Quick smoke test (numbers are not reportable):
   `python xmofs_runner.py --dataset brca --profile quick`
3. Full run:
   `python xmofs_runner.py --dataset brca --profile full --stages 3,4,6,7,8,9 --no-resume`
   (use `--dataset ov` for ovarian cancer; `prep_ov.py` first)
4. Extra metrics for the journal paper:
   `python stage10_journal_metrics.py --x <X csv> --y <y csv> --results <results folder>`

Training is stochastic and GPU non-deterministic, so repeated runs give slightly different Pareto fronts. The results in this repository are the final run of each cancer (BRCA run 2, OV run 3); the paper also reports an earlier run for comparison.

## Requirements

`pip install -r requirements.txt`
