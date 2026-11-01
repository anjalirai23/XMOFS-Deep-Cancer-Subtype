# XMOFS-Deep results: BRCA (profile: full)

## Stage 3 - filter-based baselines

| n_features | model | cv_f1_macro_mean | test_accuracy | test_f1_macro |
|---|---|---|---|---|
| 20 | RandomForest | 0.715 | 0.822 | 0.770 |
| 20 | SVM_RBF | 0.720 | 0.726 | 0.714 |
| 50 | RandomForest | 0.727 | 0.827 | 0.744 |
| 50 | SVM_RBF | 0.761 | 0.827 | 0.790 |
| 100 | RandomForest | 0.761 | 0.873 | 0.786 |
| 100 | SVM_RBF | 0.832 | 0.909 | 0.872 |
| 200 | RandomForest | 0.787 | 0.909 | 0.860 |
| 200 | SVM_RBF | 0.841 | 0.898 | 0.867 |
| 500 | RandomForest | 0.759 | 0.904 | 0.778 |
| 500 | SVM_RBF | 0.841 | 0.924 | 0.918 |

## Stage 6 - XMOFS-Deep Pareto front

| n_selected_genes | val_f1_macro | test_f1_macro | test_accuracy |
|---|---|---|---|
| 0 | 0.013 | 0.014 | 0.036 |
| 0 | 0.364 | 0.361 | 0.391 |
| 3 | 0.315 | 0.361 | 0.401 |
| 38 | 0.714 | 0.796 | 0.848 |
| 2136 | 0.836 | 0.823 | 0.873 |
| 2489 | 0.871 | 0.837 | 0.888 |
| 2842 | 0.773 | 0.828 | 0.888 |
| 3211 | 0.823 | 0.853 | 0.893 |

```
Hypervolume: 0.8567
Reference point: [1.0, 1.0]
Number of Pareto-front solutions: 8
```

## Stage 7 - hypervolume comparison

| method | hypervolume | n_front_points |
|---|---|---|
| Stage 3: Filter-based (fixed k) | 0.916 | 4 |
| Baseline #3: Shallow NSGA-II (LogReg) | 0.831 | 10 |
| XMOFS-Deep: Deep gated NSGA-II | 0.844 | 6 |

Cross-hypervolume (unique contribution of A beyond B):
```
Deep over Shallow: 0.0176
Shallow over Deep: 0.0044
Deep over Stage3:  0.0004
Stage3 over Deep:  0.0724

-- same comparison with log10(gene count) as objective 2 --
Deep over Shallow (log): 0.0499
Shallow over Deep (log): 0.0404
Deep over Stage3 (log): 0.1117
Stage3 over Deep (log): 0.0725
```

Robustness: hypervolume with log10(gene count) as the second objective:

| method | hypervolume_log |
|---|---|
| Stage 3: Filter-based (fixed k) | 0.6064 |
| Baseline #3: Shallow NSGA-II (LogReg) | 0.6361 |
| XMOFS-Deep: Deep gated NSGA-II | 0.6456 |

## Stage 9 - matched-budget ablation (mean over repeated splits)

| method | method_name | f1_mean | f1_std | acc_mean | acc_std | genes_mean |
|---|---|---|---|---|---|---|
| A | XMOFS-Deep gated net | 0.810 | 0.042 | 0.846 | 0.032 | 45.600 |
| B | SVM on XMOFS genes | 0.825 | 0.028 | 0.861 | 0.026 | 45.600 |
| C | SVM on top-k ANOVA genes | 0.794 | 0.043 | 0.818 | 0.026 | 45.600 |
| D | SVM on k random genes | 0.711 | 0.056 | 0.789 | 0.020 | 45.600 |
| E | Gated net, no sparsity (lambda=0) | 0.846 | 0.037 | 0.885 | 0.022 | 6810.500 |

## Stage 9 - paired Wilcoxon tests (Holm-corrected)

| comparison | mean_F1_diff (first - second) | wins | ties | losses | p_value | p_holm | significant_at_0.05 |
|---|---|---|---|---|---|---|---|
| A vs B | -0.0153 | 3 | 0 | 7 | 0.2324 | 0.2617 | False |
| B vs C | 0.0316 | 9 | 0 | 1 | 0.0098 | 0.0391 | True |
| B vs D | 0.1141 | 9 | 0 | 1 | 0.0039 | 0.0195 | True |
| A vs C | 0.0163 | 8 | 0 | 2 | 0.0840 | 0.2520 | False |
| A vs E | -0.0355 | 4 | 0 | 6 | 0.1309 | 0.2617 | False |

## Stage 9 - gene-selection stability
```
Usable repeats: 10/10
Genes per run: mean 45.6, min 38, max 55
Mean pairwise Jaccard: 0.009
Distinct genes ever selected: 422; in >=50% of runs: 0
PAM50 per panel: observed 1.90 vs chance 0.129; median per-run p=0.0095
```

## Pareto solution used by Stages 8 and 9
```
{
  "lambda": 0.0021726904421522,
  "gate_lr": 0.127370592870457,
  "n_selected_genes": 38,
  "val_f1_macro": 0.7139523809523809,
  "rule": "5-100 genes"
}
```