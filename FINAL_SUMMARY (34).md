# XMOFS-Deep results: OV (profile: full)

## Stage 3 - filter-based baselines

| n_features | model | cv_f1_macro_mean | test_accuracy | test_f1_macro |
|---|---|---|---|---|
| 20 | RandomForest | 0.725 | 0.610 | 0.611 |
| 20 | SVM_RBF | 0.731 | 0.644 | 0.649 |
| 50 | RandomForest | 0.746 | 0.661 | 0.655 |
| 50 | SVM_RBF | 0.733 | 0.695 | 0.696 |
| 100 | RandomForest | 0.781 | 0.661 | 0.649 |
| 100 | SVM_RBF | 0.754 | 0.661 | 0.665 |
| 200 | RandomForest | 0.797 | 0.746 | 0.740 |
| 200 | SVM_RBF | 0.772 | 0.746 | 0.743 |
| 500 | RandomForest | 0.864 | 0.814 | 0.812 |
| 500 | SVM_RBF | 0.837 | 0.814 | 0.812 |

## Stage 6 - XMOFS-Deep Pareto front

| n_selected_genes | val_f1_macro | test_f1_macro | test_accuracy |
|---|---|---|---|
| 0 | 0.111 | 0.107 | 0.271 |
| 0 | 0.247 | 0.098 | 0.119 |
| 1 | 0.216 | 0.225 | 0.339 |
| 2 | 0.424 | 0.449 | 0.458 |
| 4 | 0.600 | 0.471 | 0.559 |
| 17 | 0.829 | 0.773 | 0.780 |
| 18 | 0.714 | 0.797 | 0.797 |
| 940 | 0.888 | 0.810 | 0.814 |

```
Hypervolume: 0.9357
Reference point: [1.0, 1.0]
Number of Pareto-front solutions: 8
```

## Stage 7 - hypervolume comparison

| method | hypervolume | n_front_points |
|---|---|---|
| Stage 3: Filter-based (fixed k) | 0.809 | 4 |
| Baseline #3: Shallow NSGA-II (LogReg) | 0.862 | 10 |
| XMOFS-Deep: Deep gated NSGA-II | 0.809 | 7 |

Cross-hypervolume (unique contribution of A beyond B):
```
Deep over Shallow: 0.0001
Shallow over Deep: 0.0528
Deep over Stage3:  0.0026
Stage3 over Deep:  0.0022
```

## Stage 9 - matched-budget ablation (mean over repeated splits)

| method | method_name | f1_mean | f1_std | acc_mean | acc_std | genes_mean |
|---|---|---|---|---|---|---|
| A | XMOFS-Deep gated net | 0.781 | 0.035 | 0.785 | 0.037 | 20.100 |
| B | SVM on XMOFS genes | 0.759 | 0.041 | 0.764 | 0.041 | 20.100 |
| C | SVM on top-k ANOVA genes | 0.713 | 0.071 | 0.719 | 0.075 | 20.100 |
| D | SVM on k random genes | 0.604 | 0.084 | 0.612 | 0.083 | 20.100 |
| E | Gated net, no sparsity (lambda=0) | 0.846 | 0.045 | 0.847 | 0.044 | 8126.000 |

## Stage 9 - paired Wilcoxon tests (Holm-corrected)

| comparison | mean_F1_diff (first - second) | wins | ties | losses | p_value | p_holm | significant_at_0.05 |
|---|---|---|---|---|---|---|---|
| A vs B | 0.0219 | 9 | 0 | 1 | 0.0059 | 0.0176 | True |
| B vs C | 0.0460 | 8 | 0 | 2 | 0.0645 | 0.0645 | False |
| B vs D | 0.1551 | 10 | 0 | 0 | 0.0020 | 0.0098 | True |
| A vs C | 0.0679 | 8 | 0 | 2 | 0.0137 | 0.0273 | True |
| A vs E | -0.0651 | 1 | 0 | 9 | 0.0039 | 0.0156 | True |

## Stage 9 - gene-selection stability
```
Usable repeats: 10/10
Genes per run: mean 20.1, min 16, max 27
Mean pairwise Jaccard: 0.002
Distinct genes ever selected: 197; in >=50% of runs: 0
PAM50 per panel: observed 0.00 vs chance 0.000; median per-run p=1.0000
```

## Pareto solution used by Stages 8 and 9
```
{
  "lambda": 0.003889976925535,
  "gate_lr": 0.1156308973241454,
  "n_selected_genes": 17,
  "val_f1_macro": 0.829177807486631,
  "rule": "5-100 genes"
}
```