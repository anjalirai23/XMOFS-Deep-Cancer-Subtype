"""
xmofs_runner.py  -  ONE entry point for the whole XMOFS-Deep pipeline
----------------------------------------------------------------------
Runs the verified stage scripts in order, on the dataset you choose
("brca" or "ov"), with every path set from ONE place and every result
written under  results_<dataset>/ .

STAGES  (the scripts must be in SCRIPTS_DIR - upload them next to this file)
  3  stage3_baseline.py                   classical filter + RF/SVM baseline
  4  stage4_deep_baseline.py              plain deep net on all genes
  6  stage6_nsga2_search.py               NSGA-II over the gated deep net  -> Pareto front
  7  stage7_hypervolume_comparison.py     shallow NSGA-II + hypervolume / cross-hypervolume
  8  stage8_shap_explainability.py        SHAP + enrichment for one front solution
  9  stage9_stability_ablation_stats.py   repeats, ablations, Wilcoxon tests, stability
(Stage 5 is skipped: Stage 6 contains it as a special case.)

HOW THE LINK BETWEEN STAGES WORKS
  After Stage 6, the runner picks ONE "balanced" Pareto solution automatically
  and passes its lambda / gate_lr to Stages 8 and 9. Rule (uses VALIDATION F1
  only, never the test set): among front solutions with min_genes..max_genes
  genes, take the one with the highest validation F1-macro (ties -> fewer genes).
  The choice is saved in results_<dataset>/chosen_point.json.

PROFILES
  "quick": tiny budgets, only to check that everything runs (minutes).
           NEVER report quick-profile numbers.
  "full" : the real settings used in the project.

USAGE (Colab)
  1. upload the stage scripts + this file (+ your cleaned data files)
  2. set DATASET / PROFILE / STAGES below, paste this file in a cell, run
  or from a terminal:  python xmofs_runner.py --dataset ov --profile full --stages 3,4,6,7,8,9
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== CONFIG ===================================
DATASET = "brca"                 # "brca" or "ov"
PROFILE = "quick"                # "quick" (smoke test) or "full" (real run)
STAGES = ["3", "4", "6", "7", "8", "9"]
SCRIPTS_DIR = "."                # folder containing the stage scripts

DATASETS = {
    "brca": {
        "x": "X_expression_clean.csv",
        "y": "y_pam50_labels.csv",
        "out": "results_brca",
        "pam50": True,      # PAM50 gene list is meaningful only for breast cancer
    },
    "ov": {
        "x": "cleaned_data_ov/ov_X_expression_clean.csv",
        "y": "cleaned_data_ov/ov_y_subtype_labels.csv",
        "out": "results_ov",
        "pam50": False,
    },
}

# Rule for choosing the Pareto solution that Stages 8 and 9 will use
PICK_MIN_GENES = 5
PICK_MAX_GENES = 100
# ===========================================================================

# A stage counts as "done" if this file exists in its results folder.
# With resume=True (default) finished stages are skipped, so if Colab
# disconnects you can simply run the same command again.
DONE_MARKERS = {
    "3": "stage3_results/baseline_results.csv",
    "4": "stage4_results/test_report.txt",
    "6": "stage6_results/pareto_front.csv",
    "7": "stage7_results/hypervolume_summary.csv",
    "8": "stage8_results/run_summary.txt",
    "9": "stage9_results/summary_table.csv",
}

SCRIPT_FILES = {
    "3": "stage3_baseline.py",
    "4": "stage4_deep_baseline.py",
    "6": "stage6_nsga2_search.py",
    "7": "stage7_hypervolume_comparison.py",
    "8": "stage8_shap_explainability.py",
    "9": "stage9_stability_ablation_stats.py",
}

# Budgets per profile (names must match constants inside the stage scripts;
# anything not listed keeps the script's own default = the "full" setting)
PROFILE_OVERRIDES = {
    "quick": {
        "3": {"N_FOLDS": 3, "TOP_K_GENES_OPTIONS": [20, 100]},
        "4": {"MAX_EPOCHS": 10},
        "6": {"POP_SIZE": 4, "N_GENERATIONS": 2, "INNER_MAX_EPOCHS": 8, "INNER_PATIENCE": 3},
        "7": {"SHALLOW_POP_SIZE": 4, "SHALLOW_N_GENERATIONS": 2},
        "8": {"MAX_EPOCHS": 10, "RUN_ENRICHR": False},
        "9": {"N_REPEATS": 2, "MAX_EPOCHS": 8, "RUN_NO_SPARSITY_ABLATION": False},
    },
    "full": {
        "9": {"N_REPEATS": 10},
    },
}


def log(msg):
    print(f"[runner] {msg}")


# --------------------------------------------------------------------
# Choose the Pareto-front solution used by Stages 8 and 9
# --------------------------------------------------------------------
def pick_balanced_point(front_csv, min_genes=PICK_MIN_GENES, max_genes=PICK_MAX_GENES):
    front = pd.read_csv(front_csv)
    cand = front[(front["n_selected_genes"] >= min_genes) &
                 (front["n_selected_genes"] <= max_genes)]
    note = f"{min_genes}-{max_genes} genes"
    if cand.empty:
        cand = front[front["n_selected_genes"] > 0]
        note = "no solution in the preferred range; used any non-empty solution"
    if cand.empty:
        raise RuntimeError("The Pareto front has no solution with >0 genes. "
                           "Re-run Stage 6 with a smaller lambda range or more epochs.")
    cand = cand.sort_values(["val_f1_macro", "n_selected_genes"], ascending=[False, True])
    row = cand.iloc[0]
    return {"lambda": float(row["lambda"]), "gate_lr": float(row["gate_lr"]),
            "n_selected_genes": int(row["n_selected_genes"]),
            "val_f1_macro": float(row["val_f1_macro"]), "rule": note}


# --------------------------------------------------------------------
# Run one stage script with config overrides
# --------------------------------------------------------------------
def run_stage(stage, overrides, scripts_dir):
    path = Path(scripts_dir) / SCRIPT_FILES[stage]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Upload the stage scripts next to the runner.")
    src = path.read_text()
    ns = {"__name__": "xmofs_stage", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)        # defines everything, does not run main()
    unknown = [k for k in overrides if k not in ns]
    if unknown:
        raise KeyError(f"{path.name} has no setting(s) {unknown} - script and runner are out of sync.")
    ns.update(overrides)                              # override constants BEFORE running
    t0 = time.time()
    ns["main"]()
    log(f"Stage {stage} finished in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------
# Final summary (tables you can paste into the paper)
# --------------------------------------------------------------------
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.astype(object).iterrows():   # astype(object) keeps integer columns as integers
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_final_summary(out_root, dataset, profile, chosen):
    out_root = Path(out_root)
    parts = [f"# XMOFS-Deep results: {dataset.upper()} (profile: {profile})\n"]
    if profile == "quick":
        parts.append("> **QUICK profile: tiny budgets for a smoke test. Do NOT report these numbers.**\n")

    f3 = out_root / "stage3_results" / "baseline_results.csv"
    if f3.exists():
        d = pd.read_csv(f3)[["n_features", "model", "cv_f1_macro_mean", "test_accuracy", "test_f1_macro"]]
        parts += ["## Stage 3 - filter-based baselines\n", md_table(d), ""]
    f6 = out_root / "stage6_results" / "pareto_front.csv"
    if f6.exists():
        d = pd.read_csv(f6)[["n_selected_genes", "val_f1_macro", "test_f1_macro", "test_accuracy"]]
        parts += ["## Stage 6 - XMOFS-Deep Pareto front\n", md_table(d), ""]
    hv6 = out_root / "stage6_results" / "hypervolume.txt"
    if hv6.exists():
        parts += ["```", hv6.read_text().strip(), "```", ""]
    f7 = out_root / "stage7_results" / "hypervolume_summary.csv"
    if f7.exists():
        parts += ["## Stage 7 - hypervolume comparison\n", md_table(pd.read_csv(f7)), ""]
        c7 = out_root / "stage7_results" / "cross_hypervolume_summary.txt"
        if c7.exists():
            parts += ["Cross-hypervolume (unique contribution of A beyond B):", "```",
                      c7.read_text().strip(), "```", ""]
    f7l = out_root / "stage7_results" / "hypervolume_log_scale.csv"
    if f7l.exists():
        parts += ["Robustness: hypervolume with log10(gene count) as the second objective:\n",
                  md_table(pd.read_csv(f7l), "{:.4f}"), ""]
    f9 = out_root / "stage9_results" / "summary_table.csv"
    if f9.exists():
        parts += ["## Stage 9 - matched-budget ablation (mean over repeated splits)\n",
                  md_table(pd.read_csv(f9)), ""]
    t9 = out_root / "stage9_results" / "stat_tests.csv"
    if t9.exists():
        d = pd.read_csv(t9)[["comparison", "mean_F1_diff (first - second)", "wins", "ties", "losses",
                              "p_value", "p_holm", "significant_at_0.05"]]
        parts += ["## Stage 9 - paired Wilcoxon tests (Holm-corrected)\n", md_table(d, "{:.4f}"), ""]
    s9 = out_root / "stage9_results" / "stability_summary.txt"
    if s9.exists():
        parts += ["## Stage 9 - gene-selection stability", "```", s9.read_text().strip(), "```", ""]
    if chosen:
        parts += ["## Pareto solution used by Stages 8 and 9", "```", json.dumps(chosen, indent=2), "```"]
    (out_root / "FINAL_SUMMARY.md").write_text("\n".join(parts))
    log(f"Wrote {out_root / 'FINAL_SUMMARY.md'}")



def stage_done(out_root, stage, profile):
    """A stage counts as finished only if its result file exists AND the stage itself was
    produced with the same profile. (A single folder-wide profile marker was not enough:
    running stages 3,4 with 'full' made leftover 'quick' results of stages 6-9 look finished.)"""
    out_root = Path(out_root)
    marker = out_root / DONE_MARKERS[stage]
    stage_profile = out_root / f"stage{stage}_results" / ".profile"
    return marker.exists() and stage_profile.exists() and stage_profile.read_text().strip() == profile


def decide_resume(out_root, profile, resume):
    """Resume only if the results folder was produced with the SAME profile.
    Otherwise a quick smoke-test run would be mistaken for finished 'full' stages."""
    out_root = Path(out_root)
    prof_file = out_root / ".profile"
    has_results = any((out_root / m).exists() for m in DONE_MARKERS.values())
    if resume and has_results:
        old = prof_file.read_text().strip() if prof_file.exists() else "unknown"
        if old != profile:
            log(f"WARNING: {out_root}/ holds results from profile '{old}', but you asked for "
                f"'{profile}'. Resume is switched OFF and the requested stages will be re-run, "
                f"so quick-test numbers can never be mistaken for real ones.")
            resume = False
    out_root.mkdir(parents=True, exist_ok=True)
    prof_file.write_text(profile)
    return resume

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def run(dataset=DATASET, profile=PROFILE, stages=None, scripts_dir=SCRIPTS_DIR,
        x_file=None, y_file=None, out_root=None, resume=True,
        pick_lambda=None, pick_gate_lr=None):
    stages = [str(s) for s in (stages or STAGES)]
    cfg = DATASETS[dataset]
    x_file = x_file or cfg["x"]
    y_file = y_file or cfg["y"]
    out_root = out_root or cfg["out"]
    resume = decide_resume(out_root, profile, resume)

    for f in (x_file, y_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"Data file not found: {f}")
    log(f"Dataset={dataset} | profile={profile} | stages={stages}")
    log(f"X={x_file} | y={y_file} | results -> {out_root}/")
    if profile == "quick":
        log("QUICK profile: for checking that the code runs only - do not report these numbers.")

    s3 = f"{out_root}/stage3_results"
    s6 = f"{out_root}/stage6_results"
    pareto = f"{s6}/pareto_front.csv"
    base = {"X_FILE": x_file, "Y_FILE": y_file}
    chosen = None
    chosen_file = Path(out_root) / "chosen_point.json"
    if chosen_file.exists():
        chosen = json.loads(chosen_file.read_text())

    for st in stages:
        log(f"\n{'=' * 60}\nSTAGE {st}\n{'=' * 60}")
        if resume and stage_done(out_root, st, profile):
            log(f"Stage {st} already finished with profile '{profile}' - skipping. "
                f"Use --no-resume to force a re-run.")
            continue
        ov = dict(base)
        ov["OUTPUT_DIR"] = f"{out_root}/stage{st}_results"
        if st == "4":
            ov["STAGE3_RESULTS"] = f"{s3}/baseline_results.csv"
        elif st == "7":
            ov["STAGE3_RESULTS"] = f"{s3}/baseline_results.csv"
            ov["STAGE6_PARETO_FRONT"] = pareto
        elif st in ("8", "9"):
            if not Path(pareto).exists():
                raise FileNotFoundError(f"{pareto} not found - run Stage 6 first.")
            if pick_lambda is not None and pick_gate_lr is not None:
                chosen = {"lambda": float(pick_lambda), "gate_lr": float(pick_gate_lr),
                          "rule": "MANUAL override (must have been chosen from VALIDATION F1, never test F1)"}
            else:
                chosen = pick_balanced_point(pareto)
            chosen_file.write_text(json.dumps(chosen, indent=2))
            log(f"Pareto solution used: {chosen}")
            ov["LAMBDA"] = chosen["lambda"]
            ov["GATE_LR"] = chosen["gate_lr"]
            if not cfg["pam50"]:
                ov["PAM50_GENES"] = set()      # PAM50 list is meaningless outside breast cancer
        ov.update(PROFILE_OVERRIDES.get(profile, {}).get(st, {}))
        run_stage(st, ov, scripts_dir)
        (Path(out_root) / f"stage{st}_results" / ".profile").write_text(profile)   # which profile produced THIS stage

    write_final_summary(out_root, dataset, profile, chosen)
    log("\nAll requested stages finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET, choices=list(DATASETS))
    ap.add_argument("--profile", default=PROFILE, choices=["quick", "full"])
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--scripts-dir", default=SCRIPTS_DIR)
    ap.add_argument("--x", default=None)
    ap.add_argument("--y", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pick-lambda", type=float, default=None, help="manual Pareto point for stages 8/9 (needs --pick-gate-lr)")
    ap.add_argument("--pick-gate-lr", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true", help="re-run stages even if their results exist")
    args, _ = ap.parse_known_args()          # ignores Colab/Jupyter's own arguments
    run(args.dataset, args.profile, args.stages.split(","), args.scripts_dir, args.x, args.y, args.out,
        resume=not args.no_resume,
        pick_lambda=args.pick_lambda, pick_gate_lr=args.pick_gate_lr)
