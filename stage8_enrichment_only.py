"""
stage8_enrichment_only.py
Runs ONLY the pathway-enrichment part of Stage 8 on the gene list your
Stage 8 run already saved (stage8_results/selected_genes_shap.csv).
No re-training, so the gene list stays exactly the one in your plots.

Colab: run  !pip install gseapy -q   first, then run this cell.
Needs internet (Enrichr web service).
"""
import pandas as pd
import gseapy as gp

GENE_FILE = "stage8_results/selected_genes_shap.csv"   # saved by Stage 8
OUT_FILE = "stage8_results/pathway_enrichment.csv"

genes = pd.read_csv(GENE_FILE)["gene"].tolist()
print(f"{len(genes)} genes:", genes)

libs = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2021"]
import time

rows = []
for lib in libs:
    # Enrichr returns "status code: 429" (too many requests) if we call it
    # too fast, so wait between libraries and retry a few times.
    for attempt in range(1, 4):
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism="human",
                             outdir=None, cutoff=1.0)
            r = enr.results.sort_values("Adjusted P-value").head(10).copy()
            r["library"] = lib
            rows.append(r)
            break
        except Exception as e:
            wait = 15 * attempt
            print(f"Library {lib}, attempt {attempt} failed: {e}. "
                  f"Waiting {wait}s...")
            time.sleep(wait)
    else:
        print(f"Library {lib} skipped after 3 attempts.")
    time.sleep(10)   # pause before the next library

if rows:
    res = pd.concat(rows)[["library", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
    res.to_csv(OUT_FILE, index=False)
    pd.set_option("display.max_colwidth", 60, "display.width", 200)
    print(res.to_string(index=False))
    print("\nRule of thumb: only treat a pathway as a real finding if its "
          "Adjusted P-value is below 0.05. Otherwise write: no significant enrichment.")
else:
    print("No results - check internet access and the gseapy install.")
