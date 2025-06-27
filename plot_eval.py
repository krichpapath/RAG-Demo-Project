# plot_eval.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
EVAL_RESULTS_PATH = "./eval_data.json"  # Will contain P@5, R@5, MRR

# --- Load Evaluation Results ---
if not os.path.exists(EVAL_RESULTS_PATH):
    raise FileNotFoundError(f"Missing evaluation results file: {EVAL_RESULTS_PATH}")

with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)  # {"faiss_p@5": [...], "bm25_p@5": [...], ...}

# --- Plot Setup ---
metrics = ["p@5", "r@5", "mrr"]
methods = ["faiss", "bm25", "hybrid", "rerank"]
colors = {"faiss": "blue", "bm25": "green", "hybrid": "orange", "rerank": "red"}

# --- Create Subplots ---
fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(18, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    for method in methods:
        key = f"{method}_{metric}"
        if key in results:
            y = results[key]
            x = list(range(len(y)))
            ax.scatter(x, y, label=method.upper(), color=colors[method], alpha=0.7, s=30)
            ax.hlines(np.mean(y), xmin=0, xmax=len(y)-1, colors=colors[method], linestyles="dashed", label=f"{method} avg")

    ax.set_title(f"{metric.upper()} per Query")
    ax.set_xlabel("Query Index")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

plt.suptitle("Evaluation Results per Query", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("eval_plot.png")
plt.show()
