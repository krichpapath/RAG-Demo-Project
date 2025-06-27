import json
import numpy as np
import matplotlib.pyplot as plt
from embeddings import embed_queries, embed_texts
from faiss_index import FaissIndex
from reranker import compute_scores
from chunk_text import split_text_with_langchain
from bm25_index import BM25Retriever
from ocr import get_all_pdf
import os
from collections import defaultdict
import matplotlib.font_manager as fm

# Load and Prepare Data
PDF_PATH = "./pdf_files/anime.pdf"
TEXT_PATH = "./text_document/ocr_all_pdf.txt"
EVAL_SET = "./eval_data.json"

if not os.path.exists(TEXT_PATH):
    get_all_pdf(PDF_PATH, output_path=TEXT_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    big_text = f.read()

documents = split_text_with_langchain(big_text, chunk_size=1024, chunk_overlap=100)
print(f"✅ Loaded {len(documents)} chunks")

doc_embeddings = embed_texts(documents)
faiss_index = FaissIndex()
faiss_index.build_index(doc_embeddings)
bm25 = BM25Retriever(documents)

with open(EVAL_SET, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Metrics
def precision_at_k(pred, gold, k):
    return len(set(pred[:k]) & set(gold)) / k

def recall_at_k(pred, gold, k):
    return len(set(pred[:k]) & set(gold)) / len(gold) if gold else 0

def mrr(pred, gold):
    for rank, pid in enumerate(pred, start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0

# --- Evaluation Loop ---
results = defaultdict(list)
all_metrics = []

for idx, item in enumerate(eval_data):
    query = item["query"]
    relevant = item["relevant"]
    query_vec = embed_queries([query])

    # FAISS
    faiss_dist, faiss_idx = faiss_index.search(query_vec, top_k=10)
    faiss_ranked = faiss_idx[0].tolist()

    # BM25
    bm25_scores = bm25.get_scores(query)
    bm25_ranked = np.argsort(bm25_scores)[::-1][:10].tolist()

    # Hybrid
    faiss_scores = 1 - (faiss_dist[0] / max(faiss_dist[0]))
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    hybrid_scores = {}
    for i, idx_ in enumerate(faiss_idx[0]):
        hybrid_scores[idx_] = 0.7 * faiss_scores[i]
    for idx_, score in enumerate(bm25_norm):
        hybrid_scores[idx_] = hybrid_scores.get(idx_, 0) + 0.3 * score
    hybrid_ranked = [i for i, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]]

    # Reranked FAISS
    rerank_scores = compute_scores(query, [documents[i] for i in faiss_ranked])
    reranked_faiss = [i for _, i in sorted(zip(rerank_scores, faiss_ranked), reverse=True)]

    # Metrics
    metric_values = {}
    for name, ranked in zip(["FAISS", "BM25", "Hybrid", "Rerank"],
                             [faiss_ranked, bm25_ranked, hybrid_ranked, reranked_faiss]):
        p = precision_at_k(ranked, relevant, 5)
        r = recall_at_k(ranked, relevant, 5)
        m = mrr(ranked, relevant)
        results[f"{name.lower()}_p@5"].append(p)
        results[f"{name.lower()}_r@5"].append(r)
        results[f"{name.lower()}_mrr"].append(m)
        metric_values[name] = (p, r, m)

    all_metrics.append((query, metric_values))

print("\n===== Evaluation Results =====")
for metric, scores in results.items():
    print(f"{metric}: {np.mean(scores):.3f}")

# Plot for Each Query
print("\n📊 ploting graph")
os.makedirs("./eval/plots", exist_ok=True)
try:
    th_font = fm.FontProperties(fname="C:/Windows/Fonts/leelawad.ttf")
except Exception:
    th_font = fm.FontProperties()

for i, (query, metric_dict) in enumerate(all_metrics):
    models = list(metric_dict.keys())
    p_scores = [metric_dict[m][0] for m in models]
    r_scores = [metric_dict[m][1] for m in models]
    mrr_scores = [metric_dict[m][2] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))

    ax.plot(x, p_scores, marker='o', label='Precision@5')
    ax.plot(x, r_scores, marker='s', label='Recall@5')
    ax.plot(x, mrr_scores, marker='^', label='MRR')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontproperties=th_font)
    ax.set_title(f"Query {i+1}: {query[:50]}...", fontproperties=th_font)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"./eval/plots/query_{i+1}.png")
    plt.close(fig)

print("✅ All plots saved in ./eval/plots")