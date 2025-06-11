from embeddings import embed_texts,embed_queries
from faiss_index import FaissIndex
import numpy as np

def main():
    # Sample documents & queries
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Python is a popular programming language.",
        "FastAPI is a modern web framework for building APIs with Python."
    ]

    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "Tell me about Python programming.",
        "What is FastAPI?"
    ]

    # 1. Embed documents and queries
    print("Embedding documents...")
    doc_embeddings = embed_texts(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print(type(doc_embeddings.shape))

    print("Embedding queries...")
    query_embeddings = embed_queries(queries)
    print(f"Query embeddings shape: {query_embeddings.shape}")

    # 2. Build FAISS index with document embeddings
    dimension = doc_embeddings.shape[1]
    faiss_index = FaissIndex(dimension=dimension)
    faiss_index.build_index(doc_embeddings)

    # 3. Search the index with each query embedding
    top_k = 2
    distances, indices = faiss_index.search(query_embeddings, top_k=top_k)

    # 4. Print results
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        print("Top matches:")
        for rank, idx in enumerate(indices[i]):
            print(f"  Rank {rank+1}: Document '{documents[idx]}' (distance={distances[i][rank]:.4f})")

if __name__ == "__main__":
    main()
