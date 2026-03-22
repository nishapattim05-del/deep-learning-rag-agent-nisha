from vector_store import build_vector_store, model
import numpy as np

def search(query, k=3):
    index, texts = build_vector_store()

    # Convert query → embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search in FAISS
    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        results.append(texts[i])

    return results


if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search(query)

    print("\nTop relevant chunks:\n")
    for i, res in enumerate(results):
        print(f"{i+1}. {res}")
        print("-" * 50)