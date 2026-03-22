from chunker import create_chunks
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store():
    chunks = create_chunks()
    
    texts = [chunk["content"] for chunk in chunks]

    # Convert text → embeddings
    embeddings = model.encode(texts)

    # Convert to numpy array
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, texts


if __name__ == "__main__":
    index, texts = build_vector_store()

    print(f"Stored {len(texts)} chunks in vector DB")