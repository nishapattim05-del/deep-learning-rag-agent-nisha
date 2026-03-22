from loader import load_corpus

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - overlap

    return chunks


def create_chunks():
    documents = load_corpus()
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        
        for chunk in chunks:
            all_chunks.append({
                "source": doc["filename"],
                "content": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()
    
    print(f"Total chunks created: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1}:")
        print(chunk["content"])
        print("-" * 50)