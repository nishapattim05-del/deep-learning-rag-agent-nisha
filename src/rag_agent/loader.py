import os

def load_corpus(folder_path="data/corpus"):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                documents.append({
                    "filename": filename,
                    "content": content
                })

    return documents


if __name__ == "__main__":
    docs = load_corpus()
    
    for doc in docs:
        print(f"Loaded: {doc['filename']}")