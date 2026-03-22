from retriever import search

def generate_answer(query):
    chunks = search(query)

    context = "\n\n".join(chunks)

    answer = f"""
Based on the retrieved information:

{context}

Summary:
This answer is generated from your knowledge base.
"""

    return answer


if __name__ == "__main__":
    query = input("Ask a question: ")

    answer = generate_answer(query)

    print("\nAI Answer:\n")
    print(answer)