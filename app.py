import streamlit as st
from src.rag_agent.rag_pipeline import generate_answer

st.title("RAG AI Agent 🤖")

query = st.text_input("Ask a question:")

if query:
    answer = generate_answer(query)
    st.write(answer)