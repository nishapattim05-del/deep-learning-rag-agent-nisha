from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# INGESTION PANEL (FINAL FIXED VERSION)
# ---------------------------------------------------------------------------

def render_ingestion_panel(store: VectorStoreManager, chunker: DocumentChunker):
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload .pdf or .md files",
        type=["pdf", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.sidebar.button("Ingest Documents"):
            tmp_dir = Path("temp_uploads")
            tmp_dir.mkdir(exist_ok=True)

            file_paths = []

            # ✅ Save files with ORIGINAL names (CRITICAL FIX)
            for uploaded_file in uploaded_files:
                tmp_path = tmp_dir / uploaded_file.name

                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_paths.append(tmp_path)

            try:
                # ✅ Batch chunking (correct)
                chunks = chunker.chunk_files(file_paths)

                # ✅ Ingest
                result = store.ingest(chunks)

                st.sidebar.success(
                    f"✅ {result.ingested} chunks added, {result.skipped} duplicates skipped"
                )

                # Refresh document list
                st.session_state.ingested_documents = store.list_documents()

            except Exception as e:
                st.sidebar.error(f"❌ Error during ingestion: {e}")

    # -----------------------------------------------------------------------
    # Document List
    # -----------------------------------------------------------------------
    docs = store.list_documents()

    if docs:
        st.sidebar.subheader("📚 Documents")

        for doc in docs:
            col1, col2 = st.sidebar.columns([3, 1])

            with col1:
                st.write(f"📄 {doc['source']}")

            with col2:
                if st.button("🗑", key=f"delete_{doc['source']}"):
                    deleted = store.delete_document(doc["source"])
                    st.sidebar.success(f"Deleted {deleted} chunks")
                    st.rerun()


# ---------------------------------------------------------------------------
# DOCUMENT VIEWER
# ---------------------------------------------------------------------------

def render_document_viewer(store: VectorStoreManager):
    st.subheader("📄 Document Viewer")

    docs = store.list_documents()

    if not docs:
        st.info("Ingest documents using the sidebar.")
        return

    selected = st.selectbox(
        "Select document",
        options=[doc["source"] for doc in docs]
    )

    chunks = store.get_document_chunks(selected)

    st.write(f"Total chunks: {len(chunks)}")

    viewer = st.container(height=400)

    with viewer:
        for chunk in chunks:
            with st.expander(
                f"{chunk.metadata.topic} | {chunk.metadata.difficulty}"
            ):
                st.write(chunk.chunk_text)


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

def render_chat_interface(graph):
    st.subheader("💬 Interview Prep Chat")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.write(s)

            if msg.get("no_context_found"):
                st.warning("⚠️ No relevant context found in corpus.")

    query = st.chat_input("Ask a deep learning question...")

    if query:
        # User message
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )

            response = result["final_response"]

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "no_context_found": response.no_context_found,
            })

            with st.chat_message("assistant"):
                st.markdown(response.answer)

                if response.sources:
                    with st.expander("📎 Sources"):
                        for s in response.sources:
                            st.write(s)

                if response.no_context_found:
                    st.warning("⚠️ No relevant context found")

        except Exception as e:
            st.error(f"❌ Chat error: {e}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption("RAG-powered interview prep using LangGraph + ChromaDB")

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    render_ingestion_panel(store, chunker)

    col1, col2 = st.columns(2)

    with col1:
        render_document_viewer(store)

    with col2:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()