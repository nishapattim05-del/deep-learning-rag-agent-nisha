"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.
"""
 
from __future__ import annotations
 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
 
from rag_agent.agent.prompts import NO_CONTEXT_RESPONSE, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse
from rag_agent.config import LLMFactory
from rag_agent.vectorstore.store import VectorStoreManager
 
 
# ---------------------------------------------------------------------------
# Query Rewrite Node
# ---------------------------------------------------------------------------
 
def query_rewrite_node(state: dict) -> dict:
    messages = state.get("messages", [])
 
    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
 
    return {
        "original_query": original_query,
        "rewritten_query": original_query,
    }
 
 
# ---------------------------------------------------------------------------
# Retrieval Node
# ---------------------------------------------------------------------------
 
def retrieval_node(state: dict) -> dict:
    manager = VectorStoreManager()
 
    query = state.get("rewritten_query") or state.get("original_query", "")
 
    chunks = manager.query(query_text=query)
 
    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }
 
    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }
 
 
# ---------------------------------------------------------------------------
# Generation Node
# ---------------------------------------------------------------------------
 
def generation_node(state: dict) -> dict:
    llm = LLMFactory().create()
 
    query = state.get("original_query", "")
 
    # ---- No context guard ----
    if state.get("no_context_found"):
        msg = NO_CONTEXT_RESPONSE
 
        return {
            "final_response": AgentResponse(
                answer=msg,
                sources=[],
                confidence=0.0,
                no_context_found=True,
                rewritten_query=query,
            ),
            "messages": [AIMessage(content=msg)],
        }
 
    # ---- Build context ----
    context = ""
    sources = []
    scores = []
 
    for chunk in state.get("retrieved_chunks", []):
        citation = f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]"
        context += f"{citation}\n{chunk.chunk_text}\n\n"
        sources.append(citation)
        scores.append(chunk.score)
 
    avg_confidence = sum(scores) / len(scores) if scores else 0.0
 
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{query}"
        ),
    ]
 
    result = llm.invoke(messages)
    answer = result.content.strip()
 
    return {
        "final_response": AgentResponse(
            answer=answer,
            sources=list(set(sources)),
            confidence=avg_confidence,
            no_context_found=False,
            rewritten_query=query,
        ),
        "messages": [AIMessage(content=answer)],
    }
 
 
# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
 
def should_retry_retrieval(state: dict) -> str:
    if state.get("no_context_found"):
        return "end"
    return "generate"