"""
graph.py
========
LangGraph agent graph definition and compilation.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        # 1. Create graph
        graph = StateGraph(AgentState)

        # 2. Add nodes
        graph.add_node("query_rewrite", query_rewrite_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("generation", generation_node)

        # 3. Add edges
        graph.add_edge(START, "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")

        # 4. Always go to generation (handles both found and no-context cases)
        graph.add_edge("retrieval", "generation")

        # 5. Final edge
        graph.add_edge("generation", END)

        # 6. Compile graph
        return graph.compile(checkpointer=self._checkpointer)


@lru_cache(maxsize=1)
def get_compiled_graph():
    return AgentGraphBuilder().build()