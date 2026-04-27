"""
LangGraph graph definition — wires retrieval → reasoning → fact-check → output.
"""

from langgraph.graph import StateGraph, END
from core.state import ResearchState
from agents.retrieval_agent import retrieval_node
from agents.reasoning_agent import reasoning_node
from agents.fact_check_agent import fact_check_node
from agents.output_agent import output_node


def should_flag(state: ResearchState) -> str:
    """Conditional edge: route based on fact-check verdict."""
    if state.get("flagged"):
        return "flagged_output"
    return "clean_output"


def build_graph(docs_dir: str = "data/sample_docs") -> StateGraph:
    """
    Build and compile the multi-agent LangGraph pipeline.

    Graph topology:
        retrieval → reasoning → fact_check → (conditional) → output → END
    """

    # Inject docs_dir into the retrieval node via closure
    async def retrieval_with_dir(state: ResearchState) -> ResearchState:
        return await retrieval_node(state, docs_dir=docs_dir)

    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("retrieval", retrieval_with_dir)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("output", output_node)

    # Entry point
    graph.set_entry_point("retrieval")

    # Linear edges
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "fact_check")

    # Conditional routing after fact-check
    graph.add_conditional_edges(
        "fact_check",
        should_flag,
        {
            "flagged_output": "output",
            "clean_output": "output",
        },
    )

    graph.add_edge("output", END)

    return graph.compile()
