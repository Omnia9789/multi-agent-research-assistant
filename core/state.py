"""
Shared state schema for the LangGraph multi-agent pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional


class ResearchState(TypedDict):
    """The shared state object passed between all agents in the graph."""

    # Input
    query: str

    # Retrieval agent output
    retrieved_chunks: List[Dict[str, Any]]  # [{text, score, source}, ...]

    # Reasoning agent output
    reasoning_output: str

    # Fact-check agent output
    fact_check_result: Dict[str, Any]  # {issues: [...], verdict: str}
    confidence_score: float
    flagged: bool

    # Final synthesized answer
    final_answer: str

    # Observability
    agent_logs: List[str]
    error: Optional[str]
