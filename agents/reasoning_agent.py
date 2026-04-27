"""
Reasoning Agent — synthesizes retrieved context into a coherent answer via LLM.
"""

import logging
from core.state import ResearchState
from core.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meticulous research analyst. Your job is to synthesize 
retrieved document excerpts into a clear, well-structured answer.

Rules:
1. Cite evidence from the provided context wherever possible.
2. If context is insufficient, clearly state what is known vs. uncertain.
3. Use markdown formatting: headings, bullet points, bold for key terms.
4. Do NOT fabricate facts not found in the context.
5. Write for a technically literate audience.
"""


def _format_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents retrieved."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        parts.append(f"[Chunk {i} | source: {source} | relevance: {score:.2f}]\n{text}")
    return "\n\n---\n\n".join(parts)


async def reasoning_node(state: ResearchState) -> ResearchState:
    """LangGraph node: synthesize retrieved chunks into a draft answer."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    logs = list(state.get("agent_logs", []))
    logs.append(f"[ReasoningAgent] Synthesizing {len(chunks)} chunk(s).")

    context_str = _format_context(chunks)
    user_message = (
        f"**Research Query:** {query}\n\n"
        f"**Retrieved Context:**\n\n{context_str}\n\n"
        f"Please provide a comprehensive, well-cited answer."
    )

    try:
        answer = await call_llm(
            system=SYSTEM_PROMPT,
            user=user_message,
            max_tokens=2048,
        )
        logs.append("[ReasoningAgent] Synthesis complete.")
        return {**state, "reasoning_output": answer, "agent_logs": logs, "error": None}
    except Exception as exc:
        logger.exception("Reasoning failed.")
        logs.append(f"[ReasoningAgent] ERROR: {exc}")
        return {**state, "reasoning_output": "", "agent_logs": logs, "error": str(exc)}
