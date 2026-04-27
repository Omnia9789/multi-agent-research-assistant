"""
Fact-Check Agent — evaluates the reasoning output for hallucinations and low-confidence claims.

Returns a structured verdict with issues list, confidence score, and flagged boolean.
"""

import json
import logging
import re
from core.state import ResearchState
from core.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous fact-checking agent. 
Given a research query, the retrieved source documents, and a synthesized answer, 
your job is to evaluate the answer for accuracy and reliability.

Respond ONLY with a valid JSON object in this exact schema:
{
  "confidence_score": <float 0.0-1.0>,
  "flagged": <boolean>,
  "verdict": "<string: PASS | WARN | FAIL>",
  "issues": [<list of specific issue strings, empty if none>],
  "unsupported_claims": [<list of claims in the answer not grounded in context>]
}

Scoring guide:
- 0.85-1.0: Answer is well-grounded, no hallucinations detected → PASS, flagged=false
- 0.60-0.84: Minor gaps or hedging issues → WARN, flagged=false
- 0.0-0.59:  Significant unsupported claims or contradictions → FAIL, flagged=true
"""


def _parse_json_response(raw: str) -> dict:
    """Robustly extract JSON from LLM response, even with markdown fences."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


async def fact_check_node(state: ResearchState) -> ResearchState:
    """LangGraph node: verify the reasoning output against retrieved sources."""
    query = state["query"]
    reasoning = state.get("reasoning_output", "")
    chunks = state.get("retrieved_chunks", [])
    logs = list(state.get("agent_logs", []))
    logs.append("[FactCheckAgent] Evaluating answer reliability.")

    # Build a short context summary for the fact-checker
    context_snippets = "\n".join(
        f"- [{c.get('source', '?')}]: {c.get('text', '')[:300]}..."
        for c in chunks[:5]
    )

    user_message = (
        f"**Query:** {query}\n\n"
        f"**Source Context (excerpts):**\n{context_snippets}\n\n"
        f"**Synthesized Answer:**\n{reasoning}\n\n"
        f"Evaluate the answer and respond with the JSON schema specified."
    )

    try:
        raw = await call_llm(
            system=SYSTEM_PROMPT,
            user=user_message,
            max_tokens=512,
        )
        result = _parse_json_response(raw)

        confidence = float(result.get("confidence_score", 0.5))
        flagged = bool(result.get("flagged", confidence < 0.6))
        issues = result.get("issues", []) + result.get("unsupported_claims", [])

        logs.append(
            f"[FactCheckAgent] Verdict: {result.get('verdict', 'N/A')} | "
            f"Confidence: {confidence:.0%} | Flagged: {flagged}"
        )
        return {
            **state,
            "fact_check_result": result,
            "confidence_score": confidence,
            "flagged": flagged,
            "agent_logs": logs,
            "error": None,
        }
    except Exception as exc:
        logger.exception("Fact-check failed.")
        logs.append(f"[FactCheckAgent] ERROR: {exc}. Defaulting to low confidence.")
        return {
            **state,
            "fact_check_result": {"issues": [str(exc)], "verdict": "FAIL"},
            "confidence_score": 0.4,
            "flagged": True,
            "agent_logs": logs,
            "error": str(exc),
        }
