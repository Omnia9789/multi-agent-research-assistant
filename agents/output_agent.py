"""
Output Agent — assembles the final answer, appending fact-check annotations.
"""

from core.state import ResearchState


_PASS_FOOTER = """
---
> ✅ **Fact-Check:** PASSED — This answer is well-grounded in the retrieved sources.
"""

_WARN_FOOTER = """
---
> ⚠️ **Fact-Check:** WARNING — Some claims may need further verification.
"""

_FAIL_FOOTER = """
---
> 🚨 **Fact-Check:** FLAGGED — This answer contains potentially unsupported or hallucinated claims. 
> Please verify against primary sources before use.
"""


def _build_footer(state: ResearchState) -> str:
    fc = state.get("fact_check_result", {})
    verdict = fc.get("verdict", "PASS")
    issues = fc.get("issues", []) + fc.get("unsupported_claims", [])
    confidence = state.get("confidence_score", 1.0)

    if verdict == "FAIL" or state.get("flagged"):
        footer = _FAIL_FOOTER
    elif verdict == "WARN":
        footer = _WARN_FOOTER
    else:
        footer = _PASS_FOOTER

    footer += f"> **Confidence Score:** {confidence:.0%}\n"

    if issues:
        footer += ">\n> **Issues detected:**\n"
        for issue in issues:
            footer += f"> - {issue}\n"

    return footer


async def output_node(state: ResearchState) -> ResearchState:
    """LangGraph node: combine reasoning output + fact-check footer into final answer."""
    logs = list(state.get("agent_logs", []))
    reasoning = state.get("reasoning_output", "No answer generated.")

    if state.get("error") and not reasoning:
        final = (
            f"## ❌ Pipeline Error\n\n"
            f"An error occurred: `{state['error']}`\n\n"
            f"Please check your API key and document directory."
        )
    else:
        footer = _build_footer(state)
        final = reasoning + "\n" + footer

    logs.append("[OutputAgent] Final answer assembled.")
    return {**state, "final_answer": final, "agent_logs": logs}
