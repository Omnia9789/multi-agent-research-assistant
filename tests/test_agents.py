"""
Unit tests for Multi-Agent Research Assistant.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.state import ResearchState
from agents.reasoning_agent import reasoning_node
from agents.fact_check_agent import fact_check_node, _parse_json_response
from agents.output_agent import output_node


# ─── Fixtures ────────────────────────────────────────────────────────────────

def base_state(**kwargs) -> ResearchState:
    defaults = {
        "query": "What is RAG?",
        "retrieved_chunks": [
            {"text": "RAG stands for Retrieval-Augmented Generation.", "score": 0.9, "source": "ai_overview.md"},
        ],
        "reasoning_output": "",
        "fact_check_result": {},
        "confidence_score": 0.0,
        "flagged": False,
        "final_answer": "",
        "agent_logs": [],
        "error": None,
    }
    return {**defaults, **kwargs}


# ─── Reasoning Agent ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reasoning_node_success():
    mock_answer = "## RAG\nRAG stands for Retrieval-Augmented Generation..."
    with patch("agents.reasoning_agent.call_llm", new_callable=AsyncMock, return_value=mock_answer):
        state = base_state()
        result = await reasoning_node(state)

    assert result["reasoning_output"] == mock_answer
    assert result["error"] is None
    assert any("ReasoningAgent" in log for log in result["agent_logs"])


@pytest.mark.asyncio
async def test_reasoning_node_error_handling():
    with patch("agents.reasoning_agent.call_llm", side_effect=Exception("API Error")):
        state = base_state()
        result = await reasoning_node(state)

    assert result["reasoning_output"] == ""
    assert "API Error" in result["error"]


# ─── Fact-Check Agent ────────────────────────────────────────────────────────

def test_parse_json_response_clean():
    raw = '{"confidence_score": 0.9, "flagged": false, "verdict": "PASS", "issues": [], "unsupported_claims": []}'
    parsed = _parse_json_response(raw)
    assert parsed["verdict"] == "PASS"
    assert parsed["confidence_score"] == 0.9


def test_parse_json_response_with_fences():
    raw = '```json\n{"confidence_score": 0.5, "flagged": true, "verdict": "FAIL", "issues": ["Unsupported claim"], "unsupported_claims": []}\n```'
    parsed = _parse_json_response(raw)
    assert parsed["verdict"] == "FAIL"
    assert parsed["flagged"] is True


@pytest.mark.asyncio
async def test_fact_check_node_pass():
    mock_resp = '{"confidence_score": 0.92, "flagged": false, "verdict": "PASS", "issues": [], "unsupported_claims": []}'
    with patch("agents.fact_check_agent.call_llm", new_callable=AsyncMock, return_value=mock_resp):
        state = base_state(reasoning_output="RAG reduces hallucination.")
        result = await fact_check_node(state)

    assert result["confidence_score"] == pytest.approx(0.92)
    assert result["flagged"] is False
    assert result["fact_check_result"]["verdict"] == "PASS"


@pytest.mark.asyncio
async def test_fact_check_node_flagged():
    mock_resp = '{"confidence_score": 0.4, "flagged": true, "verdict": "FAIL", "issues": ["Claim not in sources"], "unsupported_claims": []}'
    with patch("agents.fact_check_agent.call_llm", new_callable=AsyncMock, return_value=mock_resp):
        state = base_state(reasoning_output="Unsupported claim here.")
        result = await fact_check_node(state)

    assert result["flagged"] is True
    assert result["confidence_score"] < 0.6


# ─── Output Agent ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_output_node_clean():
    state = base_state(
        reasoning_output="## Answer\nRAG is great.",
        fact_check_result={"verdict": "PASS", "issues": [], "unsupported_claims": []},
        confidence_score=0.95,
        flagged=False,
    )
    result = await output_node(state)
    assert "✅" in result["final_answer"]
    assert "PASS" in result["final_answer"]


@pytest.mark.asyncio
async def test_output_node_flagged():
    state = base_state(
        reasoning_output="## Answer\nSome answer.",
        fact_check_result={"verdict": "FAIL", "issues": ["Made-up claim"], "unsupported_claims": []},
        confidence_score=0.4,
        flagged=True,
    )
    result = await output_node(state)
    assert "🚨" in result["final_answer"]
    assert "Made-up claim" in result["final_answer"]


@pytest.mark.asyncio
async def test_output_node_pipeline_error():
    state = base_state(
        reasoning_output="",
        error="Connection timeout",
        flagged=False,
    )
    result = await output_node(state)
    assert "Pipeline Error" in result["final_answer"]
    assert "Connection timeout" in result["final_answer"]
