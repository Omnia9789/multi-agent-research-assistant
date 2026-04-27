"""
Thin wrapper around the LLM provider (Anthropic Claude by default).
Swap out the client here to use OpenAI, Ollama, etc.
"""

import os
from anthropic import AsyncAnthropic

_client: AsyncAnthropic | None = None


def get_llm_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = AsyncAnthropic(api_key=api_key)
    return _client


async def call_llm(
    system: str,
    user: str,
    model: str = "claude-opus-4-5",
    max_tokens: int = 1024,
) -> str:
    """Send a single-turn message and return the text response."""
    client = get_llm_client()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text
