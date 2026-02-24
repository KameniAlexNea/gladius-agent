import json
import os
from typing import Any


def call_llm(prompt: str, schema: dict | None = None, model: str | None = None) -> dict:
    """
    Calls the configured LLM with a structured prompt and returns parsed JSON.
    Supports OpenAI-compatible APIs. Raises on failure.
    """
    client = _get_client()
    messages = [{"role": "user", "content": prompt}]
    if schema:
        messages[0]["content"] += f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

    response = client.chat.completions.create(
        model=model or os.environ.get("LLM_MODEL", "gpt-4o"),
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


def _get_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    except ImportError:
        raise ImportError("openai package required: pip install openai")
