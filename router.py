"""
router.py — routes a user turn to the right local Ollama model.

Rule of thumb (cheap, no extra model call needed):
  - Any CJK characters in the input, OR the user explicitly wants
    tool calls (flight/tool JSON) -> Qwen2.5:7b (more stable tool/JSON output,
    native bilingual handling for the grandparents).
  - Otherwise (long English planning/reasoning turns) -> Llama3.1:8b.

Requires: `ollama serve` running locally, with both models pulled:
    ollama pull qwen2.5:7b
    ollama pull llama3.1:8b
"""

import re
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

MODEL_QWEN = "qwen2.5:7b"
MODEL_LLAMA = "llama3.1:8b"


def choose_model(user_text: str, needs_tool_call: bool) -> str:
    if needs_tool_call or CJK_PATTERN.search(user_text):
        return MODEL_QWEN
    return MODEL_LLAMA


def call_ollama(model: str, system_prompt: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """
    messages: [{"role": "user"/"assistant", "content": "..."}]
    tools: optional list of Ollama-format tool schemas (only reliably
    honored by Qwen2.5 in this setup — keep Llama calls tool-free).
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def route_and_call(user_text: str, system_prompt_qwen: str, system_prompt_llama: str,
                    messages: list[dict], tools: list[dict] | None = None) -> dict:
    needs_tool_call = tools is not None
    model = choose_model(user_text, needs_tool_call)
    system_prompt = system_prompt_qwen if model == MODEL_QWEN else system_prompt_llama
    result = call_ollama(model, system_prompt, messages, tools if model == MODEL_QWEN else None)
    result["_routed_model"] = model
    return result
