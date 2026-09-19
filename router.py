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

import json
import os
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# Local inference on a 16GB i7 with no discrete GPU is CPU-bound, so default
# to a quantized model for latency. Override via env var once you have
# faster hardware (e.g. an eGPU dock over USB4 or OCuLink — a Minisforum
# DEG1 housing an RTX 3060/4060 — at which point qwen2.5:7b or larger
# unquantized models become comfortable again).
MODEL_QWEN = os.environ.get("FAMILY_AGENT_QWEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")
# Lighter alternative if even the q4_K_M 7B is too slow on CPU-only: "qwen2.5:3b"
MODEL_LLAMA = os.environ.get("FAMILY_AGENT_LLAMA_MODEL", "llama3.1:8b")

# Keep only the last N chat turns (1 turn = 1 user + 1 assistant message) in
# the context sent to Ollama, to bound prompt-processing latency on CPU.
MAX_CONTEXT_TURNS = 5


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


def route_and_call(
    user_text: str,
    system_prompt_qwen: str,
    system_prompt_llama: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_registry: dict | None = None,
    max_tool_hops: int = 3,
) -> dict:
    """
    Routes to a model, and if it's Qwen with tools attached, actually
    executes any tool_calls the model returns and feeds results back
    until the model produces a plain-text answer (or max_tool_hops is hit).

    tool_registry: {"search_flights": search_flights, "search_web": search_web, ...}
    from tools.TOOL_REGISTRY — required if `tools` is passed.
    """
    needs_tool_call = tools is not None
    model = choose_model(user_text, needs_tool_call)
    system_prompt = system_prompt_qwen if model == MODEL_QWEN else system_prompt_llama

    active_tools = tools if model == MODEL_QWEN else None
    # Truncate to the last MAX_CONTEXT_TURNS turns (~2 messages/turn) before
    # sending to Ollama — full history isn't needed for latency-sensitive
    # local inference, and this keeps prompt-processing time bounded on CPU.
    working_messages = list(messages[-MAX_CONTEXT_TURNS * 2 :])

    for _ in range(max_tool_hops):
        response = call_ollama(model, system_prompt, working_messages, active_tools)
        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls or not active_tools:
            response["_routed_model"] = model
            return response

        # Model wants to call one or more tools — actually run them.
        working_messages.append(message)
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            fn = (tool_registry or {}).get(fn_name)
            if fn is None:
                tool_result = {"error": f"unknown tool '{fn_name}'"}
            else:
                try:
                    tool_result = fn(**fn_args)
                except Exception as e:
                    tool_result = {"error": str(e)}

            working_messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, default=str),
            })
        # loop again so the model can read the tool result and respond

    # Ran out of hops — return whatever the last response was.
    response["_routed_model"] = model
    response["_warning"] = "max_tool_hops reached without a final answer"
    return response
