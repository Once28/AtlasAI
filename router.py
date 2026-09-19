"""
router.py — calls the local Ollama model for a chat turn.

Currently pinned to a single model (qwen2.5:3b) for speed on a 16GB i7
with no discrete GPU. Dual-model routing (Qwen for tools/CJK, Llama for
long English reasoning) is commented out below rather than deleted —
flip MODEL back to the 7B/Llama setup and un-comment choose_model() to
re-enable it once you're on faster hardware or want the reasoning split.

Requires: `ollama serve` running locally, with the active model pulled:
    ollama pull qwen2.5:3b
"""

import json
import os
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# --- Active model (single-model mode) ---------------------------------
# Local inference on a 16GB i7 with no discrete GPU is CPU-bound, so this
# defaults to the smallest/fastest option. Override via env var without
# touching code (e.g. to point at a fine-tuned model per TRAINING_GUIDE.md).
MODEL = os.environ.get("FAMILY_AGENT_MODEL", "qwen2.5:3b")

# --- Dual-model setup (commented out for future dev) -------------------
# Re-enable this block + choose_model() below, and switch route_and_call's
# callers back to passing two system prompts, to restore Qwen/Llama routing.
#
# MODEL_QWEN = os.environ.get("FAMILY_AGENT_QWEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")
# MODEL_LLAMA = os.environ.get("FAMILY_AGENT_LLAMA_MODEL", "llama3.1:8b")
#
# def choose_model(user_text: str, needs_tool_call: bool) -> str:
#     # Any CJK characters, or the caller needs tool calls -> Qwen (more
#     # stable tool/JSON output, native bilingual handling). Otherwise ->
#     # Llama for long English reasoning turns.
#     if needs_tool_call or CJK_PATTERN.search(user_text):
#         return MODEL_QWEN
#     return MODEL_LLAMA
#
# eGPU note (still relevant if you revisit the 7B/Llama setup): a USB4 or
# OCuLink eGPU dock — e.g. a Minisforum DEG1 housing an RTX 3060/4060 —
# would make qwen2.5:7b or larger unquantized models comfortable again.

# Keep only the last N chat turns (1 turn = 1 user + 1 assistant message) in
# the context sent to Ollama, to bound prompt-processing latency on CPU.
MAX_CONTEXT_TURNS = 5


def call_ollama(model: str, system_prompt: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """
    messages: [{"role": "user"/"assistant", "content": "..."}]
    tools: optional list of Ollama-format tool schemas.
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
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_registry: dict | None = None,
    max_tool_hops: int = 3,
) -> dict:
    """
    Calls MODEL and, if tools are attached and the model returns tool_calls,
    actually executes them and feeds results back until the model produces
    a plain-text answer (or max_tool_hops is hit).

    tool_registry: {"search_flights": search_flights, "search_web": search_web, ...}
    from tools.TOOL_REGISTRY — required if `tools` is passed.

    (user_text is unused in single-model mode; kept in the signature so
    callers don't need to change if dual-model routing is restored — see
    the commented choose_model() above.)
    """
    active_tools = tools
    # Truncate to the last MAX_CONTEXT_TURNS turns (~2 messages/turn) before
    # sending to Ollama — full history isn't needed for latency-sensitive
    # local inference, and this keeps prompt-processing time bounded on CPU.
    working_messages = list(messages[-MAX_CONTEXT_TURNS * 2 :])

    for _ in range(max_tool_hops):
        response = call_ollama(MODEL, system_prompt, working_messages, active_tools)
        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls or not active_tools:
            response["_routed_model"] = MODEL
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
    response["_routed_model"] = MODEL
    response["_warning"] = "max_tool_hops reached without a final answer"
    return response
