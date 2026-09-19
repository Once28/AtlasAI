"""
app.py — Streamlit entrypoint. Two tabs: Chat (single-model, Qwen2.5:3b) and
Trip Review (reads/writes family_db.json directly, no model call needed for
simple edits).

Run: streamlit run app.py
Requires Ollama running locally with the active model pulled
(default: qwen2.5:3b — override via FAMILY_AGENT_MODEL env var).
"""

import json
import re
from datetime import date, datetime

import streamlit as st

from prompts import build_qwen_system_prompt
# from prompts import build_llama_system_prompt  # dual-model mode only — see router.py
from router import route_and_call
from tools import search_flights, search_web, TOOL_REGISTRY

DB_PATH = "family_db.json"

# Lightweight "already been there" detector for automatic travel-history
# logging. Not NLP-robust — it catches the phrasings this family actually
# uses ("we've already been to X", "already visited Y"). A miss just means
# the region isn't auto-logged; it can still be added manually in the Trip
# Review tab. Captures up to 4 words so multi-word places ("Hong Kong",
# "New York City") come through, at the cost of occasionally grabbing a
# trailing word or two on unpunctuated sentences — acceptable for "lightweight".
VISITED_PATTERNS = [
    re.compile(
        r"(?:we(?:'ve| have)?\s+)?already\s+(?:been to|visited)\s+"
        r"([A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"we(?:'ve| have)\s+been to\s+([A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3})",
        re.IGNORECASE,
    ),
]
_LEADING_ARTICLES = ("the ", "a ", "an ")
# Cheap trailing-word cutoff: stop the captured place name at the first
# connector/temporal word, so "Japan last spring" -> "Japan" and
# "Hong Kong and loved it" -> "Hong Kong" without needing real NLP.
_STOP_WORDS = {
    "and", "but", "or", "last", "next", "this", "that", "before", "after",
    "during", "when", "while", "in", "on", "for", "it", "was", "is", "which",
}


def _clean_place(raw: str) -> str:
    place = raw.strip().rstrip(".,!?")
    for article in _LEADING_ARTICLES:
        if place.lower().startswith(article):
            place = place[len(article):]
            break
    words = []
    for word in place.split():
        if word.lower() in _STOP_WORDS:
            break
        words.append(word)
    return " ".join(words)

FLIGHT_TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search round-trip or one-way flights",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "depart_date": {"type": "string"},
                "return_date": {"type": "string"},
            },
            "required": ["origin", "destination", "depart_date"],
        },
    },
}]


def load_db() -> dict:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db: dict) -> None:
    db["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def detect_and_log_visited_places(user_text: str, db: dict) -> list[str]:
    """
    Scans user_text for explicit "already been there" mentions and appends
    any new ones to db['regional_travel_history']['visited_regions'], saving
    to disk immediately so the next system prompt build (including this
    turn's, since this runs before build_qwen_system_prompt below) reflects
    them as lower-priority per the existing soft-guidance rules.
    """
    history = db.setdefault("regional_travel_history", {"visited_regions": [], "priority_hierarchy": {}})
    visited = history.setdefault("visited_regions", [])
    existing_lower = {v.lower() for v in visited}

    newly_logged = []
    for pattern in VISITED_PATTERNS:
        for raw in pattern.findall(user_text):
            place = _clean_place(raw)
            if place and place.lower() not in existing_lower:
                visited.append(place)
                existing_lower.add(place.lower())
                newly_logged.append(place)

    if newly_logged:
        save_db(db)
    return newly_logged


def generate_reply(db: dict) -> None:
    """
    Generates a reply for the latest (unanswered) user message in
    st.session_state.messages, appends it, and reruns to render it.
    Called identically whether the user message arrived via the initial
    centered form or the bottom-pinned chat_input.
    """
    latest_user_text = st.session_state.messages[-1]["content"]

    newly_logged = detect_and_log_visited_places(latest_user_text, db)

    # build_qwen_system_prompt doubles as the active system prompt in
    # single-model mode — it's already tool/JSON-oriented, which is what
    # qwen2.5:3b needs for the flight-search tool call below. Built after
    # detect_and_log_visited_places so any region logged this turn is
    # already reflected in the soft-guidance regional priorities.
    system_prompt = build_qwen_system_prompt(db)
    # llama_prompt = build_llama_system_prompt(db)  # dual-model mode only

    wants_flights = any(k in latest_user_text.lower() for k in ["flight", "fly", "airfare"])
    with st.spinner("Thinking..."):
        result = route_and_call(
            latest_user_text, system_prompt,
            st.session_state.messages,
            tools=FLIGHT_TOOL_SCHEMA if wants_flights else None,
            tool_registry=TOOL_REGISTRY,
        )
    reply = result.get("message", {}).get("content", "(no response)")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "model_used": result.get("_routed_model", "unknown"),
        "tool_calls_executed": result.get("_warning") is None and wants_flights,
    })
    if newly_logged:
        st.session_state["_last_logged_places"] = newly_logged
    st.rerun()


st.set_page_config(page_title="Family Travel Agent", layout="wide")
db = load_db()

tab_chat, tab_review = st.tabs(["Plan a Trip", "Trip Review"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        # Initial state: a centered welcome banner + a plain st.form input.
        # Deliberately NOT st.chat_input here — chat_input is a special
        # widget that always fixed-positions at the bottom of the viewport
        # regardless of where it's called, so it can't be centered inline.
        # A plain form renders exactly where placed, which is what lets it
        # sit centered alongside the banner.
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown("<h1 style='text-align:center;'>✈️ Family Travel Agent</h1>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;color:#888;'>Ask about flights, itineraries, "
                "or budget for the next family trip.</p>",
                unsafe_allow_html=True,
            )
            with st.form("initial_prompt_form", clear_on_submit=True):
                initial_input = st.text_input(
                    "Message",
                    label_visibility="collapsed",
                    placeholder="e.g. Find a warm, low-walking trip under $3000 for the grandparents",
                )
                submitted_initial = st.form_submit_button("Send")
            if submitted_initial and initial_input:
                st.session_state.messages.append({"role": "user", "content": initial_input})
                st.rerun()

    else:
        # Active state: full thread top-to-bottom, chat_input at the root
        # of this block so Streamlit auto-anchors it below all messages.
        st.subheader("Ask the family travel agent")
        for m in st.session_state.messages:
            st.chat_message(m["role"]).write(m["content"])
            if m["role"] == "assistant" and m.get("model_used"):
                caption = f"Routed to: {m['model_used']}"
                if m.get("tool_calls_executed"):
                    caption += " (tool calls executed)"
                st.caption(caption)

        logged = st.session_state.pop("_last_logged_places", None)
        if logged:
            st.toast(f"Logged as visited: {', '.join(logged)}")

        chat_value = st.chat_input("e.g. Find a warm, low-walking trip under $3000 for the grandparents")
        if chat_value:
            st.session_state.messages.append({"role": "user", "content": chat_value})
            st.rerun()

        # A trailing unanswered user turn means a reply is pending — true
        # whether it arrived via the initial form (previous rerun) or
        # chat_input just above (this rerun).
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            generate_reply(db)

    with st.expander("Quick tool test (bypasses the model — calls tools.py directly)"):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("search_flights")
            qo = st.text_input("Origin airport code", "SEA", key="qo")
            qd = st.text_input("Destination airport code", "BUD", key="qd")
            qdate = st.text_input("Depart date (YYYY-MM-DD)", key="qdate")
            if st.button("Run search_flights"):
                st.json(search_flights(qo, qd, qdate))
        with col2:
            st.caption("search_web")
            qtext = st.text_input("Query", "Danube cruise bilingual tours", key="qtext")
            if st.button("Run search_web"):
                st.json(search_web(qtext))

with tab_review:
    st.subheader("Trip Review — approve, reject, or log a trip")
    st.json(db["budget_rules"]["precedent_examples"])

    with st.form("review_form"):
        dest = st.text_input("Destination")
        cost = st.number_input("Estimated cost (USD)", min_value=0, step=100)
        decision = st.selectbox("Decision", ["approved", "rejected", "needs_review"])
        note = st.text_area("Notes")
        submitted = st.form_submit_button("Save to family_db.json")

    if submitted and dest:
        entry = {
            "destination": dest,
            "estimated_cost_usd": cost,
            "decision": decision,
            "notes": note,
            "logged_on": str(date.today()),
        }
        if decision == "needs_review":
            db["pending_reviews"].append(entry)
        else:
            db["trip_history"].append(entry)
        save_db(db)
        st.success(f"Saved {dest} to family_db.json ({decision}).")

    st.divider()
    st.write("Visited cities")
    st.table(db["visited_cities"])