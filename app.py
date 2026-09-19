"""
app.py — Streamlit entrypoint. Two tabs: Chat (dual-model routed) and Trip Review
(reads/writes family_db.json directly, no model call needed for simple edits).

Run: streamlit run app.py
Requires Ollama running locally with the models set in router.py pulled
(default: qwen2.5:7b-instruct-q4_K_M and llama3.1:8b — override via
FAMILY_AGENT_QWEN_MODEL / FAMILY_AGENT_LLAMA_MODEL env vars).
"""

import json
from datetime import date, datetime

import streamlit as st

from prompts import build_qwen_system_prompt, build_llama_system_prompt
from router import route_and_call
from tools import search_flights, search_web, TOOL_REGISTRY

DB_PATH = "family_db.json"

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


st.set_page_config(page_title="Family Travel Agent", layout="wide")
db = load_db()

tab_chat, tab_review = st.tabs(["Plan a Trip", "Trip Review"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        # Empty state: center a welcome header + the chat input together,
        # Gemini-style, instead of Streamlit's default bottom-pinned input.
        st.markdown(
            """
            <style>
            [data-testid="stChatInput"] {
                position: fixed !important;
                bottom: auto !important;
                top: 45% !important;
                left: 50% !important;
                transform: translate(-50%, -50%);
                width: 55% !important;
                max-width: 700px;
            }
            </style>
            <div style="position: fixed; top: 30%; left: 50%; transform: translateX(-50%);
                        text-align: center; width: 100%;">
                <h1 style="margin-bottom: 0;">✈️ Family Travel Agent</h1>
                <p style="color: #888;">Ask about flights, itineraries, or budget for the next family trip.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Active state: normal top-to-bottom history; chat_input stays
        # pinned at the bottom by Streamlit's default behavior.
        st.subheader("Ask the family travel agent")
        for m in st.session_state.messages:
            st.chat_message(m["role"]).write(m["content"])

    user_input = st.chat_input("e.g. Find a warm, low-walking trip under $3000 for the grandparents")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        qwen_prompt = build_qwen_system_prompt(db)
        llama_prompt = build_llama_system_prompt(db)

        wants_flights = any(k in user_input.lower() for k in ["flight", "fly", "airfare"])
        result = route_and_call(
            user_input, qwen_prompt, llama_prompt,
            st.session_state.messages,
            tools=FLIGHT_TOOL_SCHEMA if wants_flights else None,
            tool_registry=TOOL_REGISTRY,
        )
        reply = result.get("message", {}).get("content", "(no response)")
        model_used = result.get("_routed_model", "unknown")

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
        st.caption(f"Routed to: {model_used}" + (" (tool calls executed)" if result.get("_warning") is None and wants_flights else ""))

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
