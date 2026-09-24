"""
app.py — Streamlit entrypoint, restyled to match the "Atlas" reference design
(dark theme, icon rail + chat column + "Top Picks" side panel). Same backend
as before: MongoDB Atlas via crud.py, Ollama via router.py, single active
model (qwen2.5:3b by default — see router.py / FAMILY_AGENT_MODEL env var).
Trip Review (approve/reject/log a trip to the ledger) moved behind the
clipboard icon in the left rail instead of a visible tab, since the
reference design doesn't show a tab switcher at all.

Run: streamlit run app.py
Requires Ollama running locally with the active model pulled, and MONGO_URI
set — see .env.example / secrets.toml.template.

Note on the left-rail icons and the "Top Picks" thumbnails: Streamlit's
st.button can't render arbitrary HTML/images as its clickable surface (it's
a plain-text/emoji button), so a handful of these are implemented as a
photo-styled <div> immediately followed by a slim, minimally-chromed
button beneath it rather than a true click-anywhere card. Chasing pixel-
perfect click-anywhere cards would mean depending on undocumented Streamlit
internals (unstable across versions) for a cosmetic gain — noted inline
below at each spot. Requires Streamlit >= 1.36 for st.segmented_control
(used for the Itinerary/Flights/Hotels tabs).
"""

import json
import re
import urllib.parse
import urllib.request

import streamlit as st

import crud
from prompts import build_qwen_system_prompt
# from prompts import build_llama_system_prompt  # dual-model mode only — see router.py
from router import route_and_call, MODEL
from tools import search_flights, search_web, TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Chat-logic core — UNCHANGED from the previous UI. Same backend, same model,
# same MongoDB-backed persistence; only the rendering below this section is
# new.
# ---------------------------------------------------------------------------

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

WELCOME_MESSAGE = (
    "Hello! I'm Atlas, your AI travel companion. Where are you dreaming of "
    "going? Tell me a vibe, a season, or a destination — I'll take it from "
    "there."
)


def load_db() -> dict:
    return crud.get_full_context_db()


def detect_and_log_visited_places(user_text: str, db: dict) -> list[str]:
    """
    Scans user_text for explicit "already been there" mentions and, for
    each new one, writes it to MongoDB via crud.add_visited_region AND
    mutates the in-memory `db` dict for this turn — the write makes it
    durable, the in-memory mutation means this turn's system prompt (built
    right after this call) reflects it immediately rather than waiting on
    the cache TTL or the next rerun's re-fetch.
    """
    visited = db.setdefault("regional_travel_history", {}).setdefault("visited_regions", [])
    existing_lower = {v.lower() for v in visited}

    newly_logged = []
    for pattern in VISITED_PATTERNS:
        for raw in pattern.findall(user_text):
            place = _clean_place(raw)
            if place and place.lower() not in existing_lower:
                if crud.add_visited_region(place):
                    visited.append(place)
                    existing_lower.add(place.lower())
                    newly_logged.append(place)

    return newly_logged


def generate_reply(db: dict) -> None:
    """
    Generates a reply for the latest (unanswered) user message in
    st.session_state.messages, appends it, and reruns to render it. Called
    identically whether the user message arrived via a quick-prompt chip,
    the "Plan this trip" button, or the input bar.
    """
    latest_user_text = st.session_state.messages[-1]["content"]

    newly_logged = detect_and_log_visited_places(latest_user_text, db)

    system_prompt = build_qwen_system_prompt(db)
    # llama_prompt = build_llama_system_prompt(db)  # dual-model mode only

    wants_flights = any(k in latest_user_text.lower() for k in ["flight", "fly", "airfare"])
    with st.spinner("Atlas is thinking..."):
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


def send_message(text: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.messages.append({"role": "user", "content": text})
    st.rerun()


# ---------------------------------------------------------------------------
# Top Picks — tailored from the family's own data when it's there, falling
# back to a curated default set when it's not.
#
# "Tailored": built from db['regional_travel_history']['priority_hierarchy']
# ['high_priority_unvisited'] (the family's actual unvisited-priority
# regions from MongoDB), padded out with DEFAULT_PICKS to fill the 2x2 grid
# if the DB has fewer than 4 entries, or replaced by DEFAULT_PICKS entirely
# if the DB has none yet. We don't have live price/weather/flight-duration
# data for an arbitrary DB-driven region (that would mean a real flights +
# weather API call per card, per render — too slow/costly for a decorative
# panel), so those badges are simply omitted for DB-driven picks rather than
# fabricated; only DEFAULT_PICKS (curated by hand) carry them.
#
# "Look online for images": fetch_destination_image() below hits Wikipedia's
# public REST summary API (no key, stdlib-only HTTP) for a real photo of
# whatever name is being shown — curated or DB-driven alike — rather than
# hardcoding image URLs I can't verify are live. Cached per name for an
# hour. If the lookup fails (network issue, no matching page, no photo on
# the page), the card's gradient shows through instead — CSS renders
# `background-image: url(...), <gradient>` as two stacked layers, so a
# failed/blocked image url() layer simply reveals the gradient beneath it,
# no error, no blank card.
#
# The Itinerary/Flights/Hotels tab content below the hero card is unrelated
# to this and stays static regardless of the selected pick — see the note
# above ITINERARY_DEMO further down for why.
# ---------------------------------------------------------------------------

DEFAULT_PICKS = [
    {"name": "Santorini", "country": "Greece", "tag": "Aegean Islands",
     "temp": "26°C", "duration": "4h 20m", "price": "$890",
     "gradient": "linear-gradient(160deg, #274156 0%, #4a7c93 55%, #cfe3e8 100%)"},
    {"name": "Tokyo", "country": "Japan", "tag": "Urban Pulse",
     "temp": "18°C", "duration": "13h 50m", "price": "$1,340",
     "gradient": "linear-gradient(160deg, #1a1035 0%, #5b2a86 45%, #d94f8c 100%)"},
    {"name": "Bali", "country": "Indonesia", "tag": "Tropical Retreat",
     "temp": "29°C", "duration": "17h 05m", "price": "$1,120",
     "gradient": "linear-gradient(160deg, #17342e 0%, #2f6f5e 50%, #e8c07d 100%)"},
    {"name": "Cappadocia", "country": "Turkey", "tag": "Ancient wonder",
     "temp": "22°C", "duration": "3h 45m", "price": "$670",
     "gradient": "linear-gradient(160deg, #3a2416 0%, #a85a2e 55%, #f0b25a 100%)"},
]

# A handful of the family's actual DB region names don't map cleanly onto a
# single Wikipedia article title — redirect those specific lookups only.
WIKI_TITLE_ALIASES = {
    "Baltic region": "Baltic Sea",
    "Western Mediterranean": "Mediterranean Sea",
}

FALLBACK_GRADIENTS = [
    "linear-gradient(160deg, #274156 0%, #4a7c93 55%, #cfe3e8 100%)",
    "linear-gradient(160deg, #1a1035 0%, #5b2a86 45%, #d94f8c 100%)",
    "linear-gradient(160deg, #17342e 0%, #2f6f5e 50%, #e8c07d 100%)",
    "linear-gradient(160deg, #3a2416 0%, #a85a2e 55%, #f0b25a 100%)",
]


def _fallback_gradient(name: str) -> str:
    # Deterministic (not Python's randomized-per-process hash()) so the same
    # region gets the same gradient across app restarts, not just within one.
    idx = sum(ord(c) for c in name) % len(FALLBACK_GRADIENTS)
    return FALLBACK_GRADIENTS[idx]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_destination_image(name: str) -> str | None:
    """
    Real photo lookup via Wikipedia's public REST summary API — no API key,
    stdlib-only HTTP (urllib), so no new dependency. Returns a thumbnail URL
    or None on any failure (network error, no matching page, page has no
    image). 4s timeout so one slow/unreachable lookup can't hang the page
    for long; cached an hour so this only actually hits the network once
    per name per hour, not on every Streamlit rerun.
    """
    title = WIKI_TITLE_ALIASES.get(name, name)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AtlasTravelAgent/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("thumbnail", {}).get("source") or data.get("originalimage", {}).get("source")
    except Exception:
        return None


def get_top_picks(db: dict, limit: int = 4) -> list[dict]:
    """
    Builds the Top Picks list from the family's actual high-priority
    unvisited regions in MongoDB when available, padding with DEFAULT_PICKS
    to fill the grid; falls back to DEFAULT_PICKS entirely if the DB has no
    priority regions recorded yet.
    """
    priorities = (
        db.get("regional_travel_history", {})
        .get("priority_hierarchy", {})
        .get("high_priority_unvisited", [])
    )

    picks, seen = [], set()
    for region in priorities:
        if len(picks) >= limit:
            break
        picks.append({
            "name": region, "country": "", "tag": "Recommended for you",
            "temp": None, "duration": None, "price": None,
            "gradient": _fallback_gradient(region),
        })
        seen.add(region.lower())

    for default in DEFAULT_PICKS:
        if len(picks) >= limit:
            break
        if default["name"].lower() not in seen:
            picks.append(default)
            seen.add(default["name"].lower())

    for pick in picks:
        pick["image_url"] = fetch_destination_image(pick["name"])

    return picks[:limit]


ITINERARY_DEMO = [
    ("Day 1", "Arrive Thira · Oia check-in", "🛬"),
    ("Day 2", "Caldera hike · wine tasting", "🥾"),
    ("Day 3", "Akrotiri ruins · Red Beach", "🏛️"),
    ("Day 4", "Boat tour · hot springs", "⛵"),
    ("Day 5", "Pyrgos village · sunset", "🌅"),
    ("Day 6", "Free day · spa at leisure", "🧖"),
]

FLIGHTS_DEMO = [
    {"airline": "Aegean Airlines", "flight_no": "A3 601", "dep": "08:15", "arr": "12:35",
     "duration": "4h 20m", "stops": "Nonstop", "price": "$430"},
    {"airline": "Olympic Air", "flight_no": "OA 204", "dep": "11:40", "arr": "16:30",
     "duration": "4h 50m", "stops": "1 stop · ATH", "price": "$370"},
    {"airline": "British Airways", "flight_no": "BA 093", "dep": "14:00", "arr": "21:10",
     "duration": "7h 10m", "stops": "1 stop · LHR", "price": "$520"},
]

HOTELS_DEMO = [
    {"name": "Canaves Oia Epitome", "stars": 5, "rating": "9.8", "price": "$420/n"},
    {"name": "Grace Hotel Santorini", "stars": 5, "rating": "9.4", "price": "$290/n"},
    {"name": "Andronis Concept", "stars": 4, "rating": "9.1", "price": "$210/n"},
]

QUICK_PROMPTS = [
    "Plan a 7-day trip to Santorini",
    "Best places in Japan for autumn",
    "Hidden gems in Southeast Asia",
    "Beach escape under $1,500",
]

# ---------------------------------------------------------------------------
# Theme — dark palette + component overrides matching the reference design.
# Scoped selectors use Streamlit's key-based wrapper classes (st-key-<key>,
# added automatically for any widget given a `key=`) so this doesn't rely
# on brittle nth-child ordering. Requires a reasonably current Streamlit;
# if these classes don't apply on your version, upgrade streamlit.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
:root {
    --Atlas-bg: #0c0c0e;
    --Atlas-card: #1a1a1d;
    --Atlas-card-2: #17171a;
    --Atlas-border: #2a2a2e;
    --Atlas-accent: #eba63c;
    --Atlas-text: #e8e8ea;
    --Atlas-text-dim: #8a8a90;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--Atlas-bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }
html, body, [class*="css"] { color: var(--Atlas-text); }

/* rail icon buttons */
.st-key-rail_logo button, .st-key-rail_chat button, .st-key-rail_review button,
.st-key-rail_explore button, .st-key-rail_history button {
    background-color: var(--Atlas-card) !important;
    border: 1px solid var(--Atlas-border) !important;
    color: var(--Atlas-text-dim) !important;
    border-radius: 10px !important;
    width: 42px; height: 42px; padding: 0 !important;
}
.st-key-rail_logo button {
    background-color: var(--Atlas-accent) !important;
    color: #1a1200 !important; font-weight: 700 !important;
    border: none !important;
}

/* quick-prompt chips */
.st-key-chip_0 button, .st-key-chip_1 button, .st-key-chip_2 button, .st-key-chip_3 button {
    background-color: var(--Atlas-card) !important;
    border: 1px solid var(--Atlas-border) !important;
    color: #cfcfd2 !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.9rem !important;
    font-size: 0.82rem !important;
}

/* model badge */
.Atlas-badge {
    background-color: #241c10; color: var(--Atlas-accent);
    border: 1px solid #4a3a1a; border-radius: 999px;
    padding: 0.2rem 0.7rem; font-size: 0.75rem; font-weight: 600;
    display: inline-block;
}

/* chat header */
.Atlas-header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.5rem; }
.Atlas-header-title { font-weight: 700; font-size: 1.05rem; }
.Atlas-header-sub { color: var(--Atlas-text-dim); font-size: 0.78rem; }
.Atlas-status-dot { color: #3ecf72; font-size: 0.7rem; }

/* input row */
.st-key-user_input div[data-baseweb="input"] {
    background-color: var(--Atlas-card) !important;
    border-radius: 999px !important;
    border: 1px solid var(--Atlas-border) !important;
}
.st-key-user_input input { color: var(--Atlas-text) !important; }
.st-key-send_btn button {
    background-color: var(--Atlas-accent) !important;
    color: #1a1200 !important;
    border-radius: 999px !important;
    border: none !important;
    width: 42px; height: 42px; padding: 0 !important; font-weight: 700 !important;
}
.Atlas-disclaimer { text-align: center; color: var(--Atlas-text-dim); font-size: 0.72rem; margin-top: 0.4rem; }

/* top picks panel */
.Atlas-panel-heading { color: var(--Atlas-text-dim); font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; }
.Atlas-pick-card {
    border-radius: 12px; padding: 0.6rem; height: 78px;
    display: flex; flex-direction: column; justify-content: flex-end;
    border: 2px solid transparent; position: relative; margin-bottom: 0.4rem;
    background-size: cover;
}
.Atlas-pick-card.selected { border-color: var(--Atlas-accent); }
.Atlas-pick-card .check { position: absolute; top: 6px; right: 6px; background: var(--Atlas-accent);
    color: #1a1200; border-radius: 999px; width: 18px; height: 18px; font-size: 0.65rem;
    display: flex; align-items: center; justify-content: center; font-weight: 700; }
.Atlas-pick-title { font-weight: 700; font-size: 0.85rem; text-shadow: 0 1px 3px rgba(0,0,0,0.6); }
.Atlas-pick-sub { font-size: 0.68rem; color: #e6e6e6; text-shadow: 0 1px 3px rgba(0,0,0,0.6); }

.st-key-select_pick_0 button, .st-key-select_pick_1 button,
.st-key-select_pick_2 button, .st-key-select_pick_3 button {
    background: transparent !important; border: none !important; color: var(--Atlas-text-dim) !important;
    font-size: 0.7rem !important; padding: 0 !important; margin-top: -0.6rem;
}

.Atlas-hero { border-radius: 14px; padding: 1rem; margin-top: 0.5rem; position: relative; background-size: cover; min-height: 130px; }
.Atlas-hero-badges { display: flex; gap: 0.4rem; position: absolute; top: 10px; right: 10px; }
.Atlas-hero-badge { background: rgba(0,0,0,0.45); color: #fff; font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 999px; }
.Atlas-hero-tag { background: rgba(0,0,0,0.4); color: #f0d9a8; font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 6px; display: inline-block; margin-bottom: 0.3rem; }
.Atlas-hero-title { font-size: 1.4rem; font-weight: 800; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }
.Atlas-hero-sub { font-size: 0.8rem; color: #e6e6e6; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }

.Atlas-price-label { color: var(--Atlas-text-dim); font-size: 0.72rem; }
.Atlas-price-value { color: var(--Atlas-accent); font-size: 1.2rem; font-weight: 700; }
.Atlas-price-sub { color: var(--Atlas-text-dim); font-size: 0.68rem; }

.st-key-plan_trip_btn button {
    background-color: var(--Atlas-accent) !important; color: #1a1200 !important;
    font-weight: 700 !important; border-radius: 8px !important; border: none !important;
}

.Atlas-list-row { background: var(--Atlas-card); border: 1px solid var(--Atlas-border); border-radius: 10px;
    padding: 0.5rem 0.7rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.6rem; }
.Atlas-list-icon { background: var(--Atlas-card-2); border-radius: 8px; width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center; }
.Atlas-list-title { font-size: 0.82rem; font-weight: 600; }
.Atlas-list-sub { font-size: 0.68rem; color: var(--Atlas-text-dim); }
.Atlas-list-price { margin-left: auto; color: var(--Atlas-accent); font-weight: 700; font-size: 0.85rem; }
</style>
"""

st.set_page_config(page_title="Atlas — AI Travel Companion", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)

db = load_db()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"
if "selected_pick_index" not in st.session_state:
    st.session_state.selected_pick_index = 0

rail, main, side = st.columns([0.55, 5.2, 3.1], gap="medium")

# --- Left icon rail (mostly decorative; logo resets chat, clipboard toggles
# the Trip Review page since the reference design has no visible tab bar) ---
with rail:
    if st.button("A", key="rail_logo", help="New chat"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.active_page = "chat"
        st.rerun()
    if st.button("💬", key="rail_chat", help="Chat"):
        st.session_state.active_page = "chat"
        st.rerun()
    if st.button("📋", key="rail_review", help="Trip Review"):
        st.session_state.active_page = "review"
        st.rerun()
    st.button("🌐", key="rail_explore", help="Explore (not wired up yet)", disabled=True)
    st.button("🕐", key="rail_history", help="History (not wired up yet)", disabled=True)

if st.session_state.active_page == "review":
    with main:
        st.subheader("Trip Review — approve, reject, or log a trip")
        st.json(db["budget_rules"]["precedent_examples"])

        with st.form("review_form"):
            dest = st.text_input("Destination")
            cost = st.number_input("Estimated cost (USD)", min_value=0, step=100)
            decision = st.selectbox("Decision", ["approved", "rejected", "needs_review"])
            note = st.text_area("Notes")
            submitted = st.form_submit_button("Save to MongoDB")

        if submitted and dest:
            entry = {
                "destination": dest,
                "estimated_cost_usd": cost,
                "decision": decision,
                "notes": note,
            }
            entry_type = "pending_review" if decision == "needs_review" else "trip_history"
            crud.add_ledger_entry(entry_type, entry)
            st.success(f"Saved {dest} to MongoDB ({decision}).")

        st.divider()
        st.write("Visited cities")
        st.table(db["visited_cities"])
    with side:
        pass
else:
    with main:
        st.markdown(
            f"""
            <div class="Atlas-header">
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <div style="width:36px; height:36px; border-radius:999px; background: var(--Atlas-accent);
                                display:flex; align-items:center; justify-content:center; font-weight:800; color:#1a1200;">A</div>
                    <div>
                        <div class="Atlas-header-title">Atlas</div>
                        <div class="Atlas-header-sub"><span class="Atlas-status-dot">●</span> AI Travel Companion · online</div>
                    </div>
                </div>
                <div class="Atlas-badge">{MODEL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chip_cols = st.columns(len(QUICK_PROMPTS))
        for i, (col, prompt) in enumerate(zip(chip_cols, QUICK_PROMPTS)):
            with col:
                if st.button(prompt, key=f"chip_{i}", use_container_width=True):
                    send_message(prompt)

        st.write("")
        thread = st.container(height=380)
        with thread:
            for m in st.session_state.messages:
                avatar = "🧭" if m["role"] == "assistant" else "🧳"
                st.chat_message(m["role"], avatar=avatar).write(m["content"])
                if m["role"] == "assistant" and m.get("model_used"):
                    caption = f"Routed to: {m['model_used']}"
                    if m.get("tool_calls_executed"):
                        caption += " (tool calls executed)"
                    st.caption(caption)

        logged = st.session_state.pop("_last_logged_places", None)
        if logged:
            st.toast(f"Logged as visited: {', '.join(logged)}")

        with st.form("send_form", clear_on_submit=True):
            input_col, send_col = st.columns([9, 1])
            with input_col:
                user_text = st.text_input(
                    "Message", key="user_input", label_visibility="collapsed",
                    placeholder="Ask Atlas anything about your trip...",
                )
            with send_col:
                send_clicked = st.form_submit_button("↑", use_container_width=True, key="send_btn")
        st.markdown(
            '<div class="Atlas-disclaimer">Atlas can make mistakes. Verify important information.</div>',
            unsafe_allow_html=True,
        )

        if send_clicked and user_text.strip():
            send_message(user_text.strip())

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

    # --- Right "Top Picks" panel -------------------------------------------
    with side:
        top_row = st.columns([4, 1])
        with top_row[0]:
            st.markdown('<div class="Atlas-panel-heading">Top picks for you</div>', unsafe_allow_html=True)
        with top_row[1]:
            if st.button("View all →", key="view_all_btn"):
                st.toast("Full destination catalog coming soon")

        picks = get_top_picks(db)
        # Clamp in case the DB-driven set shrank since the index was chosen
        # (e.g. someone edited destination_priorities mid-session).
        st.session_state.selected_pick_index = min(st.session_state.selected_pick_index, len(picks) - 1)

        grid = st.columns(2)
        for i, pick in enumerate(picks):
            selected = i == st.session_state.selected_pick_index
            bg = f"url('{pick['image_url']}'), {pick['gradient']}" if pick.get("image_url") else pick["gradient"]
            with grid[i % 2]:
                st.markdown(
                    f"""
                    <div class="Atlas-pick-card {'selected' if selected else ''}" style="background-image: {bg};">
                        {'<div class="check">✓</div>' if selected else ''}
                        <div class="Atlas-pick-title">{pick['name']}</div>
                        {f'<div class="Atlas-pick-sub">{pick["country"]}</div>' if pick['country'] else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # See module docstring: a true click-anywhere photo card isn't
                # achievable with a plain st.button, so this slim link-style
                # button sits directly under the card instead.
                if st.button(f"View {pick['name']} →", key=f"select_pick_{i}"):
                    st.session_state.selected_pick_index = i
                    st.rerun()

        hero = picks[st.session_state.selected_pick_index]
        hero_bg = f"url('{hero['image_url']}'), {hero['gradient']}" if hero.get("image_url") else hero["gradient"]
        badges = "".join(
            f'<div class="Atlas-hero-badge">{icon} {value}</div>'
            for icon, value in (("☀", hero.get("temp")), ("✈", hero.get("duration")))
            if value
        )
        st.markdown(
            f"""
            <div class="Atlas-hero" style="background-image: {hero_bg};">
                <div class="Atlas-hero-badges">{badges}</div>
                <div class="Atlas-hero-tag">{hero['tag']}</div>
                <div class="Atlas-hero-title">{hero['name']}</div>
                {f'<div class="Atlas-hero-sub">{hero["country"]}</div>' if hero['country'] else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        price_col, btn_col = st.columns([1.2, 1])
        with price_col:
            if hero.get("price"):
                st.markdown(
                    f"""
                    <div style="padding-top:0.6rem;">
                        <div class="Atlas-price-label">From</div>
                        <div class="Atlas-price-value">{hero['price']}</div>
                        <div class="Atlas-price-sub">round-trip</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding-top:0.6rem;" class="Atlas-price-sub">'
                    'No fare on file yet — ask Atlas to check flights.</div>',
                    unsafe_allow_html=True,
                )
        with btn_col:
            st.write("")
            if st.button("Plan this trip →", key="plan_trip_btn", use_container_width=True):
                destination = f"{hero['name']}, {hero['country']}" if hero["country"] else hero["name"]
                send_message(f"Plan a trip to {destination}.")

        st.write("")
        active_tab = st.segmented_control(
            "View", ["Itinerary", "Flights", "Hotels"], default="Itinerary",
            label_visibility="collapsed", key="active_tab",
        )
        if active_tab is None:
            # segmented_control allows deselecting by clicking the active
            # pill again; treat that as "back to Itinerary" rather than
            # silently falling through to the Hotels branch below.
            active_tab = "Itinerary"

        # NOTE (see module docstring): the reference design's tab content is
        # the same regardless of which destination is selected above — this
        # reproduces that faithfully rather than inventing per-destination
        # data the original mockup doesn't actually have.
        if active_tab == "Itinerary":
            for day, title, icon in ITINERARY_DEMO:
                st.markdown(
                    f"""
                    <div class="Atlas-list-row">
                        <div class="Atlas-list-icon">{icon}</div>
                        <div>
                            <div class="Atlas-list-title">{title}</div>
                            <div class="Atlas-list-sub">{day}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif active_tab == "Flights":
            for f in FLIGHTS_DEMO:
                st.markdown(
                    f"""
                    <div class="Atlas-list-row">
                        <div>
                            <div class="Atlas-list-title">{f['airline']} <span class="Atlas-list-sub">{f['flight_no']}</span></div>
                            <div class="Atlas-list-sub">{f['dep']} → {f['arr']} · {f['duration']} · {f['stops']}</div>
                        </div>
                        <div class="Atlas-list-price">{f['price']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            for h in HOTELS_DEMO:
                st.markdown(
                    f"""
                    <div class="Atlas-list-row">
                        <div>
                            <div class="Atlas-list-title">{h['name']} {'★' * h['stars']}</div>
                            <div class="Atlas-list-sub">Rating {h['rating']}</div>
                        </div>
                        <div class="Atlas-list-price">{h['price']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
