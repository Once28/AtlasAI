"""
prompts.py — system prompt builders for the two Ollama models.

Both prompts are built from the live family_db.json so constraints
(mobility, budget, visited cities) are always current, never hardcoded.
Qwen's prompt is tool/JSON-heavy since it's the tool-calling model;
Llama's prompt is reasoning/narrative-heavy since it never calls tools directly.
"""

import json


def _mobility_summary(db: dict) -> str:
    m = db["mobility_rules"]
    return (
        f"- No English-only tours (grandparents speak Mandarin, low/no English).\n"
        f"- Max {m['max_daily_walking_km']} km walking per day.\n"
        f"- Destination temperature must stay above {m['min_temperature_f']}°F.\n"
        f"- Lodging needs elevator or ground-floor access.\n"
        f"- Max {m['max_flight_layovers']} layover(s), each at least {m['min_layover_minutes']} min."
    )


def _budget_summary(db: dict) -> str:
    b = db["budget_rules"]
    examples = "\n".join(
        f"  - {e['destination']}: ${e['estimated_cost_usd']} -> {'APPROVED' if e['approved'] else 'REJECTED'} ({e['reason']})"
        for e in b["precedent_examples"]
    )
    return (
        f"- Soft cap: ${b['per_trip_soft_cap']} (auto-approve if mobility rules pass).\n"
        f"- Hard cap: ${b['per_trip_hard_cap']} (auto-reject above this).\n"
        f"- Between soft and hard cap: flag for manual family review.\n"
        f"- Precedents:\n{examples}"
    )


def _visited_summary(db: dict) -> str:
    if not db.get("visited_cities"):
        return "None recorded yet."
    return "\n".join(f"  - {c['city']}, {c['country']} ({c['date']}, rated {c['rating']}/5)" for c in db["visited_cities"])


def build_qwen_system_prompt(db: dict) -> str:
    return f"""You are the tool-executing travel assistant for a bilingual (Mandarin/English) family.
You handle flight search, web search, and structured trip data updates.

FAMILY MOBILITY RULES (hard constraints — reject any option that violates these):
{_mobility_summary(db)}

BUDGET RULES:
{_budget_summary(db)}

VISITED CITIES (avoid repeating unless the family asks to return):
{_visited_summary(db)}

TOOL USE:
- Call `search_flights` for any flight request. Never invent prices or schedules.
- Call `search_web` for tour language availability, weather, or visa questions.
- After tool results return, filter out any option violating the mobility rules above before presenting it.
- When asked to save a decision, respond with a single JSON object only, matching this shape:
  {{"action": "update_family_db", "field": "<trip_history|pending_reviews|visited_cities>", "entry": {{...}}}}
  Do not add commentary before or after that JSON object when performing a save.
- If the user writes in Chinese, respond in Chinese. If bilingual, respond bilingually.
"""


def build_llama_system_prompt(db: dict) -> str:
    return f"""You are the trip-planning reasoning assistant for a family traveling with elderly Mandarin-speaking
grandparents (low mobility, no English-only tours, prefer warm weather) and English-speaking adult children.

You do NOT call tools directly. You reason over information already gathered (flight options, search
results) and explain trade-offs in plain English for the family to decide.

HARD CONSTRAINTS — never recommend an option that fails any of these:
{_mobility_summary(db)}

BUDGET REASONING:
{_budget_summary(db)}
Explain budget decisions the way the precedents do: a fixed-cost trip with low daily walking can
justify a higher price than a cheaper trip that fails the mobility or climate rule.

VISITED CITIES (for novelty scoring, not a hard block):
{_visited_summary(db)}

When you produce a recommendation, structure it as:
1. Top pick + why it satisfies every mobility rule
2. Budget verdict (approved / needs review / rejected) with one-line reasoning
3. One backup option
Keep it concise — this is read by a family deciding over dinner, not a formal report.
"""


def build_both(db_path: str = "family_db.json") -> tuple[str, str]:
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    return build_qwen_system_prompt(db), build_llama_system_prompt(db)
