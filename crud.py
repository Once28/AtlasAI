"""
crud.py — data access layer over the 5 MongoDB collections. Every read goes
through db.sanitize() so callers (Streamlit widgets, prompts.py, tools.py)
never touch pymongo or bson.ObjectId directly.

Collections:
  family_profiles        — one doc, keyed by family_name (root metadata + members)
  mobility_profiles       — one doc per profile_id (e.g. "group_low_mobility")
  budget_rules             — one doc (singleton): caps + valuation_heuristics
  destination_priorities   — one doc (singleton): visited_regions + tier_1..5
  travel_ledger             — many docs, discriminated by "type":
                               "visited_city" | "trip_history" | "pending_review"

get_full_context_db() stitches all 5 back into the exact dict shape the old
family_db.json had, so prompts.py's system-prompt builders needed ZERO
changes — they already just take a plain dict.
"""

from datetime import date

import streamlit as st
from bson import ObjectId

from db import FAMILY_NAME, get_db, sanitize


# ---------- family_profiles ----------

def get_family_profile() -> dict | None:
    doc = get_db().family_profiles.find_one({"family_name": FAMILY_NAME})
    return sanitize(doc)


def update_family_profile(updates: dict) -> None:
    get_db().family_profiles.update_one(
        {"family_name": FAMILY_NAME}, {"$set": updates}, upsert=True
    )
    get_full_context_db.clear()


# ---------- mobility_profiles ----------

def get_mobility_profile(profile_id: str) -> dict | None:
    doc = get_db().mobility_profiles.find_one({"profile_id": profile_id})
    return sanitize(doc)


def get_all_mobility_profiles() -> list[dict]:
    return sanitize(list(get_db().mobility_profiles.find({})))


def upsert_mobility_profile(profile_id: str, data: dict) -> None:
    get_db().mobility_profiles.update_one(
        {"profile_id": profile_id}, {"$set": {**data, "profile_id": profile_id}}, upsert=True
    )
    get_full_context_db.clear()


# ---------- budget_rules (singleton doc) ----------

def get_budget_rules() -> dict | None:
    doc = get_db().budget_rules.find_one({})
    return sanitize(doc)


def update_budget_rules(updates: dict) -> None:
    get_db().budget_rules.update_one({}, {"$set": updates}, upsert=True)
    get_full_context_db.clear()


# ---------- destination_priorities (singleton doc) ----------

def get_destination_tiers() -> dict | None:
    doc = get_db().destination_priorities.find_one({})
    return sanitize(doc)


def update_destination_tiers(updates: dict) -> None:
    get_db().destination_priorities.update_one({}, {"$set": updates}, upsert=True)
    get_full_context_db.clear()


def add_visited_region(region: str) -> bool:
    """
    Adds `region` to destination_priorities.visited_regions if not already
    present (case-insensitive check, since $addToSet alone is case-sensitive
    and would let "Japan" and "japan" coexist as duplicates). Returns True
    if it was newly added. Used by app.py's auto-detection of "we've
    already been to X" mentions.
    """
    coll = get_db().destination_priorities
    doc = coll.find_one({}) or {}
    existing_lower = {r.lower() for r in doc.get("visited_regions", [])}
    if region.lower() in existing_lower:
        return False
    coll.update_one({}, {"$addToSet": {"visited_regions": region}}, upsert=True)
    get_full_context_db.clear()
    return True


# ---------- travel_ledger ----------

def get_travel_ledger(entry_type: str | None = None) -> list[dict]:
    """entry_type: "visited_city" | "trip_history" | "pending_review" | None (all types)."""
    query = {"type": entry_type} if entry_type else {}
    docs = list(get_db().travel_ledger.find(query).sort("logged_on", -1))
    return sanitize(docs)


def add_ledger_entry(entry_type: str, entry: dict) -> str:
    doc = {**entry, "type": entry_type, "logged_on": entry.get("logged_on") or str(date.today())}
    result = get_db().travel_ledger.insert_one(doc)
    get_full_context_db.clear()
    return str(result.inserted_id)


def update_ledger_entry(entry_id: str, updates: dict) -> None:
    get_db().travel_ledger.update_one({"_id": ObjectId(entry_id)}, {"$set": updates})
    get_full_context_db.clear()


# ---------- stitching: reassemble the legacy family_db.json shape ----------

# TTL cache, not just a plain function: get_full_context_db() does ~7 round
# trips to Atlas, and Streamlit reruns this whole script on every widget
# interaction — without caching, a single chat turn (which calls this once
# for the system prompt) plus normal page reruns would burn through Atlas
# M0's connection/RU budget fast. 30s keeps data reasonably fresh for a
# single-family app; every write function above calls .clear() explicitly
# so an edit is reflected on the very next read, not after the TTL expires.
@st.cache_data(ttl=30, show_spinner=False)
def get_full_context_db() -> dict:
    profile = get_family_profile() or {}
    low_mobility = get_mobility_profile("group_low_mobility") or {}
    budget = get_budget_rules() or {}
    tiers = get_destination_tiers() or {}
    visited_cities = get_travel_ledger("visited_city")
    trip_history = get_travel_ledger("trip_history")
    pending_reviews = get_travel_ledger("pending_review")
    valuation = budget.get("valuation_heuristics", {})
    priority = tiers.get  # local alias, just for line-length below

    return {
        "schema_version": "2.0-mongo",
        "family_profile": {
            "family_name": profile.get("family_name"),
            "home_base": profile.get("home_base"),
            "members": profile.get("members", []),
        },
        "family_notes": {
            "translation": profile.get("translation_note", ""),
            "guidance_type": "soft",
        },
        "mobility_rules": {
            "guidance_type": low_mobility.get("guidance_type", "soft"),
            "description": low_mobility.get("description", ""),
            "max_daily_walking_km": low_mobility.get("max_walking_km_per_day"),
            "min_temperature_f": low_mobility.get("climate_preference", {}).get("min_temp_f"),
            "prefers_elevator_or_ground_floor_lodging": low_mobility.get("needs_elevator_access"),
            "max_flight_layovers_preferred": low_mobility.get("max_flight_layovers_preferred"),
            "min_layover_minutes": low_mobility.get("min_layover_minutes"),
            "hard_reject": False,
        },
        "budget_rules": {
            "guidance_type": budget.get("guidance_type", "soft"),
            "currency": budget.get("currency", "USD"),
            "per_trip_soft_cap": budget.get("per_trip_soft_cap"),
            "per_trip_hard_cap": budget.get("per_trip_hard_cap"),
            "roi_logic": valuation.get("roi_logic", ""),
            "roi_reflections": valuation.get("roi_reflections", []),
            "precedent_examples": budget.get("historical_roi_precedents", []),
            "approval_flow": budget.get("approval_flow", ""),
        },
        "regional_travel_history": {
            "guidance_type": tiers.get("guidance_type", "soft"),
            "usage_note": tiers.get("usage_note", ""),
            "visited_regions": tiers.get("visited_regions", []),
            "priority_hierarchy": {
                "high_priority_unvisited": priority("tier_1_high_priority_unvisited", []),
                "lower_priority_visited_or_repetitive": priority("tier_2_de_prioritized_visited", []),
                "lower_priority_safety_concerns": priority("tier_3_safety_concerns", []),
                "lower_priority_low_family_interest": priority("tier_4_low_interest", []),
                "lower_priority_distance_or_cost": priority("tier_5_distance_cost", []),
            },
        },
        "visited_cities": [
            {
                "city": c.get("city"), "country": c.get("country"),
                "date": c.get("date"), "rating": c.get("rating"), "notes": c.get("notes"),
            }
            for c in visited_cities
        ],
        "trip_history": trip_history,
        "pending_reviews": pending_reviews,
        "last_updated": profile.get("last_updated"),
    }
