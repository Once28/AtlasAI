"""
seed_db.py — one-time migration: reads the legacy family_db.json and loads
it into the 5 MongoDB Atlas collections used by db.py / crud.py.

Run once after creating the Atlas cluster and setting MONGO_URI:
    python seed_db.py [path/to/family_db.json]

Safe to re-run for the singleton collections (family_profiles,
mobility_profiles, budget_rules, destination_priorities) — all upserts.
travel_ledger is NOT idempotent: there's no natural unique key across
visited_cities/trip_history/pending_reviews entries in the legacy JSON, so
re-running this will re-insert duplicates there. Drop that collection first
(`db.travel_ledger.drop()` in a Mongo shell, or via Compass) if you need a
clean re-seed after already having seeded once.
"""

import json
import os
import sys

from pymongo import ASCENDING, MongoClient
from pymongo.server_api import ServerApi

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_NAME = "family_travel_db"


def load_legacy_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mobility_profiles(legacy: dict) -> list[dict]:
    mr = legacy.get("mobility_rules", {})
    members = legacy.get("family_profile", {}).get("members", [])
    # The old schema duplicated identical mobility/climate/tour_requirements
    # blocks across every low-mobility member (grandma_01, grandpa_01). This
    # pulls the shared shape from the first one found, since mobility_rules
    # (the old top-level object) is really "the low-mobility group's rules"
    # under a different name — normalizing that duplication is the point of
    # the new mobility_profiles collection.
    low_mobility_member = next(
        (m for m in members if m.get("mobility", {}).get("level") == "low"), {}
    )
    return [
        {
            "profile_id": "group_low_mobility",
            "level": "low",
            "guidance_type": mr.get("guidance_type", "soft"),
            "description": mr.get("description", ""),
            "max_walking_km_per_day": mr.get("max_daily_walking_km"),
            "requires_rest_stops_every_min": low_mobility_member.get("mobility", {}).get(
                "requires_rest_stops_every_min"
            ),
            "needs_elevator_access": mr.get("prefers_elevator_or_ground_floor_lodging"),
            "wheelchair_assist": low_mobility_member.get("mobility", {}).get("wheelchair_assist", False),
            "climate_preference": low_mobility_member.get("climate_preference", {}),
            "tour_requirements": low_mobility_member.get("tour_requirements", {}),
            "max_flight_layovers_preferred": mr.get("max_flight_layovers_preferred"),
            "min_layover_minutes": mr.get("min_layover_minutes"),
        },
        {
            "profile_id": "group_high_mobility",
            "level": "high",
            "guidance_type": "soft",
            "description": "No walking/climate constraints tracked for this group.",
        },
    ]


def build_family_profile_doc(legacy: dict) -> dict:
    fp = legacy.get("family_profile", {})
    mobility_map = {"low": "group_low_mobility", "high": "group_high_mobility"}
    members = [
        {
            "id": m.get("id"),
            "role": m.get("role"),
            "age_range": m.get("age_range"),
            "language": m.get("language"),
            "english_fluency": m.get("english_fluency"),
            # Members now reference a mobility profile instead of embedding
            # a full copy of it — this is what mobility_profiles normalizes.
            "mobility_profile_id": mobility_map.get(
                m.get("mobility", {}).get("level", "high"), "group_high_mobility"
            ),
        }
        for m in fp.get("members", [])
    ]
    return {
        "family_name": fp.get("family_name"),
        "home_base": fp.get("home_base"),
        "default_translators_present": True,
        "translation_note": legacy.get("family_notes", {}).get("translation", ""),
        "members": members,
        "last_updated": legacy.get("last_updated"),
    }


def build_budget_doc(legacy: dict) -> dict:
    b = legacy.get("budget_rules", {})
    return {
        "guidance_type": b.get("guidance_type", "soft"),
        "currency": b.get("currency", "USD"),
        "per_trip_soft_cap": b.get("per_trip_soft_cap"),
        "per_trip_hard_cap": b.get("per_trip_hard_cap"),
        "valuation_heuristics": {
            "roi_logic": b.get("roi_logic", ""),
            "roi_reflections": b.get("roi_reflections", []),
        },
        "historical_roi_precedents": b.get("precedent_examples", []),
        "approval_flow": b.get("approval_flow", ""),
    }


def build_destination_priorities_doc(legacy: dict) -> dict:
    r = legacy.get("regional_travel_history", {})
    ph = r.get("priority_hierarchy", {})
    return {
        "guidance_type": r.get("guidance_type", "soft"),
        "usage_note": r.get("usage_note", ""),
        "visited_regions": r.get("visited_regions", []),
        "tier_1_high_priority_unvisited": ph.get("high_priority_unvisited", []),
        "tier_2_de_prioritized_visited": ph.get("lower_priority_visited_or_repetitive", []),
        "tier_3_safety_concerns": ph.get("lower_priority_safety_concerns", []),
        "tier_4_low_interest": ph.get("lower_priority_low_family_interest", []),
        "tier_5_distance_cost": ph.get("lower_priority_distance_or_cost", []),
    }


def build_ledger_docs(legacy: dict) -> list[dict]:
    docs = []
    for c in legacy.get("visited_cities", []):
        docs.append({**c, "type": "visited_city"})
    for t in legacy.get("trip_history", []):
        docs.append({**t, "type": "trip_history"})
    for p in legacy.get("pending_reviews", []):
        docs.append({**p, "type": "pending_review"})
    return docs


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "family_db.json"
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise SystemExit(
            "Set MONGO_URI in your environment (or a .env file — see .env.example) "
            "before running seed_db.py."
        )

    legacy = load_legacy_json(json_path)
    client = MongoClient(uri, server_api=ServerApi("1"))
    db = client[DB_NAME]

    # Indexes — create_index() is idempotent, safe to run every time.
    db.family_profiles.create_index([("family_name", ASCENDING)], unique=True)
    db.mobility_profiles.create_index([("profile_id", ASCENDING)], unique=True)
    db.travel_ledger.create_index([("type", ASCENDING)])

    family_doc = build_family_profile_doc(legacy)
    db.family_profiles.update_one(
        {"family_name": family_doc["family_name"]}, {"$set": family_doc}, upsert=True
    )
    print(f"family_profiles: upserted '{family_doc['family_name']}'")

    mobility_profiles = build_mobility_profiles(legacy)
    for profile in mobility_profiles:
        db.mobility_profiles.update_one(
            {"profile_id": profile["profile_id"]}, {"$set": profile}, upsert=True
        )
    print(f"mobility_profiles: upserted {len(mobility_profiles)} profiles")

    db.budget_rules.update_one({}, {"$set": build_budget_doc(legacy)}, upsert=True)
    print("budget_rules: upserted singleton doc")

    db.destination_priorities.update_one(
        {}, {"$set": build_destination_priorities_doc(legacy)}, upsert=True
    )
    print("destination_priorities: upserted singleton doc")

    ledger_docs = build_ledger_docs(legacy)
    if ledger_docs:
        result = db.travel_ledger.insert_many(ledger_docs)
        print(f"travel_ledger: inserted {len(result.inserted_ids)} entries")
    else:
        print("travel_ledger: nothing to insert")

    print("\nSeed complete.")


if __name__ == "__main__":
    main()
