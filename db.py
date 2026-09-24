"""
db.py — singleton MongoDB Atlas connection for the family travel agent.

get_client() is wrapped in @st.cache_resource so the PyMongo client (and its
connection pool) is created ONCE per Streamlit server process and reused
across reruns — Streamlit reruns the whole script on every widget
interaction, so without this you'd open a new connection pool on every
keystroke, which will exhaust Atlas M0's tight connection limit fast.

Install: pip install "pymongo[srv]" python-dotenv
"""

import os

import streamlit as st
from bson import ObjectId
from pymongo import MongoClient
from pymongo.server_api import ServerApi

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into os.environ for local dev; no-op if the file doesn't exist
except ImportError:
    pass  # python-dotenv is optional if MONGO_URI is only ever provided via st.secrets

DB_NAME = "family_travel_db"

# Single-family app on an M0 free tier — one document per collection (or one
# document keyed by this name in family_profiles) rather than a users table.
# Centralized here since both crud.py and seed_db.py need the same key.
FAMILY_NAME = "Zeng Family"


def _get_mongo_uri() -> str:
    """
    st.secrets first (Streamlit Cloud deploys / .streamlit/secrets.toml),
    falling back to the MONGO_URI env var (.env locally via python-dotenv
    above) — so the same code runs unmodified in both environments.
    """
    try:
        uri = st.secrets["MONGO_URI"]
        if uri:
            return uri
    except Exception:
        pass  # no secrets.toml, or no MONGO_URI key in it — fall through to env var

    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError(
            "MONGO_URI not found in st.secrets or the environment. "
            "Copy secrets.toml.template to .streamlit/secrets.toml, or "
            ".env.example to .env, and fill in your Atlas connection string."
        )
    return uri


@st.cache_resource(show_spinner=False)
def get_client() -> MongoClient:
    uri = _get_mongo_uri()
    client = MongoClient(uri, server_api=ServerApi("1"))
    client.admin.command("ping")  # fail fast here on bad creds/network, not on the first real query
    return client


def get_db():
    return get_client()[DB_NAME]


def sanitize(value):
    """
    Recursively converts ObjectId fields to strings so query results are
    safe to hand to st.json(), json.dumps(), or an LLM prompt without
    raising "Object of type ObjectId is not JSON serializable". Works on a
    single doc, a list of docs, or nested dicts/lists; anything else (str,
    int, None, ...) passes through unchanged.
    """
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value
