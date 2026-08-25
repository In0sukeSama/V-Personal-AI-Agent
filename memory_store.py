"""
Long-term memory store for V.

Replaces the old flat key/value memory.json with a lightweight structured
format: a list of memory entries, each with a category, source, and
verification state. No database, no embeddings, no vector search - just a
JSON file and simple keyword-based relevance scoring.

SCHEMA (v2):
{
  "version": 2,
  "memories": [
    {"key": str, "value": str, "category": str, "source": str,
     "verified": bool, "ts": float}
  ]
}
"""

import os
import json
import time

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
MEMORY_BACKUP_FILE = os.path.join(os.path.dirname(__file__), "memory.json.bak")

# Default relevance retrieval limit - how many non-identity memories get
# injected into context for a given request. Configurable via env var.
DEFAULT_RETRIEVAL_LIMIT = int(os.getenv("V_MEMORY_RETRIEVAL_LIMIT", "5"))

# Common short words excluded from relevance scoring - without this, "in" or
# "the" appearing in both a query and an unrelated memory's value produces a
# false-positive overlap match.
_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or",
    "for", "with", "what", "whats", "what's", "how", "who", "do", "does",
    "did", "i", "my", "me", "you", "your", "this", "that", "was", "were",
}

# Categories that are cheap and broadly useful enough to always include,
# regardless of relevance scoring (e.g. a username needed for "open my
# github" regardless of how that request is phrased).
ALWAYS_INCLUDE_CATEGORIES = {"identity"}

# Simple keyword hints for guessing a category when one isn't specified.
# Not exhaustive by design - falls back to "general" for anything else.
_CATEGORY_HINTS = {
    "identity": ["username", "name", "email", "handle", "account"],
    "preference": ["favorite", "prefer", "like", "dislike", "editor", "color", "style"],
    "project": ["project", "repo", "repository", "codebase"],
    "routine": ["routine", "schedule", "usually", "every day", "every morning"],
}


def _guess_category(key: str) -> str:
    key_lower = key.lower()
    for category, hints in _CATEGORY_HINTS.items():
        if any(hint in key_lower for hint in hints):
            return category
    return "general"


def _migrate_flat_dict(flat: dict) -> dict:
    """Upgrade an old {key: value} memory.json into the v2 structured format.
    Every existing memory is preserved - none are discarded. Legacy entries
    are marked source="legacy" (we can't retroactively know if they came
    from an explicit user statement or a model inference) and verified=True
    (nothing suggests they're unreliable, and they've been in active use)."""
    now = time.time()
    memories = [
        {
            "key": key,
            "value": value,
            "category": _guess_category(key),
            "source": "legacy",
            "verified": True,
            "ts": now,
        }
        for key, value in flat.items()
    ]
    return {"version": 2, "memories": memories}


def _load() -> dict:
    """Load memory.json, transparently migrating the old flat format if
    found. Never deletes the original file without first writing a backup."""
    if not os.path.exists(MEMORY_FILE):
        return {"version": 2, "memories": []}

    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        # Corrupt/unreadable file - don't destroy it, just start fresh in
        # memory. The broken file stays on disk for manual inspection.
        return {"version": 2, "memories": []}

    if isinstance(data, dict) and data.get("version") == 2:
        return data

    if isinstance(data, dict):
        # Old flat {key: value} format - back up before migrating.
        try:
            if not os.path.exists(MEMORY_BACKUP_FILE):
                with open(MEMORY_BACKUP_FILE, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass
        migrated = _migrate_flat_dict(data)
        _save(migrated)
        return migrated

    return {"version": 2, "memories": []}


def _save(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def remember(key: str, value: str, category: str = None, source: str = "user") -> str:
    """Save a fact to long-term memory. Overwrites any existing entry with
    the same key. `source` defaults to "user" since remember() is invoked
    when the user has told V something directly - this is a real,
    legitimate source that doesn't require tool verification to be valid."""
    data = _load()
    category = category or _guess_category(key)
    entry = {
        "key": key,
        "value": value,
        "category": category,
        "source": source,
        "verified": True,
        "ts": time.time(),
    }
    data["memories"] = [m for m in data["memories"] if m["key"] != key]
    data["memories"].append(entry)
    _save(data)
    return f"Locked it in - {key} is now {value}."


def recall(key: str) -> str:
    """Retrieve a fact by key, with fuzzy matching (normalized substring
    overlap) since the model may not phrase the same key identically
    across turns."""
    data = _load()

    for m in data["memories"]:
        if m["key"] == key:
            return f"{m['key']}: {m['value']}"

    key_norm = key.lower().replace(" ", "").replace("_", "")
    for m in data["memories"]:
        stored_norm = m["key"].lower().replace(" ", "").replace("_", "")
        if key_norm in stored_norm or stored_norm in key_norm:
            return f"{m['key']}: {m['value']}"

    return f"No memory logged for '{key}' yet."


def list_memories() -> str:
    """Return every memory currently stored, across all categories."""
    data = _load()
    if not data["memories"]:
        return "Nothing saved in memory yet."
    return "\n".join(f"{m['key']}: {m['value']}" for m in data["memories"])


def get_relevant(query_text: str, limit: int = None) -> list:
    """Lightweight relevance retrieval - no embeddings, just keyword overlap
    plus always-included cheap categories. Returns a list of memory dicts,
    not formatted text, so the caller can format however's needed.

    Designed to be swappable later for semantic retrieval without changing
    callers - the contract is just "give me the memories relevant to this
    text", however that's computed.
    """
    limit = limit or DEFAULT_RETRIEVAL_LIMIT
    data = _load()
    memories = data["memories"]
    if not memories:
        return []

    query_words = set(query_text.lower().split()) - _STOPWORDS if query_text else set()

    always_included = [m for m in memories if m["category"] in ALWAYS_INCLUDE_CATEGORIES]
    candidates = [m for m in memories if m["category"] not in ALWAYS_INCLUDE_CATEGORIES]

    def score(m):
        text = f"{m['key']} {m['value']} {m['category']}".lower()
        text_words = set(text.replace("_", " ").split()) - _STOPWORDS
        return len(query_words & text_words)

    scored = sorted(candidates, key=score, reverse=True)
    relevant = [m for m in scored if score(m) > 0][:limit]

    # De-dupe in case a memory is in both an always-included category and
    # somehow also scored (shouldn't happen given the split above, but keep
    # keys unique defensively).
    seen_keys = set()
    result = []
    for m in always_included + relevant:
        if m["key"] not in seen_keys:
            result.append(m)
            seen_keys.add(m["key"])
    return result
