"""
Tests for Layer 3: Context & Memory Management.

Covers: working-memory budget trimming, session-history migration and
provenance, long-term-memory migration and category/relevance retrieval,
and - most importantly - the trust rule that prevents V's own past prose
from being treated as evidence that an external action actually occurred.

Run with: python test_context_memory.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

TEST_DIR = os.path.dirname(__file__)
MEMORY_FILE = os.path.join(TEST_DIR, "memory.json")
MEMORY_BAK = os.path.join(TEST_DIR, "memory.json.bak")
HISTORY_FILE = os.path.join(TEST_DIR, "conversation_history.json")
HISTORY_BAK = os.path.join(TEST_DIR, "conversation_history.json.bak")


def _cleanup():
    for f in (MEMORY_FILE, MEMORY_BAK, HISTORY_FILE, HISTORY_BAK):
        if os.path.exists(f):
            os.remove(f)


# ---------------------------------------------------------------------------
# Long-term memory: migration
# ---------------------------------------------------------------------------

def test_memory_migration_preserves_all_old_entries():
    _cleanup()
    import memory_store
    old_flat = {"github_username": "In0sukeSama", "favorite_editor": "VS Code", "project": "V"}
    with open(MEMORY_FILE, "w") as f:
        json.dump(old_flat, f)

    data = memory_store._load()
    assert data["version"] == 2
    keys = {m["key"] for m in data["memories"]}
    assert keys == set(old_flat.keys()), f"Expected all old keys preserved, got: {keys}"
    for m in data["memories"]:
        assert m["value"] == old_flat[m["key"]]
        assert m["source"] == "legacy"
        assert m["verified"] is True
    print("test_memory_migration_preserves_all_old_entries: PASS")


def test_memory_migration_writes_backup_not_destroying_original_intent():
    _cleanup()
    import memory_store
    old_flat = {"x": "y"}
    with open(MEMORY_FILE, "w") as f:
        json.dump(old_flat, f)
    memory_store._load()
    assert os.path.exists(MEMORY_BAK), "Backup of the old-format file must be written before migrating"
    with open(MEMORY_BAK) as f:
        backup_content = json.load(f)
    assert backup_content == old_flat, "Backup must contain the exact original data"
    print("test_memory_migration_writes_backup_not_destroying_original_intent: PASS")


def test_memory_already_v2_loads_without_remigration():
    _cleanup()
    import memory_store
    v2_data = {"version": 2, "memories": [{"key": "a", "value": "b", "category": "general", "source": "user", "verified": True, "ts": time.time()}]}
    with open(MEMORY_FILE, "w") as f:
        json.dump(v2_data, f)
    data = memory_store._load()
    assert data == v2_data
    assert not os.path.exists(MEMORY_BAK), "No migration should occur for an already-v2 file, so no backup should be created"
    print("test_memory_already_v2_loads_without_remigration: PASS")


def test_memory_categories_and_relevance_retrieval():
    _cleanup()
    import memory_store
    memory_store.remember("github_username", "In0sukeSama")
    memory_store.remember("favorite_editor", "VS Code")
    memory_store.remember("v_project_status", "layer 3 in progress")

    data = memory_store._load()
    by_key = {m["key"]: m for m in data["memories"]}
    assert by_key["github_username"]["category"] == "identity"
    assert by_key["favorite_editor"]["category"] == "preference"
    assert by_key["v_project_status"]["category"] == "project"

    relevant = memory_store.get_relevant("what's the weather today")
    keys = {m["key"] for m in relevant}
    assert "github_username" in keys, "Identity category should always be included"

    relevant2 = memory_store.get_relevant("continue working on the v project")
    keys2 = {m["key"] for m in relevant2}
    assert "v_project_status" in keys2, f"Project-relevant query should retrieve project memory, got: {keys2}"
    print("test_memory_categories_and_relevance_retrieval: PASS")


def test_memory_irrelevant_memories_not_injected():
    _cleanup()
    import memory_store
    memory_store.remember("favorite_editor", "VS Code")
    memory_store.remember("v_project_status", "layer 3 in progress")

    relevant = memory_store.get_relevant("what time is it in tokyo")
    keys = {m["key"] for m in relevant}
    assert "favorite_editor" not in keys
    assert "v_project_status" not in keys
    print("test_memory_irrelevant_memories_not_injected: PASS")


def test_remember_recall_list_memories_still_work():
    _cleanup()
    import tools
    r1 = tools.remember("test_key", "test_value")
    assert "test_key" in r1
    r2 = tools.recall("test_key")
    assert "test_value" in r2
    r3 = tools.recall("testkey")
    assert "test_value" in r3
    r4 = tools.list_memories()
    assert "test_value" in r4
    print("test_remember_recall_list_memories_still_work: PASS")


# ---------------------------------------------------------------------------
# Session history: migration and provenance
# ---------------------------------------------------------------------------

def test_session_history_migration_preserves_old_turns():
    _cleanup()
    import brain as brain_module
    old_format = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(old_format, f)

    summary, turns = brain_module.JarvisBrain._load_session_history()
    assert summary == ""
    contents = [t["content"] for t in turns]
    assert "hello" in contents and "hi there" in contents
    print("test_session_history_migration_preserves_old_turns: PASS")


def test_session_history_migration_legacy_assistant_turns_not_grounded():
    _cleanup()
    import brain as brain_module
    old_format = [
        {"role": "user", "content": "delete report.pdf"},
        {"role": "assistant", "content": "Done, I deleted report.pdf."},
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(old_format, f)

    _, turns = brain_module.JarvisBrain._load_session_history()
    assistant_turn = next(t for t in turns if t["role"] == "assistant")
    assert assistant_turn["grounded"] is False
    assert assistant_turn["tool_outcome"] == "none"
    print("test_session_history_migration_legacy_assistant_turns_not_grounded: PASS")


def test_session_history_v2_loads_directly():
    _cleanup()
    import brain as brain_module
    v2 = {"version": 2, "summary": "prior context", "turns": [
        {"role": "user", "content": "hi", "source": "user", "verified": True, "ts": time.time()}
    ]}
    with open(HISTORY_FILE, "w") as f:
        json.dump(v2, f)
    summary, turns = brain_module.JarvisBrain._load_session_history()
    assert summary == "prior context"
    assert len(turns) == 1
    print("test_session_history_v2_loads_directly: PASS")


def test_saved_history_contains_provenance_fields():
    _cleanup()
    import brain as brain_module
    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = ""
    v.provenance = []
    v.history = [{"role": "system", "content": "sys"}]
    v._record_turn("user", "hello", source="user", verified=True)
    v._record_turn("assistant", "hi", source="model", grounded=False, tool_outcome="none")
    v._save_turns()

    with open(HISTORY_FILE) as f:
        saved = json.load(f)
    assert saved["version"] == 2
    assert saved["turns"][0]["source"] == "user"
    assert saved["turns"][1]["source"] == "model"
    assert saved["turns"][1]["grounded"] is False
    print("test_saved_history_contains_provenance_fields: PASS")


def test_saved_history_is_safe_to_reload():
    _cleanup()
    import brain as brain_module
    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = "earlier stuff"
    v.provenance = []
    v.history = [{"role": "system", "content": "sys"}]
    v._record_turn("user", "hello again", source="user", verified=True)
    v._save_turns()

    summary, turns = brain_module.JarvisBrain._load_session_history()
    assert summary == "earlier stuff"
    assert turns[0]["content"] == "hello again"
    print("test_saved_history_is_safe_to_reload: PASS")


# ---------------------------------------------------------------------------
# Critical trust rule: model prose is never authoritative evidence
# ---------------------------------------------------------------------------

def test_action_verification_failed_tool_not_treated_as_success():
    _cleanup()
    import brain as brain_module
    bad_turn = [
        {"role": "user", "content": "delete ghost.txt"},
        {"role": "assistant", "content": "Done, ghost.txt has been deleted."},
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(bad_turn, f)

    _, turns = brain_module.JarvisBrain._load_session_history()
    assistant_turn = next(t for t in turns if t["role"] == "assistant")
    assert assistant_turn["grounded"] is False
    assert assistant_turn["tool_outcome"] == "none"
    print("test_action_verification_failed_tool_not_treated_as_success: PASS")


def test_action_verification_real_success_is_marked_grounded():
    _cleanup()
    import brain as brain_module
    import tools
    from tool_registry import registry, Tool, RISK_SAFE
    from test_helpers import FakeProvider, make_tool_call_response, make_text_response

    def ok_tool():
        return "genuinely succeeded"

    registry.register(Tool(name="__ctx_test_ok__", description="", parameters={"type": "object", "properties": {}, "required": []}, handler=ok_tool, risk=RISK_SAFE))

    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = ""
    v.provenance = []
    v.pending_actions = []
    v.provider = FakeProvider([
        make_tool_call_response([("__ctx_test_ok__", {}, "c1")]),
        make_text_response("All set."),
    ])
    v.history = [v._build_system_message()]

    v.ask("do the thing")

    assistant_entries = [t for t in v.provenance if t["role"] == "assistant"]
    assert assistant_entries, "Expected an assistant provenance entry"
    assert assistant_entries[-1]["grounded"] is True
    assert assistant_entries[-1]["tool_outcome"] == "success"
    print("test_action_verification_real_success_is_marked_grounded: PASS")


# ---------------------------------------------------------------------------
# Working memory: context budget
# ---------------------------------------------------------------------------

def test_context_budget_trims_old_tool_noise():
    _cleanup()
    import brain as brain_module

    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = ""
    v.provenance = []
    v.pending_actions = []
    from test_helpers import FakeProvider, make_text_response
    v.provider = FakeProvider([make_text_response("condensed summary of old turns") for _ in range(10)])
    v.history = [{"role": "system", "content": "sys"}]

    old_budget = brain_module.CONTEXT_TOKEN_BUDGET
    old_keep = brain_module.RECENT_MESSAGES_KEEP
    brain_module.CONTEXT_TOKEN_BUDGET = 200
    brain_module.RECENT_MESSAGES_KEEP = 4
    try:
        for i in range(20):
            v.history.append({"role": "user", "content": f"old message {i}"})
            v.history.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}]})
            v.history.append({"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 200})
            v.history.append({"role": "assistant", "content": f"reply {i}"})

        before_len = len(v.history)
        v._apply_context_budget()
        after_len = len(v.history)
        assert after_len < before_len, "Expected trimming to reduce history size"

        remaining_tool_msgs = [m for m in v.history if m.get("role") == "tool"]
        remaining_toolcall_assistants = [m for m in v.history if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(remaining_tool_msgs) == len(remaining_toolcall_assistants), "Tool call/result pairs must be removed together, never orphaned"
        print("test_context_budget_trims_old_tool_noise: PASS")
    finally:
        brain_module.CONTEXT_TOKEN_BUDGET = old_budget
        brain_module.RECENT_MESSAGES_KEEP = old_keep


def test_context_budget_preserves_recent_messages():
    _cleanup()
    import brain as brain_module

    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = ""
    v.provenance = []
    v.pending_actions = []
    from test_helpers import FakeProvider, make_text_response
    v.provider = FakeProvider([make_text_response("condensed summary of old turns") for _ in range(10)])
    v.history = [{"role": "system", "content": "sys"}]

    old_budget = brain_module.CONTEXT_TOKEN_BUDGET
    old_keep = brain_module.RECENT_MESSAGES_KEEP
    brain_module.CONTEXT_TOKEN_BUDGET = 100
    brain_module.RECENT_MESSAGES_KEEP = 3
    try:
        for i in range(15):
            v.history.append({"role": "user", "content": f"msg {i}"})
        v._apply_context_budget()
        tail_contents = [m["content"] for m in v.history[-3:]]
        assert "msg 14" in tail_contents, f"Most recent message must survive trimming, got tail: {tail_contents}"
        print("test_context_budget_preserves_recent_messages: PASS")
    finally:
        brain_module.CONTEXT_TOKEN_BUDGET = old_budget
        brain_module.RECENT_MESSAGES_KEEP = old_keep


def test_context_budget_no_trim_when_under_budget():
    _cleanup()
    import brain as brain_module
    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.session_summary = ""
    v.provenance = []
    v.pending_actions = []
    from test_helpers import FakeProvider, make_text_response
    v.provider = FakeProvider([make_text_response("condensed summary of old turns") for _ in range(10)])
    v.history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    before = list(v.history)
    v._apply_context_budget()
    assert v.history == before, "Should not modify history when already under budget"
    print("test_context_budget_no_trim_when_under_budget: PASS")


if __name__ == "__main__":
    test_memory_migration_preserves_all_old_entries()
    test_memory_migration_writes_backup_not_destroying_original_intent()
    test_memory_already_v2_loads_without_remigration()
    test_memory_categories_and_relevance_retrieval()
    test_memory_irrelevant_memories_not_injected()
    test_remember_recall_list_memories_still_work()

    test_session_history_migration_preserves_old_turns()
    test_session_history_migration_legacy_assistant_turns_not_grounded()
    test_session_history_v2_loads_directly()
    test_saved_history_contains_provenance_fields()
    test_saved_history_is_safe_to_reload()

    test_action_verification_failed_tool_not_treated_as_success()
    test_action_verification_real_success_is_marked_grounded()

    test_context_budget_trims_old_tool_noise()
    test_context_budget_preserves_recent_messages()
    test_context_budget_no_trim_when_under_budget()

    _cleanup()
    print("\nAll context/memory tests passed.")
