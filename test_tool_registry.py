"""
Small standalone tests for the Tool Registry itself (not the individual
tools' business logic - those are exercised through it).

Run with: python test_tool_registry.py
"""

from tool_registry import ToolRegistry, Tool, register_tool, registry
import json


def test_register_and_execute():
    reg = ToolRegistry()

    def add(a: int, b: int) -> str:
        return str(a + b)

    reg.register(Tool(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
        handler=add,
    ))

    assert reg.exists("add")
    result = reg.execute("add", {"a": 2, "b": 3})
    assert result["success"] is True
    assert result["result"] == "5"
    print("test_register_and_execute: PASS")


def test_unknown_tool_returns_structured_error():
    reg = ToolRegistry()
    result = reg.execute("does_not_exist", {})
    assert result["success"] is False
    assert "Unknown tool" in result["error"]
    print("test_unknown_tool_returns_structured_error: PASS")


def test_exception_in_tool_does_not_propagate():
    reg = ToolRegistry()

    def boom():
        raise RuntimeError("something broke")

    reg.register(Tool(name="boom", description="", parameters={}, handler=boom))
    result = reg.execute("boom", {})
    assert result["success"] is False
    assert "something broke" in result["error"]
    print("test_exception_in_tool_does_not_propagate: PASS")


def test_bad_arguments_reported_cleanly():
    reg = ToolRegistry()

    def needs_x(x: int) -> str:
        return str(x)

    reg.register(Tool(name="needs_x", description="", parameters={}, handler=needs_x))
    result = reg.execute("needs_x", {"y": 1})  # wrong arg name
    assert result["success"] is False
    print("test_bad_arguments_reported_cleanly: PASS")


def test_duplicate_registration_raises():
    reg = ToolRegistry()
    reg.register(Tool(name="dup", description="", parameters={}, handler=lambda: "ok"))
    try:
        reg.register(Tool(name="dup", description="", parameters={}, handler=lambda: "ok2"))
        assert False, "Expected ValueError for duplicate registration"
    except ValueError:
        pass
    print("test_duplicate_registration_raises: PASS")


def test_get_llm_tools_shape():
    reg = ToolRegistry()
    reg.register(Tool(
        name="ping",
        description="ping tool",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda: "pong",
    ))
    llm_tools = reg.get_llm_tools()
    assert len(llm_tools) == 1
    assert llm_tools[0]["type"] == "function"
    assert llm_tools[0]["function"]["name"] == "ping"
    print("test_get_llm_tools_shape: PASS")


def test_decorator_registers_into_shared_registry():
    from tool_registry import registry as shared_registry

    @register_tool(
        name="__test_decorator_tool__",
        description="temp test tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def temp_tool():
        return "ok"

    assert shared_registry.exists("__test_decorator_tool__")
    result = shared_registry.execute("__test_decorator_tool__", {})
    assert result["success"] is True
    assert result["result"] == "ok"
    print("test_decorator_registers_into_shared_registry: PASS")


def test_timeout_returns_structured_failure_quickly():
    """The core isolation guarantee: a hung tool must not block the caller
    for longer than its configured timeout. We assert on elapsed wall-clock
    time, not just the returned value, so a fake 'sleep-then-check' timeout
    implementation would fail this test."""
    import time

    reg = ToolRegistry()

    def hangs_forever():
        time.sleep(5)
        return "should never be waited for"

    reg.register(Tool(name="hangs", description="", parameters={}, handler=hangs_forever, timeout=1))

    start = time.time()
    result = reg.execute("hangs", {})
    elapsed = time.time() - start

    assert elapsed < 2, f"execute() blocked for {elapsed:.2f}s - timeout isolation isn't working"
    assert result["success"] is False
    assert result["error_type"] == "timeout"
    assert result["tool"] == "hangs"
    assert "timed out" in result["message"]
    print(f"test_timeout_returns_structured_failure_quickly: PASS (returned in {elapsed:.2f}s, not 5s)")


def test_timeout_does_not_block_subsequent_calls():
    """After one tool times out, the registry must still work normally for
    the next call - a hung tool's abandoned thread shouldn't poison anything."""
    import time

    reg = ToolRegistry()

    def hangs_forever():
        time.sleep(5)
        return "irrelevant"

    def instant():
        return "fine"

    reg.register(Tool(name="hangs2", description="", parameters={}, handler=hangs_forever, timeout=1))
    reg.register(Tool(name="instant", description="", parameters={}, handler=instant))

    timeout_result = reg.execute("hangs2", {})
    assert timeout_result["error_type"] == "timeout"

    start = time.time()
    ok_result = reg.execute("instant", {})
    elapsed = time.time() - start

    assert elapsed < 1
    assert ok_result == {"success": True, "result": "fine"}
    print("test_timeout_does_not_block_subsequent_calls: PASS")


def test_default_timeout_applied_when_not_specified():
    from tool_registry import DEFAULT_TIMEOUT_SECONDS
    reg = ToolRegistry()
    reg.register(Tool(name="no_timeout_given", description="", parameters={}, handler=lambda: "x"))
    tool = reg.get("no_timeout_given")
    assert tool.timeout == DEFAULT_TIMEOUT_SECONDS
    print("test_default_timeout_applied_when_not_specified: PASS")


def test_custom_timeout_override_respected():
    reg = ToolRegistry()
    reg.register(Tool(name="custom", description="", parameters={}, handler=lambda: "x", timeout=30))
    assert reg.get("custom").timeout == 30
    print("test_custom_timeout_override_respected: PASS")


def test_multiple_tools_sequentially():
    reg = ToolRegistry()
    reg.register(Tool(name="a", description="", parameters={}, handler=lambda: "A"))
    reg.register(Tool(name="b", description="", parameters={}, handler=lambda: "B"))
    reg.register(Tool(name="c", description="", parameters={}, handler=lambda: "C"))

    results = [reg.execute(n, {}) for n in ("a", "b", "c")]
    assert [r["result"] for r in results] == ["A", "B", "C"]
    print("test_multiple_tools_sequentially: PASS")


# ---------------------------------------------------------------------------
# Permission / consequence system tests
# ---------------------------------------------------------------------------

def test_safe_tool_executes_immediately():
    from tool_registry import RISK_SAFE
    reg = ToolRegistry()
    reg.register(Tool(name="safe_one", description="", parameters={}, handler=lambda: "ok", risk=RISK_SAFE))
    result = reg.execute("safe_one", {})
    assert result["success"] is True
    assert result["result"] == "ok"
    print("test_safe_tool_executes_immediately: PASS")


def test_low_risk_tool_executes_without_confirmation():
    from tool_registry import RISK_LOW
    reg = ToolRegistry()
    reg.register(Tool(name="low_one", description="", parameters={}, handler=lambda: "done", risk=RISK_LOW))
    result = reg.execute("low_one", {})
    assert result["success"] is True
    assert result["result"] == "done"
    print("test_low_risk_tool_executes_without_confirmation: PASS")


def test_destructive_tool_requires_confirmation_and_does_not_run():
    from tool_registry import RISK_DESTRUCTIVE

    called = {"ran": False}

    def destroy():
        called["ran"] = True
        return "destroyed"

    reg = ToolRegistry()
    reg.register(Tool(name="destroy", description="", parameters={}, handler=destroy, risk=RISK_DESTRUCTIVE))

    result = reg.execute("destroy", {"target": "report.pdf"} if False else {})
    assert result["success"] is False
    assert result["error_type"] == "confirmation_required"
    assert result["tool"] == "destroy"
    assert "pending_action" in result
    assert called["ran"] is False, "The destructive handler must NOT have run yet!"
    print("test_destructive_tool_requires_confirmation_and_does_not_run: PASS")


def test_confirmed_pending_action_executes_exactly_once():
    from tool_registry import RISK_DESTRUCTIVE

    call_count = {"n": 0}

    def destroy(target):
        call_count["n"] += 1
        return f"destroyed {target}"

    reg = ToolRegistry()
    reg.register(Tool(name="destroy2", description="", parameters={}, handler=destroy, risk=RISK_DESTRUCTIVE))

    blocked = reg.execute("destroy2", {"target": "report.pdf"})
    assert blocked["error_type"] == "confirmation_required"
    pending = blocked["pending_action"]

    confirmed = reg.execute_confirmed(pending)
    assert confirmed["success"] is True
    assert confirmed["result"] == "destroyed report.pdf"
    assert call_count["n"] == 1
    print("test_confirmed_pending_action_executes_exactly_once: PASS")


def test_consequential_risk_defaults_to_confirmation_required():
    from tool_registry import RISK_CONSEQUENTIAL
    reg = ToolRegistry()
    reg.register(Tool(name="send_thing", description="", parameters={}, handler=lambda: "sent", risk=RISK_CONSEQUENTIAL))
    result = reg.execute("send_thing", {})
    assert result["error_type"] == "confirmation_required"
    print("test_consequential_risk_defaults_to_confirmation_required: PASS")


def test_explicit_requires_confirmation_overrides_risk_default():
    from tool_registry import RISK_SAFE, RISK_DESTRUCTIVE
    reg = ToolRegistry()
    # A "safe" risk tool explicitly forced to require confirmation anyway
    reg.register(Tool(
        name="forced_confirm", description="", parameters={}, handler=lambda: "x",
        risk=RISK_SAFE, requires_confirmation=True,
    ))
    result = reg.execute("forced_confirm", {})
    assert result["error_type"] == "confirmation_required"

    # A "destructive" risk tool explicitly forced to skip confirmation
    reg.register(Tool(
        name="forced_auto", description="", parameters={}, handler=lambda: "y",
        risk=RISK_DESTRUCTIVE, requires_confirmation=False,
    ))
    result2 = reg.execute("forced_auto", {})
    assert result2["success"] is True
    print("test_explicit_requires_confirmation_overrides_risk_default: PASS")


def test_multi_tool_batch_stops_at_confirmation_gate():
    """Simulates the brain.py-level behavior: a batch of tool calls where a
    later one requires confirmation must not let subsequent calls in the
    same batch execute."""
    from tool_registry import RISK_SAFE, RISK_DESTRUCTIVE

    executed = []

    def safe_action():
        executed.append("safe_action")
        return "ok"

    def destructive_action():
        executed.append("destructive_action")  # should NEVER be appended
        return "destroyed"

    def should_not_run():
        executed.append("should_not_run")  # should NEVER be appended
        return "ran anyway"

    reg = ToolRegistry()
    reg.register(Tool(name="safe_action", description="", parameters={}, handler=safe_action, risk=RISK_SAFE))
    reg.register(Tool(name="destructive_action", description="", parameters={}, handler=destructive_action, risk=RISK_DESTRUCTIVE))
    reg.register(Tool(name="should_not_run", description="", parameters={}, handler=should_not_run, risk=RISK_SAFE))

    # Simulate brain.py's batch-processing logic directly against the registry
    call_order = ["safe_action", "destructive_action", "should_not_run"]
    blocked = False
    for name in call_order:
        if blocked:
            continue  # brain.py would substitute a "skipped" message here
        result = reg.execute(name, {})
        if result.get("error_type") == "confirmation_required":
            blocked = True

    assert executed == ["safe_action"], f"Expected only safe_action to run, got: {executed}"
    print("test_multi_tool_batch_stops_at_confirmation_gate: PASS")


# ---------------------------------------------------------------------------
# Regression tests for the "repeated confirmation ramble" bug
# (brain.py used to make a second, unbounded model call just to generate the
# confirmation text - that call could ramble into a long repeated block.
# Fix: confirmation text is now built deterministically from the
# PendingAction, with zero additional model calls.)
# ---------------------------------------------------------------------------

from test_helpers import FakeProvider, make_tool_call_response, make_text_response


def _make_test_brain(responses):
    import brain as brain_module
    v = brain_module.JarvisBrain.__new__(brain_module.JarvisBrain)
    v.provider = FakeProvider(responses)
    v.session_summary = ""
    v.provenance = []
    v.history = [v._build_system_message()]
    v.pending_actions = []
    return v


def test_confirmation_bug_one_confirmation_only():
    """Test 1: exactly one clean confirmation message, exactly one model call."""
    from tool_registry import RISK_DESTRUCTIVE

    reg_tool_calls = {"n": 0}
    def spy_delete(path):
        reg_tool_calls["n"] += 1
        return f"deleted {path}"

    registry.register(Tool(
        name="__test_confirm_delete__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy_delete, risk=RISK_DESTRUCTIVE,
    ))

    responses = [make_tool_call_response([
        ("__test_confirm_delete__", json.loads('{"path": "now_me.txt"}'), "c1")
    ])]
    v = _make_test_brain(responses)
    reply = v.ask('delete now_me.txt')

    assert v.provider.call_count == 1, f"Expected exactly 1 model call, got {v.provider.call_count}"
    assert reply.count("?") <= 1, f"Expected a single clean question, got: {reply}"
    assert "now_me.txt" in reply
    print("test_confirmation_bug_one_confirmation_only: PASS")


def test_confirmation_bug_no_repeated_tool_calls():
    """Test 2: the blocked handler must be invoked zero times before confirmation."""
    from tool_registry import RISK_DESTRUCTIVE

    calls = {"n": 0}
    def spy(path):
        calls["n"] += 1
        return "done"

    registry.register(Tool(
        name="__test_confirm_delete2__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))
    responses = [make_tool_call_response([
        ("__test_confirm_delete2__", json.loads('{"path": "x.txt"}'), "c1")
    ])]
    v = _make_test_brain(responses)
    v.ask("delete x.txt")
    assert calls["n"] == 0
    print("test_confirmation_bug_no_repeated_tool_calls: PASS")


def test_confirmation_bug_confirmed_executes_once():
    """Test 3: user says yes -> handler invoked exactly once."""
    from tool_registry import RISK_DESTRUCTIVE

    calls = {"n": 0}
    def spy(path):
        calls["n"] += 1
        return "done"

    registry.register(Tool(
        name="__test_confirm_delete3__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))
    responses = [make_tool_call_response([
        ("__test_confirm_delete3__", json.loads('{"path": "y.txt"}'), "c1")
    ])]
    v = _make_test_brain(responses)
    v.ask("delete y.txt")
    v.provider = FakeProvider([make_text_response("Done.")])
    v.ask("yes")
    assert calls["n"] == 1
    assert v.pending_actions == []
    print("test_confirmation_bug_confirmed_executes_once: PASS")


def test_confirmation_bug_second_action_no_stale_state():
    """Test 4: after confirming deletion of file A, requesting deletion of
    file B must create exactly one NEW pending confirmation - no leftover
    state from file A's now-resolved pending action."""
    from tool_registry import RISK_DESTRUCTIVE

    calls = []
    def spy(path):
        calls.append(path)
        return f"deleted {path}"

    registry.register(Tool(
        name="__test_confirm_delete4__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))

    # File A
    v = _make_test_brain([make_tool_call_response([
        ("__test_confirm_delete4__", json.loads('{"path": "fileA.txt"}'), "c1")
    ])])
    v.ask("delete fileA.txt")
    v.provider = FakeProvider([make_text_response("Done with A.")])
    v.ask("yes")
    assert calls == ["fileA.txt"]
    assert v.pending_actions == []

    # File B - fresh request
    v.provider = FakeProvider([make_tool_call_response([
        ("__test_confirm_delete4__", json.loads('{"path": "fileB.txt"}'), "c2")
    ])])
    reply = v.ask("delete fileB.txt")
    assert v.pending_actions
    assert v.pending_actions[0].arguments["path"] == "fileB.txt"
    assert calls == ["fileA.txt"], "fileB must not have executed yet"

    v.provider = FakeProvider([make_text_response("Done with B.")])
    v.ask("yes")
    assert calls == ["fileA.txt", "fileB.txt"]
    assert v.pending_actions == []
    print("test_confirmation_bug_second_action_no_stale_state: PASS")


def test_confirmation_bug_rejection_zero_calls():
    """Test 5: pending deletion + 'no' = handler invoked zero times."""
    from tool_registry import RISK_DESTRUCTIVE
    calls = {"n": 0}
    def spy(path):
        calls["n"] += 1
        return "done"
    registry.register(Tool(
        name="__test_confirm_delete5__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))
    v = _make_test_brain([make_tool_call_response([
        ("__test_confirm_delete5__", json.loads('{"path": "z.txt"}'), "c1")
    ])])
    v.ask("delete z.txt")
    v.provider = FakeProvider([make_text_response("Cancelled.")])
    v.ask("no")
    assert calls["n"] == 0
    assert v.pending_actions == []
    print("test_confirmation_bug_rejection_zero_calls: PASS")


def test_confirmation_bug_superseding_request():
    """Test 6: pending deletion + unrelated new request = deletion never executes."""
    from tool_registry import RISK_DESTRUCTIVE
    calls = {"n": 0}
    def spy(path):
        calls["n"] += 1
        return "done"
    registry.register(Tool(
        name="__test_confirm_delete6__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))
    v = _make_test_brain([make_tool_call_response([
        ("__test_confirm_delete6__", json.loads('{"path": "w.txt"}'), "c1")
    ])])
    v.ask("delete w.txt")
    v.provider = FakeProvider([make_text_response("Sure, opening Spotify.")])
    v.ask("actually open spotify")
    assert calls["n"] == 0
    assert v.pending_actions == []
    print("test_confirmation_bug_superseding_request: PASS")


def test_confirmation_bug_expiration():
    """Test 7: expired pending action + 'yes' = deletion must not execute."""
    import brain as brain_module
    from tool_registry import RISK_DESTRUCTIVE
    calls = {"n": 0}
    def spy(path):
        calls["n"] += 1
        return "done"
    registry.register(Tool(
        name="__test_confirm_delete7__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))
    v = _make_test_brain([make_tool_call_response([
        ("__test_confirm_delete7__", json.loads('{"path": "old.txt"}'), "c1")
    ])])
    v.ask("delete old.txt")
    v.pending_actions[0].created_at -= (brain_module.PENDING_ACTION_TIMEOUT_SECONDS + 10)
    v.provider = FakeProvider([make_text_response("That expired.")])
    v.ask("yes")
    assert calls["n"] == 0
    assert v.pending_actions == []
    print("test_confirmation_bug_expiration: PASS")


def test_confirmation_bug_multi_tool_boundary():
    """Test 8: a multi-tool request stops exactly at the confirmation boundary."""
    from tool_registry import RISK_SAFE, RISK_DESTRUCTIVE
    executed = []
    def safe_open(app_name):
        executed.append(f"open:{app_name}")
        return f"{app_name} opened"
    def destroy(path):
        executed.append(f"destroy:{path}")
        return "destroyed"
    def never_run(app_name):
        executed.append("SHOULD_NOT_RUN")
        return "ran"

    registry.register(Tool(name="__test_mt_open__", description="", parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}, handler=safe_open, risk=RISK_SAFE))
    registry.register(Tool(name="__test_mt_destroy__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, handler=destroy, risk=RISK_DESTRUCTIVE))
    registry.register(Tool(name="__test_mt_after__", description="", parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}, handler=never_run, risk=RISK_SAFE))

    v = _make_test_brain([make_tool_call_response([
        ("__test_mt_open__", json.loads('{"app_name": "explorer"}'), "c1"),
        ("__test_mt_destroy__", json.loads('{"path": "old_project"}'), "c2"),
        ("__test_mt_after__", json.loads('{"app_name": "vscode"}'), "c3")
    ])])
    reply = v.ask("find my old project, delete it, and open vscode")
    assert executed == ["open:explorer"], f"Expected only the safe tool to run, got: {executed}"
    assert v.pending_actions
    assert v.provider.call_count == 1
    print("test_confirmation_bug_multi_tool_boundary: PASS")


def test_batch_confirmation_covers_multiple_files_in_one_request():
    """Regression test for the 'only the first file gets confirmed, the
    second is silently skipped' bug: 'delete 4.txt and 5.txt' must produce
    ONE combined confirmation mentioning both, and a single 'yes' must
    execute both - not just the first."""
    from tool_registry import RISK_DESTRUCTIVE

    deleted = []
    def spy_delete(path):
        deleted.append(path)
        return f"deleted {path}"

    registry.register(Tool(
        name="__test_batch_delete__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy_delete, risk=RISK_DESTRUCTIVE,
    ))

    v = _make_test_brain([make_tool_call_response([
        ("__test_batch_delete__", json.loads('{"path": "4.txt"}'), "c1"),
        ("__test_batch_delete__", json.loads('{"path": "5.txt"}'), "c2")
    ])])
    reply = v.ask("delete 4.txt and 5.txt")

    assert "4.txt" in reply and "5.txt" in reply, f"Confirmation must mention both files: {reply}"
    assert len(v.pending_actions) == 2, f"Expected 2 pending actions, got {len(v.pending_actions)}"
    assert deleted == [], "Neither file should be deleted yet"

    v.provider = FakeProvider([])  # confirming must not call the model at all
    v.ask("yes")
    assert deleted == ["4.txt", "5.txt"], f"Both files should be deleted after one confirmation, got: {deleted}"
    assert v.pending_actions == []
    print("test_batch_confirmation_covers_multiple_files_in_one_request: PASS")


def test_confirmation_resume_never_leaks_internal_notes():
    """Regression test for the '[System note...]' leak bug: after a
    confirmed/rejected action, V's spoken reply must NEVER contain raw
    internal annotation text - it's built deterministically in code and
    never passed through the model for these branches, so there is no
    chance of the model echoing internal bookkeeping text back to the user."""
    from tool_registry import RISK_DESTRUCTIVE

    def spy(path):
        return f"deleted {path}"

    registry.register(Tool(
        name="__test_leak_delete__", description="", parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=spy, risk=RISK_DESTRUCTIVE,
    ))

    forbidden_markers = ["[System note", "System note:", "was confirmed and executed"]

    # Confirmed branch
    v = _make_test_brain([make_tool_call_response([
        ("__test_leak_delete__", json.loads('{"path": "leak_test.txt"}'), "c1")
    ])])
    v.ask("delete leak_test.txt")
    # No model call is scripted for the "yes" turn - if the code tried to
    # call the model here, this test would raise StopIteration and fail,
    # which itself proves the confirm branch is now fully model-call-free.
    v.provider = FakeProvider([])
    reply = v.ask("yes")
    for marker in forbidden_markers:
        assert marker not in reply, f"Leaked internal text found in reply: {reply!r}"
    assert v.provider.call_count == 0, "Confirmed branch must not call the model at all"
    print("test_confirmation_resume_never_leaks_internal_notes (confirmed): PASS")

    # Rejected branch
    v2 = _make_test_brain([make_tool_call_response([
        ("__test_leak_delete__", json.loads('{"path": "leak_test2.txt"}'), "c2")
    ])])
    v2.ask("delete leak_test2.txt")
    v2.provider = FakeProvider([])
    reply2 = v2.ask("no")
    for marker in forbidden_markers:
        assert marker not in reply2
    assert v2.provider.call_count == 0, "Rejected branch must not call the model at all"
    print("test_confirmation_resume_never_leaks_internal_notes (rejected): PASS")


if __name__ == "__main__":
    test_register_and_execute()
    test_unknown_tool_returns_structured_error()
    test_exception_in_tool_does_not_propagate()
    test_bad_arguments_reported_cleanly()
    test_duplicate_registration_raises()
    test_get_llm_tools_shape()
    test_decorator_registers_into_shared_registry()
    test_timeout_returns_structured_failure_quickly()
    test_timeout_does_not_block_subsequent_calls()
    test_default_timeout_applied_when_not_specified()
    test_custom_timeout_override_respected()
    test_multiple_tools_sequentially()
    test_safe_tool_executes_immediately()
    test_low_risk_tool_executes_without_confirmation()
    test_destructive_tool_requires_confirmation_and_does_not_run()
    test_confirmed_pending_action_executes_exactly_once()
    test_consequential_risk_defaults_to_confirmation_required()
    test_explicit_requires_confirmation_overrides_risk_default()
    test_multi_tool_batch_stops_at_confirmation_gate()
    test_confirmation_bug_one_confirmation_only()
    test_confirmation_bug_no_repeated_tool_calls()
    test_confirmation_bug_confirmed_executes_once()
    test_confirmation_bug_second_action_no_stale_state()
    test_confirmation_bug_rejection_zero_calls()
    test_confirmation_bug_superseding_request()
    test_confirmation_bug_expiration()
    test_confirmation_bug_multi_tool_boundary()
    test_batch_confirmation_covers_multiple_files_in_one_request()
    test_confirmation_resume_never_leaks_internal_notes()
    print("\nAll tool_registry tests passed.")
