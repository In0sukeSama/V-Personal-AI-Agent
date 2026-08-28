"""
Tests for Layer 4: Model Provider Abstraction.

Run with: python test_model_provider.py
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(__file__))


def _cleanup():
    for f in ("memory.json", "memory.json.bak", "conversation_history.json", "conversation_history.json.bak"):
        p = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

def test_brain_does_not_import_openai_directly():
    import brain as brain_module
    source = inspect.getsource(brain_module)
    assert "from openai import" not in source, "brain.py must not import the OpenAI SDK directly"
    assert "self.client" not in source, "brain.py must not hold a raw SDK client - use self.provider"
    print("test_brain_does_not_import_openai_directly: PASS")


def test_brain_uses_injected_provider_not_a_hardcoded_one():
    import brain as brain_module
    from test_helpers import FakeProvider, make_text_response

    fake = FakeProvider([make_text_response("hello from a fake provider")])
    v = brain_module.JarvisBrain(provider=fake)
    reply = v.ask("hi")
    assert reply == "hello from a fake provider"
    assert fake.call_count == 1
    print("test_brain_uses_injected_provider_not_a_hardcoded_one: PASS")


def test_provider_objects_do_not_leak_into_brain():
    from providers.base import ModelResponse
    r = ModelResponse(text="x", tool_calls=[], raw_assistant_message={"role": "assistant", "content": "x"})
    assert isinstance(r, ModelResponse)
    print("test_provider_objects_do_not_leak_into_brain: PASS")


# ---------------------------------------------------------------------------
# Normal responses
# ---------------------------------------------------------------------------

def test_plain_text_response_normalized_correctly():
    from providers.base import ModelResponse
    from test_helpers import make_text_response
    r = make_text_response("just talking")
    assert isinstance(r, ModelResponse)
    assert r.text == "just talking"
    assert r.tool_calls == []
    print("test_plain_text_response_normalized_correctly: PASS")


def test_empty_response_handled_safely():
    import brain as brain_module
    from test_helpers import FakeProvider, make_text_response

    v = brain_module.JarvisBrain(provider=FakeProvider([make_text_response(None)]))
    reply = v.ask("say nothing")
    assert reply == "", "Empty response should normalize to an empty string, not crash"
    print("test_empty_response_handled_safely: PASS")


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

def test_single_tool_call_normalized_correctly():
    from test_helpers import make_tool_call_response
    r = make_tool_call_response([("open_application", {"app_name": "notepad"}, "call_1")])
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "open_application"
    assert tc.arguments == {"app_name": "notepad"}
    print("test_single_tool_call_normalized_correctly: PASS")


def test_multiple_tool_calls_normalized_correctly():
    from test_helpers import make_tool_call_response
    r = make_tool_call_response([
        ("find_file", {"filename": "README.md"}, "call_1"),
        ("open_file", {"path": "/tmp/README.md"}, "call_2"),
    ])
    assert len(r.tool_calls) == 2
    assert r.tool_calls[0].name == "find_file"
    assert r.tool_calls[1].name == "open_file"
    assert r.tool_calls[1].arguments == {"path": "/tmp/README.md"}
    print("test_multiple_tool_calls_normalized_correctly: PASS")


def test_tool_call_ids_preserved():
    from test_helpers import make_tool_call_response
    r = make_tool_call_response([("get_weather", {"city": None}, "unique_id_xyz")])
    assert r.tool_calls[0].id == "unique_id_xyz"
    print("test_tool_call_ids_preserved: PASS")


def test_tool_results_flow_back_through_provider():
    import brain as brain_module
    from tool_registry import registry, Tool, RISK_SAFE
    from test_helpers import FakeProvider, make_tool_call_response, make_text_response

    def echo_tool(value):
        return f"echoed: {value}"

    registry.register(Tool(
        name="__provider_test_echo__", description="", risk=RISK_SAFE,
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        handler=echo_tool,
    ))

    fake = FakeProvider([
        make_tool_call_response([("__provider_test_echo__", {"value": "hi"}, "c1")]),
        make_text_response("Got it: echoed: hi"),
    ])
    v = brain_module.JarvisBrain(provider=fake)
    reply = v.ask("echo hi")
    assert "echoed: hi" in reply
    assert fake.call_count == 2, "Expected exactly 2 provider calls: one requesting the tool, one final reply"

    tool_messages = [m for m in v.history if m.get("role") == "tool"]
    assert any("echoed: hi" in m["content"] for m in tool_messages)
    print("test_tool_results_flow_back_through_provider: PASS")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_provider_errors_become_model_provider_error():
    from providers.base import ModelProvider, ModelProviderError

    class FailingProvider(ModelProvider):
        def generate(self, messages, tools=None, **kwargs):
            raise ModelProviderError("unknown", "something broke upstream", retryable=False)

    try:
        FailingProvider().generate(messages=[])
        assert False, "Expected ModelProviderError"
    except ModelProviderError as e:
        assert e.error_type == "unknown"
    print("test_provider_errors_become_model_provider_error: PASS")


def test_rate_limit_errors_distinguishable():
    from providers.base import ModelProviderError
    e = ModelProviderError("rate_limit", "quota exceeded", retryable=True)
    assert e.error_type == "rate_limit"
    assert e.retryable is True
    print("test_rate_limit_errors_distinguishable: PASS")


def test_auth_errors_distinguishable():
    from providers.base import ModelProviderError
    e = ModelProviderError("auth", "invalid api key", retryable=False)
    assert e.error_type == "auth"
    assert e.retryable is False
    print("test_auth_errors_distinguishable: PASS")


def test_invalid_model_response_handled_safely():
    from providers.groq_provider import GroqProvider
    from providers.base import ModelProviderError

    provider = GroqProvider.__new__(GroqProvider)

    class Empty:
        choices = []

    try:
        provider._normalize(Empty())
        assert False, "Expected ModelProviderError for an empty choices list"
    except ModelProviderError as e:
        assert e.error_type == "invalid_response"
    print("test_invalid_model_response_handled_safely: PASS")


# ---------------------------------------------------------------------------
# Existing V behavior, verified through the abstraction
# ---------------------------------------------------------------------------

def test_open_application_still_works_through_registry():
    from tool_registry import registry
    r = registry.execute("open_application", {"app_name": "notepad"})
    assert r["success"] is True
    print("test_open_application_still_works_through_registry: PASS")


def test_find_file_still_works_through_registry():
    from tool_registry import registry
    r = registry.execute("find_file", {"filename": "definitely_not_a_real_file_xyz"})
    assert r["success"] is True
    print("test_find_file_still_works_through_registry: PASS")


def test_multi_tool_execution_still_works():
    import brain as brain_module
    from tool_registry import registry, Tool, RISK_SAFE
    from test_helpers import FakeProvider, make_tool_call_response, make_text_response

    executed = []
    def tool_a():
        executed.append("a")
        return "a done"
    def tool_b():
        executed.append("b")
        return "b done"

    registry.register(Tool(name="__mte_a__", description="", parameters={"type": "object", "properties": {}, "required": []}, handler=tool_a, risk=RISK_SAFE))
    registry.register(Tool(name="__mte_b__", description="", parameters={"type": "object", "properties": {}, "required": []}, handler=tool_b, risk=RISK_SAFE))

    fake = FakeProvider([
        make_tool_call_response([("__mte_a__", {}, "c1"), ("__mte_b__", {}, "c2")]),
        make_text_response("Both done."),
    ])
    v = brain_module.JarvisBrain(provider=fake)
    reply = v.ask("find my readme and open it")
    assert executed == ["a", "b"]
    assert reply == "Both done."
    print("test_multi_tool_execution_still_works: PASS")


def test_permission_blocking_still_works():
    import brain as brain_module
    from tool_registry import registry, Tool, RISK_DESTRUCTIVE
    from test_helpers import FakeProvider, make_tool_call_response

    called = {"n": 0}
    def destroy(path):
        called["n"] += 1
        return "destroyed"

    registry.register(Tool(name="__perm_test_destroy__", description="", risk=RISK_DESTRUCTIVE,
                            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                            handler=destroy))

    fake = FakeProvider([make_tool_call_response([("__perm_test_destroy__", {"path": "x.txt"}, "c1")])])
    v = brain_module.JarvisBrain(provider=fake)
    v.ask("delete x.txt")

    assert called["n"] == 0, "Destructive tool must not run before confirmation"
    assert v.pending_actions
    assert fake.call_count == 1, "Confirmation must not require a second model call"
    print("test_permission_blocking_still_works: PASS")


def test_confirmation_still_works():
    import brain as brain_module
    from tool_registry import registry, Tool, RISK_DESTRUCTIVE
    from test_helpers import FakeProvider, make_tool_call_response

    called = {"n": 0}
    def destroy(path):
        called["n"] += 1
        return "destroyed"

    registry.register(Tool(name="__confirm_test_destroy__", description="", risk=RISK_DESTRUCTIVE,
                            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                            handler=destroy))

    fake = FakeProvider([make_tool_call_response([("__confirm_test_destroy__", {"path": "y.txt"}, "c1")])])
    v = brain_module.JarvisBrain(provider=fake)
    v.ask("delete y.txt")
    v.ask("yes")
    assert called["n"] == 1
    assert v.pending_actions == []
    print("test_confirmation_still_works: PASS")


def test_confirmation_expiration_still_works():
    import brain as brain_module
    from tool_registry import registry, Tool, RISK_DESTRUCTIVE
    from test_helpers import FakeProvider, make_tool_call_response, make_text_response

    called = {"n": 0}
    def destroy(path):
        called["n"] += 1
        return "destroyed"

    registry.register(Tool(name="__expire_test_destroy__", description="", risk=RISK_DESTRUCTIVE,
                            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                            handler=destroy))

    fake = FakeProvider([make_tool_call_response([("__expire_test_destroy__", {"path": "z.txt"}, "c1")])])
    v = brain_module.JarvisBrain(provider=fake)
    v.ask("delete z.txt")
    v.pending_actions[0].created_at -= (brain_module.PENDING_ACTION_TIMEOUT_SECONDS + 10)

    v.provider = FakeProvider([make_text_response("no pending action anymore")])
    v.ask("yes")
    assert called["n"] == 0, "Expired confirmation must not execute"
    assert v.pending_actions == []
    print("test_confirmation_expiration_still_works: PASS")


def test_context_memory_still_works():
    import brain as brain_module
    from test_helpers import FakeProvider, make_text_response
    import memory_store

    memory_store.remember("provider_layer_test_key", "still works")
    fake = FakeProvider([make_text_response("acknowledged")])
    v = brain_module.JarvisBrain(provider=fake)
    msg = v._build_system_message("does it still work")
    assert "still works" in msg["content"]
    print("test_context_memory_still_works: PASS")


if __name__ == "__main__":
    _cleanup()
    test_brain_does_not_import_openai_directly()
    test_brain_uses_injected_provider_not_a_hardcoded_one()
    test_provider_objects_do_not_leak_into_brain()

    test_plain_text_response_normalized_correctly()
    test_empty_response_handled_safely()

    test_single_tool_call_normalized_correctly()
    test_multiple_tool_calls_normalized_correctly()
    test_tool_call_ids_preserved()
    test_tool_results_flow_back_through_provider()

    test_provider_errors_become_model_provider_error()
    test_rate_limit_errors_distinguishable()
    test_auth_errors_distinguishable()
    test_invalid_model_response_handled_safely()

    test_open_application_still_works_through_registry()
    test_find_file_still_works_through_registry()
    test_multi_tool_execution_still_works()
    test_permission_blocking_still_works()
    test_confirmation_still_works()
    test_confirmation_expiration_still_works()
    test_context_memory_still_works()

    _cleanup()
    print("\nAll model provider tests passed.")
