"""
Tool Registry for V.

Each tool registers itself once via @register_tool(...), and the registry is
the single source of truth for what tools exist, the schemas the LLM sees
(get_llm_tools()), and how a tool call by name gets executed (execute()).

EXECUTION ISOLATION
Tool handlers run inside a persistent ThreadPoolExecutor rather than being
called directly on the caller's thread. This means a hung tool (a stuck
subprocess, a network call with no timeout, anything that blocks forever)
cannot freeze the rest of V - registry.execute() waits at most `timeout`
seconds for a result, then gives up and returns a structured timeout failure
so the interaction loop stays responsive.

IMPORTANT LIMITATION: Python threads cannot be forcibly killed. Timing out
means V stops WAITING for the tool - it does not mean the tool's thread has
actually been stopped. A genuinely hung tool keeps running in the background
until it finishes on its own or the process exits; it just no longer blocks
anything else. The executor has a bounded worker count so a pile-up of
abandoned threads has a ceiling, but this is not true termination. Getting
actual termination would require running tools in separate processes
(multiprocessing), which is a bigger change not taken here since nothing in
the current tool set needs it yet.

PERMISSION / CONSEQUENCE SYSTEM
Every tool declares a `risk` level and whether it needs confirmation before
running. registry.execute() checks this BEFORE submitting the handler to the
executor - a tool marked as requiring confirmation never actually runs on
this call; instead a "confirmation_required" result is returned describing
what WOULD happen, so the caller (brain.py) can ask the user and only
execute it later via execute_confirmed() once explicitly authorized. The
model cannot bypass this by re-requesting the tool - the gate is enforced
here, not in the system prompt.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_TIMEOUT_SECONDS = 10  # sensible default for simple local operations
MAX_WORKERS = 8  # bounds how many abandoned/hung tool threads can pile up

# --- Consequence levels ------------------------------------------------
# Kept intentionally small - four levels, not a sprawling policy engine.
RISK_SAFE = "safe"                # informational / effectively harmless
RISK_LOW = "low"                  # affects the local machine, generally reversible
RISK_CONSEQUENTIAL = "consequential"  # affects external systems / other people
RISK_DESTRUCTIVE = "destructive"  # hard or impossible to reverse

# Only these two levels require confirmation by default. Tools can still
# override `requires_confirmation` explicitly regardless of risk level, so a
# future policy layer isn't boxed in by risk level alone.
_DEFAULT_CONFIRM_RISK_LEVELS = {RISK_CONSEQUENTIAL, RISK_DESTRUCTIVE}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the tool's parameters
    handler: Callable[..., str]
    category: str = "general"  # optional, informational only for now
    timeout: float = DEFAULT_TIMEOUT_SECONDS  # per-tool override, in seconds

    # --- Consequence metadata ---
    risk: str = RISK_SAFE
    reversible: bool = True
    requires_confirmation: bool = field(default=None)  # None = derive from risk level

    def __post_init__(self):
        if self.requires_confirmation is None:
            self.requires_confirmation = self.risk in _DEFAULT_CONFIRM_RISK_LEVELS


@dataclass
class PendingAction:
    """A tool call that was blocked pending user confirmation. Holds
    everything needed to execute it later without the user repeating
    themselves, plus a timestamp so it can expire."""
    tool_name: str
    arguments: dict
    description: str  # human-readable summary of what WILL happen, for the confirmation prompt
    created_at: float


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        # One shared, persistent executor for the registry's lifetime - not
        # recreated per call, so we're not constantly spinning threads up.
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="v-tool")

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        return name in self._tools

    def all_tools(self) -> list:
        return list(self._tools.values())

    def get_llm_tools(self) -> list:
        """Return tool schemas in the OpenAI-compatible function-calling format
        the current model client expects (used by Groq's API)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def _describe_call(self, tool: Tool, arguments: dict) -> str:
        """Build a specific, human-readable description of what a tool call
        will do, for use in a confirmation prompt. Deliberately generic (not
        hardcoded per tool name) - falls back to naming the tool and its
        arguments plainly, which is enough for the model to build a clear
        confirmation sentence around."""
        arg_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"{tool.name}({arg_str})"

    def _run(self, tool: Tool, arguments: dict) -> dict:
        """Actually submit a tool handler to the executor and wait for it,
        with timeout isolation. This is the shared execution path used both
        by normal execute() and by execute_confirmed() once an action has
        been authorized - no permission logic lives here."""
        try:
            future = self._executor.submit(tool.handler, **arguments)
        except TypeError as e:
            return {
                "success": False,
                "error_type": "bad_arguments",
                "error": f"Bad arguments for '{tool.name}': {e}",
            }

        try:
            result = future.result(timeout=tool.timeout)
            return {"success": True, "result": result}
        except FutureTimeoutError:
            # We stop WAITING here - the thread itself may still be running
            # in the background (see module docstring). We deliberately do
            # not attempt to cancel/kill it, since Python threads can't be
            # forced to stop safely.
            return {
                "success": False,
                "error_type": "timeout",
                "error": "timeout",
                "tool": tool.name,
                "message": f"Tool execution timed out after {tool.timeout} seconds.",
            }
        except TypeError as e:
            return {
                "success": False,
                "error_type": "bad_arguments",
                "error": f"Bad arguments for '{tool.name}': {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error_type": "exception",
                "error": f"'{tool.name}' failed: {e}",
            }

    def execute(self, name: str, arguments: dict) -> dict:
        """Look up and run a tool by name, with execution isolation, a
        timeout, AND a permission/consequence gate. Never raises and never
        blocks longer than the tool's configured timeout - always returns a
        structured result dict.

        If the tool requires confirmation, the handler is NOT run. Instead a
        "confirmation_required" result is returned describing the pending
        action, for the caller to surface to the user and later resume via
        execute_confirmed(pending_action).

        Returns one of:
            {"success": True, "result": str}
            {"success": False, "error_type": "unknown_tool", "error": str}
            {"success": False, "error_type": "bad_arguments", "error": str}
            {"success": False, "error_type": "exception", "error": str}
            {"success": False, "error_type": "timeout", "error": str,
             "tool": str, "message": str}
            {"success": False, "error_type": "confirmation_required",
             "tool": str, "message": str, "pending_action": PendingAction}
        """
        tool = self.get(name)
        if tool is None:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "error": f"Unknown tool: {name}",
            }

        if tool.requires_confirmation:
            description = self._describe_call(tool, arguments)
            return {
                "success": False,
                "error_type": "confirmation_required",
                "tool": name,
                "message": f"'{name}' requires confirmation before running: {description}",
                "pending_action": PendingAction(
                    tool_name=name,
                    arguments=arguments,
                    description=description,
                    created_at=__import__("time").time(),
                ),
            }

        return self._run(tool, arguments)

    def execute_confirmed(self, pending_action: PendingAction) -> dict:
        """Execute a previously-blocked action now that the user has
        explicitly authorized it. Skips the permission gate entirely - by
        the time this is called, confirmation has already happened; this is
        the one deliberate, narrow bypass, and it only runs the EXACT stored
        tool_name/arguments, never anything freshly requested by the model."""
        tool = self.get(pending_action.tool_name)
        if tool is None:
            return {
                "success": False,
                "error_type": "unknown_tool",
                "error": f"Unknown tool: {pending_action.tool_name}",
            }
        return self._run(tool, pending_action.arguments)


# Single shared registry instance for the whole app
registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: dict,
    category: str = "general",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    risk: str = RISK_SAFE,
    reversible: bool = True,
    requires_confirmation: bool = None,
):
    """Decorator that registers a function as a tool.

    Usage:
        @register_tool(
            name="get_weather",
            description="...",
            parameters={"type": "object", "properties": {...}, "required": [...]},
            timeout=15,  # optional - defaults to DEFAULT_TIMEOUT_SECONDS
            risk="safe",  # optional - defaults to "safe". One of: safe, low,
                          # consequential, destructive
            reversible=True,  # optional - informational, defaults to True
            requires_confirmation=None,  # optional - None means "derive from
                                         # risk level" (consequential/destructive
                                         # require confirmation by default)
        )
        def get_weather(city: str = None) -> str:
            ...
    """
    def decorator(func: Callable[..., str]):
        registry.register(Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            category=category,
            timeout=timeout,
            risk=risk,
            reversible=reversible,
            requires_confirmation=requires_confirmation,
        ))
        return func
    return decorator

