"""
The "brain" of JARVIS: sends messages to Groq's free API (OpenAI-compatible),
handles tool calls, and returns a final spoken-ready text response.

Groq is used here instead of the Claude API directly because it has a genuinely
free tier (no credit card needed). Swap back to Anthropic later any time you want -
the tool-calling logic is nearly identical, just different SDK/response shapes.
"""

import os
import json
import time
import re
from openai import OpenAI
import tools  # noqa: F401 - importing triggers each tool's @register_tool decorator
from tool_registry import registry
import memory_store

MODEL = "openai/gpt-oss-120b"  # Groq's free-tier flagship model (llama-3.3-70b was deprecated)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "conversation_history.json")
HISTORY_BACKUP_FILE = os.path.join(os.path.dirname(__file__), "conversation_history.json.bak")
MAX_SAVED_TURNS = 40  # keep the most recent N user+assistant turns across sessions

PENDING_ACTION_TIMEOUT_SECONDS = 120  # a confirmation prompt goes stale after this long

# Context/memory management config - deliberately simple (no DB, no
# embeddings). Configurable via env vars, reasonable defaults otherwise.
CONTEXT_TOKEN_BUDGET = int(os.getenv("V_CONTEXT_TOKEN_BUDGET", "6000"))
RECENT_MESSAGES_KEEP = int(os.getenv("V_RECENT_MESSAGES_KEEP", "12"))  # never trimmed/summarized
MEMORY_RETRIEVAL_LIMIT = int(os.getenv("V_MEMORY_RETRIEVAL_LIMIT", "5"))

# Response generation controls - the free-tier Groq model has been observed
# to occasionally ramble into long, repeated text on open-ended replies with
# no bound on length. max_tokens caps how long any single reply can get;
# frequency_penalty directly discourages the model from repeating the same
# phrase over and over within one response.
RESPONSE_MAX_TOKENS = int(os.getenv("V_RESPONSE_MAX_TOKENS", "300"))
RESPONSE_FREQUENCY_PENALTY = float(os.getenv("V_RESPONSE_FREQUENCY_PENALTY", "0.6"))

# Words/phrases that count as confirming or rejecting a pending action. Kept
# as simple substring checks rather than a classifier - good enough for a
# clear yes/no, and anything else is treated as "not a confirmation" (see
# _check_pending_action below), which correctly falls through to normal
# processing rather than misfiring on an ambiguous reply.
_AFFIRMATIONS = {"yes", "yeah", "yep", "yup", "go ahead", "do it", "confirm", "confirmed", "proceed", "sure"}
_REJECTIONS = {"no", "nope", "cancel", "don't", "dont", "stop", "forget it", "never mind", "nevermind"}

SYSTEM_PROMPT = """You are V - a sharp, capable personal AI operating alongside your user as their strategist,
operator, and second set of eyes. Your job isn't just to follow commands - it's to understand
what the user is actually trying to accomplish and help them get there intelligently.

Understand intent. Determine the best way to achieve it. Act when the path is clear. Ask when
clarification genuinely matters. Warn when something's risky. Always think a step beyond the
immediate task.

PERSONALITY
Sharp, calm under pressure, confident without being arrogant, dry and occasionally witty,
practical. You work WITH the user, not for them - if they propose a worse approach than
necessary, say so and explain why. Never behave like a submissive servant.

Cyberpunk lingo woven in naturally, not forced: tasks are "jobs" or "runs," problems are
"static" or "glitches," good outcomes are "clean" or "locked in," the user might be "choom" or
"runner." Use these occasionally so they land - not every line, and never at the cost of clarity.

CONVERSATIONAL STYLE
Keep responses concise and easy to listen to - they'll often be converted to speech. Avoid
markdown, headers, bullet-heavy answers, and asterisks in normal conversation. Be direct. Skip
filler like "Certainly!", "I'd be happy to!", "Absolutely!" - use them only if they genuinely fit.
Don't narrate what you're about to do; just do it and report the result.

QUESTION-ASKING
Asking a good question isn't a failure to act - it prevents a bad one. Ask before acting when
the request is genuinely ambiguous, multiple interpretations lead to very different outcomes, a
needed piece of info is missing, or the action has real consequences. Don't ask when the
intended action is obvious or you can reasonably assume and refine afterward. When you do ask,
ask one focused question, not a list - or present two or three concrete options and let the
user pick.

DECISION-MAKING
When asked for advice, don't just list options - analyze and recommend, with the key tradeoff
stated briefly (e.g. "fastest path" vs "most scalable"). Confidence should come from actual
competence - if you're not sure, say so plainly rather than bluffing.

PROACTIVE THINKING
After understanding a task, consider what the user's actually trying to accomplish, what could
go wrong, and what they'll probably need next. Surface a useful next step when it's genuinely
useful, not as constant narration. Initiative should feel helpful, not pushy - don't pile on
extra suggestions after every single reply.

ERROR HANDLING
Don't dump raw stack traces or technical errors on the user. Translate them into what actually
happened and what you're doing about it - "the app isn't installed" rather than a raw exception.

SAFETY
Before anything irreversible or materially risky, make sure intent is clearly there before
acting. Reversible, clearly-authorized actions - just go.

PERMISSION AND CONFIRMATION
Some tools are gated and simply won't execute until the user confirms - this is enforced in
code, not something you decide per-request. When a tool needs confirmation, V asks a single,
fixed confirmation question built directly from what the tool would do - not something you
need to phrase yourself in that moment. Never say or imply an action was completed if it was
actually blocked pending confirmation. If the user declines or cancels, drop it entirely -
don't bring it back up or retry. Ambiguity and permission are different things: if a request is
unclear (which file, which project, what the message should say), resolve that first by asking
- don't respond to unclear requests with a confirmation prompt for a guess. When a request or a
follow-up mentions multiple targets (e.g. "delete 4.txt and 5.txt", or confirming "yes" to your
own question about several files), call the relevant tool once per target, not just the first.

TOOLS AND MEMORY
When the user asks for something a tool can do - opening or closing apps, browsing, weather,
web search, finding/moving/deleting/opening files - use the tool, don't just describe how
they'd do it themselves. Never claim to have done something you didn't actually do.

Never go silent after taking an action. Once a tool call completes, always follow up with a
short spoken confirmation of what happened - e.g. "WhatsApp's open on your desktop, sir.
Anything else?" - never end a turn with no text at all after doing something.

When the user says "open [app name]" - try the actual desktop app first via open_application.
Only fall back to a website when there genuinely is no desktop app (or open_application itself
falls back to a web version for you). When the user says "open [a file]" - a document, image,
PDF, or anything living on disk - always use open_file, never open_website. A local file should
never open as a browser tab.

For personal account pages like "open my github" or "open my [service]" - recall the relevant
username first. If found, construct the URL yourself (e.g. github.com/username) and use
open_website directly - don't ask for something you already have. If recall comes back empty,
ask for it once, remember it, then proceed the same turn.

close_application matches against whatever's actually running, so people's imprecise naming
("snipping tool" for a process called SnippingTool.exe) is expected and usually resolves fine
on its own. If it comes back with multiple plausible matches, just relay them and ask which one
- don't guess.

You have long-term memory via remember, recall, and list_memories. Save genuinely useful info
(usernames, preferences, recurring project details) without being asked. Check recall before
asking the user for something they may have already told you - if recall comes up empty for a
key you're not 100% sure of, try list_memories once before concluding you don't have it, since
the info might be saved under a slightly different key name. Don't announce that you're
"recalling from memory" - just use the information naturally.

Critical: whenever you have to ASK the user for a piece of identifying info you don't have yet
(a username, an account name, a preference) - the moment they answer, immediately call
remember to save it, in the same turn if possible. Never ask for the same piece of information
twice. If recall comes back empty for something like a username, ask once, save the answer, and
from then on use it automatically without asking again.

Deleting a file moves it to a local quarantine folder, not a permanent erase - so on a clear
deletion request, act without excessive double-checking. If the request or target file is
ambiguous, or could reasonably be important, clarify first.

CONTEXT AND MEMORY
Recent conversation is context, not proof of anything. Your own past replies are not evidence
that an action actually happened - only an actual tool result establishes that. If you're
unsure whether something you said earlier actually went through, check rather than assume it
did. Long-term memory holds persistent facts the user has told you; it's separate from the
flow of conversation and won't include everything discussed recently.
"""


class JarvisBrain:
    def __init__(self, api_key: str = None):
        # Groq's API is OpenAI-compatible, so we just point the OpenAI client
        # at Groq's base URL instead of OpenAI's.
        self.client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        # Full provenance-tagged turn log (persisted to disk, capped at save
        # time) - separate from self.history, which is the raw OpenAI-format
        # list actually sent to the API and subject to live context trimming.
        self.session_summary, self.provenance = self._load_session_history()

        # System message is NOT stored statically - it's rebuilt fresh on
        # every call via _build_system_message(), so memory updates are
        # reflected immediately without relying on the model remembering to
        # call recall.
        self.history = [self._build_system_message()]
        self.history.extend({"role": t["role"], "content": t["content"]} for t in self.provenance)

        # A list of blocked actions awaiting one combined confirmation. Empty
        # list = nothing pending. A single request can produce more than one
        # (e.g. "delete 4.txt and 5.txt"), all confirmed or rejected together.
        # Deliberately in-memory only, never persisted - a confirmation
        # should expire when V restarts, not survive across sessions.
        self.pending_actions = []

    def _build_system_message(self, user_text: str = None) -> dict:
        """Build the system message with SELECTIVELY relevant memory facts
        injected, plus the session summary if one exists. Only a small,
        relevant subset of memory is injected (not the entire store) - a
        "recall"/"list_memories" tool call is still available as a fallback
        for anything the relevance filter misses."""
        blocks = []

        try:
            relevant = memory_store.get_relevant(user_text or "", limit=MEMORY_RETRIEVAL_LIMIT)
            if relevant:
                facts = "\n".join(f"- {m['key']}: {m['value']}" for m in relevant)
                blocks.append(
                    "KNOWN FACTS RELEVANT TO THIS REQUEST (already saved in memory - use "
                    "these directly, don't ask for them again):\n" + facts
                )
        except Exception:
            pass

        if self.session_summary:
            blocks.append("SESSION SUMMARY (condensed earlier context):\n" + self.session_summary)

        content = SYSTEM_PROMPT
        if blocks:
            content += "\n\n" + "\n\n".join(blocks)
        return {"role": "system", "content": content}

    @staticmethod
    def _load_session_history() -> tuple:
        """Load conversation_history.json, transparently migrating the old
        bare-list format if found. Returns (summary_string, list_of_turn_dicts).
        Never deletes the original file without first writing a backup."""
        if not os.path.exists(HISTORY_FILE):
            return "", []

        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            return "", []

        if isinstance(data, dict) and data.get("version") == 2:
            return data.get("summary", ""), data.get("turns", [])

        if isinstance(data, list):
            # Old format: bare list of {"role": ..., "content": ...}. Back up
            # before migrating - nothing existing is silently discarded.
            try:
                if not os.path.exists(HISTORY_BACKUP_FILE):
                    with open(HISTORY_BACKUP_FILE, "w") as f:
                        json.dump(data, f, indent=2)
            except Exception:
                pass

            now = time.time()
            migrated = []
            for turn in data:
                role = turn.get("role")
                content = turn.get("content")
                if not isinstance(content, str) or not content:
                    continue
                if role == "user":
                    migrated.append({"role": "user", "content": content, "source": "user", "verified": True, "ts": now})
                elif role == "assistant":
                    # Legacy replies have no recorded provenance - treat as
                    # ungrounded conversational text, not proof anything
                    # happened, per the core trust rule.
                    migrated.append({"role": "assistant", "content": content, "source": "model", "grounded": False, "tool_outcome": "none", "ts": now})
            return "", migrated

        return "", []

    def _save_turns(self):
        """Persist provenance-tagged turns plus the session summary. Tool
        call/result messages are never persisted - only plain text turns -
        so a future session never replays stale tool_call_ids."""
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump({
                    "version": 2,
                    "summary": self.session_summary,
                    "turns": self.provenance[-MAX_SAVED_TURNS:],
                }, f, indent=2)
        except Exception as e:
            print(f"[Couldn't save conversation history: {e}]")

    def _record_turn(self, role: str, content: str, **provenance_fields):
        """Append a plain text turn to both the live working-memory history
        (self.history, what's actually sent to the API) and the persisted
        provenance log (self.provenance, what's saved to disk) in lockstep."""
        self.history.append({"role": role, "content": content})
        self.provenance.append({"role": role, "content": content, "ts": time.time(), **provenance_fields})

    # Friendly verb phrasing for known consequential tools - purely cosmetic
    # text generation for the confirmation prompt, not permission logic (the
    # actual gate lives entirely in tool_registry.py and doesn't consult this).
    _CONFIRMATION_VERBS = {
        "delete_file": "delete",
    }

    @staticmethod
    def _describe_target(pending_action) -> str:
        """Return just the filename/target for one pending action, for use
        in both single and combined confirmation phrasing."""
        for key in ("path", "source", "file", "filename"):
            if key in pending_action.arguments:
                raw = str(pending_action.arguments[key])
                return re.split(r"[\\/]+", raw)[-1]
        return pending_action.description

    @staticmethod
    def _build_confirmation_prompt(pending_actions: list) -> str:
        """Build ONE clean, natural confirmation sentence covering every
        blocked action from this request - deterministic, no model call
        involved, so there is no possibility of a rambling or repeated
        confirmation message, and no risk of only some files getting asked
        about while others silently get skipped."""
        if len(pending_actions) == 1:
            action = pending_actions[0]
            verb = JarvisBrain._CONFIRMATION_VERBS.get(action.tool_name, "run")
            target = JarvisBrain._describe_target(action)
            return f"That'll {verb} {target}. Should I go ahead?"

        # Multiple actions - group by verb so "delete 4.txt and 5.txt" reads
        # naturally even if (in the future) a batch mixes different kinds of
        # gated actions.
        by_verb = {}
        for action in pending_actions:
            verb = JarvisBrain._CONFIRMATION_VERBS.get(action.tool_name, "run")
            by_verb.setdefault(verb, []).append(JarvisBrain._describe_target(action))

        clauses = []
        for verb, targets in by_verb.items():
            if len(targets) == 1:
                clauses.append(f"{verb} {targets[0]}")
            elif len(targets) == 2:
                clauses.append(f"{verb} {targets[0]} and {targets[1]}")
            else:
                clauses.append(f"{verb} {', '.join(targets[:-1])}, and {targets[-1]}")

        return f"That'll {'; also '.join(clauses)}. Should I go ahead with all of it?"

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        """Rough token estimate (chars/4) - good enough for a budget trigger,
        not meant to be exact. Avoids adding a tokenizer dependency."""
        return sum(len(json.dumps(m, default=str)) for m in messages) // 4

    def _summarize_turns(self, turns: list) -> str:
        """Condense a block of old plain-text turns into a short summary via
        ONE model call. Only called when the context budget is actually
        exceeded - not on every turn. The prompt explicitly enforces the
        core trust rule: only phrase something as done if it was actually
        tool-confirmed, never because V previously claimed it in prose."""
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        instruction = (
            "Condense the following conversation excerpt into a short factual summary "
            "(3-6 sentences). Capture: the current task, decisions made, important "
            "user-provided information, and any unresolved questions. "
            "CRITICAL: do not state that an action (deleting a file, sending something, "
            "opening something, etc.) succeeded unless the excerpt shows clear evidence "
            "of that outcome. If an action was only requested or discussed but its result "
            "is unclear, phrase it as 'was requested' or 'was attempted', not as done.\n\n"
            f"Conversation excerpt:\n{transcript}"
        )
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": instruction}],
                max_tokens=RESPONSE_MAX_TOKENS,
                frequency_penalty=RESPONSE_FREQUENCY_PENALTY,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            # If summarization itself fails, better to lose old detail than
            # crash the interaction - the recent tail is still intact.
            return ""

    def _apply_context_budget(self):
        """Keep the live working-memory context (self.history, what's
        actually sent to the API) within a token budget. Runs once at the
        start of each ask() call - never mid tool-call-loop, so an active
        multi-tool exchange is never disturbed.

        Order of operations (cheapest/safest first):
          1. If under budget, do nothing.
          2. Strip old tool-call noise (assistant tool_call messages + their
             tool-role results) that's older than the protected recent tail -
             this is pure noise once a turn is finished, never needed again.
          3. If still over budget, summarize the oldest remaining plain
             turns via one model call, fold that into the running session
             summary, and drop those raw messages from working memory (they
             remain fully available in self.provenance/conversation_history.json
             regardless - trimming only affects what's sent to the API).
        """
        body = self.history[1:]  # everything except the system message
        if self._estimate_tokens(body) <= CONTEXT_TOKEN_BUDGET:
            return

        if len(body) > RECENT_MESSAGES_KEEP:
            head, tail = body[:-RECENT_MESSAGES_KEEP], body[-RECENT_MESSAGES_KEEP:]
        else:
            head, tail = [], body

        # Step 2: strip tool-call noise from the older portion. Both sides of
        # any tool_call_id relationship (the assistant message that made the
        # call, and the tool-role messages answering it) are always removed
        # together here, since this only ever runs on already-completed turns.
        head = [
            m for m in head
            if m.get("role") != "tool" and not (m.get("role") == "assistant" and m.get("tool_calls"))
        ]

        if self._estimate_tokens(head + tail) <= CONTEXT_TOKEN_BUDGET:
            self.history = [self.history[0]] + head + tail
            return

        # Step 3: still over budget - summarize what's left of the old
        # portion (by now just plain user/assistant text turns) and drop it
        # from live working memory. Nothing is lost from disk - the full
        # turn log is still persisted separately in self.provenance.
        plain_head = [m for m in head if isinstance(m.get("content"), str) and m.get("content")]
        if plain_head:
            addition = self._summarize_turns(plain_head)
            if addition:
                self.session_summary = (self.session_summary + "\n" + addition).strip() if self.session_summary else addition

        self.history = [self.history[0]] + tail

    def _check_pending_action(self, user_text: str) -> str:
        """If there's a pending confirmation (possibly covering several
        actions), decide whether this message confirms it, rejects it, or is
        unrelated (which supersedes/cancels the pending actions rather than
        accidentally executing them later).

        Returns a status string to guide ask(): "confirmed", "rejected",
        "expired", or "none" (nothing pending, or it wasn't addressed -
        proceed normally)."""
        if not self.pending_actions:
            return "none"

        # All actions in a batch are created at the same moment, so checking
        # the first is sufficient for the whole batch's age.
        if time.time() - self.pending_actions[0].created_at > PENDING_ACTION_TIMEOUT_SECONDS:
            self.pending_actions = []
            return "expired"

        text_norm = user_text.strip().lower().rstrip(".!")
        if text_norm in _AFFIRMATIONS:
            return "confirmed"
        if text_norm in _REJECTIONS:
            self.pending_actions = []
            return "rejected"

        # Anything else supersedes the pending actions rather than risking a
        # stale/misread "yes" executing something unrelated later - they're
        # simply dropped and the new message is handled as a normal request.
        self.pending_actions = []
        return "none"

    def ask(self, user_text: str) -> str:
        """Send user text to Groq, handle any tool calls, return final reply text."""
        # Refresh the system message with memory relevant to THIS message
        # (not the entire store) before every call, and fold in the session
        # summary if one exists.
        self.history[0] = self._build_system_message(user_text)

        # Keep live working memory within budget before adding this turn -
        # never runs mid tool-call-loop, only at the start of a fresh request.
        self._apply_context_budget()

        pending_status = self._check_pending_action(user_text)

        if pending_status == "confirmed":
            actions = self.pending_actions
            self.pending_actions = []
            results = []
            any_success = False
            for action in actions:
                outcome = registry.execute_confirmed(action)
                if outcome["success"]:
                    results.append(outcome["result"])
                    any_success = True
                else:
                    results.append(f"Ran into a problem with {self._describe_target(action)}: {outcome['error']}")
            # Deterministic reply, built directly from each tool's own result -
            # never routed through the model. The earlier bug was caused by
            # feeding an internal "[System note...]" annotation into a
            # user-role message and letting the model react to it; the model
            # sometimes just echoed the note back verbatim instead of
            # producing a natural reply. Anywhere correctness matters here,
            # V's response is built in code, not generated.
            final_text = " ".join(results)
            self._record_turn("user", user_text, source="user", verified=True)
            self._record_turn(
                "assistant", final_text, source="model",
                grounded=True, tool_outcome="success" if any_success else "error",
            )
            self._save_turns()
            return final_text

        elif pending_status == "rejected":
            final_text = "Cancelled - didn't touch it. Anything else?"
            self._record_turn("user", user_text, source="user", verified=True)
            self._record_turn("assistant", final_text, source="model", grounded=True, tool_outcome="none")
            self._save_turns()
            return final_text

        elif pending_status == "expired":
            # Per spec: an expired confirmation should be treated as a new
            # message, not answered with a canned reply - the user's actual
            # words might not even be "yes" (e.g. they moved on and asked
            # something else while the window lapsed). Just drop the stale
            # pending action and fall through to normal processing.
            self._record_turn("user", user_text, source="user", verified=True)

        else:
            # No pending action, or an unrelated message superseded one -
            # ordinary request, handled normally below.
            self._record_turn("user", user_text, source="user", verified=True)

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.history,
            tools=registry.get_llm_tools(),
            max_tokens=RESPONSE_MAX_TOKENS,
            frequency_penalty=RESPONSE_FREQUENCY_PENALTY,
        )
        message = response.choices[0].message

        last_tool_result = None
        awaiting_confirmation = False
        confirmation_message = None
        batch_pending_actions = []
        tool_calls_happened = False
        last_tool_outcome_type = "none"  # none | success | error | timeout

        # Keep handling tool calls until the model gives a final text answer
        while message.tool_calls:
            self.history.append(message.model_dump(exclude_none=True))

            blocked_this_batch = False
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool = registry.get(tool_call.function.name)

                if blocked_this_batch and not (tool and tool.requires_confirmation):
                    # A prior tool call in this same batch already hit a
                    # confirmation gate. Anything that would actually EXECUTE
                    # is skipped rather than run, since it might depend on
                    # the blocked action's outcome (e.g. "delete it, then
                    # open VS Code" must not open VS Code before the delete
                    # is approved). Other gated calls are still allowed
                    # through below so a request like "delete 4.txt and
                    # 5.txt" gets ONE combined confirmation, not just the
                    # first file.
                    result_text = (
                        "Skipped - waiting on confirmation for a previous "
                        "action in this request."
                    )
                else:
                    outcome = registry.execute(tool_call.function.name, args)
                    tool_calls_happened = True
                    if outcome["success"]:
                        result_text = outcome["result"]
                        last_tool_outcome_type = "success"
                    elif outcome.get("error_type") == "confirmation_required":
                        batch_pending_actions.append(outcome["pending_action"])
                        result_text = outcome["message"]
                        blocked_this_batch = True
                        awaiting_confirmation = True
                    elif outcome.get("error_type") == "timeout":
                        result_text = outcome["message"]
                        last_tool_outcome_type = "timeout"
                    else:
                        result_text = f"Error: {outcome['error']}"
                        last_tool_outcome_type = "error"

                last_tool_result = result_text
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            if awaiting_confirmation:
                # CRITICAL: do not call the model again this turn. Asking the
                # model to "generate the confirmation text" was the actual
                # bug - a free-form generation with no length/repetition
                # bound could ramble into a long, repeated block of text (or
                # even hallucinate that the action already succeeded). The
                # permission gate has already made its decision; there's
                # nothing left for the model to reason about. Build the
                # confirmation deterministically instead, covering every
                # blocked action from this request in one combined message,
                # then end the turn immediately without any further API call.
                self.pending_actions = batch_pending_actions
                confirmation_message = self._build_confirmation_prompt(batch_pending_actions)
                message = None
                break

            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.history,
                tools=registry.get_llm_tools(),
                max_tokens=RESPONSE_MAX_TOKENS,
                frequency_penalty=RESPONSE_FREQUENCY_PENALTY,
            )
            message = response.choices[0].message

        if confirmation_message is not None:
            final_text = confirmation_message
            grounded, tool_outcome_prov = True, "confirmation_required"
        else:
            final_text = (message.content or "").strip()
            if not final_text and last_tool_result:
                # The model completed a tool action but returned no text to speak -
                # never let V go silent after actually doing something. Fall back
                # to the tool's own result message rather than saying nothing.
                final_text = last_tool_result
            grounded = tool_calls_happened
            tool_outcome_prov = last_tool_outcome_type

        self.history.append({"role": "assistant", "content": final_text})
        self.provenance.append({
            "role": "assistant", "content": final_text, "source": "model",
            "grounded": grounded, "tool_outcome": tool_outcome_prov, "ts": time.time(),
        })
        self._save_turns()
        return final_text
