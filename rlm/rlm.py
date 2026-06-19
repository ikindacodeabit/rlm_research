"""Recursive Language Model (RLM) with a configurable MEMORY BUDGET.

This is a drop-in successor to the minimal scaffold. The paradigm is unchanged
(Zhang, Kraska & Khattab, 2025): the long prompt is NOT placed in the model's
context window; it lives as a string `context` in a persistent REPL, and the
root LM writes code to inspect it / recurse via `llm_query`.

What v6 adds
------------
1. MemoryBudget: a hard cap on the ROOT model's context window (in tokens),
   independent of document length. The full document never enters context;
   what grows in the minimal scaffold is the transcript of REPL observations.
   v6 keeps the full transcript server-side but only ever SENDS the root a
   bounded view:

       system + begin + [persistent NOTES] + last K turns that fit the budget

   Older turns are "folded" out of context by one of two strategies:
     * "evict"     - dropped cheaply (model must have saved what it needs via note())
     * "summarize" - a sub-LLM compresses them into the NOTES scratchpad first

2. note(text): a persistent scratchpad tool. Notes survive folding, so the model
   can carry evidence across evictions even under a tiny budget.

3. Efficiency: token-accurate accounting (tiktoken when available), sub-call
   memoization, adaptive observation truncation, and per-run efficiency metrics
   returned on RLMResult.metrics for budget sweeps.

Grounding note: the anti-hallucination guard (FINAL literal must appear in real
output) checks a SEPARATE server-side `seen_output` accumulator, which is never
compacted. So shrinking the budget never weakens grounding.

NOTE: model-generated code is executed with `exec`. On a shared cluster, run this
inside your own user account / a SLURM job only. For stricter isolation, wrap the
REPL in a separate process or container.
"""
from __future__ import annotations

import ast
import contextlib
import io
import re
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

from .client import NIMClient

ROOT_SYSTEM_PROMPT = """You are solving a task over a LONG document that does NOT fit in your context window.
The document is stored in a Python REPL as a string variable named `context`
(length: {ctx_len} characters). You interact with it by writing Python code.

RULES:
- Reply with exactly ONE Python code block, fenced as ```python ... ```, and then
  STOP. Write NOTHING after the code block.
- CRITICAL: You CANNOT see the result of your code until the next message. NEVER
  write, guess, or simulate the REPL output yourself. NEVER state an answer you
  have not literally seen in a real REPL output message.
- The REPL is persistent: variables survive across turns. Like Python's
  interactive shell, the value of a bare final expression is echoed back to you.
- Useful tools available in the REPL:
    * `context` (str): the full document.
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\\n" + context[i:j])
      (keep each call under ~8000 characters). Capture the result:
      ans = llm_query(...) then print(ans).
    * `note(text: str)`: save a SHORT finding to a persistent scratchpad. {note_help}
    * `print(...)`: anything you print is shown back to you in the NEXT message
      (truncated to {obs_limit} chars), so print only what you need to see.
- Strategy: peek at structure first (e.g. `print(context[:2000])`,
  `print(len(context))`, regex search), then narrow down with string ops or
  chunked `llm_query` calls. Do NOT print the whole context.
- If a search or extraction returns ZERO matches or fewer than the task implies,
  that is a signal your pattern is wrong — NOT that the answer is 0 or empty.
    Print a sample of the text around a likely keyword to see the actual format,
      then retry with a corrected pattern.
      - When the task states how many items exist, verify your extraction found that
        many before computing the final answer.
- Once (and only once) you have SEEN the information needed for the answer in a
  real REPL output, finish with a code block calling FINAL(...) on a VARIABLE or
  expression computed by your code. NEVER call FINAL with a literal guessed
  value — FINAL("42") with a made-up constant will be rejected.
- You have at most {max_steps} code turns. Be efficient.{budget_note}

Example of a correct session (3 turns):
  You:  ```python
        idx = context.find("invoice total")
        print(context[max(0, idx-200):idx+500])
        ```
  REPL: ...The invoice total for March was $4,210 including tax...
  You:  ```python
        ans = llm_query("What was the March invoice total? Answer from this text:\\n" + context[idx-200:idx+500])
        note("March invoice total = " + ans)
        print(ans)
        ```
  REPL: $4,210
  You:  ```python
        FINAL(ans)
        ```

The user's task is:
{task}
"""

SUB_SYSTEM_PROMPT = (
    "You are a helpful sub-model. Answer the question using ONLY the text "
    "provided in the prompt. Be concise and factual."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "You compress an AI agent's prior REPL findings into a few terse bullet "
    "points. Keep concrete facts, numbers, quotes, indices, and partial answers "
    "verbatim. Drop chatter. Output <= 6 short lines."
)

NOTES_TEMPLATE = (
    "[MEMORY NOTICE] {n} earlier turn(s) were dropped to stay within your memory "
    "budget. This persistent scratchpad is everything you saved/derived before:\n"
    "{notes}\n[END SCRATCHPAD] Continue from the recent REPL output below."
)

CODE_RE = re.compile(r"```(?:python|repl|py)?[ \t]*\n?(.*?)```", re.DOTALL)
STMT_KEYWORDS = r"print|import|from|for|while|if|elif|try|except|finally|with|return|note|FINAL_VAR|FINAL"
ONELINE_FIX_RE = re.compile(rf"(?<=[\)\w'\"])\s+(?=(?:{STMT_KEYWORDS})\b)")
TEXT_FINAL_RE = re.compile(r"FINAL\(\s*(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|')\s*\)", re.DOTALL)
TEXT_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*[\"'](\w+)[\"']\s*\)")


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
class TokenCounter:
    """Token counter: uses tiktoken cl100k_base if installed, else len//4."""

    def __init__(self, fn: Optional[Callable[[str], int]] = None):
        self._fn = fn
        self._enc = None
        if fn is None:
            try:  # pragma: no cover - depends on environment
                import tiktoken
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    def count(self, text: str) -> int:
        if self._fn is not None:
            return self._fn(text)
        if self._enc is not None:
            try:  # pragma: no cover
                return len(self._enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict]) -> int:
        # +4 tokens/message is the usual chat-format overhead approximation.
        return sum(self.count(m.get("content", "")) + 4 for m in messages)


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #
@dataclass
class MemoryBudget:
    """Caps the ROOT model's context window.

    max_context_tokens : hard cap on the root prompt (system + begin + notes +
                         recent turns). The headline knob to sweep.
    keep_recent_turns  : how many most-recent (assistant,observation) pairs to
                         try to keep verbatim before budget squeezing kicks in.
    strategy           : "evict" (drop folded turns) or "summarize" (compress
                         folded turns into the NOTES scratchpad via a sub-LLM).
    max_notes_tokens   : cap on the scratchpad size injected into context.
    """
    max_context_tokens: int = 4096
    keep_recent_turns: int = 3
    strategy: str = "evict"  # "evict" | "summarize"
    max_notes_tokens: int = 1024

    def __post_init__(self):
        if self.strategy not in ("evict", "summarize"):
            raise ValueError(f"strategy must be 'evict' or 'summarize', got {self.strategy!r}")


@dataclass
class RLMResult:
    answer: str | None
    steps: int
    finished: bool
    transcript: list = field(default_factory=list)
    end_reason: str = ""
    metrics: dict = field(default_factory=dict)


class RLM:
    def __init__(
        self,
        root_client: NIMClient,
        sub_client: NIMClient | None = None,
        max_steps: int = 12,
        obs_limit: int = 6000,
        max_subcall_chars: int = 32000,
        budget: Optional[MemoryBudget] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        cache_subcalls: bool = True,
    ):
        self.root = root_client
        self.sub = sub_client or root_client
        self.max_steps = max_steps
        self.obs_limit = obs_limit
        self.max_subcall_chars = max_subcall_chars
        self.budget = budget
        self.cache_subcalls = cache_subcalls
        self.tok = TokenCounter(token_counter)

    # ---------------- REPL plumbing ----------------
    def _make_env(self, context: str, metrics: dict, notes: list, cache: dict) -> dict:
        final_box: dict = {"value": None, "done": False}

        def llm_query(prompt: str) -> str:
            prompt = str(prompt)[: self.max_subcall_chars]
            if self.cache_subcalls and prompt in cache:
                metrics["sub_cache_hits"] += 1
                return cache[prompt]
            ans = self.sub.chat(
                [
                    {"role": "system", "content": SUB_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            metrics["sub_calls"] += 1
            metrics["sub_call_tokens"] += self.tok.count(prompt) + self.tok.count(ans)
            if self.cache_subcalls:
                cache[prompt] = ans
            return ans

        def note(text: str) -> str:
            text = str(text).strip()
            if text:
                notes.append(text)
                metrics["notes_saved"] += 1
            return f"[saved note #{len(notes)}]"

        def FINAL(answer) -> None:
            final_box["value"] = str(answer)
            final_box["done"] = True

        env = {
            "context": context,
            "llm_query": llm_query,
            "note": note,
            "FINAL": FINAL,
            "re": re,
        }

        def FINAL_VAR(name) -> None:
            final_box["value"] = str(env.get(str(name), f"<missing var {name}>"))
            final_box["done"] = True

        env["FINAL_VAR"] = FINAL_VAR
        env["_final_box"] = final_box
        return env

    def _exec(self, code: str, env: dict, obs_limit: int) -> str:
        code = code.strip()
        note = ""
        try:
            compile(code, "<rlm>", "exec")
        except SyntaxError:
            if "\n" not in code:
                fixed = ONELINE_FIX_RE.sub("\n", code)
                try:
                    compile(fixed, "<rlm>", "exec")
                    code = fixed
                    note = ("\n[note: your code block was written on a single line; it was "
                            "auto-reformatted. Please use real newlines inside code blocks.]")
                except SyntaxError:
                    return ("[SYNTAX ERROR] Your code block was written on a single line and "
                            "could not be parsed. Rewrite it as a properly formatted multi-line "
                            "```python block with real newlines.")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    tree = None
                literals = []
                if tree:
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == "FINAL"
                                and node.args
                                and isinstance(node.args[0], ast.Constant)):
                            literals.append(str(node.args[0].value))
                env["_rlm_final_literals"] = literals
                if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body[-1]
                    assign = ast.Assign(
                        targets=[ast.Name(id="_rlm_last", ctx=ast.Store())],
                        value=last.value,
                    )
                    tree.body[-1] = ast.copy_location(assign, last)
                    ast.fix_missing_locations(tree)
                    sentinel = object()
                    env["_rlm_last"] = sentinel
                    exec(compile(tree, "<rlm>", "exec"), env)  # noqa: S102
                    val = env.get("_rlm_last")
                    if val is not sentinel and val is not None:
                        env["_"] = val
                        print(val if isinstance(val, str) else repr(val))
                else:
                    exec(code, env)  # noqa: S102
        except Exception:
            buf.write("\n[EXCEPTION]\n" + traceback.format_exc(limit=3))
        out = buf.getvalue()
        if len(out) > obs_limit:
            half = obs_limit // 2
            out = out[:half] + f"\n...[truncated {len(out) - obs_limit} chars]...\n" + out[-half:]
        return (out if out.strip() else "[no output]") + note

    # ---------------- budget / context view ----------------
    def _budget_note(self) -> str:
        if self.budget is None:
            return ""
        return (
            f"\n- MEMORY BUDGET: your working context is capped at ~{self.budget.max_context_tokens} "
            "tokens. Older turns are DROPPED automatically once you exceed it. Anything you will "
            "need for FINAL must be saved with note('...') BEFORE it scrolls out — the scratchpad "
            "survives, raw REPL output does not."
        )

    def _note_help(self) -> str:
        if self.budget is None:
            return "Use it to keep a running list of key findings."
        return ("Notes PERSIST across memory-budget drops, so save every fact you'll need for the "
                "final answer here.")

    def _notes_block(self, notes: list) -> str:
        text = "\n".join(f"- {n}" for n in notes) if notes else "(nothing saved yet)"
        # keep the scratchpad itself within its sub-budget
        if self.budget is not None:
            cap = self.budget.max_notes_tokens
            while self.tok.count(text) > cap and len(text) > 200:
                text = text[: int(len(text) * 0.8)] + "\n- ...[notes truncated]"
        return text

    def _summarize_pairs(self, pairs: list, env, metrics: dict, notes: list) -> None:
        """Compress folded (assistant,observation) pairs into the scratchpad."""
        blob = []
        for pair in pairs:
            for m in pair:
                if m["role"] == "user":  # the observation carries the facts
                    blob.append(m["content"])
        joined = "\n".join(blob)[: self.max_subcall_chars]
        if not joined.strip():
            return
        try:
            summary = self.sub.chat(
                [
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": "Compress these prior findings:\n" + joined},
                ]
            )
            metrics["summarize_calls"] += 1
            metrics["sub_call_tokens"] += self.tok.count(joined) + self.tok.count(summary)
            notes.append("(auto) " + summary.strip())
        except Exception:
            pass

    # ---------------- main loop ----------------
    def run(self, context: str, task: str) -> RLMResult:
        metrics = {
            "steps": 0,
            "root_prompt_tokens": 0,
            "root_completion_tokens": 0,
            "peak_context_tokens": 0,
            "sub_calls": 0,
            "sub_call_tokens": 0,
            "sub_cache_hits": 0,
            "compactions": 0,
            "summarize_calls": 0,
            "notes_saved": 0,
            "budget": (self.budget.max_context_tokens if self.budget else None),
            "strategy": (self.budget.strategy if self.budget else None),
        }
        notes: list = []
        cache: dict = {}
        env = self._make_env(context, metrics, notes, cache)

        system_msg = {
            "role": "system",
            "content": ROOT_SYSTEM_PROMPT.format(
                ctx_len=len(context),
                obs_limit=self.obs_limit,
                max_steps=self.max_steps,
                task=task,
                budget_note=self._budget_note(),
                note_help=self._note_help(),
            ),
        }
        begin_msg = {"role": "user", "content": "Begin. Write your first code block."}

        full_history: list = []   # all (assistant, user) messages, server-side
        summary_index = 0         # number of leading pairs already folded into notes
        transcript = []
        nudges = 0
        seen_output = ""          # grounding accumulator — NEVER compacted

        def build_sent() -> tuple[list, int]:
            """Construct the bounded view actually sent to the root model."""
            nonlocal summary_index
            base = [system_msg, begin_msg]
            if self.budget is None:
                return base + full_history, 0

            n_pairs = len(full_history) // 2
            pairs = [full_history[i * 2:i * 2 + 2] for i in range(n_pairs)]
            keep = self.budget.keep_recent_turns
            kept = pairs[-keep:] if keep > 0 else []

            def assemble(kept_pairs, fold_n):
                msgs = list(base)
                if fold_n > 0 or notes:
                    msgs.append({"role": "user",
                                 "content": NOTES_TEMPLATE.format(n=fold_n,
                                                                  notes=self._notes_block(notes))})
                for p in kept_pairs:
                    msgs.extend(p)
                return msgs

            fold_n = n_pairs - len(kept)
            sent = assemble(kept, fold_n)
            # squeeze: drop more recent pairs until within budget
            while self.tok.count_messages(sent) > self.budget.max_context_tokens and kept:
                kept = kept[1:]
                fold_n = n_pairs - len(kept)
                sent = assemble(kept, fold_n)

            # fold newly-dropped pairs (summarize or just evict)
            if fold_n > summary_index:
                newly = pairs[summary_index:fold_n]
                if self.budget.strategy == "summarize":
                    self._summarize_pairs(newly, env, metrics, notes)
                summary_index = fold_n
                metrics["compactions"] += 1
                sent = assemble(kept, fold_n)  # rebuild (notes may have grown)
            return sent, fold_n

        for step in range(1, self.max_steps + 1):
            metrics["steps"] = step
            sent, _ = build_sent()
            ctx_tokens = self.tok.count_messages(sent)
            metrics["root_prompt_tokens"] += ctx_tokens
            metrics["peak_context_tokens"] = max(metrics["peak_context_tokens"], ctx_tokens)

            reply = self.root.chat(sent)
            metrics["root_completion_tokens"] += self.tok.count(reply)
            blocks = CODE_RE.findall(reply)

            # adaptive observation limit: never let one observation exceed the budget
            obs_limit = self.obs_limit
            if self.budget is not None:
                obs_limit = max(512, min(self.obs_limit, self.budget.max_context_tokens * 3))

            # --- No code block in the reply ---
            if not blocks:
                m = TEXT_FINAL_RE.search(reply)
                if m:
                    val = m.group(1).strip()
                    if val and (val in seen_output or val in task):
                        transcript.append({"step": step, "reply": reply, "code": None,
                                           "observation": "[FINAL parsed from prose]"})
                        return RLMResult(val, step, True, transcript, "final_in_prose", metrics)
                    nudges += 1
                    transcript.append({"step": step, "reply": reply, "code": None,
                                       "observation": "[ungrounded prose FINAL rejected]"})
                    if nudges > 2:
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append({"role": "user", "content":
                        f"REJECTED: your answer {val!r} never appeared in any actual REPL "
                        "output, so it looks like a guess. Do NOT invent answers. Write a "
                        "```python code block that finds the answer in `context` (string "
                        "search / regex / llm_query with the snippet pasted in), look at "
                        "the real output, and only then FINAL it."})
                    continue
                mv = TEXT_FINAL_VAR_RE.search(reply)
                if mv:
                    val = str(env.get(mv.group(1), f"<missing var {mv.group(1)}>"))
                    transcript.append({"step": step, "reply": reply, "code": None,
                                       "observation": "[FINAL_VAR parsed from prose]"})
                    return RLMResult(val, step, True, transcript, "final_var_in_prose", metrics)
                nudges += 1
                transcript.append({"step": step, "reply": reply, "code": None,
                                   "observation": "[no code block - nudged]"})
                if nudges > 2:
                    return RLMResult(reply.strip(), step, False, transcript, "gave_up_no_code", metrics)
                full_history.append({"role": "assistant", "content": reply})
                full_history.append({"role": "user", "content":
                    "Your reply contained no code block, so NOTHING was executed and no "
                    "answer was recorded. Reply with exactly one ```python code block. "
                    "If you already know the answer from a previous REPL output, reply with "
                    "a code block containing only: FINAL(\"your answer\")"})
                continue

            # --- Execute ONLY the first block ---
            code = blocks[0]
            obs = self._exec(code, env, obs_limit)
            if len(blocks) > 1:
                obs += ("\n[WARNING: you wrote multiple code blocks; ONLY the FIRST was "
                        "executed. Anything you wrote after it (including any 'output' you "
                        "predicted) did NOT happen.)")
            transcript.append({"step": step, "reply": reply, "code": code, "observation": obs})
            seen_output += "\n" + obs

            if env["_final_box"]["done"]:
                val = env["_final_box"]["value"]
                if (val in (env.get("_rlm_final_literals") or [])
                        and val not in seen_output and val not in task):
                    env["_final_box"]["done"] = False
                    env["_final_box"]["value"] = None
                    nudges += 1
                    if nudges > 2:
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append({"role": "user", "content":
                        f"REJECTED: FINAL({val!r}) is a literal constant that never appeared "
                        "in any REPL output — it looks like a guess. Find the real answer in "
                        "`context` first (e.g. re.search / context.find / llm_query with the "
                        "snippet pasted in), print it, then call FINAL on the variable "
                        "holding it."})
                    continue
                return RLMResult(val, step, True, transcript, "final_called", metrics)

            full_history.append({"role": "assistant", "content": reply})
            full_history.append({"role": "user", "content":
                f"ACTUAL REPL output:\n```\n{obs}\n```\n"
                f"Continue. ({self.max_steps - step} turns left) "
                "Remember: one code block only; finish with FINAL(...) once you have "
                "seen the answer in a real output."})

        return RLMResult(None, self.max_steps, False, transcript, "max_steps", metrics)


def vanilla_answer(client: NIMClient, context: str, task: str, char_limit: int = 400_000) -> str:
    """Baseline: stuff (possibly truncated) context directly into the prompt."""
    truncated = context[:char_limit]
    note = "" if len(context) <= char_limit else "\n[NOTE: document truncated]"
    prompt = f"Document:\n{truncated}{note}\n\nTask: {task}\nAnswer concisely."
    return client.chat([{"role": "user", "content": prompt}])
