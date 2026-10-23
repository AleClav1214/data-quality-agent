"""The agent loop.

    goal + tools  ->  model plans  ->  model calls a tool  ->  we run it
         ^                                                       |
         |------------- model reads the result  <----------------|

The loop repeats until the model calls `finish`, stops asking for
tools, or hits the step limit (a guardrail against runaway cost).
"""

from __future__ import annotations

import os

import pandas as pd

from .registry import TOOL_SPECS, dispatch, to_model_text
from .tools import AuditState

BUILTIN_MODEL = "claude-sonnet-5-5"  # used for the final runs we report
DEFAULT_MODEL = os.environ.get("DQ_AGENT_MODEL", BUILTIN_MODEL)
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

SYSTEM_PROMPT = """You are a careful data quality auditor working for a consulting team.
Your goal: find the problems in a dataset that would mislead a dashboard or break a
system migration, and explain them so a business user can act.

How you work:
- You can only see the data through tools. Never invent counts, rows, or values.
  Every number you mention must come from a tool result.
- You cannot change the data. You recommend fixes; a person approves them.
- Start with profile_dataset, then choose checks that fit what each column means.
- Use business sense. For example, set realistic min/max limits instead of relying
  on a statistical outlier rule, and only check uniqueness on IDs that should be unique.
- When a result looks surprising, investigate further before deciding (for example,
  use check_missing with group_by to see whether blanks are expected).
- Log each real problem once with log_issue. Do NOT log findings that are expected
  given what the data represents, and say why you skipped them in your summary.
- Finish by calling finish with a short, plain-language summary."""


def _block_to_dict(block) -> dict:
    """Convert SDK content blocks to plain dicts so the transcript is easy
    to store and the loop also works with a fake client in tests."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if hasattr(block, "model_dump"):  # other block types (e.g. thinking) are passed back unchanged
        return block.model_dump()
    return {"type": block.type}


def run_agent(df: pd.DataFrame, *, dataset_description: str = "", client=None,
              model: str = DEFAULT_MODEL, max_steps: int = 30,
              reference_date: str | None = None, verbose: bool = True) -> AuditState:
    """Run the audit. Returns the AuditState holding findings, logged
    issues, the tool log, and the agent's summary."""
    state = AuditState(df=df, model=model)
    state.usage = {"api_calls": 0, **{k: 0 for k in USAGE_FIELDS}}
    if reference_date:
        state.reference_date = pd.Timestamp(reference_date)

    if client is None:
        import anthropic  # imported here so tests and the baseline don't need it

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    goal = (
        f"Audit this dataset ({len(df)} rows, columns: {list(df.columns)}). "
        f"Today's date for this audit is {state.reference_date.date()}. "
        + (f"What the data represents: {dataset_description}" if dataset_description else "")
    )
    messages: list[dict] = [{"role": "user", "content": goal}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM_PROMPT,
            tools=TOOL_SPECS, messages=messages,
        )
        state.usage["api_calls"] += 1
        usage = getattr(response, "usage", None)  # token counts as billed (the fake test client has none)
        for k in USAGE_FIELDS:
            state.usage[k] += getattr(usage, k, None) or 0
        content = [_block_to_dict(b) for b in response.content]
        messages.append({"role": "assistant", "content": content})

        if verbose:
            for b in content:
                if b["type"] == "text" and b["text"].strip():
                    print(f"[step {step}] thinking: {b['text'].strip()[:300]}")

        tool_calls = [b for b in content if b["type"] == "tool_use"]
        if not tool_calls:
            state.stop_reason = state.stop_reason or "model stopped without calling finish"
            break

        results = []
        for call in tool_calls:
            result = dispatch(state, call["name"], call["input"])
            if verbose:
                found = [f["finding_id"] for f in result.get("findings", [])]
                status = result.get("error") or (f"findings {found}" if found else
                                                 result.get("status") or "no problems")
                print(f"[step {step}] {call['name']}({call['input']}) -> {status}")
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": to_model_text(result),
                            "is_error": "error" in result})
        messages.append({"role": "user", "content": results})

        if state.done:
            break
    else:
        state.stop_reason = f"step limit ({max_steps}) reached"

    state.transcript = messages  # type: ignore[attr-defined]
    return state
