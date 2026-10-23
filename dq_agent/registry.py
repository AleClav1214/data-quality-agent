"""Tool registry: tells the model what tools exist (JSON schemas) and
routes the model's tool calls to the Python functions in tools.py.

This is "function calling": the model never runs code itself. It asks
for a tool by name with arguments; we run it and send back the result.
"""

from __future__ import annotations

import json
from typing import Callable

from . import tools
from .tools import AuditState

_str = {"type": "string"}
_num = {"type": ["number", "null"]}

TOOL_SPECS: list[dict] = [
    {
        "name": "profile_dataset",
        "description": "Overview of every column: inferred type, missing count, unique count, "
                       "and most common values. Call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_missing",
        "description": "Find blank or placeholder values ('', 'N/A', 'null'). Optionally break "
                       "missing rows down by another column (group_by) to judge whether the "
                       "blanks are expected, e.g. no pickup date because status is 'Waiting'.",
        "input_schema": {"type": "object", "properties": {
            "columns": {"type": "array", "items": _str,
                        "description": "Columns to check; omit for all."},
            "group_by": {"type": "string", "description": "Optional column to break missing rows down by."},
        }},
    },
    {
        "name": "check_duplicates",
        "description": "Find rows that are exact copies of each other.",
        "input_schema": {"type": "object", "properties": {
            "subset": {"type": "array", "items": _str,
                       "description": "Columns to compare; omit to compare whole rows."},
        }},
    },
    {
        "name": "check_duplicate_keys",
        "description": "Find IDs that should be unique (like an order or package ID) but appear on "
                       "rows with different details. Do not use on columns that are expected to "
                       "repeat (like a customer ID in an orders table).",
        "input_schema": {"type": "object", "properties": {"column": _str}, "required": ["column"]},
    },
    {
        "name": "check_numeric_range",
        "description": "Flag numbers outside a business rule [min_value, max_value]. Prefer giving "
                       "realistic limits from domain knowledge; if both are omitted a statistical "
                       "outlier rule is used, which can flag unusual-but-valid values.",
        "input_schema": {"type": "object", "properties": {
            "column": _str, "min_value": _num, "max_value": _num,
        }, "required": ["column"]},
    },
    {
        "name": "check_date_values",
        "description": "Check that a date column uses one format (default YYYY-MM-DD, i.e. %Y-%m-%d) "
                       "and falls within [min_date, max_date]. max_date defaults to the reference "
                       "date, so future dates are flagged.",
        "input_schema": {"type": "object", "properties": {
            "column": _str,
            "expected_format": {"type": "string", "description": "strftime format, e.g. %Y-%m-%d"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD"},
        }, "required": ["column"]},
    },
    {
        "name": "check_date_order",
        "description": "Flag rows where end_column is earlier than start_column "
                       "(e.g. a package picked up before it arrived).",
        "input_schema": {"type": "object", "properties": {
            "start_column": _str, "end_column": _str,
            "expected_format": {"type": "string"},
        }, "required": ["start_column", "end_column"]},
    },
    {
        "name": "check_category_variants",
        "description": "Find misspelled or inconsistent labels in a category column "
                       "(e.g. 'fedex', 'Fed Ex' vs 'FedEx').",
        "input_schema": {"type": "object", "properties": {"column": _str}, "required": ["column"]},
    },
    {
        "name": "check_pattern",
        "description": "Check values against a format. pattern can be 'email', 'us_phone', "
                       "'us_zip', or a regular expression.",
        "input_schema": {"type": "object", "properties": {"column": _str, "pattern": _str},
                         "required": ["column", "pattern"]},
    },
    {
        "name": "log_issue",
        "description": "Add a finding to the final report with your judgment. Only log real "
                       "problems; skip findings that are expected given what the data represents.",
        "input_schema": {"type": "object", "properties": {
            "finding_id": {"type": "string", "description": "ID from a check result, e.g. F003"},
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "why_it_matters": {"type": "string", "description": "One sentence on the business impact."},
            "suggested_fix": {"type": "string", "description": "What a person should do. You cannot change the data."},
        }, "required": ["finding_id", "severity", "why_it_matters", "suggested_fix"]},
    },
    {
        "name": "finish",
        "description": "End the audit with a 2-4 sentence summary for a non-technical manager.",
        "input_schema": {"type": "object", "properties": {"summary": _str}, "required": ["summary"]},
    },
]

_FUNCTIONS: dict[str, Callable] = {spec["name"]: getattr(tools, spec["name"]) for spec in TOOL_SPECS}

MAX_RESULT_CHARS = 6000  # cap what goes back into the model's context


def dispatch(state: AuditState, name: str, args: dict | None) -> dict:
    """Run one tool call. Errors are returned to the model as data so it
    can correct itself (e.g. a misspelled column name) instead of crashing."""
    args = {k: v for k, v in (args or {}).items() if v is not None}
    entry = {"tool": name, "args": args}
    try:
        if name not in _FUNCTIONS:
            raise ValueError(f"Unknown tool '{name}'")
        result = _FUNCTIONS[name](state, **args)
        entry["findings"] = [f["finding_id"] for f in result.get("findings", [])]
    except Exception as exc:  # noqa: BLE001 - we want every error surfaced to the agent
        result = {"error": f"{type(exc).__name__}: {exc}"}
        entry["error"] = result["error"]
    state.tool_log.append(entry)
    return result


def to_model_text(result: dict) -> str:
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + '... [truncated]'
    return text
