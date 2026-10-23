"""Baseline: a plain script with no AI.

It runs every check on every column it can guess a type for and logs
everything it finds. It is useful two ways:
  1. You can run the whole project without an API key.
  2. It is the comparison point: does the agent's judgment beat a script?
"""

from __future__ import annotations

import pandas as pd

from . import tools
from .tools import AuditState

DEFAULT_SEVERITY = {
    "duplicate_key": "high", "date_order": "high", "date_out_of_range": "high",
    "out_of_range": "high", "missing": "medium", "duplicate_row": "medium",
    "date_format": "medium", "invalid_format": "medium", "category_variant": "low",
}

DEFAULT_FIX = {
    "missing": "Fill in from the source system or mark as unknown; confirm whether blanks are expected.",
    "duplicate_row": "Remove the extra copies after confirming they are not separate real events.",
    "duplicate_key": "Investigate which record is correct and assign a new ID to the other.",
    "out_of_range": "Check against the source record; likely a typo or unit error.",
    "date_format": "Convert to the standard YYYY-MM-DD format.",
    "date_out_of_range": "Check for a mistyped year and correct it.",
    "date_order": "Check both dates against the source; one of them is likely wrong.",
    "category_variant": "Map the variant spelling to the standard label.",
    "invalid_format": "Correct the value or contact the owner for the right one.",
}


def run_baseline(df: pd.DataFrame, reference_date: str | None = None) -> AuditState:
    state = AuditState(df=df)
    if reference_date:
        state.reference_date = pd.Timestamp(reference_date)

    profile = tools.profile_dataset(state)
    state.tool_log.append({"tool": "profile_dataset", "args": {}})
    calls: list[tuple] = [(tools.check_missing, {}), (tools.check_duplicates, {})]

    date_cols = []
    for col in profile["columns"]:
        name, kind = col["column"], col["inferred_type"]
        if kind == "numeric":
            calls.append((tools.check_numeric_range, {"column": name}))  # IQR rule: no domain knowledge
        elif kind == "date":
            date_cols.append(name)
            calls.append((tools.check_date_values, {"column": name}))
        elif kind == "category":
            calls.append((tools.check_category_variants, {"column": name}))
        if "email" in name.lower():
            calls.append((tools.check_pattern, {"column": name, "pattern": "email"}))
        if name.lower().endswith("id") and col["unique_ratio"] > 0.95:
            calls.append((tools.check_duplicate_keys, {"column": name}))
    if len(date_cols) == 2:
        calls.append((tools.check_date_order, {"start_column": date_cols[0], "end_column": date_cols[1]}))

    for fn, args in calls:
        result = fn(state, **args)
        state.tool_log.append({"tool": fn.__name__, "args": args,
                               "findings": [f["finding_id"] for f in result.get("findings", [])]})

    for fid, finding in state.findings.items():  # a script cannot tell expected from wrong
        tools.log_issue(state, fid, DEFAULT_SEVERITY.get(finding.category, "medium"),
                        "Flagged automatically by a rule-based check.",
                        DEFAULT_FIX.get(finding.category, "Review manually."))
    tools.finish(state, f"Rule-based scan logged {len(state.issues)} issues. "
                        "No judgment was applied about which findings are expected.")
    state.stop_reason = "baseline complete"
    return state
