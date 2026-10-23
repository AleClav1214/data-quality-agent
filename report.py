"""Turn an audit into files a person can review.

- report.md         readable summary for a manager
- issues.csv        one row per issue, with an empty 'approved' column for
                    the human reviewer (human-in-the-loop)
- flagged_rows.csv  every affected row, with its Excel row number
- run.json          everything, for evaluation and debugging
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .tools import AuditState

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def excel_row(row_index: int) -> int:
    return row_index + 2  # +1 for 1-based rows, +1 for the header row


def build_markdown(state: AuditState, title: str, mode: str) -> str:
    issues = sorted(state.issues, key=lambda i: (SEVERITY_ORDER[i["severity"]], -len(i["rows"])))
    n_rows = len({r for i in issues for r in i["rows"]})
    lines = [
        f"# Data Quality Report: {title}",
        "",
        f"**Mode:** {mode} · **Rows audited:** {len(state.df):,} · "
        f"**Issues:** {len(issues)} · **Rows affected:** {n_rows:,} "
        f"({n_rows / max(len(state.df), 1):.1%})",
        "",
        f"**Data unchanged during audit:** {'yes (verified by fingerprint)' if state.data_unchanged() else 'NO'}",
        "",
        "## Summary",
        "",
        state.summary or "_No summary provided._",
        "",
        "## Issues",
        "",
        "| ID | Severity | Column | Problem | Rows | Suggested fix |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for i in issues:
        lines.append(f"| {i['issue_id']} | {i['severity']} | {i['column'] or 'all'} | "
                     f"{i['description']} | {len(i['rows'])} | {i['suggested_fix']} |")
    lines += ["", "## Details", ""]
    for i in issues:
        rows_preview = ", ".join(str(excel_row(r)) for r in i["rows"][:10])
        more = f" (+{len(i['rows']) - 10} more)" if len(i["rows"]) > 10 else ""
        lines += [
            f"### {i['issue_id']} · {i['severity'].upper()} · {i['description']}",
            "",
            f"- **Why it matters:** {i['why_it_matters']}",
            f"- **Suggested fix:** {i['suggested_fix']}",
            f"- **Examples:** {'; '.join(str(e) for e in i['examples'][:5])}",
            f"- **Spreadsheet rows:** {rows_preview}{more}",
            "",
        ]
    lines += ["## Steps the auditor took", ""]
    for n, t in enumerate(state.tool_log, 1):
        note = f" → error: {t['error']}" if t.get("error") else \
            (f" → {', '.join(t['findings'])}" if t.get("findings") else "")
        lines.append(f"{n}. `{t['tool']}` {t.get('args') or ''}{note}")
    if state.stop_reason:
        lines += ["", f"_Stopped because: {state.stop_reason}_"]
    return "\n".join(lines) + "\n"


def save_outputs(state: AuditState, out_dir: str | Path, title: str, mode: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.md").write_text(build_markdown(state, title, mode), encoding="utf-8")

    with open(out / "issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue_id", "severity", "category", "column", "description",
                    "rows_affected", "why_it_matters", "suggested_fix", "approved"])
        for i in state.issues:
            w.writerow([i["issue_id"], i["severity"], i["category"], i["column"], i["description"],
                        len(i["rows"]), i["why_it_matters"], i["suggested_fix"], ""])

    with open(out / "flagged_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "excel_row", "issue_id", "category", "column"])
        for i in state.issues:
            for r in i["rows"]:
                w.writerow([r, excel_row(r), i["issue_id"], i["category"], i["column"]])

    run = {
        "mode": mode,
        "title": title,
        "rows": len(state.df),
        "summary": state.summary,
        "stop_reason": state.stop_reason,
        "data_unchanged": state.data_unchanged(),
        "issues": state.issues,
        "skipped_findings": [f.summary() for fid, f in state.findings.items()
                             if fid not in {i["finding_id"] for i in state.issues}],
        "tool_log": state.tool_log,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return out
