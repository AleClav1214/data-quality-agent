"""Compare the agent with the rule-based script and with reference points.

Every number here is computed when you run it, from files on disk
(run.json, the answer keys, the clean data). Nothing is copied from the
README. REPORTED holds what the README currently says, only so we can
warn when a fresh run no longer matches it.

Reference points for Auto MPG (not agent results):
  projection  the script's issues PLUS a weight business rule (1,500-6,000 lbs),
              i.e. what we'd expect if the agent sets that rule and does
              everything else the script does
  floor       every planted error caught (from the answer key); the best
              any auditor could do
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import tools
from .evaluate import score
from .fixes import numeric_pairs
from .regression import X, Y, cleaned_fit, fit, issues_from_answer_key, slope_error_pct
from .tools import AuditState, load_csv

WEIGHT_RULE = (1500, 6000)

# What the README reports today (earlier real runs). Used only for drift warnings.
REPORTED = {
    ("mpg", "baseline"): {"caught": 66, "planted": 72, "false_alarms": 0, "slope_error_pct": 4.5},
    ("package_log", "baseline"): {"caught": 227, "planted": 240, "false_alarms": 197},
    ("mpg", "raw"): {"slope_error_pct": 93.1},
    ("mpg", "projection"): {"slope_error_pct": 2.5},
    ("mpg", "floor"): {"slope_error_pct": 1.0},
}

# USD per million tokens (input, output), first-party API list prices as of 2026-10.
# Used only to estimate what a run cost; check your console for the real bill.
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
                  "claude-opus-5-5": (4.00, 20.00)}

DATASETS = ("mpg", "package_log")


def totals(issues: list[dict], truth: pd.DataFrame) -> dict:
    row = score(issues, truth).set_index("category").loc["ALL"]
    return {k: (int(row[k]) if k in ("caught", "planted", "false_alarms", "flagged") else float(row[k]))
            for k in ("caught", "planted", "flagged", "false_alarms", "precision", "recall", "f1")}


def kg_weight_rows(messy: pd.DataFrame, truth: pd.DataFrame) -> set[int]:
    """Rows where weight was planted in kilograms. The answer key files both
    planted weight errors as out_of_range; the other kind (an extra zero) is
    always over 16,000 lbs, so anything under 6,000 is a kilogram entry."""
    w = pd.to_numeric(messy["weight"], errors="coerce")
    planted = truth[(truth["category"] == "out_of_range") & (truth["column"] == "weight")]["row_index"]
    return {int(r) for r in planted if w.get(r, float("nan")) < WEIGHT_RULE[1]}


def kg_caught(issues: list[dict], kg_rows: set[int]) -> int:
    flagged = {int(r) for i in issues if i["category"] == "out_of_range" for r in i["rows"]}
    return len(kg_rows & flagged)


def projection_issues(messy: pd.DataFrame, baseline_issues: list[dict]) -> list[dict]:
    """The script's issues plus the weight business rule, run through the real tool."""
    state = AuditState(df=messy)
    result = tools.check_numeric_range(state, "weight", *WEIGHT_RULE)
    for f in result["findings"]:
        if f["category"] == "out_of_range":
            tools.log_issue(state, f["finding_id"], "high", "weight outside business rule", "exclude")
    extra = [{**i, "issue_id": f"P{n:02d}"} for n, i in enumerate(state.issues, 1)]
    return baseline_issues + extra


def estimated_cost(model: str, usage: dict) -> float | None:
    if model not in PRICE_PER_MTOK or not usage:
        return None
    p_in, p_out = PRICE_PER_MTOK[model]
    tokens_in = sum(usage.get(k, 0) for k in
                    ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return (tokens_in * p_in + usage.get("output_tokens", 0) * p_out) / 1e6


def _row(dataset: str, auditor: str, run: dict | None, truth: pd.DataFrame, **extra) -> dict:
    row = {"dataset": dataset, "auditor": auditor, "kind": "result", **extra}
    if run is not None:
        row.update(totals(run["issues"], truth))
        row.update(model=run.get("model", ""), stop_reason=run.get("stop_reason", ""),
                   data_unchanged=run.get("data_unchanged"), tool_calls=len(run.get("tool_log", [])),
                   usage=run.get("usage") or {})
        row["estimated_cost_usd"] = estimated_cost(row["model"], row["usage"])
    return row


def build_summary(runs: dict[tuple[str, str], dict], data_dir: str | Path = "data") -> dict:
    """runs maps (dataset, 'baseline'|'agent') -> loaded run.json."""
    data = Path(data_dir)
    rows: list[dict] = []

    messy, clean = load_csv(data / "mpg_messy.csv"), load_csv(data / "mpg_clean.csv")
    truth = pd.read_csv(data / "mpg_ground_truth.csv")
    true_fit = fit(numeric_pairs(clean, X, Y))
    ts = true_fit["slope"]
    kg_rows = kg_weight_rows(messy, truth)

    raw = fit(numeric_pairs(messy, X, Y))
    rows.append({"dataset": "mpg", "auditor": "No cleaning", "kind": "reference",
                 "slope_error_pct": slope_error_pct(raw["slope"], ts), "n": raw["n"]})
    for mode, label in (("baseline", "Baseline script"), ("agent", "AI agent")):
        run = runs.get(("mpg", mode))
        if run is None:
            continue
        f = cleaned_fit(messy, run["issues"], ts)
        rows.append(_row("mpg", label, run, truth, slope_error_pct=f["slope_error_pct"], n=f["n"],
                         kg_caught=kg_caught(run["issues"], kg_rows), kg_planted=len(kg_rows)))
    if ("mpg", "baseline") in runs:
        proj = projection_issues(messy, runs[("mpg", "baseline")]["issues"])
        f = cleaned_fit(messy, proj, ts)
        rows.append({"dataset": "mpg", "auditor": "Projection: script + weight rule 1,500-6,000 lbs",
                     "kind": "projection", "slope_error_pct": f["slope_error_pct"], "n": f["n"],
                     "kg_caught": kg_caught(proj, kg_rows), "kg_planted": len(kg_rows)})
    f = cleaned_fit(messy, issues_from_answer_key(truth), ts)
    rows.append({"dataset": "mpg", "auditor": "Floor: every planted error caught", "kind": "reference",
                 "slope_error_pct": f["slope_error_pct"], "n": f["n"]})

    pk_truth = pd.read_csv(data / "package_log_ground_truth.csv")
    for mode, label in (("baseline", "Baseline script"), ("agent", "AI agent")):
        if ("package_log", mode) in runs:
            rows.append(_row("package_log", label, runs[("package_log", mode)], pk_truth))

    return {"true_slope_per_1000lbs": ts * 1000, "rows": rows,
            "checks": expectation_checks(rows), "drift": drift_warnings(rows)}


def _find(rows: list[dict], dataset: str, auditor_prefix: str) -> dict | None:
    return next((r for r in rows if r["dataset"] == dataset and r["auditor"].startswith(auditor_prefix)), None)


def expectation_checks(rows: list[dict]) -> list[str]:
    """Plain-language comparison of the agent with the script and the projection.
    Reports whatever happened, including when the agent does worse."""
    out: list[str] = []
    agent, base = _find(rows, "mpg", "AI agent"), _find(rows, "mpg", "Baseline")
    proj, floor = _find(rows, "mpg", "Projection"), _find(rows, "mpg", "Floor")
    if agent and base and proj and floor:
        a, b, p, fl = (r["slope_error_pct"] for r in (agent, base, proj, floor))
        out.append(f"MPG slope error: agent {a:.1f}% vs script {b:.1f}% -> agent is "
                   f"{'better' if a < b else 'worse' if a > b else 'the same'} by {abs(a - b):.1f} points.")
        out.append(f"MPG vs projection ({p:.1f}%): agent is {'lower' if a < p else 'higher' if a > p else 'equal'} "
                   f"by {abs(a - p):.1f} points. (The projection is a what-if, not an agent result.)")
        out.append(f"Expected range {fl:.1f}%-{b:.1f}% (floor to script): agent is "
                   f"{'inside' if fl <= a <= b else 'OUTSIDE'} it.")
        out.append(f"Kilogram weights caught: agent {agent['kg_caught']}/{agent['kg_planted']}, "
                   f"script {base['kg_caught']}/{base['kg_planted']}, projection {proj['kg_caught']}/{proj['kg_planted']}.")
    for ds in ("mpg", "package_log"):
        agent, base = _find(rows, ds, "AI agent"), _find(rows, ds, "Baseline")
        if agent and base:
            out.append(f"{ds}: recall {agent['recall']:.0%} vs {base['recall']:.0%} (script), "
                       f"precision {agent['precision']:.0%} vs {base['precision']:.0%}, "
                       f"false alarms {agent['false_alarms']} vs {base['false_alarms']}.")
    for r in rows:
        if r["auditor"] == "AI agent":
            if r.get("stop_reason") != "agent finished":
                out.append(f"WARNING {r['dataset']}: agent did not finish normally ({r.get('stop_reason')}).")
            if r.get("data_unchanged") is not True:
                out.append(f"WARNING {r['dataset']}: data fingerprint changed during the audit.")
    return out


def drift_warnings(rows: list[dict]) -> list[str]:
    """Flag fresh numbers that no longer match what the README reports."""
    lookup = {("mpg", "baseline"): _find(rows, "mpg", "Baseline"),
              ("package_log", "baseline"): _find(rows, "package_log", "Baseline"),
              ("mpg", "raw"): _find(rows, "mpg", "No cleaning"),
              ("mpg", "projection"): _find(rows, "mpg", "Projection"),
              ("mpg", "floor"): _find(rows, "mpg", "Floor")}
    out = []
    for key, expected in REPORTED.items():
        row = lookup.get(key)
        if row is None:
            continue
        for field, value in expected.items():
            got = row.get(field)
            same = round(got, 1) == value if field == "slope_error_pct" else got == value
            if not same:
                out.append(f"{key[0]} {key[1]} {field}: README says {value}, this run gives {got}")
    return out


def _pct(v, digits=0) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v * 100:.{digits}f}%"


def format_table(summary: dict) -> str:
    """Markdown table in the same shape as the README results table."""
    lines = ["| Dataset | Auditor | Caught | Recall | False alarms | Precision | kg weights caught | Slope error |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["rows"]:
        caught = f"{r['caught']} / {r['planted']}" if "caught" in r else "—"
        kg = f"{r['kg_caught']} / {r['kg_planted']}" if "kg_caught" in r else ""
        slope = f"{r['slope_error_pct']:.1f}%" if "slope_error_pct" in r else "—"
        label = f"*{r['auditor']}*" if r["kind"] != "result" else f"**{r['auditor']}**"
        lines.append(f"| {r['dataset']} | {label} | {caught} | {_pct(r.get('recall'))} | "
                     f"{r.get('false_alarms', '—')} | {_pct(r.get('precision'))} | {kg} | {slope} |")
    return "\n".join(lines)


def format_summary(summary: dict, header: str = "") -> str:
    out = [header] if header else []
    out += [f"True slope: {summary['true_slope_per_1000lbs']:.2f} mpg per 1,000 lbs", "", format_table(summary), ""]
    agents = [r for r in summary["rows"] if r["auditor"] == "AI agent"]
    for r in agents:
        u = r["usage"]
        cost = r["estimated_cost_usd"]
        out.append(f"Agent on {r['dataset']}: model {r['model']}, {u.get('api_calls', 0)} API calls, "
                   f"{r['tool_calls']} tool calls, {u.get('input_tokens', 0):,} input + "
                   f"{u.get('output_tokens', 0):,} output tokens"
                   + (f", about ${cost:.3f}" if cost is not None else "") + f"; stop: {r['stop_reason']}")
    if agents:
        out.append("")
    if summary["checks"]:
        out += ["What the numbers say:"] + [f"  - {c}" for c in summary["checks"]] + [""]
    if summary["drift"]:
        out += ["README drift (fresh numbers differ from what the README reports):"]
        out += [f"  - {d}" for d in summary["drift"]] + [""]
    else:
        out += ["README check: the script, raw, projection and floor numbers match the README.", ""]
    return "\n".join(out)
