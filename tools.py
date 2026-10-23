"""Data quality check functions (the agent's "tools").

Design rule: CODE DOES THE COUNTING, THE AI DOES THE REASONING.
Every number, row and example the agent ever sees comes from these
functions. The model never computes anything itself, so it cannot
hallucinate a count.

Guardrail: every function here is READ-ONLY. None of them modify the
DataFrame. AuditState fingerprints the data at the start of the audit
so we can prove at the end that nothing changed.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Text that people commonly type to mean "no value".
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-", "?"}

# Named patterns the agent can ask for instead of writing its own regex.
PATTERNS = {
    "email": r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$",
    "us_phone": r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
    "us_zip": r"^\d{5}(-\d{4})?$",
}

MAX_EXAMPLES = 5  # keep tool results small so the agent's context stays manageable


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass
class Finding:
    """Something a check discovered. Rows are stored here, in code, so the
    model can refer to a finding by ID without ever restating row numbers."""

    finding_id: str
    check: str
    category: str
    column: str | None
    description: str
    rows: list[int]
    examples: list[Any]

    def summary(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "column": self.column,
            "description": self.description,
            "rows_affected": len(self.rows),
            "examples": self.examples[:MAX_EXAMPLES],
        }


@dataclass
class AuditState:
    """Everything the agent has done and found so far (its working memory)."""

    df: pd.DataFrame
    reference_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today().normalize())
    findings: dict[str, Finding] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    summary: str = ""
    done: bool = False
    stop_reason: str = ""
    fingerprint_start: str = ""

    def __post_init__(self) -> None:
        self.fingerprint_start = fingerprint(self.df)

    def add_finding(self, check: str, category: str, column: str | None,
                    description: str, rows, examples) -> Finding:
        fid = f"F{len(self.findings) + 1:03d}"
        finding = Finding(fid, check, category, column, description,
                          sorted(int(r) for r in rows), list(examples))
        self.findings[fid] = finding
        return finding

    def data_unchanged(self) -> bool:
        return fingerprint(self.df) == self.fingerprint_start


def fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the full table, used to verify the audit was read-only."""
    hashed = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed.tobytes() + "|".join(df.columns).encode()).hexdigest()


def load_csv(path: str) -> pd.DataFrame:
    """Load everything as raw text so messy values (bad dates, '999 lbs')
    survive exactly as typed instead of being silently converted."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Unknown column '{column}'. Columns are: {list(df.columns)}")


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), "").astype(str)


def _is_missing(series: pd.Series) -> pd.Series:
    return _as_text(series).str.strip().str.lower().isin(MISSING_TOKENS)


def _present(df: pd.DataFrame, column: str) -> pd.Series:
    """The non-missing values of a column, with their original row index."""
    s = _as_text(df[column])
    return s[~_is_missing(df[column])]


def _result(state: AuditState, check: str, column, findings: list[Finding], **extra) -> dict:
    out = {
        "check": check,
        "column": column,
        "rows_checked": len(state.df),
        "findings": [f.summary() for f in findings],
    }
    if not findings:
        out["message"] = "No problems found by this check."
    out.update(extra)
    return out


def _parse_dates_strict(values: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(values, format=fmt, errors="coerce")


def _parse_dates_loose(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="mixed", errors="coerce")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def profile_dataset(state: AuditState) -> dict:
    """Overview of every column: type guess, missing count, unique count,
    most common values. This is usually the agent's first step."""
    df = state.df
    columns = []
    for col in df.columns:
        present = _present(df, col)
        n_present = len(present)
        numeric_share = pd.to_numeric(present, errors="coerce").notna().mean() if n_present else 0
        date_share = _parse_dates_loose(present).notna().mean() if n_present else 0
        if numeric_share >= 0.9:
            inferred = "numeric"
        elif date_share >= 0.8 and numeric_share < 0.5:
            inferred = "date"
        elif present.nunique() <= max(20, 0.02 * n_present):
            inferred = "category"
        else:
            inferred = "text / identifier"
        top = present.value_counts().head(5)
        columns.append({
            "column": col,
            "inferred_type": inferred,
            "missing": int(_is_missing(df[col]).sum()),
            "unique_values": int(present.nunique()),
            "unique_ratio": round(present.nunique() / n_present, 3) if n_present else 0,
            "most_common": {str(k): int(v) for k, v in top.items()},
        })
    return {"rows": len(df), "columns": columns,
            "reference_date": str(state.reference_date.date())}


def check_missing(state: AuditState, columns: list[str] | None = None,
                  group_by: str | None = None) -> dict:
    """Find blank or placeholder values ('N/A', 'null', ...).
    group_by shows how the missing rows break down by another column,
    which helps decide whether blanks are expected (e.g. no pickup date
    because the package is still waiting)."""
    df = state.df
    cols = columns or list(df.columns)
    if group_by:
        _require_column(df, group_by)
    findings = []
    for col in cols:
        _require_column(df, col)
        mask = _is_missing(df[col])
        if not mask.any():
            continue
        desc = f"{int(mask.sum())} missing values in '{col}'"
        examples = []
        if group_by:
            breakdown = _as_text(df.loc[mask, group_by]).value_counts().head(8)
            examples = [f"{group_by}={k}: {v} rows" for k, v in breakdown.items()]
            desc += f" (broken down by '{group_by}' in examples)"
        else:
            examples = [repr(v) for v in _as_text(df.loc[mask, col]).value_counts().index[:MAX_EXAMPLES]]
        findings.append(state.add_finding("check_missing", "missing", col, desc,
                                          df.index[mask], examples))
    return _result(state, "check_missing", columns, findings)


def check_duplicates(state: AuditState, subset: list[str] | None = None) -> dict:
    """Find rows that are exact copies of each other (all rows in each
    group are returned, including the first)."""
    df = state.df
    for col in subset or []:
        _require_column(df, col)
    mask = df.duplicated(subset=subset, keep=False)
    findings = []
    if mask.any():
        dupes = df[mask]
        n_groups = dupes.groupby(list(subset or df.columns), dropna=False).ngroups
        examples = [dupes.iloc[i].to_dict() for i in range(min(2, len(dupes)))]
        findings.append(state.add_finding(
            "check_duplicates", "duplicate_row", None,
            f"{int(mask.sum())} rows are exact duplicates ({n_groups} groups of identical rows)",
            df.index[mask], examples))
    return _result(state, "check_duplicates", None, findings)


def check_duplicate_keys(state: AuditState, column: str) -> dict:
    """Find IDs that should be unique but appear on rows with DIFFERENT
    details (exact copies are left to check_duplicates)."""
    df = state.df
    _require_column(df, column)
    present = _present(df, column)
    repeated = present[present.duplicated(keep=False)]
    conflict_rows = []
    examples = []
    for key, idx in repeated.groupby(repeated).groups.items():
        group = df.loc[idx]
        if len(group.drop_duplicates()) > 1:  # same ID, different details
            conflict_rows.extend(idx)
            if len(examples) < MAX_EXAMPLES:
                examples.append(f"{column}={key} appears on {len(idx)} different rows")
    findings = []
    if conflict_rows:
        findings.append(state.add_finding(
            "check_duplicate_keys", "duplicate_key", column,
            f"{len(conflict_rows)} rows share an ID in '{column}' but have different details",
            conflict_rows, examples))
    return _result(state, "check_duplicate_keys", column, findings,
                   unique_ratio=round(present.nunique() / max(len(present), 1), 3))


def check_numeric_range(state: AuditState, column: str, min_value: float | None = None,
                        max_value: float | None = None) -> dict:
    """Flag numbers outside a business rule (min/max). If no limits are
    given, falls back to a statistical outlier rule (3 x IQR), which is
    less reliable because unusual is not the same as wrong."""
    df = state.df
    _require_column(df, column)
    present = _present(df, column)
    numbers = pd.to_numeric(present, errors="coerce")
    findings = []

    not_numbers = numbers.isna()
    if not_numbers.any():
        findings.append(state.add_finding(
            "check_numeric_range", "invalid_format", column,
            f"{int(not_numbers.sum())} values in '{column}' are not numbers",
            present.index[not_numbers], present[not_numbers].unique()[:MAX_EXAMPLES].tolist()))

    valid = numbers.dropna()
    if min_value is None and max_value is None:
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        method = f"statistical outlier rule (3 x IQR): outside [{lo:.2f}, {hi:.2f}]"
    else:
        lo = -np.inf if min_value is None else float(min_value)
        hi = np.inf if max_value is None else float(max_value)
        method = f"business rule: outside [{min_value}, {max_value}]"
    out = valid[(valid < lo) | (valid > hi)]
    if len(out):
        findings.append(state.add_finding(
            "check_numeric_range", "out_of_range", column,
            f"{len(out)} values in '{column}' fall outside the {method}",
            out.index, sorted(out.unique().tolist())[:MAX_EXAMPLES]))
    return _result(state, "check_numeric_range", column, findings, rule_used=method,
                   observed_min=float(valid.min()) if len(valid) else None,
                   observed_max=float(valid.max()) if len(valid) else None)


def check_date_values(state: AuditState, column: str, expected_format: str = "%Y-%m-%d",
                      min_date: str | None = None, max_date: str | None = None) -> dict:
    """Check that dates use one consistent format and fall in a sensible
    window. max_date defaults to the reference date (no future dates)."""
    df = state.df
    _require_column(df, column)
    present = _present(df, column)
    parsed = _parse_dates_strict(present, expected_format)
    findings = []

    bad = parsed.isna()
    if bad.any():
        findings.append(state.add_finding(
            "check_date_values", "date_format", column,
            f"{int(bad.sum())} values in '{column}' do not match the format {expected_format}",
            present.index[bad], present[bad].unique()[:MAX_EXAMPLES].tolist()))

    hi = pd.Timestamp(max_date) if max_date else state.reference_date
    lo = pd.Timestamp(min_date) if min_date else None
    ok = parsed.dropna()
    outside = ok[(ok > hi) | ((ok < lo) if lo is not None else False)]
    if len(outside):
        findings.append(state.add_finding(
            "check_date_values", "date_out_of_range", column,
            f"{len(outside)} dates in '{column}' fall outside "
            f"[{lo.date() if lo is not None else 'any'}, {hi.date()}]",
            outside.index, present[outside.index].unique()[:MAX_EXAMPLES].tolist()))
    return _result(state, "check_date_values", column, findings)


def check_date_order(state: AuditState, start_column: str, end_column: str,
                     expected_format: str = "%Y-%m-%d") -> dict:
    """Flag rows where the end date is before the start date
    (e.g. picked up before it arrived). Unparseable dates are skipped."""
    df = state.df
    _require_column(df, start_column)
    _require_column(df, end_column)
    start = _parse_dates_strict(_as_text(df[start_column]), expected_format)
    end = _parse_dates_strict(_as_text(df[end_column]), expected_format)
    mask = start.notna() & end.notna() & (end < start)
    findings = []
    if mask.any():
        examples = [f"{start_column}={df.at[i, start_column]}, {end_column}={df.at[i, end_column]}"
                    for i in df.index[mask][:MAX_EXAMPLES]]
        findings.append(state.add_finding(
            "check_date_order", "date_order", f"{start_column} -> {end_column}",
            f"{int(mask.sum())} rows have '{end_column}' earlier than '{start_column}'",
            df.index[mask], examples))
    return _result(state, "check_date_order", [start_column, end_column], findings)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def category_variant_map(present: pd.Series, similarity: float = 0.8) -> tuple[dict, list]:
    """Map each non-standard spelling to its standard label.
    Returns (variant -> standard, list of standard labels)."""
    counts = present.value_counts()
    threshold = max(3, 0.02 * len(present))
    standard = [v for v, c in counts.items() if c >= threshold]
    variant_map = {}
    # 1) Same letters, different case/spacing/punctuation ('UPS' vs 'ups' vs 'U.P.S.'):
    #    the most common spelling in the group is the standard one.
    groups: dict[str, list[str]] = {}
    for value in counts.index:  # value_counts is sorted most common first
        groups.setdefault(_norm(value), []).append(value)
    for spellings in groups.values():
        for v in spellings[1:]:
            variant_map[v] = spellings[0]
        if spellings[0] not in standard and len(spellings) > 1:
            standard.append(spellings[0])
    # 2) Close misspellings of a common value ('UPSS', 'Amazn').
    for value, count in counts.items():
        if value in standard or value in variant_map:
            continue
        best, best_score = None, 0.0
        for s in standard:
            score = 1.0 if _norm(value) == _norm(s) else \
                difflib.SequenceMatcher(None, _norm(value), _norm(s)).ratio()
            if score > best_score:
                best, best_score = s, score
        if best is not None and best_score >= similarity:
            variant_map[value] = best
    return variant_map, standard


def check_category_variants(state: AuditState, column: str, similarity: float = 0.8) -> dict:
    """Find misspelled or inconsistently formatted category labels
    ('fedex', 'Fed Ex', 'UPSS') by comparing rare values to common ones."""
    df = state.df
    _require_column(df, column)
    present = _present(df, column)
    counts = present.value_counts()
    if len(counts) > 50:
        return _result(state, "check_category_variants", column, [],
                       message="Too many distinct values for a category column; skipped.")
    variant_map, standard = category_variant_map(present, similarity)
    findings = []
    if variant_map:
        rows = present.index[present.isin(list(variant_map))]
        examples = [f"'{v}' looks like '{s}' ({int(counts[v])} rows)" for v, s in variant_map.items()]
        findings.append(state.add_finding(
            "check_category_variants", "category_variant", column,
            f"{len(rows)} rows in '{column}' use a non-standard spelling of a category",
            rows, examples))
    return _result(state, "check_category_variants", column, findings,
                   standard_values=standard)


def check_pattern(state: AuditState, column: str, pattern: str) -> dict:
    """Check values against a format. pattern is a name ('email',
    'us_phone', 'us_zip') or a regular expression."""
    df = state.df
    _require_column(df, column)
    regex = PATTERNS.get(pattern, pattern)
    re.compile(regex)  # raises a clear error if the agent wrote a bad regex
    present = _present(df, column)
    bad = ~present.str.fullmatch(regex)
    findings = []
    if bad.any():
        findings.append(state.add_finding(
            "check_pattern", "invalid_format", column,
            f"{int(bad.sum())} values in '{column}' do not match the {pattern} pattern",
            present.index[bad], present[bad].unique()[:MAX_EXAMPLES].tolist()))
    return _result(state, "check_pattern", column, findings, pattern_used=regex)


# --------------------------------------------------------------------------
# Reporting tools (the agent's judgment is recorded here)
# --------------------------------------------------------------------------

SEVERITIES = ("high", "medium", "low")


def log_issue(state: AuditState, finding_id: str, severity: str,
              why_it_matters: str, suggested_fix: str) -> dict:
    """Add a finding to the final report. The agent supplies judgment
    (severity, impact, fix); the rows come from the stored finding, not
    from the model."""
    if finding_id not in state.findings:
        raise ValueError(f"No finding with id {finding_id}. Run a check first.")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    if any(i["finding_id"] == finding_id for i in state.issues):
        return {"status": "already logged", "finding_id": finding_id}
    f = state.findings[finding_id]
    state.issues.append({
        "issue_id": f"I{len(state.issues) + 1:02d}",
        "finding_id": finding_id,
        "category": f.category,
        "column": f.column,
        "description": f.description,
        "severity": severity,
        "why_it_matters": why_it_matters,
        "suggested_fix": suggested_fix,
        "rows": f.rows,
        "examples": f.examples[:MAX_EXAMPLES],
    })
    return {"status": "logged", "finding_id": finding_id, "rows_affected": len(f.rows)}


def finish(state: AuditState, summary: str) -> dict:
    """End the audit with a short summary for a non-technical reader."""
    state.summary = summary
    state.done = True
    state.stop_reason = "agent finished"
    return {"status": "audit complete", "issues_logged": len(state.issues)}
