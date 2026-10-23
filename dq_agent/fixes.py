"""Apply the fixes for logged issues to produce a cleaned copy of the data.

Human-in-the-loop: pass `approved` (a set of issue IDs) to apply only the
fixes a person signed off on in issues.csv. The original DataFrame is
never modified; a new one is returned.

Fix policy by issue category:
  duplicate_row     keep the first copy, drop the rest
  category_variant  rename to the standard label
  date_format       convert to YYYY-MM-DD (drop the row if it can't be read)
  invalid_format    numbers typed with units/commas ('3,504 lbs') -> 3504;
                    anything else is dropped
  everything else   (missing, out_of_range, bad dates, duplicate IDs)
                    the row is excluded, since the true value is unknown
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .tools import _is_missing, _present, category_variant_map


def _numeric_column(df: pd.DataFrame, column: str) -> bool:
    present = _present(df, column)
    return len(present) > 0 and pd.to_numeric(present, errors="coerce").notna().mean() >= 0.8


def apply_fixes(df: pd.DataFrame, issues: list[dict],
                approved: set[str] | None = None) -> tuple[pd.DataFrame, Counter]:
    out = df.copy()
    exclude: set[int] = set()
    log: Counter = Counter()

    for issue in issues:
        if approved is not None and issue["issue_id"] not in approved:
            log["not approved (left as is)"] += len(issue["rows"])
            continue
        cat, col, rows = issue["category"], issue["column"], [r for r in issue["rows"] if r in out.index]

        if cat == "duplicate_row":
            group = out.loc[rows]
            extra = group[group.duplicated(keep="first")].index
            exclude.update(int(r) for r in extra)
            log["duplicate copies removed"] += len(extra)

        elif cat == "category_variant" and col in out.columns:
            mapping, _ = category_variant_map(_present(out, col))
            for r in rows:
                if out.at[r, col] in mapping:
                    out.at[r, col] = mapping[out.at[r, col]]
                    log["labels standardized"] += 1

        elif cat == "date_format" and col in out.columns:
            for r in rows:
                d = pd.to_datetime(out.at[r, col], format="mixed", errors="coerce")
                if pd.isna(d):
                    exclude.add(r)
                    log["rows excluded"] += 1
                else:
                    out.at[r, col] = d.strftime("%Y-%m-%d")
                    log["dates reformatted"] += 1

        elif cat == "invalid_format" and col in out.columns and _numeric_column(df, col):
            for r in rows:
                cleaned = pd.to_numeric(pd.Series([out.at[r, col]]).str.replace(r"[^0-9.\-]", "", regex=True),
                                        errors="coerce").iloc[0]
                if pd.isna(cleaned):
                    exclude.add(r)
                    log["rows excluded"] += 1
                else:
                    out.at[r, col] = f"{cleaned:g}"
                    log["numbers recovered from text"] += 1

        else:  # value is unknown or impossible: leave it out of the analysis
            new = set(rows) - exclude
            exclude.update(new)
            log["rows excluded"] += len(new)

    return out.drop(index=sorted(exclude)), log


def numeric_pairs(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows where both x and y read as numbers (how most tools would treat raw data)."""
    xs = pd.to_numeric(df[x].where(~_is_missing(df[x])), errors="coerce")
    ys = pd.to_numeric(df[y].where(~_is_missing(df[y])), errors="coerce")
    pairs = pd.DataFrame({x: xs, y: ys}).dropna()
    return pairs
