"""Does cleaning change the answer? Fit mpg ~ weight and compare the
slope with the TRUE slope from the clean data.

Shared by scripts/regression_demo.py (the chart) and dq_agent/compare.py
(the summary table) so both always report the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fixes import apply_fixes, numeric_pairs

X, Y = "weight", "mpg"


def fit(pairs: pd.DataFrame) -> dict:
    slope, intercept = np.polyfit(pairs[X], pairs[Y], 1)
    r = np.corrcoef(pairs[X], pairs[Y])[0, 1]
    return {"slope": slope, "intercept": intercept, "r2": r * r, "n": len(pairs)}


def slope_error_pct(slope: float, true_slope: float) -> float:
    return abs(slope - true_slope) / abs(true_slope) * 100


def cleaned_fit(messy: pd.DataFrame, issues: list[dict], true_slope: float) -> dict:
    """Apply every fix for `issues` to a copy of the data, then fit."""
    fixed, log = apply_fixes(messy, issues)
    f = fit(numeric_pairs(fixed, X, Y))
    f["slope_error_pct"] = slope_error_pct(f["slope"], true_slope)
    f["fixes"] = dict(log)
    return f


def issues_from_answer_key(truth: pd.DataFrame) -> list[dict]:
    """A perfect auditor: one issue per (category, column) in the answer key.
    Duplicate copies must share one issue so apply_fixes can keep the first."""
    return [{"issue_id": f"K{n:02d}", "category": cat, "column": col or None,
             "rows": sorted(int(r) for r in group["row_index"])}
            for n, ((cat, col), group) in enumerate(truth.fillna("").groupby(["category", "column"]), 1)]
