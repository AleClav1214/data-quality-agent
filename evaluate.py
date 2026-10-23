"""Grade an audit against the answer key.

A prediction is a (row, error category) pair the auditor logged.
  precision = of everything flagged, how much was a real planted error
  recall    = of all planted errors, how many were caught
  F1        = balance of the two
"""

from __future__ import annotations

import pandas as pd


def score(issues: list[dict], truth: pd.DataFrame) -> pd.DataFrame:
    predicted = {(int(r), i["category"]) for i in issues for r in i["rows"]}
    actual = {(int(r), c) for r, c in zip(truth["row_index"], truth["category"])}
    categories = sorted({c for _, c in predicted | actual})

    rows = []
    for cat in categories + ["ALL"]:
        p = {x for x in predicted if cat == "ALL" or x[1] == cat}
        a = {x for x in actual if cat == "ALL" or x[1] == cat}
        tp, fp, fn = len(p & a), len(p - a), len(a - p)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"category": cat, "planted": len(a), "flagged": len(p), "caught": tp,
                     "false_alarms": fp, "missed": fn,
                     "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})
    return pd.DataFrame(rows)


def format_scorecard(df: pd.DataFrame, label: str = "") -> str:
    out = [f"Scorecard{': ' + label if label else ''}", ""]
    out.append(df.to_string(index=False))
    total = df[df["category"] == "ALL"].iloc[0]
    out += ["", f"Caught {total['caught']} of {total['planted']} planted errors "
                f"(recall {total['recall']:.0%}); {total['false_alarms']} false alarms "
                f"(precision {total['precision']:.0%})."]
    return "\n".join(out)
