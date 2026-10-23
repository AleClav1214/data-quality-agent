import numpy as np
import pandas as pd

from dq_agent import mpg_data
from dq_agent.baseline import run_baseline
from dq_agent.evaluate import score
from dq_agent.fixes import apply_fixes, numeric_pairs
from dq_agent.tools import AuditState, check_numeric_range, log_issue


def _slope(df):
    p = numeric_pairs(df, "weight", "mpg")
    return np.polyfit(p["weight"], p["mpg"], 1)[0]


def test_fixes_do_not_touch_original():
    df = pd.DataFrame({"w": ["1", "2,000", "3", "4", "5"], "c": ["usa", "usa", "USA", "usa", "japan"]})
    issues = [{"issue_id": "I01", "category": "invalid_format", "column": "w", "rows": [1]},
              {"issue_id": "I02", "category": "category_variant", "column": "c", "rows": [2]}]
    before = df.copy()
    fixed, log = apply_fixes(df, issues)
    pd.testing.assert_frame_equal(df, before)
    assert fixed.at[1, "w"] == "2000" and fixed.at[2, "c"] == "usa"
    assert log["numbers recovered from text"] == 1 and log["labels standardized"] == 1


def test_only_approved_fixes_are_applied():
    df = pd.DataFrame({"v": ["1", "999", "3"]})
    issues = [{"issue_id": "I01", "category": "out_of_range", "column": "v", "rows": [1]}]
    fixed, log = apply_fixes(df, issues, approved=set())
    assert len(fixed) == 3 and log["not approved (left as is)"] == 1
    fixed, _ = apply_fixes(df, issues, approved={"I01"})
    assert len(fixed) == 2


def test_duplicate_fix_keeps_one_copy():
    df = pd.DataFrame({"a": ["x", "x", "y"]})
    fixed, _ = apply_fixes(df, [{"issue_id": "I01", "category": "duplicate_row", "column": None,
                                 "rows": [0, 1]}])
    assert fixed["a"].tolist() == ["x", "y"]


def test_mpg_answer_key_and_true_slope():
    messy, truth, clean = mpg_data.generate(seed=42)
    assert len(clean) == 392
    assert len(messy) == len(clean) + mpg_data.INJECTIONS["duplicate_row"]
    assert abs(_slope(clean) * 1000 - (-7.65)) < 0.01


def test_cleaning_moves_slope_toward_truth():
    messy, truth, clean = mpg_data.generate(seed=42)
    true_slope = _slope(clean)
    raw_err = abs(_slope(messy) - true_slope)

    state = run_baseline(messy)
    script_fixed, _ = apply_fixes(messy, state.issues)
    script_err = abs(_slope(script_fixed) - true_slope)
    assert script_err < raw_err / 5

    # A business rule on weight (what an agent with domain sense would add)
    # catches kilogram entries that the statistical rule misses.
    rule = AuditState(df=messy)
    res = check_numeric_range(rule, "weight", min_value=1500, max_value=6000)
    for f in res["findings"]:
        log_issue(rule, f["finding_id"], "high", "x", "y")
    extra = [dict(i, issue_id="R" + i["issue_id"]) for i in rule.issues]
    rule_fixed, _ = apply_fixes(messy, state.issues + extra)
    assert abs(_slope(rule_fixed) - true_slope) < script_err

    card = score(state.issues, truth).set_index("category")
    assert card.loc["out_of_range", "missed"] == 6  # the kilogram entries
