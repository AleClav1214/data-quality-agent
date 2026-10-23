import pandas as pd
import pytest

from dq_agent import tools
from dq_agent.registry import TOOL_SPECS, dispatch
from dq_agent.tools import AuditState


@pytest.fixture
def state():
    df = pd.DataFrame({
        "order_id": ["A1", "A2", "A3", "A3", "A5", "A6", "A6"],
        "email": ["a@x.com", "bad-email", "", "c@x.com", "N/A", "d@x.com", "d@x.com"],
        "carrier": ["UPS", "UPS", "ups", "FedEx", "FedEx", "FedEx", "FedEx"],
        "start": ["2026-01-01", "2026-01-02", "01/03/2026", "2026-01-04", "2027-01-05", "2026-01-06", "2026-01-06"],
        "end": ["2026-01-02", "2025-12-30", "2026-01-04", "2026-01-05", "2027-01-06", "2026-01-07", "2026-01-07"],
        "weight": ["1.5", "-2", "3.0", "abc", "", "4.0", "4.0"],
    })
    return AuditState(df=df, reference_date=pd.Timestamp("2026-06-01"))


def rows(result, category):
    return sorted(r for f in result["findings"] if f["category"] == category
                  for r in [f["finding_id"]])


def finding_rows(state, result, category):
    ids = [f["finding_id"] for f in result["findings"] if f["category"] == category]
    return sorted(r for i in ids for r in state.findings[i].rows)


def test_missing_counts_placeholders(state):
    res = tools.check_missing(state, columns=["email", "weight"])
    assert finding_rows(state, res, "missing") == [2, 4, 4]  # email rows 2 and 4, weight row 4


def test_missing_group_by_breakdown(state):
    res = tools.check_missing(state, columns=["email"], group_by="carrier")
    assert any("carrier=" in e for e in res["findings"][0]["examples"])


def test_exact_duplicates(state):
    res = tools.check_duplicates(state)
    assert finding_rows(state, res, "duplicate_row") == [5, 6]


def test_duplicate_keys_ignores_exact_copies(state):
    res = tools.check_duplicate_keys(state, "order_id")
    assert finding_rows(state, res, "duplicate_key") == [2, 3]  # A3 differs; A6 rows are identical


def test_numeric_range_business_rule(state):
    res = tools.check_numeric_range(state, "weight", min_value=0.1, max_value=150)
    assert finding_rows(state, res, "out_of_range") == [1]
    assert finding_rows(state, res, "invalid_format") == [3]


def test_date_values_format_and_future(state):
    res = tools.check_date_values(state, "start")
    assert finding_rows(state, res, "date_format") == [2]
    assert finding_rows(state, res, "date_out_of_range") == [4]


def test_date_order(state):
    res = tools.check_date_order(state, "start", "end")
    assert finding_rows(state, res, "date_order") == [1]


def test_category_variants(state):
    res = tools.check_category_variants(state, "carrier")
    assert finding_rows(state, res, "category_variant") == [2]


def test_pattern_email(state):
    res = tools.check_pattern(state, "email", "email")
    assert finding_rows(state, res, "invalid_format") == [1]  # blanks are left to check_missing


def test_tools_never_modify_data(state):
    before = state.df.copy()
    for spec in TOOL_SPECS:
        if spec["name"] in ("log_issue", "finish"):
            continue
        args = {"column": "weight"} if "column" in spec["input_schema"].get("required", []) else {}
        if spec["name"] == "check_pattern":
            args = {"column": "email", "pattern": "email"}
        if spec["name"] == "check_date_order":
            args = {"start_column": "start", "end_column": "end"}
        dispatch(state, spec["name"], args)
    pd.testing.assert_frame_equal(state.df, before)
    assert state.data_unchanged()


def test_dispatch_returns_errors_instead_of_crashing(state):
    res = dispatch(state, "check_missing", {"columns": ["not_a_column"]})
    assert "error" in res and "Unknown column" in res["error"]
    assert state.tool_log[-1]["error"]


def test_log_issue_rows_come_from_code_not_model(state):
    res = tools.check_date_order(state, "start", "end")
    fid = res["findings"][0]["finding_id"]
    tools.log_issue(state, fid, "high", "why", "fix")
    assert state.issues[0]["rows"] == [1]
    with pytest.raises(ValueError):
        tools.log_issue(state, "F999", "high", "why", "fix")
