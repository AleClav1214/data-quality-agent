import pandas as pd

from dq_agent.baseline import run_baseline
from dq_agent.dataset import INJECTIONS, REFERENCE_DATE, generate
from dq_agent.evaluate import score


def test_generation_is_reproducible():
    a, ta, _ = generate(1000, seed=5)
    b, tb, _ = generate(1000, seed=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


def test_answer_key_matches_injections():
    messy, truth, clean = generate(2000, seed=3)
    counts = truth["category"].value_counts()
    assert counts["duplicate_row"] == 2 * INJECTIONS["duplicate_row"]
    assert counts["duplicate_key"] == 2 * INJECTIONS["duplicate_key"]
    assert counts["missing"] == INJECTIONS["missing"]
    assert len(messy) == len(clean) + INJECTIONS["duplicate_row"] + INJECTIONS["duplicate_key"]
    assert truth["row_index"].max() < len(messy)


def test_baseline_has_high_recall_but_false_alarms():
    messy, truth, _ = generate(3000, seed=4)
    state = run_baseline(messy, reference_date=REFERENCE_DATE)
    card = score(state.issues, truth).set_index("category")
    assert card.loc["ALL", "recall"] > 0.85
    assert card.loc["missing", "false_alarms"] > 50   # flags expected blank pickup dates
    assert card.loc["out_of_range", "recall"] < 1.0   # IQR rule misses negative weights
    assert state.data_unchanged()
