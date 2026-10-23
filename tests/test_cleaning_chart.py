"""Test the script-vs-agent chart (scripts/cleaning_chart.py): it shows the
agent as pending until a real run exists and never invents agent numbers."""

import sys
from pathlib import Path

from dq_agent.baseline import run_baseline
from dq_agent.report import save_outputs
from dq_agent.tools import load_csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cleaning_chart  # noqa: E402

DATA = ROOT / "data"


def test_agent_pending_without_runs_and_script_rerun(tmp_path):
    results = cleaning_chart.collect(tmp_path, DATA)  # empty folder: no run.json at all
    assert results[("mpg", "agent")] is None and results[("package_log", "agent")] is None
    assert results["model"] is None
    mpg, pk = results[("mpg", "script")], results[("package_log", "script")]
    assert (mpg["caught"], mpg["planted"], mpg["false_alarms"]) == (66, 72, 0)
    assert (pk["caught"], pk["planted"], pk["false_alarms"]) == (227, 240, 197)


def test_agent_run_is_read_and_can_be_hidden(tmp_path):
    state = run_baseline(load_csv(DATA / "mpg_messy.csv"))
    state.issues = state.issues[:1]  # a stand-in "agent" that logged one issue
    state.model = "m-test"
    save_outputs(state, tmp_path / "mpg_agent", "mpg_messy", "agent")

    results = cleaning_chart.collect(tmp_path, DATA)
    assert results[("mpg", "agent")]["caught"] == len(state.issues[0]["rows"])
    assert results[("package_log", "agent")] is None  # only the runs that exist
    assert results["model"] == "m-test"
    assert cleaning_chart.collect(tmp_path, DATA, include_agent=False)[("mpg", "agent")] is None


def test_chart_file_is_written(tmp_path):
    out = tmp_path / "chart.png"
    cleaning_chart.main(["--runs-dir", str(tmp_path), "--out", str(out)])
    assert out.exists() and out.stat().st_size > 10_000
