"""Test the one-command evaluation (scripts/run_all.py) and the comparison
it prints (dq_agent/compare.py), without an API key or any API calls."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dq_agent.agent import BUILTIN_MODEL, run_agent
from dq_agent.baseline import run_baseline
from dq_agent.compare import build_summary, kg_weight_rows, projection_issues
from dq_agent.report import save_outputs
from dq_agent.tools import load_csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_all  # noqa: E402

SECRET = "sk-ant-test-DO-NOT-PRINT-1234"


def _run(args, env_extra=None, drop_key=True):
    env = {k: v for k, v in os.environ.items() if not (drop_key and k == "ANTHROPIC_API_KEY")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "scripts/run_all.py", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)


def _baseline_run(name, ref=None):
    st = run_baseline(load_csv(ROOT / f"data/{name}_messy.csv"), reference_date=ref)
    return {"issues": st.issues, "model": "", "usage": {}, "tool_log": st.tool_log,
            "stop_reason": st.stop_reason, "data_unchanged": True}


# ---- failing safely ------------------------------------------------------

def test_missing_key_fails_before_any_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, "ROOT", tmp_path)  # so a real .env in the project isn't read
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "out"
    assert run_all.main(["--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY is not set" in err and "--baseline-only" in err
    assert not out.exists()  # nothing ran


def test_placeholder_key_is_rejected():
    assert "placeholder" in run_all.api_key_problem({"ANTHROPIC_API_KEY": "your-key-here"})
    assert run_all.api_key_problem({"ANTHROPIC_API_KEY": " "}) == "ANTHROPIC_API_KEY is not set."
    assert run_all.api_key_problem({"ANTHROPIC_API_KEY": SECRET}) is None


def test_asks_before_spending_and_never_prints_key(tmp_path):
    out = tmp_path / "out"
    proc = _run(["--out", str(out)], {"ANTHROPIC_API_KEY": SECRET})
    assert proc.returncode == 3  # no one to confirm, so it stops
    assert "up to 60 API calls" in proc.stdout
    assert SECRET not in proc.stdout + proc.stderr
    assert not out.exists()


def test_final_runs_ignore_dev_model_override():
    env = {"DQ_AGENT_MODEL": "claude-haiku-5-5"}
    assert run_all.choose_model(final=True, env=env) == BUILTIN_MODEL
    assert run_all.choose_model(final=False, env=env) == "claude-haiku-5-5"
    assert run_all.choose_model(final=False, env={}) == run_all.DEV_MODEL


def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# comment\nDQ_TEST_A=from_file\nDQ_TEST_B='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("DQ_TEST_A", "from_env")
    monkeypatch.delenv("DQ_TEST_B", raising=False)
    run_all.load_dotenv(tmp_path / ".env")
    assert os.environ["DQ_TEST_A"] == "from_env"
    assert os.environ["DQ_TEST_B"] == "quoted"
    monkeypatch.delenv("DQ_TEST_B")


# ---- the comparison numbers ----------------------------------------------

def test_reference_numbers_match_readme():
    runs = {("mpg", "baseline"): _baseline_run("mpg"),
            ("package_log", "baseline"): _baseline_run("package_log", "2026-04-30")}
    summary = build_summary(runs, ROOT / "data")
    assert summary["drift"] == []  # the README's script, raw, projection and floor numbers reproduce
    rows = {r["auditor"].split(":")[0]: r for r in summary["rows"] if r["dataset"] == "mpg"}
    assert rows["Baseline script"]["kg_caught"] == 0
    assert rows["Projection"]["kg_caught"] == 4 and rows["Projection"]["kind"] == "projection"


def test_kg_rows_include_the_hard_cases():
    messy = load_csv(ROOT / "data/mpg_messy.csv")
    kg = kg_weight_rows(messy, pd.read_csv(ROOT / "data/mpg_ground_truth.csv"))
    values = sorted(int(messy.at[r, "weight"]) for r in kg)
    assert len(kg) == 6 and 1601 in values and 1794 in values


def test_checks_report_an_agent_that_does_worse():
    runs = {("mpg", "baseline"): _baseline_run("mpg"),
            ("mpg", "agent"): {"issues": [], "model": "claude-haiku-5-5", "usage": {}, "tool_log": [],
                               "stop_reason": "step limit (30) reached", "data_unchanged": True}}
    checks = " ".join(build_summary(runs, ROOT / "data")["checks"])
    assert "agent is worse" in checks and "OUTSIDE" in checks
    assert "did not finish normally" in checks


def test_agent_records_model_and_token_usage(tmp_path):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_creation_input_tokens=None,
                            cache_read_input_tokens=0)
    calls = iter([[SimpleNamespace(type="tool_use", id="1", name="profile_dataset", input={})],
                  [SimpleNamespace(type="tool_use", id="2", name="finish", input={"summary": "ok"})]])

    class Client:
        messages = SimpleNamespace(create=lambda **k: SimpleNamespace(content=next(calls), usage=usage))

    state = run_agent(load_csv(ROOT / "data/mpg_messy.csv"), client=Client(), model="m-test", verbose=False)
    assert state.usage == {"api_calls": 2, "input_tokens": 200, "output_tokens": 40,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    run = json.loads((save_outputs(state, tmp_path, "mpg_messy", "agent") / "run.json").read_text())
    assert run["model"] == "m-test" and run["usage"]["api_calls"] == 2


# ---- the whole pipeline, with the agent steps faked -----------------------

def test_full_run_with_faked_agent_steps(tmp_path, monkeypatch):
    """Every real step runs; only the two agent subprocesses are replaced by a
    stand-in that applies the weight rule, so no API call is made."""
    real_step = run_all.run_step
    seen = {}

    def fake_step(name, args, log_dir, echo=False):
        if not name.endswith("_agent"):
            return real_step(name, args, log_dir, echo)
        seen[name] = args
        dataset = name.removesuffix("_agent")
        messy = load_csv(ROOT / f"data/{dataset}_messy.csv")
        state = run_baseline(messy, reference_date="2026-04-30" if dataset == "package_log" else None)
        if dataset == "mpg":
            state.issues = projection_issues(messy, state.issues)
        state.model = args[args.index("--model") + 1]
        save_outputs(state, args[args.index("--out") + 1], f"{dataset}_messy", "agent")

    out = tmp_path / "out"
    (out / "mpg_agent").mkdir(parents=True)
    (out / "mpg_agent" / "stale.txt").write_text("old run")
    monkeypatch.setattr(run_all, "run_step", fake_step)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.delenv("DQ_AGENT_MODEL", raising=False)

    assert run_all.main(["--yes", "--out", str(out)]) == 0
    assert all("claude-haiku-5-5" in a for a in seen.values())  # dev runs use the cheap model
    assert not (out / "mpg_agent" / "stale.txt").exists()      # old agent output cleared first
    assert (out / "regression_comparison.png").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    agent = next(r for r in summary["rows"] if r["dataset"] == "mpg" and r["auditor"] == "AI agent")
    assert agent["model"] == "claude-haiku-5-5"
    assert round(agent["slope_error_pct"], 1) == 2.5 and agent["kg_caught"] == 4
    assert SECRET not in (out / "summary.md").read_text(encoding="utf-8")
