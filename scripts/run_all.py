"""One command for the whole evaluation: baseline + agent on both datasets,
scorecards, the regression chart with the agent panel, the script-vs-agent
cleaning chart, and a summary table.

    python scripts/run_all.py                  # development run (cheap model, outputs/dev/)
    python scripts/run_all.py --final          # the run we report (default model, outputs/final/)
    python scripts/run_all.py --baseline-only  # no API key, no API calls

The agent steps need ANTHROPIC_API_KEY, either as an environment variable
or in a .env file in the project folder (never committed; see .env.example).
The script never prints the key. Before any API call it says how many calls
it may make and asks you to confirm (or pass --yes).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dq_agent.agent import BUILTIN_MODEL  # noqa: E402
from dq_agent.compare import DATASETS, build_summary, format_summary  # noqa: E402

DEV_MODEL = "claude-haiku-5-5"
MAX_STEPS = 30
PLACEHOLDER_KEYS = {"your-key-here", "your-api-key", "sk-ant-..."}

KEY_HELP = """\
{problem}

The agent steps need an Anthropic API key (console.anthropic.com). Either:
  PowerShell, this window only:   $env:ANTHROPIC_API_KEY = "your-key"
  PowerShell, permanently:        setx ANTHROPIC_API_KEY "your-key"   (then open a new window)
  macOS/Linux/Git Bash:           export ANTHROPIC_API_KEY=your-key
  or a .env file in the project folder:  ANTHROPIC_API_KEY=your-key   (.env is git-ignored)

To check everything else without a key:  python scripts/run_all.py --baseline-only"""


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment without
    overriding variables that are already set. Values are never printed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key_problem(env: dict) -> str | None:
    key = env.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "ANTHROPIC_API_KEY is not set."
    if key in PLACEHOLDER_KEYS or key.lower().startswith("your"):
        return "ANTHROPIC_API_KEY still has the placeholder value from .env.example."
    return None


def choose_model(final: bool, env: dict) -> str:
    """Final runs always use the built-in default model so reported numbers
    are comparable; development runs use DQ_AGENT_MODEL or the cheap model."""
    return BUILTIN_MODEL if final else env.get("DQ_AGENT_MODEL") or DEV_MODEL


def confirm_spend(model: str, assume_yes: bool) -> bool:
    calls = len(DATASETS) * MAX_STEPS
    print(f"About to run the AI agent on {len(DATASETS)} datasets with {model}.\n"
          f"That is up to {calls} API calls ({MAX_STEPS}-step limit per dataset), billed to your key.")
    if assume_yes:
        print("Confirmed with --yes.\n")
        return True
    if not sys.stdin.isatty():
        print("No terminal to ask for confirmation. Re-run with --yes to go ahead.")
        return False
    try:
        return input("Continue? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:  # no answer possible (e.g. input redirected): treat as no
        print()
        return False


def run_step(name: str, args: list[str], log_dir: Path, echo: bool = False) -> None:
    """Run one existing script as a subprocess, saving its output to a log.
    Stops the whole evaluation (exit 1) if the step fails."""
    log = log_dir / f"{name}.log"
    print(f"-> {name}", flush=True)
    start = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            f.write(line)
            if echo:
                print("   " + line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0:
        print(f"\nStep '{name}' failed (exit {proc.returncode}). Full output: {log}")
        print("".join(log.read_text(encoding="utf-8").splitlines(keepends=True)[-15:]))
        sys.exit(1)
    print(f"   done in {time.time() - start:.0f}s (log: {log.relative_to(ROOT) if log.is_relative_to(ROOT) else log})")


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true",
                    help=f"Use the default model ({BUILTIN_MODEL}) for the numbers we report.")
    ap.add_argument("--baseline-only", action="store_true", help="Skip the agent (no key, no API calls).")
    ap.add_argument("--yes", action="store_true", help="Don't ask before making API calls.")
    ap.add_argument("--out", default=None, help="Output folder (default outputs/final or outputs/dev).")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    except AttributeError:
        pass

    load_dotenv(ROOT / ".env")
    use_agent = not args.baseline_only
    model = choose_model(args.final, os.environ) if use_agent else ""

    # Fail before doing any work if the agent can't run.
    if use_agent:
        problem = api_key_problem(os.environ)
        if problem:
            print(KEY_HELP.format(problem=problem), file=sys.stderr)
            return 2
        if not confirm_spend(model, args.yes):
            print("Stopped before any API call.")
            return 3

    out = Path(args.out) if args.out else ROOT / "outputs" / ("final" if args.final else "dev")
    out = out if out.is_absolute() else ROOT / out
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not all((ROOT / "data" / f"{d}_{kind}.csv").exists() for d in DATASETS
               for kind in ("messy", "clean", "ground_truth")):
        run_step("make_dataset", ["scripts/make_dataset.py"], log_dir)

    for d in DATASETS:
        run_step(f"{d}_baseline", ["scripts/run_audit.py", f"data/{d}_messy.csv", "--mode", "baseline",
                                   "--out", str(out / f"{d}_baseline")], log_dir)
    if use_agent:
        for d in DATASETS:
            shutil.rmtree(out / f"{d}_agent", ignore_errors=True)  # never score a stale run
            run_step(f"{d}_agent", ["scripts/run_audit.py", f"data/{d}_messy.csv", "--mode", "agent",
                                    "--model", model, "--max-steps", str(MAX_STEPS),
                                    "--out", str(out / f"{d}_agent")], log_dir, echo=True)

    modes = ("baseline", "agent") if use_agent else ("baseline",)
    run_paths = [str(out / f"{d}_{m}" / "run.json") for d in DATASETS for m in modes]
    run_step("evaluate", ["scripts/evaluate.py", *run_paths], log_dir)
    chart = out / "regression_comparison.png"
    run_step("regression_chart", ["scripts/regression_demo.py",
                                  "--baseline-run", str(out / "mpg_baseline" / "run.json"),
                                  "--agent-run", str(out / "mpg_agent" / "run.json"),
                                  *([] if use_agent else ["--no-agent"]), "--out", str(chart)], log_dir)
    cleaning = out / "cleaning_comparison.png"
    run_step("cleaning_chart", ["scripts/cleaning_chart.py", "--runs-dir", str(out),
                                *([] if use_agent else ["--no-agent"]), "--out", str(cleaning)], log_dir)

    runs = {(d, m): json.loads((out / f"{d}_{m}" / "run.json").read_text(encoding="utf-8"))
            for d in DATASETS for m in modes}
    summary = build_summary(runs, ROOT / "data")
    summary.update(run_at=datetime.now().isoformat(timespec="seconds"), git_commit=git_commit(),
                   model=model or None, kind="final" if args.final else "development",
                   agent_included=use_agent)
    header = (f"# Evaluation summary ({summary['kind']} run{', model ' + model if model else ', baseline only'})\n\n"
              f"Run {summary['run_at']} at commit {summary['git_commit']}. "
              f"Italic rows are reference points, not auditor results.\n")
    text = format_summary(summary, header)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 78 + "\n")
    print(text)
    print(f"Saved: {out / 'summary.md'}\n       {chart}\n       {cleaning}\n"
          f"       scorecards in {log_dir / 'evaluate.log'}")
    if not use_agent:
        print("\nAgent skipped (--baseline-only). Add ANTHROPIC_API_KEY and run without the flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
