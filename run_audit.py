"""Audit a CSV file.

    # AI agent (needs ANTHROPIC_API_KEY)
    python scripts/run_audit.py data/mpg_messy.csv --mode agent

    # Rule-based baseline (no API key needed)
    python scripts/run_audit.py data/mpg_messy.csv --mode baseline

Outputs go to outputs/<dataset>_<mode>/: report.md, issues.csv, flagged_rows.csv, run.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dq_agent import dataset, mpg_data  # noqa: E402
from dq_agent.baseline import run_baseline  # noqa: E402
from dq_agent.report import save_outputs  # noqa: E402
from dq_agent.tools import load_csv  # noqa: E402

# Built-in demo files get a description and audit date automatically.
DEMOS = {
    "package_log_messy.csv": (dataset.DESCRIPTION, dataset.REFERENCE_DATE),
    "mpg_messy.csv": (mpg_data.DESCRIPTION, None),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--mode", choices=["agent", "baseline"], default="agent")
    ap.add_argument("--description", default=None,
                    help="What the data represents (helps the agent use business sense).")
    ap.add_argument("--reference-date", default=None, help="'Today' for the audit, YYYY-MM-DD.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_csv(args.csv)
    demo_description, demo_date = DEMOS.get(Path(args.csv).name, ("", None))
    description = args.description or demo_description
    ref = args.reference_date or demo_date

    if args.mode == "agent":
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set. Get a key at console.anthropic.com, then:\n"
                     "  export ANTHROPIC_API_KEY=your-key-here\n"
                     "Or run without AI: --mode baseline")
        from dq_agent.agent import DEFAULT_MODEL, run_agent

        state = run_agent(df, dataset_description=description, model=args.model or DEFAULT_MODEL,
                          max_steps=args.max_steps, reference_date=ref)
    else:
        state = run_baseline(df, reference_date=ref)

    name = Path(args.csv).stem.removesuffix("_messy")
    out = save_outputs(state, args.out or f"outputs/{name}_{args.mode}", Path(args.csv).stem, args.mode)
    print(f"\n{len(state.issues)} issues logged. Data unchanged: {state.data_unchanged()}.")
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
