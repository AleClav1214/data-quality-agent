"""Grade one or more audit runs against the answer key.

    python scripts/evaluate.py outputs/mpg_baseline/run.json outputs/mpg_agent/run.json

The answer key is found from the audited file's name
(mpg_messy -> data/mpg_ground_truth.csv) unless --truth is given.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dq_agent.evaluate import format_scorecard, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run.json files from run_audit.py")
    ap.add_argument("--truth", default=None)
    args = ap.parse_args()

    summary = []
    for path in args.runs:
        run = json.loads(Path(path).read_text())
        truth_path = args.truth or f"data/{run['title'].removesuffix('_messy')}_ground_truth.csv"
        truth = pd.read_csv(truth_path)
        card = score(run["issues"], truth)
        print(format_scorecard(card, f"{run['mode']} ({path})"))
        print("\n" + "-" * 80 + "\n")
        total = card[card["category"] == "ALL"].iloc[0].to_dict()
        summary.append({"run": run["mode"], **{k: total[k] for k in
                        ["caught", "planted", "false_alarms", "precision", "recall", "f1"]}})
    if len(summary) > 1:
        print("Side by side:\n")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
