"""Does the agent clean smarter? Compare the rule-based script with the AI
agent on recall, precision and false alarms, for both datasets.

    python scripts/cleaning_chart.py                          # reads outputs/final/
    python scripts/cleaning_chart.py --runs-dir outputs/dev
    python scripts/cleaning_chart.py --no-agent --out docs/cleaning_comparison_baseline.png

Until an agent run exists, its bars show as "pending" (nothing is drawn for
results we don't have). If the script's run.json is missing, the script is
re-run here; it is deterministic and needs no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dq_agent import dataset  # noqa: E402
from dq_agent.baseline import run_baseline  # noqa: E402
from dq_agent.compare import DATASETS, totals  # noqa: E402
from dq_agent.tools import load_csv  # noqa: E402

# Same surface, ink and grid as the regression chart; series colors are
# categorical slots 1 and 2 of the reference palette (validated together).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a1", "#e6e5e1"
SERIES = {"script": "#2a78d6", "agent": "#eb6834"}
LABELS = {"script": "Rule-based script", "agent": "AI agent"}
NAMES = {"mpg": "Auto MPG", "package_log": "Package log"}
REFERENCE_DATES = {"package_log": dataset.REFERENCE_DATE}
METRICS = [("recall", "Recall", "share of planted errors caught · higher is better"),
           ("precision", "Precision", "share of flags that were real errors · higher is better"),
           ("false_alarms", "False alarms", "flags that weren't planted errors · lower is better")]


def collect(runs_dir: Path, data_dir: Path, include_agent: bool = True) -> dict:
    """{(dataset, 'script'|'agent'): totals dict, or None when pending}, plus the agent model."""
    out: dict = {"model": None}
    for d in DATASETS:
        truth = pd.read_csv(data_dir / f"{d}_ground_truth.csv")
        base = runs_dir / f"{d}_baseline" / "run.json"
        if base.exists():
            issues = json.loads(base.read_text(encoding="utf-8"))["issues"]
        else:
            issues = run_baseline(load_csv(data_dir / f"{d}_messy.csv"), REFERENCE_DATES.get(d)).issues
        out[(d, "script")] = totals(issues, truth)

        agent = runs_dir / f"{d}_agent" / "run.json"
        if include_agent and agent.exists():
            run = json.loads(agent.read_text(encoding="utf-8"))
            out[(d, "agent")] = totals(run["issues"], truth)
            out["model"] = out["model"] or run.get("model")
        else:
            out[(d, "agent")] = None
    return out


def draw(results: dict, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(13.5, 6.2), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    fa_max = max([results[k]["false_alarms"] for k in results if k != "model" and results[k]] + [10]) * 1.18
    order = ["script", "agent"]  # top to bottom

    for i, d in enumerate(DATASETS):
        for j, (key, title, subtitle) in enumerate(METRICS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            is_pct = key != "false_alarms"
            xmax = 1.42 if is_pct else fa_max  # room for the value label past a full bar
            for row, who in enumerate(order):
                y = len(order) - 1 - row
                r = results[(d, who)]
                if r is None:
                    ax.text(xmax * 0.01, y, "pending", va="center", fontsize=9.5, color=INK_2, style="italic")
                    continue
                v = r[key]
                ax.barh(y, v, height=0.52, color=SERIES[who], zorder=2)
                text = f"{v:.0%}" if is_pct else f"{v:,}"
                if key == "recall":
                    text += f"  ({r['caught']}/{r['planted']})"
                ax.text(v + xmax * 0.015, y, text, va="center", fontsize=9.5, color=INK)
            ax.set_xlim(0, xmax)
            ax.set_ylim(-0.6, len(order) - 0.4)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([LABELS[w] for w in reversed(order)] if j == 0 else [])
            ax.tick_params(axis="y", length=0)
            if is_pct:
                ax.set_xticks([0, 0.5, 1.0])
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
            else:
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if i == 0:  # column headers, above the first dataset's label
                ax.text(0, 1.52, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold")
                ax.text(0, 1.37, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)
            if j == 0:
                planted = results[(d, "script")]["planted"]
                ax.text(-0.42, 1.1, f"{NAMES[d]}  ·  {planted} planted errors",
                        transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)

    pending = any(results[(d, "agent")] is None for d in DATASETS)
    note = ("AI agent: pending. Its bars appear after a real run (python scripts/run_all.py --final)."
            if pending else f"AI agent model: {results['model']}. One run per dataset.")
    fig.text(0.08, -0.02, note + "  Scored against the answer key of planted errors.",
             fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(color=SERIES[w], label=LABELS[w]) for w in order], loc="upper right",
               bbox_to_anchor=(0.905, 1.125), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Does the agent clean smarter?  Script vs. AI agent on planted errors",
                 x=0.08, ha="left", y=1.11, fontsize=13.5, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/final")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-agent", action="store_true", help="Show the agent as pending even if a run exists.")
    ap.add_argument("--out", default=None, help="Default: <runs-dir>/cleaning_comparison.png")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    results = collect(runs_dir, Path(args.data_dir), include_agent=not args.no_agent)
    out = Path(args.out) if args.out else runs_dir / "cleaning_comparison.png"
    draw(results, out)
    for d in DATASETS:
        for who in ("script", "agent"):
            r = results[(d, who)]
            print(f"{d:12} {LABELS[who]:18} " + ("pending" if r is None else
                  f"recall {r['recall']:.0%}  precision {r['precision']:.0%}  false alarms {r['false_alarms']}"))
    print(f"Chart: {out}")


if __name__ == "__main__":
    main()
