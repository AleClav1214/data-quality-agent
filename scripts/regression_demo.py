"""Show how data quality changes an analysis: fit mpg ~ weight on
  1. the raw messy data,
  2. the data cleaned with the baseline script's findings,
  3. the data cleaned with the AI agent's findings (if you've run it),
and compare each fitted line with the TRUE line from the clean data.

    python scripts/regression_demo.py
    python scripts/regression_demo.py --agent-run outputs/mpg_agent/run.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dq_agent.fixes import apply_fixes, numeric_pairs  # noqa: E402
from dq_agent.regression import X, Y, fit, slope_error_pct  # noqa: E402
from dq_agent.tools import load_csv  # noqa: E402

# Chart colors (reference palette from the dataviz guidelines)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a1", "#e6e5e1"
FIT = "#2a78d6"
XLIM, YLIM = (1400, 5400), (0, 50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messy", default="data/mpg_messy.csv")
    ap.add_argument("--clean", default="data/mpg_clean.csv")
    ap.add_argument("--baseline-run", default="outputs/mpg_baseline/run.json")
    ap.add_argument("--agent-run", default="outputs/mpg_agent/run.json")
    ap.add_argument("--no-agent", action="store_true", help="Leave out the agent panel even if a run exists.")
    ap.add_argument("--out", default="outputs/regression_comparison.png")
    args = ap.parse_args()

    messy, clean = load_csv(args.messy), load_csv(args.clean)
    versions = [("Raw data (no cleaning)", numeric_pairs(messy, X, Y), None)]
    runs = [("Cleaned by rule-based script", args.baseline_run)]
    if not args.no_agent:
        runs.append(("Cleaned by AI agent", args.agent_run))
    for label, path in runs:
        if Path(path).exists():
            issues = json.loads(Path(path).read_text())["issues"]
            fixed, log = apply_fixes(messy, issues)
            versions.append((label, numeric_pairs(fixed, X, Y), log))
        else:
            print(f"(skipping '{label}': {path} not found)")

    truth = fit(numeric_pairs(clean, X, Y))
    results = []
    for label, pairs, log in versions:
        f = fit(pairs)
        f["label"] = label
        f["slope_error_pct"] = slope_error_pct(f["slope"], truth["slope"])
        f["off_chart"] = int(((pairs[X] < XLIM[0]) | (pairs[X] > XLIM[1]) |
                              (pairs[Y] < YLIM[0]) | (pairs[Y] > YLIM[1])).sum())
        f["pairs"] = pairs
        results.append(f)

    # ---- chart -----------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    k = len(results)
    fig = plt.figure(figsize=(4.6 * k, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, k, height_ratios=[3, 1.15], hspace=0.55, wspace=0.12)
    xs = np.linspace(*XLIM, 50)

    for j, f in enumerate(results):
        ax = fig.add_subplot(gs[0, j], facecolor=SURFACE)
        p = f["pairs"]
        ax.scatter(p[X], p[Y], s=14, color=MUTED, alpha=0.75, linewidths=0, zorder=2)
        ax.plot(xs, truth["intercept"] + truth["slope"] * xs, color=INK, lw=1.5, ls=(0, (5, 4)),
                zorder=3, label="True relationship (clean data)")
        ax.plot(xs, f["intercept"] + f["slope"] * xs, color=FIT, lw=2.4, zorder=4,
                label="Fitted line on this version")
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_title(f["label"], loc="left", fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Weight (lbs)")
        if j == 0:
            ax.set_ylabel("Fuel efficiency (mpg)")
        else:
            ax.set_yticklabels([])
        ax.set_xticks([2000, 3000, 4000, 5000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        note = (f"Slope: {f['slope'] * 1000:+.2f} mpg per 1,000 lbs\n"
                f"True:  {truth['slope'] * 1000:+.2f}   ·   error {f['slope_error_pct']:.1f}%\n"
                f"R² = {f['r2']:.2f}   ·   n = {f['n']}")
        if f["off_chart"]:
            note += f"\n{f['off_chart']} impossible points outside this view"
        ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
                color=INK, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=GRID))
        if j == 0:
            ax.legend(loc="lower left", fontsize=8.5, frameon=False)

    bx = fig.add_subplot(gs[1, :], facecolor=SURFACE)
    labels = [f["label"] for f in results][::-1]
    errs = [f["slope_error_pct"] for f in results][::-1]
    bars = bx.barh(labels, errs, color=FIT, height=0.5, zorder=2)
    for b, e in zip(bars, errs):
        bx.text(b.get_width() + max(errs) * 0.01, b.get_y() + b.get_height() / 2, f"{e:.1f}%",
                va="center", fontsize=10, color=INK)
    bx.set_title("How far each fitted slope is from the true slope (lower is better)",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    bx.set_xlim(0, max(errs) * 1.12)
    bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    bx.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_color(GRID)
    bx.tick_params(axis="y", length=0)

    fig.suptitle("Does cleaning the data change the answer?  Car weight vs. fuel efficiency (Auto MPG)",
                 x=0.08, ha="left", y=0.995, fontsize=13.5, fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)

    table = pd.DataFrame([{k2: f[k2] for k2 in ("label", "n", "slope", "r2", "slope_error_pct")}
                          for f in results])
    table["slope"] = (table["slope"] * 1000).round(2)
    table = table.rename(columns={"slope": "slope_mpg_per_1000lbs"})
    print(f"True slope: {truth['slope'] * 1000:.2f} mpg per 1,000 lbs (R² {truth['r2']:.2f}, n {truth['n']})\n")
    print(table.round(2).to_string(index=False))
    for label, _, log in versions[1:]:
        print(f"\nFixes applied ({label}): {dict(log)}")
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nChart: {out}")


if __name__ == "__main__":
    main()
