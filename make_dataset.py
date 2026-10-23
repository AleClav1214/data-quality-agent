"""Create a test dataset with planted errors and its answer key.

    python scripts/make_dataset.py                     # Auto MPG (real data) + package log (synthetic)
    python scripts/make_dataset.py --dataset mpg
    python scripts/make_dataset.py --dataset packages --rows 5000 --seed 7
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dq_agent import dataset, mpg_data  # noqa: E402


def write(name: str, messy, truth, clean, out: Path) -> None:
    messy.to_csv(out / f"{name}_messy.csv", index=False)
    clean.to_csv(out / f"{name}_clean.csv", index=False)
    truth.to_csv(out / f"{name}_ground_truth.csv", index=False)
    print(f"{name}: {len(messy):,} rows -> {out / f'{name}_messy.csv'}; "
          f"{len(truth)} planted (row, error) pairs -> {out / f'{name}_ground_truth.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["all", "mpg", "packages"], default="all")
    ap.add_argument("--rows", type=int, default=3000, help="package log only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "mpg"):
        write("mpg", *mpg_data.generate(args.seed), out)
    if args.dataset in ("all", "packages"):
        write("package_log", *dataset.generate(args.rows, args.seed), out)


if __name__ == "__main__":
    main()
