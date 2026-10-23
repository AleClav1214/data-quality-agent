"""Real-data demo: the Auto MPG dataset with planted errors.

Source: Auto MPG, UCI Machine Learning Repository (Quinlan, 1993),
CC BY 4.0, via the seaborn-data copy saved in data/source/auto_mpg.csv.

Heavier cars get fewer miles per gallon (r = -0.83), a well-known
relationship. We treat the published data (minus 6 cars with no
horsepower) as the ground truth, then plant the kinds of mistakes people
make when typing data into a spreadsheet. Because we planted them, we
have an answer key AND we know the true regression slope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "data" / "source" / "auto_mpg.csv"

INJECTIONS = {
    "weight_extra_zero": 6,   # 3504 typed as 35040
    "weight_in_kg": 6,        # entered in kilograms instead of pounds
    "mpg_decimal_shift": 5,   # 24.5 typed as 245
    "missing": 10,            # blank or N/A in mpg / weight
    "horsepower_unknown": 6,  # '?' placeholder, as in the original UCI file
    "invalid_format": 8,      # weight typed with units or commas: '3,504' or '3504 lbs'
    "category_variant": 15,   # 'USA', 'U.S.A.', 'Japan ' instead of usa / japan
    "duplicate_row": 8,       # car entered twice
}

ORIGIN_VARIANTS = {
    "usa": ["USA", "U.S.A.", "Usa", "usa "],
    "japan": ["Japan", "JAPAN", "japan "],
    "europe": ["Europe", "EUROPE", "europe."],
}

DESCRIPTION = (
    "Specifications of about 400 car models from 1970-1982 (the classic Auto MPG dataset). "
    "One row per car model. mpg is fuel efficiency in miles per gallon (realistic range 5-60). "
    "weight is curb weight in pounds (passenger cars realistically weigh 1,500-6,000 lbs). "
    "horsepower is engine power (40-300). model_year is the 2-digit year 70-82. "
    "origin is usa, japan, or europe. The same car name can repeat across model years."
)


def load_clean() -> pd.DataFrame:
    """The published data, minus the 6 cars with unknown horsepower, as text."""
    df = pd.read_csv(SOURCE).dropna().reset_index(drop=True)
    out = df.astype(str)
    for col in ("mpg", "displacement", "horsepower", "acceleration"):
        out[col] = df[col].map(lambda v: f"{v:g}")
    out["weight"] = df["weight"].astype(int).astype(str)
    return out


def generate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (messy_df, ground_truth, clean_df)."""
    rng = np.random.default_rng(seed)
    clean = load_clean()
    df = clean.copy()
    df["_uid"] = np.arange(len(df))
    df["_sort"] = np.arange(len(df), dtype=float)
    truth: list[tuple[int, str, str]] = []
    used: set[int] = set()

    def take(k: int) -> np.ndarray:
        avail = np.array([i for i in df.index if i not in used])
        chosen = rng.choice(avail, k, replace=False)
        used.update(int(c) for c in chosen)
        return chosen

    for i in take(INJECTIONS["weight_extra_zero"]):
        df.at[i, "weight"] = str(int(df.at[i, "weight"]) * 10)
        truth.append((i, "out_of_range", "weight"))

    for i in take(INJECTIONS["weight_in_kg"]):
        df.at[i, "weight"] = str(round(int(df.at[i, "weight"]) / 2.2046))
        truth.append((i, "out_of_range", "weight"))

    for i in take(INJECTIONS["mpg_decimal_shift"]):
        df.at[i, "mpg"] = f"{float(df.at[i, 'mpg']) * 10:g}"
        truth.append((i, "out_of_range", "mpg"))

    for col, idx in zip(["mpg", "weight"], np.array_split(take(INJECTIONS["missing"]), 2)):
        for i in idx:
            df.at[i, col] = rng.choice(["", "N/A"])
            truth.append((i, "missing", col))

    for i in take(INJECTIONS["horsepower_unknown"]):
        df.at[i, "horsepower"] = "?"
        truth.append((i, "missing", "horsepower"))

    for i in take(INJECTIONS["invalid_format"]):
        w = int(df.at[i, "weight"])
        df.at[i, "weight"] = rng.choice([f"{w:,}", f"{w} lbs"])
        truth.append((i, "invalid_format", "weight"))

    for i in take(INJECTIONS["category_variant"]):
        df.at[i, "origin"] = rng.choice(ORIGIN_VARIANTS[df.at[i, "origin"]])
        truth.append((i, "category_variant", "origin"))

    extra = []
    next_uid = len(df)
    for i in take(INJECTIONS["duplicate_row"]):
        row = df.loc[i].copy()
        row["_uid"], row["_sort"] = next_uid, df.at[i, "_sort"] + 0.5
        truth += [(i, "duplicate_row", ""), (next_uid, "duplicate_row", "")]
        extra.append(row)
        next_uid += 1

    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df = df.sort_values("_sort", kind="stable").reset_index(drop=True)
    uid_to_row = dict(zip(df["_uid"].astype(int), df.index))
    gt = pd.DataFrame(truth, columns=["_uid", "category", "column"])
    gt["row_index"] = gt["_uid"].map(uid_to_row)
    gt = gt[["row_index", "category", "column"]].sort_values(["row_index", "category"]).reset_index(drop=True)
    return df.drop(columns=["_uid", "_sort"]), gt, clean
