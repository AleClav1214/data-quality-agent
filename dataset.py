"""Synthetic test data: a fake residential-college mailroom package log.

Why synthetic? Real package logs contain residents' names and emails.
Using made-up data avoids any privacy risk, and because we plant the
errors ourselves we have an answer key (ground truth) to grade the
auditor against.

The clean data also contains two "decoys" that look suspicious but are
NOT errors, to test judgment:
  - pickup_date is blank for packages still 'Waiting' or 'Returned'
  - a few genuinely heavy packages that a statistical outlier rule flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CARRIERS = ["Amazon", "UPS", "USPS", "FedEx", "DHL"]
CARRIER_P = [0.40, 0.22, 0.20, 0.15, 0.03]
CARRIER_VARIANTS = {
    "Amazon": ["amazon", "AMAZON", "Amazn"],
    "UPS": ["ups", "UPSS", "U.P.S."],
    "USPS": ["usps", "USPS "],
    "FedEx": ["fedex", "Fed Ex", "FEDEX"],
    "DHL": ["dhl"],
}

# How many of each planted error (before duplicates are added).
INJECTIONS = {
    "date_order": 20,
    "date_format": 30,
    "date_out_of_range": 10,
    "out_of_range": 20,
    "missing": 45,
    "category_variant": 30,
    "invalid_format": 15,
    "duplicate_row": 25,
    "duplicate_key": 10,
}

DESCRIPTION = (
    "A mailroom package log for a residential college of about 400 students, exported as a "
    "snapshot on 2026-04-30. One row per package. package_id should be unique; resident_id "
    "repeats because residents receive many packages. pickup_date is blank until a package is "
    "picked up. status is Picked Up, Waiting, or Returned. weight_lbs is the package weight in "
    "pounds (mailroom packages realistically weigh between 0.1 and 150 lbs)."
)
REFERENCE_DATE = "2026-04-30"


def _clean(n: int, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-12")
    days = 15 * 7
    arrival = np.sort(start + pd.to_timedelta(rng.integers(0, days, n), unit="D"))
    resident = rng.integers(1, 401, n)
    weight = np.round(rng.gamma(2.0, 2.5, n) + 0.2, 1)
    wait = rng.geometric(0.45, n) - 1
    last_week = arrival >= start + pd.Timedelta(days=days - 7)
    u = rng.random(n)
    status = np.where(last_week & (u < 0.6), "Waiting", np.where(u < 0.03, "Returned", "Picked Up"))
    pickup_ts = pd.Series(arrival + pd.to_timedelta(wait, unit="D"))
    # The log is a snapshot: anything not collected by the export date is still waiting.
    status = np.where((status == "Picked Up") & (pickup_ts > pd.Timestamp(REFERENCE_DATE)).to_numpy(),
                      "Waiting", status)
    pickup = pickup_ts.dt.strftime("%Y-%m-%d")
    pickup[status != "Picked Up"] = ""
    return pd.DataFrame({
        "package_id": [f"PKG-{i:05d}" for i in range(1, n + 1)],
        "resident_id": [f"R-{r:04d}" for r in resident],
        "resident_email": [f"r{r:04d}@example.edu" for r in resident],
        "carrier": rng.choice(CARRIERS, n, p=CARRIER_P),
        "arrival_date": pd.Series(arrival).dt.strftime("%Y-%m-%d"),
        "pickup_date": pickup,
        "status": status,
        "weight_lbs": [f"{w:.1f}" for w in weight],
        "size_category": np.select([weight < 3, weight < 15], ["Small", "Medium"], "Large"),
        "_uid": np.arange(n),
        "_sort": np.arange(n, dtype=float),
    })


def generate(n: int = 3000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (messy_df, ground_truth, clean_df)."""
    if n < 600:
        raise ValueError("Use at least 600 rows so every error type fits.")
    rng = np.random.default_rng(seed)
    df = _clean(n, rng)
    clean = df.drop(columns=["_uid", "_sort"]).copy()
    truth: list[tuple[int, str, str]] = []  # (_uid, category, column)

    picked = df.index[df["status"] == "Picked Up"].to_numpy()
    used: set[int] = set()

    def take(k: int, pool=None) -> np.ndarray:
        pool = df.index.to_numpy() if pool is None else pool
        avail = np.array([i for i in pool if i not in used])
        chosen = rng.choice(avail, k, replace=False)
        used.update(int(c) for c in chosen)
        return chosen

    def ts(i, col):
        return pd.Timestamp(df.at[i, col])

    for i in take(INJECTIONS["date_order"], picked):  # picked up before it arrived
        df.at[i, "pickup_date"] = (ts(i, "arrival_date") - pd.Timedelta(days=int(rng.integers(1, 6)))).strftime("%Y-%m-%d")
        truth.append((i, "date_order", "arrival_date -> pickup_date"))

    for i in take(INJECTIONS["date_format"], picked):  # someone typed the date differently
        fmt = rng.choice(["%m/%d/%Y", "%B %d, %Y", "%d-%m-%Y"])
        df.at[i, "arrival_date"] = ts(i, "arrival_date").strftime(fmt)
        truth.append((i, "date_format", "arrival_date"))

    for i in take(INJECTIONS["date_out_of_range"], picked):  # year typo: 2027 instead of 2026
        for col in ("arrival_date", "pickup_date"):
            d = ts(i, col)
            df.at[i, col] = d.replace(year=d.year + 1).strftime("%Y-%m-%d")
        truth.append((i, "date_out_of_range", "arrival_date"))

    for i in take(INJECTIONS["out_of_range"]):  # impossible weights
        w = -rng.uniform(0.5, 3) if rng.random() < 0.5 else rng.uniform(500, 1500)
        df.at[i, "weight_lbs"] = f"{w:.1f}"
        truth.append((i, "out_of_range", "weight_lbs"))

    for col, idx in zip(["carrier", "resident_email", "weight_lbs"],
                        np.array_split(take(INJECTIONS["missing"]), 3)):
        for i in idx:
            df.at[i, col] = rng.choice(["", "", "N/A"])
            truth.append((i, "missing", col))

    for i in take(INJECTIONS["category_variant"]):
        df.at[i, "carrier"] = rng.choice(CARRIER_VARIANTS[df.at[i, "carrier"]])
        truth.append((i, "category_variant", "carrier"))

    for i in take(INJECTIONS["invalid_format"]):
        e = df.at[i, "resident_email"]
        df.at[i, "resident_email"] = rng.choice([e.replace("@", ""), e.replace(".edu", ""),
                                                  e.replace("@", "@@"), e.replace("@", " @")])
        truth.append((i, "invalid_format", "resident_email"))

    extra = []
    next_uid = n
    for i in take(INJECTIONS["duplicate_row"]):  # same package scanned twice
        row = df.loc[i].copy()
        row["_uid"], row["_sort"] = next_uid, df.at[i, "_sort"] + 0.5
        truth += [(i, "duplicate_row", ""), (next_uid, "duplicate_row", "")]
        extra.append(row)
        next_uid += 1

    for i in take(INJECTIONS["duplicate_key"], picked):  # ID reused for a different package
        row = df.loc[i].copy()
        r = int(rng.integers(1, 401))
        shift = pd.Timedelta(days=int(rng.integers(1, 4)))
        row["resident_id"], row["resident_email"] = f"R-{r:04d}", f"r{r:04d}@example.edu"
        row["carrier"] = rng.choice(CARRIERS, p=CARRIER_P)
        row["weight_lbs"] = f"{rng.gamma(2.0, 2.5) + 0.2:.1f}"
        row["arrival_date"] = (ts(i, "arrival_date") + shift).strftime("%Y-%m-%d")
        row["pickup_date"] = (ts(i, "pickup_date") + shift).strftime("%Y-%m-%d")
        row["_uid"], row["_sort"] = next_uid, df.at[i, "_sort"] + 3.5
        truth += [(i, "duplicate_key", "package_id"), (next_uid, "duplicate_key", "package_id")]
        extra.append(row)
        next_uid += 1

    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df = df.sort_values("_sort", kind="stable").reset_index(drop=True)
    uid_to_row = dict(zip(df["_uid"].astype(int), df.index))

    gt = pd.DataFrame(truth, columns=["_uid", "category", "column"])
    gt["row_index"] = gt["_uid"].map(uid_to_row)
    gt = gt[["row_index", "category", "column"]].sort_values(["row_index", "category"]).reset_index(drop=True)
    messy = df.drop(columns=["_uid", "_sort"])
    return messy, gt, clean
