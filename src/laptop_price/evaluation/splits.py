"""Three ways to split the data, reported side by side.

Why three
---------
* **stratified random** - what the original notebooks reported (R² 0.827). Kept
  for comparability with the existing reports.
* **grouped on spec signature** - 43% of rows are exact duplicates in feature
  space, so a random split can put the same configuration on both sides. The
  audit measured that this barely moves the score, but "we checked and it
  doesn't" is only worth saying if you actually check.
* **time-based** - train on <=2024, test on 2025. Listings span 2018-2025 and
  carry seven years of dinar inflation and tech depreciation. A random split
  over time-ordered data flatters the model; this is the number that answers
  "can we price a laptop listed tomorrow", and it is the honest headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from laptop_price.config import CONFIG


@dataclass(frozen=True)
class Split:
    """One train/validation/test partition, described well enough to report."""

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    description: str

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_idx),
            "val": len(self.val_idx),
            "test": len(self.test_idx),
        }


def stratified_split(
    y: pd.Series,
    *,
    n_bins: int = 10,
    random_state: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
) -> Split:
    """60/20/20, stratified on price decile.

    Stratification matters here because the price distribution has a long right
    tail; without it a test fold can end up with almost no premium machines.
    """
    cfg = CONFIG.model
    random_state = cfg.random_state if random_state is None else random_state
    test_size = cfg.test_size if test_size is None else test_size
    val_size = cfg.val_size if val_size is None else val_size

    positions = np.arange(len(y))
    bins = pd.qcut(np.log1p(y.to_numpy(dtype=float)), q=n_bins, labels=False, duplicates="drop")

    temp_idx, test_idx = train_test_split(
        positions, test_size=test_size, random_state=random_state, stratify=bins
    )
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_size, random_state=random_state, stratify=bins[temp_idx]
    )
    return Split(
        name="stratified_random",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        description=f"random 60/20/20 stratified on {n_bins} price bins",
    )


def grouped_split(
    groups: pd.Series,
    *,
    random_state: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
) -> Split:
    """60/20/20 where an identical configuration never straddles the split."""
    cfg = CONFIG.model
    random_state = cfg.random_state if random_state is None else random_state
    test_size = cfg.test_size if test_size is None else test_size
    val_size = cfg.val_size if val_size is None else val_size

    unique = pd.Index(groups.unique())
    rng = np.random.default_rng(random_state)
    shuffled = unique[rng.permutation(len(unique))]

    n_test = int(len(shuffled) * test_size)
    n_val = int((len(shuffled) - n_test) * val_size)
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test : n_test + n_val])

    positions = np.arange(len(groups))
    is_test = groups.isin(test_groups).to_numpy()
    is_val = groups.isin(val_groups).to_numpy()

    return Split(
        name="grouped_by_spec",
        train_idx=positions[~is_test & ~is_val],
        val_idx=positions[is_val],
        test_idx=positions[is_test],
        description=f"60/20/20 grouped on {len(unique):,} unique spec signatures",
    )


def time_based_split(
    years: pd.Series,
    *,
    train_max_year: int | None = None,
    val_fraction: float = 0.15,
    random_state: int | None = None,
) -> Split:
    """Train on listings up to ``train_max_year``, test on everything after.

    This is the headline number: it measures whether the model can price a
    listing it has never seen *in time*, which is the only question a deployed
    pricing service is ever asked.
    """
    cfg = CONFIG.model
    train_max_year = cfg.time_split_train_max_year if train_max_year is None else train_max_year
    random_state = cfg.random_state if random_state is None else random_state

    numeric_years = pd.to_numeric(years, errors="coerce")
    positions = np.arange(len(numeric_years))

    is_test = (numeric_years > train_max_year).fillna(False).to_numpy()
    # Rows with an unparseable date cannot be placed in time; keep them in train.
    train_pool = positions[~is_test]

    rng = np.random.default_rng(random_state)
    shuffled = train_pool[rng.permutation(len(train_pool))]
    n_val = int(len(shuffled) * val_fraction)

    return Split(
        name="time_based",
        train_idx=shuffled[n_val:],
        val_idx=shuffled[:n_val],
        test_idx=positions[is_test],
        description=f"train <= {train_max_year}, test > {train_max_year}",
    )


def group_kfold(
    groups: pd.Series, *, n_splits: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds where a spec signature stays on one side."""
    n_splits = CONFIG.model.n_splits if n_splits is None else n_splits
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(groups))
    return list(splitter.split(placeholder, groups=groups))


def stratified_kfold(
    y: pd.Series, *, n_splits: int | None = None, n_bins: int = 10, random_state: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds stratified on price decile."""
    cfg = CONFIG.model
    n_splits = cfg.n_splits if n_splits is None else n_splits
    random_state = cfg.random_state if random_state is None else random_state

    bins = pd.qcut(np.log1p(y.to_numpy(dtype=float)), q=n_bins, labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), bins))


def all_splits(features: pd.DataFrame, target: pd.Series) -> list[Split]:
    """Build every split strategy for the standard results table."""
    return [
        stratified_split(target),
        grouped_split(features["spec_signature"]),
        time_based_split(features["listing_year"]),
    ]


__all__ = [
    "Split",
    "all_splits",
    "group_kfold",
    "grouped_split",
    "stratified_kfold",
    "stratified_split",
    "time_based_split",
]
