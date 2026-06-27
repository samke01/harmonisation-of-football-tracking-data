"""Sampling regimes and split helpers for downstream evaluations."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PER_TARGET_REGIMES = [
    "target_only",
    "other_only",
    "merged_same_n",
    "target_plus_other_2n",
]


def _sample_train(
    df: pd.DataFrame,
    regime: str,
    train_size: int,
    seed: int,
) -> pd.DataFrame:
    if regime == "dfl":
        pool = df[df["source"] == "DFL"]
    elif regime == "skillcorner":
        pool = df[df["source"] == "SkillCorner"]
    elif regime == "merged":
        pool = df
    else:
        raise ValueError(f"Unknown regime: {regime}")

    if len(pool) < train_size:
        raise ValueError(f"Regime {regime} has only {len(pool)} rows, need {train_size}")
    return pool.sample(n=train_size, random_state=seed)


def _sample_train_per_target(
    target_train_pool: pd.DataFrame,
    other_pool: pd.DataFrame,
    regime: str,
    n: int,
    seed: int,
) -> pd.DataFrame | None:
    if regime == "target_only":
        if len(target_train_pool) < n:
            return None
        return target_train_pool.sample(n=n, random_state=seed)
    if regime == "other_only":
        if len(other_pool) < n:
            return None
        return other_pool.sample(n=n, random_state=seed)
    if regime == "merged_same_n":
        half = n // 2
        rest = n - half
        if len(target_train_pool) < half or len(other_pool) < rest:
            return None
        return pd.concat(
            [
                target_train_pool.sample(n=half, random_state=seed),
                other_pool.sample(n=rest, random_state=seed + 100),
            ],
            ignore_index=True,
        )
    if regime == "target_plus_other_2n":
        if len(target_train_pool) < n or len(other_pool) < n:
            return None
        return pd.concat(
            [
                target_train_pool.sample(n=n, random_state=seed),
                other_pool.sample(n=n, random_state=seed + 100),
            ],
            ignore_index=True,
        )
    raise ValueError(f"Unknown per-target regime: {regime}")


def _iter_target_splits(
    target_df: pd.DataFrame,
    label_col: str,
    seeds: list[int],
    test_size: float,
    cv_scheme: str,
) -> Iterator[tuple[int, str, pd.DataFrame, pd.DataFrame]]:
    """Yield (sample_seed, fold_id, target_train_pool, test_df) per fold."""
    if cv_scheme == "lomo":
        sample_seed = int(seeds[0])
        match_ids = sorted(target_df["match_id"].astype(str).unique().tolist())
        for held_out in match_ids:
            test_df = target_df[target_df["match_id"].astype(str) == held_out].copy()
            target_train_pool = target_df[target_df["match_id"].astype(str) != held_out].copy()
            yield sample_seed, f"lomo:{held_out}", target_train_pool, test_df
        return

    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(
            splitter.split(target_df, target_df[label_col], groups=target_df["match_id"])
        )
        target_train_pool = target_df.iloc[train_idx].copy()
        test_df = target_df.iloc[test_idx].copy()
        yield int(seed), f"shuffle:{seed}", target_train_pool, test_df
