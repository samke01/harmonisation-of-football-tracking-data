"""Statistical summaries for downstream harmonization reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ci95(values: pd.Series) -> float:
    """Return normal-approximate 95% CI half-width."""
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _bootstrap_ci(
    values: pd.Series,
    n_resamples: int = 5000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap-percentile CI for the mean of `values`."""
    arr = values.dropna().to_numpy()
    if len(arr) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import bootstrap

        res = bootstrap(
            (arr,),
            np.mean,
            confidence_level=1 - alpha,
            n_resamples=n_resamples,
            method="BCa",
            random_state=0,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        rng = np.random.default_rng(0)
        means = np.array(
            [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
        )
        return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _wilcoxon_p(values: pd.Series) -> float:
    """One-sample two-sided Wilcoxon signed-rank p-value vs. zero."""
    arr = values.dropna().to_numpy()
    arr = arr[arr != 0]
    if len(arr) < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def _paired_mean_summary(values: pd.Series) -> dict:
    """Standardised summary of a paired-difference series."""
    lo, hi = _bootstrap_ci(values)
    return {
        "mean": float(values.dropna().mean()) if values.notna().any() else float("nan"),
        "std": float(values.dropna().std(ddof=1)) if values.notna().sum() > 1 else 0.0,
        "ci95_normal": _ci95(values),
        "ci95_bootstrap_low": lo,
        "ci95_bootstrap_high": hi,
        "wilcoxon_p": _wilcoxon_p(values),
        "n_pairs": int(values.notna().sum()),
    }
