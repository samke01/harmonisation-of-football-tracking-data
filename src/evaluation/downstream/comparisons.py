"""Paired comparison aggregators for downstream reports."""

from __future__ import annotations

import pandas as pd

from src.evaluation.stats import _ci95, _paired_mean_summary


def _same_n_classification_comparisons(measurements: pd.DataFrame) -> list[dict]:
    """Compare merged against single-source regimes at identical N and seed."""
    rows = []
    for source_regime in ["dfl", "skillcorner"]:
        pairs = measurements.merge(
            measurements[measurements["regime"] == "merged"],
            on=["seed", "train_size"],
            suffixes=("_source", "_merged"),
        )
        pairs = pairs[pairs["regime_source"] == source_regime].copy()
        if pairs.empty:
            continue
        pairs["delta_macro_f1"] = pairs["macro_f1_merged"] - pairs["macro_f1_source"]
        pairs["delta_balanced_accuracy"] = (
            pairs["balanced_accuracy_merged"] - pairs["balanced_accuracy_source"]
        )
        summary = (
            pairs.groupby("train_size")
            .agg(
                source_regime=("regime_source", "first"),
                delta_macro_f1_mean=("delta_macro_f1", "mean"),
                delta_macro_f1_std=("delta_macro_f1", "std"),
                delta_macro_f1_ci95=("delta_macro_f1", _ci95),
                delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
                delta_balanced_accuracy_std=("delta_balanced_accuracy", "std"),
                delta_balanced_accuracy_ci95=("delta_balanced_accuracy", _ci95),
                n_pairs=("delta_macro_f1", "size"),
            )
            .reset_index()
        )
        rows.extend(summary.to_dict(orient="records"))
    return rows


def _same_n_regression_comparisons(measurements: pd.DataFrame) -> list[dict]:
    """Compare merged against single-source regimes at identical N, seed, target."""
    rows = []
    merged = measurements[measurements["regime"] == "merged"]
    for source_regime in ["dfl", "skillcorner"]:
        pairs = measurements.merge(
            merged,
            on=["seed", "target", "train_size"],
            suffixes=("_source", "_merged"),
        )
        pairs = pairs[pairs["regime_source"] == source_regime].copy()
        if pairs.empty:
            continue
        pairs["delta_mae_source_minus_merged"] = pairs["mae_source"] - pairs["mae_merged"]
        pairs["delta_r2_merged_minus_source"] = pairs["r2_merged"] - pairs["r2_source"]
        summary = (
            pairs.groupby(["target", "train_size"])
            .agg(
                source_regime=("regime_source", "first"),
                delta_mae_mean=("delta_mae_source_minus_merged", "mean"),
                delta_mae_std=("delta_mae_source_minus_merged", "std"),
                delta_mae_ci95=("delta_mae_source_minus_merged", _ci95),
                delta_r2_mean=("delta_r2_merged_minus_source", "mean"),
                delta_r2_std=("delta_r2_merged_minus_source", "std"),
                delta_r2_ci95=("delta_r2_merged_minus_source", _ci95),
                n_pairs=("delta_mae_source_minus_merged", "size"),
            )
            .reset_index()
        )
        rows.extend(summary.to_dict(orient="records"))
    return rows


def _same_n_target_classification_comparisons(measurements: pd.DataFrame) -> list[dict]:
    """Compare each non-target regime against target_only at identical pairs."""
    pair_key = "fold_id" if "fold_id" in measurements.columns else "seed"
    base = (
        measurements[measurements["regime"] == "target_only"][
            [pair_key, "target_source", "train_size", "macro_f1", "balanced_accuracy"]
        ]
        .rename(
            columns={
                "macro_f1": "macro_f1_target_only",
                "balanced_accuracy": "balanced_accuracy_target_only",
            }
        )
    )
    rows = []
    for compare_regime in ["merged_same_n", "target_plus_other_2n", "other_only"]:
        comp = (
            measurements[measurements["regime"] == compare_regime][
                [pair_key, "target_source", "train_size", "macro_f1", "balanced_accuracy"]
            ]
            .rename(
                columns={
                    "macro_f1": "macro_f1_compare",
                    "balanced_accuracy": "balanced_accuracy_compare",
                }
            )
        )
        pairs = base.merge(comp, on=[pair_key, "target_source", "train_size"])
        if pairs.empty:
            continue
        pairs["delta_macro_f1"] = pairs["macro_f1_compare"] - pairs["macro_f1_target_only"]
        pairs["delta_balanced_accuracy"] = (
            pairs["balanced_accuracy_compare"] - pairs["balanced_accuracy_target_only"]
        )
        for (target_source, train_size), grp in pairs.groupby(["target_source", "train_size"]):
            f1_summary = _paired_mean_summary(grp["delta_macro_f1"])
            ba_summary = _paired_mean_summary(grp["delta_balanced_accuracy"])
            rows.append(
                {
                    "target_source": target_source,
                    "train_size": int(train_size),
                    "compare_regime": compare_regime,
                    "delta_macro_f1_mean": f1_summary["mean"],
                    "delta_macro_f1_std": f1_summary["std"],
                    "delta_macro_f1_ci95": f1_summary["ci95_normal"],
                    "delta_macro_f1_ci95_bootstrap_low": f1_summary["ci95_bootstrap_low"],
                    "delta_macro_f1_ci95_bootstrap_high": f1_summary["ci95_bootstrap_high"],
                    "delta_macro_f1_wilcoxon_p": f1_summary["wilcoxon_p"],
                    "delta_balanced_accuracy_mean": ba_summary["mean"],
                    "delta_balanced_accuracy_std": ba_summary["std"],
                    "delta_balanced_accuracy_ci95": ba_summary["ci95_normal"],
                    "delta_balanced_accuracy_ci95_bootstrap_low": ba_summary[
                        "ci95_bootstrap_low"
                    ],
                    "delta_balanced_accuracy_ci95_bootstrap_high": ba_summary[
                        "ci95_bootstrap_high"
                    ],
                    "delta_balanced_accuracy_wilcoxon_p": ba_summary["wilcoxon_p"],
                    "n_pairs": f1_summary["n_pairs"],
                }
            )
    return rows


def _same_n_target_regression_comparisons(measurements: pd.DataFrame) -> list[dict]:
    """Paired regression comparisons with positive deltas meaning compare is better."""
    pair_key = "fold_id" if "fold_id" in measurements.columns else "seed"
    base = (
        measurements[measurements["regime"] == "target_only"][
            [pair_key, "target_source", "target", "train_size", "mae", "r2"]
        ]
        .rename(columns={"mae": "mae_target_only", "r2": "r2_target_only"})
    )
    rows = []
    for compare_regime in ["merged_same_n", "target_plus_other_2n", "other_only"]:
        comp = (
            measurements[measurements["regime"] == compare_regime][
                [pair_key, "target_source", "target", "train_size", "mae", "r2"]
            ]
            .rename(columns={"mae": "mae_compare", "r2": "r2_compare"})
        )
        pairs = base.merge(comp, on=[pair_key, "target_source", "target", "train_size"])
        if pairs.empty:
            continue
        pairs["delta_mae"] = pairs["mae_target_only"] - pairs["mae_compare"]
        pairs["delta_r2"] = pairs["r2_compare"] - pairs["r2_target_only"]
        for (target_source, target, train_size), grp in pairs.groupby(
            ["target_source", "target", "train_size"]
        ):
            mae_summary = _paired_mean_summary(grp["delta_mae"])
            r2_summary = _paired_mean_summary(grp["delta_r2"])
            rows.append(
                {
                    "target_source": target_source,
                    "target": target,
                    "train_size": int(train_size),
                    "compare_regime": compare_regime,
                    "delta_mae_mean": mae_summary["mean"],
                    "delta_mae_std": mae_summary["std"],
                    "delta_mae_ci95": mae_summary["ci95_normal"],
                    "delta_mae_ci95_bootstrap_low": mae_summary["ci95_bootstrap_low"],
                    "delta_mae_ci95_bootstrap_high": mae_summary["ci95_bootstrap_high"],
                    "delta_mae_wilcoxon_p": mae_summary["wilcoxon_p"],
                    "delta_r2_mean": r2_summary["mean"],
                    "delta_r2_std": r2_summary["std"],
                    "delta_r2_ci95": r2_summary["ci95_normal"],
                    "delta_r2_ci95_bootstrap_low": r2_summary["ci95_bootstrap_low"],
                    "delta_r2_ci95_bootstrap_high": r2_summary["ci95_bootstrap_high"],
                    "delta_r2_wilcoxon_p": r2_summary["wilcoxon_p"],
                    "n_pairs": mae_summary["n_pairs"],
                }
            )
    return rows
