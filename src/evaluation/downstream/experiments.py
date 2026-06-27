"""Downstream evaluation for merged DFL/SkillCorner data.

The direct source-shift metrics answer whether the sources remain
distinguishable. This module asks the next question: does merged training help
for a concrete task when the training budget is controlled?

Final tasks implemented here:

- Player-match aggregate position/role proxy classification
- Player-match kinematic regression
- Ball-status prediction as a robustness task
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.evaluation.downstream.comparisons import (
    _same_n_classification_comparisons,
    _same_n_regression_comparisons,
    _same_n_target_classification_comparisons,
    _same_n_target_regression_comparisons,
)
from src.evaluation.downstream.correction import apply_train_only_correction
from src.evaluation.datasets import (
    _add_pass_event_features,
    _label_to_bucket,
    _load_pass_events_base,
    _metadata_position_lookup,
    _tracking_context_for_match,
    load_ball_status_dataset,
    load_kinematic_regression_dataset,
    load_kinematic_regression_visibility_corrected_dataset,
    load_pass_success_event_dataset,
    load_pass_success_tracking_context_dataset,
    load_player_aggregate_position_dataset,
    load_player_aggregate_position_no_kinematic_dataset,
    load_player_aggregate_position_with_tracking_context_dataset,
    load_shot_success_event_dataset,
)
from src.evaluation.serialization import _json_default, save_report
from src.evaluation.metrics import _evaluate_predictions, _evaluate_regression
from src.evaluation.models import _model, _regression_model
from src.evaluation.downstream.plots import (
    _plot_data_acquisition_heatmap,
    _plot_learning_curves,
    _plot_regression_learning_curves,
    _plot_regression_transfer,
    _plot_target_augmentation_classification,
    _plot_target_augmentation_regression,
    _plot_transfer,
    build_forest_plot_summary,
)
from src.evaluation.downstream.regimes import (
    PER_TARGET_REGIMES,
    _iter_target_splits,
    _sample_train,
    _sample_train_per_target,
)
from src.evaluation.stats import (
    _bootstrap_ci,
    _ci95,
    _paired_mean_summary,
    _wilcoxon_p,
)

DATA_PATH = Path(settings.data_path)
MATCHES_DIR = DATA_PATH / "merged" / "matches"
REPORTS_DIR = DATA_PATH / "reports"
PLOTS_DIR = Path("plots")
RANDOM_SEED = 42

TRACKING_TRAIN_SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000]
SEEDS = [42, 43, 44, 45, 46]
# Extended seed list used by Phase-1 statistical-robustness runs.
# 20 seeds give tight enough CIs that paired tests are no longer underpowered.
SEEDS_EXTENDED = list(range(42, 62))
# Optional very-small-N grid for tracking-derived player-match tasks.
# Probes whether merging helps in the extreme low-data regime where the
# within-source learning curve is still steep.
PLAYER_AGG_TINY_SIZES = [10, 20, 30, 50, 100, 150]


# CDF Appendix C labels → legacy 6-class tactical bucket used by the
# downstream player-aggregate position classification task. The CDF
# ``position_group`` (DF/MF/FW) is too coarse for a meaningful
# classification target on a 370-player dataset; the tactical bucket
# preserves the thesis task at its original granularity.


def run_position_label_audit() -> dict:
    """Audit position-group metadata consistency across DFL and SkillCorner."""
    rows = []
    for meta_path in sorted(MATCHES_DIR.glob("*/*_metadata.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        match_id = str(meta["match_id"])
        source = meta["source"]
        for side in ("home", "away"):
            team = meta.get("teams", {}).get(side, {})
            for player in team.get("players", []):
                rows.append(
                    {
                        "source": source,
                        "match_id": match_id,
                        "team_id": str(team.get("id")),
                        "team_name": team.get("name"),
                        "player_id": str(player.get("id")),
                        "player_name": player.get("name"),
                        "position_raw": player.get("position_raw"),
                        "position_group": player.get("position_group"),
                        "starting": player.get("starting"),
                        "start_time": player.get("start_time"),
                        "end_time": player.get("end_time"),
                    }
                )

    roster = pd.DataFrame(rows)
    tracking_rows = []
    for path in sorted(MATCHES_DIR.glob("*/*_tracking_10hz.parquet")):
        cols = ["source", "match_id", "player_id", "position_group", "is_visible"]
        df = pd.read_parquet(path, columns=cols)
        visible = df[df["is_visible"] == True]
        tracking_rows.append(
            {
                "source": df["source"].iloc[0],
                "match_id": str(df["match_id"].iloc[0]),
                "tracking_players": int(df["player_id"].nunique()),
                "visible_players": int(visible["player_id"].nunique()),
                "tracking_position_group_nonnull_rate": float(df["position_group"].notna().mean()),
            }
        )
    tracking = pd.DataFrame(tracking_rows)

    grouped = roster.groupby("source")
    summary = grouped.agg(
        roster_players=("player_id", "count"),
        players_with_position_raw=("position_raw", lambda s: int(s.notna().sum())),
        players_with_position_group=("position_group", lambda s: int(s.notna().sum())),
        starters=("starting", lambda s: int((s == True).sum())),
        used_players=("start_time", lambda s: int(s.notna().sum())),
    ).reset_index()
    summary["position_group_rate"] = (
        summary["players_with_position_group"] / summary["roster_players"]
    )

    position_counts = (
        roster.groupby(["source", "position_group"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    raw_counts = (
        roster.groupby(["source", "position_raw"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    report = {
        "description": (
            "Audit of position labels used by position-group downstream tasks. "
            "Both sources expose position labels via match metadata/team sheets; "
            "SkillCorner does not repeat them on each tracking row, so the "
            "evaluation joins metadata labels by match_id/player_id."
        ),
        "risk_assessment": (
            "Labels are available for both sources but may encode provider/team-sheet "
            "semantics rather than observed tactical role. Treat position tasks as "
            "role-proxy tasks unless manually validated."
        ),
        "summary": summary.to_dict(orient="records"),
        "position_group_counts": position_counts.astype({"position_group": "string"}).to_dict(orient="records"),
        "position_raw_counts": raw_counts.astype({"position_raw": "string"}).to_dict(orient="records"),
        "tracking_position_coverage": tracking.to_dict(orient="records"),
    }
    save_report(report, REPORTS_DIR / "position_label_audit.json")
    return report


def _run_learning_curves_for_loader(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    train_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
) -> dict:
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    regimes = ["dfl", "skillcorner", "merged"]
    rows = []

    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, df[label_col], groups=df["match_id"]))
        train_pool = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        for regime in regimes:
            for train_size in train_sizes:
                try:
                    train_df = _sample_train(train_pool, regime, train_size, seed)
                except ValueError:
                    continue
                clf = _model(seed)
                clf.fit(train_df[feature_cols], train_df[label_col])
                pred = clf.predict(test_df[feature_cols])
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "train_size": train_size,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "test_matches": sorted(test_df["match_id"].unique().tolist()),
                        **_evaluate_predictions(test_df[label_col].to_numpy(), pred),
                    }
                )

    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["regime", "train_size"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    same_n = _same_n_classification_comparisons(measurements)
    report = {
        "task": task_name,
        "label": label_col,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df[label_col].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
    }
    save_report(report, REPORTS_DIR / f"{output_prefix}_learning_curves.json")
    _plot_learning_curves(summary, PLOTS_DIR / f"{output_prefix}_learning_curves.png")
    return report


def run_ball_status_learning_curves() -> dict:
    """Run learning curves for ball-in-play prediction."""
    return _run_learning_curves_for_loader(
        load_ball_status_dataset,
        "ball_status_prediction",
        "ball_status",
        TRACKING_TRAIN_SIZES,
    )


def run_player_aggregate_position_learning_curves() -> dict:
    """Run learning curves for player-match aggregate position classification."""
    return _run_learning_curves_for_loader(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        "player_aggregate_position",
        [50, 100, 150, 200, 250, 300],
    )


def run_kinematic_regression_learning_curves(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Run learning curves for player-match kinematic regression."""
    train_sizes = train_sizes or [50, 100, 150, 200, 250, 300]
    seeds = seeds or SEEDS
    df, feature_cols, target_cols = load_kinematic_regression_dataset()
    regimes = ["dfl", "skillcorner", "merged"]
    rows = []
    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
        train_pool = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        for regime in regimes:
            for train_size in train_sizes:
                try:
                    train_df = _sample_train(train_pool, regime, train_size, seed)
                except ValueError:
                    continue
                for target in target_cols:
                    model = _regression_model(seed)
                    model.fit(train_df[feature_cols], train_df[target])
                    pred = model.predict(test_df[feature_cols])
                    rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "target": target,
                            "train_size": train_size,
                            "n_train": len(train_df),
                            "n_test": len(test_df),
                            "test_matches": sorted(test_df["match_id"].unique().tolist()),
                            **_evaluate_regression(test_df[target].to_numpy(), pred),
                        }
                    )
    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target", "regime", "train_size"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_ci95=("mae", _ci95),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_runs=("mae", "size"),
        )
        .reset_index()
    )
    same_n = _same_n_regression_comparisons(measurements)
    report = {
        "task": "player_match_kinematic_regression",
        "feature_policy": (
            "role/location descriptors only; direct speed/distance aggregate "
            "features and n_frames excluded"
        ),
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "targets": target_cols,
        "feature_cols": feature_cols,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
    }
    save_report(report, REPORTS_DIR / "kinematic_regression_learning_curves.json")
    _plot_regression_learning_curves(summary, PLOTS_DIR / "kinematic_regression_learning_curves.png")
    return report


def _run_transfer_for_loader(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    seeds: list[int] | None = None,
) -> dict:
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    rows = []
    for seed in seeds:
        for target_source in ["DFL", "SkillCorner"]:
            other_source = "SkillCorner" if target_source == "DFL" else "DFL"
            target_df = df[df["source"] == target_source].copy()
            other_df = df[df["source"] == other_source].copy()

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
            target_train_idx, target_test_idx = next(
                splitter.split(target_df, target_df[label_col], groups=target_df["match_id"])
            )
            target_train = target_df.iloc[target_train_idx]
            target_test = target_df.iloc[target_test_idx]

            train_budget = min(len(other_df), len(target_train), 32_000)
            if train_budget < 50:
                continue
            half_budget = train_budget // 2
            train_sets = {
                f"{other_source.lower()}_to_{target_source.lower()}": other_df.sample(
                    n=train_budget, random_state=seed
                ),
                f"{target_source.lower()}_only_to_{target_source.lower()}": target_train.sample(
                    n=train_budget, random_state=seed
                ),
                f"merged_to_{target_source.lower()}": pd.concat(
                    [
                        other_df.sample(n=half_budget, random_state=seed),
                        target_train.sample(n=train_budget - half_budget, random_state=seed + 100),
                    ],
                    ignore_index=True,
                ),
            }
            for exp_name, train_df in train_sets.items():
                clf = _model(seed)
                clf.fit(train_df[feature_cols], train_df[label_col])
                pred = clf.predict(target_test[feature_cols])
                rows.append(
                    {
                        "seed": seed,
                        "experiment": exp_name,
                        "train_source": (
                            "merged_balanced"
                            if exp_name.startswith("merged")
                            else train_df["source"].iloc[0]
                        ),
                        "test_source": target_source,
                        "n_train": len(train_df),
                        "n_test": len(target_test),
                        "test_matches": sorted(target_test["match_id"].unique().tolist()),
                        **_evaluate_predictions(target_test[label_col].to_numpy(), pred),
                    }
                )

    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["experiment", "train_source", "test_source"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    report = {
        "task": task_name,
        "label": label_col,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df[label_col].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    save_report(report, REPORTS_DIR / f"{output_prefix}_transfer.json")
    _plot_transfer(summary, PLOTS_DIR / f"{output_prefix}_transfer.png")
    return report


def run_ball_status_transfer() -> dict:
    """Run transfer evaluation for ball-in-play prediction."""
    return _run_transfer_for_loader(
        load_ball_status_dataset,
        "ball_status_prediction",
        "ball_status",
    )


def run_player_aggregate_position_transfer() -> dict:
    """Run transfer evaluation for player-match aggregate position classification."""
    return _run_transfer_for_loader(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        "player_aggregate_position",
    )


def run_kinematic_regression_transfer(seeds: list[int] | None = None) -> dict:
    """Run transfer evaluation for player-match kinematic regression."""
    seeds = seeds or SEEDS
    df, feature_cols, target_cols = load_kinematic_regression_dataset()
    rows = []
    for seed in seeds:
        for target_source in ["DFL", "SkillCorner"]:
            other_source = "SkillCorner" if target_source == "DFL" else "DFL"
            target_df = df[df["source"] == target_source].copy()
            other_df = df[df["source"] == other_source].copy()

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
            target_train_idx, target_test_idx = next(
                splitter.split(target_df, groups=target_df["match_id"])
            )
            target_train = target_df.iloc[target_train_idx]
            target_test = target_df.iloc[target_test_idx]

            train_budget = min(len(other_df), len(target_train), 300)
            if train_budget < 50:
                continue
            half_budget = train_budget // 2
            train_sets = {
                f"{other_source.lower()}_to_{target_source.lower()}": other_df.sample(
                    n=train_budget, random_state=seed
                ),
                f"{target_source.lower()}_only_to_{target_source.lower()}": target_train.sample(
                    n=train_budget, random_state=seed
                ),
                f"merged_to_{target_source.lower()}": pd.concat(
                    [
                        other_df.sample(n=half_budget, random_state=seed),
                        target_train.sample(n=train_budget - half_budget, random_state=seed + 100),
                    ],
                    ignore_index=True,
                ),
            }
            for exp_name, train_df in train_sets.items():
                for target in target_cols:
                    model = _regression_model(seed)
                    model.fit(train_df[feature_cols], train_df[target])
                    pred = model.predict(target_test[feature_cols])
                    rows.append(
                        {
                            "seed": seed,
                            "experiment": exp_name,
                            "target": target,
                            "train_source": (
                                "merged_balanced"
                                if exp_name.startswith("merged")
                                else train_df["source"].iloc[0]
                            ),
                            "test_source": target_source,
                            "n_train": len(train_df),
                            "n_test": len(target_test),
                            "test_matches": sorted(target_test["match_id"].unique().tolist()),
                            **_evaluate_regression(target_test[target].to_numpy(), pred),
                        }
                    )
    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target", "experiment", "train_source", "test_source"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_ci95=("mae", _ci95),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_runs=("mae", "size"),
        )
        .reset_index()
    )
    report = {
        "task": "player_match_kinematic_regression",
        "feature_policy": (
            "role/location descriptors only; direct speed/distance aggregate "
            "features and n_frames excluded"
        ),
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "targets": target_cols,
        "feature_cols": feature_cols,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    save_report(report, REPORTS_DIR / "kinematic_regression_transfer.json")
    _plot_regression_transfer(summary, PLOTS_DIR / "kinematic_regression_transfer.png")
    return report


def _run_target_augmentation_classification(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    train_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
    correction: str = "none",
) -> dict:
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, label_col, seeds, test_size, cv_scheme
        ):
            for n in train_sizes:
                for regime in PER_TARGET_REGIMES:
                    train_df = _sample_train_per_target(
                        target_train_pool, other_df, regime, n, sample_seed
                    )
                    if train_df is None:
                        continue
                    correction_result = apply_train_only_correction(
                        train_df,
                        test_df,
                        feature_cols,
                        method=correction,
                        reference_batch=target_source,
                        random_state=sample_seed,
                    )
                    train_fit = correction_result.train_df
                    test_eval = correction_result.test_df
                    clf = _model(sample_seed, kind=model_kind)
                    clf.fit(train_fit[feature_cols], train_fit[label_col])
                    pred = clf.predict(test_eval[feature_cols])
                    row = {
                        "seed": sample_seed,
                        "fold_id": fold_id,
                        "target_source": target_source,
                        "regime": regime,
                        "train_size": n,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "test_matches": sorted(test_df["match_id"].unique().tolist()),
                        **_evaluate_predictions(test_eval[label_col].to_numpy(), pred),
                    }
                    if correction_result.method != "none":
                        row.update(
                            {
                                "correction": correction_result.method,
                                "correction_applied": correction_result.applied,
                                "correction_reason": correction_result.reason,
                            }
                        )
                    rows.append(row)

    measurements = pd.DataFrame(rows)
    # Use fold_id as the pairing key for LOMO; legacy seed pairing for shuffle.
    measurements["_pair_key"] = measurements.get(
        "fold_id", measurements.get("seed", "")
    ).astype(str)
    summary = (
        measurements.groupby(["target_source", "regime", "train_size"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    same_n = _same_n_target_classification_comparisons(measurements)
    report = {
        "task": task_name,
        "evaluation_design": (
            "per-target augmentation: fixed target-source test set, four train "
            "regimes (target_only, other_only, merged_same_n, "
            "target_plus_other_2n) at matched budgets"
        ),
        "model_kind": model_kind,
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "label": label_col,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df[label_col].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "regimes": PER_TARGET_REGIMES,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
    }
    if correction.lower() != "none":
        report["correction"] = correction
        report["correction_protocol"] = (
            "train-only feature correction fitted inside each fold on the "
            "sampled train regime. Single-source regimes are left unchanged; "
            "mixed-source regimes align non-reference provider rows to the "
            "target-source reference batch."
        )
    save_report(report, REPORTS_DIR / f"{output_prefix}_target_augmentation.json")
    _plot_target_augmentation_classification(
        summary, PLOTS_DIR / f"{output_prefix}_target_augmentation.png"
    )
    return report


def _run_target_augmentation_regression(
    loader: Callable[[], tuple[pd.DataFrame, list[str], list[str]]],
    task_name: str,
    output_prefix: str,
    train_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
    correction: str = "none",
) -> dict:
    seeds = seeds or SEEDS
    df, feature_cols, target_cols = loader()
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, target_cols[0], seeds, test_size, cv_scheme
        ):
            for n in train_sizes:
                for regime in PER_TARGET_REGIMES:
                    train_df = _sample_train_per_target(
                        target_train_pool, other_df, regime, n, sample_seed
                    )
                    if train_df is None:
                        continue
                    correction_result = apply_train_only_correction(
                        train_df,
                        test_df,
                        feature_cols,
                        method=correction,
                        reference_batch=target_source,
                        random_state=sample_seed,
                    )
                    train_fit = correction_result.train_df
                    test_eval = correction_result.test_df
                    for target in target_cols:
                        model = _regression_model(sample_seed, kind=model_kind)
                        model.fit(train_fit[feature_cols], train_fit[target])
                        pred = model.predict(test_eval[feature_cols])
                        row = {
                            "seed": sample_seed,
                            "fold_id": fold_id,
                            "target_source": target_source,
                            "regime": regime,
                            "target": target,
                            "train_size": n,
                            "n_train": len(train_df),
                            "n_test": len(test_df),
                            "test_matches": sorted(test_df["match_id"].unique().tolist()),
                            **_evaluate_regression(
                                test_eval[target].to_numpy(), pred
                            ),
                        }
                        if correction_result.method != "none":
                            row.update(
                                {
                                    "correction": correction_result.method,
                                    "correction_applied": correction_result.applied,
                                    "correction_reason": correction_result.reason,
                                }
                            )
                        rows.append(row)

    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target_source", "regime", "target", "train_size"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_ci95=("mae", _ci95),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_runs=("mae", "size"),
        )
        .reset_index()
    )
    same_n = _same_n_target_regression_comparisons(measurements)
    report = {
        "task": task_name,
        "evaluation_design": (
            "per-target augmentation: fixed target-source test set, four train "
            "regimes (target_only, other_only, merged_same_n, "
            "target_plus_other_2n) at matched budgets"
        ),
        "model_kind": model_kind,
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "targets": target_cols,
        "feature_cols": feature_cols,
        "regimes": PER_TARGET_REGIMES,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
    }
    if correction.lower() != "none":
        report["correction"] = correction
        report["correction_protocol"] = (
            "train-only feature correction fitted inside each fold on the "
            "sampled train regime. Single-source regimes are left unchanged; "
            "mixed-source regimes align non-reference provider rows to the "
            "target-source reference batch."
        )
    save_report(report, REPORTS_DIR / f"{output_prefix}_target_augmentation.json")
    _plot_target_augmentation_regression(
        summary, PLOTS_DIR / f"{output_prefix}_target_augmentation.png"
    )
    return report


def run_player_aggregate_position_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
    correction: str = "none",
) -> dict:
    """Run per-target augmentation eval for player-aggregate position task."""
    return _run_target_augmentation_classification(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        f"player_aggregate_position{output_suffix}",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction=correction,
    )


def run_player_aggregate_position_no_kinematic_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
    correction: str = "none",
) -> dict:
    """Run position target augmentation without speed/acceleration aggregates."""
    return _run_target_augmentation_classification(
        load_player_aggregate_position_no_kinematic_dataset,
        "player_aggregate_position_group_classification_no_kinematic",
        f"player_aggregate_position_no_kinematic{output_suffix}",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction=correction,
    )


def run_kinematic_regression_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
    correction: str = "none",
) -> dict:
    """Run per-target augmentation eval for kinematic regression task."""
    return _run_target_augmentation_regression(
        load_kinematic_regression_dataset,
        "player_match_kinematic_regression",
        f"kinematic_regression{output_suffix}",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction=correction,
    )


def run_player_aggregate_position_ls_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_ls",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run train-only location/scale correction sensitivity for position."""
    return run_player_aggregate_position_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="ls",
    )


def run_kinematic_regression_ls_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_ls",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run train-only location/scale correction sensitivity for kinematics."""
    return run_kinematic_regression_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="ls",
    )


def run_player_aggregate_position_ls_permuted_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_ls_permuted",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run location/scale correction with permuted provider labels as a null."""
    return run_player_aggregate_position_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="ls_permuted",
    )


def run_kinematic_regression_ls_permuted_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_ls_permuted",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run location/scale correction with permuted provider labels as a null."""
    return run_kinematic_regression_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="ls_permuted",
    )


def run_player_aggregate_position_combat_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_combat_p",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run train-only parametric ComBat sensitivity for position."""
    return run_player_aggregate_position_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="combat_p",
    )


def run_kinematic_regression_combat_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_combat_p",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run train-only parametric ComBat sensitivity for kinematics."""
    return run_kinematic_regression_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="combat_p",
    )


def run_player_aggregate_position_combat_permuted_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_combat_p_permuted",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run parametric ComBat with permuted provider labels as a null."""
    return run_player_aggregate_position_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="combat_p_permuted",
    )


def run_kinematic_regression_combat_permuted_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_combat_p_permuted",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Run parametric ComBat with permuted provider labels as a null."""
    return run_kinematic_regression_target_augmentation(
        train_sizes=train_sizes,
        seeds=seeds,
        output_suffix=output_suffix,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
        correction="combat_p_permuted",
    )


# ---------------------------------------------------------------------------
# Pass-success event tasks
# ---------------------------------------------------------------------------
#
# Two variants:
#  - load_pass_success_event_dataset: event-only features (start/end coords,
#    pass length, angle, forward progress, period). Pure event task, no
#    tracking confound.
#  - load_pass_success_tracking_context_dataset: event-only features plus
#    tracking-frame context at the pass timestamp (teammate / opponent
#    density and proximity). Showcases what the merged tracking + event
#    schema enables.
#
# Both are restricted to ``action_type == 'pass'`` (cross-dataset comparable
# per ``CROSS_DATASET_COMPARABLE`` in ``src.harmonization.events``). Label
# is binary success (``result == 'success'``).

PASS_SUCCESS_TRAIN_SIZES = [500, 1000, 2000, 4000]


def run_pass_success_event_learning_curves() -> dict:
    """Mixed-pool learning curves for event-only pass-success."""
    return _run_learning_curves_for_loader(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        "pass_success_event",
        PASS_SUCCESS_TRAIN_SIZES,
    )


def run_pass_success_event_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Per-target augmentation for event-only pass-success."""
    return _run_target_augmentation_classification(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        f"pass_success_event{output_suffix}",
        train_sizes if train_sizes is not None else PASS_SUCCESS_TRAIN_SIZES,
        seeds=seeds,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
    )


def run_pass_success_event_transfer() -> dict:
    """Naive cross-source transfer for event-only pass-success."""
    return _run_transfer_for_loader(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        "pass_success_event",
    )


def run_pass_success_tracking_learning_curves() -> dict:
    """Mixed-pool learning curves for pass-success with tracking context."""
    return _run_learning_curves_for_loader(
        load_pass_success_tracking_context_dataset,
        "pass_success_tracking_context_classification",
        "pass_success_tracking",
        PASS_SUCCESS_TRAIN_SIZES,
    )


def run_pass_success_tracking_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Per-target augmentation for pass-success with tracking context."""
    return _run_target_augmentation_classification(
        load_pass_success_tracking_context_dataset,
        "pass_success_tracking_context_classification",
        f"pass_success_tracking{output_suffix}",
        train_sizes if train_sizes is not None else PASS_SUCCESS_TRAIN_SIZES,
        seeds=seeds,
        cv_scheme=cv_scheme,
        model_kind=model_kind,
    )


def run_pass_success_tracking_transfer() -> dict:
    """Naive cross-source transfer for pass-success with tracking context."""
    return _run_transfer_for_loader(
        load_pass_success_tracking_context_dataset,
        "pass_success_tracking_context_classification",
        "pass_success_tracking",
    )


# ---------------------------------------------------------------------------
# Phase 3a: pretrain-finetune protocol
# ---------------------------------------------------------------------------
#
# Tests whether sequential transfer learning beats joint training. The
# joint-training merged_same_n regime mixes target and other rows in one
# fit step; pretrain-finetune fits on `other_pool` first (potentially with
# a much larger budget than N), then continues fitting on N target rows
# only via warm-start. RandomForest cannot do this naturally; sklearn
# MLPClassifier supports `warm_start=True` and is a sufficient minimal
# model for the comparison.
#
# Two finetune regimes are added on top of the existing four:
#   pretrain_other_finetune_target — pretrain on `other_pool` (capped at
#       OTHER_PRETRAIN_BUDGET), then finetune on N target rows.
#   target_only_mlp                — same model class trained only on N
#       target rows. Provides the within-source MLP baseline.
#
# These are reported in a *separate* report file with the suffix
# "_pretrain_finetune" so they don't clutter the main RF/LGBM tables.

PRETRAIN_FINETUNE_REGIMES = [
    "target_only_mlp",
    "pretrain_other_finetune_target",
]
OTHER_PRETRAIN_BUDGET = 4_000


def _fit_pretrain_finetune_classifier(
    pretrain_df: pd.DataFrame,
    finetune_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    seed: int,
    pretrain_max_iter: int = 200,
    finetune_max_iter: int = 200,
) -> Pipeline:
    """Fit a warm-start MLPClassifier: full fit on pretrain, then warm-start
    fit on finetune. The Pipeline imputer/scaler are fit on pretrain only —
    finetune reuses the fitted preprocessor.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    pre_imputer = SimpleImputer(strategy="median").fit(pretrain_df[feature_cols])
    pre_scaler = StandardScaler().fit(pre_imputer.transform(pretrain_df[feature_cols]))

    le = LabelEncoder().fit(
        pd.concat([pretrain_df[label_col], finetune_df[label_col]], ignore_index=True)
    )

    Xp = pre_scaler.transform(pre_imputer.transform(pretrain_df[feature_cols]))
    yp = le.transform(pretrain_df[label_col])
    Xf = pre_scaler.transform(pre_imputer.transform(finetune_df[feature_cols]))
    yf = le.transform(finetune_df[label_col])

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=False,
        random_state=seed,
    )
    classes = np.arange(len(le.classes_))
    # `MLPClassifier.fit(..., warm_start=True)` requires every subsequent
    # `y` to contain the same class set. Small target finetune samples often
    # miss a role class, so use incremental updates with the full class list.
    clf.partial_fit(Xp, yp, classes=classes)
    for _ in range(max(pretrain_max_iter - 1, 0)):
        clf.partial_fit(Xp, yp, classes=classes)
    for _ in range(finetune_max_iter):
        clf.partial_fit(Xf, yf, classes=classes)

    pipeline = Pipeline([
        ("imputer", pre_imputer),
        ("scaler", pre_scaler),
        ("clf", _PretrainedClassifierAdapter(clf, le)),
    ])
    return pipeline


class _PretrainedClassifierAdapter:
    """Thin adapter that exposes the MLPClassifier through the Pipeline API.

    Stores the fitted MLP plus the label encoder used during pretrain, so
    `predict` returns labels in the original space rather than encoded ints.
    Implements only the methods the eval harness needs.
    """

    def __init__(self, clf, le):
        self.clf = clf
        self.le = le

    def fit(self, X, y):  # noqa: D401 - never called; pre-fitted
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.clf.predict(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def _run_pretrain_finetune_classification(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    train_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
    cv_scheme: str = "shuffle",
    other_pretrain_budget: int = OTHER_PRETRAIN_BUDGET,
) -> dict:
    """Per-target evaluation of pretrain-finetune vs. target-only MLP."""
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, label_col, seeds, test_size, cv_scheme
        ):
            pretrain_n = min(other_pretrain_budget, len(other_df))
            pretrain_df = other_df.sample(n=pretrain_n, random_state=sample_seed)
            for n in train_sizes:
                if len(target_train_pool) < n:
                    continue
                finetune_df = target_train_pool.sample(n=n, random_state=sample_seed)
                # 1. target_only_mlp baseline (same model class, no pretrain)
                base_clf = _model(sample_seed, kind="mlp")
                base_clf.fit(finetune_df[feature_cols], finetune_df[label_col])
                base_pred = base_clf.predict(test_df[feature_cols])
                rows.append({
                    "seed": sample_seed,
                    "fold_id": fold_id,
                    "target_source": target_source,
                    "regime": "target_only_mlp",
                    "train_size": n,
                    "n_train": len(finetune_df),
                    "n_pretrain": 0,
                    "n_test": len(test_df),
                    **_evaluate_predictions(test_df[label_col].to_numpy(), base_pred),
                })
                # 2. pretrain on other, then finetune on target
                pf_clf = _fit_pretrain_finetune_classifier(
                    pretrain_df=pretrain_df,
                    finetune_df=finetune_df,
                    feature_cols=feature_cols,
                    label_col=label_col,
                    seed=sample_seed,
                )
                pf_pred = pf_clf.predict(test_df[feature_cols])
                rows.append({
                    "seed": sample_seed,
                    "fold_id": fold_id,
                    "target_source": target_source,
                    "regime": "pretrain_other_finetune_target",
                    "train_size": n,
                    "n_train": len(finetune_df),
                    "n_pretrain": pretrain_n,
                    "n_test": len(test_df),
                    **_evaluate_predictions(test_df[label_col].to_numpy(), pf_pred),
                })

    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target_source", "regime", "train_size"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    # Same-N pair test: pretrain_other_finetune_target vs. target_only_mlp
    pair_key = "fold_id" if "fold_id" in measurements.columns else "seed"
    base = measurements[measurements["regime"] == "target_only_mlp"][
        [pair_key, "target_source", "train_size", "macro_f1", "balanced_accuracy"]
    ].rename(columns={"macro_f1": "f1_base", "balanced_accuracy": "ba_base"})
    pf = measurements[measurements["regime"] == "pretrain_other_finetune_target"][
        [pair_key, "target_source", "train_size", "macro_f1", "balanced_accuracy"]
    ].rename(columns={"macro_f1": "f1_pf", "balanced_accuracy": "ba_pf"})
    pairs = base.merge(pf, on=[pair_key, "target_source", "train_size"])
    same_n = []
    if not pairs.empty:
        pairs["delta_macro_f1"] = pairs["f1_pf"] - pairs["f1_base"]
        pairs["delta_balanced_accuracy"] = pairs["ba_pf"] - pairs["ba_base"]
        for (target_source, train_size), grp in pairs.groupby(["target_source", "train_size"]):
            f1_summary = _paired_mean_summary(grp["delta_macro_f1"])
            ba_summary = _paired_mean_summary(grp["delta_balanced_accuracy"])
            same_n.append({
                "target_source": target_source,
                "train_size": int(train_size),
                "compare_regime": "pretrain_vs_target_only_mlp",
                "delta_macro_f1_mean": f1_summary["mean"],
                "delta_macro_f1_ci95_bootstrap_low": f1_summary["ci95_bootstrap_low"],
                "delta_macro_f1_ci95_bootstrap_high": f1_summary["ci95_bootstrap_high"],
                "delta_macro_f1_wilcoxon_p": f1_summary["wilcoxon_p"],
                "delta_balanced_accuracy_mean": ba_summary["mean"],
                "delta_balanced_accuracy_wilcoxon_p": ba_summary["wilcoxon_p"],
                "n_pairs": f1_summary["n_pairs"],
            })
    report = {
        "task": task_name,
        "evaluation_design": (
            "pretrain on other source (budget capped at "
            f"{other_pretrain_budget}), then warm-start finetune on N target "
            "rows; compared to target-only MLP at the same N"
        ),
        "model_kind": "mlp",
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "label": label_col,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "regimes": PRETRAIN_FINETUNE_REGIMES,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
    }
    save_report(report, REPORTS_DIR / f"{output_prefix}_pretrain_finetune.json")
    return report


def run_player_aggregate_position_pretrain_finetune(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    cv_scheme: str = "shuffle",
) -> dict:
    return _run_pretrain_finetune_classification(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        "player_aggregate_position",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds,
        cv_scheme=cv_scheme,
    )


def run_pass_success_event_pretrain_finetune(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    cv_scheme: str = "shuffle",
) -> dict:
    return _run_pretrain_finetune_classification(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        "pass_success_event",
        train_sizes if train_sizes is not None else PASS_SUCCESS_TRAIN_SIZES,
        seeds=seeds,
        cv_scheme=cv_scheme,
    )


# ---------------------------------------------------------------------------
# Phase 3b: data-acquisition curve (N_target × N_other heatmap)
# ---------------------------------------------------------------------------
#
# Sweeps N_target and N_other independently. The output heatmap answers
# "given M target rows, how many other-source rows do I need for a given
# performance level?" — the practitioner-facing form of the merge question.
# The diagonal `N_target = N_other` recovers a slice through the per-target
# augmentation table; the row N_other = 0 recovers the single-source
# learning curve.

def _run_data_acquisition_classification(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    target_sizes: list[int],
    other_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, label_col, seeds, test_size, cv_scheme
        ):
            for nt in target_sizes:
                if len(target_train_pool) < nt:
                    continue
                target_sub = target_train_pool.sample(n=nt, random_state=sample_seed)
                for no in other_sizes:
                    if no > 0 and len(other_df) < no:
                        continue
                    if no == 0:
                        train_df = target_sub
                    else:
                        other_sub = other_df.sample(n=no, random_state=sample_seed + 100)
                        train_df = pd.concat([target_sub, other_sub], ignore_index=True)
                    clf = _model(sample_seed, kind=model_kind)
                    clf.fit(train_df[feature_cols], train_df[label_col])
                    pred = clf.predict(test_df[feature_cols])
                    rows.append({
                        "seed": sample_seed,
                        "fold_id": fold_id,
                        "target_source": target_source,
                        "n_target": nt,
                        "n_other": no,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        **_evaluate_predictions(test_df[label_col].to_numpy(), pred),
                    })
    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target_source", "n_target", "n_other"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    report = {
        "task": task_name,
        "evaluation_design": (
            "data acquisition curve: sweep N_target × N_other independently "
            "on a fixed target-source test set. Row N_other=0 = single-source "
            "learning curve; diagonal N_target=N_other ≈ target_plus_other_2n."
        ),
        "model_kind": model_kind,
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "label": label_col,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "target_sizes": target_sizes,
        "other_sizes": other_sizes,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    save_report(report, REPORTS_DIR / f"{output_prefix}_data_acquisition.json")
    _plot_data_acquisition_heatmap(summary, PLOTS_DIR / f"{output_prefix}_data_acquisition.png")
    return report


def run_player_aggregate_position_data_acquisition(
    seeds: list[int] | None = None, cv_scheme: str = "shuffle", model_kind: str = "rf"
) -> dict:
    return _run_data_acquisition_classification(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        "player_aggregate_position",
        target_sizes=[25, 50, 100, 150],
        other_sizes=[0, 50, 100, 200, 300],
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


def run_pass_success_event_data_acquisition(
    seeds: list[int] | None = None, cv_scheme: str = "shuffle", model_kind: str = "rf"
) -> dict:
    return _run_data_acquisition_classification(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        "pass_success_event",
        target_sizes=[100, 250, 500, 1000, 2000],
        other_sizes=[0, 250, 500, 1000, 2000, 4000],
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


# ---------------------------------------------------------------------------
# Phase 4a: position task with tracking-context features
# ---------------------------------------------------------------------------


def run_player_aggregate_position_with_tracking_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_with_tracking",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    return _run_target_augmentation_classification(
        load_player_aggregate_position_with_tracking_context_dataset,
        "player_aggregate_position_with_tracking_classification",
        f"player_aggregate_position{output_suffix}",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


# ---------------------------------------------------------------------------
# Phase 4b: importance-weighted joint training
# ---------------------------------------------------------------------------
#
# In merged_same_n we mix N/2 target + N/2 other rows with equal weight in
# the model fit. Importance weighting estimates the density ratio
# p_target(x) / p_other(x) on the feature distribution and weights other
# rows by that ratio (clipped). Other rows that "look like" the target
# distribution get higher weight; outliers in other-source feature space
# get downweighted. RF supports sample_weight natively.

def _density_ratio_weights(
    target_X: np.ndarray, other_X: np.ndarray, clip: tuple[float, float] = (0.05, 20.0),
) -> np.ndarray:
    """Estimate target_density / other_density via a probabilistic source
    classifier. Implementation: train a logistic regression on (other=0,
    target=1), then take p(target|x) / p(other|x) for each other-row.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS

    Xall = np.vstack([target_X, other_X])
    Xall = np.nan_to_num(Xall, nan=0.0)
    yall = np.array([1] * len(target_X) + [0] * len(other_X))
    scaler = SS().fit(Xall)
    Xs = scaler.transform(Xall)
    lr = LogisticRegression(max_iter=500, C=1.0).fit(Xs, yall)
    other_Xs = scaler.transform(np.nan_to_num(other_X, nan=0.0))
    p_target = lr.predict_proba(other_Xs)[:, 1]
    p_other = 1.0 - p_target
    ratio = p_target / np.clip(p_other, 1e-6, 1.0)
    return np.clip(ratio, clip[0], clip[1])


def _run_importance_weighted_classification(
    loader: Callable[[], tuple[pd.DataFrame, list[str], str]],
    task_name: str,
    output_prefix: str,
    train_sizes: list[int],
    seeds: list[int] | None = None,
    test_size: float = 0.30,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    """Same as merged_same_n, but weight other-source rows by their
    estimated density ratio to the target distribution.
    """
    seeds = seeds or SEEDS
    df, feature_cols, label_col = loader()
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, label_col, seeds, test_size, cv_scheme
        ):
            for n in train_sizes:
                half = n // 2
                rest = n - half
                if len(target_train_pool) < half or len(other_df) < rest:
                    continue
                t_sub = target_train_pool.sample(n=half, random_state=sample_seed)
                o_sub = other_df.sample(n=rest, random_state=sample_seed + 100)
                # Compute density-ratio weights on the feature matrix.
                target_X = SimpleImputer(strategy="median").fit_transform(t_sub[feature_cols])
                other_X = SimpleImputer(strategy="median").fit_transform(o_sub[feature_cols])
                w_other = _density_ratio_weights(target_X, other_X)
                weights = np.concatenate([np.ones(len(t_sub)), w_other])
                train_df = pd.concat([t_sub, o_sub], ignore_index=True)
                clf = _model(sample_seed, kind=model_kind)
                # Pipeline.fit accepts step-specific kwargs via "stepname__param".
                fit_kwargs = {f"{clf.steps[-1][0]}__sample_weight": weights}
                clf.fit(train_df[feature_cols], train_df[label_col], **fit_kwargs)
                pred = clf.predict(test_df[feature_cols])
                rows.append({
                    "seed": sample_seed,
                    "fold_id": fold_id,
                    "target_source": target_source,
                    "regime": "merged_same_n_iw",
                    "train_size": n,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "weight_mean_other": float(np.mean(w_other)),
                    "weight_std_other": float(np.std(w_other)),
                    **_evaluate_predictions(test_df[label_col].to_numpy(), pred),
                })
    measurements = pd.DataFrame(rows)
    summary = (
        measurements.groupby(["target_source", "regime", "train_size"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci95=("macro_f1", _ci95),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_ci95=("balanced_accuracy", _ci95),
            n_runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    report = {
        "task": task_name,
        "evaluation_design": (
            "importance-weighted joint training: merged_same_n with other-source "
            "rows weighted by p_target(x)/p_other(x) (LR-based density ratio, "
            "clipped to [0.05, 20]). RF/LGBM accept sample_weight natively."
        ),
        "model_kind": model_kind,
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "label": label_col,
        "n_rows": int(len(df)),
        "feature_cols": feature_cols,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    save_report(report, REPORTS_DIR / f"{output_prefix}_importance_weighted.json")
    return report


def run_player_aggregate_position_importance_weighted(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    return _run_importance_weighted_classification(
        load_player_aggregate_position_dataset,
        "player_aggregate_position_group_classification",
        "player_aggregate_position",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


def run_pass_success_event_importance_weighted(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    return _run_importance_weighted_classification(
        load_pass_success_event_dataset,
        "pass_success_event_classification",
        "pass_success_event",
        train_sizes if train_sizes is not None else PASS_SUCCESS_TRAIN_SIZES,
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


# ---------------------------------------------------------------------------
# Phase 5a: shot-outcome (binary on-target / scoring) task
# ---------------------------------------------------------------------------
#
# Small-N regime where merging is most likely to help. Restrict to
# action_type ∈ {shot, shot_freekick, shot_penalty} (cross_dataset_comparable=True).
# Label is binary "successful shot" (DFL: SuccessfulShot child, SC:
# lead_to_goal). Both sources mark this with the SPADL `result == 'success'`.

SHOT_TRAIN_SIZES = [50, 100, 150, 200]


def run_shot_success_event_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    return _run_target_augmentation_classification(
        load_shot_success_event_dataset,
        "shot_success_event_classification",
        f"shot_success_event{output_suffix}",
        train_sizes if train_sizes is not None else SHOT_TRAIN_SIZES,
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )


def run_shot_success_event_learning_curves() -> dict:
    return _run_learning_curves_for_loader(
        load_shot_success_event_dataset,
        "shot_success_event_classification",
        "shot_success_event",
        SHOT_TRAIN_SIZES,
    )


def run_shot_success_event_transfer() -> dict:
    return _run_transfer_for_loader(
        load_shot_success_event_dataset,
        "shot_success_event_classification",
        "shot_success_event",
    )


# ---------------------------------------------------------------------------
# Phase 5b: SC visibility-corrected kinematic regression
# ---------------------------------------------------------------------------
#
# Restrict the rate-target denominator to *continuously-tracked* segments
# longer than 5 s, rather than counting all visible frames as observed
# minutes. Removes part of the SC broadcast-CV bias toward ball-near
# (and therefore high-activity) segments.


def run_kinematic_regression_visibility_corrected_target_augmentation(
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    output_suffix: str = "_visibility_corrected",
    cv_scheme: str = "shuffle",
    model_kind: str = "rf",
) -> dict:
    return _run_target_augmentation_regression(
        load_kinematic_regression_visibility_corrected_dataset,
        "player_match_kinematic_regression_visibility_corrected",
        f"kinematic_regression{output_suffix}",
        train_sizes if train_sizes is not None else [50, 100, 150],
        seeds=seeds, cv_scheme=cv_scheme, model_kind=model_kind,
    )
