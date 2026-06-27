"""Plotting helpers for downstream harmonization reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _plot_learning_curves(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"dfl": "#486B8A", "skillcorner": "#8A5A44", "merged": "#4B7F52"}
    for regime, group in summary.groupby("regime"):
        group = group.sort_values("train_size")
        y = group["macro_f1_mean"].to_numpy()
        std = group["macro_f1_std"].fillna(0).to_numpy()
        x = group["train_size"].to_numpy()
        ax.plot(x, y, marker="o", label=regime, color=colors.get(regime))
        ax.fill_between(x, y - std, y + std, alpha=0.18, color=colors.get(regime))
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Macro F1")
    ax.set_title("Downstream learning curves")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_regression_learning_curves(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    targets = summary["target"].unique().tolist()
    fig, axes = plt.subplots(len(targets), 1, figsize=(7, 3.2 * len(targets)), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    colors = {"dfl": "#486B8A", "skillcorner": "#8A5A44", "merged": "#4B7F52"}
    for ax, target in zip(axes, targets):
        sub = summary[summary["target"] == target]
        for regime, group in sub.groupby("regime"):
            group = group.sort_values("train_size")
            x = group["train_size"].to_numpy()
            y = group["mae_mean"].to_numpy()
            ci = group["mae_ci95"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", label=regime, color=colors.get(regime))
            ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=colors.get(regime))
        ax.set_xscale("log")
        ax.set_ylabel("MAE")
        ax.set_title(target)
    axes[-1].set_xlabel("Training samples")
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_regression_transfer(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    targets = summary["target"].unique().tolist()
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, 3.4 * len(targets)))
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        sub = summary[summary["target"] == target].copy()
        labels = sub["experiment"].tolist()
        x = np.arange(len(labels))
        ax.bar(x, sub["mae_mean"], yerr=sub["mae_ci95"].fillna(0), color="#486B8A")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title(target)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_transfer(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = summary["experiment"].tolist()
    x = np.arange(len(labels))
    y = summary["macro_f1_mean"].to_numpy()
    err = summary["macro_f1_std"].fillna(0).to_numpy()
    ax.bar(x, y, yerr=err, color="#486B8A", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Cross-source transfer")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_target_augmentation_classification(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    targets = sorted(summary["target_source"].unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 4.5), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    colors = {
        "target_only": "#486B8A",
        "other_only": "#8A5A44",
        "merged_same_n": "#4B7F52",
        "target_plus_other_2n": "#7B5A8C",
    }
    for ax, target_source in zip(axes, targets):
        sub = summary[summary["target_source"] == target_source]
        for regime, group in sub.groupby("regime"):
            group = group.sort_values("train_size")
            x = group["train_size"].to_numpy()
            y = group["macro_f1_mean"].to_numpy()
            ci = group["macro_f1_ci95"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", label=regime, color=colors.get(regime))
            ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=colors.get(regime))
        ax.set_xlabel("Training samples (target-side budget N)")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Test on {target_source}")
        ax.legend(fontsize=8)
    fig.suptitle("Per-target augmentation - fixed target test set")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_target_augmentation_regression(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    targets = summary["target"].unique().tolist()
    target_sources = sorted(summary["target_source"].unique().tolist())
    fig, axes = plt.subplots(
        len(targets),
        len(target_sources),
        figsize=(6 * len(target_sources), 3.2 * len(targets)),
        sharex=True,
    )
    if len(targets) == 1 and len(target_sources) == 1:
        axes = np.array([[axes]])
    elif len(targets) == 1:
        axes = np.array([axes])
    elif len(target_sources) == 1:
        axes = np.array([[a] for a in axes])
    colors = {
        "target_only": "#486B8A",
        "other_only": "#8A5A44",
        "merged_same_n": "#4B7F52",
        "target_plus_other_2n": "#7B5A8C",
    }
    for i, target in enumerate(targets):
        for j, target_source in enumerate(target_sources):
            ax = axes[i, j]
            sub = summary[
                (summary["target"] == target) & (summary["target_source"] == target_source)
            ]
            for regime, group in sub.groupby("regime"):
                group = group.sort_values("train_size")
                x = group["train_size"].to_numpy()
                y = group["mae_mean"].to_numpy()
                ci = group["mae_ci95"].fillna(0).to_numpy()
                ax.plot(x, y, marker="o", label=regime, color=colors.get(regime))
                ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=colors.get(regime))
            ax.set_title(f"{target} | test on {target_source}")
            ax.set_ylabel("MAE")
            ax.set_xlabel("Training samples (target-side budget N)")
            ax.legend(fontsize=7)
    fig.suptitle("Per-target augmentation - fixed target test set")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_data_acquisition_heatmap(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    target_sources = sorted(summary["target_source"].unique().tolist())
    fig, axes = plt.subplots(1, len(target_sources), figsize=(6.5 * len(target_sources), 4.5))
    if len(target_sources) == 1:
        axes = [axes]
    for ax, target_source in zip(axes, target_sources):
        sub = summary[summary["target_source"] == target_source]
        pivot = sub.pivot(index="n_target", columns="n_other", values="macro_f1_mean")
        im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("N_other (rows from other source)")
        ax.set_ylabel("N_target (rows from target source)")
        ax.set_title(f"Test on {target_source}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                if pd.notna(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color="white" if v < 0.5 else "black",
                        fontsize=7,
                    )
        fig.colorbar(im, ax=ax, label="macro F1")
    fig.suptitle("Data-acquisition curve - fixed target test set")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_forest_plot_summary(
    report_paths: list[Path],
    output_path: Path,
    metric: str = "delta_macro_f1",
    compare_regime: str = "merged_same_n",
) -> dict:
    """Read per-target reports and emit a forest plot summary."""
    import matplotlib.pyplot as plt

    rows = []
    for path in report_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        task = d.get("task", path.stem)
        for entry in d.get("same_n_comparisons", []):
            if entry.get("compare_regime") != compare_regime:
                continue
            mean = entry.get(f"{metric}_mean")
            lo = entry.get(f"{metric}_ci95_bootstrap_low")
            hi = entry.get(f"{metric}_ci95_bootstrap_high")
            p = entry.get(f"{metric}_wilcoxon_p")
            rows.append(
                {
                    "task": task,
                    "target_source": entry["target_source"],
                    "train_size": entry["train_size"],
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "p": p,
                    "label": f'{task} | {entry["target_source"]} | N={entry["train_size"]}',
                }
            )

    if not rows:
        return {"n_rows": 0, "output_path": str(output_path)}

    rows = sorted(rows, key=lambda r: (r["task"], r["target_source"], r["train_size"]))
    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    los = [r["lo"] if r["lo"] is not None and not np.isnan(r["lo"]) else r["mean"] for r in rows]
    his = [r["hi"] if r["hi"] is not None and not np.isnan(r["hi"]) else r["mean"] for r in rows]
    sig = [r["p"] is not None and not np.isnan(r["p"]) and r["p"] < 0.05 for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(rows))))
    y = np.arange(len(rows))
    ax.errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
        fmt="o",
        color="#486B8A",
        ecolor="#888",
        capsize=3,
    )
    for i, is_sig in enumerate(sig):
        if is_sig:
            ax.scatter(
                means[i],
                y[i],
                color="#B8473A",
                s=40,
                zorder=3,
                label="p<0.05 (Wilcoxon)" if i == sig.index(True) else None,
            )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"{metric} ({compare_regime} - target_only) - bootstrap 95 % CI")
    ax.set_title(f"Per-target merge effect ({compare_regime})")
    if any(sig):
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return {"n_rows": len(rows), "output_path": str(output_path)}
