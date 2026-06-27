"""Self-supervised tracking representations and linear-probe evaluation.

This module intentionally lives beside, rather than inside, ``downstream.py``:
other evaluation work can keep moving while this representation experiment
reuses the established per-target split and same-N report helpers.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.evaluation.downstream.comparisons import _same_n_target_classification_comparisons
from src.evaluation.datasets import RANDOM_SEED, _label_to_bucket, _metadata_position_lookup
from src.evaluation.metrics import _evaluate_predictions
from src.evaluation.downstream.regimes import PER_TARGET_REGIMES, _iter_target_splits, _sample_train_per_target
from src.evaluation.serialization import save_report
from src.evaluation.stats import _ci95

DATA_PATH = Path(settings.data_path)
MATCHES_DIR = DATA_PATH / "merged" / "matches"
CACHE_DIR = DATA_PATH / "cache"
REPORTS_DIR = DATA_PATH / "reports"
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

SNIPPET_LENGTH = 50
SNIPPET_FEATURES = ["x", "y", "speed_kmh_filtered", "acceleration_ms2_filtered"]
EMBEDDING_DIM = 64


@dataclass(frozen=True)
class SnippetBuildConfig:
    output_path: Path = CACHE_DIR / "snippets.npz"
    snippet_length: int = SNIPPET_LENGTH
    min_segment_frames: int = 50
    stride: int = 10
    max_snippets: int | None = None
    seed: int = RANDOM_SEED


def _try_import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except Exception:
        return None, None, None, None


def _reservoir_accept(rng: np.random.Generator, seen: int, max_items: int | None) -> int | None:
    if max_items is None:
        return seen
    if seen < max_items:
        return seen
    j = int(rng.integers(0, seen + 1))
    return j if j < max_items else None


def _normalise_source(value: object) -> str:
    text = str(value)
    lower = text.lower()
    if lower in {"dfl", "skillcorner"}:
        return "DFL" if lower == "dfl" else "SkillCorner"
    return text


def _read_tracking_columns(path: Path) -> pd.DataFrame:
    cols = [
        "match_id",
        "source",
        "frame_id",
        "period",
        "player_id",
        "position_label",
        "x",
        "y",
        "speed_kmh_filtered",
        "acceleration_ms2_filtered",
        "is_visible",
    ]
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        required = [c for c in cols if c != "position_label"]
        return pd.read_parquet(path, columns=required)


def build_snippet_cache(config: SnippetBuildConfig | None = None) -> dict:
    """Extract overlapping visible trajectory snippets into a compressed NPZ.

    Rows are split by ``(source, match_id, period, player_id)`` and then by
    frame continuity. Only visible continuous segments of at least 5 seconds
    at 10 Hz are used. The NPZ stores ``X`` as ``float32`` with shape
    ``(n_snippets, 50, 4)`` plus aligned metadata arrays.
    """
    config = config or SnippetBuildConfig()
    rng = np.random.default_rng(config.seed)
    lookup = _metadata_position_lookup()
    paths = sorted(MATCHES_DIR.glob("*/*_tracking_10hz.parquet"))
    snippets: list[np.ndarray] = []
    meta_source: list[str] = []
    meta_match: list[str] = []
    meta_player: list[str] = []
    meta_period: list[str] = []
    meta_position: list[str] = []
    meta_start_frame: list[int] = []
    seen = 0
    n_segments = 0

    for path in paths:
        df = _read_tracking_columns(path)
        df = df[(df["is_visible"] == True) & df["player_id"].notna()].copy()
        df = df.dropna(subset=["x", "y", "speed_kmh_filtered", "acceleration_ms2_filtered"])
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
        df["source"] = df["source"].map(_normalise_source)
        if "position_label" in df.columns:
            df["position_group"] = df["position_label"].map(_label_to_bucket)
        else:
            df["position_group"] = None
        missing = df["position_group"].isna()
        if missing.any():
            keys = list(zip(df.loc[missing, "match_id"], df.loc[missing, "player_id"]))
            df.loc[missing, "position_group"] = [lookup.get(k) for k in keys]

        sort_cols = ["source", "match_id", "period", "player_id", "frame_id"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        group_cols = ["source", "match_id", "period", "player_id"]
        for (source, match_id, period, player_id), group in df.groupby(group_cols, sort=False):
            frame_diff = group["frame_id"].diff().fillna(1)
            seg_id = (frame_diff != 1).cumsum()
            for _, segment in group.groupby(seg_id, sort=False):
                if len(segment) < config.min_segment_frames:
                    continue
                n_segments += 1
                values = segment[SNIPPET_FEATURES].to_numpy(dtype=np.float32, copy=True)
                frames = segment["frame_id"].to_numpy()
                position = segment["position_group"].dropna()
                position_group = str(position.iloc[0]) if not position.empty else ""
                for start in range(0, len(segment) - config.snippet_length + 1, config.stride):
                    idx = _reservoir_accept(rng, seen, config.max_snippets)
                    seen += 1
                    if idx is None:
                        continue
                    snippet = values[start : start + config.snippet_length]
                    if idx == len(snippets):
                        snippets.append(snippet)
                        meta_source.append(str(source))
                        meta_match.append(str(match_id))
                        meta_player.append(str(player_id))
                        meta_period.append(str(period))
                        meta_position.append(position_group)
                        meta_start_frame.append(int(frames[start]))
                    else:
                        snippets[idx] = snippet
                        meta_source[idx] = str(source)
                        meta_match[idx] = str(match_id)
                        meta_player[idx] = str(player_id)
                        meta_period[idx] = str(period)
                        meta_position[idx] = position_group
                        meta_start_frame[idx] = int(frames[start])

    if not snippets:
        raise RuntimeError(f"No snippets extracted from {MATCHES_DIR}")

    X = np.stack(snippets).astype(np.float32, copy=False)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        X=X,
        source=np.asarray(meta_source),
        match_id=np.asarray(meta_match),
        player_id=np.asarray(meta_player),
        period=np.asarray(meta_period),
        position_group=np.asarray(meta_position),
        start_frame=np.asarray(meta_start_frame, dtype=np.int64),
        feature_names=np.asarray(SNIPPET_FEATURES),
        snippet_length=np.asarray(config.snippet_length),
        stride=np.asarray(config.stride),
    )
    return {
        "snippet_path": str(config.output_path),
        "n_snippets": int(len(X)),
        "n_seen_before_reservoir": int(seen),
        "n_continuous_segments": int(n_segments),
        "max_snippets": config.max_snippets,
        "stride": config.stride,
    }


def load_snippets(path: Path = CACHE_DIR / "snippets.npz") -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def describe_snippet_cache(path: Path = CACHE_DIR / "snippets.npz") -> dict:
    data = load_snippets(path)
    source, source_counts = np.unique(data["source"].astype(str), return_counts=True)
    return {
        "snippet_path": str(path),
        "n_snippets": int(data["X"].shape[0]),
        "shape": [int(v) for v in data["X"].shape],
        "feature_names": data["feature_names"].astype(str).tolist(),
        "source_counts": {
            str(k): int(v) for k, v in zip(source.tolist(), source_counts.tolist())
        },
        "stride": int(np.asarray(data.get("stride", 0)).item()),
    }


def _standardise_snippets(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0).astype(np.float32)
    std = X.reshape(-1, X.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((X - mean) / std).astype(np.float32), mean, std


def _torch_autoencoder_classes(nn):
    class TrackingEncoder(nn.Module):
        def __init__(self, in_channels: int = 4, embedding_dim: int = EMBEDDING_DIM):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.proj = nn.Linear(64, embedding_dim)

        def forward(self, x):
            x = x.transpose(1, 2)
            h = self.conv(x).mean(dim=-1)
            return self.proj(h)

    class TrackingAutoencoder(nn.Module):
        def __init__(self, snippet_length: int = SNIPPET_LENGTH, n_features: int = 4):
            super().__init__()
            self.encoder = TrackingEncoder(n_features, EMBEDDING_DIM)
            self.decoder = nn.Sequential(
                nn.Linear(EMBEDDING_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, snippet_length * n_features),
            )
            self.snippet_length = snippet_length
            self.n_features = n_features

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out.view(-1, self.snippet_length, self.n_features)

    return TrackingEncoder, TrackingAutoencoder


def pretrain_encoder(
    snippet_path: Path = CACHE_DIR / "snippets.npz",
    model_path: Path = MODELS_DIR / "tracking_encoder.pt",
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    max_train_snippets: int | None = 50_000,
    seed: int = RANDOM_SEED,
    prefer_torch: bool = True,
) -> dict:
    """Pretrain a 64-dim snippet encoder with a reconstruction objective."""
    data = load_snippets(snippet_path)
    X = data["X"].astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(X))
    if max_train_snippets is not None and len(train_idx) > max_train_snippets:
        train_idx = rng.choice(train_idx, size=max_train_snippets, replace=False)
    X_std, mean, std = _standardise_snippets(X)
    X_train = X_std[train_idx]

    torch, nn, DataLoader, TensorDataset = _try_import_torch()
    if prefer_torch and torch is not None:
        torch.manual_seed(seed)
        TrackingEncoder, TrackingAutoencoder = _torch_autoencoder_classes(nn)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrackingAutoencoder(snippet_length=X.shape[1], n_features=X.shape[2]).to(device)
        dataset = TensorDataset(torch.from_numpy(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        history = []
        model.train()
        for epoch in range(epochs):
            losses = []
            for (batch,) in loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "reconstruction_mse": float(np.mean(losses))})
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": "torch",
                "encoder_state_dict": model.encoder.state_dict(),
                "autoencoder_state_dict": model.state_dict(),
                "feature_names": SNIPPET_FEATURES,
                "feature_mean": mean,
                "feature_std": std,
                "snippet_length": int(X.shape[1]),
                "embedding_dim": EMBEDDING_DIM,
                "history": history,
            },
            model_path,
        )
        return {
            "backend": "torch",
            "model_path": str(model_path),
            "device": str(device),
            "epochs": epochs,
            "n_train_snippets": int(len(X_train)),
            "final_reconstruction_mse": history[-1]["reconstruction_mse"],
        }

    flat_train = X_train.reshape(len(X_train), -1)
    scaler = StandardScaler()
    flat_train_s = scaler.fit_transform(flat_train)
    pca = PCA(n_components=min(EMBEDDING_DIM, flat_train_s.shape[1]), random_state=seed)
    z = pca.fit_transform(flat_train_s)
    recon = pca.inverse_transform(z)
    pca_mse = float(mean_squared_error(flat_train_s, recon))
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, EMBEDDING_DIM, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(batch_size, len(flat_train_s)),
        learning_rate_init=learning_rate,
        max_iter=epochs,
        random_state=seed,
        early_stopping=False,
        verbose=False,
    )
    mlp.fit(flat_train_s, flat_train_s)
    payload = {
        "backend": "sklearn_pca_mlp_surrogate",
        "feature_names": SNIPPET_FEATURES,
        "feature_mean": mean,
        "feature_std": std,
        "flat_scaler": scaler,
        "pca": pca,
        "mlp_autoencoder": mlp,
        "snippet_length": int(X.shape[1]),
        "embedding_dim": int(pca.n_components_),
        "pca_reconstruction_mse": pca_mse,
        "mlp_loss": float(mlp.loss_),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    return {
        "backend": "sklearn_pca_mlp_surrogate",
        "model_path": str(model_path),
        "epochs": epochs,
        "n_train_snippets": int(len(X_train)),
        "final_reconstruction_mse": float(mlp.loss_),
        "pca_reconstruction_mse": pca_mse,
        "note": "Torch unavailable; PCA embeddings plus an MLP reconstruction surrogate were saved.",
    }


def _load_torch_encoder(model_path: Path):
    torch, nn, _, _ = _try_import_torch()
    if torch is None:
        return None, None, None
    # This checkpoint is produced locally by ``pretrain_encoder`` and includes
    # numpy scaler arrays alongside tensors, so PyTorch >=2.6 needs explicit
    # trusted loading instead of the stricter weights-only default.
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if ckpt.get("backend") != "torch":
        return None, None, ckpt
    TrackingEncoder, _ = _torch_autoencoder_classes(nn)
    encoder = TrackingEncoder(4, int(ckpt.get("embedding_dim", EMBEDDING_DIM)))
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    return torch, encoder, ckpt


def encode_snippets(
    snippet_path: Path = CACHE_DIR / "snippets.npz",
    model_path: Path = MODELS_DIR / "tracking_encoder.pt",
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Return one mean-pooled embedding row per ``source/match/player``."""
    data = load_snippets(snippet_path)
    X = data["X"].astype(np.float32, copy=False)
    torch, encoder, ckpt = _load_torch_encoder(model_path)
    if encoder is not None:
        mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        X_std = ((X - mean) / std).astype(np.float32)
        embs = []
        with torch.no_grad():
            for start in range(0, len(X_std), batch_size):
                batch = torch.from_numpy(X_std[start : start + batch_size])
                embs.append(encoder(batch).cpu().numpy())
        Z = np.vstack(embs).astype(np.float32)
        backend = "torch"
    else:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        flat = X.reshape(len(X), -1)
        flat_s = payload["flat_scaler"].transform(flat)
        Z = payload["pca"].transform(flat_s).astype(np.float32)
        backend = payload.get("backend", "sklearn_pca_mlp_surrogate")

    meta = pd.DataFrame(
        {
            "source": data["source"].astype(str),
            "match_id": data["match_id"].astype(str),
            "player_id": data["player_id"].astype(str),
            "position_group": data["position_group"].astype(str),
        }
    )
    emb_cols = [f"emb_{i:03d}" for i in range(Z.shape[1])]
    emb = pd.DataFrame(Z, columns=emb_cols)
    pooled = pd.concat([meta, emb], axis=1)
    pooled = pooled[pooled["position_group"].notna() & (pooled["position_group"] != "")]
    pooled = pooled[pooled["position_group"] != "GK"].copy()
    grouped = pooled.groupby(["source", "match_id", "player_id", "position_group"], dropna=False)
    out = grouped[emb_cols].mean().reset_index()
    out.attrs["encoder_backend"] = backend
    return out


def _linear_probe_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _rf_baseline_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _run_probe_measurements(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    train_sizes: list[int],
    seeds: list[int],
    cv_scheme: str,
    model_factory,
) -> pd.DataFrame:
    rows = []
    for target_source in ["DFL", "SkillCorner"]:
        other_source = "SkillCorner" if target_source == "DFL" else "DFL"
        target_df = df[df["source"] == target_source].copy()
        other_df = df[df["source"] == other_source].copy()
        if target_df.empty or other_df.empty:
            continue
        for sample_seed, fold_id, target_train_pool, test_df in _iter_target_splits(
            target_df, label_col, seeds, 0.30, cv_scheme
        ):
            for n in train_sizes:
                for regime in PER_TARGET_REGIMES:
                    train_df = _sample_train_per_target(
                        target_train_pool, other_df, regime, n, sample_seed
                    )
                    if train_df is None:
                        continue
                    model = model_factory(sample_seed)
                    model.fit(train_df[feature_cols], train_df[label_col])
                    pred = model.predict(test_df[feature_cols])
                    rows.append(
                        {
                            "seed": int(sample_seed),
                            "fold_id": str(fold_id),
                            "target_source": target_source,
                            "regime": regime,
                            "train_size": int(n),
                            "n_train": int(len(train_df)),
                            "n_test": int(len(test_df)),
                            "test_matches": sorted(test_df["match_id"].astype(str).unique().tolist()),
                            **_evaluate_predictions(test_df[label_col].to_numpy(), pred),
                        }
                    )
    return pd.DataFrame(rows)


def _summarise_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    return (
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


def _headline_table(
    linear_summary: pd.DataFrame,
    same_n: list[dict],
    rf_summary: pd.DataFrame | None,
    rf_same_n: list[dict] | None,
) -> list[dict]:
    rows = []
    compare_regimes = ["merged_same_n", "target_plus_other_2n"]
    linear_same = [row for row in same_n if row.get("compare_regime") in compare_regimes]
    rf_lookup = {}
    if rf_same_n:
        rf_lookup = {
            (row["target_source"], int(row["train_size"]), row.get("compare_regime")): row
            for row in rf_same_n
            if row.get("compare_regime") in compare_regimes
        }
    for row in linear_same:
        target = row["target_source"]
        n = int(row["train_size"])
        compare_regime = row["compare_regime"]
        target_only = linear_summary[
            (linear_summary["target_source"] == target)
            & (linear_summary["train_size"] == n)
            & (linear_summary["regime"] == "target_only")
        ]
        compare = linear_summary[
            (linear_summary["target_source"] == target)
            & (linear_summary["train_size"] == n)
            & (linear_summary["regime"] == compare_regime)
        ]
        out = {
            "target_source": target,
            "train_size": n,
            "compare_regime": compare_regime,
            "linear_probe_target_only_macro_f1": float(target_only["macro_f1_mean"].iloc[0])
            if not target_only.empty
            else float("nan"),
            "linear_probe_compare_macro_f1": float(compare["macro_f1_mean"].iloc[0])
            if not compare.empty
            else float("nan"),
            "linear_probe_delta_macro_f1": row["delta_macro_f1_mean"],
            "linear_probe_wilcoxon_p": row["delta_macro_f1_wilcoxon_p"],
            "n_pairs": row.get("n_pairs"),
        }
        if rf_summary is not None:
            rf_target = rf_summary[
                (rf_summary["target_source"] == target)
                & (rf_summary["train_size"] == n)
                & (rf_summary["regime"] == "target_only")
            ]
            rf_compare = rf_summary[
                (rf_summary["target_source"] == target)
                & (rf_summary["train_size"] == n)
                & (rf_summary["regime"] == compare_regime)
            ]
            rf_same = rf_lookup.get((target, n, compare_regime), {})
            out.update(
                {
                    "rf_target_only_macro_f1": float(rf_target["macro_f1_mean"].iloc[0])
                    if not rf_target.empty
                    else float("nan"),
                    "rf_compare_macro_f1": float(rf_compare["macro_f1_mean"].iloc[0])
                    if not rf_compare.empty
                    else float("nan"),
                    "rf_delta_macro_f1": rf_same.get("delta_macro_f1_mean", float("nan")),
                    "rf_wilcoxon_p": rf_same.get("delta_macro_f1_wilcoxon_p", float("nan")),
                }
            )
        rows.append(out)
    return rows


def _plot_probe(summary: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if summary.empty:
        return
    regimes = ["target_only", "merged_same_n", "target_plus_other_2n", "other_only"]
    colors = {
        "target_only": "#486B8A",
        "merged_same_n": "#4B7F52",
        "target_plus_other_2n": "#B9823A",
        "other_only": "#8A4B56",
    }
    targets = sorted(summary["target_source"].unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        sub = summary[summary["target_source"] == target]
        for regime in regimes:
            r = sub[sub["regime"] == regime].sort_values("train_size")
            if r.empty:
                continue
            ax.errorbar(
                r["train_size"],
                r["macro_f1_mean"],
                yerr=r["macro_f1_ci95"].fillna(0),
                marker="o",
                capsize=3,
                label=regime,
                color=colors.get(regime),
            )
        ax.set_title(target)
        ax.set_xlabel("Train size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Representation Linear Probe: Position Classification")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _with_output_suffix(path: Path, output_suffix: str) -> Path:
    if not output_suffix:
        return path
    return path.with_name(f"{path.stem}{output_suffix}{path.suffix}")


def _load_rf_baseline_from_report(path: Path) -> tuple[pd.DataFrame | None, list[dict] | None]:
    if not path.exists():
        return None, None
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return pd.DataFrame(report.get("summary", [])), report.get("same_n_comparisons", [])


def run_representation_linear_probe(
    snippet_path: Path = CACHE_DIR / "snippets.npz",
    model_path: Path = MODELS_DIR / "tracking_encoder.pt",
    report_path: Path = REPORTS_DIR / "representation_linear_probe.json",
    plot_path: Path = PLOTS_DIR / "representation_linear_probe.png",
    train_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    cv_scheme: str = "shuffle",
    include_rf_baseline: bool = True,
    rf_baseline_report_path: Path = (
        REPORTS_DIR / "player_aggregate_position_rf_extended_target_augmentation.json"
    ),
    output_suffix: str = "",
) -> dict:
    report_path = _with_output_suffix(report_path, output_suffix)
    plot_path = _with_output_suffix(plot_path, output_suffix)
    train_sizes = train_sizes or [50, 100, 150]
    seeds = seeds or [RANDOM_SEED, 43, 44, 45, 46]
    df = encode_snippets(snippet_path, model_path)
    feature_cols = [c for c in df.columns if c.startswith("emb_")]
    label_col = "position_group"
    measurements = _run_probe_measurements(
        df, feature_cols, label_col, train_sizes, seeds, cv_scheme, _linear_probe_model
    )
    summary = _summarise_measurements(measurements)
    same_n = _same_n_target_classification_comparisons(measurements)

    rf_summary = None
    rf_same_n = None
    if include_rf_baseline:
        rf_summary, rf_same_n = _load_rf_baseline_from_report(rf_baseline_report_path)
        if rf_summary is None:
            rf_measurements = _run_probe_measurements(
                df, feature_cols, label_col, train_sizes, seeds, cv_scheme, _rf_baseline_model
            )
            rf_summary = _summarise_measurements(rf_measurements)
            rf_same_n = _same_n_target_classification_comparisons(rf_measurements)

    report = {
        "task": "self_supervised_tracking_representation_linear_probe_position_classification",
        "evaluation_design": (
            "50-frame visible trajectory snippets; encoder frozen; snippets mean-pooled "
            "per player-match; multinomial logistic-regression linear probe with the "
            "existing per-target augmentation regimes."
        ),
        "snippet_cache": describe_snippet_cache(snippet_path),
        "encoder_backend": df.attrs.get("encoder_backend"),
        "snippet_path": str(snippet_path),
        "model_path": str(model_path),
        "report_path": str(report_path),
        "plot_path": str(plot_path),
        "cv_scheme": cv_scheme,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df[label_col].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "regimes": PER_TARGET_REGIMES,
        "measurements": measurements.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "same_n_comparisons": same_n,
        "rf_baseline_summary": rf_summary.to_dict(orient="records") if rf_summary is not None else [],
        "rf_baseline_same_n_comparisons": rf_same_n or [],
        "headline_table": _headline_table(summary, same_n, rf_summary, rf_same_n),
    }
    save_report(report, report_path)
    _plot_probe(summary, plot_path)
    return report


def run_full_representation_experiment(args: argparse.Namespace) -> dict:
    build_info = None
    if args.rebuild_snippets or not args.snippet_path.exists():
        build_info = build_snippet_cache(
            SnippetBuildConfig(
                output_path=args.snippet_path,
                stride=args.stride,
                max_snippets=args.max_snippets,
                seed=args.seed,
            )
        )
    pretrain_info = None
    if args.retrain_encoder or not args.model_path.exists():
        pretrain_info = pretrain_encoder(
            snippet_path=args.snippet_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_snippets=args.max_train_snippets,
            seed=args.seed,
            prefer_torch=not args.no_torch,
        )
    report = run_representation_linear_probe(
        snippet_path=args.snippet_path,
        model_path=args.model_path,
        report_path=args.report_path,
        plot_path=args.plot_path,
        train_sizes=args.train_sizes,
        seeds=args.seeds,
        cv_scheme=args.cv_scheme,
        include_rf_baseline=not args.no_rf_baseline,
        output_suffix=args.output_suffix,
    )
    report["snippet_build"] = build_info
    report["pretraining"] = pretrain_info
    save_report(report, Path(report.get("report_path", args.report_path)))
    return report


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--snippet-path", type=Path, default=CACHE_DIR / "snippets.npz")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "tracking_encoder.pt")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "representation_linear_probe.json",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=PLOTS_DIR / "representation_linear_probe.png",
    )
    parser.add_argument("--rebuild-snippets", action="store_true")
    parser.add_argument("--retrain-encoder", action="store_true")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-snippets", type=int, default=None)
    parser.add_argument("--max-train-snippets", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50, 100, 150])
    parser.add_argument("--cv-scheme", choices=["shuffle", "lomo"], default="shuffle")
    parser.add_argument("--no-torch", action="store_true")
    parser.add_argument("--no-rf-baseline", action="store_true")
    parser.add_argument("--output-suffix", default="")
    return parser


__all__ = [
    "SnippetBuildConfig",
    "build_snippet_cache",
    "pretrain_encoder",
    "encode_snippets",
    "run_representation_linear_probe",
    "run_full_representation_experiment",
    "add_cli_arguments",
]
