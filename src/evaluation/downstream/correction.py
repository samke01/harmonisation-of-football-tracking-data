"""Train-only statistical correction helpers for downstream sensitivity runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CorrectionResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    applied: bool
    reason: str
    method: str
    reference_batch: str | None


def apply_train_only_correction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    method: str = "none",
    batch_col: str = "source",
    reference_batch: str | None = None,
    random_state: int | None = None,
) -> CorrectionResult:
    """Fit correction parameters on training rows and apply them to train/test.

    ``ls`` aligns non-reference batches to the reference batch's training-set
    mean and standard deviation feature-by-feature. If the train set contains
    only one provider, correction is skipped so target-only and other-only
    regimes remain honest single-source controls.
    """
    method = "none" if method is None else method.lower()
    if method in {"none", ""}:
        return CorrectionResult(train_df, test_df, False, "method_none", "none", reference_batch)
    if method in {"ls", "location_scale"}:
        return _apply_location_scale_correction(
            train_df,
            test_df,
            feature_cols,
            batch_col=batch_col,
            reference_batch=reference_batch,
            permute_batches=False,
            random_state=random_state,
        )
    if method in {"ls_permuted", "ls_perm"}:
        return _apply_location_scale_correction(
            train_df,
            test_df,
            feature_cols,
            batch_col=batch_col,
            reference_batch=reference_batch,
            permute_batches=True,
            random_state=random_state,
        )
    if method in {"combat_p", "combat_parametric"}:
        return _apply_parametric_combat_correction(
            train_df,
            test_df,
            feature_cols,
            batch_col=batch_col,
            permute_batches=False,
            random_state=random_state,
        )
    if method in {"combat_p_permuted", "combat_parametric_permuted"}:
        return _apply_parametric_combat_correction(
            train_df,
            test_df,
            feature_cols,
            batch_col=batch_col,
            permute_batches=True,
            random_state=random_state,
        )
    raise ValueError(f"Unknown correction method: {method}")


def _batch_stats(df: pd.DataFrame, feature_cols: list[str], batch_col: str) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for batch, group in df.groupby(batch_col):
        batch_key = str(batch)
        values = group[feature_cols]
        stats[batch_key] = {
            "mean": values.mean(skipna=True),
            "std": values.std(skipna=True, ddof=0).replace(0.0, np.nan),
        }
    return stats


def _transform_location_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_col: str,
    stats: dict[str, dict],
    ref_mean: pd.Series,
    ref_std: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = out[feature_cols].astype(float)
    safe_ref_std = ref_std.replace(0.0, np.nan).fillna(1.0)
    for batch, group in out.groupby(batch_col):
        batch_key = str(batch)
        if batch_key not in stats:
            continue
        mean = stats[batch_key]["mean"]
        std = stats[batch_key]["std"].fillna(1.0)
        adjusted = (group[feature_cols] - mean) / std
        adjusted = adjusted * safe_ref_std + ref_mean
        out.loc[group.index, feature_cols] = adjusted
    return out


def _apply_location_scale_correction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    batch_col: str,
    reference_batch: str | None,
    permute_batches: bool,
    random_state: int | None,
) -> CorrectionResult:
    batches = sorted(train_df[batch_col].dropna().astype(str).unique().tolist())
    method = "ls_permuted" if permute_batches else "ls"
    if len(batches) < 2:
        return CorrectionResult(
            train_df,
            test_df,
            False,
            "single_train_batch",
            method,
            reference_batch,
        )
    stats_df = train_df.copy()
    if permute_batches:
        rng = np.random.default_rng(random_state)
        stats_df[batch_col] = rng.permutation(stats_df[batch_col].to_numpy())
    stats = _batch_stats(stats_df, feature_cols, batch_col)
    reference_batch = reference_batch or batches[0]
    if reference_batch not in stats:
        return CorrectionResult(
            train_df,
            test_df,
            False,
            "reference_batch_absent_from_train",
            method,
            reference_batch,
        )
    ref_mean = stats[reference_batch]["mean"]
    ref_std = stats[reference_batch]["std"].fillna(1.0)
    corrected_train = _transform_location_scale(
        train_df, feature_cols, batch_col, stats, ref_mean, ref_std
    )
    corrected_test = _transform_location_scale(
        test_df, feature_cols, batch_col, stats, ref_mean, ref_std
    )
    return CorrectionResult(
        corrected_train,
        corrected_test,
        True,
        "applied_permuted_batches" if permute_batches else "applied",
        method,
        reference_batch,
    )


def _inverse_gamma_prior(delta_hat: np.ndarray) -> tuple[float, float]:
    means = float(np.nanmean(delta_hat))
    variances = float(np.nanvar(delta_hat, ddof=1))
    variances = variances if np.isfinite(variances) and variances > 1e-12 else 1e-12
    means = means if np.isfinite(means) and means > 1e-12 else 1.0
    a_prior = (2.0 * variances + means**2) / variances
    b_prior = (means * variances + means**3) / variances
    return float(a_prior), float(b_prior)


def _it_sol(
    s_data: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_bar: float,
    t2: float,
    a_prior: float,
    b_prior: float,
    *,
    conv: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_old = gamma_hat.copy()
    delta_old = np.where(np.isfinite(delta_hat) & (delta_hat > 1e-12), delta_hat, 1.0)
    n_valid = np.sum(np.isfinite(s_data), axis=1).astype(float)
    n_valid = np.maximum(n_valid, 1.0)

    for _ in range(max_iter):
        denom = t2 * n_valid + delta_old
        gamma_new = (t2 * n_valid * gamma_hat + delta_old * gamma_bar) / denom
        centered = s_data - gamma_new[:, None]
        sum_sq = np.nansum(centered**2, axis=1)
        delta_new = (0.5 * sum_sq + b_prior) / (0.5 * n_valid + a_prior - 1.0)
        delta_new = np.where(np.isfinite(delta_new) & (delta_new > 1e-12), delta_new, 1.0)

        change_gamma = np.nanmax(
            np.abs(gamma_new - gamma_old) / np.maximum(np.abs(gamma_old), 1e-8)
        )
        change_delta = np.nanmax(
            np.abs(delta_new - delta_old) / np.maximum(np.abs(delta_old), 1e-8)
        )
        gamma_old, delta_old = gamma_new, delta_new
        if max(change_gamma, change_delta) < conv:
            break
    return gamma_old, delta_old


def _fit_parametric_combat(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    batch_col: str,
) -> dict:
    """Fit a small no-covariate parametric ComBat model on training rows."""
    batches = sorted(train_df[batch_col].dropna().astype(str).unique().tolist())
    if len(batches) < 2:
        raise ValueError("Parametric ComBat needs at least two train batches.")

    X = train_df[feature_cols].astype(float).to_numpy()
    batch_values = train_df[batch_col].astype(str).to_numpy()
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("Parametric ComBat needs non-empty data.")

    site_design = np.column_stack([(batch_values == batch).astype(float) for batch in batches])
    batch_sizes = site_design.sum(axis=0)
    data = X.T
    beta_hat = np.linalg.pinv(site_design.T @ site_design) @ site_design.T @ data.T
    grand_mean = (batch_sizes / float(n_samples)) @ beta_hat
    fitted = (site_design @ beta_hat).T
    var_pooled = ((data - fitted) ** 2) @ (np.ones((n_samples, 1)) / float(n_samples))
    var_pooled = var_pooled.ravel()
    var_pooled = np.where(np.isfinite(var_pooled) & (var_pooled > 1e-12), var_pooled, 1.0)
    pooled_std = np.sqrt(var_pooled)

    standardized_mean = grand_mean[:, None] @ np.ones((1, n_samples))
    s_data = (data - standardized_mean) / (pooled_std[:, None] @ np.ones((1, n_samples)))

    gamma_hat = np.linalg.pinv(site_design.T @ site_design) @ site_design.T @ s_data.T
    delta_hat = [
        np.var(s_data[:, batch_values == batch], axis=1, ddof=1)
        for batch in batches
    ]

    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    t2 = np.where(np.isfinite(t2) & (t2 > 1e-12), t2, 1e-12)

    gamma_star = {}
    delta_star = {}
    for i, batch in enumerate(batches):
        Sb = s_data[:, batch_values == batch]
        a_prior, b_prior = _inverse_gamma_prior(delta_hat[i])
        gamma_i, delta_i = _it_sol(
            Sb,
            gamma_hat[i],
            delta_hat[i],
            float(gamma_bar[i]),
            float(t2[i]),
            a_prior,
            b_prior,
        )
        gamma_star[batch] = gamma_i
        delta_star[batch] = delta_i

    return {
        "batches": batches,
        "grand_mean": grand_mean,
        "pooled_std": pooled_std,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
    }


def _transform_parametric_combat(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_col: str,
    params: dict,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = out[feature_cols].astype(float)
    grand_mean = params["grand_mean"]
    pooled_std = params["pooled_std"]
    for batch, group in out.groupby(batch_col):
        batch_key = str(batch)
        if batch_key not in params["gamma_star"]:
            continue
        gamma = params["gamma_star"][batch_key]
        delta = params["delta_star"][batch_key]
        X = group[feature_cols].astype(float).to_numpy()
        standardized = (X - grand_mean) / pooled_std
        adjusted = (standardized - gamma) / np.sqrt(delta)
        out.loc[group.index, feature_cols] = adjusted * pooled_std + grand_mean
    return out


def _apply_parametric_combat_correction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    batch_col: str,
    permute_batches: bool,
    random_state: int | None,
) -> CorrectionResult:
    batches = sorted(train_df[batch_col].dropna().astype(str).unique().tolist())
    method = "combat_p_permuted" if permute_batches else "combat_p"
    if len(batches) < 2:
        return CorrectionResult(
            train_df,
            test_df,
            False,
            "single_train_batch",
            method,
            None,
        )
    fit_df = train_df.copy()
    if permute_batches:
        rng = np.random.default_rng(random_state)
        fit_df[batch_col] = rng.permutation(fit_df[batch_col].to_numpy())
    params = _fit_parametric_combat(fit_df, feature_cols, batch_col)
    corrected_train = _transform_parametric_combat(train_df, feature_cols, batch_col, params)
    corrected_test = _transform_parametric_combat(test_df, feature_cols, batch_col, params)
    return CorrectionResult(
        corrected_train,
        corrected_test,
        True,
        "applied_permuted_batches" if permute_batches else "applied",
        method,
        None,
    )
