"""Post-processing: position smoothing, speed, acceleration, QC, frame IDs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from src.harmonization.utils import MAX_ACCEL_MS2, MAX_SPEED_KMH, SOURCE_POS_SMOOTH_WINDOW

_FLOAT_COLS = [
    "x", "y", "ball_x", "ball_y", "ball_z", "speed_kmh",
    "speed_kmh_filtered", "acceleration_ms2", "acceleration_ms2_filtered",
    "distance_m",
]


def apply_savgol_filter(
    df: pd.DataFrame, column: str, window: int = 5, order: int = 2
) -> pd.DataFrame:
    """Apply Savitzky-Golay filter per player per match per period, per
    contiguous non-NaN segment.

    Each contiguous run of non-NaN values is filtered independently. Gaps
    (raw NaN values) are preserved as NaN and the filter never bridges them.

    Segments shorter than ``window`` are left as their raw values (the filter
    cannot run on them) rather than being filled with interpolated content.

    Args:
        df: Merged DataFrame.
        column: Column name to filter.
        window: Window length in frames.
        order: Polynomial order.

    Returns:
        DataFrame with additional ``{column}_filtered`` column.
    """
    filtered_col = f"{column}_filtered"
    df[filtered_col] = df[column].copy()

    if savgol_filter is None:
        warnings.warn("scipy not installed, skipping Savitzky-Golay filter")
        return df

    for (_, _, _), group in df.groupby(
        ["match_id", "player_id", "period"], sort=False
    ):
        vals = group[column].values.astype(float)
        n = len(vals)
        if n < window:
            continue
        mask = ~np.isnan(vals)
        if not mask.any():
            continue

        out = vals.copy()
        in_seg = False
        seg_start = 0
        for i in range(n):
            if mask[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not mask[i] and in_seg:
                if (i - seg_start) >= window:
                    out[seg_start:i] = savgol_filter(
                        vals[seg_start:i], window_length=window, polyorder=order
                    )
                in_seg = False
        if in_seg and (n - seg_start) >= window:
            out[seg_start:n] = savgol_filter(
                vals[seg_start:n], window_length=window, polyorder=order
            )

        df.loc[group.index, filtered_col] = out

    return df


def smooth_positions_per_segment(
    df: pd.DataFrame,
    exclude_mask: pd.Series | None = None,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Per-segment Savitzky-Golay smoothing of (x, y), source-aware.

    Smoothing positions BEFORE differentiating attenuates the noise that
    broadcast tracking RMSE injects into single-frame derivatives. With
    σ_position ≈ 1 m for SkillCorner the raw single-frame speed at 10 Hz
    carries ~50 km/h of noise per axis; smoothing speed alone does much
    less than smoothing positions.

    Window length is taken from ``SOURCE_POS_SMOOTH_WINDOW``. NaN positions
    act as segment boundaries; segments shorter than the applicable window
    remain unsmoothed. Rows flagged in ``exclude_mask`` are treated as NaN
    inside the polynomial fit so that single-frame teleports do not bias
    the smoothed window.

    Adds ``x_smooth`` and ``y_smooth`` columns.
    """
    df["x_smooth"] = df["x"].astype(float)
    df["y_smooth"] = df["y"].astype(float)

    if savgol_filter is None:
        warnings.warn("scipy not installed, skipping position smoothing")
        return df

    if exclude_mask is None:
        exclude_mask = pd.Series(False, index=df.index)
    excl_full = exclude_mask.astype(bool)

    for (_, _, _, src), group in df.groupby(
        ["match_id", "player_id", "period", "source"], sort=False
    ):
        window = SOURCE_POS_SMOOTH_WINDOW.get(src)
        if window is None:
            continue
        if window % 2 == 0:
            window += 1
        idx = group.index
        excl = excl_full.loc[idx].values

        for raw_col, smooth_col in (("x", "x_smooth"), ("y", "y_smooth")):
            vals = group[raw_col].astype(float).values.copy()
            vals[excl] = np.nan
            n = len(vals)
            if n < window:
                continue
            mask = ~np.isnan(vals)
            if not mask.any():
                continue

            out = vals.copy()
            in_seg = False
            seg_start = 0
            for i in range(n):
                if mask[i] and not in_seg:
                    seg_start = i
                    in_seg = True
                elif not mask[i] and in_seg:
                    if (i - seg_start) >= window:
                        out[seg_start:i] = savgol_filter(
                            vals[seg_start:i],
                            window_length=window,
                            polyorder=polyorder,
                        )
                    in_seg = False
            if in_seg and (n - seg_start) >= window:
                out[seg_start:n] = savgol_filter(
                    vals[seg_start:n], window_length=window, polyorder=polyorder
                )

            df.loc[idx, smooth_col] = out

    return df


def derive_speed_from_smoothed_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``speed_kmh_filtered`` from (x_smooth, y_smooth).

    Per-row dt is taken from the inter-frame timestamp difference, which
    correctly handles DFL's variable 0.08 / 0.12 s alternating cadence.
    NaN-safe: missing positions or timestamps propagate to NaN speed.
    """
    df["speed_kmh_filtered"] = np.nan
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="ISO8601")

    for (_, _, _), group in df.groupby(
        ["match_id", "player_id", "period"], sort=False
    ):
        idx = group.index
        if len(idx) < 2:
            continue
        x = group["x_smooth"].values.astype(float)
        y = group["y_smooth"].values.astype(float)
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        dt = ts.loc[idx].diff().dt.total_seconds().values
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.sqrt(dx ** 2 + dy ** 2) / dt * 3.6
        df.loc[idx, "speed_kmh_filtered"] = speed

    return df


def derive_acceleration(
    df: pd.DataFrame, speed_col: str = "speed_kmh_filtered"
) -> pd.DataFrame:
    """Derive acceleration (m/s²) from filtered speed.

    ``speed[i]`` is the average velocity over [t[i-1], t[i]] and is
    conceptually at midpoint t_mid[i]. The correct backward-difference
    denominator at sample i is (dt[i-1] + dt[i]) / 2 — which equals a
    constant 0.10 s for DFL's alternating 0.08/0.12 s cadence, avoiding
    the ±25 % alternating bias that using ``dt[i]`` naively would inject.

    Args:
        df: Merged DataFrame with filtered speed column.
        speed_col: Column name for filtered speed (km/h).

    Returns:
        DataFrame with added ``acceleration_ms2`` and filtered variant.
    """
    df["acceleration_ms2"] = np.nan
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="ISO8601")

    for (_, _, _), group in df.groupby(
        ["match_id", "player_id", "period"], sort=False
    ):
        idx = group.index
        if len(idx) < 3:
            continue
        speed_ms = group[speed_col].astype(float).values / 3.6
        dt = ts.loc[idx].diff().dt.total_seconds().values
        avg_dt = (dt[:-1] + dt[1:]) / 2
        accel = np.empty_like(speed_ms)
        accel[:] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            accel[2:] = (speed_ms[2:] - speed_ms[1:-1]) / avg_dt[1:]
        df.loc[idx, "acceleration_ms2"] = accel

    if savgol_filter is not None:
        df = apply_savgol_filter(df, "acceleration_ms2", window=7, order=2)

    return df


def apply_is_visible_nulling(df: pd.DataFrame) -> pd.DataFrame:
    """Set x, y, speed, acceleration to NaN for SC frames where is_visible=False.

    CDF §5.2: missing positions must be explicit null, not interpolated.

    Args:
        df: Merged DataFrame.

    Returns:
        DataFrame with position and kinematic nulls applied.
    """
    mask = (df["source"] == "SkillCorner") & (~df["is_visible"].fillna(False))
    kinematic_cols = [
        "speed_kmh", "speed_kmh_filtered",
        "acceleration_ms2", "acceleration_ms2_filtered",
        "distance_m",
    ]
    for col in kinematic_cols:
        if col in df.columns:
            df.loc[mask, col] = np.nan

    n_masked = mask.sum()
    print(f"  Nulled x/y/kinematics for {n_masked:,} non-visible SC rows")
    return df


def assign_frame_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign monotonic frame_id per match (starting at 0), preserving frame_id_source.

    CDF Table 4: frame_id must be monotonically increasing per match from 0.

    Args:
        df: Merged DataFrame with frame_id_source column.

    Returns:
        DataFrame with frame_id column added.
    """
    period_order = df["period"].map({"first_half": 0, "second_half": 1, None: -1}).fillna(-1)
    sort_key = period_order * 10_000_000 + df["frame_id_source"].astype(float).fillna(0)
    df = df.copy()
    df["_sort_key"] = sort_key
    df["frame_id"] = df.groupby("match_id")["_sort_key"].transform(
        lambda x: x.rank(method="dense").astype(int) - 1
    )
    df = df.drop(columns=["_sort_key"])
    return df


def _assign_frame_ids_single(df: pd.DataFrame) -> pd.DataFrame:
    """Assign monotonic frame_id for a single-match DataFrame (starting at 0)."""
    period_order = df["period"].map({"first_half": 0, "second_half": 1}).fillna(-1)
    sort_key = period_order * 10_000_000 + df["frame_id_source"].astype(float).fillna(0)
    df = df.copy()
    df["frame_id"] = sort_key.rank(method="dense").astype(int) - 1
    return df
