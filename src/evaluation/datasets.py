"""Dataset builders for downstream harmonization tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

DATA_PATH = Path(settings.data_path)
MATCHES_DIR = DATA_PATH / "merged" / "matches"
RANDOM_SEED = 42

_LABEL_TO_BUCKET = {
    "GK": "GK",
    "LB": "FB", "RB": "FB",
    "LCB": "CB", "CB": "CB", "RCB": "CB",
    "LDM": "DM", "CDM": "DM", "RDM": "DM",
    "LCM": "CM/AM", "CM": "CM/AM", "RCM": "CM/AM",
    "LM": "CM/AM", "RM": "CM/AM",
    "LAM": "CM/AM", "CAM": "CM/AM", "RAM": "CM/AM",
    "LW": "WG", "RW": "WG",
    "LCF": "FW", "CF": "FW", "RCF": "FW",
}

def _label_to_bucket(label: str | None) -> str | None:
    if label is None:
        return None
    return _LABEL_TO_BUCKET.get(label)

def _metadata_position_lookup() -> dict[tuple[str, str], str]:
    """Return ``(match_id, player_id) -> tactical bucket`` from metadata JSON.

    Uses the CDF ``position_label`` field (added after the CDF-taxonomy
    migration) and reduces it to the legacy 6-class tactical bucket
    (CB/FB/DM/CM/AM/WG/FW/GK) so downstream classification remains at the
    original granularity. Falls back to ``position_group`` for older
    metadata files that predate the migration.
    """
    lookup: dict[tuple[str, str], str] = {}
    for path in MATCHES_DIR.glob("*/*_metadata.json"):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        match_id = str(meta["match_id"])
        for side in ("home", "away"):
            for player in meta.get("teams", {}).get(side, {}).get("players", []):
                label = player.get("position_label")
                bucket = _label_to_bucket(label) if label else player.get("position_group")
                pid = player.get("id")
                if pid and bucket:
                    lookup[(match_id, str(pid))] = bucket
    return lookup

def load_ball_status_dataset(
    frames_per_match: int = 8_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], str]:
    """Build frame-level ball-status prediction data from tracking.

    The label is ``ball_in_play`` from `ball_status`. Features exclude the
    label itself and use tracking-frame state: ball position, visible-player
    count, team spread, and speed summaries.
    """
    cols = [
        "match_id", "source", "frame_id", "period", "team_id", "x", "y",
        "ball_x", "ball_y", "ball_z", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "is_visible", "ball_status",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    for path in MATCHES_DIR.glob("*/*_tracking_10hz.parquet"):
        df = pd.read_parquet(path, columns=cols)
        frame_ids = df["frame_id"].drop_duplicates()
        if len(frame_ids) > frames_per_match:
            frame_ids = frame_ids.sample(
                n=frames_per_match,
                random_state=int(rng.randint(0, 1_000_000)),
            )
            df = df[df["frame_id"].isin(frame_ids)]
        visible = df[df["is_visible"] == True].copy()
        grouped = visible.groupby(["source", "match_id", "frame_id", "period"], dropna=False)
        feat = grouped.agg(
            ball_in_play=("ball_status", "first"),
            ball_x=("ball_x", "first"),
            ball_y=("ball_y", "first"),
            ball_z=("ball_z", "first"),
            n_visible=("player_id" if "player_id" in visible.columns else "x", "size"),
            mean_x=("x", "mean"),
            mean_y=("y", "mean"),
            spread_x=("x", "std"),
            spread_y=("y", "std"),
            mean_speed=("speed_kmh_filtered", "mean"),
            std_speed=("speed_kmh_filtered", "std"),
            mean_accel=("acceleration_ms2_filtered", "mean"),
        ).reset_index()
        feat["period_second_half"] = (feat["period"] == "second_half").astype(int)
        feat["ball_in_play"] = feat["ball_in_play"].astype(int)
        rows.append(feat)
    out = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "ball_x", "ball_y", "ball_z", "n_visible", "mean_x", "mean_y",
        "spread_x", "spread_y", "mean_speed", "std_speed", "mean_accel",
        "period_second_half",
    ]
    return out[["source", "match_id", "ball_in_play", *feature_cols]], feature_cols, "ball_in_play"

def load_player_aggregate_position_dataset(
    max_frames_per_player: int = 4_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], str]:
    """Build player-match aggregate features for position-group classification.

    One row represents one player in one match. Features summarize the player's
    spatial role and movement profile over many frames. This is more stable and
    tactically meaningful than classifying individual player-frame rows.
    """
    lookup = _metadata_position_lookup()
    cols = [
        "match_id", "source", "frame_id", "period", "player_id", "team_id",
        "position_label", "x", "y", "ball_x", "ball_y", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "distance_m", "is_visible", "is_home",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    for path in MATCHES_DIR.glob("*/*_tracking_10hz.parquet"):
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
        df["position_group"] = df["position_label"].map(_label_to_bucket)
        missing = df["position_group"].isna()
        if missing.any():
            keys = list(zip(df.loc[missing, "match_id"], df.loc[missing, "player_id"]))
            df.loc[missing, "position_group"] = [lookup.get(k) for k in keys]
        df = df[df["position_group"].notna()].copy()
        df = df[df["position_group"] != "GK"].copy()
        if df.empty:
            continue

        sampled = []
        for _, group in df.groupby("player_id"):
            if len(group) > max_frames_per_player:
                group = group.sample(
                    n=max_frames_per_player,
                    random_state=int(rng.randint(0, 1_000_000)),
                )
            sampled.append(group)
        df = pd.concat(sampled, ignore_index=True)

        # Team centroid per frame lets us express a player's role relative to
        # teammates, reducing match-specific pitch-position effects.
        centroids = (
            df.groupby(["team_id", "frame_id"])[["x", "y"]]
            .mean()
            .rename(columns={"x": "team_centroid_x", "y": "team_centroid_y"})
            .reset_index()
        )
        df = df.merge(centroids, on=["team_id", "frame_id"], how="left")
        df["rel_x"] = df["x"] - df["team_centroid_x"]
        df["rel_y"] = df["y"] - df["team_centroid_y"]
        df["dist_to_ball"] = np.sqrt((df["ball_x"] - df["x"]) ** 2 + (df["ball_y"] - df["y"]) ** 2)
        df["period_second_half"] = (df["period"] == "second_half").astype(int)
        df["is_home"] = df["is_home"].astype(int)

        grouped = df.groupby(["source", "match_id", "player_id", "position_group"], dropna=False)
        feat = grouped.agg(
            n_frames=("frame_id", "nunique"),
            x_median=("x", "median"),
            y_median=("y", "median"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
            x_std=("x", "std"),
            y_std=("y", "std"),
            x_q10=("x", lambda s: s.quantile(0.10)),
            x_q90=("x", lambda s: s.quantile(0.90)),
            y_q10=("y", lambda s: s.quantile(0.10)),
            y_q90=("y", lambda s: s.quantile(0.90)),
            rel_x_median=("rel_x", "median"),
            rel_y_median=("rel_y", "median"),
            rel_x_std=("rel_x", "std"),
            rel_y_std=("rel_y", "std"),
            speed_mean=("speed_kmh_filtered", "mean"),
            speed_median=("speed_kmh_filtered", "median"),
            speed_std=("speed_kmh_filtered", "std"),
            speed_q90=("speed_kmh_filtered", lambda s: s.quantile(0.90)),
            accel_mean=("acceleration_ms2_filtered", "mean"),
            accel_std=("acceleration_ms2_filtered", "std"),
            distance_sum=("distance_m", "sum"),
            dist_to_ball_median=("dist_to_ball", "median"),
            dist_to_ball_mean=("dist_to_ball", "mean"),
            is_home=("is_home", "first"),
            second_half_share=("period_second_half", "mean"),
        ).reset_index()
        feat["x_range"] = feat["x_q90"] - feat["x_q10"]
        feat["y_range"] = feat["y_q90"] - feat["y_q10"]
        feat = feat[feat["n_frames"] >= 200].copy()
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "n_frames", "x_median", "y_median", "x_mean", "y_mean", "x_std",
        "y_std", "x_range", "y_range", "rel_x_median", "rel_y_median",
        "rel_x_std", "rel_y_std", "speed_mean", "speed_median", "speed_std",
        "speed_q90", "accel_mean", "accel_std", "distance_sum",
        "dist_to_ball_median", "dist_to_ball_mean", "is_home",
        "second_half_share",
    ]
    # player_id is kept in the public output so downstream tasks (e.g.
    # the Phase 4a tracking-context-augmented variant) can join on the
    # full (match_id, player_id) key. It is not in feature_cols so it is
    # not consumed by the model fit.
    return out[["source", "match_id", "player_id", "position_group", *feature_cols]], feature_cols, "position_group"

def load_player_aggregate_position_no_kinematic_dataset(
    max_frames_per_player: int = 4_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], str]:
    """Build the position aggregate task without speed/acceleration features."""
    df, feature_cols, label_col = load_player_aggregate_position_dataset(
        max_frames_per_player=max_frames_per_player,
        seed=seed,
    )
    feature_cols = [
        col for col in feature_cols
        if not (
            col.startswith("speed_")
            or col.startswith("accel_")
            or col == "distance_sum"
        )
    ]
    return df[["source", "match_id", "player_id", label_col, *feature_cols]], feature_cols, label_col

def load_kinematic_regression_dataset(
    max_frames_per_player: int = 5_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build player-match samples for kinematic regression.

    Main targets are tracking-derived intensity rates:
    distance per observed minute, high-speed-running distance per observed
    minute, and mean speed. Absolute distance targets are deliberately avoided
    because they are dominated by exposure time (``n_frames``). Sprint count and
    maximum speed are computed internally for exploratory checks but are
    excluded from the main target list because initial experiments showed weak
    learnability. Features are role/location descriptors and deliberately
    exclude direct speed/distance aggregates and ``n_frames`` to avoid trivial
    exposure-time leakage.
    """
    cols = [
        "match_id", "source", "frame_id", "period", "player_id", "team_id",
        "position_label", "x", "y", "ball_x", "ball_y", "speed_kmh_filtered",
        "distance_m", "is_visible", "is_home",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    lookup = _metadata_position_lookup()
    for path in MATCHES_DIR.glob("*/*_tracking_10hz.parquet"):
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
        df["position_group"] = df["position_label"].map(_label_to_bucket)
        missing = df["position_group"].isna()
        if missing.any():
            keys = list(zip(df.loc[missing, "match_id"], df.loc[missing, "player_id"]))
            df.loc[missing, "position_group"] = [lookup.get(k) for k in keys]
        df = df[df["position_group"].notna()].copy()
        df = df[df["position_group"] != "GK"].copy()
        if df.empty:
            continue

        sampled = []
        for _, group in df.groupby("player_id"):
            if len(group) > max_frames_per_player:
                group = group.sample(
                    n=max_frames_per_player,
                    random_state=int(rng.randint(0, 1_000_000)),
                )
            sampled.append(group)
        df = pd.concat(sampled, ignore_index=True)

        centroids = (
            df.groupby(["team_id", "frame_id"])[["x", "y"]]
            .mean()
            .rename(columns={"x": "team_centroid_x", "y": "team_centroid_y"})
            .reset_index()
        )
        df = df.merge(centroids, on=["team_id", "frame_id"], how="left")
        df["rel_x"] = df["x"] - df["team_centroid_x"]
        df["rel_y"] = df["y"] - df["team_centroid_y"]
        df["dist_to_ball"] = np.sqrt((df["ball_x"] - df["x"]) ** 2 + (df["ball_y"] - df["y"]) ** 2)
        df["hsr_distance_m"] = np.where(df["speed_kmh_filtered"] >= 19.8, df["distance_m"], 0.0)
        sprint_flag = df["speed_kmh_filtered"] >= 25.2
        # Count sprint starts, not every sprint frame.
        df["sprint_start"] = (
            sprint_flag
            & ~sprint_flag.groupby([df["player_id"]]).shift(fill_value=False)
        ).astype(int)
        df["is_home"] = df["is_home"].astype(int)
        df["period_second_half"] = (df["period"] == "second_half").astype(int)

        grouped = df.groupby(["source", "match_id", "player_id", "position_group"], dropna=False)
        feat = grouped.agg(
            n_frames=("frame_id", "nunique"),
            x_median=("x", "median"),
            y_median=("y", "median"),
            x_std=("x", "std"),
            y_std=("y", "std"),
            rel_x_median=("rel_x", "median"),
            rel_y_median=("rel_y", "median"),
            rel_x_std=("rel_x", "std"),
            rel_y_std=("rel_y", "std"),
            dist_to_ball_median=("dist_to_ball", "median"),
            dist_to_ball_std=("dist_to_ball", "std"),
            is_home=("is_home", "first"),
            second_half_share=("period_second_half", "mean"),
            total_distance_m=("distance_m", "sum"),
            hsr_distance_m=("hsr_distance_m", "sum"),
            sprint_count=("sprint_start", "sum"),
            vmax_kmh=("speed_kmh_filtered", "max"),
            mean_speed_kmh=("speed_kmh_filtered", "mean"),
        ).reset_index()
        feat = feat[feat["n_frames"] >= 200].copy()
        observed_minutes = feat["n_frames"] / 600.0
        feat["distance_per_minute_m"] = feat["total_distance_m"] / observed_minutes
        feat["hsr_per_minute_m"] = feat["hsr_distance_m"] / observed_minutes
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    out = pd.get_dummies(out, columns=["position_group"], dtype=int)
    target_cols = ["distance_per_minute_m", "hsr_per_minute_m", "mean_speed_kmh"]
    # Player-matches with no valid tracking-derived speed (all-NaN
    # speed_kmh_filtered for the visible frames) yield NaN regression
    # targets. sklearn does not accept NaN in y, so drop those rows here
    # rather than per-call. The 200-frame minimum upstream usually
    # prevents this, but broadcast-CV gaps can still leave a rate target
    # undefined.
    out = out.dropna(subset=target_cols)
    exploratory_target_cols = [
        "total_distance_m", "hsr_distance_m", "sprint_count", "vmax_kmh", "n_frames",
    ]
    feature_cols = [
        col for col in out.columns
        if col not in {"source", "match_id", "player_id", *target_cols, *exploratory_target_cols}
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    return out[["source", "match_id", *target_cols, *feature_cols]], feature_cols, target_cols

def _add_pass_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pass_dx"] = df["end_x"] - df["start_x"]
    df["pass_dy"] = df["end_y"] - df["start_y"]
    df["pass_length_m"] = np.sqrt(df["pass_dx"] ** 2 + df["pass_dy"] ** 2)
    df["pass_angle_rad"] = np.arctan2(df["pass_dy"], df["pass_dx"])
    df["forward_progress_m"] = df["pass_dx"]
    df["second_half"] = (df["period"] == "second_half").astype(int)
    return df

def _load_pass_events_base() -> pd.DataFrame:
    """Load all pass actions across matches with shared event-level features."""
    rows = []
    for path in sorted(MATCHES_DIR.glob("*/*_events_spadl.parquet")):
        df = pd.read_parquet(path)
        df = df[df["action_type"] == "pass"].copy()
        df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["team_id"] = df["team_id"].astype(str)
        df["pass_success"] = (df["result"] == "success").astype(int)
        df = _add_pass_event_features(df)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_pass_success_event_dataset() -> tuple[pd.DataFrame, list[str], str]:
    """Pass-success classification with event-only features."""
    df = _load_pass_events_base()
    feature_cols = [
        "start_x", "start_y", "end_x", "end_y",
        "pass_length_m", "pass_angle_rad", "forward_progress_m",
        "second_half",
    ]
    keep = ["source", "match_id", "pass_success", *feature_cols]
    return df[keep].copy(), feature_cols, "pass_success"

def _tracking_context_for_match(
    events: pd.DataFrame,
    tracking_path: Path,
) -> pd.DataFrame:
    """Compute tracking-context features for one match's pass events.

    For each event, the closest tracking frame (asof, ±0.2 s tolerance) is
    matched and a few density / proximity features are computed against the
    pass start and end positions:

    - n_visible_frame: number of visible players in the matched frame
    - teammates_within_10m_start, opponents_within_10m_start
    - opponents_within_3m_start: close-pressure on the passer
    - opponents_within_5m_end: target-area congestion
    - min_dist_opponent_to_start, min_dist_teammate_to_end
    """
    tracking_cols = [
        "match_id", "frame_id", "period", "timestamp",
        "team_id", "player_id", "x", "y", "is_visible",
    ]
    tracking = pd.read_parquet(tracking_path, columns=tracking_cols)
    tracking = tracking[tracking["is_visible"] == True].copy()
    if tracking.empty or events.empty:
        return pd.DataFrame()
    tracking["timestamp"] = pd.to_datetime(
        tracking["timestamp"], errors="coerce", utc=True
    )
    tracking = tracking.dropna(subset=["timestamp"])
    tracking["team_id"] = tracking["team_id"].astype(str)

    # One representative timestamp per frame for the asof join.
    frame_index = (
        tracking.groupby("frame_id", as_index=False)["timestamp"].first()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    events = events.sort_values("timestamp").reset_index(drop=True)
    events_with_frame = pd.merge_asof(
        events,
        frame_index,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("0.2s"),
    )
    events_with_frame = events_with_frame.dropna(subset=["frame_id"]).copy()
    if events_with_frame.empty:
        return pd.DataFrame()
    events_with_frame["frame_id"] = events_with_frame["frame_id"].astype(int)
    events_with_frame["event_uid"] = (
        events_with_frame["match_id"].astype(str)
        + "::"
        + events_with_frame["event_id"].astype(str)
        + "::"
        + events_with_frame.groupby(["match_id", "event_id"]).cumcount().astype(str)
    )

    used_frames = events_with_frame["frame_id"].unique()
    track_sub = tracking[tracking["frame_id"].isin(used_frames)][
        ["frame_id", "team_id", "x", "y"]
    ].copy()

    ev_min = events_with_frame[
        ["event_uid", "frame_id", "team_id", "start_x", "start_y", "end_x", "end_y"]
    ].rename(columns={"team_id": "passer_team_id"})
    merged = ev_min.merge(track_sub, on="frame_id", how="left")
    merged["dist_to_start"] = np.sqrt(
        (merged["x"] - merged["start_x"]) ** 2 + (merged["y"] - merged["start_y"]) ** 2
    )
    merged["dist_to_end"] = np.sqrt(
        (merged["x"] - merged["end_x"]) ** 2 + (merged["y"] - merged["end_y"]) ** 2
    )
    merged["is_teammate"] = (merged["team_id"] == merged["passer_team_id"]).astype(int)

    n_visible = merged.groupby("event_uid").size().rename("n_visible_frame")

    teammates = merged[merged["is_teammate"] == 1]
    opponents = merged[merged["is_teammate"] == 0]

    teammate_features = teammates.groupby("event_uid").agg(
        teammates_within_10m_start=(
            "dist_to_start", lambda d: int((d <= 10).sum())
        ),
        min_dist_teammate_to_end=("dist_to_end", "min"),
    )
    opponent_features = opponents.groupby("event_uid").agg(
        opponents_within_10m_start=(
            "dist_to_start", lambda d: int((d <= 10).sum())
        ),
        opponents_within_3m_start=(
            "dist_to_start", lambda d: int((d <= 3).sum())
        ),
        opponents_within_5m_end=(
            "dist_to_end", lambda d: int((d <= 5).sum())
        ),
        min_dist_opponent_to_start=("dist_to_start", "min"),
    )

    feat = (
        events_with_frame.set_index("event_uid")
        .join(n_visible, how="left")
        .join(teammate_features, how="left")
        .join(opponent_features, how="left")
        .reset_index()
    )
    return feat

def load_pass_success_tracking_context_dataset() -> tuple[pd.DataFrame, list[str], str]:
    """Pass-success classification with event + tracking-context features.

    For each pass event, the closest tracking frame is matched (asof, 0.2 s
    tolerance); event-only features are augmented with per-frame density and
    proximity features around the pass start and end positions.
    """
    rows = []
    for match_dir in sorted(MATCHES_DIR.iterdir()):
        if not match_dir.is_dir():
            continue
        match_id = match_dir.name
        events_path = match_dir / f"{match_id}_events_spadl.parquet"
        tracking_path = match_dir / f"{match_id}_tracking_10hz.parquet"
        if not events_path.exists() or not tracking_path.exists():
            continue
        events = pd.read_parquet(events_path)
        events = events[events["action_type"] == "pass"].copy()
        events = events.dropna(
            subset=["start_x", "start_y", "end_x", "end_y", "timestamp", "team_id"]
        )
        if events.empty:
            continue
        events["match_id"] = events["match_id"].astype(str)
        events["team_id"] = events["team_id"].astype(str)
        events["pass_success"] = (events["result"] == "success").astype(int)
        events = _add_pass_event_features(events)
        events["timestamp"] = pd.to_datetime(
            events["timestamp"], errors="coerce", utc=True
        )
        events = events.dropna(subset=["timestamp"])
        feat = _tracking_context_for_match(events, tracking_path)
        if feat.empty:
            continue
        rows.append(feat)

    if not rows:
        return pd.DataFrame(), [], "pass_success"
    out = pd.concat(rows, ignore_index=True)
    # Drop events whose nearest frame had no other visible players (degenerate).
    out = out[out["n_visible_frame"].fillna(0) >= 1].copy()
    feature_cols = [
        "start_x", "start_y", "end_x", "end_y",
        "pass_length_m", "pass_angle_rad", "forward_progress_m",
        "second_half",
        "n_visible_frame",
        "teammates_within_10m_start",
        "opponents_within_10m_start",
        "opponents_within_3m_start",
        "opponents_within_5m_end",
        "min_dist_opponent_to_start",
        "min_dist_teammate_to_end",
    ]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    keep = ["source", "match_id", "pass_success", *feature_cols]
    return out[keep].copy(), feature_cols, "pass_success"

def load_player_aggregate_position_with_tracking_context_dataset(
    max_frames_per_player: int = 4_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], str]:
    """Player-aggregate position task with tracking-context features added.

    Augments the existing role/location aggregates with per-player
    pressure / density statistics computed against the full visible-player
    set on each sampled frame. Tests whether the tracking-context
    enhancement that mildly helped pass-success also helps position
    prediction.
    """
    base_df, base_features, label_col = load_player_aggregate_position_dataset(
        max_frames_per_player=max_frames_per_player, seed=seed,
    )
    cols = [
        "match_id", "source", "frame_id", "period", "player_id", "team_id",
        "x", "y", "is_visible",
    ]
    extra_rows = []
    for path in MATCHES_DIR.glob("*/*_tracking_10hz.parquet"):
        df = pd.read_parquet(path, columns=cols)
        df = df[df["is_visible"] == True].copy()
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
        df["team_id"] = df["team_id"].astype(str)
        rng = np.random.RandomState(seed)
        sampled = []
        for _, group in df.groupby("player_id"):
            if len(group) > max_frames_per_player:
                group = group.sample(
                    n=max_frames_per_player,
                    random_state=int(rng.randint(0, 1_000_000)),
                )
            sampled.append(group)
        df = pd.concat(sampled, ignore_index=True)

        # Per-frame: opponents and teammates of each visible player.
        # For efficiency, group by frame_id and compute pairwise distances
        # within each frame in numpy.
        frame_ctx_rows = []
        for fid, frame in df.groupby("frame_id"):
            xy = frame[["x", "y"]].to_numpy()
            teams = frame["team_id"].to_numpy()
            pids = frame["player_id"].to_numpy()
            # pairwise distance matrix within this frame
            diffs = xy[:, None, :] - xy[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)  # exclude self
            same_team = teams[:, None] == teams[None, :]
            opp_dist = np.where(same_team, np.inf, dists)
            teammate_dist = np.where(same_team, dists, np.inf)
            for i in range(len(pids)):
                frame_ctx_rows.append({
                    "match_id": frame["match_id"].iloc[i],
                    "frame_id": fid,
                    "player_id": pids[i],
                    "opponents_within_5m": int((opp_dist[i] <= 5).sum()),
                    "opponents_within_10m": int((opp_dist[i] <= 10).sum()),
                    "teammates_within_5m": int((teammate_dist[i] <= 5).sum()),
                    "min_dist_opponent": float(opp_dist[i].min())
                        if np.isfinite(opp_dist[i].min()) else np.nan,
                    "min_dist_teammate": float(teammate_dist[i].min())
                        if np.isfinite(teammate_dist[i].min()) else np.nan,
                })
        if not frame_ctx_rows:
            continue
        frame_ctx = pd.DataFrame(frame_ctx_rows)
        # Aggregate per player-match.
        agg = (
            frame_ctx.groupby(["match_id", "player_id"])
            .agg(
                ctx_opponents_5m_mean=("opponents_within_5m", "mean"),
                ctx_opponents_10m_mean=("opponents_within_10m", "mean"),
                ctx_teammates_5m_mean=("teammates_within_5m", "mean"),
                ctx_min_dist_opponent_median=("min_dist_opponent", "median"),
                ctx_min_dist_teammate_median=("min_dist_teammate", "median"),
            )
            .reset_index()
        )
        extra_rows.append(agg)

    if not extra_rows:
        return base_df, base_features, label_col
    extra = pd.concat(extra_rows, ignore_index=True)
    merged = base_df.merge(extra, on=["match_id", "player_id"], how="left")
    feature_cols = base_features + [
        "ctx_opponents_5m_mean", "ctx_opponents_10m_mean",
        "ctx_teammates_5m_mean", "ctx_min_dist_opponent_median",
        "ctx_min_dist_teammate_median",
    ]
    keep = ["source", "match_id", label_col, *feature_cols]
    return merged[keep].copy(), feature_cols, label_col

def load_shot_success_event_dataset() -> tuple[pd.DataFrame, list[str], str]:
    """Shot-outcome binary classification with event-only features."""
    rows = []
    shot_types = {"shot", "shot_freekick", "shot_penalty"}
    for path in sorted(MATCHES_DIR.glob("*/*_events_spadl.parquet")):
        df = pd.read_parquet(path)
        df = df[df["action_type"].isin(shot_types)].copy()
        df = df.dropna(subset=["start_x", "start_y"])
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        # Treat "successful" / "success" as goal; everything else (including
        # saved/blocked/wide/woodwork on DFL, fail on SC) as non-goal.
        success_tokens = {"success", "successful"}
        df["shot_success"] = df["result"].isin(success_tokens).astype(int)
        df["second_half"] = (df["period"] == "second_half").astype(int)
        df["abs_y"] = df["start_y"].abs()
        # Distance to centre of opposing goal at (52.5, 0). We use absolute
        # x because home-attacks-left-to-right is normalized in CDF orientation,
        # so all shots head toward x = +52.5 regardless of team.
        df["dist_to_goal_m"] = np.sqrt(
            (52.5 - df["start_x"]) ** 2 + df["start_y"] ** 2
        )
        df["angle_to_goal"] = np.arctan2(df["start_y"].abs(), (52.5 - df["start_x"]).clip(lower=0.1))
        df["is_set_piece"] = (df["action_type"] != "shot").astype(int)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "start_x", "start_y", "abs_y",
        "dist_to_goal_m", "angle_to_goal",
        "second_half", "is_set_piece",
    ]
    keep = ["source", "match_id", "shot_success", *feature_cols]
    return out[keep].copy(), feature_cols, "shot_success"

def load_kinematic_regression_visibility_corrected_dataset(
    max_frames_per_player: int = 5_000,
    min_segment_seconds: float = 5.0,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cols = [
        "match_id", "source", "frame_id", "period", "player_id", "team_id",
        "position_label", "x", "y", "ball_x", "ball_y", "speed_kmh_filtered",
        "distance_m", "is_visible", "is_home", "timestamp",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    lookup = _metadata_position_lookup()
    min_seg_frames = int(min_segment_seconds * 10)  # tracking is 10 Hz
    for path in MATCHES_DIR.glob("*/*_tracking_10hz.parquet"):
        df = pd.read_parquet(path, columns=cols)
        df = df[df["team_id"].notna()].copy()
        if df.empty:
            continue
        df["match_id"] = df["match_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
        df["position_group"] = df["position_label"].map(_label_to_bucket)
        missing = df["position_group"].isna()
        if missing.any():
            keys = list(zip(df.loc[missing, "match_id"], df.loc[missing, "player_id"]))
            df.loc[missing, "position_group"] = [lookup.get(k) for k in keys]
        df = df[df["position_group"].notna()].copy()
        df = df[df["position_group"] != "GK"].copy()
        if df.empty:
            continue
        # Identify continuous-visibility segments per player.
        df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
        df["_visible_int"] = df["is_visible"].astype(int)
        # Segment id increments at each visibility transition per player.
        df["_seg"] = (
            (df["_visible_int"] != df.groupby("player_id")["_visible_int"].shift(fill_value=0))
            .groupby(df["player_id"]).cumsum()
        )
        # Keep only visible rows belonging to long enough segments.
        seg_lengths = df[df["_visible_int"] == 1].groupby(["player_id", "_seg"]).size()
        long_segs = seg_lengths[seg_lengths >= min_seg_frames].index
        keep_mask = (df["_visible_int"] == 1) & (
            df.set_index(["player_id", "_seg"]).index.isin(long_segs)
        )
        df = df[keep_mask].copy()
        if df.empty:
            continue
        sampled = []
        for _, group in df.groupby("player_id"):
            if len(group) > max_frames_per_player:
                group = group.sample(
                    n=max_frames_per_player,
                    random_state=int(rng.randint(0, 1_000_000)),
                )
            sampled.append(group)
        df = pd.concat(sampled, ignore_index=True)

        centroids = (
            df.groupby(["team_id", "frame_id"])[["x", "y"]]
            .mean()
            .rename(columns={"x": "team_centroid_x", "y": "team_centroid_y"})
            .reset_index()
        )
        df = df.merge(centroids, on=["team_id", "frame_id"], how="left")
        df["rel_x"] = df["x"] - df["team_centroid_x"]
        df["rel_y"] = df["y"] - df["team_centroid_y"]
        df["dist_to_ball"] = np.sqrt((df["ball_x"] - df["x"]) ** 2 + (df["ball_y"] - df["y"]) ** 2)
        df["hsr_distance_m"] = np.where(df["speed_kmh_filtered"] >= 19.8, df["distance_m"], 0.0)
        df["is_home"] = df["is_home"].astype(int)
        df["period_second_half"] = (df["period"] == "second_half").astype(int)

        grouped = df.groupby(["source", "match_id", "player_id", "position_group"], dropna=False)
        feat = grouped.agg(
            n_frames_continuous=("frame_id", "nunique"),
            x_median=("x", "median"),
            y_median=("y", "median"),
            x_std=("x", "std"),
            y_std=("y", "std"),
            rel_x_median=("rel_x", "median"),
            rel_y_median=("rel_y", "median"),
            rel_x_std=("rel_x", "std"),
            rel_y_std=("rel_y", "std"),
            dist_to_ball_median=("dist_to_ball", "median"),
            dist_to_ball_std=("dist_to_ball", "std"),
            is_home=("is_home", "first"),
            second_half_share=("period_second_half", "mean"),
            total_distance_m=("distance_m", "sum"),
            hsr_distance_m=("hsr_distance_m", "sum"),
            mean_speed_kmh=("speed_kmh_filtered", "mean"),
        ).reset_index()
        feat = feat[feat["n_frames_continuous"] >= 200].copy()
        observed_minutes = feat["n_frames_continuous"] / 600.0
        feat["distance_per_minute_m"] = feat["total_distance_m"] / observed_minutes
        feat["hsr_per_minute_m"] = feat["hsr_distance_m"] / observed_minutes
        rows.append(feat)

    if not rows:
        return pd.DataFrame(), [], []
    out = pd.concat(rows, ignore_index=True)
    out = pd.get_dummies(out, columns=["position_group"], dtype=int)
    target_cols = ["distance_per_minute_m", "hsr_per_minute_m", "mean_speed_kmh"]
    out = out.dropna(subset=target_cols)
    exploratory_target_cols = [
        "total_distance_m", "hsr_distance_m", "n_frames_continuous",
    ]
    feature_cols = [
        col for col in out.columns
        if col not in {"source", "match_id", "player_id", *target_cols, *exploratory_target_cols}
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    return out[["source", "match_id", *target_cols, *feature_cols]], feature_cols, target_cols
