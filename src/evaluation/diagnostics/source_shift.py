"""Direct harmonization metrics for DFL and SkillCorner.

This module implements source-shift tests that are common in domain
adaptation literature:

- Maximum Mean Discrepancy with an RBF kernel (Gretton et al., 2012)
- Domain-classifier accuracy / proxy A-distance intuition
  (Ben-David et al., 2010)
- Per-feature Kolmogorov-Smirnov tests

The functions are intentionally independent of the formation task. They answer
the first evaluation question directly: can DFL and SkillCorner still be
distinguished after the merge pipeline?
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.evaluation.serialization import _json_default, save_report

DATA_PATH = Path(settings.data_path)
MATCHES_DIR = DATA_PATH / "merged" / "matches"
REPORTS_DIR = DATA_PATH / "reports"
PLOTS_DIR = Path("plots")
DFL_RAW_DIR = DATA_PATH / "DFL"
SC_RAW_DIR = DATA_PATH / "SkillCorner" / "matches"

RANDOM_SEED = 42
DFL_PERIOD_MAP = {"firstHalf": "first_half", "secondHalf": "second_half"}


def _tracking_files() -> list[Path]:
    files = sorted(MATCHES_DIR.glob("*/*_tracking_10hz.parquet"))
    if not files:
        raise FileNotFoundError(f"No tracking parquet files found below {MATCHES_DIR}")
    return files


def _event_files() -> list[Path]:
    return sorted(MATCHES_DIR.glob("*/*_events_spadl.parquet"))


def _balanced_sample(
    df: pd.DataFrame,
    source_col: str = "source",
    per_source: int = 20_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a source-balanced sample from a DataFrame."""
    parts = []
    for _, group in df.groupby(source_col):
        n = min(len(group), per_source)
        if n > 0:
            parts.append(group.sample(n=n, random_state=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_player_tracking_features(
    rows_per_match: int = 8_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build row-level player tracking features from visible player rows.

    The feature view intentionally excludes identifiers, team names, and
    provider metadata. It keeps only numeric tracking quantities that should be
    comparable after harmonization.
    """
    cols = [
        "match_id", "source", "period", "x", "y", "ball_x", "ball_y", "ball_z",
        "speed_kmh_filtered", "acceleration_ms2_filtered", "distance_m",
        "is_visible", "is_home", "ball_status",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    for path in _tracking_files():
        df = pd.read_parquet(path, columns=cols)
        df = df[df["is_visible"] == True].copy()
        if len(df) > rows_per_match:
            df = df.sample(n=rows_per_match, random_state=int(rng.randint(0, 1_000_000)))
        df["period_second_half"] = (df["period"] == "second_half").astype(int)
        df["is_home"] = df["is_home"].astype(int)
        df["ball_status"] = df["ball_status"].astype(int)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "x", "y", "ball_x", "ball_y", "ball_z", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "distance_m", "is_home", "ball_status",
        "period_second_half",
    ]
    return out[["source", "match_id", *feature_cols]]


def build_player_tracking_core_features(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a stricter player-tracking view without obvious context artefacts."""
    df = build_player_tracking_features(seed=seed)
    cols = [
        "source", "match_id", "x", "y", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "distance_m",
    ]
    return df[cols]


def build_player_tracking_strict_features(
    rows_per_match: int = 8_000,
    min_outfield_per_team: int = 9,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build coverage-matched player tracking features.

    Keeps only team-frames where at least ``min_outfield_per_team`` outfield
    players (GK excluded) are visible. Goalkeepers are excluded from the count
    because broadcast cameras rarely detect them (≈18 % visibility vs ≈61 % for
    outfield), so including them in the threshold would discard structurally
    valid formation frames where only the GK is off-camera.
    """
    cols = [
        "match_id", "source", "frame_id", "team_id", "x", "y",
        "speed_kmh_filtered", "acceleration_ms2_filtered", "distance_m",
        "is_visible", "position_label",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    for path in _tracking_files():
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        outfield = df[df["position_label"] != "GK"]
        counts = (
            outfield.groupby(["match_id", "team_id", "frame_id"])["x"]
            .size()
            .rename("n_outfield_visible")
            .reset_index()
        )
        valid = counts[counts["n_outfield_visible"] >= min_outfield_per_team]
        df = df.merge(valid[["match_id", "team_id", "frame_id"]], on=["match_id", "team_id", "frame_id"])
        if len(df) > rows_per_match:
            df = df.sample(n=rows_per_match, random_state=int(rng.randint(0, 1_000_000)))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    cols_out = [
        "source", "match_id", "x", "y", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "distance_m",
    ]
    return out[cols_out]


def _parse_iso_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return pd.Timestamp(ts).timestamp()
    except (ValueError, TypeError):
        return None


def _sample_rows_per_match(
    rows: list[dict],
    rows_per_match: int,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    if len(rows) <= rows_per_match:
        return rows
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rows), size=rows_per_match, replace=False)
    return [rows[i] for i in idx]


def _build_raw_dfl_tracking_common_features(
    rows_per_match: int = 8_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Extract a minimal common tracking view directly from DFL raw XML.

    DFL is downsampled from 25 Hz to approximately 10 Hz using the same
    alternating 2/3-frame cadence as the merge pipeline. Distances, speed, and
    acceleration are recomputed after downsampling from actual selected-frame
    timestamps. No interpolation, final parquet QC, kinematic smoothing,
    metadata joining, or feature hygiene is applied.
    """
    rows = []
    rng = np.random.RandomState(seed)
    for raw_path in sorted(DFL_RAW_DIR.glob("*positions_raw*.xml")):
        match_id = raw_path.stem.split("_")[-1]
        match_rows = []
        current_team = None
        current_period = None
        current_player = None
        selected_next = 0
        step2 = True
        frame_position = 0
        prev_x = prev_y = prev_t = prev_speed_ms = None

        for event, elem in ET.iterparse(str(raw_path), events=("start", "end")):
            if event == "start" and elem.tag == "FrameSet":
                current_team = elem.get("TeamId")
                current_player = elem.get("PersonId")
                current_period = DFL_PERIOD_MAP.get(
                    elem.get("GameSection", ""), elem.get("GameSection", "")
                )
                selected_next = 0
                step2 = True
                frame_position = 0
                prev_x = prev_y = prev_t = prev_speed_ms = None

            elif event == "end" and elem.tag == "Frame":
                if (
                    current_team not in {"BALL", "referee", None}
                    and current_player is not None
                    and frame_position == selected_next
                ):
                    x_raw = elem.get("X")
                    y_raw = elem.get("Y")
                    t_raw = elem.get("T")
                    if x_raw is not None and y_raw is not None:
                        x = float(x_raw)
                        y = float(y_raw)
                        t = _parse_iso_seconds(t_raw)
                        distance_m = speed_kmh = acceleration_ms2 = None
                        if prev_x is not None and prev_t is not None and t is not None:
                            dt = t - prev_t
                            if dt > 0:
                                distance_m = float(np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2))
                                speed_ms = distance_m / dt
                                speed_kmh = speed_ms * 3.6
                                if prev_speed_ms is not None:
                                    acceleration_ms2 = (speed_ms - prev_speed_ms) / dt
                                prev_speed_ms = speed_ms
                        match_rows.append(
                            {
                                "source": "DFL",
                                "match_id": match_id,
                                "period": current_period,
                                "x": x,
                                "y": y,
                                "speed_kmh": speed_kmh,
                                "acceleration_ms2": acceleration_ms2,
                                "distance_m": distance_m,
                            }
                        )
                        prev_x, prev_y, prev_t = x, y, t
                    selected_next += 2 if step2 else 3
                    step2 = not step2
                frame_position += 1
                elem.clear()

            elif event == "end" and elem.tag == "FrameSet":
                elem.clear()

        rows.extend(
            _sample_rows_per_match(
                match_rows,
                rows_per_match,
                seed=int(rng.randint(0, 1_000_000)),
            )
        )

    return pd.DataFrame(rows)


def _build_raw_sc_tracking_common_features(
    rows_per_match: int = 8_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Extract a minimal common tracking view directly from SkillCorner JSONL.

    SkillCorner is native 10 Hz, so distances, speed, and acceleration are
    recomputed from consecutive detected 0.1 s frames after visibility gaps are
    reset. This mirrors the finite-difference units used for the DFL raw common
    view: metres, km/h, and m/s^2.
    """
    rows = []
    rng = np.random.RandomState(seed)
    for jsonl_path in sorted(SC_RAW_DIR.glob("*/*_tracking_extrapolated.jsonl")):
        match_id = jsonl_path.parent.name
        match_rows = []
        prev_positions: dict[tuple, tuple[float, float, float | None]] = {}
        prev_speeds: dict[tuple, float] = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                period = frame.get("period")
                if period is None:
                    continue
                frame_id = frame.get("frame")
                if frame_id is None:
                    continue
                t = float(frame_id) / 10.0
                period_name = "first_half" if period == 1 else "second_half"
                for player in frame.get("player_data", []):
                    player_id = player.get("player_id")
                    if player_id is None:
                        continue
                    key = (player_id, period)
                    if not bool(player.get("is_detected", False)):
                        prev_positions.pop(key, None)
                        prev_speeds.pop(key, None)
                        continue
                    x_raw = player.get("x")
                    y_raw = player.get("y")
                    if x_raw is None or y_raw is None:
                        continue
                    x = float(x_raw)
                    y = float(y_raw)
                    distance_m = speed_kmh = acceleration_ms2 = None
                    prev = prev_positions.get(key)
                    if prev is not None:
                        prev_x, prev_y, prev_t = prev
                        dt = t - prev_t if prev_t is not None else 0.1
                        if dt > 0:
                            distance_m = float(np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2))
                            speed_ms = distance_m / dt
                            speed_kmh = speed_ms * 3.6
                            if key in prev_speeds:
                                acceleration_ms2 = (speed_ms - prev_speeds[key]) / dt
                            prev_speeds[key] = speed_ms
                    prev_positions[key] = (x, y, t)
                    match_rows.append(
                        {
                            "source": "SkillCorner",
                            "match_id": match_id,
                            "period": period_name,
                            "x": x,
                            "y": y,
                            "speed_kmh": speed_kmh,
                            "acceleration_ms2": acceleration_ms2,
                            "distance_m": distance_m,
                        }
                    )

        rows.extend(
            _sample_rows_per_match(
                match_rows,
                rows_per_match,
                seed=int(rng.randint(0, 1_000_000)),
            )
        )

    return pd.DataFrame(rows)


def build_raw_tracking_common_features(
    rows_per_match: int = 8_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build a minimal-alignment raw common-feature baseline.

    This baseline reads directly from the provider raw tracking files and keeps
    only features available in both formats. It aligns DFL to 10 Hz so the
    source classifier does not simply learn frame rate, but it deliberately
    avoids the final merged-parquet QC, smoothing, and strict feature hygiene.
    """
    dfl = _build_raw_dfl_tracking_common_features(rows_per_match, seed)
    sc = _build_raw_sc_tracking_common_features(rows_per_match, seed + 1)
    out = pd.concat([dfl, sc], ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    cols = ["source", "match_id", "x", "y", "speed_kmh", "acceleration_ms2", "distance_m"]
    return out[cols]


def build_team_shape_features(
    frames_per_match: int = 2_500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build frame-team shape features from visible player positions.

    This view asks whether team shape distributions still reveal the provider.
    It is less sensitive to individual missing rows than row-level tracking.
    """
    cols = [
        "match_id", "source", "frame_id", "period", "team_id", "x", "y",
        "speed_kmh_filtered", "is_visible", "is_home", "position_label",
    ]
    rows = []
    rng = np.random.RandomState(seed)
    for path in _tracking_files():
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        frame_ids = df["frame_id"].drop_duplicates()
        if len(frame_ids) > frames_per_match:
            sample_ids = frame_ids.sample(
                n=frames_per_match,
                random_state=int(rng.randint(0, 1_000_000)),
            )
            df = df[df["frame_id"].isin(sample_ids)]

        df["is_outfield"] = (df["position_label"] != "GK").astype(int)
        grouped = df.groupby(["source", "match_id", "team_id", "frame_id", "period", "is_home"])
        feat = grouped.agg(
            n_visible=("x", "size"),
            n_outfield_visible=("is_outfield", "sum"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
            width_y=("y", lambda s: s.max() - s.min()),
            depth_x=("x", lambda s: s.max() - s.min()),
            spread_x=("x", "std"),
            spread_y=("y", "std"),
            mean_speed=("speed_kmh_filtered", "mean"),
            std_speed=("speed_kmh_filtered", "std"),
        ).reset_index()
        feat = feat[feat["n_visible"] >= 8].copy()
        feat["period_second_half"] = (feat["period"] == "second_half").astype(int)
        feat["is_home"] = feat["is_home"].astype(int)
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "n_visible", "centroid_x", "centroid_y", "width_y", "depth_x",
        "spread_x", "spread_y", "mean_speed", "std_speed", "is_home",
        "period_second_half",
    ]
    # n_outfield_visible is a filter metadata column, not a feature; it is
    # returned alongside source/match_id for use by the strict builder.
    return out[["source", "match_id", "n_outfield_visible", *feature_cols]]


def build_team_shape_core_features(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build team-shape features without visibility/home indicators."""
    df = build_team_shape_features(seed=seed)
    cols = [
        "source", "match_id", "centroid_x", "centroid_y", "width_y",
        "depth_x", "spread_x", "spread_y", "mean_speed", "std_speed",
    ]
    return df[cols]


def build_team_shape_strict_features(
    seed: int = RANDOM_SEED,
    min_outfield_visible: int = 9,
) -> pd.DataFrame:
    """Build team-shape features with coverage filter and no shape-range features.

    Drops width/depth/spread because these quantities shrink mechanically when
    players are missing from the frame, coupling them to SkillCorner's
    broadcast-visibility gaps. Restricts to frames with at least
    ``min_outfield_visible`` outfield players (GK excluded) visible. Goalkeepers
    are excluded from the threshold because broadcast cameras rarely detect them
    (≈18 % visibility), so including them would drop valid formation frames
    where only the GK is off-camera.
    """
    df = build_team_shape_features(seed=seed)
    df = df[df["n_outfield_visible"] >= min_outfield_visible].copy()
    cols = [
        "source", "match_id", "centroid_x", "centroid_y",
        "mean_speed", "std_speed",
    ]
    return df[cols]


# ---------------------------------------------------------------------------
# CDF §5.2 orientation-normalized views
# ---------------------------------------------------------------------------

def _load_orientation_meta() -> dict[str, dict[str, object]]:
    """Read orientation metadata from the merged metadata JSON files."""
    mapping: dict[str, dict[str, object]] = {}
    for meta_path in sorted(MATCHES_DIR.glob("*/*_metadata.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        mapping[str(meta["match_id"])] = {
            "play_direction": meta.get("play_direction", {}) or {},
            "coordinates_normalized_to_cdf": bool(meta.get("coordinates_normalized_to_cdf", False)),
        }
    return mapping


def _orientation_flip_mask(
    match_ids: pd.Series, periods: pd.Series, orientation_meta: dict[str, dict[str, object]]
) -> np.ndarray:
    """Return a bool mask marking rows that need x → -x to normalize to CDF orientation.

    CDF §5.2 requires the home team to always play left-to-right. A row must
    be flipped iff the home team's original attacking direction for that
    period was right-to-left (``"right_to_left"`` in merged metadata).
    """
    def _flip(mid: str, period: str) -> bool:
        meta = orientation_meta.get(str(mid), {})
        if meta.get("coordinates_normalized_to_cdf"):
            return False
        direction = (meta.get("play_direction") or {}).get(str(period), "")
        return direction in ("right_to_left", "right_left")

    return np.fromiter(
        (_flip(mid, period) for mid, period in zip(match_ids, periods)),
        dtype=bool,
        count=len(match_ids),
    )


def build_oriented_player_tracking_strict_features(
    rows_per_match: int = 8_000,
    min_outfield_per_team: int = 9,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Strict player-tracking with CDF-compliant home-left-to-right orientation.

    Identical to ``player_tracking_strict`` except x is flipped per row
    according to the metadata play_direction so the home team always attacks
    left-to-right. Uses outfield-only coverage threshold (GK excluded); see
    ``build_player_tracking_strict_features`` for rationale.
    """
    cols = [
        "match_id", "source", "frame_id", "team_id", "period",
        "x", "y", "speed_kmh_filtered", "acceleration_ms2_filtered",
        "distance_m", "is_visible", "position_label",
    ]
    direction_map = _load_orientation_meta()
    rows = []
    rng = np.random.RandomState(seed)
    for path in _tracking_files():
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        outfield = df[df["position_label"] != "GK"]
        counts = (
            outfield.groupby(["match_id", "team_id", "frame_id"])["x"]
            .size()
            .rename("n_outfield_visible")
            .reset_index()
        )
        valid = counts[counts["n_outfield_visible"] >= min_outfield_per_team]
        df = df.merge(valid[["match_id", "team_id", "frame_id"]],
                      on=["match_id", "team_id", "frame_id"])
        flip = _orientation_flip_mask(df["match_id"], df["period"], direction_map)
        df.loc[flip, "x"] = -df.loc[flip, "x"].to_numpy()
        if len(df) > rows_per_match:
            df = df.sample(n=rows_per_match, random_state=int(rng.randint(0, 1_000_000)))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    cols_out = [
        "source", "match_id", "x", "y", "speed_kmh_filtered",
        "acceleration_ms2_filtered", "distance_m",
    ]
    return out[cols_out]


def build_oriented_team_shape_strict_features(
    frames_per_match: int = 2_500,
    min_outfield_visible: int = 9,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Strict team-shape features with CDF-compliant orientation normalization.

    The centroid_x of a team covaries with the team's attacking direction;
    since the two sources differ in how often the home team originally
    attacked left-to-right (86 % right-to-left first halves for DFL vs. 50 %
    for SkillCorner), un-normalized centroid_x carries orientation signal
    that the domain classifier picks up. Normalizing x so the home team
    always attacks left-to-right removes that confound. Uses outfield-only
    coverage threshold (GK excluded); see
    ``build_player_tracking_strict_features`` for rationale.
    """
    cols = [
        "match_id", "source", "frame_id", "period", "team_id",
        "x", "y", "speed_kmh_filtered", "is_visible", "is_home",
        "position_label",
    ]
    direction_map = _load_orientation_meta()
    rows = []
    rng = np.random.RandomState(seed)
    for path in _tracking_files():
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["is_visible"] == True) & df["team_id"].notna()].copy()
        if df.empty:
            continue
        frame_ids = df["frame_id"].drop_duplicates()
        if len(frame_ids) > frames_per_match:
            sample_ids = frame_ids.sample(
                n=frames_per_match,
                random_state=int(rng.randint(0, 1_000_000)),
            )
            df = df[df["frame_id"].isin(sample_ids)]

        flip = _orientation_flip_mask(df["match_id"], df["period"], direction_map)
        df.loc[flip, "x"] = -df.loc[flip, "x"].to_numpy()

        df["is_outfield"] = (df["position_label"] != "GK").astype(int)
        grouped = df.groupby(["source", "match_id", "team_id", "frame_id", "period", "is_home"])
        feat = grouped.agg(
            n_outfield_visible=("is_outfield", "sum"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
            mean_speed=("speed_kmh_filtered", "mean"),
            std_speed=("speed_kmh_filtered", "std"),
        ).reset_index()
        feat = feat[feat["n_outfield_visible"] >= min_outfield_visible].copy()
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True)
    cols_out = ["source", "match_id", "centroid_x", "centroid_y",
                "mean_speed", "std_speed"]
    return out[cols_out]


def build_event_features() -> pd.DataFrame:
    """Build event-level SPADL features with simple one-hot action metadata."""
    files = _event_files()
    if not files:
        raise FileNotFoundError(f"No event parquet files found below {MATCHES_DIR}")
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    base = df[
        [
            "source", "match_id", "action_type", "result", "start_x", "start_y",
            "end_x", "end_y", "cross_dataset_comparable",
        ]
    ].copy()
    base["dx"] = base["end_x"] - base["start_x"]
    base["dy"] = base["end_y"] - base["start_y"]
    base["distance"] = np.sqrt(base["dx"] ** 2 + base["dy"] ** 2)
    base["cross_dataset_comparable"] = base["cross_dataset_comparable"].astype(int)
    encoded = pd.get_dummies(base, columns=["action_type", "result"], dtype=int)
    return encoded


def build_comparable_event_features() -> pd.DataFrame:
    """Build event features restricted to cross-dataset comparable SPADL actions."""
    df = build_event_features()
    df = df[df["cross_dataset_comparable"] == 1].copy()
    cols = [
        col for col in df.columns
        if col in {"source", "match_id", "start_x", "start_y", "end_x", "end_y", "dx", "dy", "distance"}
        or col.startswith("action_type_")
        or col.startswith("result_")
    ]
    return df[cols]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col not in {"source", "match_id"} and pd.api.types.is_numeric_dtype(df[col])
    ]


def _prepare_arrays(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    max_per_source: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Extract imputed numeric arrays for DFL and SkillCorner.

    Returns the DFL feature matrix, SkillCorner feature matrix, ordered feature
    column names, and the per-row ``match_id`` arrays for DFL and SkillCorner
    (used as groups for match-level cross-validation).
    """
    carry_cols = ["source"]
    if "match_id" in df.columns:
        carry_cols.append("match_id")
    work = df[[*carry_cols, *feature_cols]].copy()
    if max_per_source is not None:
        work = _balanced_sample(work, per_source=max_per_source, seed=seed)

    feature_cols = list(feature_cols)
    dfl = work[work["source"] == "DFL"]
    sc = work[work["source"] == "SkillCorner"]
    if dfl.empty or sc.empty:
        raise ValueError("Both DFL and SkillCorner samples are required.")

    imputer = SimpleImputer(strategy="median")
    combined = pd.concat([dfl[feature_cols], sc[feature_cols]], ignore_index=True)
    imputer.fit(combined)
    X_dfl = imputer.transform(dfl[feature_cols])
    X_sc = imputer.transform(sc[feature_cols])

    if "match_id" in work.columns:
        groups_dfl = dfl["match_id"].to_numpy()
        groups_sc = sc["match_id"].to_numpy()
    else:
        groups_dfl = np.full(len(dfl), "dfl_unknown", dtype=object)
        groups_sc = np.full(len(sc), "sc_unknown", dtype=object)

    return X_dfl, X_sc, feature_cols, groups_dfl, groups_sc


def _standardize_pair(X_source: np.ndarray, X_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    pooled = np.vstack([X_source, X_target])
    scaler.fit(pooled)
    return scaler.transform(X_source), scaler.transform(X_target)


def _rbf_kernel_from_sqdist(sqdist: np.ndarray, bandwidth: float) -> np.ndarray:
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    return np.exp(-gamma * sqdist)


def _mmd_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    n = len(X)
    m = len(Y)
    if n < 2 or m < 2:
        raise ValueError("MMD requires at least two samples per group.")

    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    Kxx = _rbf_kernel_from_sqdist(XX, bandwidth)
    Kyy = _rbf_kernel_from_sqdist(YY, bandwidth)
    Kxy = _rbf_kernel_from_sqdist(XY, bandwidth)

    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sum_xx + sum_yy - 2.0 * Kxy.mean())


def compute_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_permutations: int = 100,
    subsample: int = 1_500,
    seed: int = RANDOM_SEED,
) -> dict:
    """Compute unbiased RBF-MMD squared with a permutation p-value.

    Args:
        X_source: Feature matrix for DFL.
        X_target: Feature matrix for SkillCorner.
        n_permutations: Number of label permutations for the null test.
        subsample: Maximum rows per source for quadratic kernel computation.
        seed: Random seed.

    Returns:
        Dictionary with MMD statistic, p-value, bandwidth, and sample sizes.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X_source, dtype=float)
    Y = np.asarray(X_target, dtype=float)
    if len(X) > subsample:
        X = X[rng.choice(len(X), size=subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), size=subsample, replace=False)]

    X, Y = _standardize_pair(X, Y)
    pooled = np.vstack([X, Y])
    distances = pdist(pooled, metric="euclidean")
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    observed = _mmd_unbiased(X, Y, bandwidth)
    n = len(X)
    permuted = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(pooled))
        Xp = pooled[idx[:n]]
        Yp = pooled[idx[n:]]
        permuted.append(_mmd_unbiased(Xp, Yp, bandwidth))
    permuted_arr = np.asarray(permuted)
    p_value = float((np.sum(permuted_arr >= observed) + 1) / (n_permutations + 1))

    return {
        "mmd_sq": observed,
        "p_value": p_value,
        "bandwidth": bandwidth,
        "n_source": int(len(X)),
        "n_target": int(len(Y)),
        "n_permutations": int(n_permutations),
        "permutation_mean": float(permuted_arr.mean()),
        "permutation_std": float(permuted_arr.std()),
    }


def domain_classifier_accuracy(
    X_source: np.ndarray,
    X_target: np.ndarray,
    feature_names: list[str] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    groups_source: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
) -> dict:
    """Train a source classifier and report cross-validated accuracy.

    Accuracy near 0.5 means the sources are hard to distinguish from the given
    feature set. High accuracy means provider/domain signal remains.

    When ``groups_source`` and ``groups_target`` are provided (typically the
    ``match_id`` per row), cross-validation uses ``StratifiedGroupKFold`` so
    that every match is entirely in train or test. This prevents the classifier
    from exploiting per-match quirks (leakage) and yields a more honest
    source-discrimination estimate.
    """
    X = np.vstack([X_source, X_target])
    y = np.array([0] * len(X_source) + [1] * len(X_target))
    groups: np.ndarray | None = None
    split_scheme = "StratifiedKFold"
    if groups_source is not None and groups_target is not None:
        groups = np.concatenate([np.asarray(groups_source), np.asarray(groups_target)])
        source_groups = np.unique(groups_source)
        target_groups = np.unique(groups_target)
        n_group_splits = min(n_splits, len(source_groups), len(target_groups))
        if n_group_splits >= 2:
            source_group_folds = KFold(
                n_splits=n_group_splits, shuffle=True, random_state=seed
            ).split(source_groups)
            target_group_folds = KFold(
                n_splits=n_group_splits, shuffle=True, random_state=seed + 1
            ).split(target_groups)
            splits = []
            for (_, src_test_idx), (_, tgt_test_idx) in zip(source_group_folds, target_group_folds):
                test_groups = set(source_groups[src_test_idx]) | set(target_groups[tgt_test_idx])
                test_idx = np.where(np.isin(groups, list(test_groups)))[0]
                train_idx = np.setdiff1d(np.arange(len(groups)), test_idx)
                if len(np.unique(y[test_idx])) == 2 and len(np.unique(y[train_idx])) == 2:
                    splits.append((train_idx, test_idx))
            if splits:
                cv = splits
                split_scheme = "SourceBalancedGroupKFold"
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                groups = None
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            groups = None
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    scores = cross_validate(
        clf,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
        return_estimator=True,
        n_jobs=1,
        groups=groups,
    )

    importances = None
    if feature_names:
        vals = []
        for estimator in scores["estimator"]:
            vals.append(estimator.named_steps["rf"].feature_importances_)
        mean_imp = np.mean(vals, axis=0)
        order = np.argsort(mean_imp)[::-1]
        importances = [
            {"feature": feature_names[i], "importance": float(mean_imp[i])}
            for i in order[:15]
        ]

    acc = scores["test_accuracy"]
    bal = scores["test_balanced_accuracy"]
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "balanced_accuracy_mean": float(bal.mean()),
        "balanced_accuracy_std": float(bal.std()),
        "per_fold_accuracy": [float(v) for v in acc],
        "baseline": 0.5,
        "n_source": int(len(X_source)),
        "n_target": int(len(X_target)),
        "top_feature_importances": importances or [],
        "cv_scheme": split_scheme,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
    }


def per_feature_ks(
    X_source: np.ndarray,
    X_target: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Run two-sample KS tests per feature."""
    rows = []
    n_features = len(feature_names)
    for idx, name in enumerate(feature_names):
        xs = X_source[:, idx]
        ys = X_target[:, idx]
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        if len(xs) == 0 or len(ys) == 0:
            stat, p_value = np.nan, np.nan
        else:
            res = ks_2samp(xs, ys)
            stat, p_value = float(res.statistic), float(res.pvalue)
        rows.append(
            {
                "feature": name,
                "ks_stat": stat,
                "p_value": p_value,
                "p_value_bonferroni": min(float(p_value) * n_features, 1.0)
                if np.isfinite(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)


def location_scale_provider_diagnostic(
    X_dfl: np.ndarray,
    X_sc: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Summarize per-feature provider location/scale effects.

    This is the lightweight linear-model diagnostic suggested by the ComBat
    review: for each feature, the two-level provider coefficient is the
    SkillCorner minus DFL mean shift. Scale effects are reported as each
    provider variance relative to the pooled variance and as the SC/DFL
    variance ratio. It is diagnostic only; no correction is applied.
    """
    rows = []
    eps = 1e-12
    for idx, name in enumerate(feature_names):
        dfl = X_dfl[:, idx]
        sc = X_sc[:, idx]
        dfl = dfl[np.isfinite(dfl)]
        sc = sc[np.isfinite(sc)]
        if len(dfl) == 0 or len(sc) == 0:
            rows.append(
                {
                    "feature": name,
                    "n_dfl": int(len(dfl)),
                    "n_skillcorner": int(len(sc)),
                    "gamma_skillcorner_minus_dfl": float("nan"),
                    "gamma_pooled_sd": float("nan"),
                    "dfl_delta2_overall_var": float("nan"),
                    "skillcorner_delta2_overall_var": float("nan"),
                    "variance_ratio_skillcorner_over_dfl": float("nan"),
                }
            )
            continue

        pooled = np.concatenate([dfl, sc])
        dfl_mean = float(np.mean(dfl))
        sc_mean = float(np.mean(sc))
        gamma = sc_mean - dfl_mean
        pooled_std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        pooled_var = float(np.var(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        dfl_var = float(np.var(dfl, ddof=1)) if len(dfl) > 1 else 0.0
        sc_var = float(np.var(sc, ddof=1)) if len(sc) > 1 else 0.0
        rows.append(
            {
                "feature": name,
                "n_dfl": int(len(dfl)),
                "n_skillcorner": int(len(sc)),
                "dfl_mean": dfl_mean,
                "skillcorner_mean": sc_mean,
                "gamma_skillcorner_minus_dfl": float(gamma),
                "pooled_std": pooled_std,
                "gamma_pooled_sd": float(gamma / pooled_std)
                if pooled_std > eps else float("nan"),
                "dfl_variance": dfl_var,
                "skillcorner_variance": sc_var,
                "pooled_variance": pooled_var,
                "dfl_delta2_overall_var": float(dfl_var / pooled_var)
                if pooled_var > eps else float("nan"),
                "skillcorner_delta2_overall_var": float(sc_var / pooled_var)
                if pooled_var > eps else float("nan"),
                "variance_ratio_skillcorner_over_dfl": float(sc_var / dfl_var)
                if dfl_var > eps else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out["_abs_gamma_pooled_sd"] = out["gamma_pooled_sd"].abs()
    return (
        out.sort_values("_abs_gamma_pooled_sd", ascending=False)
        .drop(columns=["_abs_gamma_pooled_sd"])
        .reset_index(drop=True)
    )


def evaluate_feature_set(
    name: str,
    df: pd.DataFrame,
    max_per_source: int = 20_000,
    mmd_subsample: int = 1_500,
    n_permutations: int = 100,
    seed: int = RANDOM_SEED,
) -> dict:
    """Evaluate one feature set with MMD, source classifier, and KS tests."""
    feature_cols = _feature_columns(df)
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found for {name}")
    X_dfl, X_sc, feature_cols, groups_dfl, groups_sc = _prepare_arrays(
        df,
        feature_cols,
        max_per_source=max_per_source,
        seed=seed,
    )
    mmd = compute_mmd(
        X_dfl,
        X_sc,
        n_permutations=n_permutations,
        subsample=mmd_subsample,
        seed=seed,
    )
    clf = domain_classifier_accuracy(
        X_dfl,
        X_sc,
        feature_names=feature_cols,
        seed=seed,
        groups_source=groups_dfl,
        groups_target=groups_sc,
    )
    ks = per_feature_ks(X_dfl, X_sc, feature_cols)
    location_scale = location_scale_provider_diagnostic(X_dfl, X_sc, feature_cols)
    significant = int((ks["p_value_bonferroni"] < 0.05).sum())
    return {
        "name": name,
        "n_rows_total": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "source_counts_total": df["source"].value_counts().to_dict(),
        "mmd": mmd,
        "domain_classifier": clf,
        "ks": ks.to_dict(orient="records"),
        "location_scale_diagnostic": location_scale.to_dict(orient="records"),
        "n_ks_significant_bonferroni": significant,
    }


def _plot_summary(report: dict, path: Path) -> None:
    """Create a compact comparison plot for all feature sets."""
    import matplotlib.pyplot as plt

    names = list(report["feature_sets"].keys())
    mmd = [report["feature_sets"][n]["mmd"]["mmd_sq"] for n in names]
    acc = [
        report["feature_sets"][n]["domain_classifier"]["accuracy_mean"]
        for n in names
    ]
    ks_mean = [
        float(pd.DataFrame(report["feature_sets"][n]["ks"])["ks_stat"].mean())
        for n in names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(names, mmd, color="#486B8A")
    axes[0].set_title("MMD squared")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(names, acc, color="#8A5A44")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_title("Domain classifier accuracy")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(names, ks_mean, color="#4B7F52")
    axes[2].set_title("Mean KS statistic")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _accuracy_drop(feature_sets: dict, before: str, after: str) -> dict:
    """Summarize source-classifier change between two related feature views."""
    before_acc = feature_sets[before]["domain_classifier"]["accuracy_mean"]
    after_acc = feature_sets[after]["domain_classifier"]["accuracy_mean"]
    before_mmd = feature_sets[before]["mmd"]["mmd_sq"]
    after_mmd = feature_sets[after]["mmd"]["mmd_sq"]
    return {
        "before": before,
        "after": after,
        "source_accuracy_before": before_acc,
        "source_accuracy_after": after_acc,
        "source_accuracy_drop": before_acc - after_acc,
        "mmd_before": before_mmd,
        "mmd_after": after_mmd,
        "mmd_drop": before_mmd - after_mmd,
    }


def summarize_harmonization_ablation(feature_sets: dict) -> dict:
    """Summarize how much artefact-hygiene steps reduce source signal.

    This is not a full raw-data baseline. It compares increasingly stricter
    feature views inside the merged schema: rich/provider-sensitive views,
    core views, and strict coverage-/artifact-hygiene views.
    """
    comparisons = [
        _accuracy_drop(feature_sets, "raw_tracking_common", "player_tracking_strict"),
        _accuracy_drop(feature_sets, "player_tracking", "player_tracking_core"),
        _accuracy_drop(feature_sets, "player_tracking", "player_tracking_strict"),
        _accuracy_drop(feature_sets, "player_tracking_strict", "player_tracking_strict_oriented"),
        _accuracy_drop(feature_sets, "team_shape", "team_shape_core"),
        _accuracy_drop(feature_sets, "team_shape", "team_shape_strict"),
        _accuracy_drop(feature_sets, "team_shape_strict", "team_shape_strict_oriented"),
        _accuracy_drop(feature_sets, "event_spadl", "event_spadl_comparable"),
    ]
    return {
        "type": "within-merged feature hygiene ablation",
        "caveat": (
            "This is a diagnostic mechanism check, not a pure raw-concat baseline. "
            "It quantifies how much source distinguishability drops when "
            "provider-sensitive features and coverage artefacts are removed."
        ),
        "comparisons": comparisons,
    }


def run_harmonization_evaluation(
    max_per_source: int = 20_000,
    mmd_subsample: int = 1_500,
    n_permutations: int = 100,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run direct harmonization evaluation on tracking and event feature views."""
    feature_builders = {
        "raw_tracking_common": build_raw_tracking_common_features,
        "player_tracking": build_player_tracking_features,
        "player_tracking_core": build_player_tracking_core_features,
        "player_tracking_strict": build_player_tracking_strict_features,
        "player_tracking_strict_oriented": build_oriented_player_tracking_strict_features,
        "team_shape": build_team_shape_features,
        "team_shape_core": build_team_shape_core_features,
        "team_shape_strict": build_team_shape_strict_features,
        "team_shape_strict_oriented": build_oriented_team_shape_strict_features,
        "event_spadl": build_event_features,
        "event_spadl_comparable": build_comparable_event_features,
    }
    results = {}
    for name, builder in feature_builders.items():
        print(f"\nBuilding feature set: {name}")
        df = builder() if name.startswith("event_spadl") else builder(seed=seed)
        print(f"  rows={len(df):,}, features={len(_feature_columns(df))}")
        results[name] = evaluate_feature_set(
            name,
            df,
            max_per_source=max_per_source,
            mmd_subsample=mmd_subsample,
            n_permutations=n_permutations,
            seed=seed,
        )
        acc = results[name]["domain_classifier"]["accuracy_mean"]
        mmd = results[name]["mmd"]["mmd_sq"]
        print(f"  MMD²={mmd:.4f}, source-accuracy={acc:.3f}")

    report = {
        "description": (
            "Direct DFL-vs-SkillCorner source-shift evaluation after the CDF "
            "merge pipeline. Lower MMD, source-classifier accuracy closer to "
            "0.5, and lower KS statistics indicate stronger harmonization."
        ),
        "references": {
            "mmd": "Gretton et al. (2012), A Kernel Two-Sample Test, JMLR.",
            "domain_classifier": (
                "Ben-David et al. (2010), A theory of learning from different "
                "domains, Machine Learning."
            ),
        },
        "parameters": {
            "max_per_source": max_per_source,
            "mmd_subsample": mmd_subsample,
            "n_permutations": n_permutations,
            "seed": seed,
        },
        "minimal_alignment_baseline": {
            "feature_set": "raw_tracking_common",
            "description": (
                "Reads directly from DFL raw tracking XML and SkillCorner raw "
                "tracking JSONL. DFL is decimated from 25 Hz to approximately "
                "10 Hz with the same alternating 2/3-frame cadence used by the "
                "merge pipeline; SkillCorner is native 10 Hz. Distance, speed, "
                "and acceleration are recomputed from consecutive selected/"
                "detected frames using metres, km/h, and m/s^2. No interpolation, "
                "final parquet QC, kinematic smoothing, or strict feature hygiene "
                "is applied. The source-classifier pipeline applies the same "
                "median imputation and StandardScaler preprocessing to every "
                "feature set, including this baseline."
            ),
            "caveat": (
                "This is a minimal-alignment baseline, not a pure raw-concat "
                "baseline. It removes trivial frame-rate/schema distinguishability "
                "before source classification. The comparison with harmonized "
                "tracking still confounds coordinate/schema alignment with the "
                "smoothing and QC steps used in the merged parquet pipeline."
            ),
        },
        "feature_sets": results,
        "ablation_summary": summarize_harmonization_ablation(results),
    }

    report_path = REPORTS_DIR / "harmonization_evaluation.json"
    plot_path = PLOTS_DIR / "harmonization_evaluation.png"
    save_report(report, report_path)
    _plot_summary(report, plot_path)
    print(f"\nSaved report: {report_path}")
    print(f"Saved plot:   {plot_path}")
    return report
