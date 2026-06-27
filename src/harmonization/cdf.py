"""CDF metadata building and JSONL tracking output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.harmonization.utils import (
    CDF_VERSION,
    _json_value,
    _kmh_to_ms,
    _left_team_id_for_period,
    _normalize_utc_iso,
    _right_team_id_for_period,
    _winning_team_id,
)

_PERIOD_ORDER = {
    "first_half": 0,
    "second_half": 1,
    "first_half_extratime": 2,
    "second_half_extratime": 3,
    "shootout": 4,
}


def _build_cdf_periods(df: pd.DataFrame, match_meta: dict) -> list[dict]:
    """Build CDF-style per-period metadata from the saved tracking rows."""
    if df.empty:
        return []

    periods = (
        df[["period", "frame_id", "timestamp"]]
        .drop_duplicates()
        .assign(_period_order=lambda x: x["period"].map(_PERIOD_ORDER).fillna(99))
        .sort_values(["_period_order", "frame_id"])
    )

    out: list[dict] = []
    for period, period_df in periods.groupby("period", sort=False):
        first = period_df.iloc[0]
        last = period_df.iloc[-1]
        out.append(
            {
                "id": period,
                "time_start": _json_value(first["timestamp"]),
                "time_end": _json_value(last["timestamp"]),
                "frame_id_start": _json_value(first["frame_id"]),
                "frame_id_end": _json_value(last["frame_id"]),
                "left_team_id": _left_team_id_for_period(match_meta, period),
                "right_team_id": _right_team_id_for_period(match_meta, period),
                "play_direction": (match_meta.get("play_direction") or {}).get(period),
            }
        )
    return out


def _build_cdf_whistles(periods: list[dict]) -> list[dict]:
    """Construct whistle timestamps for period starts / ends from tracking bounds."""
    whistles: list[dict] = []
    for period in periods:
        period_id = period.get("id")
        start_time = period.get("time_start")
        end_time = period.get("time_end")
        if start_time is not None:
            whistles.append({"type": period_id, "sub_type": "start", "time": start_time})
        if end_time is not None:
            whistles.append({"type": period_id, "sub_type": "end", "time": end_time})
    return whistles


def apply_cdf_orientation(df: pd.DataFrame, play_direction: dict[str, str]) -> pd.DataFrame:
    """Flip x / ball_x so the home team always attacks left-to-right."""
    if df.empty:
        return df

    df = df.copy()
    flip_periods = {period for period, direction in (play_direction or {}).items() if direction == "right_left"}
    if not flip_periods:
        return df

    mask = df["period"].isin(flip_periods)
    if "x" in df.columns:
        df.loc[mask, "x"] = -df.loc[mask, "x"]
    if "ball_x" in df.columns:
        df.loc[mask, "ball_x"] = -df.loc[mask, "ball_x"]
    return df


def _cdf_player_meta_record(player: dict, team_id: str | None = None) -> dict:
    """Translate the legacy per-player roster dict to CDF Tables 1 / 6 keys."""
    started = bool(player.get("starting"))
    has_played = started or player.get("start_time") is not None
    return {
        "id": player.get("id"),
        "first_name": player.get("first_name"),
        "last_name": player.get("last_name"),
        "name": player.get("name"),
        "team_id": player.get("team_id") or team_id,
        "jersey_number": player.get("shirt_number"),
        "is_starter": started,
        "has_played": bool(has_played),
        "position": player.get("position_label"),
        "position_group": player.get("position_group"),
        "start_time": player.get("start_time"),
        "end_time": player.get("end_time"),
    }


def _team_id_for_player(team_rosters: dict, player_id: str | None) -> str | None:
    if not player_id:
        return None
    for tid, team in team_rosters.items():
        for p in team.get("players", []):
            if p.get("id") == player_id:
                return tid
    return None


def _build_cdf_match_events(match_meta: dict) -> dict:
    """Reshape per-player goal / card / sub records into CDF Table 1 lists."""
    teams = match_meta.get("teams", {})
    rosters = {"home": teams.get("home", {}), "away": teams.get("away", {})}
    home_id = (rosters["home"] or {}).get("id")
    away_id = (rosters["away"] or {}).get("id")
    home_score_final = match_meta.get("home_score")
    away_score_final = match_meta.get("away_score")

    goals: list[dict] = []
    cards: list[dict] = []
    substitutions: list[dict] = []

    for side, team in rosters.items():
        team_id = team.get("id") if team else None
        for p in (team.get("players", []) if team else []):
            n_goals = int(p.get("goals") or 0)
            for _ in range(n_goals):
                goals.append({
                    "time": None,
                    "player_id": p.get("id"),
                    "assist_id": None,
                    "team_id": team_id,
                    "is_own_goal": False,
                    "is_penalty": False,
                    "score": {"home": None, "away": None},
                })
            n_own = int(p.get("own_goals") or 0)
            for _ in range(n_own):
                goals.append({
                    "time": None,
                    "player_id": p.get("id"),
                    "assist_id": None,
                    "team_id": away_id if side == "home" else home_id,
                    "is_own_goal": True,
                    "is_penalty": False,
                    "score": {"home": None, "away": None},
                })

            n_yc = int(p.get("yellow_cards") or 0)
            n_rc = int(p.get("red_cards") or 0)
            for _ in range(n_yc):
                cards.append({
                    "time": None,
                    "player_id": p.get("id"),
                    "type": "yellow_card",
                    "team_id": team_id,
                })
            for _ in range(n_rc):
                cards.append({
                    "time": None,
                    "player_id": p.get("id"),
                    "type": "red_card",
                    "team_id": team_id,
                })

    for side, team in rosters.items():
        team_id = team.get("id") if team else None
        if not team:
            continue
        on_subs = [p for p in team.get("players", []) if p.get("start_time") and p.get("start_time") != "00:00:00"]
        off_subs = [p for p in team.get("players", []) if p.get("starting") and p.get("end_time")]
        off_remaining = list(off_subs)
        for in_p in on_subs:
            in_time = in_p.get("start_time")
            partner = None
            for off_p in off_remaining:
                if off_p.get("end_time") == in_time:
                    partner = off_p
                    break
            if partner is None and off_remaining:
                partner = min(
                    (op for op in off_remaining if op.get("end_time")),
                    key=lambda op: op.get("end_time") or "",
                    default=None,
                )
            if partner is not None:
                off_remaining.remove(partner)
            substitutions.append({
                "in_time": in_time,
                "in_player_id": in_p.get("id"),
                "out_time": partner.get("end_time") if partner else None,
                "out_player_id": partner.get("id") if partner else None,
                "team_id": team_id,
            })

    return {
        "goals": goals,
        "cards": cards,
        "substitutions": substitutions,
        "score_final": {"home": home_score_final, "away": away_score_final},
    }


def build_cdf_metadata(match_meta: dict, df: pd.DataFrame) -> dict:
    """Build the CDF-aligned per-match metadata dict.

    Returns a dict with both a CDF nested view (``match`` / ``stadium`` /
    ``periods`` / ``whistles`` / ``meta`` / ``referees`` / ``events``) and
    a flat legacy view for backwards-compatible consumers.
    """
    teams = match_meta.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})

    metadata = dict(match_meta)
    periods = _build_cdf_periods(df, match_meta)
    metadata["coordinates_normalized_to_cdf"] = True

    kickoff_utc = _normalize_utc_iso(match_meta.get("kickoff_time_utc"))
    home_score = match_meta.get("home_score")
    away_score = match_meta.get("away_score")
    winning_team_id = _winning_team_id(match_meta)

    half_time = match_meta.get("first_half_score") or {}
    second_half = match_meta.get("second_half_score") or {}

    cdf_match_events = _build_cdf_match_events(match_meta)

    cdf_periods = []
    for p in periods:
        cdf_periods.append({
            "type": p.get("id"),
            "play_direction": p.get("play_direction"),
            "time_start": p.get("time_start"),
            "time_end": p.get("time_end"),
            "frame_id_start": p.get("frame_id_start"),
            "frame_id_end": p.get("frame_id_end"),
            "left_team_id": p.get("left_team_id"),
            "right_team_id": p.get("right_team_id"),
        })

    metadata["match"] = {
        "id": match_meta.get("match_id"),
        "competition": {
            "id": match_meta.get("competition_id"),
            "name": match_meta.get("competition_name") or match_meta.get("competition"),
        },
        "season": {
            "id": match_meta.get("season_id"),
            "name": match_meta.get("season_name"),
        },
        "match_day": match_meta.get("match_day"),
        "kickoff_time": kickoff_utc,
        "scheduled_kickoff_time": kickoff_utc,
        "result": {
            "winning_team_id": winning_team_id,
            "final": {
                "home": home_score,
                "away": away_score,
                "winning_team_id": winning_team_id,
            },
            "first_half": {
                "home": half_time.get("home"),
                "away": half_time.get("away"),
            },
            "second_half": {
                "home": second_half.get("home"),
                "away": second_half.get("away"),
            },
        },
        "status": {
            "is_neutral": bool(match_meta.get("is_neutral", False)),
            "has_extratime": bool(match_meta.get("has_extratime", False)),
            "has_shootout": bool(match_meta.get("has_shootout", False)),
        },
        "periods": cdf_periods,
        "whistles": _build_cdf_whistles(periods),
        "teams": {
            "home": {
                "id": home.get("id"),
                "name": home.get("name"),
                "players": [_cdf_player_meta_record(p, home.get("id")) for p in home.get("players", [])],
            },
            "away": {
                "id": away.get("id"),
                "name": away.get("name"),
                "players": [_cdf_player_meta_record(p, away.get("id")) for p in away.get("players", [])],
            },
        },
        "events": {
            "goals": cdf_match_events["goals"],
            "cards": cdf_match_events["cards"],
            "substitutions": cdf_match_events["substitutions"],
        },
    }
    metadata["periods"] = cdf_periods
    metadata["whistles"] = _build_cdf_whistles(periods)
    metadata["referees"] = match_meta.get("referees", [])

    metadata["stadium"] = {
        "id": match_meta.get("stadium_id"),
        "name": match_meta.get("stadium_name"),
        "capacity": match_meta.get("stadium_capacity"),
        "pitch_length": match_meta.get("pitch_length"),
        "pitch_width": match_meta.get("pitch_width"),
    }

    vendor = match_meta.get("tracking_name_original")
    fps_out = match_meta.get("fps_output")
    metadata["meta"] = {
        "vendor": vendor,
        "cdf": {"version": match_meta.get("cdf_version", CDF_VERSION)},
        "tracking": {
            "name": vendor,
            "version": str(match_meta.get("fps_original", "")) or None,
            "fps": fps_out,
            "collection_timing": "post_match",
            "name_original": vendor,
            "fps_original": match_meta.get("fps_original"),
            "fps_output": fps_out,
            "ball_status_source": match_meta.get("ball_status_source"),
        },
        "event": {
            "name": vendor,
            "version": None,
            "collection_timing": "post_match",
        },
        "meta": {
            "name": vendor,
            "version": CDF_VERSION,
        },
        "system": {
            "domain": match_meta.get("source"),
            "tracking_type": match_meta.get("tracking_type"),
        },
        "representation": {
            "coordinates_normalized_to_cdf": True,
        },
    }
    return metadata


def _cdf_player_record(row: dict) -> dict:
    """Convert one player-row record into the per-frame CDF JSONL structure."""
    player = {
        "id": _json_value(row.get("player_id")),
        "x": _json_value(row.get("x")),
        "y": _json_value(row.get("y")),
    }

    optional = {
        "position": _json_value(row.get("position_label")),
        "position_group": _json_value(row.get("position_group")),
        "is_visible": _json_value(row.get("is_visible")),
        "vel": _kmh_to_ms(row.get("speed_kmh_filtered")),
        "acc": _json_value(row.get("acceleration_ms2_filtered")),
        "dist": _json_value(row.get("distance_m")),
    }
    for key, value in optional.items():
        if value is not None:
            player[key] = value

    return player


def write_cdf_tracking_jsonl(df: pd.DataFrame, match_meta: dict, out_path: Path) -> None:
    """Write one UTF-8 JSONL tracking file with one object per frame."""
    teams = match_meta.get("teams", {})
    home_id = teams.get("home", {}).get("id")
    away_id = teams.get("away", {}).get("id")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for frame_id, frame_rows in df.groupby("frame_id", sort=True):
            records = frame_rows.to_dict("records")
            first = records[0]

            home_players: list[dict] = []
            away_players: list[dict] = []

            for row in records:
                player = _cdf_player_record(row)
                team_id = row.get("team_id")
                if team_id == home_id or (team_id is None and bool(row.get("is_home"))):
                    home_players.append(player)
                else:
                    away_players.append(player)

            ball = {
                "id": "ball",
                "x": _json_value(first.get("ball_x")),
                "y": _json_value(first.get("ball_y")),
                "z": _json_value(first.get("ball_z")),
            }
            if first.get("ball_status") is not None:
                ball["status"] = _json_value(first.get("ball_status"))
            if first.get("ball_poss_team_id") is not None:
                ball["poss_team_id"] = _json_value(first.get("ball_poss_team_id"))

            frame_obj = {
                "frame_id": _json_value(frame_id),
                "timestamp": _json_value(first.get("timestamp")),
                "period": _json_value(first.get("period")),
                "match": {"id": _json_value(first.get("match_id"))},
                "ball": ball,
                "teams": {
                    "home": {"id": home_id, "players": home_players},
                    "away": {"id": away_id, "players": away_players},
                },
            }
            f.write(json.dumps(frame_obj, ensure_ascii=False))
            f.write("\n")
