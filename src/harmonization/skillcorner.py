"""SkillCorner broadcast CV parsing: match metadata and tracking positions."""

from __future__ import annotations

import bisect
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.harmonization.utils import (
    CDF_VERSION,
    DATA_PATH,
    SC_POS_MAP,
    _pos_entry,
)

SC_DIR = DATA_PATH / "Skillcorner" / "matches"
SC_MATCHES_JSON = DATA_PATH / "Skillcorner" / "matches.json"

ASSUMED_HALFTIME_BREAK_SECS = 15 * 60  # 15 min; used for UTC derivation


def load_sc_match_meta(match_dir: Path, matches_index: dict) -> dict:
    """Load SkillCorner match metadata from match.json and matches.json index.

    Returns:
        dict with player_meta, team info, pitch dims, period info, play_direction,
        kickoff_time_utc, period_offsets
    """
    match_id = match_dir.name
    match_json = match_dir / f"{match_id}_match.json"
    phases_csv = match_dir / f"{match_id}_phases_of_play.csv"

    with open(match_json, encoding="utf-8") as f:
        mdata = json.load(f)

    home_team_id = str(mdata["home_team"]["id"])
    away_team_id = str(mdata["away_team"]["id"])
    home_team_name = mdata["home_team"].get("short_name", "")
    away_team_name = mdata["away_team"].get("short_name", "")
    pitch_length = float(mdata.get("pitch_length") or 105.0)
    pitch_width = float(mdata.get("pitch_width") or 68.0)

    home_score = mdata.get("home_team_score")
    away_score = mdata.get("away_team_score")
    comp_ed = mdata.get("competition_edition") or {}
    comp = comp_ed.get("competition") or {}
    season = comp_ed.get("season") or {}
    comp_round = mdata.get("competition_round") or {}
    stadium = mdata.get("stadium") or {}

    competition_name = comp.get("name")
    competition_id = str(comp.get("id")) if comp.get("id") is not None else None
    season_name = season.get("name")
    season_id = str(season.get("id")) if season.get("id") is not None else None
    match_day = comp_round.get("round_number")
    try:
        match_day = int(match_day) if match_day is not None else None
    except (ValueError, TypeError):
        match_day = None
    stadium_name = stadium.get("name")
    stadium_id = str(stadium.get("id")) if stadium.get("id") is not None else None
    stadium_capacity = stadium.get("capacity")
    try:
        stadium_capacity = int(stadium_capacity) if stadium_capacity is not None else None
    except (ValueError, TypeError):
        stadium_capacity = None

    referees: list[dict] = []
    for ref in mdata.get("referees", []) or []:
        rid = ref.get("id")
        if rid is None:
            continue
        referees.append({
            "id": str(rid),
            "first_name": ref.get("first_name"),
            "last_name": ref.get("last_name"),
            "short_name": ref.get("short_name"),
            "official_type": ref.get("type") or ref.get("role"),
        })

    kickoff_utc_str: str | None = None
    for entry in matches_index:
        if str(entry.get("id")) == match_id:
            kickoff_utc_str = entry.get("date_time")
            break

    period_offsets: dict[int, float] = {1: 0.0}
    match_periods = mdata.get("match_periods", [])
    p1_duration_secs = 0.0
    for mp in match_periods:
        if mp.get("period") == 1:
            p1_duration_secs = mp.get("duration_minutes", 45.0) * 60.0
    period_offsets[2] = p1_duration_secs + ASSUMED_HALFTIME_BREAK_SECS

    sc_to_cdf = {
        "right_to_left": "right_left",
        "left_to_right": "left_right",
    }
    play_direction: dict[str, str] = {}
    period_names = ["first_half", "second_half"]
    for i, entry in enumerate(mdata.get("home_team_side", [])):
        if i >= 2:
            break
        if isinstance(entry, str):
            play_direction[period_names[i]] = sc_to_cdf.get(entry, entry)
        elif isinstance(entry, dict):
            raw = entry.get("attacking_side", "unknown")
            play_direction[period_names[i]] = sc_to_cdf.get(raw, raw)

    # player_meta keyed by SkillCorner player 'id' (the integer in tracking JSONL).
    # NOT trackable_object — that identifier does not match JSONL keys.
    player_meta: dict[int, dict] = {}
    player_id_to_team: dict[str, str] = {}
    teams_roster: dict[str, dict] = {
        home_team_id: {"id": home_team_id, "name": home_team_name, "players": []},
        away_team_id: {"id": away_team_id, "name": away_team_name, "players": []},
    }

    for p in mdata.get("players", []):
        pid = p.get("id")
        tid = str(p.get("team_id", ""))
        pos_acronym = None
        role = p.get("player_role")
        if role:
            pos_acronym = role.get("acronym")
        name = p.get("short_name", "")
        shirt = p.get("number")
        started = p.get("start_time") == "00:00:00"

        pos_label, pos_group = _pos_entry(pos_acronym, SC_POS_MAP)
        if pid is not None:
            player_meta[pid] = {
                "player_id": str(pid),
                "player_name": name,
                "team_id": tid,
                "team_name": home_team_name if tid == home_team_id else away_team_name,
                "position_raw": pos_acronym,
                "position_label": pos_label,
                "position_group": pos_group,
            }
            player_id_to_team[str(pid)] = tid

        if tid in teams_roster:
            teams_roster[tid]["players"].append({
                "id": str(pid) if pid else None,
                "name": name,
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "shirt_number": int(shirt) if shirt is not None else None,
                "position_raw": pos_acronym,
                "position_label": pos_label,
                "position_group": pos_group,
                "starting": started,
                "start_time": p.get("start_time"),
                "end_time": p.get("end_time"),
                "yellow_cards": int(p.get("yellow_card") or 0),
                "red_cards": int(p.get("red_card") or 0),
                "goals": int(p.get("goal") or 0),
                "own_goals": int(p.get("own_goal") or 0),
            })

    return {
        "match_id": match_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "kickoff_time_utc": kickoff_utc_str,
        "period_offsets": period_offsets,
        "play_direction": play_direction,
        "home_score": home_score,
        "away_score": away_score,
        "competition_id": competition_id,
        "competition_name": competition_name,
        "season_name": season_name,
        "season_id": season_id,
        "match_day": match_day,
        "stadium_id": stadium_id,
        "stadium_name": stadium_name,
        "stadium_capacity": stadium_capacity,
        "referees": referees,
        "player_meta": player_meta,
        "player_id_to_team": player_id_to_team,
        "teams_roster": teams_roster,
        "phases_csv": phases_csv,
    }


def _sc_frame_clock_to_seconds(clock_str: str) -> float:
    """Convert SkillCorner frame timestamp 'HH:MM:SS.ss' to seconds."""
    try:
        parts = clock_str.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except (IndexError, ValueError):
        return 0.0


def _sc_derive_utc(
    clock_str: str,
    period: int,
    kickoff_utc_str: str | None,
    period_offsets: dict[int, float],
) -> str | None:
    """Derive UTC ISO-8601 timestamp for a SkillCorner frame."""
    if not kickoff_utc_str or not clock_str:
        return None
    try:
        kickoff_dt = datetime.fromisoformat(kickoff_utc_str.replace("Z", "+00:00"))
        clock_secs = _sc_frame_clock_to_seconds(clock_str)
        offset = period_offsets.get(period, 0.0)
        result_dt = kickoff_dt + timedelta(seconds=offset + clock_secs)
        return result_dt.isoformat()
    except Exception:
        return None


def _sc_build_ball_status_lookup(phases_csv: Path) -> list[tuple[int, int]]:
    """Build sorted list of (frame_start, frame_end) active-play intervals from phases CSV."""
    if not phases_csv.exists():
        return []
    try:
        df = pd.read_csv(phases_csv)
        intervals = []
        for _, row in df.iterrows():
            fs = row.get("frame_start")
            fe = row.get("frame_end")
            if pd.notna(fs) and pd.notna(fe):
                intervals.append((int(fs), int(fe)))
        intervals.sort(key=lambda x: x[0])
        return intervals
    except Exception as exc:
        warnings.warn(f"Could not read phases_of_play {phases_csv}: {exc}")
        return []


def _is_in_active_phase(frame_n: int, intervals: list[tuple[int, int]]) -> bool | None:
    """Binary-search whether frame_n falls within any active possession phase."""
    if not intervals:
        return None
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(starts, frame_n) - 1
    if idx >= 0 and intervals[idx][0] <= frame_n <= intervals[idx][1]:
        return True
    return False


def load_sc_match(match_dir: Path, match_id: str, matches_index: list) -> pd.DataFrame:
    """Load one SkillCorner match from raw JSONL at native 10 Hz.

    Args:
        match_dir: Directory for this match (contains JSONL, match.json, etc.)
        match_id: Match identifier string
        matches_index: Parsed matches.json list (for kickoff UTC)

    Returns:
        DataFrame with CDF-aligned columns.
    """
    print(f"  Loading SC {match_id}...")
    meta = load_sc_match_meta(match_dir, matches_index)
    jsonl_path = match_dir / f"{match_id}_tracking_extrapolated.jsonl"

    if not jsonl_path.exists():
        warnings.warn(f"Tracking JSONL not found: {jsonl_path}")
        return pd.DataFrame()

    ball_status_intervals = _sc_build_ball_status_lookup(meta["phases_csv"])
    has_phases = len(ball_status_intervals) > 0

    player_meta = meta["player_meta"]
    player_id_to_team = meta["player_id_to_team"]
    home_team_id = meta["home_team_id"]
    away_team_id = meta["away_team_id"]
    period_offsets = meta["period_offsets"]
    kickoff_utc = meta["kickoff_time_utc"]

    rows: list[dict] = []
    prev_positions: dict[tuple, tuple] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            frame_n = data.get("frame")
            if frame_n is None:
                continue

            period_int = data.get("period")
            if period_int is None:
                continue
            period_str = "first_half" if period_int == 1 else "second_half"

            clock_str = data.get("timestamp")
            timestamp_utc = _sc_derive_utc(clock_str, period_int, kickoff_utc, period_offsets)

            ball_d = data.get("ball_data") or {}
            ball_x = ball_d.get("x")
            ball_y = ball_d.get("y")
            ball_z = ball_d.get("z")

            ball_status: bool | None = None
            if has_phases:
                in_phase = _is_in_active_phase(frame_n, ball_status_intervals)
                if in_phase is not None:
                    ball_status = in_phase

            poss = data.get("possession") or {}
            poss_pid = poss.get("player_id")
            ball_poss_team_id: str | None = None
            if poss_pid is not None:
                ball_poss_team_id = player_id_to_team.get(str(poss_pid))
            if ball_poss_team_id is None:
                poss_group = poss.get("group")
                if poss_group == "home team":
                    ball_poss_team_id = home_team_id
                elif poss_group == "away team":
                    ball_poss_team_id = away_team_id

            match_minute: int | None = None
            if clock_str:
                base = 45 if period_int == 2 else 0
                match_minute = base + int(_sc_frame_clock_to_seconds(clock_str) // 60) + 1

            for p in data.get("player_data", []):
                pid = p.get("player_id")
                if pid is None:
                    continue

                is_visible = bool(p.get("is_detected", False))
                raw_x = p.get("x")
                raw_y = p.get("y")

                x = round(float(raw_x), 3) if (is_visible and raw_x is not None) else None
                y = round(float(raw_y), 3) if (is_visible and raw_y is not None) else None

                speed_kmh = None
                distance_m = None
                prev = prev_positions.get((pid, period_int))
                if is_visible and x is not None and y is not None and prev is not None:
                    dx = x - prev[0]
                    dy = y - prev[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    distance_m = round(dist, 3)
                    speed_kmh = round((dist / 0.1) * 3.6, 3)

                if is_visible and x is not None and y is not None:
                    prev_positions[(pid, period_int)] = (x, y)
                elif not is_visible:
                    prev_positions.pop((pid, period_int), None)

                pm = player_meta.get(pid, {})
                is_home = pm.get("team_id") == home_team_id

                rows.append({
                    "match_id": match_id,
                    "frame_id_source": frame_n,
                    "period": period_str,
                    "timestamp": timestamp_utc,
                    "player_id": pm.get("player_id", str(pid)),
                    "player_name": pm.get("player_name"),
                    "team_id": pm.get("team_id"),
                    "team_name": pm.get("team_name"),
                    "position_raw": pm.get("position_raw"),
                    "position_label": pm.get("position_label"),
                    "position_group": pm.get("position_group"),
                    "x": x,
                    "y": y,
                    "ball_x": round(float(ball_x), 3) if ball_x is not None else None,
                    "ball_y": round(float(ball_y), 3) if ball_y is not None else None,
                    "ball_z": round(float(ball_z), 3) if ball_z is not None else None,
                    "ball_status": ball_status,
                    "ball_poss_team_id": ball_poss_team_id,
                    "speed_kmh": speed_kmh,
                    "acceleration_ms2": None,
                    "distance_m": distance_m,
                    "match_minute": match_minute,
                    "is_visible": is_visible,
                    "is_home": is_home,
                    "source": "SkillCorner",
                    "tracking_type": "broadcast",
                    "kinematic_source": "finite_diff_native",
                    "competition": "A-League",
                })

    df = pd.DataFrame(rows)
    n_frames = df["frame_id_source"].nunique() if len(df) > 0 else 0
    print(f"    -> {len(df):,} rows, {df['player_id'].nunique()} players, {n_frames} frames")
    return df


def load_all_sc() -> tuple[pd.DataFrame, dict]:
    """Load all SkillCorner matches. Returns (DataFrame, match_metadata_dict)."""
    if not SC_MATCHES_JSON.exists():
        warnings.warn(f"matches.json not found at {SC_MATCHES_JSON}")
        matches_index = []
    else:
        with open(SC_MATCHES_JSON, encoding="utf-8") as f:
            matches_index = json.load(f)

    match_dirs = sorted([d for d in SC_DIR.iterdir() if d.is_dir()])
    match_metadata: dict[str, dict] = {}
    dfs: list[pd.DataFrame] = []

    for match_dir in match_dirs:
        match_id = match_dir.name
        df = load_sc_match(match_dir, match_id, matches_index)
        if len(df) > 0:
            dfs.append(df)

        try:
            meta = load_sc_match_meta(match_dir, matches_index)
            roster = meta["teams_roster"]
            match_metadata[match_id] = {
                "match_id": match_id,
                "source": "SkillCorner",
                "competition": "A-League",
                "kickoff_time_utc": meta["kickoff_time_utc"],
                "pitch_length": meta["pitch_length"],
                "pitch_width": meta["pitch_width"],
                "tracking_type": "broadcast",
                "tracking_name_original": "SkillCorner broadcast CV",
                "fps_original": 10,
                "fps_output": 10,
                "play_direction": meta["play_direction"],
                "ball_status_source": "phases_approx",
                "cdf_version": CDF_VERSION,
                "teams": {
                    "home": roster.get(meta["home_team_id"], {"id": meta["home_team_id"], "name": meta["home_team_name"], "players": []}),
                    "away": roster.get(meta["away_team_id"], {"id": meta["away_team_id"], "name": meta["away_team_name"], "players": []}),
                },
            }
        except Exception as exc:
            warnings.warn(f"Could not build metadata for SC {match_id}: {exc}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), match_metadata
