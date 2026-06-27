"""DFL TRACAB XML parsing: match info, play direction, substitutions, positions."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.harmonization.utils import (
    CDF_VERSION,
    DATA_PATH,
    DFL_POS_MAP,
    _pos_entry,
    _secs_to_hms,
)

DFL_DIR = DATA_PATH / "DFL"

DFL_PERIOD_MAP = {"firstHalf": "first_half", "secondHalf": "second_half"}
DFL_COMPETITION_MAP = {"DFL-COM-000001": "Bundesliga", "DFL-COM-000002": "2.Bundesliga"}


def parse_dfl_matchinfo(meta_path: Path) -> dict:
    """Parse DFL match info XML.

    Returns:
        dict with keys: home_team_id, away_team_id, pitch_length, pitch_width,
        kickoff_time_utc, match_id_official, competition, player_meta, team_names
    """
    tree = ET.parse(str(meta_path))
    root = tree.getroot()

    general = root.find(".//General")
    env = root.find(".//Environment")

    home_team_id = general.get("HomeTeamId")
    away_team_id = general.get("GuestTeamId")
    kickoff_time_utc = general.get("KickoffTime")
    match_id_official = general.get("MatchId")
    competition_id = general.get("CompetitionId", "")
    competition = DFL_COMPETITION_MAP.get(competition_id, competition_id)
    competition_name = general.get("CompetitionName")
    season_name = general.get("Season")
    season_id = general.get("SeasonId")

    home_score: int | None = None
    away_score: int | None = None
    result_str = general.get("Result", "") or ""
    if ":" in result_str:
        try:
            h, a = result_str.split(":")
            home_score = int(h)
            away_score = int(a)
        except (ValueError, TypeError):
            pass

    def _parse_score_pair(value: str | None) -> dict[str, int | None]:
        if not value or ":" not in value:
            return {"home": None, "away": None}
        try:
            h_s, a_s = value.split(":")
            return {"home": int(h_s), "away": int(a_s)}
        except (ValueError, TypeError):
            return {"home": None, "away": None}

    first_half_score = _parse_score_pair(general.get("ResultFirstHalf"))
    second_half_score = _parse_score_pair(general.get("ResultSecondHalf"))

    try:
        match_day = int(general.get("MatchDay")) if general.get("MatchDay") else None
    except (ValueError, TypeError):
        match_day = None

    pitch_length = float(env.get("PitchX", 105.0))
    pitch_width = float(env.get("PitchY", 68.0))
    stadium_name = env.get("StadiumName") if env is not None else None
    stadium_id = env.get("StadiumId") if env is not None else None
    try:
        stadium_capacity = (
            int(env.get("StadiumCapacity"))
            if env is not None and env.get("StadiumCapacity")
            else None
        )
    except (ValueError, TypeError):
        stadium_capacity = None

    referees: list[dict] = []
    for ref_elem in root.findall(".//Referee"):
        rid = ref_elem.get("PersonId") or ref_elem.get("RefereeId")
        if not rid:
            continue
        referees.append({
            "id": rid,
            "first_name": ref_elem.get("FirstName"),
            "last_name": ref_elem.get("LastName"),
            "short_name": ref_elem.get("Shortname"),
            "official_type": ref_elem.get("Role"),
        })

    team_names: dict[str, str] = {}
    player_meta: dict[str, dict] = {}
    teams_roster: dict[str, dict] = {}

    for team_elem in root.findall(".//Team"):
        tid = team_elem.get("TeamId") or team_elem.get("id")
        if not tid:
            continue
        tname = (
            team_elem.get("TeamName")
            or team_elem.get("ShortName")
            or team_elem.get("LongName")
            or team_elem.get("Name")
            or tid
        )
        team_names[tid] = tname
        team_players = []

        for player_elem in team_elem.findall(".//Player"):
            pid = player_elem.get("PersonId") or player_elem.get("PlayerId")
            if not pid:
                continue
            first = player_elem.get("FirstName", "")
            last = player_elem.get("LastName", "")
            pos = player_elem.get("PlayingPosition")
            shirt = player_elem.get("ShirtNumber")
            starting = player_elem.get("Starting", "").lower() == "true"
            name = f"{first} {last}".strip()

            pos_label, pos_group = _pos_entry(pos, DFL_POS_MAP)
            player_meta[pid] = {
                "player_name": name,
                "team_id": tid,
                "team_name": tname,
                "position_raw": pos,
                "position_label": pos_label,
                "position_group": pos_group,
            }
            team_players.append({
                "id": pid,
                "name": name,
                "first_name": first or None,
                "last_name": last or None,
                "shirt_number": int(shirt) if shirt and shirt.isdigit() else None,
                "position_raw": pos,
                "position_label": pos_label,
                "position_group": pos_group,
                "starting": starting,
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "own_goals": 0,
            })

        teams_roster[tid] = {"id": tid, "name": tname, "players": team_players}

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "kickoff_time_utc": kickoff_time_utc,
        "match_id_official": match_id_official,
        "competition": competition,
        "competition_id": competition_id,
        "competition_name": competition_name,
        "season_name": season_name,
        "season_id": season_id,
        "match_day": match_day,
        "home_score": home_score,
        "away_score": away_score,
        "first_half_score": first_half_score,
        "second_half_score": second_half_score,
        "stadium_id": stadium_id,
        "stadium_name": stadium_name,
        "stadium_capacity": stadium_capacity,
        "referees": referees,
        "player_meta": player_meta,
        "team_names": team_names,
        "teams_roster": teams_roster,
    }


def parse_dfl_play_direction(event_path: Path, home_team_id: str) -> dict[str, str]:
    """Parse DFL events XML to extract play direction per period.

    Returns:
        {'first_half': 'left_right'|'right_left',
         'second_half': 'left_right'|'right_left'}
    """
    play_direction: dict[str, str] = {}
    try:
        tree = ET.parse(str(event_path))
        root = tree.getroot()
        for kickoff in root.findall(".//KickOff"):
            section = kickoff.get("GameSection", "")
            period = DFL_PERIOD_MAP.get(section)
            if not period:
                continue
            team_left = kickoff.get("TeamLeft")
            if team_left == home_team_id:
                play_direction[period] = "left_right"
            else:
                play_direction[period] = "right_left"
    except Exception as exc:
        warnings.warn(f"Could not parse play direction from {event_path}: {exc}")

    if "first_half" in play_direction and "second_half" not in play_direction:
        play_direction["second_half"] = (
            "right_left" if play_direction["first_half"] == "left_right"
            else "left_right"
        )
    elif "second_half" in play_direction and "first_half" not in play_direction:
        play_direction["first_half"] = (
            "right_left" if play_direction["second_half"] == "left_right"
            else "left_right"
        )

    return play_direction


def parse_dfl_substitutions(
    event_path: Path | None, kickoff_utc_str: str | None
) -> dict[str, dict]:
    """Extract per-player substitution timings from the DFL events XML.

    Returns:
        {player_id: {"sub_in_secs": float, "sub_out_secs": float, "in_position": str}}
    """
    result: dict[str, dict] = {}
    if not event_path or not event_path.exists() or not kickoff_utc_str:
        return result
    try:
        kickoff_dt = datetime.fromisoformat(kickoff_utc_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return result

    try:
        tree = ET.parse(str(event_path))
        root = tree.getroot()
    except Exception as exc:
        warnings.warn(f"Could not parse DFL events {event_path}: {exc}")
        return result

    for event_elem in root.findall(".//Event"):
        event_time = event_elem.get("EventTime")
        sub = event_elem.find("Substitution")
        if sub is None or not event_time:
            continue
        try:
            ev_dt = datetime.fromisoformat(event_time)
        except (ValueError, TypeError):
            continue
        elapsed = (ev_dt - kickoff_dt).total_seconds()
        if elapsed < 0:
            continue
        player_in = sub.get("PlayerIn")
        player_out = sub.get("PlayerOut")
        position = sub.get("PlayingPosition")
        if player_in:
            entry = result.setdefault(player_in, {})
            entry["sub_in_secs"] = elapsed
            if position:
                entry["in_position"] = position
        if player_out:
            result.setdefault(player_out, {})["sub_out_secs"] = elapsed
    return result


def parse_dfl_player_stats(
    event_path: Path | None, player_team: dict[str, str]
) -> dict[str, dict]:
    """Extract per-player card and goal counts from the DFL events XML.

    Returns:
        {player_id: {"yellow_cards": int, "red_cards": int, "goals": int, "own_goals": int}}
    """
    result: dict[str, dict] = {}
    if not event_path or not event_path.exists():
        return result
    try:
        tree = ET.parse(str(event_path))
        root = tree.getroot()
    except Exception as exc:
        warnings.warn(f"Could not parse DFL events for stats {event_path}: {exc}")
        return result

    def _entry(pid: str) -> dict:
        return result.setdefault(
            pid,
            {"yellow_cards": 0, "red_cards": 0, "goals": 0, "own_goals": 0},
        )

    for caution in root.findall(".//Caution"):
        pid = caution.get("Player")
        color = caution.get("CardColor")
        if not pid or not color:
            continue
        if color == "yellow":
            _entry(pid)["yellow_cards"] += 1
        elif color in ("red", "yellowRed"):
            _entry(pid)["red_cards"] += 1

    for shot in root.findall(".//ShotAtGoal"):
        if shot.find("SuccessfulShot") is None:
            continue
        pid = shot.get("Player")
        if not pid:
            continue
        shot_team = shot.get("Team")
        scorer_team = player_team.get(pid)
        entry = _entry(pid)
        if shot_team and scorer_team and shot_team != scorer_team:
            entry["own_goals"] += 1
        else:
            entry["goals"] += 1

    return result


def apply_dfl_player_stats(
    teams_roster: dict, stats: dict[str, dict]
) -> None:
    """Apply per-player card/goal counts to the DFL roster in-place."""
    for team in teams_roster.values():
        for p in team.get("players", []):
            pid = p.get("id")
            s = stats.get(pid, {}) if pid else {}
            p["yellow_cards"] = s.get("yellow_cards", 0)
            p["red_cards"] = s.get("red_cards", 0)
            p["goals"] = s.get("goals", 0)
            p["own_goals"] = s.get("own_goals", 0)


def apply_dfl_sub_timings(
    teams_roster: dict, sub_timings: dict[str, dict]
) -> None:
    """Augment each DFL roster player with SkillCorner-style start/end time fields."""
    for team in teams_roster.values():
        for p in team.get("players", []):
            pid = p.get("id")
            timings = sub_timings.get(pid, {}) if pid else {}
            if p.get("starting"):
                p["start_time"] = "00:00:00"
                p["end_time"] = _secs_to_hms(timings.get("sub_out_secs"))
                continue
            sub_in = timings.get("sub_in_secs")
            if sub_in is None:
                p["start_time"] = None
                p["end_time"] = None
                if p.get("position_raw") is None:
                    p["position_raw"] = "SUB"
                    p["position_label"] = None
                    p["position_group"] = "SUB"
            else:
                p["start_time"] = _secs_to_hms(sub_in)
                p["end_time"] = _secs_to_hms(timings.get("sub_out_secs"))
                if p.get("position_raw") is None and timings.get("in_position"):
                    in_pos = timings["in_position"]
                    label, group = _pos_entry(in_pos, DFL_POS_MAP)
                    p["position_raw"] = in_pos
                    p["position_label"] = label
                    p["position_group"] = group


def _dfl_positions_pass1(raw_path: Path) -> tuple[dict, dict, dict]:
    """First pass: collect ball data, frame timestamps, and frame lists per period.

    Returns:
        ball_data: {N: {x, y, z, ball_status, ball_poss}}
        frame_meta: {N: (timestamp_str, period_str)}
        selected_by_period: {period_str: [(position_in_sorted, N), ...]} (downsampled)
    """
    ball_data: dict[int, dict] = {}
    frame_meta: dict[int, tuple] = {}
    frames_by_period: dict[str, set] = {}

    current_team: str | None = None
    current_period: str | None = None

    for event, elem in ET.iterparse(str(raw_path), events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_team = elem.get("TeamId")
            current_period = DFL_PERIOD_MAP.get(
                elem.get("GameSection", ""), elem.get("GameSection", "")
            )

        elif event == "end" and elem.tag == "Frame":
            n_str = elem.get("N")
            if n_str is None:
                elem.clear()
                continue
            n = int(n_str)
            t = elem.get("T")

            if n not in frame_meta:
                m_str = elem.get("M")
                frame_meta[n] = (t, current_period, int(m_str) if m_str else None)
                frames_by_period.setdefault(current_period, set()).add(n)

            if current_team == "BALL":
                bs = elem.get("BallStatus")
                bp = elem.get("BallPossession")
                x_str = elem.get("X")
                y_str = elem.get("Y")
                ball_data[n] = {
                    "x": float(x_str) if x_str else None,
                    "y": float(y_str) if y_str else None,
                    "z": float(elem.get("Z", 0.0)),
                    "ball_status": (int(bs) == 1) if bs is not None else None,
                    "ball_poss": int(bp) if bp is not None else None,
                }
            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()

    # Sort and downsample (25 Hz → 10 Hz via alternating step 2/3)
    selected_by_period: dict[str, list] = {}
    for period, ns in frames_by_period.items():
        sorted_ns = sorted(ns)
        sel = []
        i = 0
        step2 = True
        while i < len(sorted_ns):
            sel.append(sorted_ns[i])
            i += 2 if step2 else 3
            step2 = not step2
        selected_by_period[period] = sel

    return ball_data, frame_meta, selected_by_period


def _dfl_positions_pass2(
    raw_path: Path,
    ball_data: dict,
    frame_meta: dict,
    selected_by_period: dict,
    player_meta: dict,
    home_team_id: str,
    away_team_id: str,
    match_id: str,
    competition: str,
) -> list[dict]:
    """Second pass: collect per-player positions and build output rows."""
    rows: list[dict] = []

    selected_sets = {p: set(ns) for p, ns in selected_by_period.items()}

    current_team: str | None = None
    current_pid: str | None = None
    current_period: str | None = None
    player_frames: dict[int, tuple] = {}

    for event, elem in ET.iterparse(str(raw_path), events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_team = elem.get("TeamId")
            current_pid = elem.get("PersonId")
            current_period = DFL_PERIOD_MAP.get(
                elem.get("GameSection", ""), elem.get("GameSection", "")
            )
            player_frames = {}

        elif event == "end" and elem.tag == "Frame":
            if current_team not in ("BALL", "referee") and current_team is not None:
                n_str = elem.get("N")
                x_str = elem.get("X")
                y_str = elem.get("Y")
                if n_str and x_str and y_str:
                    n = int(n_str)
                    if n in selected_sets.get(current_period, set()):
                        s_str = elem.get("S")
                        a_str = elem.get("A")
                        player_frames[n] = (
                            float(x_str),
                            float(y_str),
                            float(s_str) if s_str else None,
                            float(a_str) if a_str else None,
                        )
            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            if (
                current_team not in ("BALL", "referee")
                and current_team is not None
                and current_pid is not None
            ):
                meta = player_meta.get(current_pid, {})
                is_home = meta.get("team_id") == home_team_id
                sel_ns = selected_by_period.get(current_period, [])

                prev_x = prev_y = None

                for n in sel_ns:
                    pos = player_frames.get(n)
                    t_str, _, match_minute = frame_meta.get(n, (None, None, None))

                    if pos is None:
                        prev_x = prev_y = None
                        continue

                    x, y, native_s, native_a = pos
                    # Native TRACAB S (km/h) and A (m/s²) are computed by the
                    # vendor from the full 25 Hz trajectory before downsampling,
                    # avoiding the quantisation loss inherent in 10 Hz position
                    # finite-differences. distance_m is still derived from
                    # downsampled positions (native D is not a per-frame value).
                    speed_kmh = round(native_s, 3) if native_s is not None else None
                    acceleration_ms2 = round(native_a, 3) if native_a is not None else None
                    distance_m = None
                    if prev_x is not None:
                        dx = x - prev_x
                        dy = y - prev_y
                        distance_m = round(np.sqrt(dx**2 + dy**2), 3)

                    ball = ball_data.get(n, {})
                    bp_raw = ball.get("ball_poss")
                    if bp_raw == 1:
                        ball_poss_team_id = home_team_id
                    elif bp_raw == 2:
                        ball_poss_team_id = away_team_id
                    else:
                        ball_poss_team_id = None

                    rows.append({
                        "match_id": match_id,
                        "frame_id_source": n,
                        "period": current_period,
                        "timestamp": t_str,
                        "player_id": current_pid,
                        "player_name": meta.get("player_name"),
                        "team_id": meta.get("team_id"),
                        "team_name": meta.get("team_name"),
                        "position_raw": meta.get("position_raw"),
                        "position_label": meta.get("position_label"),
                        "position_group": meta.get("position_group"),
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "ball_x": round(ball["x"], 3) if ball.get("x") is not None else None,
                        "ball_y": round(ball["y"], 3) if ball.get("y") is not None else None,
                        "ball_z": round(ball.get("z", 0.0), 3),
                        "ball_status": ball.get("ball_status"),
                        "ball_poss_team_id": ball_poss_team_id,
                        "speed_kmh": speed_kmh,
                        "acceleration_ms2": acceleration_ms2,
                        "distance_m": distance_m,
                        "match_minute": match_minute,
                        "is_visible": True,
                        "is_home": is_home,
                        "source": "DFL",
                        "tracking_type": "in_stadium",
                        "kinematic_source": "native_tracab",
                        "competition": competition,
                    })

                    prev_x, prev_y = x, y

            elem.clear()

    return rows


def load_dfl_match(
    meta_path: Path, event_path: Path | None, raw_path: Path, match_id: str
) -> pd.DataFrame:
    """Load one DFL match from raw XML, downsample to 10 Hz, return CDF-aligned rows.

    Args:
        meta_path: Path to DFL_02_01_matchinformation_*.xml
        event_path: Path to DFL_03_02_events_raw_*.xml (for play_direction)
        raw_path: Path to DFL_04_03_positions_raw_observed_*.xml
        match_id: Identifier string for this match

    Returns:
        DataFrame with CDF-aligned columns.
    """
    print(f"  Loading DFL {match_id}...")
    info = parse_dfl_matchinfo(meta_path)

    play_direction: dict[str, str] = {}
    if event_path and event_path.exists():
        play_direction = parse_dfl_play_direction(event_path, info["home_team_id"])

    print(f"    Pass 1: indexing frames and ball data...")
    ball_data, frame_meta, selected_by_period = _dfl_positions_pass1(raw_path)

    total_selected = sum(len(v) for v in selected_by_period.values())
    print(f"    Selected {total_selected} frames across {len(selected_by_period)} periods")

    print(f"    Pass 2: building player rows...")
    rows = _dfl_positions_pass2(
        raw_path=raw_path,
        ball_data=ball_data,
        frame_meta=frame_meta,
        selected_by_period=selected_by_period,
        player_meta=info["player_meta"],
        home_team_id=info["home_team_id"],
        away_team_id=info["away_team_id"],
        match_id=match_id,
        competition=info["competition"],
    )

    df = pd.DataFrame(rows)
    n_frames = df["frame_id_source"].nunique() if len(df) > 0 else 0
    print(f"    -> {len(df):,} rows, {df['player_id'].nunique()} players, {n_frames} frames")

    df.attrs["play_direction"] = play_direction
    df.attrs["pitch_length"] = info["pitch_length"]
    df.attrs["pitch_width"] = info["pitch_width"]
    df.attrs["kickoff_time_utc"] = info["kickoff_time_utc"]
    df.attrs["home_team_id"] = info["home_team_id"]
    df.attrs["away_team_id"] = info["away_team_id"]

    return df


def load_all_dfl() -> tuple[pd.DataFrame, dict]:
    """Load all 7 DFL matches. Returns (DataFrame, match_metadata_dict)."""
    meta_files = sorted(DFL_DIR.glob("*matchinformation*.xml"))
    pos_files = sorted(DFL_DIR.glob("*positions_raw*.xml"))
    event_files = sorted(DFL_DIR.glob("*events_raw*.xml"))

    event_map: dict[str, Path] = {}
    for ef in event_files:
        parts = ef.stem.split("_")
        mid = parts[-1]
        event_map[mid] = ef

    match_metadata: dict[str, dict] = {}
    dfs: list[pd.DataFrame] = []

    for meta_path, raw_path in zip(meta_files, pos_files):
        parts = meta_path.stem.split("_")
        match_id = parts[-1]
        event_path = event_map.get(match_id)

        df = load_dfl_match(meta_path, event_path, raw_path, match_id)
        dfs.append(df)

        info = parse_dfl_matchinfo(meta_path)
        play_dir = parse_dfl_play_direction(event_path, info["home_team_id"]) if event_path else {}
        roster = info["teams_roster"]
        match_metadata[match_id] = {
            "match_id": match_id,
            "source": "DFL",
            "competition": info["competition"],
            "kickoff_time_utc": info["kickoff_time_utc"],
            "pitch_length": info["pitch_length"],
            "pitch_width": info["pitch_width"],
            "tracking_type": "in_stadium",
            "tracking_name_original": "TRACAB Gen5",
            "fps_original": 25,
            "fps_output": 10,
            "play_direction": play_dir,
            "cdf_version": CDF_VERSION,
            "teams": {
                "home": roster.get(info["home_team_id"], {"id": info["home_team_id"], "name": "", "players": []}),
                "away": roster.get(info["away_team_id"], {"id": info["away_team_id"], "name": "", "players": []}),
            },
        }

    return pd.concat(dfs, ignore_index=True), match_metadata
