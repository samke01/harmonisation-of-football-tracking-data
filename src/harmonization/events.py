"""
SPADL-based event normalization plus CDF event export for DFL and SkillCorner.

Converts both event formats into a unified SPADL-inspired schema
(Decroos et al., 2019) with the following columns per action:

    match_id, timestamp, action_type, player_id, player_name,
    team_id, team_name, result, start_x, start_y, end_x, end_y,
    source, competition, cross_dataset_comparable

DFL events are discrete on-ball actions (Pass, Shot, Tackle, etc.).
SkillCorner events are possession-level snapshots — we extract the
on-ball action at the END of each player_possession + on_ball_engagements.

Cross-dataset comparability:
    Only action types present with consistent semantics in BOTH sources
    are flagged as cross_dataset_comparable=True. Source-specific action
    types (tackle, interception, pressing, etc.) are retained but flagged
    as False to prevent invalid cross-dataset comparisons.

    body_part is excluded from the schema because SkillCorner does not
    record this information — including it would create a systematically
    incomplete column that could mislead downstream analysis.

Usage:
    python -m src.harmonization.events
"""

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.config import settings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path(settings.data_path)
DFL_DIR = DATA_PATH / "DFL"
SC_DIR = DATA_PATH / "Skillcorner" / "matches"
MATCHES_DIR = DATA_PATH / "merged" / "matches"

# Shot outcome labels shared across ShotAtGoal, FreeKick/ShotAtGoal, Penalty/ShotAtGoal.
# "success" (not "successful") keeps the result token consistent with pass/foul/clearance
# so that result == "success" is a reliable goal signal across both sources.
# OwnGoal is not present in the current 7-match DFL sample but is included defensively;
# it maps to "own_goal" which _result_success_flag treats as False (unsuccessful for the
# shooting team) per SPADL Table 4.
_DFL_SHOT_OUTCOME = {
    "SuccessfulShot": "success",
    "OwnGoal": "own_goal",
    "SavedShot": "saved",
    "BlockedShot": "blocked",
    "ShotWide": "wide",
    "ShotWoodWork": "woodwork",
}

# SPADL action types (subset relevant to our data)
# Based on Decroos et al. (2019), Table A.1, plus "pressing" extension
ACTION_TYPES = [
    "pass", "cross", "throw_in", "freekick_short", "freekick_cross",
    "corner_short", "corner_cross", "goalkick", "shot", "shot_freekick",
    "shot_penalty", "tackle", "interception", "clearance", "dribble",
    "foul", "keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up",
    "take_on", "bad_touch", "non_action", "pressing",
]

# Action types that are semantically comparable across DFL and SkillCorner.
# Only these should be used for cross-dataset aggregation / modelling.
# Criteria: both sources must have NON-ZERO counts of the action type with
# broadly consistent counting logic.
# Excluded despite DFL having them:
#   shot_penalty: SC does not capture the penalty kick as a possession event
#     (structural gap); SC count is 0 → not comparable.
#   freekick_short / corner_short / goalkick: SC has no set-piece kick
#     extraction; SC counts receptions not kicks → structural gap.
# shot_freekick IS comparable: SC detects it via game_interruption_before =
#   free_kick_for on shot possessions (~1.1/match each source).
CROSS_DATASET_COMPARABLE = {"pass", "shot", "shot_freekick", "throw_in"}

# DFL event data: corner-origin (0→L, 0→W); convert to CDF centre-origin metres
DFL_PITCH_LENGTH = 105.0
DFL_PITCH_WIDTH = 68.0

# SkillCorner event data: already centre-origin metres — no conversion needed
SC_PITCH_LENGTH = 105.0
SC_PITCH_WIDTH = 68.0
ASSUMED_HALFTIME_BREAK_SECS = 15 * 60
DFL_PERIOD_MAP = {"firstHalf": "first_half", "secondHalf": "second_half"}
SC_PERIOD_MAP = {1: "first_half", 2: "second_half", 3: "first_half_extratime", 4: "second_half_extratime"}


def _normalize_utc_iso(raw: str | None) -> str | None:
    """Normalise an ISO-8601 timestamp to a UTC offset string."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return str(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _flip_event_x(x, period: str | None, play_direction: dict[str, str]) -> float:
    """Flip x when the original period orientation had the home team right-to-left."""
    if pd.isna(x):
        return np.nan
    if period and play_direction.get(period) == "right_left":
        return -float(x)
    return float(x)


def _json_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    return None if pd.isna(value) else value


def _result_success_flag(result: str | None) -> bool | None:
    """True only when the action ended in the (cross-source) success state.

    All action types use ``"success"`` as the unified success token.
    Shot-detail outcomes (``saved``, ``blocked``, ``wide``, ``woodwork``)
    are unsuccessful from the shooter's perspective → ``False``.
    ``own_goal`` is also ``False`` (unsuccessful for the shooting team).
    """
    if result == "success":
        return True
    if result in {"fail", "offside", "saved", "blocked", "wide", "woodwork", "own_goal"}:
        return False
    return None


def _coerce_id(value) -> str | None:
    """Cast a possibly-numeric ID to a clean string ('51649' not '51649.0').

    pandas reads numeric ID columns as float when nulls are present, so a
    naive ``str(value)`` produces the trailing-``.0`` artefact. We try
    ``int`` first via ``float`` to handle string-encoded floats too
    (``'51649.0'`` → ``51649``).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _cdf_type_and_subtype(action_type: str) -> tuple[str, str | None]:
    """Map our SPADL-style action_type to (CDF event type, sub_type).

    CDF Table 3 closed sets:
      - shot.sub_type ∈ {None, penalty_kick, free_kick, corner_kick}
      - pass.sub_type ∈ {None, throw_in, free_kick, corner_kick, goal_kick, kick_off}
      - misc.sub_type ∈ {other_ball_action, chance_without_shot, tackle}
      - referee.sub_type ∈ {final_whistle, foul, caution, offside, substitution,
        player_on, player_off}
    """
    mapping = {
        "pass": ("pass", None),
        "cross": ("pass", None),        # CDF has no cross sub_type; treat as open-play pass
        "freekick_short": ("pass", "free_kick"),
        "freekick_cross": ("pass", "free_kick"),
        "corner_short": ("pass", "corner_kick"),
        "corner_cross": ("pass", "corner_kick"),
        "throw_in": ("pass", "throw_in"),
        "goalkick": ("pass", "goal_kick"),
        "shot": ("shot", None),
        "shot_freekick": ("shot", "free_kick"),
        "shot_penalty": ("shot", "penalty_kick"),
        "foul": ("referee", "foul"),
        "clearance": ("misc", "other_ball_action"),
        "bad_touch": ("misc", "other_ball_action"),
        "non_action": ("misc", "other_ball_action"),
    }
    return mapping.get(action_type, ("misc", "other_ball_action"))


def _cdf_outcome_type(action_type: str, result: str | None) -> str | None:
    """Map our internal result to a CDF Table 3 outcome value.

    CDF closed sets:
      - shot: successful, saved, blocked, wide, woodwork, own_goal
      - pass: successful, out_of_play, intercepted (e.g.)
      - referee: start, end, injury, yellow_card, red_card, second_yellow_card
      - misc: successful, unsuccessful

    Shot-detail values (saved / blocked / wide / woodwork / own_goal) are
    enriched at parse time and arrive here as ``result`` values. Generic
    "fail" / "success" fall back to the binary mapping.
    """
    # CDF shot outcome_type enum: {successful, saved, blocked, wide, woodwork, own_goal}.
    # The result column now uses "success" (not "successful") as the goal token, so
    # "success" is mapped → "successful" here. Detail tokens (saved, blocked, …) come
    # from DFL only; SC exposes only lead_to_goal so its non-goal shots arrive as
    # result="fail" and return None (CDF §5.2: explicit null for unknown detail).
    shot_detail = {"saved", "blocked", "wide", "woodwork", "own_goal"}
    if action_type.startswith("shot"):
        if result in shot_detail:
            return result
        if result == "success":
            return "successful"
        return None
    if action_type in {"pass", "cross", "freekick_short", "freekick_cross", "corner_short", "corner_cross", "throw_in", "goalkick"}:
        if result == "success":
            return "successful"
        # offside ends in a dead ball / restart: nearest CDF pass outcome.
        if result in {"offside", "fail"}:
            return "out_of_play"
        return None
    if action_type == "foul":
        # CDF referee outcomes describe whistle states, not foul markers.
        return None
    if action_type in {"clearance"}:
        return "successful"
    if action_type in {"bad_touch", "non_action"}:
        return "unsuccessful"
    return None


def _write_cdf_event_jsonl(match_df: pd.DataFrame, out_path: Path) -> None:
    """Write one UTF-8 JSONL event file with one object per event."""
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in match_df.to_dict("records"):
            event_type, event_sub_type = _cdf_type_and_subtype(row["action_type"])
            event_obj = {
                "match": {"id": _json_value(row["match_id"])},
                "meta": {"is_synced": True},
                "event": {
                    "id": _json_value(row.get("event_id")),
                    "time": _json_value(row.get("timestamp")),
                    "period": _json_value(row.get("period")),
                    "type": event_type,
                    "sub_type": event_sub_type,
                    "is_successful": _result_success_flag(row.get("result")),
                    "outcome_type": _cdf_outcome_type(row["action_type"], row.get("result")),
                    "player_id": _json_value(row.get("player_id")),
                    "team_id": _json_value(row.get("team_id")),
                    "receiver_id": _json_value(row.get("receiver_id")),
                    "receiver_time": _json_value(row.get("receiver_time")),
                    "x": _json_value(row.get("start_x")),
                    "y": _json_value(row.get("start_y")),
                    "x_end": _json_value(row.get("end_x")),
                    "y_end": _json_value(row.get("end_y")),
                    "body_part": _json_value(row.get("body_part")),
                    "related_event_ids": _json_value(row.get("related_event_ids")),
                },
            }
            f.write(json.dumps(event_obj, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# DFL helpers
# ---------------------------------------------------------------------------

def _parse_dfl_match_info(match_info_path):
    """Extract player/team lookup from DFL match information XML."""
    tree = ET.parse(match_info_path)
    root = tree.getroot()

    # Match ID
    general = root.find(".//General")
    match_id = general.get("MatchId") if general is not None and general.get("MatchId") else None
    home_team_id = general.get("HomeTeamId") if general is not None else None
    away_team_id = general.get("GuestTeamId") if general is not None else None

    players = {}  # person_id -> {name, team_id, team_name, position}
    teams = {}    # team_id -> team_name

    for team in root.findall(".//Teams/Team"):
        tid = team.get("TeamId")
        tname = team.get("TeamName")
        teams[tid] = tname
        for player in team.findall(".//Player"):
            pid = player.get("PersonId")
            players[pid] = {
                "player_name": player.get("Shortname", ""),
                "team_id": tid,
                "team_name": tname,
                "position": player.get("PlayingPosition", ""),
            }

    return {
        "match_id": match_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "players": players,
        "teams": teams,
    }


def _parse_dfl_play_direction_and_kickoffs(root: ET.Element, home_team_id: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """Read home-team play direction and kickoff timestamps per period from XML."""
    play_direction: dict[str, str] = {}
    kickoff_times: dict[str, str] = {}
    for event_elem in root.findall(".//Event"):
        kickoff = event_elem.find("KickOff")
        if kickoff is None:
            continue
        game_section = kickoff.get("GameSection")
        period = DFL_PERIOD_MAP.get(game_section)
        if period is None:
            continue
        timestamp = _normalize_utc_iso(event_elem.get("EventTime"))
        if timestamp:
            kickoff_times[period] = timestamp
        team_left = kickoff.get("TeamLeft")
        if home_team_id and team_left == home_team_id:
            play_direction[period] = "left_right"
        elif home_team_id:
            play_direction[period] = "right_left"
    if "first_half" in play_direction and "second_half" not in play_direction:
        play_direction["second_half"] = "right_left" if play_direction["first_half"] == "left_right" else "left_right"
    return play_direction, kickoff_times


def _infer_dfl_period(timestamp: str | None, kickoff_times: dict[str, str]) -> str | None:
    """Infer DFL event period from UTC event timestamps and second-half kickoff."""
    if timestamp is None:
        return None
    try:
        event_dt = datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return None
    second_half = kickoff_times.get("second_half")
    if second_half:
        try:
            second_half_dt = datetime.fromisoformat(second_half)
            if event_dt >= second_half_dt:
                return "second_half"
        except (ValueError, TypeError):
            pass
    return "first_half"



def _dfl_normalize_coords(x, y):
    """Convert DFL event coords (corner-origin 0–105, 0–68) to CDF centre-origin metres."""
    nx = float(x) - DFL_PITCH_LENGTH / 2 if x else np.nan
    ny = float(y) - DFL_PITCH_WIDTH / 2 if y else np.nan
    return nx, ny


def load_dfl_events(events_path, match_info_path):
    """Parse one DFL event XML into SPADL-format rows."""
    match_id_from_file = events_path.stem.split("_")[-1]
    match_info = _parse_dfl_match_info(match_info_path)
    players = match_info["players"]
    teams = match_info["teams"]

    tree = ET.parse(events_path)
    root = tree.getroot()
    play_direction, kickoff_times = _parse_dfl_play_direction_and_kickoffs(root, match_info["home_team_id"])

    rows = []

    for event_elem in root.findall(".//Event"):
        ea = event_elem.attrib
        timestamp = _normalize_utc_iso(ea.get("EventTime"))
        period = _infer_dfl_period(timestamp, kickoff_times)
        start_x_raw, start_y = _dfl_normalize_coords(
            ea.get("X-Source-Position", ea.get("X-Position")),
            ea.get("Y-Source-Position", ea.get("Y-Position")),
        )
        end_x_raw, end_y = _dfl_normalize_coords(
            ea.get("X-Position"), ea.get("Y-Position")
        )
        start_x = _flip_event_x(start_x_raw, period, play_direction)
        end_x = _flip_event_x(end_x_raw, period, play_direction)

        # Base row template
        base = {
            "event_id": ea.get("EventId"),
            "match_id": match_id_from_file,
            "timestamp": timestamp,
            "period": period,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "source": "DFL",
            "body_part": None,
            "receiver_id": None,
            "receiver_time": None,
            "related_event_ids": None,
        }

        # --- Open-play Pass / Cross ---
        # event_elem.find("Play") matches only DIRECT children. Set-piece
        # passes are nested inside FreeKick/CornerKick/GoalKick/ThrowIn/
        # KickOff wrappers and are handled by their own blocks below.
        #
        # A <Cross> child inside <Play> marks a crossing pass (SPADL Table 4:
        # "cross"). We emit action_type="cross" for these (~20/match) rather
        # than silently collapsing into "pass". cross_dataset_comparable=False
        # (set downstream via CROSS_DATASET_COMPARABLE) because SkillCorner
        # cannot distinguish crosses from ordinary passes in its event schema.
        play = event_elem.find("Play")
        if play is not None:
            pa = play.attrib
            player_id = pa.get("Player", "")
            pinfo = players.get(player_id, {})
            result = "success" if pa.get("Evaluation") == "successfullyCompleted" else "fail"
            receiver_id = pa.get("Recipient") if result == "success" else None
            is_cross = play.find("Cross") is not None
            rows.append({
                **base,
                "action_type": "cross" if is_cross else "pass",
                "player_id": player_id,
                "player_name": pinfo.get("player_name", ""),
                "team_id": pinfo.get("team_id", pa.get("Team", "")),
                "team_name": pinfo.get("team_name", ""),
                "receiver_id": receiver_id,
                "receiver_time": None,
                "result": result,
            })
            continue

        # --- Shot (open play, direct free-kick shot not inside FreeKick wrapper) ---
        shot = event_elem.find("ShotAtGoal")
        if shot is not None:
            sa = shot.attrib
            player_id = sa.get("Player", "")
            pinfo = players.get(player_id, {})
            result = "fail"
            for child in shot:
                mapped = _DFL_SHOT_OUTCOME.get(child.tag)
                if mapped is not None:
                    result = mapped
                    break
            # TakerSetup=penalty: penalty kick. TakerSetup=freeKick AND
            # BuildUp=freeKick: direct free-kick shot (not a rebound after FK).
            if sa.get("TakerSetup") == "penalty":
                action_type = "shot_penalty"
            elif sa.get("TakerSetup") == "freeKick" and sa.get("BuildUp") == "freeKick":
                action_type = "shot_freekick"
            else:
                action_type = "shot"
            rows.append({
                **base,
                "action_type": action_type,
                "player_id": player_id,
                "player_name": pinfo.get("player_name", ""),
                "team_id": pinfo.get("team_id", sa.get("Team", "")),
                "team_name": pinfo.get("team_name", ""),
                "result": result,
            })
            continue

        # --- FreeKick wrapper: freekick_short pass or shot_freekick ---
        # DFL nests Play or ShotAtGoal inside the FreeKick element.
        fk = event_elem.find("FreeKick")
        if fk is not None:
            fk_play = fk.find("Play")
            if fk_play is not None:
                pa = fk_play.attrib
                player_id = pa.get("Player", "")
                pinfo = players.get(player_id, {})
                result = "success" if pa.get("Evaluation") == "successfullyCompleted" else "fail"
                receiver_id = pa.get("Recipient") if result == "success" else None
                # Cross child → SPADL "Crossed free-kick"; otherwise "Short free-kick"
                fk_type = "freekick_cross" if fk_play.find("Cross") is not None else "freekick_short"
                rows.append({
                    **base,
                    "action_type": fk_type,
                    "player_id": player_id,
                    "player_name": pinfo.get("player_name", ""),
                    "team_id": pinfo.get("team_id", pa.get("Team", "")),
                    "team_name": pinfo.get("team_name", ""),
                    "receiver_id": receiver_id,
                    "receiver_time": None,
                    "result": result,
                })
            else:
                fk_shot = fk.find("ShotAtGoal")
                if fk_shot is not None:
                    sa = fk_shot.attrib
                    player_id = sa.get("Player", "")
                    pinfo = players.get(player_id, {})
                    result = "fail"
                    for child in fk_shot:
                        mapped = _DFL_SHOT_OUTCOME.get(child.tag)
                        if mapped is not None:
                            result = mapped
                            break
                    rows.append({
                        **base,
                        "action_type": "shot_freekick",
                        "player_id": player_id,
                        "player_name": pinfo.get("player_name", ""),
                        "team_id": pinfo.get("team_id", sa.get("Team", "")),
                        "team_name": pinfo.get("team_name", ""),
                        "result": result,
                    })
            continue

        # --- CornerKick wrapper: corner_cross or corner_short ---
        # 53/57 DFL corners have a Cross child → SPADL "Crossed corner".
        # Only 4 are played short → SPADL "Short corner".
        ck = event_elem.find("CornerKick")
        if ck is not None:
            ck_play = ck.find("Play")
            if ck_play is not None:
                pa = ck_play.attrib
                player_id = pa.get("Player", "")
                pinfo = players.get(player_id, {})
                result = "success" if pa.get("Evaluation") == "successfullyCompleted" else "fail"
                receiver_id = pa.get("Recipient") if result == "success" else None
                ck_type = "corner_cross" if ck_play.find("Cross") is not None else "corner_short"
                rows.append({
                    **base,
                    "action_type": ck_type,
                    "player_id": player_id,
                    "player_name": pinfo.get("player_name", ""),
                    "team_id": pinfo.get("team_id", pa.get("Team", "")),
                    "team_name": pinfo.get("team_name", ""),
                    "receiver_id": receiver_id,
                    "receiver_time": None,
                    "result": result,
                })
            continue

        # --- Penalty wrapper: shot_penalty ---
        # DFL nests ShotAtGoal inside the Penalty element (3 events).
        penalty = event_elem.find("Penalty")
        if penalty is not None:
            pen_shot = penalty.find("ShotAtGoal")
            if pen_shot is not None:
                sa = pen_shot.attrib
                player_id = sa.get("Player", "")
                pinfo = players.get(player_id, {})
                result = "fail"
                for child in pen_shot:
                    mapped = _DFL_SHOT_OUTCOME.get(child.tag)
                    if mapped is not None:
                        result = mapped
                        break
                rows.append({
                    **base,
                    "action_type": "shot_penalty",
                    "player_id": player_id,
                    "player_name": pinfo.get("player_name", ""),
                    "team_id": pinfo.get("team_id", sa.get("Team", "")),
                    "team_name": pinfo.get("team_name", ""),
                    "result": result,
                })
            continue

        # --- KickOff ---
        # EXCLUDED: Kick-offs (32 events, ~4.6/match) have no SkillCorner
        # equivalent. Capturing them would inflate the DFL pass count without
        # a comparable SC signal.
        if event_elem.find("KickOff") is not None:
            continue

        # --- Tackle / TacklingGame ---
        # EXCLUDED: DFL TacklingGame (tackle + take_on) has no SkillCorner
        # equivalent. SC on_ball_engagement is pressing, not tackling.
        # Including either would create source-specific features unusable
        # in cross-dataset models. Raw data remains available for
        # single-dataset analysis.

        # --- BallClaiming (Interception) ---
        # EXCLUDED: DFL-only action type with no SkillCorner equivalent.

        # --- Foul ---
        foul = event_elem.find("Foul")
        if foul is not None:
            fa = foul.attrib
            # Use the fouled player (victim) as player_id to align with SC,
            # which records the player who was fouled out of possession.
            fouled_id = fa.get("Fouled", "")
            pinfo = players.get(fouled_id, {})
            rows.append({
                **base,
                "action_type": "foul",
                "player_id": fouled_id,
                "player_name": pinfo.get("player_name", ""),
                "team_id": pinfo.get("team_id", fa.get("TeamFouled", "")),
                "team_name": pinfo.get("team_name", ""),
                "result": "fail",
            })
            continue

        # --- Clearance (OtherBallAction with DefensiveClearance=true) ---
        # OtherBallAction without DefensiveClearance (1540 events) has no
        # SkillCorner equivalent and is intentionally excluded.
        oba = event_elem.find("OtherBallAction")
        if oba is not None:
            oa = oba.attrib
            if oa.get("DefensiveClearance") == "true":
                player_id = oa.get("Player", "")
                pinfo = players.get(player_id, {})
                rows.append({
                    **base,
                    "action_type": "clearance",
                    "player_id": player_id,
                    "player_name": pinfo.get("player_name", ""),
                    "team_id": pinfo.get("team_id", oa.get("Team", "")),
                    "team_name": pinfo.get("team_name", ""),
                    "result": "success",
                })
            continue

        # --- GoalKick wrapper: goalkick ---
        # GoalKick/Play carries the taker's player_id.
        gk = event_elem.find("GoalKick")
        if gk is not None:
            gk_play = gk.find("Play")
            player_id = gk_play.get("Player", "") if gk_play is not None else ""
            pinfo = players.get(player_id, {})
            team_id = pinfo.get("team_id", gk.get("Team", ""))
            rows.append({
                **base,
                "action_type": "goalkick",
                "player_id": player_id,
                "player_name": pinfo.get("player_name", ""),
                "team_id": team_id,
                "team_name": pinfo.get("team_name", teams.get(gk.get("Team", ""), "")),
                "result": "success",
            })
            continue

        # --- ThrowIn wrapper: throw_in ---
        # ThrowIn/Play carries the thrower's player_id (available for 305/308
        # events; FaultExecution and FairPlay sub-types have no Play child).
        ti = event_elem.find("ThrowIn")
        if ti is not None:
            ti_play = ti.find("Play")
            player_id = ti_play.get("Player", "") if ti_play is not None else ""
            pinfo = players.get(player_id, {})
            team_id = pinfo.get("team_id", ti.get("Team", ""))
            rows.append({
                **base,
                "action_type": "throw_in",
                "player_id": player_id,
                "player_name": pinfo.get("player_name", ""),
                "team_id": team_id,
                "team_name": pinfo.get("team_name", teams.get(ti.get("Team", ""), "")),
                "result": "success",
            })
            continue

        # --- Offside ---
        # DFL records offside as a standalone <Offside> event that immediately
        # follows the pass that triggered it. SPADL and SkillCorner both model
        # offside as result="offside" on the triggering pass row, so we patch
        # the most-recently emitted pass-type row retroactively.
        #
        # Eligibility: only pass, freekick_short, freekick_cross, throw_in can
        # cause offside (FIFA LOTG 11). Corner kicks and goal kicks are exempt,
        # so they are excluded from the lookback. We search at most 4 rows back
        # to guard against rare cases where a non-pass event was emitted between
        # the offside pass and the <Offside> marker (e.g. a referee event).
        offside_elem = event_elem.find("Offside")
        if offside_elem is not None:
            _offside_eligible = {"pass", "cross", "freekick_short", "freekick_cross", "throw_in"}
            for i in range(len(rows) - 1, max(len(rows) - 5, -1), -1):
                if rows[i].get("action_type") in _offside_eligible and rows[i]["result"] == "fail":
                    rows[i]["result"] = "offside"
                    break
            continue

    # Down-harmonise throw-ins to SkillCorner's reception perspective.
    # DFL records the *thrower* at the touchline; SkillCorner only sees the
    # *receival* (the receiver at the reception point) and cannot recover the
    # thrower. To make throw_in cross-dataset comparable we re-key each DFL
    # throw-in to its receival: the next same-team on-ball event identifies the
    # receiver and its start location is the reception point. Throws that are
    # immediately lost/contested (next event is the opponent) are left keyed to
    # the thrower -- those are exactly the cases SkillCorner does not log as a
    # throw_in_reception either. The original thrower/origin remain in the raw
    # DFL XML for single-source analyses.
    for i, ti_row in enumerate(rows):
        if ti_row.get("action_type") != "throw_in":
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is None or not nxt.get("player_id") or nxt.get("team_id") != ti_row.get("team_id"):
            continue
        ti_row["player_id"] = nxt["player_id"]
        ti_row["player_name"] = nxt.get("player_name", ti_row.get("player_name", ""))
        if nxt.get("start_x") is not None and not pd.isna(nxt.get("start_x")):
            ti_row["start_x"] = nxt["start_x"]
            ti_row["start_y"] = nxt["start_y"]

    return rows


def load_all_dfl_events():
    """Load and normalize events from all DFL matches."""
    event_files = sorted(DFL_DIR.glob("DFL_03_02_events_raw_*.xml"))
    info_files = sorted(DFL_DIR.glob("DFL_02_01_matchinformation_*.xml"))

    # Match event files to info files by match ID suffix
    info_map = {}
    for f in info_files:
        mid = f.stem.split("_")[-1]
        info_map[mid] = f

    all_rows = []
    for ef in event_files:
        mid = ef.stem.split("_")[-1]
        mif = info_map.get(mid)
        if mif is None:
            warnings.warn(f"No match info for {mid}, skipping")
            continue

        rows = load_dfl_events(ef, mif)
        # Add competition from info file
        tree = ET.parse(mif)
        comp = tree.find(".//General").get("CompetitionName", "")
        for r in rows:
            r["competition"] = comp
        all_rows.extend(rows)
        print(f"  DFL {mid}: {len(rows)} actions")

    return all_rows


# ---------------------------------------------------------------------------
# SkillCorner helpers
# ---------------------------------------------------------------------------

def _sc_normalize_coords(x, y):
    """Pass through SkillCorner event coords — already CDF centre-origin metres."""
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    return float(x), float(y)


def _sc_frame_clock_to_seconds(clock_str: str | None) -> float:
    """Convert SkillCorner clock strings like HH:MM.S or HH:MM:SS.ss to seconds."""
    if not clock_str:
        return 0.0
    try:
        parts = str(clock_str).split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def _sc_period_offsets(match_json: dict) -> dict[int, float]:
    period_offsets = {1: 0.0}
    p1_duration_secs = 45.0 * 60.0
    for mp in match_json.get("match_periods", []):
        if mp.get("period") == 1:
            p1_duration_secs = float(mp.get("duration_minutes", 45.0)) * 60.0
    period_offsets[2] = p1_duration_secs + ASSUMED_HALFTIME_BREAK_SECS
    return period_offsets


def _sc_derive_utc(clock_str: str | None, period: int, kickoff_time_utc: str | None, period_offsets: dict[int, float]) -> str | None:
    if not kickoff_time_utc:
        return None
    try:
        kickoff_dt = datetime.fromisoformat(kickoff_time_utc.replace("Z", "+00:00"))
    except ValueError:
        return None
    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    offset = float(period_offsets.get(period, 0.0))
    dt = kickoff_dt + timedelta(seconds=offset + _sc_frame_clock_to_seconds(clock_str))
    return dt.astimezone(timezone.utc).isoformat()


def load_sc_events(match_dir):
    """Parse one SkillCorner match's dynamic_events CSV into SPADL-format rows."""
    match_id = Path(match_dir).name
    events_file = Path(match_dir) / f"{match_id}_dynamic_events.csv"
    match_file = Path(match_dir) / f"{match_id}_match.json"

    if not events_file.exists():
        return []

    df = pd.read_csv(events_file)

    competition = "A-League"
    kickoff_time_utc = None
    period_offsets = {1: 0.0}
    play_direction = {}
    if match_file.exists():
        with open(match_file, "r", encoding="utf-8") as f:
            mdata = json.load(f)
            competition = (
                (mdata.get("competition") or {}).get("name")
                or ((mdata.get("competition_edition") or {}).get("competition") or {}).get("name")
                or "A-League"
            )
            kickoff_time_utc = _normalize_utc_iso(mdata.get("date_time"))
            period_offsets = _sc_period_offsets(mdata)
            home_sides = mdata.get("home_team_side", [])
            for idx, side in enumerate(home_sides[:2], start=1):
                period_name = SC_PERIOD_MAP.get(idx)
                if period_name:
                    if side == "left_to_right":
                        play_direction[period_name] = "left_right"
                    elif side == "right_to_left":
                        play_direction[period_name] = "right_left"

    rows = []

    # --- player_possession events → pass, shot, clearance, foul, bad_touch ---
    pp = df[df["event_type"] == "player_possession"].copy()
    for _, row in pp.iterrows():
        period = SC_PERIOD_MAP.get(int(row.get("period", 0))) if pd.notna(row.get("period")) else None
        timestamp = _sc_derive_utc(row.get("time_start"), int(row.get("period", 1)), kickoff_time_utc, period_offsets)
        raw_start_x, start_y = _sc_normalize_coords(row.get("x_start"), row.get("y_start"))
        raw_end_x, end_y = _sc_normalize_coords(row.get("x_end"), row.get("y_end"))
        start_x = _flip_event_x(raw_start_x, period, play_direction)
        end_x = _flip_event_x(raw_end_x, period, play_direction)

        end_type = str(row.get("end_type", "")).lower()
        start_type = str(row.get("start_type", "")).lower()

        # Derive throw-in from start_type: SC marks possessions that begin
        # at the sideline after a throw-in as throw_in_reception. Emit a
        # separate throw_in action so counts align with DFL's per-event model.
        if start_type == "throw_in_reception":
            rows.append({
                "event_id": (_coerce_id(row.get("event_id")) or "") + "_ti",
                "match_id": str(match_id),
                "timestamp": timestamp,
                "period": period,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "action_type": "throw_in",
                "player_id": _coerce_id(row.get("player_id")) or "",
                "player_name": str(row.get("player_name", "")),
                "team_id": _coerce_id(row.get("team_id")) or "",
                "team_name": str(row.get("team_shortname", "")),
                "receiver_id": None,
                "receiver_time": None,
                "body_part": None,
                "related_event_ids": None,
                "result": "success",
                "source": "SkillCorner",
                "competition": competition,
            })

        receiver_id = None
        receiver_time = None
        if end_type == "pass":
            action_type = "pass"
            outcome = str(row.get("pass_outcome", "")).lower()
            if outcome == "successful":
                result = "success"
                receiver_id = _coerce_id(row.get("player_targeted_id"))
                receiver_time = _sc_derive_utc(row.get("time_end"), int(row.get("period", 1)), kickoff_time_utc, period_offsets)
            elif outcome == "offside":
                result = "offside"
            else:
                result = "fail"
        elif end_type == "shot":
            # game_interruption_before = free_kick_for identifies direct free
            # kick shots (start_type = free_kick_reception covers 8/11; the
            # remaining 3 have start_type = unknown but the interruption signal
            # is unambiguous). Penalty kicks are not captured as possession
            # events in SC (structural gap) so shot_penalty stays DFL-only.
            gi_before = str(row.get("game_interruption_before", "")).lower()
            action_type = "shot_freekick" if gi_before == "free_kick_for" else "shot"
            if str(row.get("lead_to_goal", "")).lower() == "true":
                result = "success"
            else:
                result = "fail"
        elif end_type == "clearance":
            action_type = "clearance"
            result = "success"
        elif end_type == "foul_suffered":
            action_type = "foul"
            result = "fail"
        elif end_type == "possession_loss":
            action_type = "bad_touch"
            result = "fail"
        else:
            action_type = "non_action"
            result = "fail"

        rows.append({
            "event_id": _coerce_id(row.get("event_id")) or "",
            "match_id": str(match_id),
            "timestamp": timestamp,
            "period": period,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "action_type": action_type,
            "player_id": _coerce_id(row.get("player_id")) or "",
            "player_name": str(row.get("player_name", "")),
            "team_id": _coerce_id(row.get("team_id")) or "",
            "team_name": str(row.get("team_shortname", "")),
            "receiver_id": receiver_id,
            "receiver_time": receiver_time,
            "body_part": None,
            "related_event_ids": None,
            "result": result,
            "source": "SkillCorner",
            "competition": competition,
        })

    # --- on_ball_engagement events (pressing) ---
    # EXCLUDED: SC on_ball_engagement (pressing, pressure, recovery_press,
    # counter_press) are proximity-based defensive events with no DFL
    # equivalent. Including them would create a source-specific feature.
    # Raw SC data remains available for single-dataset analysis.

    return rows


def load_all_sc_events():
    """Load and normalize events from all SkillCorner matches."""
    all_rows = []
    match_dirs = sorted(SC_DIR.iterdir())

    for md in match_dirs:
        if not md.is_dir():
            continue
        rows = load_sc_events(md)
        all_rows.extend(rows)
        print(f"  SC {md.name}: {len(rows)} actions")

    return all_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_normalization():
    """Run the full SPADL event normalization pipeline."""
    print("=" * 60)
    print("SPADL Event Normalization Pipeline")
    print("=" * 60)

    # Load DFL
    print("\n[1/3] Loading DFL events...")
    dfl_rows = load_all_dfl_events()
    print(f"  Total DFL actions: {len(dfl_rows)}")

    # Load SkillCorner
    print("\n[2/3] Loading SkillCorner events...")
    sc_rows = load_all_sc_events()
    print(f"  Total SkillCorner actions: {len(sc_rows)}")

    # Merge
    print("\n[3/3] Merging and saving...")
    all_rows = dfl_rows + sc_rows
    df = pd.DataFrame(all_rows)

    # Add cross-dataset comparability flag
    df["cross_dataset_comparable"] = df["action_type"].isin(CROSS_DATASET_COMPARABLE)

    # Enforce column order
    cols = [
        "event_id", "match_id", "timestamp", "period", "action_type", "player_id", "player_name",
        "team_id", "team_name", "receiver_id", "receiver_time", "body_part", "related_event_ids", "result",
        "start_x", "start_y", "end_x", "end_y",
        "source", "competition", "cross_dataset_comparable",
    ]
    df = df[cols]

    # Type cleanup
    for col in ("event_id", "player_id", "team_id", "match_id"):
        df[col] = df[col].astype(str)
    for col in ("start_x", "start_y", "end_x", "end_y"):
        df[col] = df[col].round(3)

    # Save one analytical parquet + one CDF JSONL file per match.
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for match_id, match_df in df.groupby("match_id"):
        match_dir = MATCHES_DIR / str(match_id)
        match_dir.mkdir(parents=True, exist_ok=True)
        parquet_out = match_dir / f"{match_id}_events_spadl.parquet"
        jsonl_out = match_dir / f"{match_id}_events.jsonl"
        match_df.to_parquet(parquet_out, index=False)
        _write_cdf_event_jsonl(match_df, jsonl_out)
        total += len(match_df)
        print(f"  {match_id}: {len(match_df)} events → {parquet_out.name}, {jsonl_out.name}")
    print(f"\n  Total actions saved: {total}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nBy source:")
    print(df["source"].value_counts().to_string())
    print(f"\nBy action_type:")
    print(df["action_type"].value_counts().to_string())
    print(f"\nBy result:")
    print(df["result"].value_counts().to_string())
    print(f"\nBy competition:")
    print(df["competition"].value_counts().to_string())

    # Comparability summary
    comparable = df[df["cross_dataset_comparable"]]
    print(f"\nCross-dataset comparable actions: {len(comparable)} "
          f"({len(comparable)/len(df)*100:.1f}%)")
    print(comparable.groupby(["source", "action_type"]).size()
          .unstack(fill_value=0).to_string())

    print("\n" + "-" * 40)
    print("Note: Player position filtering is done downstream")
    print("in the feature engineering / clustering step, not here.")
    print("-" * 40)

    return df


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_normalization()
