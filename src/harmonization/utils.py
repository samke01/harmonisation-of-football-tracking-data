"""Domain constants and low-level helpers shared across harmonisation modules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

DATA_PATH = Path(settings.data_path)

CDF_VERSION = "1.0.0"

MAX_SPEED_KMH = 40.0
MAX_ACCEL_MS2 = 12.0

# Per-source Savitzky-Golay window (frames). Only SkillCorner is configured:
# its uniform 0.1 s sampling makes the equal-spacing assumption exactly valid.
# DFL is absent intentionally — native TRACAB S/A are used directly.
SOURCE_POS_SMOOTH_WINDOW = {
    "SkillCorner": 13,  # 1300 ms at 10 Hz
}

# CDF Appendix C position_label and position_group mappings.
SUB_ENTRY = (None, "SUB")

DFL_POS_MAP: dict[str, tuple[str | None, str]] = {
    "TW":  ("GK",  "GK"),
    "IVL": ("LCB", "DF"), "IVR": ("RCB", "DF"), "IVZ": ("CB",  "DF"),
    "LV":  ("LB",  "DF"), "RV":  ("RB",  "DF"),
    "DML": ("LDM", "MF"), "DMR": ("RDM", "MF"), "DMZ": ("CDM", "MF"),
    "ZO":  ("CAM", "MF"),
    "LM":  ("LM",  "MF"), "RM":  ("RM",  "MF"),
    "DLM": ("LCM", "MF"), "DRM": ("RCM", "MF"),
    "OLM": ("LAM", "MF"), "ORM": ("RAM", "MF"),
    "LA":  ("LW",  "FW"), "RA":  ("RW",  "FW"),
    "HL":  ("LW",  "FW"), "HR":  ("RW",  "FW"),
    "STZ": ("CF",  "FW"), "STL": ("LCF", "FW"), "STR": ("RCF", "FW"),
    "SUB": SUB_ENTRY,
}

SC_POS_MAP: dict[str, tuple[str | None, str]] = {
    "GK":  ("GK",  "GK"),
    "LCB": ("LCB", "DF"), "RCB": ("RCB", "DF"), "CB":  ("CB",  "DF"),
    "LB":  ("LB",  "DF"), "RB":  ("RB",  "DF"),
    "LWB": ("LB",  "DF"), "RWB": ("RB",  "DF"),
    "DM":  ("CDM", "MF"), "LDM": ("LDM", "MF"), "RDM": ("RDM", "MF"),
    "AM":  ("CAM", "MF"),
    "LM":  ("LM",  "MF"), "RM":  ("RM",  "MF"),
    "LW":  ("LW",  "FW"), "RW":  ("RW",  "FW"),
    "LF":  ("LCF", "FW"), "RF":  ("RCF", "FW"),
    "CF":  ("CF",  "FW"),
    "SUB": SUB_ENTRY,
}


def _pos_entry(
    raw: str | None, mapping: dict[str, tuple[str | None, str]]
) -> tuple[str | None, str | None]:
    """Return ``(label, group)`` for a raw position code, both None on miss."""
    if raw is None:
        return None, None
    entry = mapping.get(raw)
    if entry is None:
        return None, None
    return entry


def _dt_seconds(t1: str, t2: str) -> float | None:
    """Compute elapsed seconds between two ISO-8601 timestamps."""
    try:
        dt1 = datetime.fromisoformat(t1)
        dt2 = datetime.fromisoformat(t2)
        return (dt2 - dt1).total_seconds()
    except (ValueError, TypeError):
        return None


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _secs_to_hms(secs: float | None) -> str | None:
    """Convert elapsed seconds to ``"HH:MM:SS"`` (SkillCorner roster format)."""
    if secs is None or secs < 0:
        return None
    total = int(secs)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalize_utc_iso(raw: str | None) -> str | None:
    """Normalise an ISO-8601 timestamp to use ``+00:00`` offset (never ``Z``)."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError):
        return raw


def _json_value(value):
    """Convert pandas / numpy scalars into JSON-safe Python values."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    return None if pd.isna(value) else value


def _kmh_to_ms(value) -> float | None:
    value = _json_value(value)
    if value is None:
        return None
    return round(float(value) / 3.6, 3)


def _winning_team_id(match_meta: dict) -> str | None:
    """Return the winning team ID when the final score is available."""
    home_score = match_meta.get("home_score")
    away_score = match_meta.get("away_score")
    teams = match_meta.get("teams", {})
    home_id = teams.get("home", {}).get("id")
    away_id = teams.get("away", {}).get("id")
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return home_id
    if away_score > home_score:
        return away_id
    return None


def _left_team_id_for_period(match_meta: dict, period: str) -> str | None:
    """Derive which team is on the left side from the home's attacking side."""
    teams = match_meta.get("teams", {})
    home_id = teams.get("home", {}).get("id")
    away_id = teams.get("away", {}).get("id")
    play_direction = (match_meta.get("play_direction") or {}).get(period)
    if play_direction == "left_right":
        return home_id
    if play_direction == "right_left":
        return away_id
    return None


def _right_team_id_for_period(match_meta: dict, period: str) -> str | None:
    """Derive which team is on the right side from the home's attacking side."""
    teams = match_meta.get("teams", {})
    home_id = teams.get("home", {}).get("id")
    away_id = teams.get("away", {}).get("id")
    play_direction = (match_meta.get("play_direction") or {}).get(period)
    if play_direction == "left_right":
        return away_id
    if play_direction == "right_left":
        return home_id
    return None
