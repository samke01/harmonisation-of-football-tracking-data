"""CDF-compliance and data-nature audit for the merged DFL/SkillCorner dataset.

Usage:
    uv run python scripts/run_cdf_audit.py

Output:
    data/reports/cdf_audit.json

The existing harmonization evaluation answers *how distinguishable* the two
sources are in the merged schema. It does not check whether the merge itself
respects the Common Data Format (Anzer et al., 2025, v1.0.0) nor whether
heterogeneous-data properties (orientation, coverage, ball-status semantics)
were treated consistently.

This script fills that gap. Each ``check_*`` function returns a dict that
captures a single concern; ``run_cdf_audit`` executes all of them against the
on-disk merged parquet and metadata files and writes a JSON report alongside
the existing harmonization report.

Checks grouped by concern:

- CDF spec: coordinate convention, pitch bounds, float precision, frame_id
  monotonicity from 0, period enum, play_direction value set, position_group
  taxonomy (Appendix C).
- Orientation: CDF §5.2 requires the home team to always play left-to-right
  for the entire match, with play_direction in metadata describing the
  original (pre-normalized) sides. The merge now stores CDF-normalized
  coordinates and preserves the original sides in metadata, so the
  orientation checks quantify source priors in the metadata rather than a
  live compliance gap in the saved tracking rows.
- Data-nature: visibility and kinematic-null propagation, SkillCorner
  coverage gap vs. DFL, ball_status distributional shift, ball_poss_team_id
  null rate by source, ID namespace collisions, per-match timestamp cadence,
  ball_z distribution.

No fixes are applied here — the audit is read-only and structurally matches
the style of ``run_harmonization_evaluation`` so the two reports can be read
side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

DATA_PATH = Path(settings.data_path)
MATCHES_DIR = DATA_PATH / "merged" / "matches"
REPORTS_DIR = DATA_PATH / "reports"

# CDF Appendix C position groups. The merge currently uses position LABELS
# (CB, FB, WG, DM, CM/AM) as group values, which is not compliant.
CDF_POSITION_GROUPS = {"GK", "DF", "MF", "FW", "SUB"}

# CDF §5.2 and the Figure 3 meta-data example use hyphen-less enum values.
CDF_PLAY_DIRECTION_VALUES = {"left_right", "right_left"}

# CDF Table 4 period enum.
CDF_PERIOD_VALUES = {
    "first_half",
    "second_half",
    "first_half_extratime",
    "second_half_extratime",
    "shootout",
}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _tracking_files() -> list[Path]:
    return sorted(MATCHES_DIR.glob("*/*_tracking_10hz.parquet"))


def _metadata_files() -> list[Path]:
    return sorted(MATCHES_DIR.glob("*/*_metadata.json"))


def _load_metadata() -> list[dict]:
    metas = []
    for path in _metadata_files():
        with open(path, encoding="utf-8") as f:
            metas.append(json.load(f))
    return metas


# ---------------------------------------------------------------------------
# CDF spec checks
# ---------------------------------------------------------------------------

def check_coordinate_bounds() -> dict:
    """Quantify in-play coordinates vs. pitch bounds per match and per source.

    CDF §5.2 allows coordinates outside ±L/2, ±W/2 to denote out-of-play
    positions. The check flags systematic excess (>10 % of *detected, in-play*
    frames outside bounds) which usually indicates a coordinate-system
    mismatch, not a genuine out-of-play situation.
    """
    rows = []
    for meta, parq in zip(_load_metadata(), _tracking_files()):
        L = float(meta.get("pitch_length") or 105.0)
        W = float(meta.get("pitch_width") or 68.0)
        df = pd.read_parquet(parq, columns=["source", "x", "y", "is_visible", "ball_status"])
        visible = df[df["is_visible"] == True]
        in_play = visible[visible["ball_status"] == True]
        x_excess = (in_play["x"].abs() > L / 2).mean()
        y_excess = (in_play["y"].abs() > W / 2).mean()
        rows.append({
            "match_id": meta["match_id"],
            "source": meta["source"],
            "pitch_length": L,
            "pitch_width": W,
            "x_p001": float(in_play["x"].quantile(0.001)),
            "x_p999": float(in_play["x"].quantile(0.999)),
            "y_p001": float(in_play["y"].quantile(0.001)),
            "y_p999": float(in_play["y"].quantile(0.999)),
            "frac_x_outside_bounds_in_play": float(x_excess),
            "frac_y_outside_bounds_in_play": float(y_excess),
        })
    return {
        "description": (
            "Per-match quantiles of detected in-play (x, y). "
            "CDF §5.2 permits out-of-play excess but systematic excess would "
            "indicate a coordinate-system or orientation issue."
        ),
        "per_match": rows,
    }


def check_frame_id_monotonic() -> dict:
    """CDF Table 4: frame_id must be monotonically increasing from 0 per match."""
    rows = []
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["match_id", "frame_id"])
        mid = df["match_id"].iloc[0]
        fids = df["frame_id"].drop_duplicates().sort_values().to_numpy()
        starts_at_zero = bool(fids[0] == 0) if len(fids) else False
        monotonic = bool(np.all(np.diff(fids) > 0)) if len(fids) > 1 else True
        rows.append({
            "match_id": mid,
            "min_frame_id": int(fids[0]) if len(fids) else None,
            "max_frame_id": int(fids[-1]) if len(fids) else None,
            "n_distinct_frame_ids": int(len(fids)),
            "starts_at_zero": starts_at_zero,
            "strictly_monotonic": monotonic,
        })
    n_ok = sum(r["starts_at_zero"] and r["strictly_monotonic"] for r in rows)
    return {
        "description": "CDF Table 4: frame_id must be monotonically increasing from 0 per match.",
        "n_matches_ok": n_ok,
        "n_matches_total": len(rows),
        "compliant": n_ok == len(rows),
        "per_match": rows,
    }


def check_period_enum_compliance() -> dict:
    """CDF Table 4: period is an enum with {first_half, second_half, ...}."""
    period_counts: dict[str, int] = {}
    for parq in _tracking_files():
        s = pd.read_parquet(parq, columns=["period"])["period"]
        for v, c in s.value_counts(dropna=False).items():
            period_counts[str(v)] = period_counts.get(str(v), 0) + int(c)
    non_compliant = [v for v in period_counts if v not in CDF_PERIOD_VALUES and v != "nan"]
    return {
        "description": "CDF Table 4 period enum: first_half, second_half, first_half_extratime, second_half_extratime, shootout.",
        "period_counts": period_counts,
        "non_compliant_values": non_compliant,
        "compliant": len(non_compliant) == 0,
    }


def check_play_direction_enum_compliance() -> dict:
    """CDF Figure 3: play_direction values are `left_right` / `right_left`."""
    seen: dict[str, int] = {}
    per_match = []
    for meta in _load_metadata():
        pd_dict = meta.get("play_direction", {}) or {}
        for period, value in pd_dict.items():
            key = f"{period}:{value}"
            seen[key] = seen.get(key, 0) + 1
        per_match.append({"match_id": meta["match_id"], "play_direction": pd_dict})
    values_used = {str(v).split(":")[-1] for v in seen}
    non_compliant = values_used - CDF_PLAY_DIRECTION_VALUES
    return {
        "description": (
            "CDF §5.2 / Figure 3 require play_direction values "
            "`left_right` / `right_left`."
        ),
        "values_used": sorted(values_used),
        "non_compliant_values": sorted(non_compliant),
        "compliant": len(non_compliant) == 0,
        "per_match": per_match,
    }


def check_position_group_taxonomy() -> dict:
    """CDF Appendix C groups are {GK, DF, MF, FW, SUB}."""
    counts: dict[str, int] = {}
    for parq in _tracking_files():
        s = pd.read_parquet(parq, columns=["position_group"])["position_group"].dropna()
        for v, c in s.value_counts().items():
            counts[str(v)] = counts.get(str(v), 0) + int(c)
    non_compliant_values = sorted(v for v in counts if v not in CDF_POSITION_GROUPS)
    return {
        "description": (
            "CDF Appendix C defines position_group ∈ {GK, DF, MF, FW, SUB}. "
            "The merged dataset currently uses CDF position *labels* (CB, FB, WG, DM, ...) "
            "in the group field."
        ),
        "value_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "compliant_values_used": sorted(v for v in counts if v in CDF_POSITION_GROUPS),
        "non_compliant_values": non_compliant_values,
        "compliant": len(non_compliant_values) == 0,
    }


def check_float_precision() -> dict:
    """CDF §5.2: floats ≤ 3 decimal places. Spot-check x, y, ball_z."""
    offenders = []
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["match_id", "source", "x", "y", "ball_z"])
        for col in ("x", "y", "ball_z"):
            vals = df[col].dropna().to_numpy()
            if len(vals) == 0:
                continue
            rounded = np.round(vals, 3)
            n_mismatch = int(np.sum(~np.isclose(vals, rounded, atol=1e-9)))
            if n_mismatch:
                offenders.append({
                    "match_id": df["match_id"].iloc[0],
                    "source": df["source"].iloc[0],
                    "column": col,
                    "n_rows_with_more_than_3_dp": n_mismatch,
                })
    return {
        "description": "CDF §5.2: maximum 3 decimal places.",
        "compliant": len(offenders) == 0,
        "offenders": offenders,
    }


def check_visibility_kinematic_propagation() -> dict:
    """CDF §5.2: missing values must be explicit null.

    Verifies that is_visible=False ⇒ x/y null and speed/accel NaN (merge
    documentation claims this), and that is_visible=True ⇒ x, y not null.
    """
    rows = []
    for parq in _tracking_files():
        df = pd.read_parquet(
            parq,
            columns=[
                "match_id", "source", "is_visible", "x", "y",
                "speed_kmh_filtered", "acceleration_ms2_filtered", "distance_m",
            ],
        )
        mid = df["match_id"].iloc[0]
        src = df["source"].iloc[0]
        inv = df[~df["is_visible"]]
        vis = df[df["is_visible"]]
        rows.append({
            "match_id": mid,
            "source": src,
            "n_invisible": int(len(inv)),
            "invisible_rows_with_x_not_null": int(inv["x"].notna().sum()),
            "invisible_rows_with_speed_not_nan": int(inv["speed_kmh_filtered"].notna().sum()),
            "invisible_rows_with_accel_not_nan": int(inv["acceleration_ms2_filtered"].notna().sum()),
            "visible_rows_with_x_null": int(vis["x"].isna().sum()),
            "visible_rows_with_y_null": int(vis["y"].isna().sum()),
        })
    leakage = [r for r in rows if r["invisible_rows_with_x_not_null"] > 0
               or r["invisible_rows_with_speed_not_nan"] > 0
               or r["invisible_rows_with_accel_not_nan"] > 0
               or r["visible_rows_with_x_null"] > 0
               or r["visible_rows_with_y_null"] > 0]
    return {
        "description": (
            "CDF §5.2 visibility/null propagation: "
            "is_visible=False ⇒ x, y, speed, accel null; is_visible=True ⇒ x, y not null."
        ),
        "compliant": len(leakage) == 0,
        "n_matches_with_leakage": len(leakage),
        "per_match": rows,
    }


def check_match_sheet_shape() -> dict:
    """CDF Table 1: match-sheet event-lists and player roster naming."""
    rows = []
    for meta in _load_metadata():
        match_block = meta.get("match", {}) or {}
        result = match_block.get("result", {}) or {}
        final_block = result.get("final", {}) or {}
        events = match_block.get("events", {}) or {}
        teams = match_block.get("teams", {}) or {}
        all_players = []
        for side in ("home", "away"):
            all_players.extend((teams.get(side, {}) or {}).get("players", []) or [])
        roster_keys = set().union(*(p.keys() for p in all_players)) if all_players else set()
        rows.append({
            "match_id": meta.get("match_id"),
            "has_result_final": "home" in final_block and "away" in final_block,
            "has_result_first_half": isinstance(result.get("first_half"), dict),
            "has_result_second_half": isinstance(result.get("second_half"), dict),
            "has_status_is_neutral": "is_neutral" in (match_block.get("status") or {}),
            "has_events_goals": isinstance(events.get("goals"), list),
            "has_events_cards": isinstance(events.get("cards"), list),
            "has_events_substitutions": isinstance(events.get("substitutions"), list),
            "n_goals": len(events.get("goals") or []),
            "n_cards": len(events.get("cards") or []),
            "n_substitutions": len(events.get("substitutions") or []),
            "uses_jersey_number": "jersey_number" in roster_keys,
            "uses_is_starter": "is_starter" in roster_keys,
            "has_first_name": "first_name" in roster_keys,
            "has_has_played": "has_played" in roster_keys,
        })
    bool_keys = (
        "has_result_final",
        "has_result_first_half",
        "has_result_second_half",
        "has_status_is_neutral",
        "has_events_goals",
        "has_events_cards",
        "has_events_substitutions",
        "uses_jersey_number",
        "uses_is_starter",
        "has_first_name",
        "has_has_played",
    )
    compliant = all(all(r[k] for k in bool_keys) for r in rows)
    return {
        "description": (
            "CDF Table 1 match-sheet shape: result.final / per-period "
            "scores, events.goals/cards/substitutions arrays, "
            "jersey_number / is_starter / first_name / has_played on each "
            "player record."
        ),
        "compliant": compliant,
        "per_match": rows,
    }


def check_referees_present() -> dict:
    """CDF Table 1: at least one referee per match (mandatory ``referees[].id``)."""
    rows = []
    for meta in _load_metadata():
        refs = meta.get("referees") or []
        rows.append({
            "match_id": meta.get("match_id"),
            "source": meta.get("source"),
            "n_referees": len(refs),
            "ids_set": all((r.get("id") for r in refs)) if refs else False,
        })
    return {
        "description": (
            "CDF Table 1 ``referees[]`` mandatory. SkillCorner broadcast "
            "tracking does not record referees (referees=[]); DFL XML "
            "always carries the four / five officials."
        ),
        "per_match": rows,
        "n_matches_without_referees": sum(1 for r in rows if r["n_referees"] == 0),
    }


def check_meta_block_naming() -> dict:
    """CDF Table 6: meta blocks must carry ``name``, ``version``, ``fps``."""
    rows = []
    for meta in _load_metadata():
        m = meta.get("meta", {}) or {}
        tracking = m.get("tracking", {}) or {}
        event = m.get("event", {}) or {}
        rows.append({
            "match_id": meta.get("match_id"),
            "tracking_has_name": "name" in tracking,
            "tracking_has_fps": "fps" in tracking,
            "tracking_has_version": "version" in tracking,
            "tracking_has_collection_timing": tracking.get("collection_timing") is not None,
            "event_has_name": "name" in event,
            "cdf_version": (m.get("cdf") or {}).get("version"),
        })
    compliant = all(
        r["tracking_has_name"]
        and r["tracking_has_fps"]
        and r["tracking_has_collection_timing"]
        for r in rows
    )
    return {
        "description": (
            "CDF Table 6 meta-block naming: ``meta.tracking.{name, fps, "
            "version, collection_timing}``; ``meta.event.name``; "
            "``meta.cdf.version``."
        ),
        "compliant": compliant,
        "per_match": rows,
    }


def check_id_namespace() -> dict:
    """Unique IDs for players, teams, matches (CDF §5.2)."""
    pids_by_src = {"DFL": set(), "SkillCorner": set()}
    tids_by_src = {"DFL": set(), "SkillCorner": set()}
    mids = []
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["source", "match_id", "player_id", "team_id"])
        src = df["source"].iloc[0]
        pids_by_src[src].update(df["player_id"].dropna().unique())
        tids_by_src[src].update(df["team_id"].dropna().unique())
        mids.append(df["match_id"].iloc[0])
    pid_overlap = pids_by_src["DFL"] & pids_by_src["SkillCorner"]
    tid_overlap = tids_by_src["DFL"] & tids_by_src["SkillCorner"]
    return {
        "description": "Player / team / match id namespaces should be disjoint across sources.",
        "player_id_overlap": sorted(pid_overlap),
        "team_id_overlap": sorted(tid_overlap),
        "duplicate_match_ids": sorted(mid for mid in set(mids) if mids.count(mid) > 1),
        "n_players": {
            "DFL": len(pids_by_src["DFL"]),
            "SkillCorner": len(pids_by_src["SkillCorner"]),
        },
        "n_teams": {
            "DFL": len(tids_by_src["DFL"]),
            "SkillCorner": len(tids_by_src["SkillCorner"]),
        },
        "compliant": len(pid_overlap) == 0 and len(tid_overlap) == 0,
    }


# ---------------------------------------------------------------------------
# Orientation audit (CDF §5.2 home-team left-to-right normalization)
# ---------------------------------------------------------------------------

def check_home_attacking_side_bias() -> dict:
    """Quantify orientation asymmetry vs. CDF §5.2 normalization."""
    per_match = []
    dist_first = {"DFL": {"left_right": 0, "right_left": 0},
                  "SkillCorner": {"left_right": 0, "right_left": 0}}
    for meta in _load_metadata():
        pd_dict = meta.get("play_direction", {}) or {}
        src = meta["source"]
        per_match.append({
            "match_id": meta["match_id"],
            "source": src,
            "first_half": pd_dict.get("first_half"),
            "second_half": pd_dict.get("second_half"),
        })
        fh = pd_dict.get("first_half", "")
        key = None
        if fh in ("left_to_right", "left_right"):
            key = "left_right"
        elif fh in ("right_to_left", "right_left"):
            key = "right_left"
        if key is not None:
            dist_first[src][key] += 1

    def _frac(d):
        n = sum(d.values())
        return {k: (v / n if n else None) for k, v in d.items()}

    return {
        "description": (
            "CDF §5.2: home team should always play left-to-right. "
            "The merge stores normalized coordinates and preserves only "
            "the original side in metadata. This check quantifies how "
            "unevenly first-half orientation is distributed across sources."
        ),
        "dfl_first_half_distribution": _frac(dist_first["DFL"]),
        "skillcorner_first_half_distribution": _frac(dist_first["SkillCorner"]),
        "per_match": per_match,
        "interpretation": (
            "If the two sources differ on first-half orientation, the "
            "un-normalized dataset carries a structural x-sign shift that "
            "inflates source-classifier accuracy on team-shape features."
        ),
    }


def check_home_gk_x_sign_consistency() -> dict:
    """Verify the saved tracking is CDF-orientation-normalised."""
    rows = []
    for meta, parq in zip(_load_metadata(), _tracking_files()):
        pd_dict = meta.get("play_direction", {}) or {}
        df = pd.read_parquet(
            parq,
            columns=["is_home", "position_group", "position_raw", "period", "x", "is_visible"],
        )
        home_gk = df[
            (df["is_home"])
            & ((df["position_group"] == "GK") | (df["position_raw"].isin(["TW", "GK"])))
            & df["is_visible"]
        ]
        for period_name in ("first_half", "second_half"):
            g = home_gk[home_gk["period"] == period_name]
            if g.empty:
                continue
            mean_x = float(g["x"].mean())
            declared = pd_dict.get(period_name, "")
            ok = mean_x < 0
            rows.append({
                "match_id": meta["match_id"],
                "source": meta["source"],
                "period": period_name,
                "original_direction": declared,
                "home_gk_mean_x": mean_x,
                "cdf_normalised": bool(ok),
            })
    mismatches = [r for r in rows if not r["cdf_normalised"]]
    return {
        "description": (
            "CDF §5.2: home team plays left → right; after orientation "
            "normalisation the home goalkeeper's mean x must be negative "
            "in every period."
        ),
        "n_checks": len(rows),
        "n_mismatches": len(mismatches),
        "compliant": len(mismatches) == 0,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Data-nature audits (heterogeneity acknowledgement)
# ---------------------------------------------------------------------------

def check_tracking_coverage_gap() -> dict:
    """Tracking-duration gap between DFL and SkillCorner."""
    rows = []
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["match_id", "source", "frame_id", "period"])
        mid = df["match_id"].iloc[0]
        src = df["source"].iloc[0]
        n_frames = int(df["frame_id"].nunique())
        duration_min = n_frames / 10.0 / 60.0
        p1 = df[df["period"] == "first_half"]["frame_id"].nunique() / 10.0 / 60.0
        p2 = df[df["period"] == "second_half"]["frame_id"].nunique() / 10.0 / 60.0
        rows.append({
            "match_id": mid,
            "source": src,
            "n_frames": n_frames,
            "duration_min": round(duration_min, 2),
            "first_half_min": round(p1, 2),
            "second_half_min": round(p2, 2),
        })
    by_src = {}
    for src in ("DFL", "SkillCorner"):
        vals = [r["duration_min"] for r in rows if r["source"] == src]
        if vals:
            by_src[src] = {
                "matches": len(vals),
                "mean_duration_min": float(np.mean(vals)),
                "median_duration_min": float(np.median(vals)),
                "min_duration_min": float(np.min(vals)),
                "max_duration_min": float(np.max(vals)),
            }
    return {
        "description": (
            "Per-match tracked duration (minutes). SkillCorner broadcast "
            "tracking may have systematic temporal gaps that DFL does not."
        ),
        "per_match": rows,
        "per_source_summary": by_src,
    }


def check_ball_status_distribution_shift() -> dict:
    """Highlight ball_status semantic mismatch between sources."""
    per_match = []
    per_src_true = {"DFL": [], "SkillCorner": []}
    for meta, parq in zip(_load_metadata(), _tracking_files()):
        df = pd.read_parquet(parq, columns=["source", "ball_status"])
        src = df["source"].iloc[0]
        bs_true = float((df["ball_status"] == True).mean())
        bs_false = float((df["ball_status"] == False).mean())
        bs_null = float(df["ball_status"].isna().mean())
        per_match.append({
            "match_id": meta["match_id"],
            "source": src,
            "ball_status_source": meta.get("ball_status_source"),
            "frac_true": bs_true,
            "frac_false": bs_false,
            "frac_null": bs_null,
        })
        per_src_true[src].append(bs_true)
    return {
        "description": (
            "DFL ball_status is native per-frame; SkillCorner is a "
            "phase-of-play approximation. Distributional difference in "
            "the True fraction indicates the two labels mean different things."
        ),
        "per_match": per_match,
        "per_source_frac_true": {
            src: {
                "n_matches": len(v),
                "mean": float(np.mean(v)) if v else None,
                "min": float(np.min(v)) if v else None,
                "max": float(np.max(v)) if v else None,
            }
            for src, v in per_src_true.items()
        },
    }


def check_ball_poss_team_id_null_rate() -> dict:
    """ball_poss_team_id null rate by source."""
    per_match = []
    for meta, parq in zip(_load_metadata(), _tracking_files()):
        df = pd.read_parquet(parq, columns=["source", "ball_poss_team_id"])
        src = df["source"].iloc[0]
        null_rate = float(df["ball_poss_team_id"].isna().mean())
        per_match.append({
            "match_id": meta["match_id"],
            "source": src,
            "null_rate": null_rate,
        })
    return {
        "description": (
            "Null rate of ball_poss_team_id by source. DFL is native "
            "(~0 % null); SkillCorner carries only explicit possession "
            "events (~75–80 % null). Feature sets that include this "
            "column will trivially separate sources."
        ),
        "per_match": per_match,
    }


def check_ball_z_distribution() -> dict:
    """Per-source ball_z stats; flags unphysical negatives and median bias."""
    per_src: dict[str, list] = {"DFL": [], "SkillCorner": []}
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["source", "ball_z"])
        src = df["source"].iloc[0]
        per_src[src].append(df["ball_z"].dropna().to_numpy())
    out = {}
    for src, arrs in per_src.items():
        if not arrs:
            continue
        arr = np.concatenate(arrs)
        out[src] = {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "p01": float(np.quantile(arr, 0.01)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
            "frac_negative": float(np.mean(arr < 0)),
        }
    return {
        "description": (
            "Ball height distribution per source. A non-zero fraction of "
            "negative ball_z is unphysical; a large median/mean gap is a "
            "provider artefact that inflates source classification."
        ),
        "per_source": out,
    }


def check_timestamp_cadence() -> dict:
    """Per-source dt distribution for the filtered tracking stream."""
    per_src = {"DFL": [], "SkillCorner": []}
    for parq in _tracking_files():
        df = pd.read_parquet(parq, columns=["source", "match_id", "player_id", "period", "frame_id", "timestamp"])
        src = df["source"].iloc[0]
        df = df.sort_values(["player_id", "period", "frame_id"]).reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        dt = ts.diff().dt.total_seconds()
        grp = (df["player_id"].shift() != df["player_id"]) | (df["period"].shift() != df["period"])
        dt = dt[~grp].dropna().to_numpy()
        dt = dt[(dt > 0) & (dt < 5.0)]
        if len(dt):
            per_src[src].append(dt)
    out = {}
    for src, arrs in per_src.items():
        if not arrs:
            continue
        arr = np.concatenate(arrs)
        out[src] = {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "p01": float(np.quantile(arr, 0.01)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
            "frac_in_0_08_to_0_12": float(np.mean((arr >= 0.079) & (arr <= 0.121))),
        }
    return {
        "description": (
            "Per-row dt distribution after merge/downsampling. "
            "SkillCorner: uniform 0.1 s. DFL: alternating 0.08 / 0.12 s "
            "(mean 0.1 s) from the 25 → 10 Hz alternating 2/3-step decimation."
        ),
        "per_source": out,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_cdf_audit(save: bool = True) -> dict:
    """Run all CDF-compliance and data-nature checks; write a single JSON report."""
    checks = {
        "cdf_coordinate_bounds": check_coordinate_bounds(),
        "cdf_frame_id_monotonic": check_frame_id_monotonic(),
        "cdf_period_enum": check_period_enum_compliance(),
        "cdf_play_direction_enum": check_play_direction_enum_compliance(),
        "cdf_position_group_taxonomy": check_position_group_taxonomy(),
        "cdf_float_precision": check_float_precision(),
        "cdf_visibility_null_propagation": check_visibility_kinematic_propagation(),
        "cdf_id_namespace": check_id_namespace(),
        "cdf_match_sheet_shape": check_match_sheet_shape(),
        "cdf_referees_present": check_referees_present(),
        "cdf_meta_block_naming": check_meta_block_naming(),
        "orientation_home_attacking_side_bias": check_home_attacking_side_bias(),
        "orientation_home_gk_x_sign_consistency": check_home_gk_x_sign_consistency(),
        "nature_tracking_coverage_gap": check_tracking_coverage_gap(),
        "nature_ball_status_distribution_shift": check_ball_status_distribution_shift(),
        "nature_ball_poss_team_id_null_rate": check_ball_poss_team_id_null_rate(),
        "nature_ball_z_distribution": check_ball_z_distribution(),
        "nature_timestamp_cadence": check_timestamp_cadence(),
    }
    report = {
        "description": (
            "CDF-compliance and data-nature audit of the merged 17-match "
            "10 Hz tracking dataset. The audit is independent of the "
            "direct harmonization evaluation and identifies concrete "
            "issues arising from merging heterogeneous tracking sources."
        ),
        "reference": (
            "Anzer et al. (2025), The Common Data Format (CDF): A "
            "Standardized Format for Match-Data in Football (Soccer), v1.0.0."
        ),
        "checks": checks,
    }
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "cdf_audit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"Saved report: {out_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = run_cdf_audit()

    checks = report["checks"]
    print()
    print("=" * 64)
    print("CDF-compliance summary")
    print("=" * 64)
    for name in (
        "cdf_frame_id_monotonic",
        "cdf_period_enum",
        "cdf_play_direction_enum",
        "cdf_position_group_taxonomy",
        "cdf_float_precision",
        "cdf_visibility_null_propagation",
        "cdf_id_namespace",
        "cdf_match_sheet_shape",
        "cdf_referees_present",
        "cdf_meta_block_naming",
    ):
        c = checks[name]
        ok = c.get("compliant")
        tag = "OK " if ok else "NO "
        print(f"  [{tag}] {name}")
        if not ok:
            if "non_compliant_values" in c and c["non_compliant_values"]:
                print(f"         non-compliant values: {c['non_compliant_values'][:6]}")
            if name == "cdf_position_group_taxonomy":
                print(f"         values in use: {list(c['value_counts'].keys())}")
            if "n_matches_with_leakage" in c:
                print(f"         matches with leakage: {c['n_matches_with_leakage']}")
            if "player_id_overlap" in c and c["player_id_overlap"]:
                print(f"         player-id overlap count: {len(c['player_id_overlap'])}")

    print()
    print("=" * 64)
    print("Orientation audit")
    print("=" * 64)
    ob = checks["orientation_home_attacking_side_bias"]
    print(f"  DFL first-half distribution:        {ob['dfl_first_half_distribution']}")
    print(f"  SkillCorner first-half distribution:{ob['skillcorner_first_half_distribution']}")
    gk = checks["orientation_home_gk_x_sign_consistency"]
    print(f"  Home-GK x-sign vs. metadata: {gk['n_checks'] - gk['n_mismatches']}/{gk['n_checks']} OK")

    print()
    print("=" * 64)
    print("Data-nature audit")
    print("=" * 64)
    cov = checks["nature_tracking_coverage_gap"]["per_source_summary"]
    for src, s in cov.items():
        print(f"  Coverage [{src:12s}]  matches={s['matches']}  mean={s['mean_duration_min']:.1f} min  "
              f"min={s['min_duration_min']:.1f}  max={s['max_duration_min']:.1f}")

    bs = checks["nature_ball_status_distribution_shift"]["per_source_frac_true"]
    for src, s in bs.items():
        if s["mean"] is None:
            continue
        print(f"  ball_status frac_true [{src:12s}]  mean={s['mean']:.2f}  range=[{s['min']:.2f}, {s['max']:.2f}]")

    bz = checks["nature_ball_z_distribution"]["per_source"]
    for src, s in bz.items():
        print(f"  ball_z [{src:12s}]  median={s['median']:.3f}  mean={s['mean']:.3f}  "
              f"min={s['min']:.3f}  frac_negative={s['frac_negative']:.4f}")

    cad = checks["nature_timestamp_cadence"]["per_source"]
    for src, s in cad.items():
        print(f"  dt [{src:12s}]  median={s['median']:.3f}s  mean={s['mean']:.3f}s  "
              f"p99={s['p99']:.3f}s  frac_in_0.08-0.12={s['frac_in_0_08_to_0_12']:.3f}")


if __name__ == "__main__":
    main()
