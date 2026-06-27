"""
Harmonise DFL and SkillCorner tracking datasets into a unified 10 Hz CDF-compliant DataFrame.

Pipeline:
    1. Load DFL tracking (25 Hz) from raw TRACAB XML, downsample to 10 Hz
    2. Load SkillCorner tracking (10 Hz native) from raw JSONL
    3. Concatenate; assign monotonic frame_id per match (preserve frame_id_source)
    4. Build source-aware filtered speed (DFL native copy; SC smoothed positions)
    5. Build source-aware acceleration (DFL native copy; SC finite difference)
    6. Quality control: null encoding, outlier capping, precision rounding
    7. Save analytical Parquet plus CDF delivery JSON/JSONL per match

Coordinate system (CDF-compliant):
    - Origin (0, 0): pitch centre
    - X: along sidelines [−L/2, +L/2] metres
    - Y: along goal lines [−W/2, +W/2] metres
    - Coordinates are orientation-normalized so the home team always attacks
      left-to-right, per CDF §5.2
    - Original play_direction is preserved in metadata for provenance

Usage:
    python -m src.harmonization.tracking
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.harmonization.cdf import apply_cdf_orientation, build_cdf_metadata, write_cdf_tracking_jsonl
from src.harmonization.dfl import (
    DFL_DIR,
    apply_dfl_player_stats,
    apply_dfl_sub_timings,
    load_dfl_match,
    parse_dfl_matchinfo,
    parse_dfl_play_direction,
    parse_dfl_player_stats,
    parse_dfl_substitutions,
)
from src.harmonization.kinematics import (
    _FLOAT_COLS,
    _assign_frame_ids_single,
    apply_is_visible_nulling,
    apply_savgol_filter,
    derive_acceleration,
    derive_speed_from_smoothed_positions,
    smooth_positions_per_segment,
)
from src.harmonization.skillcorner import (
    SC_DIR,
    SC_MATCHES_JSON,
    load_sc_match,
    load_sc_match_meta,
)
from src.harmonization.utils import CDF_VERSION, DATA_PATH, MAX_ACCEL_MS2, MAX_SPEED_KMH, _normalize_utc_iso

OUTPUT_DIR = DATA_PATH / "merged"


def _dfl_match_meta(info: dict, play_dir: dict, match_id: str) -> dict:
    roster = info["teams_roster"]
    return {
        "match_id": match_id,
        "source": "DFL",
        "competition": info["competition"],
        "competition_id": info.get("competition_id"),
        "competition_name": info.get("competition_name"),
        "season_name": info.get("season_name"),
        "season_id": info.get("season_id"),
        "match_day": info.get("match_day"),
        "kickoff_time_utc": _normalize_utc_iso(info["kickoff_time_utc"]),
        "home_score": info.get("home_score"),
        "away_score": info.get("away_score"),
        "first_half_score": info.get("first_half_score"),
        "second_half_score": info.get("second_half_score"),
        "pitch_length": info["pitch_length"],
        "pitch_width": info["pitch_width"],
        "stadium_id": info.get("stadium_id"),
        "stadium_name": info.get("stadium_name"),
        "stadium_capacity": info.get("stadium_capacity"),
        "referees": info.get("referees", []),
        "tracking_type": "in_stadium",
        "tracking_name_original": "TRACAB Gen5",
        "fps_original": 25,
        "fps_output": 10,
        "play_direction": play_dir,
        "ball_status_source": "native",
        "cdf_version": CDF_VERSION,
        "teams": {
            "home": roster.get(info["home_team_id"], {"id": info["home_team_id"], "name": "", "players": []}),
            "away": roster.get(info["away_team_id"], {"id": info["away_team_id"], "name": "", "players": []}),
        },
    }


def _sc_match_meta(meta: dict, match_id: str) -> dict:
    roster = meta["teams_roster"]
    return {
        "match_id": match_id,
        "source": "SkillCorner",
        "competition": "A-League",
        "competition_id": meta.get("competition_id"),
        "competition_name": meta.get("competition_name"),
        "season_name": meta.get("season_name"),
        "season_id": meta.get("season_id"),
        "match_day": meta.get("match_day"),
        "kickoff_time_utc": _normalize_utc_iso(meta["kickoff_time_utc"]),
        "home_score": meta.get("home_score"),
        "away_score": meta.get("away_score"),
        "first_half_score": None,
        "second_half_score": None,
        "pitch_length": meta["pitch_length"],
        "pitch_width": meta["pitch_width"],
        "stadium_id": meta.get("stadium_id"),
        "stadium_name": meta.get("stadium_name"),
        "stadium_capacity": meta.get("stadium_capacity"),
        "referees": meta.get("referees", []),
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


def process_and_save_match(
    df: pd.DataFrame,
    match_id: str,
    match_meta: dict,
    matches_dir: Path,
) -> dict:
    """Run all post-processing steps on a single match and write output files.

    Steps: frame ID assignment → speed filter → acceleration → QC → save.
    Writes analytical parquet plus CDF delivery JSON/JSONL files:
    {match_id}_tracking_10hz.parquet, {match_id}_tracking_10hz.jsonl,
    {match_id}_metadata.json.

    Args:
        df: Raw tracking DataFrame for this match only.
        match_id: Match identifier string.
        match_meta: Metadata dict to write as JSON.
        matches_dir: Root matches directory (data/merged/matches/).

    Returns:
        Dict with keys: rows, players, frames, size_mb, visible_pct (SC only).
    """
    for col in ("player_id", "team_id", "match_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["is_visible"] = df["is_visible"].fillna(True).astype(bool)
    df["is_home"] = df["is_home"].fillna(False).astype(bool)
    if "acceleration_ms2" not in df.columns:
        df["acceleration_ms2"] = np.nan
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _assign_frame_ids_single(df)

    raw_spike = df["speed_kmh"].abs() > MAX_SPEED_KMH

    source = df["source"].iloc[0]

    if source == "SkillCorner":
        # σ_position ≈ 1 m at uniform 0.1 s sampling — smooth positions
        # first (window 13), then differentiate. Position-first smoothing
        # attenuates the noise that single-frame derivatives inflate ~10×
        # into ~50 km/h of per-axis speed noise far more effectively than
        # smoothing speed after the fact.
        df = smooth_positions_per_segment(df, exclude_mask=raw_spike)
        df = derive_speed_from_smoothed_positions(df)
        df = df.drop(columns=["x_smooth", "y_smooth"])
    else:
        # DFL: speed_kmh and acceleration_ms2 are TRACAB native values
        # computed by the vendor from the full 25 Hz trajectory before
        # downsampling (cf. dfl_native_kinematics.md). Preserve them as
        # sampled native values; smoothing them again would double-smooth
        # peaks and sprint/acceleration onsets.
        if raw_spike.any():
            df.loc[raw_spike, "speed_kmh"] = np.nan
        df["speed_kmh_filtered"] = df["speed_kmh"].copy()

    if raw_spike.any():
        df.loc[raw_spike, "speed_kmh"] = np.nan

    if source == "SkillCorner":
        df = derive_acceleration(df, speed_col="speed_kmh_filtered")
    else:
        accel_spike = df["acceleration_ms2"].abs() > MAX_ACCEL_MS2
        if accel_spike.any():
            df.loc[accel_spike, "acceleration_ms2"] = np.nan
        df["acceleration_ms2_filtered"] = df["acceleration_ms2"].copy()

    df = apply_is_visible_nulling(df)

    mask = df["speed_kmh_filtered"].abs() > MAX_SPEED_KMH
    if mask.any():
        df.loc[mask, "speed_kmh_filtered"] = np.nan

    for accel_col in ("acceleration_ms2", "acceleration_ms2_filtered"):
        if accel_col in df.columns:
            mask = df[accel_col].abs() > MAX_ACCEL_MS2
            if mask.any():
                df.loc[mask, accel_col] = np.nan

    propagated = df["speed_kmh_filtered"].isna()
    for accel_col in ("acceleration_ms2", "acceleration_ms2_filtered"):
        if accel_col in df.columns:
            df.loc[propagated, accel_col] = np.nan

    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].round(3)

    df = apply_cdf_orientation(df, match_meta.get("play_direction", {}))

    match_dir = matches_dir / str(match_id)
    match_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["frame_id", "player_id"]).reset_index(drop=True)
    cdf_meta = build_cdf_metadata(match_meta, df)

    tracking_out = match_dir / f"{match_id}_tracking_10hz.parquet"
    df.to_parquet(tracking_out, index=False)

    tracking_jsonl_out = match_dir / f"{match_id}_tracking_10hz.jsonl"
    write_cdf_tracking_jsonl(df, cdf_meta, tracking_jsonl_out)

    meta_out = match_dir / f"{match_id}_metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(cdf_meta, f, indent=2, ensure_ascii=False)

    size_mb = tracking_out.stat().st_size / 1024 / 1024
    sc_rows = df[df["source"] == "SkillCorner"]
    visible_pct = sc_rows["is_visible"].mean() * 100 if len(sc_rows) > 0 else None

    return {
        "rows": len(df),
        "players": df["player_id"].nunique(),
        "frames": df["frame_id"].nunique(),
        "size_mb": size_mb,
        "visible_pct": visible_pct,
    }


def run_merge(resume: bool = True) -> None:
    """Process all matches one at a time, saving each before moving to the next.

    Each match is loaded, post-processed, written to disk, and freed from memory
    before the next match begins. If a match fails, already-saved matches are
    unaffected and the pipeline continues with remaining matches.

    Args:
        resume: If True (default), skip matches whose output file already exists.
                Set to False to reprocess everything from scratch.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches_dir = OUTPUT_DIR / "matches"
    matches_dir.mkdir(exist_ok=True)

    ok: list[dict] = []
    skipped: list[str] = []
    failed: list[dict] = []

    def _tracking_exists(mid: str) -> bool:
        match_dir = matches_dir / mid
        return (
            (match_dir / f"{mid}_tracking_10hz.parquet").exists()
            and (match_dir / f"{mid}_tracking_10hz.jsonl").exists()
            and (match_dir / f"{mid}_metadata.json").exists()
        )

    print("=" * 60)
    print("DFL matches")
    print("=" * 60)

    meta_files = sorted(DFL_DIR.glob("*matchinformation*.xml"))
    pos_files = sorted(DFL_DIR.glob("*positions_raw*.xml"))
    event_map = {
        ef.stem.split("_")[-1]: ef
        for ef in sorted(DFL_DIR.glob("*events_raw*.xml"))
    }

    for meta_path, raw_path in zip(meta_files, pos_files):
        match_id = meta_path.stem.split("_")[-1]

        if resume and _tracking_exists(match_id):
            print(f"  [{match_id}] already exists — skipping")
            skipped.append(match_id)
            continue

        try:
            event_path = event_map.get(match_id)
            info = parse_dfl_matchinfo(meta_path)
            play_dir = parse_dfl_play_direction(event_path, info["home_team_id"]) if event_path else {}
            sub_timings = parse_dfl_substitutions(event_path, info["kickoff_time_utc"])
            apply_dfl_sub_timings(info["teams_roster"], sub_timings)
            player_team = {
                pid: meta["team_id"] for pid, meta in info["player_meta"].items()
            }
            player_stats = parse_dfl_player_stats(event_path, player_team)
            apply_dfl_player_stats(info["teams_roster"], player_stats)
            match_meta = _dfl_match_meta(info, play_dir, match_id)

            df = load_dfl_match(meta_path, event_path, raw_path, match_id)
            stats = process_and_save_match(df, match_id, match_meta, matches_dir)
            del df
            gc.collect()

            print(
                f"  [{match_id}] OK  {stats['rows']:,} rows  "
                f"{stats['players']} players  {stats['size_mb']:.1f} MB"
            )
            ok.append({"match_id": match_id, "source": "DFL", **stats})

        except Exception as exc:
            print(f"  [{match_id}] FAILED: {exc}")
            failed.append({"match_id": match_id, "source": "DFL", "error": str(exc)})
            gc.collect()

    print()
    print("=" * 60)
    print("SkillCorner matches")
    print("=" * 60)

    if not SC_MATCHES_JSON.exists():
        warnings.warn(f"matches.json not found at {SC_MATCHES_JSON}")
        matches_index = []
    else:
        with open(SC_MATCHES_JSON, encoding="utf-8") as f:
            matches_index = json.load(f)

    match_dirs = sorted([d for d in SC_DIR.iterdir() if d.is_dir()])

    for match_dir in match_dirs:
        match_id = match_dir.name

        if resume and _tracking_exists(match_id):
            print(f"  [{match_id}] already exists — skipping")
            skipped.append(match_id)
            continue

        try:
            meta = load_sc_match_meta(match_dir, matches_index)
            match_meta = _sc_match_meta(meta, match_id)

            df = load_sc_match(match_dir, match_id, matches_index)
            if len(df) == 0:
                raise ValueError("empty DataFrame returned")

            stats = process_and_save_match(df, match_id, match_meta, matches_dir)
            del df
            gc.collect()

            vis = f"  {stats['visible_pct']:.1f}% visible" if stats["visible_pct"] is not None else ""
            print(
                f"  [{match_id}] OK  {stats['rows']:,} rows  "
                f"{stats['players']} players  {stats['size_mb']:.1f} MB{vis}"
            )
            ok.append({"match_id": match_id, "source": "SkillCorner", **stats})

        except Exception as exc:
            print(f"  [{match_id}] FAILED: {exc}")
            failed.append({"match_id": match_id, "source": "SkillCorner", "error": str(exc)})
            gc.collect()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_rows = sum(r["rows"] for r in ok)
    total_mb = sum(r["size_mb"] for r in ok)
    print(f"  Processed:  {len(ok)} matches  ({total_rows:,} rows  {total_mb:.1f} MB)")
    print(f"  Skipped:    {len(skipped)} already-complete matches")
    print(f"  Failed:     {len(failed)}")

    if ok:
        print()
        for src in ("DFL", "SkillCorner"):
            src_matches = [r for r in ok if r["source"] == src]
            if src_matches:
                src_rows = sum(r["rows"] for r in src_matches)
                src_players = sum(r["players"] for r in src_matches)
                print(f"  {src}: {len(src_matches)} matches  {src_rows:,} rows  {src_players} players (incl. duplicates across matches)")

    if failed:
        print()
        print("  Failed matches:")
        for f in failed:
            print(f"    [{f['match_id']}] {f['error']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Harmonise DFL and SkillCorner tracking data.")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess all matches even if output already exists."
    )
    args = parser.parse_args()
    run_merge(resume=not args.no_resume)
