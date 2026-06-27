import importlib
import json
import sys

import pandas as pd


def _load_merge_tracking(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    sys.modules.pop("src.harmonization.tracking", None)
    sys.modules.pop("src.config", None)
    module = importlib.import_module("src.harmonization.tracking")
    return importlib.reload(module)


def test_process_and_save_match_writes_cdf_jsonl_and_nested_metadata(monkeypatch, tmp_path):
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)

    df = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "frame_id_source": 10,
                "period": "first_half",
                "timestamp": "2025-01-01T12:00:00+00:00",
                "player_id": "h1",
                "player_name": "Home One",
                "team_id": "home",
                "team_name": "Home FC",
                "position_raw": "CB",
                "position_label": "CB",
                "position_group": "DF",
                "x": 0.0,
                "y": 0.0,
                "ball_x": 0.1,
                "ball_y": 0.2,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": None,
                "distance_m": None,
                "is_visible": True,
                "is_home": True,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            },
            {
                "match_id": "m1",
                "frame_id_source": 10,
                "period": "first_half",
                "timestamp": "2025-01-01T12:00:00+00:00",
                "player_id": "a1",
                "player_name": "Away One",
                "team_id": "away",
                "team_name": "Away FC",
                "position_raw": "CF",
                "position_label": "CF",
                "position_group": "FW",
                "x": 1.0,
                "y": 1.0,
                "ball_x": 0.1,
                "ball_y": 0.2,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": None,
                "distance_m": None,
                "is_visible": True,
                "is_home": False,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            },
            {
                "match_id": "m1",
                "frame_id_source": 12,
                "period": "first_half",
                "timestamp": "2025-01-01T12:00:00.100000+00:00",
                "player_id": "h1",
                "player_name": "Home One",
                "team_id": "home",
                "team_name": "Home FC",
                "position_raw": "CB",
                "position_label": "CB",
                "position_group": "DF",
                "x": 0.5,
                "y": 0.0,
                "ball_x": 0.3,
                "ball_y": 0.4,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": 18.0,
                "distance_m": 0.5,
                "is_visible": True,
                "is_home": True,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            },
            {
                "match_id": "m1",
                "frame_id_source": 12,
                "period": "first_half",
                "timestamp": "2025-01-01T12:00:00.100000+00:00",
                "player_id": "a1",
                "player_name": "Away One",
                "team_id": "away",
                "team_name": "Away FC",
                "position_raw": "CF",
                "position_label": "CF",
                "position_group": "FW",
                "x": 1.5,
                "y": 1.0,
                "ball_x": 0.3,
                "ball_y": 0.4,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": 18.0,
                "distance_m": 0.5,
                "is_visible": True,
                "is_home": False,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            },
        ]
    )

    match_meta = {
        "match_id": "m1",
        "source": "DFL",
        "competition": "Bundesliga",
        "competition_id": "bund",
        "competition_name": "Bundesliga",
        "season_name": "2024/25",
        "season_id": "2024",
        "match_day": 1,
        "kickoff_time_utc": "2025-01-01T12:00:00+00:00",
        "home_score": 1,
        "away_score": 0,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "stadium_name": "Test Arena",
        "stadium_capacity": 1000,
        "tracking_type": "in_stadium",
        "tracking_name_original": "TRACAB Gen5",
        "fps_original": 25,
        "fps_output": 10,
        "play_direction": {"first_half": "left_right"},
        "ball_status_source": "native",
        "cdf_version": "1.0.0",
        "teams": {
            "home": {"id": "home", "name": "Home FC", "players": []},
            "away": {"id": "away", "name": "Away FC", "players": []},
        },
    }

    stats = merge_tracking.process_and_save_match(
        df=df,
        match_id="m1",
        match_meta=match_meta,
        matches_dir=tmp_path / "merged" / "matches",
    )

    assert stats["frames"] == 2

    match_dir = tmp_path / "merged" / "matches" / "m1"
    parquet_path = match_dir / "m1_tracking_10hz.parquet"
    jsonl_path = match_dir / "m1_tracking_10hz.jsonl"
    metadata_path = match_dir / "m1_metadata.json"

    assert parquet_path.exists()
    assert jsonl_path.exists()
    assert metadata_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_frame = json.loads(lines[0])
    second_frame = json.loads(lines[1])

    assert first_frame["frame_id"] == 0
    assert first_frame["teams"]["home"]["id"] == "home"
    assert first_frame["teams"]["away"]["id"] == "away"
    assert first_frame["ball"]["status"] is True
    assert second_frame["teams"]["home"]["players"][0]["vel"] == 5.0
    assert second_frame["teams"]["home"]["players"][0]["dist"] == 0.5

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["meta"]["cdf"]["version"] == "1.0.0"
    assert metadata["meta"]["system"]["domain"] == "DFL"
    assert metadata["match"]["result"]["winning_team_id"] == "home"
    assert metadata["periods"][0]["frame_id_start"] == 0
    assert metadata["periods"][0]["frame_id_end"] == 1


def test_process_and_save_match_normalizes_tracking_x_and_writes_whistles(monkeypatch, tmp_path):
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)

    df = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "frame_id_source": 1,
                "period": "first_half",
                "timestamp": "2025-01-01T12:00:00+00:00",
                "player_id": "h1",
                "player_name": "Home One",
                "team_id": "home",
                "team_name": "Home FC",
                "position_raw": "CB",
                "position_label": "CB",
                "position_group": "DF",
                "x": 10.0,
                "y": 0.0,
                "ball_x": 20.0,
                "ball_y": 0.0,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": None,
                "distance_m": None,
                "is_visible": True,
                "is_home": True,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            }
        ]
    )

    match_meta = {
        "match_id": "m2",
        "source": "DFL",
        "competition": "Bundesliga",
        "competition_id": "bund",
        "competition_name": "Bundesliga",
        "season_name": "2024/25",
        "season_id": "2024",
        "match_day": 1,
        "kickoff_time_utc": "2025-01-01T12:00:00+00:00",
        "home_score": 0,
        "away_score": 0,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "stadium_name": "Test Arena",
        "stadium_capacity": 1000,
        "tracking_type": "in_stadium",
        "tracking_name_original": "TRACAB Gen5",
        "fps_original": 25,
        "fps_output": 10,
        "play_direction": {"first_half": "right_left"},
        "ball_status_source": "native",
        "cdf_version": "1.0.0",
        "teams": {
            "home": {"id": "home", "name": "Home FC", "players": []},
            "away": {"id": "away", "name": "Away FC", "players": []},
        },
    }

    merge_tracking.process_and_save_match(
        df=df,
        match_id="m2",
        match_meta=match_meta,
        matches_dir=tmp_path / "merged" / "matches",
    )

    match_dir = tmp_path / "merged" / "matches" / "m2"
    saved = pd.read_parquet(match_dir / "m2_tracking_10hz.parquet")
    assert saved.loc[0, "x"] == -10.0
    assert saved.loc[0, "ball_x"] == -20.0

    metadata = json.loads((match_dir / "m2_metadata.json").read_text(encoding="utf-8"))
    assert metadata["coordinates_normalized_to_cdf"] is True
    assert metadata["meta"]["representation"]["coordinates_normalized_to_cdf"] is True
    assert metadata["periods"][0]["left_team_id"] == "away"
    assert metadata["periods"][0]["right_team_id"] == "home"
    assert metadata["whistles"][0]["sub_type"] == "start"
    assert metadata["whistles"][1]["sub_type"] == "end"


def test_dfl_native_kinematics_are_copied_to_filtered_columns(monkeypatch, tmp_path):
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)

    speeds = [10.0, 12.0, 25.0, 14.0, 30.0, 16.0, 22.0, 18.0]
    accels = [0.0, 0.5, 4.0, -2.0, 3.0, 1.0, -1.0, 2.0]
    base = pd.Timestamp("2025-01-01T12:00:00+00:00")
    df = pd.DataFrame(
        [
            {
                "match_id": "m3",
                "frame_id_source": i,
                "period": "first_half",
                "timestamp": (base + pd.Timedelta(milliseconds=100 * i)).isoformat(),
                "player_id": "h1",
                "player_name": "Home One",
                "team_id": "home",
                "team_name": "Home FC",
                "position_raw": "CB",
                "position_label": "CB",
                "position_group": "DF",
                "x": float(i),
                "y": 0.0,
                "ball_x": 0.0,
                "ball_y": 0.0,
                "ball_z": 0.0,
                "ball_status": True,
                "ball_poss_team_id": "home",
                "speed_kmh": speeds[i],
                "acceleration_ms2": accels[i],
                "distance_m": 0.1 if i else None,
                "is_visible": True,
                "is_home": True,
                "source": "DFL",
                "tracking_type": "in_stadium",
                "kinematic_source": "native_tracab",
                "competition": "Bundesliga",
            }
            for i in range(len(speeds))
        ]
    )

    match_meta = {
        "match_id": "m3",
        "source": "DFL",
        "competition": "Bundesliga",
        "competition_id": "bund",
        "competition_name": "Bundesliga",
        "season_name": "2024/25",
        "season_id": "2024",
        "match_day": 1,
        "kickoff_time_utc": "2025-01-01T12:00:00+00:00",
        "home_score": 0,
        "away_score": 0,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "stadium_name": "Test Arena",
        "stadium_capacity": 1000,
        "tracking_type": "in_stadium",
        "tracking_name_original": "TRACAB Gen5",
        "fps_original": 25,
        "fps_output": 10,
        "play_direction": {"first_half": "left_right"},
        "ball_status_source": "native",
        "cdf_version": "1.0.0",
        "teams": {
            "home": {"id": "home", "name": "Home FC", "players": []},
            "away": {"id": "away", "name": "Away FC", "players": []},
        },
    }

    merge_tracking.process_and_save_match(
        df=df,
        match_id="m3",
        match_meta=match_meta,
        matches_dir=tmp_path / "merged" / "matches",
    )

    saved = pd.read_parquet(
        tmp_path / "merged" / "matches" / "m3" / "m3_tracking_10hz.parquet"
    )
    assert saved["speed_kmh"].tolist() == speeds
    assert saved["speed_kmh_filtered"].tolist() == speeds
    assert saved["acceleration_ms2"].tolist() == accels
    assert saved["acceleration_ms2_filtered"].tolist() == accels
