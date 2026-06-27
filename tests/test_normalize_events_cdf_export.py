import importlib
import json
import sys

import pandas as pd


def _load_normalize_events(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    sys.modules.pop("src.harmonization.events", None)
    sys.modules.pop("src.config", None)
    module = importlib.import_module("src.harmonization.events")
    return importlib.reload(module)


def test_write_cdf_event_jsonl_outputs_mandatory_event_fields(monkeypatch, tmp_path):
    normalize_events = _load_normalize_events(monkeypatch, tmp_path)

    df = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "match_id": "m1",
                "timestamp": "2025-01-01T12:00:00+00:00",
                "period": "first_half",
                "action_type": "pass",
                "player_id": "p1",
                "player_name": "Player One",
                "team_id": "home",
                "team_name": "Home FC",
                "receiver_id": "p2",
                "receiver_time": "2025-01-01T12:00:01+00:00",
                "body_part": None,
                "related_event_ids": None,
                "result": "success",
                "start_x": -5.0,
                "start_y": 1.0,
                "end_x": 2.0,
                "end_y": 3.0,
                "source": "DFL",
                "competition": "Bundesliga",
                "cross_dataset_comparable": True,
            }
        ]
    )

    out_path = tmp_path / "events.jsonl"
    normalize_events._write_cdf_event_jsonl(df, out_path)

    line = out_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    event = payload["event"]

    assert payload["match"]["id"] == "m1"
    assert payload["meta"]["is_synced"] is True
    assert event["id"] == "e1"
    assert event["time"] == "2025-01-01T12:00:00+00:00"
    assert event["period"] == "first_half"
    assert event["type"] == "pass"
    assert event["sub_type"] is None
    assert event["is_successful"] is True
    assert event["outcome_type"] == "successful"
    assert event["receiver_id"] == "p2"
    assert event["receiver_time"] == "2025-01-01T12:00:01+00:00"
    assert event["x"] == -5.0
    assert event["x_end"] == 2.0
