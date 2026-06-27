"""Regression tests for tracking kinematics derivation.

The original bug (commit fixing pandas 2.x ISO8601 parsing): pd.to_datetime
without an explicit format= silently coerces ISO-8601 strings with
fractional seconds (e.g. "2024-11-30T04:00:00.100000+00:00") to NaT under
errors='coerce'. At 10 Hz, 9 of every 10 SkillCorner timestamps carry
fractional seconds, so the entire derive_speed and derive_acceleration
output collapsed to NaN for SC. DFL's odd-second timestamps survived,
masking the bug; only the merged tracking parquet revealed it via 100 %
NaN speed_kmh_filtered for SC.

These tests pin the fix so the regression cannot return silently.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd


def _load_merge_tracking(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    sys.modules.pop("src.harmonization.tracking", None)
    sys.modules.pop("src.config", None)
    module = importlib.import_module("src.harmonization.tracking")
    return importlib.reload(module)


def _build_uniform_player_df(
    n_frames: int,
    dt_seconds: float,
    fractional: bool,
    source: str = "SkillCorner",
) -> pd.DataFrame:
    """Build a single-player single-period dataframe walking along x at 1 m/s."""
    base = pd.Timestamp("2025-01-01T00:00:00+00:00")
    timestamps = []
    for i in range(n_frames):
        ts = base + pd.Timedelta(seconds=i * dt_seconds)
        if fractional:
            timestamps.append(ts.isoformat())
        else:
            # Force a no-fractional rendering by stripping microseconds.
            ts_no_frac = ts.replace(microsecond=0)
            timestamps.append(ts_no_frac.isoformat())
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append(
            {
                "match_id": "m1",
                "player_id": "p1",
                "team_id": "t1",
                "period": "first_half",
                "timestamp": ts,
                "x": float(i) * dt_seconds,  # 1 m/s along x
                "y": 0.0,
                "x_smooth": float(i) * dt_seconds,
                "y_smooth": 0.0,
                "speed_kmh_filtered": np.nan,
                "is_visible": True,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def test_derive_speed_handles_fractional_iso8601_timestamps(monkeypatch, tmp_path):
    """The 10 Hz fractional-second case must not produce all-NaN speed."""
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)
    df = _build_uniform_player_df(n_frames=20, dt_seconds=0.1, fractional=True)
    out = merge_tracking.derive_speed_from_smoothed_positions(df.copy())
    speed = out["speed_kmh_filtered"]
    # First row has no prior frame → NaN by construction; remaining 19 must be valid.
    assert speed.iloc[1:].notna().all(), (
        "Fractional-second timestamps still produce NaN speed — the ISO8601 "
        "format= argument was likely lost again. Got NaN count: "
        f"{speed.iloc[1:].isna().sum()} / 19"
    )
    # Walking at 1 m/s = 3.6 km/h; values should cluster there (within float epsilon).
    np.testing.assert_allclose(speed.iloc[1:].to_numpy(), 3.6, atol=1e-6)


def test_derive_acceleration_handles_fractional_iso8601_timestamps(monkeypatch, tmp_path):
    """Acceleration derivation must also survive fractional timestamps."""
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)
    df = _build_uniform_player_df(n_frames=20, dt_seconds=0.1, fractional=True)
    df["speed_kmh_filtered"] = 3.6  # constant speed
    out = merge_tracking.derive_acceleration(df.copy(), speed_col="speed_kmh_filtered")
    accel = out["acceleration_ms2"]
    # First two rows are NaN by the (i-1, i, i+1) finite-difference design;
    # remaining 18 must be valid (and ~0 for constant speed).
    assert accel.iloc[2:].notna().all(), (
        "Fractional-second timestamps still produce NaN acceleration. "
        f"Got NaN count: {accel.iloc[2:].isna().sum()} / 18"
    )
    np.testing.assert_allclose(accel.iloc[2:].to_numpy(), 0.0, atol=1e-9)


def test_derive_speed_handles_integer_second_timestamps(monkeypatch, tmp_path):
    """Integer-second ISO-8601 timestamps (no fractional component, like
    most DFL kickoff times) must also produce valid speeds. Together with
    the fractional case above, this confirms the parser handles both
    ISO-8601 forms used in the merged tracking parquets.

    Note: a previous version of this test compared the fractional and
    integer-second pipelines at dt=1.0, which was tautological because
    Timestamp.isoformat() omits the fractional part when the microsecond
    is zero — both fixtures produced byte-identical strings. This rewrite
    tests the integer-second case independently.
    """
    merge_tracking = _load_merge_tracking(monkeypatch, tmp_path)
    df = _build_uniform_player_df(n_frames=20, dt_seconds=1.0, fractional=False)
    # Sanity-check the fixture: at dt=1.0s every timestamp has microsecond=0,
    # so isoformat() strings must contain no '.' separator.
    assert all("." not in t for t in df["timestamp"].tolist()), (
        "fixture is supposed to emit integer-second strings but produced "
        f"fractional ones: {df['timestamp'].iloc[0]!r}"
    )
    out = merge_tracking.derive_speed_from_smoothed_positions(df.copy())
    speed = out["speed_kmh_filtered"]
    # n_frames=20 with dt=1.0 covers 19 valid speed rows (first row has no
    # prior frame). Walking 1 m/s = 3.6 km/h.
    assert speed.iloc[1:].notna().all(), (
        "Integer-second timestamps produced NaN speed. "
        f"NaN count: {speed.iloc[1:].isna().sum()} / 19"
    )
    np.testing.assert_allclose(speed.iloc[1:].to_numpy(), 3.6, atol=1e-6)
