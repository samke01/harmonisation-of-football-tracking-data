"""Validate kinematics in the merged tracking dataset.

Since commit 3a790c9 the DFL pipeline reads the TRACAB native ``S`` and
``A`` attributes off each player ``<Frame>`` and writes them to
``speed_kmh`` / ``acceleration_ms2`` (kinematic_source = "native_tracab").
SkillCorner has no native frame-level kinematics, so its ``speed_kmh`` is
finite-differenced from position-smoothed coordinates and
``acceleration_ms2`` is finite-differenced from exported filtered speed
(kinematic_source = "finite_diff_native").

This script therefore performs:

1. Native S/A precondition: confirms the TRACAB XML actually exposes
   non-empty ``S`` / ``A`` per player frame — required for the
   "native_tracab" path to produce real values.
2. Internal consistency: per-frame distance ≈ |Δposition|. Catches
   arithmetic drift in the loader.
3. Source-aware speed/distance consistency: SC raw ``speed_kmh`` is
   distance/dt by construction; DFL raw ``speed_kmh`` is native vendor
   speed and is expected to deviate from distance/dt by ~0.2 km/h MAE
   (cf. dfl_native_kinematics.md).
4. Source-aware acceleration check: SC ``acceleration_ms2`` is derived
   from filtered speed and must equal Δ(filtered_speed)/dt. DFL
   ``acceleration_ms2`` is the native vendor signal and is independent
   of the speed pipeline — large deviation from Δ(filtered_speed)/dt
   is expected, not a bug.
5. Distribution-level summary per source (mean / median / 99th / max)
   for speed and acceleration so the merged dataset's behaviour can be
   eyeballed against published references (e.g. Bassek et al. 2025;
   Buchheit et al. 2014).

Usage:
    uv run python scripts/validate_kinematics.py [--match MATCH_ID]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DATA_PATH = Path(settings.data_path)
DFL_DIR = DATA_PATH / "DFL"
MERGED_DIR = DATA_PATH / "merged" / "matches"


def check_native_kinematics_present(dfl_xml: Path, max_frames: int = 50_000) -> dict:
    """Scan a TRACAB XML and report whether player S/A attributes are populated."""
    n = 0
    nonzero_s = 0
    nonzero_a = 0
    samples_s: list[str] = []
    samples_a: list[str] = []
    current_team = None

    # NOTE: elem.clear() must run only on "end" events. Calling it on "start"
    # discards element attributes before the matching "end" can read them,
    # which previously made this function silently report False on every XML.
    for event, elem in ET.iterparse(str(dfl_xml), events=("start", "end")):
        if event == "start":
            if elem.tag == "FrameSet":
                current_team = elem.get("TeamId")
            continue
        if elem.tag == "Frame":
            if current_team not in (None, "BALL", "referee"):
                n += 1
                s = elem.get("S") or ""
                a = elem.get("A") or ""
                if s and float(s) != 0.0:
                    nonzero_s += 1
                    if len(samples_s) < 3:
                        samples_s.append(s)
                if a and float(a) != 0.0:
                    nonzero_a += 1
                    if len(samples_a) < 3:
                        samples_a.append(a)
                if n >= max_frames:
                    break
            elem.clear()

    return {
        "frames_checked": n,
        "nonzero_speed": nonzero_s,
        "nonzero_accel": nonzero_a,
        "speed_samples": samples_s,
        "accel_samples": samples_a,
        "native_speed_present": nonzero_s > 0,
        "native_accel_present": nonzero_a > 0,
    }


def check_distance_speed_consistency(df: pd.DataFrame) -> dict:
    """Verify ``distance_m`` matches sqrt(Δx² + Δy²) within float precision."""
    df = df.sort_values(["match_id", "player_id", "period", "frame_id"]).copy()
    df["dx"] = df.groupby(["match_id", "player_id", "period"])["x"].diff()
    df["dy"] = df.groupby(["match_id", "player_id", "period"])["y"].diff()
    df["recomputed_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    mask = df["distance_m"].notna() & df["recomputed_dist"].notna()
    diff = (df.loc[mask, "distance_m"] - df.loc[mask, "recomputed_dist"]).abs()
    return {
        "n_compared": int(mask.sum()),
        "mean_abs_diff_m": float(diff.mean()),
        "max_abs_diff_m": float(diff.max()),
        "frac_above_1mm": float((diff > 1e-3).mean()),
    }


def check_speed_distance_consistency(df: pd.DataFrame, dt: float = 0.1) -> dict:
    """Verify raw ``speed_kmh`` ≈ distance / dt · 3.6.

    Uses ``dt = 0.1s`` since the merged dataset is at 10 Hz. For DFL the
    raw ``speed_kmh`` is the TRACAB native ``S`` attribute (computed by the
    vendor from the full 25 Hz signal before downsampling), so it is
    *expected* to deviate from distance/dt by ~0.2 km/h MAE — that gap is
    the smoothing benefit, not a bug. SkillCorner ``speed_kmh`` is
    distance/dt by construction.
    """
    mask = df["speed_kmh"].notna() & df["distance_m"].notna()
    derived = df.loc[mask, "distance_m"] / dt * 3.6
    diff = (df.loc[mask, "speed_kmh"] - derived).abs()
    return {
        "n_compared": int(mask.sum()),
        "mean_abs_diff_kmh": float(diff.mean()),
        "p95_abs_diff_kmh": float(diff.quantile(0.95)),
        "p99_abs_diff_kmh": float(diff.quantile(0.99)),
    }


def check_accel_speed_consistency(df: pd.DataFrame, dt: float = 0.1) -> dict:
    """Compare ``acceleration_ms2`` against Δ(filtered_speed_ms)/dt.

    Equality holds only for SkillCorner (``acceleration_ms2`` is derived
    from ``speed_kmh_filtered`` by ``derive_acceleration``). For DFL the
    raw ``acceleration_ms2`` is the TRACAB native ``A`` and is independent
    of the speed pipeline, so MAE >> 0 here is expected — call it as a
    descriptive metric of how far native A diverges from the exported
    speed derivative, not as a pass/fail consistency check.
    """
    df = df.sort_values(["match_id", "player_id", "period", "frame_id"]).copy()
    df["v_ms"] = df["speed_kmh_filtered"] / 3.6
    df["dv_ms"] = df.groupby(["match_id", "player_id", "period"])["v_ms"].diff()
    df["recomputed_a"] = df["dv_ms"] / dt

    mask = df["acceleration_ms2"].notna() & df["recomputed_a"].notna()
    diff = (df.loc[mask, "acceleration_ms2"] - df.loc[mask, "recomputed_a"]).abs()
    return {
        "n_compared": int(mask.sum()),
        "mean_abs_diff_ms2": float(diff.mean()),
        "p99_abs_diff_ms2": float(diff.quantile(0.99)),
        "max_abs_diff_ms2": float(diff.max()),
    }


def kinematic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source summary of derived speed / acceleration distributions."""
    rows = []
    for src, grp in df.groupby("source"):
        spd = grp["speed_kmh_filtered"].dropna()
        acc = grp["acceleration_ms2_filtered"].dropna()
        rows.append({
            "source": src,
            "n_rows": len(grp),
            "n_visible": int(grp["is_visible"].sum()),
            "speed_mean_kmh": spd.mean(),
            "speed_median_kmh": spd.median(),
            "speed_p99_kmh": spd.quantile(0.99),
            "speed_max_kmh": spd.max(),
            "speed_negative_frac": float((spd < 0).mean()),
            "accel_mean_ms2": acc.mean(),
            "accel_p99_abs_ms2": acc.abs().quantile(0.99),
            "accel_max_abs_ms2": acc.abs().max(),
        })
    return pd.DataFrame(rows)


def speed_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Player-time spent in each speed band (Bradley et al. 2009 thresholds)."""
    bands = [
        ("Standing 0–6", 0, 6),
        ("Jogging 6–11", 6, 11),
        ("Running 11–19.8", 11, 19.8),
        ("HSR 19.8–25.2", 19.8, 25.2),
        ("Sprint >25.2", 25.2, float("inf")),
    ]
    out = []
    for src, grp in df.groupby("source"):
        spd = grp["speed_kmh_filtered"].dropna()
        n = len(spd)
        row = {"source": src, "n_visible_kinematic": n}
        for name, lo, hi in bands:
            row[name] = float(((spd >= lo) & (spd < hi)).sum() / n) if n else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match",
        default="DFL-MAT-J03WOH",
        help="DFL match ID to use for native-attribute check",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("1) Native TRACAB S/A attribute presence (precondition for native_tracab)")
    print("=" * 64)
    xml = next(DFL_DIR.glob(f"*positions_raw*{args.match}*.xml"), None)
    if xml is None:
        print(f"  XML not found for {args.match}", file=sys.stderr)
        sys.exit(1)
    native = check_native_kinematics_present(xml)
    print(f"  {xml.name}")
    print(f"    frames_checked       = {native['frames_checked']:,}")
    print(f"    native_speed_present = {native['native_speed_present']}")
    print(f"    native_accel_present = {native['native_accel_present']}")
    if native["speed_samples"]:
        print(f"    speed samples        = {native['speed_samples']}")
    if native["accel_samples"]:
        print(f"    accel samples        = {native['accel_samples']}")
    if not (native["native_speed_present"] and native["native_accel_present"]):
        print(
            "  --> WARNING: S or A attributes are missing in this match XML.\n"
            "      The DFL pipeline now relies on native S/A — populated\n"
            "      DFL speed_kmh / acceleration_ms2 columns require both."
        )

    print()
    print("=" * 64)
    print("2) Internal consistency on all merged matches")
    print("=" * 64)

    parquets = sorted(MERGED_DIR.glob("*/*_tracking_10hz.parquet"))
    if not parquets:
        print(f"  No merged matches under {MERGED_DIR}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for path in parquets:
        df = pd.read_parquet(path)
        all_rows.append(df)
    merged = pd.concat(all_rows, ignore_index=True)
    print(f"  Loaded {len(merged):,} rows across {len(parquets)} matches")

    print()
    print("  Distance ↔ Δposition:")
    res = check_distance_speed_consistency(merged)
    for k, v in res.items():
        print(f"    {k}: {v}")

    print()
    print("  Raw speed_kmh ↔ distance/dt:")
    print("    DFL: native vendor S, ~0.2 km/h MAE expected vs distance/dt.")
    print("    SC : equality by construction.")
    for src, grp in merged.groupby("source"):
        res = check_speed_distance_consistency(grp)
        print(f"    [{src}] {res}")

    print()
    print("  Acceleration ↔ Δ(filtered_speed)/dt:")
    print("    SC : equality by construction (acceleration_ms2 = Δfiltered/dt).")
    print("    DFL: descriptive — native vendor A is independent of the speed")
    print("         pipeline, so MAE here is expected to be large.")
    for src, grp in merged.groupby("source"):
        res = check_accel_speed_consistency(grp)
        print(f"    [{src}] {res}")

    print()
    print("=" * 64)
    print("3) Distribution summary per source")
    print("=" * 64)
    summary = kinematic_summary(merged)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("=" * 64)
    print("4) Speed-band occupancy (Bradley thresholds)")
    print("=" * 64)
    bands = speed_band_distribution(merged)
    print(bands.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
