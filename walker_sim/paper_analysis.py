"""Reproduces Table IV from the paper.

For each (robot, scaling law) pair this script collects all the saved runs
under `<law>/<robot>/saved_sim_data/`, extracts the steady-state body-attitude
amplitudes and walking velocity at each scale, looks up the minimum required
torque, and fits power laws of the form `metric ~ L^alpha` across scales.

The fit results are meant to match Table IV of:

    Oke et al., "Allometric Scaling Laws for Bipedal Robots",
    arXiv:2603.22560
-----------------------------
* Amplitudes: mean(peaks) - mean(valleys),
  reported in Table IV in degrees. The first 5 s of every trial are discarded
  so we only measure the steady-state limit cycle.
* Velocity: total ground-plane displacement during the stable period divided
  by elapsed time.
* Minimum torque: "manually tuned" per scale to find the
  smallest value that produces consistent walking. The Zippy and
  scaled-Mugatu models ship the tuned (scale, torque) pairs in the
  `get_hip_torque(scale)` function. Within the tuned range, the
  function linearly interpolates between known points; outside the range it
  falls back to the fitted power law. The Mugatu controller is sinusoidal
  position control so it has no tuned torque ceiling (so for Mugatu we
  fall back to the peak hip torque measured during the stable region of the
  saved CSV).
* Power-law fit: least-squares on log10(L) vs log10(metric); reports alpha
  (the exponent) and R^2.

Usage
-----
    # Process everything saved across both laws:
    python -m walker_sim.paper_analysis

    # Just one (robot, law) pair:
    python -m walker_sim.paper_analysis --robot zippy --law L2

Results are printed to the console and written under `analysis_out/`.
"""

from __future__ import annotations
import argparse
import glob
import importlib
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_STABILIZATION_S = 5.0   # discard first 5 s of every trial

# Reference leg length at scale=1, in metres (Table III).
REF_LEG_LENGTH_M = {
    "zippy":         0.0247,
    "scaled_mugatu": 0.0243,
    "mugatu":        0.153,
}

# Robots whose torque ceiling is manually tuned in get_hip_torque().
TUNED_TORQUE_ROBOTS = {"zippy", "scaled_mugatu"}

# Maps each (robot, law) pair to its expected Table-IV row, for printing only.
TABLE_IV = {
    ("zippy",         "L2"): dict(v_exp=0.47, tau_exp=2.92, theta_R=9.28,  theta_P=25.02, theta_Y=7.14),
    ("scaled_mugatu", "L2"): dict(v_exp=0.48, tau_exp=2.93, theta_R=16.27, theta_P=25.02, theta_Y=7.45),
    ("mugatu",        "L2"): dict(v_exp=0.50, tau_exp=3.49, theta_R=19.91, theta_P=10.83, theta_Y=10.06),
    ("zippy",         "L3"): dict(v_exp=0.47, tau_exp=3.90, theta_R=10.09, theta_P=27.69, theta_Y=8.43),
    ("scaled_mugatu", "L3"): dict(v_exp=0.52, tau_exp=3.96, theta_R=16.60, theta_P=29.22, theta_Y=7.82),
    ("mugatu",        "L3"): dict(v_exp=0.51, tau_exp=4.07, theta_R=19.91, theta_P=10.82, theta_Y=9.82),
}

# ---------------------------------------------------------------------------
# Model-side torque lookup
# ---------------------------------------------------------------------------
def load_torque_function(robot: str, law: str):
    """Return the robot/law's `get_hip_torque(scale) -> Nm`, or None for Mugatu.

    The model's `get_hip_torque` already does the right thing: linear
    interpolation between manually tuned (scale, torque) pairs, with a power-
    law fallback only outside the tuned range. We just need to import it.
    """
    if robot == "mugatu":
        # Mugatu uses sinusoidal position control; no tuned torque ceiling.
        return None
    variant = {"L2": "ml2_scaling", "L3": "ml3_scaling"}[law]
    variant_dir = os.path.join(REPO_DIR, variant)
    if variant_dir not in sys.path:
        sys.path.insert(0, variant_dir)
    # Forget any other variant's cached import so we always get the right one.
    for m in list(sys.modules):
        if m == robot or m.startswith(robot + "."):
            del sys.modules[m]
    model = importlib.import_module(f"{robot}.model_definition")
    return getattr(model, "get_hip_torque", None)


def torque_lookup_diagnostics(get_hip_torque, scale: float):
    """Inspect what get_hip_torque is doing for a given scale.

    Returns (torque, source) where source is 'tuned' / 'interpolated' /
    'extrapolated' so the report can flag any out-of-range queries.
    """
    if get_hip_torque is None:
        return np.nan, "n/a"
    # The function reads its SCALE_DATA / TORQUE_DATA arrays from inside its
    # closure; we can inspect them through its __globals__ or just check the
    # bounds via duplicate hard-coded knowledge. The simplest robust check is
    # to call it on the boundary and see if the result matches the fall-back
    # polynomial.
    import inspect
    src = inspect.getsource(get_hip_torque)
    # Pull out the SCALE_DATA literal numbers; this is a stable layout.
    nums = re.findall(r'(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', src.split("TORQUE_DATA")[0])
    scales_in_table = sorted(set(float(n) for n in nums if 0 < float(n) < 1000))
    if not scales_in_table:
        return float(get_hip_torque(scale)), "unknown"
    lo, hi = scales_in_table[0], scales_in_table[-1]
    if scale < lo or scale > hi:
        source = "extrapolated"
    elif any(abs(scale - s) < 1e-5 for s in scales_in_table):
        source = "tuned"
    else:
        source = "interpolated"
    return float(get_hip_torque(scale)), source


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------
@dataclass
class TrialMetrics:
    csv_path: str
    robot: str
    law: str
    scale: float
    leg_length_m: float
    mass_kg: float
    duration_s: float
    velocity_mps: float
    min_torque_Nm: float
    torque_source: str           # 'tuned' / 'interpolated' / 'extrapolated' / 'measured'
    roll_amp_deg: float
    pitch_amp_deg: float
    yaw_amp_deg: float
    walked: bool                 # whether the trial actually walked
    walked_reason: str           # why a failed trial was excluded


def _peak_to_valley_amplitude(signal):
    """Half the average peak-to-valley swing over the stable signal.

    Intra-stride wobble can produce small spurious peaks; we filter them out
    by requiring each accepted peak to be prominent compared with the signal
    range.
    """

    s = np.asarray(signal, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 4:
        return np.nan
    # Only count peaks whose prominence is at least 10% of the signal range -
    # enough to discard within-stride wobble but well below stride-scale swings.
    prom = 0.1 * (s.max() - s.min())
    peaks, _ = find_peaks(s, prominence=prom)
    valleys, _ = find_peaks(-s, prominence=prom)
    if peaks.size == 0 or valleys.size == 0:
        # Fall back to unfiltered detection (small-amplitude signal).
        peaks, _ = find_peaks(s)
        valleys, _ = find_peaks(-s)
    if peaks.size == 0 or valleys.size == 0:
        return np.nan
    return (np.mean(s[peaks]) - np.mean(s[valleys])) / 2.0


def _stable_velocity(x, y, time_s, settle_idx):
    """Steady-state forward speed: |d/dt of straight-line ground displacement|.
    """

    if settle_idx >= len(time_s) - 2:
        return np.nan
    t = np.asarray(time_s[settle_idx:], dtype=float)
    x = np.asarray(x[settle_idx:], dtype=float)
    y = np.asarray(y[settle_idx:], dtype=float)
    if len(t) < 2:
        return np.nan
    vx, _ = np.polyfit(t, x, 1)
    vy, _ = np.polyfit(t, y, 1)
    return float(np.hypot(vx, vy))


def _scale_from_path(csv_path: str) -> float:
    """Pull the scale out of a folder name like 'zippy_sc6_12_2026-...'."""

    m = re.search(r"sc(\d+(?:_\d+)?)", os.path.basename(os.path.dirname(csv_path)))
    if m is None:
        raise ValueError(f"could not parse scale from {csv_path}")
    return float(m.group(1).replace("_", "."))


def extract_trial(csv_path: str, robot: str, law: str, get_hip_torque,
                  stabilization_s: float = DEFAULT_STABILIZATION_S) -> TrialMetrics:
    """Pull every Table-IV metric out of one saved-run CSV."""

    from .robots import ROBOTS
    cfg = ROBOTS[robot]
    df = pd.read_csv(csv_path)
    scale = _scale_from_path(csv_path)

    time_s = df["time"].to_numpy()
    T = float(np.diff(time_s[:5]).mean()) if len(time_s) > 5 else 0.001
    settle_idx = min(int(stabilization_s / T), max(len(df) - 2, 0))
    tail = df.iloc[settle_idx:]

    def col_first(name):
        if name in df.columns:
            v = df[name].dropna()
            if not v.empty:
                return float(v.iloc[0])
        return np.nan

    # Always recompute amplitudes and velocity from the raw data: the runner's
    # cached values use a simpler formula that doesn't filter intra-stride
    # wobble, so they would skew the fit.
    #
    # Axis convention note. The saved CSV reports Drake's world-frame
    # Roll/Pitch/Yaw. For robots whose URDF imports lying down we rotate the
    # base -90 deg about X at setup so it stands upright. After that rotation
    # Drake's "Roll_deg" tracks the forward-rocking motion (oscillates at the
    # stride frequency, once per step) and Drake's "Pitch_deg" tracks the
    # sideways-tipping motion (oscillates at half the stride frequency, once
    # per gait cycle). We change everything to match the convention where "roll" is
    # sideways tipping and "pitch" is forward rocking. We swap per-robot
    # (config.paper_swaps_roll_pitch) so Table IV is the right thing to
    # compare against.
    if cfg.paper_swaps_roll_pitch:
        roll  = _peak_to_valley_amplitude(tail["Pitch_deg"])   # paper roll  = sideways
        pitch = _peak_to_valley_amplitude(tail["Roll_deg"])    # paper pitch = forward
    else:
        roll  = _peak_to_valley_amplitude(tail["Roll_deg"])
        pitch = _peak_to_valley_amplitude(tail["Pitch_deg"])
    yaw   = _peak_to_valley_amplitude(tail["Yaw_deg"])
    velocity = _stable_velocity(df["x"].to_numpy(), df["y"].to_numpy(),
                                time_s, settle_idx)

    # Torque: from the manually-tuned lookup when available, otherwise the peak
    # measured value in the saved CSV.
    if get_hip_torque is not None:
        min_torque, torque_source = torque_lookup_diagnostics(get_hip_torque, scale)
    elif "hip_real_torque" in df.columns:
        min_torque = float(np.nanmax(np.abs(tail["hip_real_torque"].to_numpy())))
        torque_source = "measured"
    else:
        min_torque, torque_source = np.nan, "missing"

    mass = col_first("Total_mass_kg")
    leg_length = REF_LEG_LENGTH_M[robot] * scale

    # Did the robot actually walk? The
    # gait should show real attitude amplitude and the robot should have moved
    # at least a few body lengths during the stable region. Both Zippy and
    # Mugatu have body length ~1.5x leg length, so we use that.
    walked, reason = True, "ok"
    body_length = 1.5 * leg_length
    tail_x = df["x"].to_numpy()[settle_idx:]
    tail_y = df["y"].to_numpy()[settle_idx:]
    if tail_x.size >= 2:
        displacement = float(np.hypot(tail_x[-1] - tail_x[0], tail_y[-1] - tail_y[0]))
    else:
        displacement = 0.0
    if not (np.isfinite(velocity) and np.isfinite(roll) and np.isfinite(pitch)):
        walked, reason = False, "missing amplitude or velocity"
    elif displacement < 3 * body_length:
        walked, reason = (
            False,
            f"moved {displacement:.3f} m (< 3 body lengths = {3*body_length:.2f} m)",
        )
    elif roll < 0.5 and scale >= 5:
        walked, reason = False, "roll amplitude < 0.5 deg (no gait)"

    return TrialMetrics(
        csv_path=csv_path, robot=robot, law=law, scale=scale,
        leg_length_m=leg_length, mass_kg=mass, duration_s=float(time_s[-1]),
        velocity_mps=velocity, min_torque_Nm=min_torque, torque_source=torque_source,
        roll_amp_deg=roll, pitch_amp_deg=pitch, yaw_amp_deg=yaw,
        walked=walked, walked_reason=reason,
    )


# ---------------------------------------------------------------------------
# Cross-scale power-law fit
# ---------------------------------------------------------------------------
def fit_power_law(L, y):
    """Least-squares fit of log10(y) = alpha * log10(L) + beta. Returns
    (alpha, beta, R^2). Non-positive / non-finite samples are dropped."""
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(L) & np.isfinite(y) & (L > 0) & (y > 0)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    lx = np.log10(L[ok])
    ly = np.log10(y[ok])
    alpha, beta = np.polyfit(lx, ly, 1)
    pred = alpha * lx + beta
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(alpha), float(beta), r2


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def find_run_csvs(robot: str, law: str):
    variant = {"L2": "ml2_scaling", "L3": "ml3_scaling"}[law]
    pattern = os.path.join(REPO_DIR, variant, robot, "saved_sim_data",
                           "*", "all_data_*.csv")
    return sorted(glob.glob(pattern))


def analyze_pair(robot: str, law: str, stabilization_s: float = DEFAULT_STABILIZATION_S):
    """Build the per-trial table and the cross-scale fits for one (robot, law)."""
    csvs = find_run_csvs(robot, law)
    if not csvs:
        return None, None

    get_hip_torque = load_torque_function(robot, law)

    rows = []
    for c in csvs:
        try:
            rows.append(extract_trial(c, robot, law, get_hip_torque, stabilization_s))
        except Exception as exc:
            print(f"  skipping {c}: {exc}")
    if not rows:
        return None, None
    trials = pd.DataFrame([r.__dict__ for r in rows]).sort_values("scale").reset_index(drop=True)

    # Cross-scale fits use successful trials only
    valid = trials[trials["walked"]].copy()

    fits = {}
    L = valid["leg_length_m"].to_numpy()
    for key, col in [("velocity", "velocity_mps"), ("min_torque", "min_torque_Nm")]:
        a, b, r2 = fit_power_law(L, valid[col].to_numpy())
        fits[key] = dict(alpha=a, intercept_log10=b, r2=r2,
                         coefficient=10 ** b if np.isfinite(b) else np.nan)
    for key, col in [("roll_amplitude", "roll_amp_deg"),
                     ("pitch_amplitude", "pitch_amp_deg"),
                     ("yaw_amplitude", "yaw_amp_deg")]:
        v = valid[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        fits[key] = dict(mean_deg=float(v.mean()) if v.size else np.nan,
                         std_deg=float(v.std(ddof=0)) if v.size else np.nan)
    fits["n_valid"] = len(valid)
    fits["n_total"] = len(trials)
    return trials, fits


def print_pair_report(robot, law, trials, fits):
    target = TABLE_IV.get((robot, law), {})
    print(f"\n===== {robot}  /  {law}  =====")
    if trials is None:
        print("  (no saved runs)")
        return
    scales = sorted(trials["scale"].unique().tolist())
    print(f"  scales analysed:  {scales}")
    print(f"  trials:           {fits['n_total']} ({fits['n_valid']} walked successfully)")
    excluded = trials[~trials["walked"]]
    if not excluded.empty:
        for r in excluded.itertuples():
            print(f"    [excluded] sc{r.scale}: {r.walked_reason}")
    # Show what each torque value came from so I know whether it's
    # a tuned value, interpolated between two, or fell out of range.
    sources = trials[trials["walked"]][["scale", "torque_source"]]
    if not sources["torque_source"].eq("measured").all():
        print(f"  torque sources:   "
              + ", ".join(f"sc{r.scale}={r.torque_source}"
                          for r in sources.itertuples()))
    print()
    print(f"  Velocity   [m/s]   v   ~ L^{fits['velocity']['alpha']:.2f}  "
          f"(R^2={fits['velocity']['r2']:.3f})   paper: L^{target.get('v_exp','?')}")
    print(f"  Min torque [N m]   tau ~ L^{fits['min_torque']['alpha']:.2f}  "
          f"(R^2={fits['min_torque']['r2']:.3f})   paper: L^{target.get('tau_exp','?')}")
    print(f"  Roll  amp  [deg]   {fits['roll_amplitude']['mean_deg']:6.2f} "
          f"+/- {fits['roll_amplitude']['std_deg']:.2f}    paper: {target.get('theta_R','?')}")
    print(f"  Pitch amp  [deg]   {fits['pitch_amplitude']['mean_deg']:6.2f} "
          f"+/- {fits['pitch_amplitude']['std_deg']:.2f}    paper: {target.get('theta_P','?')}")
    print(f"  Yaw   amp  [deg]   {fits['yaw_amplitude']['mean_deg']:6.2f} "
          f"+/- {fits['yaw_amplitude']['std_deg']:.2f}    paper: {target.get('theta_Y','?')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(REF_LEG_LENGTH_M),
                    help="Limit analysis to one robot.")
    ap.add_argument("--law", choices=["L2", "L3"],
                    help="Limit analysis to one law.")
    ap.add_argument("--stabilization", type=float, default=DEFAULT_STABILIZATION_S,
                    help="Seconds of settling time to discard (default 5).")
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "analysis_out"),
                    help="Directory to write results CSVs.")
    args = ap.parse_args()

    robots = [args.robot] if args.robot else list(REF_LEG_LENGTH_M)
    laws = [args.law] if args.law else ["L2", "L3"]

    os.makedirs(args.out, exist_ok=True)
    summary_rows = []

    for robot in robots:
        for law in laws:
            trials, fits = analyze_pair(robot, law, args.stabilization)
            print_pair_report(robot, law, trials, fits)
            if trials is None:
                continue
            trials.to_csv(os.path.join(args.out, f"trials_{robot}_{law}.csv"),
                          index=False)
            summary_rows.append({
                "robot": robot, "law": law, "n_trials": len(trials),
                "v_alpha":    fits["velocity"]["alpha"],
                "v_R2":       fits["velocity"]["r2"],
                "tau_alpha":  fits["min_torque"]["alpha"],
                "tau_R2":     fits["min_torque"]["r2"],
                "roll_mean":  fits["roll_amplitude"]["mean_deg"],
                "roll_std":   fits["roll_amplitude"]["std_deg"],
                "pitch_mean": fits["pitch_amplitude"]["mean_deg"],
                "pitch_std":  fits["pitch_amplitude"]["std_deg"],
                "yaw_mean":   fits["yaw_amplitude"]["mean_deg"],
                "yaw_std":    fits["yaw_amplitude"]["std_deg"],
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.out, "paper_table_iv_reproduction.csv")
        summary.to_csv(summary_csv, index=False)
        print(f"\nSummary written to {summary_csv}")
    else:
        print("\nNo runs analysed. Generate some with --save-data first, e.g.:")
        print("  python run.py --robot zippy --law L2 --scales 1 6.12 40 "
              "--simulate --duration 25 --save-data")


if __name__ == "__main__":
    main()
