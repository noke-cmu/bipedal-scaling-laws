"""Post-processing of a simulation run.

Everything here is plot-free cuz I used to plot everything. Uses raw state arrays and returns the derived
quantities the CSV and the plotting code both need. Keeping the math separate
from the figure-drawing lets us cache the results and skip the slow plotting
step entirely when only data is needed.
"""
import numpy as np
from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion


def quat_to_rpy(qw, qx, qy, qz):
    """Vectorised quaternion -> (roll, pitch, yaw)"""

    quats = np.column_stack([qw, qx, qy, qz])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rpy = np.array([RollPitchYaw(Quaternion(q)).vector() for q in quats])
    return (np.unwrap(rpy[:, 0]), np.unwrap(rpy[:, 1]), np.unwrap(rpy[:, 2]))


def angular_velocity_to_rpy_rates(omega_x, omega_y, omega_z, roll, pitch):
    """Body-frame angular velocity -> Euler-rate (roll/pitch/yaw rate)."""

    cosp = np.cos(pitch)
    # Singular at +/- pi/2; in this regime we never approach that.
    roll_rate = omega_x + np.sin(roll) * np.tan(pitch) * omega_y + np.cos(roll) * np.tan(pitch) * omega_z
    pitch_rate = np.cos(roll) * omega_y - np.sin(roll) * omega_z
    yaw_rate = (np.sin(roll) / cosp) * omega_y + (np.cos(roll) / cosp) * omega_z
    return roll_rate, pitch_rate, yaw_rate


def amplitude_after_stable(signal, stabilization_period):
    """Mean peak-to-valley amplitude over the post-settling part of the signal.
    """

    from scipy.signal import find_peaks
    tail = np.asarray(signal[stabilization_period:])
    if tail.size == 0:
        return np.nan, np.nan
    peaks, _ = find_peaks(tail)
    valleys, _ = find_peaks(-tail)
    if peaks.size == 0 or valleys.size == 0:
        return np.nan, np.nan
    avg_amp = (np.mean(tail[peaks]) - np.mean(tail[valleys])) / 2.0
    max_amp = (tail.max() - tail.min()) / 2.0
    return avg_amp, max_amp


def split_states(states, T):
    """Unpack a (N, 15) state matrix into a dict of named columns.
    Drake gives us the floating-base quaternion + xyz + hip angle, then the
    six body-velocity components plus the hip velocity, all stacked into a
    flat state vector.
    """

    s = np.asarray(states)
    N = len(s)
    return {
        "time": np.arange(N) * T,
        "qw": s[:, 0], "qx": s[:, 1], "qy": s[:, 2], "qz": s[:, 3],
        "x":  s[:, 4], "y":  s[:, 5], "z":  s[:, 6],
        "hip_real_angle": s[:, 7],
        "omega_x": s[:, 8], "omega_y": s[:, 9], "omega_z": s[:, 10],
        "xdot": s[:, 11], "ydot": s[:, 12], "zdot": s[:, 13],
        "hip_real_angvel": s[:, 14],
    }


def grf_components(left_forces, left_points, right_forces, right_points):
    """Unpack contact-force/point lists into Numpy x/y/z column arrays."""

    lf, lp = np.asarray(left_forces), np.asarray(left_points)
    rf, rp = np.asarray(right_forces), np.asarray(right_points)
    return {
        "left_x": lp[:, 0], "left_y": lp[:, 1],
        "right_x": rp[:, 0], "right_y": rp[:, 1],
        "left_fx": lf[:, 0], "left_fy": lf[:, 1], "left_fz": lf[:, 2],
        "right_fx": rf[:, 0], "right_fy": rf[:, 1], "right_fz": rf[:, 2],
    }


def power_and_energy(hip_angvel, hip_torque, T):
    """Instantaneous mechanical power and cumulative energy at the hip."""

    power = np.abs(np.asarray(hip_angvel) * np.asarray(hip_torque).squeeze())
    energy = np.cumsum(power) * T
    return power, energy


def steady_state_velocity(x, y, time_array, stabilization_period):
    """Average horizontal walking speed after the gait has settled."""
    
    dx = x[-1] - x[stabilization_period]
    dy = y[-1] - y[stabilization_period]
    dt = time_array[-1] - time_array[stabilization_period]
    if dt <= 0:
        return np.nan
    return np.hypot(dx, dy) / dt
