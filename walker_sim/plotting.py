
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # always headless: writes files, never opens a window
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.figure import Figure as _Figure

plt.rcParams.update({
    "figure.max_open_warning": 0,
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
    "agg.path.chunksize": 10000,
})

# Light PNG compression: smaller writes, smaller file barely changes.
_orig_savefig = _Figure.savefig
def _fast_savefig(self, *a, **k):
    k.setdefault("pil_kwargs", {"compress_level": 1})
    return _orig_savefig(self, *a, **k)
_Figure.savefig = _fast_savefig


# ---------------------------------------------------------------------------
# Small helpers (the building blocks every figure uses)
# ---------------------------------------------------------------------------
def _decimate(values, max_points=1000):
    """Down-sample for rendering only - never touches the data on disk."""
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[0] <= max_points:
        return arr
    return arr[:: max(1, arr.shape[0] // max_points)]


def _lineplot(folder, name, x, y, *, xlabel, ylabel, title, figsize=None):
    """One 2-D line plot -> one PNG."""
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    fig.savefig(os.path.join(folder, name + ".png"))
    plt.close(fig)


def _phase_plot(folder, name, a, b, settle, *, xlabel, ylabel, title):
    """Phase-portrait style plot, with settling vs steady-state coloured."""
    fig = plt.figure()
    plt.plot(a[:settle], b[:settle], color="g", label="Gait Stabilization Period")
    plt.plot(a[settle:], b[settle:], color="r", label="Stable Gait")
    plt.plot(a[0],  b[0],  "o", color="purple", label="Start point")
    plt.plot(a[-1], b[-1], "bD",                label="End point")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, name + ".png"))
    plt.close(fig)


def _line3d_collection(folder, name, x, y, z, *,
                       xlabel, ylabel, zlabel, title, color_by=None, cbar_label="Time"):
    """3-D coloured line - the slow one we always decimate before drawing."""
    x, y, z = _decimate(x), _decimate(y), _decimate(z)
    color_by = _decimate(z if color_by is None else color_by)
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=color_by.min(), vmax=color_by.max())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lc = Line3DCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(color_by)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(lc, ax=ax, label=cbar_label)
    fig.savefig(os.path.join(folder, name + ".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# The plot suites (one function per logical group, all using the helpers)
# ---------------------------------------------------------------------------
def plot_attitude(folder, time_array, rolls_deg, pitches_deg, yaws_deg, settle):
    """Roll/pitch/yaw v time, plus 2-D and 3-D phase plots between them."""
    _lineplot(folder, "roll_v_time",  time_array, rolls_deg,
              xlabel="Time [s]", ylabel="Roll [degrees]",  title="Roll")
    _lineplot(folder, "pitch_v_time", time_array, pitches_deg,
              xlabel="Time [s]", ylabel="Pitch [degrees]", title="Pitch")
    _lineplot(folder, "yaw_v_time",   time_array, yaws_deg,
              xlabel="Time [s]", ylabel="Yaw [degrees]",   title="Yaw")
    _phase_plot(folder, "roll_v_yaw",   rolls_deg, yaws_deg,    settle,
                xlabel="Roll [degrees]", ylabel="Yaw [degrees]",   title="Roll vs Yaw")
    _phase_plot(folder, "roll_v_pitch", rolls_deg, pitches_deg, settle,
                xlabel="Roll [degrees]", ylabel="Pitch [degrees]", title="Roll vs Pitch")
    _line3d_collection(folder, "roll_v_yaw3D",   rolls_deg, yaws_deg,    time_array,
                       xlabel="Roll [deg]", ylabel="Yaw [deg]",   zlabel="Time [s]",
                       title="Roll vs Yaw vs Time")
    _line3d_collection(folder, "roll_v_pitch3D", rolls_deg, pitches_deg, time_array,
                       xlabel="Roll [deg]", ylabel="Pitch [deg]", zlabel="Time [s]",
                       title="Roll vs Pitch vs Time")
    _line3d_collection(folder, "rpy3D",          rolls_deg, pitches_deg, yaws_deg,
                       xlabel="Roll [deg]", ylabel="Pitch [deg]", zlabel="Yaw [deg]",
                       title="Roll vs Pitch vs Yaw", cbar_label="Yaw")


def plot_rates(folder, time_array, omega_x, omega_y, omega_z, roll_rate, pitch_rate, yaw_rate):
    """Body-frame angular velocity and the corresponding RPY rates."""
    for fname, title, components in [
        ("euler_rates", "Body angular velocities",
         [("X", "r", omega_x), ("Y", "b", omega_y), ("Z", "g", omega_z)]),
        ("rpy_rates",   "Roll/pitch/yaw rates",
         [("Roll", "r", roll_rate), ("Yaw", "b", yaw_rate), ("Pitch", "g", pitch_rate)]),
    ]:
        fig = plt.figure(figsize=(12, 6))
        for label, color, data in components:
            plt.plot(time_array, data, color=color, label=label)
        plt.title(title); plt.xlabel("Time [s]"); plt.ylabel("[radians/s]")
        plt.legend()
        fig.savefig(os.path.join(folder, fname + ".png"))
        plt.close(fig)


def plot_com(folder, time_array, com_x, com_y, com_z, com_vx, com_vy, com_vz, total_mass, T):
    """Centre-of-mass position / velocity / acceleration / force."""
    # 3-panel position over time
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for ax, signal, label in zip(axes, (com_x, com_y, com_z), ("X", "Y", "Z")):
        ax.plot(time_array, signal)
        ax.set_ylabel(f"COM in {label} [m]")
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Center of mass position over time")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "com_position.png"))
    plt.close(fig)

    # 2-D top-down trajectory
    _lineplot(folder, "com_xy", com_x, com_y,
              xlabel="COM in X [m]", ylabel="COM in Y [m]",
              title="Centre-of-mass trajectory (top-down)")

    # 3-D trajectory (decimated)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(_decimate(com_x), _decimate(com_y), _decimate(com_z),
              label="CoM Trajectory", color="blue")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D Center of Mass Trajectory")
    fig.savefig(os.path.join(folder, "com_position_3d.png"), bbox_inches="tight")
    plt.close(fig)

    # Velocity, acceleration, force (derived from position)
    com_accx = np.diff(com_vx) / T
    com_accy = np.diff(com_vy) / T
    com_accz = np.diff(com_vz) / T
    for fname, title, ylabel, signals in [
        ("com_vel",   "COM velocity",     "[m/s]",       (com_vx, com_vy, com_vz)),
        ("com_acc",   "COM acceleration", "[m/s^2]",     (com_accx, com_accy, com_accz)),
        ("com_force", "COM force",        "[N]",
            (com_accx * total_mass, com_accy * total_mass, com_accz * total_mass)),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        for ax, signal, axname in zip(axes, signals, ("X", "Y", "Z")):
            ax.plot(signal)
            ax.set_ylabel(f"{axname} {ylabel}")
        axes[0].set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, fname + ".png"))
        plt.close(fig)


def plot_link_coms(folder, com_per_link):
    """Per-link centre-of-mass trajectories: one 3-D overlay + a few 2-D views."""
    if not com_per_link:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for name, coms in com_per_link.items():
        c = _decimate(coms)
        ax.plot3D(c[:, 0], c[:, 1], c[:, 2], label=name)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Per-link COM trajectories (3D)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)
    fig.savefig(os.path.join(folder, "com_all_links_3d.png"))
    plt.close(fig)

    for fname, ix, iy, xlabel, ylabel in [
        ("com_all_links_xz", 0, 2, "X [m]", "Z [m]"),
        ("com_all_links_xy", 0, 1, "X [m]", "Y [m]"),
        ("com_all_links_yz", 1, 2, "Y [m]", "Z [m]"),
    ]:
        fig = plt.figure()
        for name, coms in com_per_link.items():
            c = _decimate(coms)
            plt.plot(c[:, ix], c[:, iy], label=name)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(f"Per-link COMs: {xlabel} vs {ylabel}")
        plt.legend(fontsize="x-small", loc="best")
        fig.savefig(os.path.join(folder, fname + ".png"))
        plt.close(fig)


def plot_grf(folder, time_array, contact_times, grf):
    """Ground reaction forces and contact point trajectories."""
    n = min(len(contact_times), len(grf["left_fx"]))
    t = np.asarray(contact_times[:n])
    for fname, signals, title, ylabel in [
        ("grf_left",  ("left_fx",  "left_fy",  "left_fz"),  "Left foot GRF",  "Force [N]"),
        ("grf_right", ("right_fx", "right_fy", "right_fz"), "Right foot GRF", "Force [N]"),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for ax, key, axname in zip(axes, signals, ("X", "Y", "Z")):
            data = grf[key][:n]
            # Z forces are saved sign-flipped to match Drake's convention
            ax.plot(t, -data if axname == "Z" else data)
            ax.set_ylabel(f"{axname} {ylabel}")
        axes[-1].set_xlabel("Time [s]"); axes[0].set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, fname + ".png"))
        plt.close(fig)

    # Contact-point trajectories (top-down)
    fig = plt.figure()
    plt.plot(grf["left_x"][:n],  grf["left_y"][:n],  label="left foot")
    plt.plot(grf["right_x"][:n], grf["right_y"][:n], label="right foot")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title("Foot contact-point trajectories"); plt.legend()
    fig.savefig(os.path.join(folder, "contact_points_xy.png"))
    plt.close(fig)


def plot_position_and_velocity(folder, time_array, x, y, z, xdot, ydot, zdot):
    """Position and velocity time series, six lines total."""
    for fname, title, ylabel, signals, labels in [
        ("position", "Body position",   "[m]",
            (x, y, z),         ("X", "Y", "Z")),
        ("velocity", "Body velocity",   "[m/s]",
            (xdot, ydot, zdot), ("Xdot", "Ydot", "Zdot")),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for ax, sig, lbl in zip(axes, signals, labels):
            ax.plot(time_array, sig)
            ax.set_ylabel(f"{lbl} {ylabel}")
        axes[-1].set_xlabel("Time [s]"); axes[0].set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, fname + ".png"))
        plt.close(fig)


def plot_controls(folder, time_array, hip_real_torque, hip_target_signal,
                  hip_real_angle, hip_real_angvel, target_label):
    """Hip torque, hip target signal, joint angle and velocity."""
    series = [
        ("hip_real_torque", "Hip motor torque",  "Torque [Nm]", hip_real_torque),
        ("hip_target",      f"Hip target ({target_label})", "Target", hip_target_signal),
        ("hip_real_angle",  "Hip angle",         "Angle [rad]", hip_real_angle),
        ("hip_real_angvel", "Hip angular vel.",  "Ang. vel. [rad/s]", hip_real_angvel),
    ]
    for fname, title, ylabel, data in series:
        _lineplot(folder, fname, time_array, np.asarray(data).squeeze(),
                  xlabel="Time [s]", ylabel=ylabel, title=title)


def plot_power(folder, time_array, power, energy):
    """Instantaneous power and cumulative energy."""
    _lineplot(folder, "power",  time_array, power,
              xlabel="Time [s]", ylabel="Power [W]",  title="Hip mechanical power")
    _lineplot(folder, "energy", time_array, energy,
              xlabel="Time [s]", ylabel="Energy [J]", title="Cumulative mechanical energy")


def plot_stability(folder, settle, rolls, roll_rate, pitches, pitch_rate, yaws, yaw_rate,
                   x, y, z, xdot, ydot, zdot):
    """Limit-cycle phase portraits: angle vs rate, position vs velocity."""
    pairs = [
        ("stability_p_v_prate", pitches,    pitch_rate, "Pitch [rad]",   "Pitch rate [rad/s]",   "Pitch limit cycle"),
        ("stability_r_v_rrate", rolls,      roll_rate,  "Roll [rad]",    "Roll rate [rad/s]",    "Roll limit cycle"),
        ("stability_y_v_yrate", yaws,       yaw_rate,   "Yaw [rad]",     "Yaw rate [rad/s]",     "Yaw limit cycle"),
        ("stability_x_v_xdot",  x,          xdot,       "X [m]",         "X velocity [m/s]",     "X limit cycle"),
        ("stability_y_v_ydot",  y,          ydot,       "Y [m]",         "Y velocity [m/s]",     "Y limit cycle"),
        ("stability_z_v_zdot",  z,          zdot,       "Z [m]",         "Z velocity [m/s]",     "Z limit cycle"),
    ]
    for name, a, b, xl, yl, title in pairs:
        _phase_plot(folder, name, np.asarray(a), np.asarray(b), settle,
                    xlabel=xl, ylabel=yl, title=title)
