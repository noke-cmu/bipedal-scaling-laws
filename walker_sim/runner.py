"""One runner for every robot, every scaling law, every scale. Replaces my OG simulate and multi_simulate scripts.

  - `load_robot()`:  import a robot's model_definition under a given scaling law
  - `simulate()`:    run one simulation and return everything we measured
  - `save_run()`:    write CSV + plots for a finished run
  - `run()`:         glue for the command line - takes the friendly args from
                     `run.py` and dispatches to simulate/save as needed

Each robot's `model_definition.py` is left untouched: that's where the
genuinely-different physics and per-law torque tuning lives.
"""
from __future__ import annotations
import importlib
import os
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from pydrake.all import (
    AddDefaultVisualization, ContactVisualizer, ContactVisualizerParams,
    JointSliders, MeshcatVisualizer, RigidTransform, RotationMatrix,
    ScopedName, Simulator,
)

from . import analysis as ana
from . import plotting as plots
from .robots import LAWS, ROBOTS, RobotConfig


REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
STABILIZATION_SECONDS = 5.0   # gait considered "settled" after this long


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_robot(robot: str, law: str):
    """Import the model_definition for a (robot, law) pair from its own package.
    """

    if robot not in ROBOTS:
        raise ValueError(f"unknown robot {robot!r}; choose from {list(ROBOTS)}")
    if law not in LAWS:
        raise ValueError(f"unknown law {law!r}; choose from {list(LAWS)}")
    variant = LAWS[law][0]
    variant_dir = os.path.join(REPO_DIR, variant)
    if variant_dir not in sys.path:
        sys.path.insert(0, variant_dir)
    # Forget any other variant's cached import so we always get the right one.
    for m in list(sys.modules):
        if m == robot or m.startswith(robot + "."):
            del sys.modules[m]
    return importlib.import_module(f"{robot}.model_definition")


# ---------------------------------------------------------------------------
# One simulation
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Everything one simulated trial produces, ready for CSV + plotting."""

    config: RobotConfig
    scale: float
    T: float
    duration: float
    total_mass: float
    states: np.ndarray
    hip_real_torque: np.ndarray
    hip_target: np.ndarray
    com_robot: np.ndarray            # (N, 3)
    com_robot_vel: np.ndarray        # (N, 3)
    com_per_link: dict               # name -> (N, 3)
    contact_times: np.ndarray
    left_forces: np.ndarray
    left_points: np.ndarray
    right_forces: np.ndarray
    right_points: np.ndarray
    meshcat_url: str = ""


def _build_diagram(model, config, scale, ground_friction, feet_friction, T, controller_period,
                   meshcat, collect_contacts: bool):
    """Plant + controller + meshcat, ready to simulate."""
    plant, scene_graph, builder, instance = model.setup_walker_plant(
        scale=scale, ground_friction=ground_friction,
        feet_friction=feet_friction, timestep=T)

    controller = builder.AddSystem(model.Controller(
        scale=scale, ground_friction=ground_friction,
        feet_friction=feet_friction, control_period=controller_period))
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("state"))
    builder.Connect(controller.get_output_port(),  plant.get_actuation_input_port())

    contact_system = None
    if collect_contacts:
        pairs = [
            [ScopedName("walker", config.contact_body_left),  ScopedName("walker", "ground")],
            [ScopedName("walker", config.contact_body_right), ScopedName("walker", "ground")],
        ]
        # Older robots' ContactResultsToArray takes a timestep; newer ones don't.
        try:
            contact_system = builder.AddSystem(
                model.ContactResultsToArray(plant, scene_graph, T, pairs))
        except TypeError:
            contact_system = builder.AddSystem(
                model.ContactResultsToArray(plant, scene_graph, pairs))
        builder.Connect(plant.get_contact_results_output_port(),
                        contact_system.GetInputPort("contact_results"))

    meshcat.Delete()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    ContactVisualizer.AddToBuilder(
        builder, plant, meshcat, ContactVisualizerParams(radius=0.002 * scale))

    return plant, builder.Build(), controller, contact_system, instance


def simulate(robot: str, law: str, *, scale=1.0, duration=None,
             ground_friction=0.9, feet_friction=0.9, collect_data=True, meshcat=None):
    """Simulate one (robot, law, scale) trial and return everything we measured."""
    model = load_robot(robot, law)
    config = ROBOTS[robot]
    if duration is None:
        duration = config.default_duration
    if meshcat is None:
        from pydrake.all import StartMeshcat
        meshcat = StartMeshcat()

    T = 0.001
    n_steps = int(duration / T)

    plant, diagram, controller, contact_system, instance = _build_diagram(
        model, config, scale, ground_friction, feet_friction,
        T=T, controller_period=10 * T, meshcat=meshcat,
        collect_contacts=collect_data)

    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(sim_context)

    # Walker meshes come in lying down; rotate so the base sits upright.
    if config.needs_x_rotation:
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName(config.base_body),
                              RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2)))
    plant.SetPositionsAndVelocities(plant_context, model.get_home_state(scale))

    total_mass = plant.CalcTotalMass(plant_context, [instance])
    print(f"Total mass at scale {scale}: {total_mass * 1000:.2f} g")

    # Cache the body lookups: we'd otherwise call GetBodyByName per link per step.
    exclude = {"ground", "inclined_plane"}
    link_names = [plant.get_body(i).name() for i in plant.GetBodyIndices(instance)
                  if plant.get_body(i).name() not in exclude]
    link_bodies = {n: plant.GetBodyByName(n) for n in link_names}

    visualizer_systems = [s for s in diagram.GetSystems()
                          if "Meshcat" in type(s).__name__]
    for v in visualizer_systems:
        if hasattr(v, "StartRecording"):
            v.StartRecording(False)

    if not collect_data:
        # No-save fast path: just run the physics, skip the per-step capture.
        simulator.AdvanceTo(n_steps * T)
        for v in visualizer_systems:
            if hasattr(v, "PublishRecording"):
                v.PublishRecording()
        return RunResult(
            config=config, scale=scale, T=T, duration=duration, total_mass=total_mass,
            states=np.empty((0, 15)), hip_real_torque=np.empty(0), hip_target=np.empty(0),
            com_robot=np.empty((0, 3)), com_robot_vel=np.empty((0, 3)),
            com_per_link={}, contact_times=np.empty(0),
            left_forces=np.empty((0, 3)),  left_points=np.empty((0, 3)),
            right_forces=np.empty((0, 3)), right_points=np.empty((0, 3)),
            meshcat_url=meshcat.web_url(),
        )

    # Data-collection fast path: append into Python lists, then arrayify once.
    states, hip_torque, hip_target = [], [], []
    com_robot, com_robot_vel = [], []
    com_per_link = {n: [] for n in link_names}
    get_target = config.get_target

    t0 = time.time()
    for idx in range(n_steps):
        simulator.AdvanceTo(T * (idx + 1))
        states.append(plant.GetPositionsAndVelocities(plant_context))
        hip_torque.append(controller.control_signal.copy())
        hip_target.append(get_target(controller))
        com_robot.append(plant.CalcCenterOfMassPositionInWorld(plant_context, [instance]))
        com_robot_vel.append(plant.CalcCenterOfMassTranslationalVelocityInWorld(
            plant_context, [instance]))
        for name, body in link_bodies.items():
            X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
            p_Bcm = body.CalcCenterOfMassInBodyFrame(plant_context)
            com_per_link[name].append((X_WB @ np.append(p_Bcm, 1.0))[:3])
    sim_seconds = time.time() - t0

    # Stop and publish the meshcat recording once.
    for v in visualizer_systems:
        if hasattr(v, "PublishRecording"):
            v.PublishRecording()

    # Contact-force log: keep only the timestamps that have entries on both feet.
    if contact_system is not None:
        force_dict, point_dict = contact_system.get_forces_and_points()
        times = np.array(list(force_dict.keys()), dtype=float)
        left_f  = np.array([force_dict[t]["left_foot_force"]  for t in force_dict])
        right_f = np.array([force_dict[t]["right_foot_force"] for t in force_dict])
        left_p  = np.array([point_dict[t]["left_foot_point"]  for t in point_dict])
        right_p = np.array([point_dict[t]["right_foot_point"] for t in point_dict])
    else:
        times = np.empty(0)
        left_f = right_f = left_p = right_p = np.empty((0, 3))

    print(f"Simulated {duration:.1f} s of walking in {sim_seconds:.1f} s wall-clock.")
    return RunResult(
        config=config, scale=scale, T=T, duration=duration, total_mass=total_mass,
        states=np.array(states),
        hip_real_torque=np.array(hip_torque),
        hip_target=np.array(hip_target),
        com_robot=np.array(com_robot),
        com_robot_vel=np.array(com_robot_vel),
        com_per_link={n: np.array(v) for n, v in com_per_link.items()},
        contact_times=times,
        left_forces=left_f, left_points=left_p,
        right_forces=right_f, right_points=right_p,
        meshcat_url=meshcat.web_url(),
    )


# ---------------------------------------------------------------------------
# Save + plot one run
# ---------------------------------------------------------------------------
def save_run(result: RunResult, output_root: str, save_meshcat_recording: bool = True):
    """Compute amplitudes, write the CSV, write every plot, and save the recording."""
    s = ana.split_states(result.states, result.T)
    time_array = s["time"]
    N = len(time_array)
    settle = min(int(STABILIZATION_SECONDS / result.T), max(N - 2, 0))

    # Numerical post-processing (the math the plots also need).
    rolls, pitches, yaws = ana.quat_to_rpy(s["qw"], s["qx"], s["qy"], s["qz"])
    rolls_deg, pitches_deg, yaws_deg = np.degrees(rolls), np.degrees(pitches), np.degrees(yaws)
    roll_rate, pitch_rate, yaw_rate = ana.angular_velocity_to_rpy_rates(
        s["omega_x"], s["omega_y"], s["omega_z"], rolls, pitches)
    avg_roll, _  = ana.amplitude_after_stable(rolls_deg,  settle)
    avg_pitch, _ = ana.amplitude_after_stable(pitches_deg, settle)
    avg_yaw, _   = ana.amplitude_after_stable(yaws_deg,    settle)

    power, energy = ana.power_and_energy(s["hip_real_angvel"], result.hip_real_torque, result.T)
    vel_steady = ana.steady_state_velocity(s["x"], s["y"], time_array, settle)
    grf = ana.grf_components(result.left_forces, result.left_points,
                             result.right_forces, result.right_points)

    # Folder layout: saved_sim_data/<robot>_<law>_sc<scale>_<timestamp>/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scale_str = f"sc{result.scale}".replace(".", "_")
    folder = os.path.join(output_root,
                          f"{result.config.name}_{scale_str}_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    print(f"Saving to {folder}")

    # CSV: one row per sim step, with all the time-series columns and a few
    # scalar "stats" columns repeated on the first row (matches the old layout).
    n_contacts = min(len(result.contact_times), N)
    data = {
        "time": time_array,
        "qw": s["qw"], "qx": s["qx"], "qy": s["qy"], "qz": s["qz"],
        "x":  s["x"],  "y":  s["y"],  "z":  s["z"],
        "x_dot": s["xdot"], "y_dot": s["ydot"], "z_dot": s["zdot"],
        "COM_x_position": result.com_robot[:, 0],
        "COM_y_position": result.com_robot[:, 1],
        "COM_z_position": result.com_robot[:, 2],
        "COM_x_velocity": result.com_robot_vel[:, 0],
        "COM_y_velocity": result.com_robot_vel[:, 1],
        "COM_z_velocity": result.com_robot_vel[:, 2],
        "Roll_rad":  rolls,  "Pitch_rad":  pitches,  "Yaw_rad":  yaws,
        "Roll_deg":  rolls_deg, "Pitch_deg":  pitches_deg, "Yaw_deg":  yaws_deg,
        "Roll_rate_radpersec":  roll_rate,
        "Pitch_rate_radpersec": pitch_rate,
        "Yaw_rate_radpersec":   yaw_rate,
        "X_euler_rate_radpersec": s["omega_x"],
        "Y_euler_rate_radpersec": s["omega_y"],
        "Z_euler_rate_radpersec": s["omega_z"],
        "hip_real_angle":   s["hip_real_angle"],
        "hip_real_angvel":  s["hip_real_angvel"],
        result.config.target_column: np.asarray(result.hip_target).squeeze(),
        "hip_real_torque":  np.asarray(result.hip_real_torque).squeeze(),
        "total_power":      power,
        "cumulative_energy": energy,
    }
    # Contact-force columns: padded with NaN to match N.
    for key in ("left_x", "left_y", "right_x", "right_y",
                "left_fx", "left_fy", "left_fz",
                "right_fx", "right_fy", "right_fz"):
        col = np.full(N, np.nan)
        col[:n_contacts] = grf[key][:n_contacts]
        data[key] = col
    df = pd.DataFrame(data)
    # Scalar stats: written once on the first row.
    df.loc[0, "Total_mass_kg"] = result.total_mass
    df.loc[0, "Average_roll_amplitude_after_stable_walking_degrees"]  = avg_roll
    df.loc[0, "Average_pitch_amplitude_after_stable_walking_degrees"] = avg_pitch
    df.loc[0, "Average_yaw_amplitude_after_stable_walking_degrees"]   = avg_yaw
    df.loc[0, "Steady_state_velocity_m_per_s"] = vel_steady

    csv_path = os.path.join(folder, f"all_data_{timestamp}.csv")
    df.to_csv(csv_path, index=True)
    print(f"Sim data saved to {csv_path}")

    # All the plots, in one place.
    print(f"Writing plots to {folder}")
    plots_t0 = time.time()
    plots.plot_attitude(folder, time_array, rolls_deg, pitches_deg, yaws_deg, settle)
    plots.plot_rates(folder, time_array, s["omega_x"], s["omega_y"], s["omega_z"],
                     roll_rate, pitch_rate, yaw_rate)
    plots.plot_com(folder, time_array, result.com_robot[:, 0], result.com_robot[:, 1],
                   result.com_robot[:, 2], result.com_robot_vel[:, 0],
                   result.com_robot_vel[:, 1], result.com_robot_vel[:, 2],
                   result.total_mass, result.T)
    plots.plot_link_coms(folder, result.com_per_link)
    if n_contacts > 0:
        plots.plot_grf(folder, time_array, result.contact_times, grf)
    plots.plot_position_and_velocity(folder, time_array,
                                     s["x"], s["y"], s["z"],
                                     s["xdot"], s["ydot"], s["zdot"])
    plots.plot_controls(folder, time_array, result.hip_real_torque, result.hip_target,
                        s["hip_real_angle"], s["hip_real_angvel"],
                        result.config.target_column)
    plots.plot_power(folder, time_array, power, energy)
    plots.plot_stability(folder, settle, rolls, roll_rate, pitches, pitch_rate,
                         yaws, yaw_rate, s["x"], s["y"], s["z"],
                         s["xdot"], s["ydot"], s["zdot"])
    print(f"Plots done in {time.time() - plots_t0:.1f} s.")

    if save_meshcat_recording and result.meshcat_url:
        try:
            r = requests.get(result.meshcat_url + "/download", timeout=10)
            path = os.path.join(folder, f"meshcat_recording_scale_{result.scale}.html")
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"Meshcat recording saved to {path}")
        except requests.RequestException as exc:
            print(f"(Meshcat recording skipped: {exc})")

    print(
        f"\n=== summary for {result.config.name} at scale {result.scale} ===\n"
        f"  mass:                {result.total_mass * 1000:7.2f} g\n"
        f"  roll  amplitude:     {avg_roll:6.3f} deg\n"
        f"  pitch amplitude:     {avg_pitch:6.3f} deg\n"
        f"  yaw   amplitude:     {avg_yaw:6.3f} deg\n"
        f"  steady-state speed:  {vel_steady:6.3f} m/s\n"
    )
    return folder


# ---------------------------------------------------------------------------
# Joint-slider viewer (no walking, just dragging joints)
# ---------------------------------------------------------------------------
def show_joint_sliders(robot, law, scale=1.0, ground_friction=0.9, feet_friction=0.9):
    """Open the Drake JointSliders viewer instead of running a simulation."""
    model = load_robot(robot, law)
    config = ROBOTS[robot]

    from pydrake.all import StartMeshcat
    meshcat = StartMeshcat()

    plant, scene_graph, builder, _ = model.setup_walker_plant(
        scale=scale, ground_friction=ground_friction, feet_friction=feet_friction)
    builder.AddSystem(JointSliders(meshcat, plant))
    meshcat.Delete()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    ContactVisualizer.AddToBuilder(builder, plant, meshcat,
                                   ContactVisualizerParams(radius=0.001 * scale))
    AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    sim = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(sim.get_mutable_context())
    if config.needs_x_rotation:
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName(config.base_body),
                              RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2)))
    plant.SetPositionsAndVelocities(plant_context, model.get_home_state(scale))

    print(f"Joint-slider viewer for {robot} ({law}) at scale {scale}.")
    print(f"Open in your browser:  {meshcat.web_url()}")
    print("Ctrl-C in the terminal when you're done.")
    try:
        sim.AdvanceTo(1e9)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Command-line glue (called from run.py)
# ---------------------------------------------------------------------------
def run(robot, law, *, simulate_flag=False, scale=None, scales=None,
        duration=None, save_data=False, ground_friction=0.9, feet_friction=0.9):
    """The single dispatch point."""
    if save_data and duration is not None and duration < 6:
        sys.exit("[setup] --save-data needs at least ~6 s of simulation so the\n"
                 "        analysis has steady-state walking to measure (it\n"
                 "        discards the first 5 s). Try --duration 25.")

    if not simulate_flag and not scales:
        show_joint_sliders(robot, law, scale=scale or 1.0,
                           ground_friction=ground_friction, feet_friction=feet_friction)
        return 0

    scale_list = scales if scales else [scale if scale is not None else 1.0]
    output_root = os.path.join(REPO_DIR, LAWS[law][0], robot, "saved_sim_data")
    if save_data:
        os.makedirs(output_root, exist_ok=True)

    from pydrake.all import StartMeshcat
    meshcat = StartMeshcat()
    print(f"Meshcat is live at: {meshcat.web_url()}")

    for s in scale_list:
        print(f"\n--- {robot} ({law}) at scale {s} ---")
        result = simulate(robot, law, scale=s, duration=duration,
                          ground_friction=ground_friction, feet_friction=feet_friction,
                          collect_data=save_data, meshcat=meshcat)
        if save_data:
            save_run(result, output_root)
        else:
            # No-save case: just leave the meshcat link open so the user can
            # download a recording manually if they want one.
            print(f"View the recording at: {result.meshcat_url}/download")
            try:
                webbrowser.open(result.meshcat_url + "/download")
            except Exception:
                pass
    return 0
