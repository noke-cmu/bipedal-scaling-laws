"""Per-robot configuration: documents changes between the robots, naming, and the sims based on my OG code
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    name: str
    contact_body_left: str    # link name used in the left-foot/ground contact pair
    contact_body_right: str   # link name used in the right-foot/ground contact pair
    base_body: str            # link Drake treats as the floating base (for SetFreeBodyPose)
    needs_x_rotation: bool    # True if the URDF imports lying down and needs -90deg around X
    paper_swaps_roll_pitch: bool   # True if Drake's roll/pitch swap to match the paper's labels
    get_target: callable      # function(controller) -> scalar, the target hip signal for this step
    target_column: str        # what to call this signal in the saved CSV
    default_duration: float


ROBOTS = {
    "zippy": RobotConfig(
        name="zippy",
        contact_body_left="left_leg",
        contact_body_right="right_leg",
        base_body="left_leg",
        needs_x_rotation=True,
        # After the -90 deg X rotation Drake's Roll_deg tracks forward-rocking
        # (stride frequency) and Pitch_deg tracks sideways-tipping (half stride
        # frequency). The paper calls them the other way round, so for paper
        # comparisons we swap.
        paper_swaps_roll_pitch=True,
        get_target=lambda c: c.ang_vel_input,
        target_column="hip_target_angular_velocity",
        default_duration=7.0,
    ),
    "mugatu": RobotConfig(
        name="mugatu",
        contact_body_left="left_foot",
        contact_body_right="right_foot",
        base_body="right_foot",
        needs_x_rotation=True,
        paper_swaps_roll_pitch=True,
        get_target=lambda c: float(c.target_state[7]),
        target_column="hip_target_angle",
        default_duration=30.0,
    ),
    "scaled_mugatu": RobotConfig(
        name="scaled_mugatu",
        contact_body_left="left_leg",
        contact_body_right="right_leg",
        base_body="left_leg",
        needs_x_rotation=False,   # URDF already imports upright; SetFreeBodyPose fails here
        # No X rotation -- Drake's Roll/Pitch already match the paper's labels.
        paper_swaps_roll_pitch=False,
        get_target=lambda c: c.ang_vel_input,
        target_column="hip_target_angular_velocity",
        default_duration=7.0,
    ),
}


LAWS = {
    "L2": ("ml2_scaling", "mass scales as L^2 (allometric)"),
    "L3": ("ml3_scaling", "mass scales as L^3 (isometric)"),
}
