"""Simulation framework for the bipedal scaling walkers.

This package contains one runner, one plotting module, two analysis
modules, and a robot-config table that work for all three robots
(Zippy, Mugatu, Scaled-Mugatu) under both scaling laws (L^2, L^3). The
per-robot physics lives in each robot's `model_definition.py`
under `ml2_scaling/` or `ml3_scaling/`.

Modules
-------
runner          : build + run a simulation, save its CSV + plots
plotting        : per-run figures
analysis        : numerical post-processing of a single run
paper_analysis  : reproduces scaling fits across many runs like Table IV from the paper
robots          : per-robot config (contact bodies and controller field defs and mapping)
"""
from .robots import ROBOTS, LAWS
from .runner import run, simulate, save_run, show_joint_sliders, load_robot, RunResult

__all__ = ["ROBOTS", "LAWS", "run", "simulate", "save_run",
           "show_joint_sliders", "load_robot", "RunResult"]
