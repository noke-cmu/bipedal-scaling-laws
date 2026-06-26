#!/usr/bin/env python3
"""run.py - one entry point for every simulation in this repository.

Examples
--------
    # See what's available.
    python run.py --list

    # Watch Zippy walk at its real size (the m proportional to L^2 law).
    python run.py --robot zippy --law L2 --simulate

    # Same, but run for 25 s and save the data + plots.
    python run.py --robot zippy --law L2 --simulate --duration 25 --save-data

    # A 2x-scaled Mugatu under the m proportional to L^3 law.
    python run.py --robot mugatu --law L3 --scale 2 --simulate

    # Sweep several sizes in one go.
    python run.py --robot zippy --law L2 --scales 1 6.12 40 --simulate --save-data

    # Just open the interactive viewer (drag the joints around).
    python run.py --robot zippy --law L2
"""
import argparse
import os
import sys
import textwrap

# Make sure the repo root is importable when run.py is invoked from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from walker_sim import ROBOTS, LAWS, run as run_sim


def print_list():
    print("\nRobots (--robot):")
    descriptions = {
        "zippy":         "Smallest walker; ellipsoidal feet, bang-bang hip torque.",
        "mugatu":        "Larger walker; spherical feet, sinusoidal hip control.",
        "scaled_mugatu": "Zippy body with scaled-down Mugatu (spherical) feet.",
    }
    for name in ROBOTS:
        print(f"  {name:15s} {descriptions.get(name, '')}")
    print("\nMass-scaling laws (--law):")
    for name, (_, desc) in LAWS.items():
        print(f"  {name:15s} {desc}")
    print("\nRun  python run.py --help  for the full list of options.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Launch a bipedal-walker scaling simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__.split("Examples")[1]),
    )
    parser.add_argument("--list", action="store_true",
                        help="List the robots and scaling laws, then exit.")
    parser.add_argument("--robot", choices=list(ROBOTS),
                        help="Which walker to simulate.")
    parser.add_argument("--law", choices=list(LAWS), default="L2",
                        help="Mass-scaling law (default: L2).")

    parser.add_argument("--simulate", action="store_true",
                        help="Run the walking simulation. Without it, you get the "
                             "interactive joint-slider viewer instead.")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--scale", type=float,
                      help="A single size to simulate (leg-length factor, default 1.0).")
    size.add_argument("--scales", type=float, nargs="+",
                      help="Several sizes to simulate in sequence.")

    parser.add_argument("--duration", type=float, default=30,
                        help="Simulation length in seconds (default: per-robot).")
    parser.add_argument("--save-data", action="store_true",
                        help="Save CSVs and plots under saved_sim_data/.")
    parser.add_argument("--ground-friction", type=float, default=0.9,
                        help="Friction coefficient for the ground (default 0.9).")
    parser.add_argument("--feet-friction", type=float, default=0.9,
                        help="Friction coefficient for the feet (default 0.9).")
    args = parser.parse_args()

    if args.list or args.robot is None:
        if args.robot is None and not args.list:
            print("Please choose a --robot (or use --list to see the options).")
        print_list()
        return 0

    return run_sim(
        args.robot, args.law,
        simulate_flag=args.simulate,
        scale=args.scale, scales=args.scales,
        duration=args.duration,
        save_data=args.save_data,
        ground_friction=args.ground_friction,
        feet_friction=args.feet_friction,
    )


if __name__ == "__main__":
    raise SystemExit(main())
