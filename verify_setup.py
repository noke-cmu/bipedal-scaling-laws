#!/usr/bin/env python3
"""verify_setup.py - a 10-second check that your installation works.

Run this right after installing Drake and the requirements:

    python verify_setup.py

It builds each robot model under both scaling laws and confirms the masses
scale the way they should (L^2 vs L^3). It does NOT open a viewer or run a
full walking trial. It just proves that Drake, the meshes, and the robot
models all load correctly. If this passes, `python run.py` shhhoulllddd work.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    try:
        import pydrake  # noqa: F401
    except ImportError:
        print("FAIL: pydrake is not installed. Run `pip install drake` first.")
        return 1

    from walker_sim import ROBOTS, LAWS, load_robot

    print("Checking that each robot builds and scales correctly...\n")
    all_ok = True
    for law, (_, _) in LAWS.items():
        exponent = {"L2": 2, "L3": 3}[law]
        for robot in ROBOTS:
            try:
                model = load_robot(robot, law)

                def total_mass(scale):
                    plant, _, builder, instance = model.setup_walker_plant(
                        scale=scale, ground_friction=0.9, feet_friction=0.9, timestep=0.001)
                    diagram = builder.Build()
                    ctx = diagram.CreateDefaultContext()
                    return plant.CalcTotalMass(plant.GetMyContextFromRoot(ctx), [instance])

                m1, m2 = total_mass(1.0), total_mass(2.0)
                ratio, expected = m2 / m1, 2 ** exponent
                ok = abs(ratio - expected) < 0.05
                all_ok &= ok
                status = "ok " if ok else "FAIL"
                print(f"  [{status}] {robot:14s} {law}: {m1 * 1000:6.1f} g at 1x, "
                      f"x{ratio:.1f} at 2x (expected x{expected} for m proportional to L^{exponent})")
            except Exception as exc:
                all_ok = False
                print(f"  [FAIL] {robot:14s} {law}: {type(exc).__name__}: {exc}")

    print()
    if all_ok:
        print("Everything works. Try:  python run.py --robot zippy --law L2 --simulate")
        return 0
    print("Something went wrong above. Check that Drake installed cleanly and that "
          "the meshes/ folder is present.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
