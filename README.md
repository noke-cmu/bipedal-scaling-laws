# Bipedal Robot Scaling Simulations

This repository lets you **simulate small walking robots at many different
sizes** and see how their behaviour changes as they grow or shrink. It is the
simulation code behind the paper:

> **Allometric Scaling Laws for Bipedal Robots**
> Naomi Oke, Aja M. Carter, Ben Gu, Cordelia Pride, Steven Man,
> Sarah Bergbreiter, and Aaron M. Johnson (Carnegie Mellon University).
> arXiv:[2603.22560](https://arxiv.org/abs/2603.22560).

You do **not** need to know anything about these robots or the simulator to use
it. Install three things, run one command, and watch a robot walk.

Minor note: This is a refactored script from my original code that I tried to 
clean up for others to use. Please let me know if you have any bugs.
---

## Quick start

```bash
git clone https://github.com//bipedal-scaling-laws.git
cd bipedal-scaling-laws
```

Create an isolated environment using either a Python venv or a conda env.

**Option A: Python venv:**

```bash
python3 -m venv scaling-env
source scaling-env/bin/activate            # Windows: scaling-env\Scripts\activate
```

**Option B: conda:**

```bash
conda create -n scaling-env python=3.11 -y
conda activate scaling-env
```

Then, in either case, install Drake and the project's dependencies:

```bash
pip install drake                  # the robotics simulator (provides pydrake)
pip install -r requirements.txt    # numpy, pandas, matplotlib, etc.

python verify_setup.py             # confirms install (~10 s, six [ok] lines)
python run.py --robot zippy --law L2 --simulate    # watch Zippy walk
```

A MeshCat link prints in the terminal. Open it in your browser to watch the
3-D viewer.

## What this simulates

The tiny "passive-style" walkers: each has a single motor at the hip and curved
feet, and rocks forward almost like a wind-up toy. You can scale either of them
up or down (the `--scale` number is the leg-length multiplier) to study how
speed, torque, and stability change with size.

| Robot | Size | Feet | Control |
|-------|------|------|---------|
| **Zippy (Ellipsoidal Feet)** | ~2.5 cm legs | Ellipsoidal (egg-shaped) | Bang-bang torque with mechanical end-stops. |
| **Mugatu** | ~15 cm legs | Spherical (round) | Smooth sinusoidal control at the hip. |
| **Zippy (Spherical Feet aka Scaled-Mugatu)** | Zippy-sized | Spherical, scaled-down | Zippy's body with Mugatu-style round feet -- a control case for foot shape. |

When you scale a robot up, how much heavier should it get? There are two rules
you can pick from, each separately tuned:

- **L^2**: mass grows with leg length *squared* (m ∝ L²). Real bipedal robots actually do this.
- **L^3**: mass grows with leg length *cubed* (m ∝ L³). Volumetric scaling derive from isometric scaling.
  It is also seen in biology. It's kept here for comparison purposes. 

The two laws use different motor-torque tuning because a bigger robot needs
much more torque under L3 than under L2. These two scaling laws live in two separate folders
(`ml2_scaling/` and `ml3_scaling/`).

---

## `run.py` : used to run the sim

```bash
python run.py --list          # show robots and laws
python run.py --help          # full option list
```

Common things to do:

```bash
# Open the interactive viewer (no walking, visualie body/URDF, drag the joints around).
python run.py --robot zippy --law L2

# Simulate walking and save data + plots (needs --duration of at least 6 s).
python run.py --robot zippy --law L2 --simulate --duration 25 --save-data

# Example: A 2x-scaled Mugatu under the L3 law.
python run.py --robot mugatu --law L3 --scale 2 --simulate

# Sweep several sizes at once.
python run.py --robot zippy --law L2 --scales 1 6.12 40 --simulate --save-data
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--robot {zippy,mugatu,scaled_mugatu}` | — | Which walker (required). |
| `--law {L2,L3}` | `L2` | Mass-scaling law. |
| `--simulate` | off | Run the walking simulation. Omit it to get the joint-slider viewer. |
| `--scale <x>` | `1.0` | One size to run (leg-length multiplier). |
| `--scales <x ...>` | — | Several sizes to run in sequence. |
| `--duration <s>` | per-robot | How long to simulate, in seconds. |
| `--save-data` | off | Save CSVs and plots under the robot's `saved_sim_data/`. |
| `--ground-friction <u>` | `0.9` | Friction with the ground. |
| `--feet-friction <u>` | `0.9` | Friction at the feet. |

Saved runs land in `<law-folder>/<robot>/saved_sim_data/<robot>_sc<scale>_<timestamp>/`
as a folder with the data CSV, the plots, and a MeshCat recording you can replay.

---

## What's in the repository

```
bipedal-scaling-laws/
├── run.py                  ← start here: launches any simulation
├── verify_setup.py         ← quick "is my install OK?" check
├── requirements.txt
├── walker_sim/             ← the shared simulation framework
│   ├── runner.py           ←   builds + runs the simulation, saves output
│   ├── plotting.py         ←   creates all the figures
│   ├── analysis.py         ←   numerical post-processing (math only, no plotting)
│   └── robots.py           ←   per-robot config (contact bodies, etc.)
├── meshes/                 ← 3-D robot parts (.obj), shared by both laws
│   ├── zippy/  mugatu/  scaled_mugatu/
├── data/                   ← measured motor data used by the models
├── ml2_scaling/            ← the L2 (m ∝ L²) physics
│   ├── zippy/  mugatu/  scaled_mugatu/
│   │   ├── model_definition.py   ← URDF, plant setup, controller
│   │   └── analysis.py           ← paper-specific amplitude analysis
└── ml3_scaling/            ← the L3 (m ∝ L³) physics, same shape as ml2
```

The two `*_scaling` folders are deliberate near-copies: the robot geometry is
the same, but the torque tuning and motor coefficients differ per law. The
shared assets (meshes, motor data) live once at the top level, and **all of
the simulation, plotting, and post-processing code** lives once in
`walker_sim/`. Each `model_definition.py` is self-contained and just defines
the physics for one (robot, law) pair.

If you want to write your own scripts on top of the framework, the public API is:

```python
from walker_sim import simulate, save_run

result = simulate("zippy", "L2", scale=1.0, duration=25, collect_data=True)
save_run(result, output_root="my_runs/")
```

---

## Torques used in paper and corresponding scales

Both Zippy and Scaled-Mugatu use a bang-bang torque controller, so the
walker only walks if you supply a torque ceiling that is large enough to
drive the gait but not so large that it tips the robot over. The paper
manually tuned this ceiling per scale; those values ship with the code,
inside each model's `get_hip_torque(scale)` function in
`<law>/<robot>/model_definition.py`.

`get_hip_torque(scale)` does three things, in order:

1. **Exact match.** If `scale` matches a tuned point within `1e-5`, you
   get that point's tuned value verbatim.
2. **Inside the tuned range.** Otherwise, return `np.interp` between the
   nearest two tuned points.
3. **Outside the tuned range.** Fall back to the fitted power law
   `POWER_C * scale ** POWER_ALPHA`. The fitted constants are:

   | Robot          | Law | `POWER_C` | `POWER_ALPHA` |
   |----------------|-----|-----------|---------------|
   | Zippy          | L2  | 0.00409   | 2.92          |
   | Zippy          | L3  | 0.00507   | 3.90          |
   | Scaled-Mugatu  | L2  | 0.00191   | 2.93          |
   | Scaled-Mugatu  | L3  | 0.00208   | 3.96          |

Mugatu has no torque table because it uses sinusoidal position control instead
of a torque ceiling, so the torque emerges from the dynamics rather than
being tuned. For paper-comparison purposes, `paper_analysis`
extracts Mugatu's torque from peak `hip_real_torque` during the stable
portion of each saved run.

The abbreviated table below shows the paper's reference scales plus
a sampling of the rest of the curve. Read the full arrays directly out of
`get_hip_torque()` which can be found in `model_definition.py` if you need them.

### Zippy (Ellipsoidal Feet)

`L₁` = 24.7 mm at scale 1.0

| Scale     | Leg length [mm] | Torque [Nm], L²  | Torque [Nm], L³  |
|-----------|-----------------|------------------|------------------|
| 0.9       | 22.2            | 0.003            | 0.003            |
| 1         | 24.7            | 0.005            | 0.005            |
| 1.03      | 25.4            | 0.008            | 0.007            |
| 1.178     | 29.1            | 0.009            | 0.01             |
| 1.348     | 33.3            | 0.01             | 0.02             |
| 1.542     | 38.1            | 0.012            | 0.03             |
| 1.764     | 43.6            | 0.019            | 0.05             |
| 2.018     | 49.8            | 0.028            | 0.08             |
| 2.308     | 57.0            | 0.044            | 0.11             |
| 2.64      | 65.2            | 0.066            | 0.2              |
| 3.021     | 74.6            | 0.082            | 0.3              |
| 3.456     | 85.4            | 0.1              | 0.6              |
| 3.953     | 97.6            | 0.2              | 1                |
| 4         | 98.8            | 0.3              | 1.3              |
| 4.523     | 111.7           | 0.4              | 2                |
| 5.174     | 127.8           | 0.5              | 3                |
| 5.919     | 146.2           | 0.65             | 4.8              |
| 6         | 148.2           | 0.7              | 5.8              |
| 6.12      | 151.2           | 0.72             | 6                |
| 6.194     | 153.0           | 0.8              | 7                |
| 6.772     | 167.3           | 1                | 8                |
| 7.747     | 191.3           | 1.5              | 13               |
| 8         | 197.6           | 1.6              | 15               |
| 8.862     | 218.9           | 2.3              | 25               |
| 10.14     | 250.4           | 3.3              | 40               |
| 11        | 271.7           | 4.2              | 50               |
| 11.6      | 286.5           | 5                | 75               |
| 13.27     | 327.7           | 7.4              | 140              |
| 15        | 370.5           | 11               | 195              |
| 15.18     | 374.9           | 13               | 200              |
| 17.37     | 428.9           | 22               | 390              |
| 19.87     | 490.7           | 26               | 770              |
| 22.73     | 561.4           | 40               | 985              |
| 26        | 642.2           | 55               | 1800             |
| 29.75     | 734.7           | 88               | 2800             |
| 30        | 741.0           | 91               | 3100             |
| 34.03     | 840.5           | 130              | 5000             |
| 38.93     | 961.6           | 197              | 7600             |
| 40        | 988.0           | 205              | 8500             |
| 40.49     | 1000.0          | 210              | 9000             |
| 41        | 1012.7          | 220              | 11000            |
| 44.54     | 1100.0          | 300              | 14000            |

### Zippy (Spherical Feet) aka Scaled-Mugatu

`L₁` = 24.3 mm at scale 1.0. 

| Scale     | Leg length [mm] | Torque [Nm], L²  | Torque [Nm], L³  |
|-----------|-----------------|------------------|------------------|
| 0.9       | 21.9            | 0.0014           | 0.0023           |
| 1         | 24.3            | 0.0025           | 0.0025           |
| 1.014     | 24.6            | 0.0025           | 0.0026           |
| 1.03      | 25.0            | 0.0025           | 0.0027           |
| 1.179     | 28.7            | 0.0028           | 0.0034           |
| 1.349     | 32.8            | 0.004            | 0.0055           |
| 1.545     | 37.5            | 0.0065           | 0.011            |
| 1.768     | 43.0            | 0.0092           | 0.015            |
| 2.023     | 49.2            | 0.015            | 0.03             |
| 2.316     | 56.3            | 0.021            | 0.048            |
| 2.651     | 64.4            | 0.033            | 0.09             |
| 3.034     | 73.7            | 0.045            | 0.15             |
| 3.473     | 84.4            | 0.069            | 0.27             |
| 3.975     | 96.6            | 0.095            | 0.41             |
| 4.549     | 110.6           | 0.17             | 0.77             |
| 5.207     | 126.5           | 0.23             | 1.29             |
| 5.96      | 144.8           | 0.33             | 2.3              |
| 6         | 145.8           | 0.34             | 3                |
| 6.283     | 152.7           | 0.39             | 4                |
| 6.822     | 165.8           | 0.55             | 4.2              |
| 7.808     | 189.7           | 0.77             | 6.7              |
| 8.937     | 217.2           | 1                | 11.5             |
| 10.23     | 248.6           | 1.9              | 25               |
| 11.71     | 284.5           | 3                | 38               |
| 13.4      | 325.6           | 3.7              | 60               |
| 15.34     | 372.7           | 5.2              | 100              |
| 17.55     | 426.6           | 7.5              | 164              |
| 20.09     | 488.3           | 15               | 290              |
| 23        | 558.8           | 25               | 545              |
| 26.32     | 639.6           | 28               | 880              |
| 30.13     | 732.1           | 41               | 1500             |
| 34.48     | 837.9           | 54               | 3400             |
| 39.47     | 959.1           | 90               | 4180             |
| 40        | 972.0           | 95               | 4400             |
| 40.49     | 983.8           | 99               | 4650             |
| 41.07     | 998.0           | 102              | 5000             |
| 42        | 1020.6          | 108              | 5500             |
| 45.17     | 1097.8          | 148              | 7200             |

### Mugatu

`L₁` = 153.0 mm at scale 1.0. 

| Scale     | Leg length [mm] | 
|-----------|-----------------|
| 0.15      | 22.9            | 
| 0.1614    | 24.7            | 
| 0.163     | 24.9            | 
| 0.167     | 25.6            | 
| 0.1717    | 26.3            | 
| 0.1965    | 30.1            | 
| 0.2248    | 34.4            |
| 0.2573    | 39.4            | 
| 0.2945    | 45.1            | 
| 0.337     | 51.6            | 
| 0.3856    | 59.0            | 
| 0.4413    | 67.5            | 
| 0.5051    | 77.3            | 
| 0.578     | 88.4            | 
| 0.6615    | 101.2           | 
| 0.757     | 115.8           | 
| 0.8664    | 132.6           | 
| 0.9915    | 151.7           |
| 1         | 153.0           | 
| 1.135     | 173.6           | 
| 1.299     | 198.7           | 
| 1.486     | 227.4           | 
| 1.701     | 260.2           |
| 1.946     | 297.8           | 
| 2.227     | 340.8           |
| 2.549     | 390.0           | 
| 2.917     | 446.3           | 
| 3.338     | 510.8           |
| 3.821     | 584.6           | 
| 4.372     | 669.0           | 
| 5.004     | 765.6           | 
| 5.727     | 876.2           | 
| 6.536     | 1000.0          | 
| 6.554     | 1002.7          | 
| 6.667     | 1020.1          | 
| 7.5       | 1147.5          | 
---

## Reproducing the paper's analysis (Table IV)

`walker_sim.paper_analysis` reads every saved run under
`<law>/<robot>/saved_sim_data/` and reproduces Table IV: per-scale
body-attitude amplitudes, walking velocity, and minimum required torque,
plus power-law fits across scales.

```bash
# 1. Generate runs at the paper's three reference scales (or your own set).
python run.py --robot zippy --law L2 --scales 1 6.194 40.486 --simulate \
    --duration 25 --save-data

# 2. Run the analysis -- prints a side-by-side comparison with the paper's
#    Table IV, and writes CSVs to analysis_out/.
python -m walker_sim.paper_analysis --robot zippy --law L2
```

The paper analysis script looks at:

* **Amplitudes** from the mean-of-peaks minus mean-of-valleys of the
  stable portion of each gait (first 5 s of settling discarded), halved. This is
  the same peak-to-valley measure the paper uses.
* **Velocity** is ground-plane displacement divided by elapsed time over
  the stable portion.
* **Minimum torque** uses the manually-tuned values from each model's
  `get_hip_torque(scale)` table (Zippy and Scaled-Mugatu). Inside the
  tabulated scale range the script returns the tuned value exactly when
  the scale matches and linearly interpolates between adjacent tuned
  points otherwise; outside the tabulated range it extrapolates with
  `POWER_C * scale ** POWER_ALPHA`. For Mugatu, which uses sinusoidal
  position control instead of a torque table, the minimum torque is
  measured from the recorded `hip_real_torque` column.
* **Fits** are least-squares power laws on log10(L) vs log10(metric),
  reporting alpha and R^2.

By default the analysis flags trials where the robot did not travel at least
three body lengths during the stable portion of the run, and excludes them
from the cross-scale fits. The output lists which trials were excluded and
why, so you know which ones to re-run with a longer `--duration`. A robot
that fell or never started walking is not part of the paper's "successful
trials".

---

## Troubleshooting

- **`--save-data` crashes with an empty-array error.** Your `--duration` is too
  short. The analysis throws away the first 5 s (settling) before measuring,
  so use `--duration 6` or more. `run.py` will stop early with a clear message
  if you ask for less.
- **`pip install drake` complains about numpy.** Current Drake needs numpy ≥ 2;
  don't pin numpy below 2. A fresh virtual environment avoids conflicts.
- **No browser / running on a server.** Everything runs headless: plots are
  written straight to files (the code selects a non-interactive backend
  automatically), and the MeshCat link simply won't have anyone viewing it.

---

## Citation and license

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)
for the full entry). Released under the MIT [`LICENSE`](LICENSE). This work was
supported in part by the National Science Foundation under Grant CMMI-2408884.
