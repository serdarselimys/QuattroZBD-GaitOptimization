# Quattro ZBD – Gait Optimization

Find stable walking parameters for the **Quattro ZBD** 3D-printed quadruped robot dog in simulation ([PyBullet](https://pybullet.org)), review them visually, and drive the simulated robot with a gamepad or keyboard before putting anything on the real hardware.

The pipeline has three steps, one script each:

```
 ┌──────────────────┐     opt_results.csv     ┌────────────────────┐    control_params.csv    ┌──────────────────┐
 │ 1. quattro_opt   │ ──────────────────────▶ │ 2. quattro_playback│ ──── (pick the best) ──▶ │ 3. quattro_control│
 │ search with      │   firmware_rows.txt     │ replay every result│                          │ drive the robot   │
 │ Optuna           │   runs/ (TensorBoard)   │ in the 3D window   │                          │ in simulation     │
 └──────────────────┘                         └────────────────────┘                          └──────────────────┘
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `quattro_opt.py` | Searches for the best `step_amplitude` and `frequency` (and body sway for straight walking) for every direction you pick: straight, sideways, diagonal and spin. Writes `opt_results.csv`. |
| 2 | `quattro_playback.py` | Reads `opt_results.csv` and replays each result in the PyBullet window so you can judge the walk with your own eyes and compare it with its score. |
| 3 | `quattro_control.py` | Loads the gaits you chose and lets you drive the simulated robot with a gamepad or the keyboard, like the real robot. |

## Repository contents

```
QuattroZBD-GaitOptimization/
├── quattro_opt.py          # step 1 – optimization
├── quattro_playback.py     # step 2 – replay
├── quattro_control.py      # step 3 – gamepad / keyboard control
├── quattro_zbd.urdf        # robot model
├── meshes/                 # 3D meshes used by the URDF (keep next to the .urdf)
├── examples/               # example output files (see below)
├── requirements.txt
└── README.md
```

> Always run the scripts **from the repository folder**, because they read and write their files (`opt_results.csv`, `control_params.csv`, …) in the current directory.

---

## 1. Install

### 1.1 Install Python

You need **Python 3.10 or 3.11** (recommended: most widely tested with PyBullet). Newer versions may work, but if no ready-made PyBullet package exists for your version, `pip` will try to compile it and you need a C++ compiler (see [Troubleshooting](#troubleshooting)).

<details>
<summary><b>Windows</b></summary>

1. Download the installer from <https://www.python.org/downloads/windows/> (Python 3.11).
2. In the installer, **tick “Add python.exe to PATH”**, then click *Install Now*.
3. Open *PowerShell* and check: `python --version`
</details>

<details>
<summary><b>macOS</b></summary>

With [Homebrew](https://brew.sh): `brew install python@3.11`
or use the installer from <https://www.python.org/downloads/macos/>.
Check: `python3 --version`
</details>

<details>
<summary><b>Linux (Debian / Ubuntu)</b></summary>

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk git
python3 --version
```
`python3-tk` is only used to size the simulation window to your screen.
</details>

### 1.2 Install Git and clone the repository

Install Git from <https://git-scm.com/downloads> (Linux: `sudo apt install git`, macOS: `brew install git`), then:

```bash
git clone https://github.com/serdarselimys/QuattroZBD-GaitOptimization.git
cd QuattroZBD-GaitOptimization
```

To get later updates, run `git pull` inside the folder.

### 1.3 Create a virtual environment and install the requirements

A virtual environment keeps this project's packages separate from the rest of your system.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use *Command Prompt* and `.venv\Scripts\activate.bat`.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Your prompt shows `(.venv)` while the environment is active. Activate it again in every new terminal before running the scripts.

Quick check:
```bash
python -c "import pybullet, optuna, pandas, pygame, tensorboardX; print('all good')"
```

What the requirements are for: `pybullet` (physics and 3D window), `optuna` + `cmaes` (the optimizer), `pandas`/`numpy`/`tqdm` (data and progress bars), `pygame` (gamepad input), `tensorboard` + `tensorboardX` (live charts of the optimization).

---

## 2. Step 1 – Optimize gaits (`quattro_opt.py`)

All settings are at the top of the file (section *1. SETTINGS & GLOBALS*), each with a short comment. The ones you will change most:

| Setting | What it does |
|---------|--------------|
| `RENDER_MODE` | `"windowed"` or `"headless"`. See [below](#windowed-vs-headless). |
| `MODES` | Which movements to optimize, run in order: `"straight"`, `"sideway"`, `"diagonal"`, `"spin"`. E.g. `["straight"]` or `["straight", "spin"]`. |
| `DIRECTIONS` | `+1` / `-1` for the opposite directions of straight (forward/back), sideway (left/right) and spin (CCW/CW). Use `[1, -1]` to do both. |
| `DIAGONAL_DIRECTIONS` | Heading in degrees from forward for the diagonal mode (`45`, `135`, `225`, `315`). |
| `TARGET_SPEEDS` | The speed the score rewards, per mode (m/s; spin in rad/s). **One separate experiment runs for every value you list**, giving you one parameter set per speed. |
| `BODY_HEIGHTS`, `STEP_HEIGHTS` | Standing height and foot-lift height to test. Every combination is a separate experiment. |
| `GENE_BOUNDS` | The search range for `step_amplitude` (half the step length), `frequency` and, for straight, `sway_amp`. If the best result sits on the edge of a range, widen it. |
| `GAITS` | Leg timing per mode: `"trot"`, `"walk"` (straight only makes sense) or `"diag_trot"` (diagonal). |
| `NUM_ROBOTS` | Robots (candidate gaits) simulated side by side per experiment, headless mode. |
| `GAIT_TRIALS` | Number of generations. Each generation tests `NUM_ROBOTS` new candidates, so an experiment tests `GAIT_TRIALS × NUM_ROBOTS` candidates. |
| `DURATION` | Seconds each candidate walks before it is scored. Longer = more accurate but slower. |
| `NUM_EXPERIMENTS` | How many times each setting is repeated from scratch (headless only). More repeats = more confidence the result is not luck. |
| `SAMPLER_MODE` | `"cmaes"` (evolves a population, default) or `"tpe"`. |
| `LATERAL_FRICTION` | Floor/foot friction. Lower = more slipping. |

Run it:
```bash
python quattro_opt.py
```

### Windowed vs headless

| | `RENDER_MODE = "windowed"` | `RENDER_MODE = "headless"` |
|---|---|---|
| What you see | A PyBullet window with the robots walking, running in real time | No window, a progress bar in the terminal |
| Speed | Real time only: about `GAIT_TRIALS × (DURATION + 0.5)` seconds per experiment (~1 min 45 s with the defaults) | As fast as your CPU allows; experiments run **in parallel**, one per CPU core (minus one) |
| Repeats | Each setting runs **once**; `NUM_EXPERIMENTS` is ignored | Each setting runs `NUM_EXPERIMENTS` times |
| Robots in the scene | 10 | `NUM_ROBOTS` per experiment |
| Experiments | One after another; the camera view you leave carries over to the next window | Spread over all workers |
| Use it for | Checking that a setup works, watching how gaits look, debugging | The real optimization runs |

**Recommended workflow:** first try a small run in `windowed` mode (one mode, one speed, `GAIT_TRIALS = 5`) to confirm everything works and the robots walk sensibly, then switch to `headless` with several `NUM_EXPERIMENTS` for the real search.

### How many experiments will run?

```
experiments = modes × gaits × target speeds × body heights × step heights × directions × repeats
```
(`repeats` = `NUM_EXPERIMENTS` in headless, 1 in windowed.) The script prints the total before it starts. For example `MODES = ["straight", "spin"]`, one gait each, one speed each, one height, one step height, `DIRECTIONS = [1]` and `NUM_EXPERIMENTS = 5` gives 2 × 1 × 1 × 1 × 1 × 1 × 5 = **10 experiments**. Adding two more speeds to each mode triples that, so grow the lists gradually.

### Single run vs multiple experiments

* **1 experiment (quick test):** `RENDER_MODE = "windowed"`, or headless with `NUM_EXPERIMENTS = 1`.
* **Multiple experiments (reliable result):** headless with `NUM_EXPERIMENTS = 5` or more. Every repeat starts from scratch; the best-scoring repeat is the one that ends up in your control table, and the spread between repeats shows how consistent the result is.

### Outputs

| File | Content |
|------|---------|
| `opt_results.csv` | One row per experiment: the best parameters found, plus context and quality metrics. |
| `firmware_rows.txt` | The best row per (name, speed) formatted like the firmware's `quadGaitMatrix` table: `{ "trot", speed, amplitude, frequency, sway, duty },` |
| `runs/` | TensorBoard logs. |

Main columns of `opt_results.csv`: `firmware_name` (trot / walk / sideway / diagonal / spin), `target_vel`, `step_amplitude`, `frequency`, `sway_amp`, `duty` (these six are what the firmware table needs), then `mode`, `exp_id`, `gait`, `target_height`, `step_height`, `direction`, `score` and metrics such as `distance_traveled`, `avg_speed`, `lateral_deviation`, `yaw_quality`, `stability_quality`, `swing_drag_frac` for manual review. `step_amplitude` is **half** the step length (for spin: half the sweep angle in radians).

### Watching the optimization in TensorBoard

In a second terminal (with the virtual environment active):
```bash
tensorboard --logdir runs
```
then open <http://localhost:6006>. Each setting appears as **one run**, even when its repeats ran in parallel; the charts combine all repeats and robots:

* `score/best_in_generation`, `score/mean_in_generation` – how the population improves.
* `score/all_time_best` – best score found so far in any repeat.
* `score/all_time_best_worst_repeat` – best score of the weakest repeat; if it is close to `all_time_best` the repeats agree.
* `best_dna/*` and `best_metrics/*` – parameters and metrics of the best candidate.

A chart point appears once every repeat has finished that generation.

---

## 3. Step 2 – Replay and review (`quattro_playback.py`)

Replays the results of `opt_results.csv` one robot at a time in a PyBullet window. It imports `quattro_opt.py`, so the robot, gait and scoring code are identical to the ones used for the optimization. For every row it prints the parameters, the distance / speed / drift achieved, the replayed score next to the CSV score, and how hard the joints were driven (peak speed and torque against their limits).

```bash
python quattro_playback.py                        # every row in the CSV, one by one
python quattro_playback.py --best                 # the 3 best rows of each mode
python quattro_playback.py --mode straight spin   # only these modes
python quattro_playback.py --gait walk --top 3    # the 3 best walk rows overall
python quattro_playback.py --speed 0.5 --follow   # half speed, camera follows the robot
python quattro_playback.py --csv examples/opt_results.csv
```

| Flag | Meaning |
|------|---------|
| `--csv FILE` | Result file (default `opt_results.csv`). |
| `--mode ...` | Only these modes (`straight`, `sideway`, `diagonal`, `spin`). |
| `--gait NAME` | Only this gait (`trot`, `walk`, `diag_trot`). |
| `--best` | Only the 3 best-scoring rows of each mode (change `BEST_PER_MODE` in the script). |
| `--top N` | Only the N best rows overall. |
| `--speed X` | Playback speed: `1` real time, `0.5` half speed, `0` as fast as possible. |
| `--follow` | Camera follows the robot. |

Notes:
* Press `Ctrl+C` in the terminal to stop.
* The replayed score can differ a little from the CSV score: the optimizer ran many robots at once, the replay runs one robot, and it uses the optimizer's *current* settings. A big difference usually means you changed the optimizer after producing the CSV.
* Pick the rows that look good **and** have a low `swing drag`, little lateral drift and peak velocity/torque comfortably under the limits.

---

## 4. Step 3 – Drive it (`quattro_control.py`)

### 4.1 Choose the gaits to load

The control script reads `control_params.csv` (set by `FILE_PATH` at the top of the script). Build it by copying the rows you liked from `opt_results.csv`, one or more speeds per gait. Rows are matched by name (`firmware_name`/`gait`), so you want entries for `trot`, `walk`, `sideway`, `diagonal` and `spin`. When a stick is pushed further, the script moves to the row with the next higher `target_vel`, so several speeds per gait give you a speed range.

Two shortcuts:
* Set `FILE_PATH = "opt_results.csv"`: the script then automatically picks the best-scoring row for each gait, speed and direction.
* Try it without running anything first: copy the sample from `examples/` (`cp examples/control_params.csv .`, on Windows `copy examples\control_params.csv .`).

If no file is found, the script falls back to a few built-in rows.

### 4.2 Run

```bash
python quattro_control.py
```
The robot starts **sitting**. Press `S` or the gamepad **Start** button to stand up.

### 4.3 Controls

| Action | Gamepad | Keyboard (PyBullet window must be focused) |
|--------|---------|------------------------------------------|
| Walk / strafe | Left stick (any direction, diagonals blend smoothly) | Arrow keys (`Shift` = full speed) |
| Spin | Right stick left/right | `Q` / `E` |
| Stand / sit | Start (or hold L1 + R1 for 2 s) | `S` |
| Body height − / + | L1 / R1 | `F` / `R` |
| Pitch trim − / + | L2 / R2 | `Z` / `X` |
| Toggle trot ↔ walk for forward/back | A | `W` |
| Idle “breathing” motion on/off | – | `I` |
| Auto-sit after inactivity on/off | – | `A` |
| Gamepad debug output / backend switch | – | `G` / `B` |

Keyboard and gamepad can be used at the same time. Keyboard speed is set by `KEYBOARD_SPEED` at the top of the script (default `0.5` of full stick; holding `Shift` gives 100 %).

Other settings at the top of the script: `STRAIGHT_GAIT` (trot or walk to start with), `START_STANDING`, `IDLE_ENABLED`, `AUTO_SLEEP_S`, height limits and smoothing times.

### 4.4 Gamepad troubleshooting

Most pads (Xbox, PlayStation, Logitech F310/F710, 8BitDo, …) work automatically and can be plugged in while the script runs. If something is off, start with `python quattro_control.py --gamepad-debug` to see the live values, then try `--backend raw` or `--backend sdl`, or press `B` in the window to switch live. `--pad N` locks to one pad; `--raw-layout xbox_win|xbox_linux|dinput` forces a layout.

---

## Using the results on the real robot

`firmware_rows.txt` contains the best row per (name, speed) in the format of the firmware gait table:

```
{ "trot",     0.150, <amplitude>, <frequency>, <sway>, <duty> },
```
`amplitude` is the half step length in metres (spin: radians), `frequency` is in Hz. Paste the rows into your firmware table. The simulated robot is only a model, so always test a new gait on the real robot slowly and with the robot lifted or held at first.

## Adapting it to another robot

* Replace `quattro_zbd.urdf` and `meshes/` (keep the meshes next to the URDF).
* The scripts expect joints named `FL1 FL2 FL3 … RR1 RR2 RR3` (1 = knee, 2 = thigh, 3 = hip) – rename the joints in the URDF or adapt the code.
* Update the robot block at the top of `quattro_opt.py` (`L1`, `L2`, `URDF_X_OFFSET`, `BODY_OFFSET`, torque and velocity limits) and the sign tables (`THIGH_SIGN`, `KNEE_SIGN`, …) to match your legs.
* `quattro_control.py` has its own copy of these robot constants (section *ROBOT CONSTANTS*); keep it identical to the optimizer, otherwise the robot will not move the way it was tuned.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip install` fails while building `pybullet` | Use Python 3.10/3.11, or install a C++ compiler: Windows – *Microsoft C++ Build Tools* (<https://visualstudio.microsoft.com/visual-cpp-build-tools/>, “Desktop development with C++”); Linux – `sudo apt install build-essential python3-dev`; macOS – `xcode-select --install`. Then rerun `pip install -r requirements.txt`. |
| Errors mentioning NumPy when importing pybullet | `pip install "numpy<2"` |
| `ModuleNotFoundError` | The virtual environment is not active – activate it again (section 1.3). |
| `ModuleNotFoundError: cmaes` | `pip install cmaes`, or set `SAMPLER_MODE = "tpe"`. |
| “TensorBoard logging disabled” | `pip install tensorboard tensorboardX`, or set `ENABLE_TENSORBOARD = False` in `quattro_opt.py`. |
| Window opens small / default size | On Linux install `python3-tk` so the window can be sized to the screen. |
| Robot model not found | Run the script from the repository folder; `quattro_zbd.urdf` and `meshes/` must sit next to the scripts. |
| Gamepad not detected | Plug it in before or while the script runs, check with `--gamepad-debug`; the keyboard works without one. |
| Keys do nothing | Click the PyBullet window first so it has keyboard focus. |
| Replay score differs from the CSV | See the notes in step 2. |
