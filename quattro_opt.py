import pybullet as p
import pybullet_data
import math
import numpy as np
import copy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import os
import queue
import threading
from tqdm import tqdm

PATH = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. SETTINGS & GLOBALS
# ============================================================
# Robot Settings
URDF_FILE = "quattro_zbd.urdf"   # robot model loaded into every simulation
L1 = 0.1000                      # thigh length (m): sets how far the foot can reach in the leg IK
L2 = 0.1000                      # shin length (m): sets how far the foot can reach in the leg IK
URDF_X_OFFSET = -0.010           # fore/aft shift (m) between the hip joint and the foot's neutral position
BODY_OFFSET = 0.0544             # vertical shift (m) between the body-height reference and the thigh axis
GLOBAL_MAX_TORQUE = 1.8          # torque limit per joint (N*m): lower = weaker, slower-reacting legs
GLOBAL_MAX_VELOCITY = 6.0        # speed limit per joint (rad/s): caps how fast a leg can move

# Optimization Environment Settings
NUM_ROBOTS = 10         # robots simulated side by side in headless mode; each tests a different candidate per generation
DURATION = 10           # seconds each candidate walks before it is scored (longer = more accurate, slower)
SPACING = 1.5           # distance (m) between neighbouring robots so they never collide
GAIT_TRIALS = 10        # generations per experiment; candidates tested = GAIT_TRIALS x robots
NUM_EXPERIMENTS = 5    # independent repeats of every setting from scratch (headless only; windowed runs once)
PHYSICS_FREQ = 240      # physics steps per second of simulated time
SERVO_FREQ = 100        # how often (Hz) the leg targets are updated; matches the real servo update rate
LATERAL_FRICTION = 1.0  # floor and foot friction: lower = more slipping
SPINNING_FRICTION = {"diagonal": 0.02}  # foot twisting friction per mode (modes not listed use none): resists the feet pivoting, i.e. yaw slip
RENDER_MODE = "headless"  # "windowed" = watch in the pybullet window, experiments run one at a time; "headless" = no window, experiments run in parallel
SAMPLER_MODE = "cmaes"    # how Optuna proposes new gaits: "cmaes" = evolves a population, "tpe" = Bayesian search (see Optuna docs for others)

# Optimization Ranges
MODES = ["straight", "spin"]            # movement types to optimize, run in order: "straight", "sideway", "diagonal", "spin"
DIRECTIONS = [1]                # +1 / -1 = opposite directions (forward/back, left/right, CCW/CW); skipped modes need a non-empty list
DIAGONAL_DIRECTIONS = [45]      # diagonal heading in degrees from forward (45, 135, 225, 315); used only by "diagonal"

BODY_HEIGHTS = [0.20]           # standing hip-to-foot height (m): lower = crouched; e.g. 0.14, 0.16, 0.18, 0.22
STEP_HEIGHTS = [0.010]          # how high (m) each foot lifts during swing; e.g. 0.015, 0.02

TARGET_SPEEDS = {               # speed the score rewards (m/s, spin in rad/s); one experiment per value
    "straight": [0.15],         # 0.05, 0.075, 0.1, 0.125, 0.15, 0.175
    "sideway":  [0.075],        # 0.05, 0.1
    "diagonal": [0.10],         # 0.05, 0.1
    "spin":     [0.6],          # 0.3, 0.8
}

# Search range (low, high) Optuna may pick from for each mode:
#   step_amplitude : half the step length (m); how far each foot swings fore/aft from neutral. Spin: half sweep angle (rad)
#   frequency      : gait cycles per second (Hz); how fast the legs move
#   sway_amp       : sideways body sway (m) that shifts weight over the stance legs (optional)
# Body speed ~ 2 * step_amplitude * frequency / duty, i.e. 4 * step_amplitude * frequency for the trot.
GENE_BOUNDS = {
    "straight": {"step_amplitude": (0.050, 0.060), "frequency": (1.4, 1.6), "sway_amp": (0.0, 0.01)},
    "sideway":  {"step_amplitude": (0.0125, 0.0175), "frequency": (1.1, 1.6)},
    "diagonal": {"step_amplitude": (0.015, 0.025), "frequency": (1.1, 1.6)},
    "spin":     {"step_amplitude": (0.08, 0.10),   "frequency": (1.4, 1.6)},
}

# Gaits tried for each mode, each run as its own experiment: "trot", "walk", "diag_trot"
GAITS = {
    "straight": ["trot"],       # "walk"
    "sideway":  ["trot"],
    "diagonal": ["diag_trot"],
    "spin":     ["trot"],
}

# Leg order is [FL, RL, FR, RR].
#   offsets : how far each leg's cycle is delayed (0.5 = half a cycle), sets which legs step together
#   duty    : fraction of the cycle each foot stays on the ground; higher = more feet down at once
GAIT_LIBRARY = {
    "trot": {
        "offsets": [0.0, 0.5, 0.5, 0.0],
        "duty": 0.5,
    },
    "walk": {
        "offsets": [0.75, 0.0, 0.25, 0.5],
        "duty": 0.60,
    },
    "diag_trot": {
        "offsets": [0.0, 0.5, 0.5, 0.0],
        "duty": 0.55,
    },
}

# Foot-path shaping, set per mode:
#   SWING_PROFILE      : "quintic" = smooth swing, then constant-speed stance so grounded feet don't scrub the floor
#                        "cycloid" = foot slows to zero at touchdown and lift-off
#   SWING_DRAG_PENALTY : score is multiplied by (1 - penalty x fraction of mid-swing time the foot touches the floor)
SWING_PROFILE = {"straight": "quintic", "sideway": "cycloid", "diagonal": "cycloid", "spin": "cycloid"}
SWING_DRAG_PENALTY = {"straight": 0.8, "sideway": 0.0, "diagonal": 0.0, "spin": 0.0}

# ============================================================
# 2. IK
# ============================================================
LEG_PREFIXES = ["FL", "RL", "FR", "RR"]
THIGH_SIGN = {"FL": -1.0, "RL": -1.0, "FR": 1.0, "RR": 1.0}
KNEE_SIGN = {"FL": 1.0, "RL": 1.0, "FR": 1.0, "RR": -1.0}

def solve_leg_ik_3dof(target_x, target_y, target_z):
    x = target_x + URDF_X_OFFSET
    y = target_y
    z_from_thigh = -(target_z - BODY_OFFSET)
    hip_angle = math.atan2(y, -z_from_thigh)
    z_sagittal = -math.sqrt(y**2 + z_from_thigh**2)

    dist_sq = x**2 + z_sagittal**2
    dist = math.sqrt(dist_sq)

    max_reach = (L1 + L2) * 0.97
    min_reach = abs(L1 - L2) * 1.05
    if dist > max_reach or dist < min_reach:
        scale = (max_reach if dist > max_reach else min_reach) / max(dist, 1e-9)
        x *= scale
        z_sagittal *= scale
        dist_sq = x**2 + z_sagittal**2
        dist = math.sqrt(dist_sq)

    cos_phi = (L1**2 + L2**2 - dist_sq) / (2 * L1 * L2)
    phi = math.acos(np.clip(cos_phi, -1.0, 1.0))
    knee_angle = math.pi - phi

    alpha = math.atan2(z_sagittal, x)
    cos_beta = (L1**2 + dist_sq - L2**2) / (2 * L1 * max(dist, 1e-9))
    beta = math.acos(np.clip(cos_beta, -1.0, 1.0))
    thigh_angle = alpha - beta + (math.pi / 2)

    return hip_angle, thigh_angle, knee_angle


def command_leg(r_id, joint_map, prefix, h, th, kn, client_id, max_vel=None):
    kwargs = dict(force=GLOBAL_MAX_TORQUE, physicsClientId=client_id)
    if max_vel is not None:
        kwargs["maxVelocity"] = max_vel
    p.setJointMotorControl2(r_id, joint_map[f"{prefix}3"], p.POSITION_CONTROL, h, **kwargs)
    p.setJointMotorControl2(r_id, joint_map[f"{prefix}2"], p.POSITION_CONTROL, THIGH_SIGN[prefix] * th, **kwargs)
    p.setJointMotorControl2(r_id, joint_map[f"{prefix}1"], p.POSITION_CONTROL, KNEE_SIGN[prefix] * kn, **kwargs)

# ============================================================
# 3. TRAJECTORY DISPATCH (per-mode foot target)
# ============================================================
LEG_ROOTS = {"FL": [0.12, 0.10], "RL": [-0.12, 0.10],
             "FR": [0.12, -0.10], "RR": [-0.12, -0.10]}
FOOT_X_SIGN = -1.0
FOOT_Y_SIGN = 1.0
LIFT_SIGN = {"straight": 1.0, "sideway": -1.0, "diagonal": -1.0, "spin": -1.0}
HIP_SIDE_SIGN = {"L": 1.0, "R": 1.0}


def quintic_ease(s):
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def cycloid_ease(s):
    return s - math.sin(2.0 * math.pi * s) / (2.0 * math.pi)


def gait_curve(phase, duty, profile):
    swing_frac = 1.0 - duty
    if phase < swing_frac:
        s = phase / swing_frac
        if profile == "quintic":
            q = quintic_ease(s)
            return 1.0 - 2.0 * q, 4.0 * q * (1.0 - q), True, s
        return 1.0 - 2.0 * cycloid_ease(s), math.sin(math.pi * s), True, s
    s = (phase - swing_frac) / duty
    if profile == "quintic":
        return -1.0 + 2.0 * s, 0.0, False, s
    return -1.0 + 2.0 * cycloid_ease(s), 0.0, False, s


def foot_target(mode, prefix, phase_multiplier, tz, dna, direction, sway=0.0):
    tx, ty, tz = _foot_path(mode, prefix, phase_multiplier, tz, dna, direction)
    return tx, ty + sway * HIP_SIDE_SIGN[prefix[-1]], tz


def _foot_path(mode, prefix, phase_multiplier, tz, dna, direction):
    amp = dna["step_amplitude"]

    if mode == "straight":
        return FOOT_X_SIGN * direction * amp * phase_multiplier, 0.0, tz

    if mode == "sideway":
        return 0.0, FOOT_Y_SIGN * direction * amp * phase_multiplier, tz

    if mode == "diagonal":
        rad = math.radians(direction)
        tx = amp * math.cos(rad) * phase_multiplier
        ty = FOOT_Y_SIGN * amp * math.sin(rad) * phase_multiplier
        return tx, ty, tz

    if mode == "spin":
        rx, ry = LEG_ROOTS[prefix]
        angle = amp * phase_multiplier
        tx = -ry * angle * direction
        ty = FOOT_Y_SIGN * (rx * angle * direction)
        return tx, ty, tz

    raise ValueError(f"Unknown mode {mode}")

# ============================================================
# 4. SCORING DISPATCH
# ============================================================
def compute_translational_score(mode, start_pos, current_pos, roll, pitch,
                                target_vel, current_orn, dna, direction):
    move_vec = np.array(current_pos[:2]) - np.array(start_pos[:2])

    if mode == "straight":
        fwd = np.array([1.0, 0.0])
        dist_along_dir = np.dot(move_vec, fwd) * direction
        lateral_deviation = abs(move_vec[1])
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.01, 0.005, 0.05, 0.001
    elif mode == "sideway":
        fwd = np.array([0.0, 1.0])
        dist_along_dir = np.dot(move_vec, fwd) * direction
        lateral_deviation = abs(move_vec[0])
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.01, 0.001, 0.05, 0.05
    else:
        rad = math.radians(direction)
        target_dir_vec = np.array([math.cos(rad), math.sin(rad)])
        dist_along_dir = np.dot(move_vec, target_dir_vec)
        lateral_deviation = float(np.linalg.norm(move_vec - dist_along_dir * target_dir_vec))
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.045, 0.0032, 0.01, 0.0128

    drift_penalty = math.exp(-(lateral_deviation ** 2) / drift_sigma_sq)
    actual_vel = dist_along_dir / DURATION
    speed_reward = math.exp(-((target_vel - actual_vel) ** 2) / speed_sigma_sq)

    vertical_deviation = current_pos[2] - start_pos[2]
    stability_error = roll ** 2 + pitch ** 2 + vertical_deviation ** 2
    stability_reward = math.exp(-stability_error / stab_sigma_sq)

    _, _, yaw = p.getEulerFromQuaternion(current_orn)
    yaw_reward = math.exp(-(yaw ** 2) / yaw_sigma_sq)

    score = max(0.0, dist_along_dir)
    score *= (40.0 * speed_reward * stability_reward * yaw_reward * drift_penalty)

    num_steps = dna["frequency"] * DURATION
    metrics = {
        "distance_traveled":   dist_along_dir,
        "avg_speed":           actual_vel,
        "speed_match_quality": speed_reward,
        "stability_quality":   stability_reward,
        "yaw_quality":         yaw_reward,
        "drift_quality":       drift_penalty,
        "lateral_deviation":   lateral_deviation,
        "stride_length":       dist_along_dir / (num_steps + 1e-6),
        "vertical_deviation":  abs(vertical_deviation),
    }
    return score, metrics


def compute_spin_score(total_yaw, current_orn, current_pos, start_xy,
                       target_speed, spin_dir, dna):
    roll, pitch, _ = p.getEulerFromQuaternion(current_orn)

    actual_rot_speed = (total_yaw * spin_dir) / DURATION
    speed_error = abs(target_speed - actual_rot_speed)

    speed_reward = math.exp(-(speed_error ** 2) / 0.002)
    stability_reward = math.exp(-(abs(roll) + abs(pitch)) / 0.15)

    drift = float(np.linalg.norm(np.array(current_pos[:2]) - start_xy))
    drift_reward = math.exp(-drift / 0.05)

    score = max(0.0, total_yaw * spin_dir)
    score *= 40.0 * speed_reward * stability_reward * drift_reward

    num_steps = dna["frequency"] * DURATION
    yaw_per_cycle = (total_yaw * spin_dir) / (num_steps + 1e-6)
    efficiency_reward = math.tanh(yaw_per_cycle / 0.5)
    score *= (1.0 + 0.5 * efficiency_reward)

    if abs(roll) > 0.8 or abs(pitch) > 0.8:
        score *= 0.01

    metrics = {
        "actual_rot_speed":    actual_rot_speed,
        "speed_match_quality": speed_reward,
        "stability_quality":   stability_reward,
        "drift_quality":       drift_reward,
        "yaw_per_cycle":       yaw_per_cycle,
        "efficiency_reward":   efficiency_reward,
    }
    return score, metrics

# ============================================================
# 5. RESET / STANCE HELPERS
# ============================================================
_RENDER_MAP = {"windowed": p.GUI, "headless": p.DIRECT}
_SAMPLER_CHOICES = ("tpe", "cmaes")

assert RENDER_MODE in _RENDER_MAP, f"RENDER_MODE must be one of {list(_RENDER_MAP)}"
PYBULLET_CONNECT_MODE = _RENDER_MAP[RENDER_MODE]
assert SAMPLER_MODE in _SAMPLER_CHOICES, f"SAMPLER_MODE must be one of {_SAMPLER_CHOICES}"

SPAWN_ORI = p.getQuaternionFromEuler([0, 0, 0])
SPAWN_CLEARANCE = 0.005

INITIAL_SETTLE_SEC = 1.0
GEN_SETTLE_SEC = 0.5
START_RAMP_TIME = 0.3

CAMERA_ZOOM = 0.4
_last_camera = None


def set_camera(client_id):
    if _last_camera is not None:
        dist, yaw, pitch, target = _last_camera
    else:
        cam = p.getDebugVisualizerCamera(physicsClientId=client_id)
        dist, yaw, pitch, target = cam[10] * CAMERA_ZOOM, cam[8], cam[9], cam[11]
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target, physicsClientId=client_id)


def save_camera(client_id):
    global _last_camera
    cam = p.getDebugVisualizerCamera(physicsClientId=client_id)
    _last_camera = (cam[10], cam[8], cam[9], cam[11])

ENABLE_TENSORBOARD = True
TENSORBOARD_LOGDIR = "runs"


def gui_options():
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return f"--width={w} --height={h}"
    except Exception:
        pass
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return f"--width={user32.GetSystemMetrics(0)} --height={user32.GetSystemMetrics(1)}"
    except Exception:
        return ""


def make_tb_writer(log_dir):
    if not ENABLE_TENSORBOARD:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter
        except Exception:
            print("TensorBoard logging disabled: install `tensorboard` plus `torch` or `tensorboardX`.")
            return None
    return SummaryWriter(log_dir=log_dir)


class RunLogger:
    def __init__(self, logdir, repeats):
        self.logdir = logdir
        self.repeats = repeats
        self.writers = {}
        self.pending = {}
        self.lock = threading.Lock()

    def put(self, msg):
        with self.lock:
            bucket = self.pending.setdefault((msg["key"], msg["gen"]), {})
            bucket[msg["exp"]] = msg
            if len(bucket) >= self.repeats:
                self._flush(msg["key"], msg["gen"])

    def _flush(self, key, gen):
        msgs = list(self.pending.pop((key, gen)).values())
        if key not in self.writers:
            self.writers[key] = make_tb_writer(os.path.join(self.logdir, key))
        writer = self.writers[key]
        if writer is None:
            return
        all_scores = [sc for m in msgs for sc in m["scores"]]
        top = max(msgs, key=lambda m: m["best_score"])
        writer.add_scalar("score/best_in_generation", max(all_scores), gen)
        writer.add_scalar("score/mean_in_generation", float(np.mean(all_scores)), gen)
        writer.add_scalar("score/all_time_best", top["best_score"], gen)
        writer.add_scalar("score/all_time_best_worst_repeat", min(m["best_score"] for m in msgs), gen)
        for k, v in top["dna"].items():
            writer.add_scalar(f"best_dna/{k}", v, gen)
        for k, v in top["metrics"].items():
            writer.add_scalar(f"best_metrics/{k}", v, gen)
        writer.flush()

    def close(self):
        with self.lock:
            for key, gen in sorted(self.pending, key=lambda kg: kg[1]):
                self._flush(key, gen)
            for writer in self.writers.values():
                if writer is not None:
                    writer.close()


_LOG_SINK = None


def _init_worker(sink):
    global _LOG_SINK
    _LOG_SINK = sink.put if sink is not None else None


def apply_spinning_friction(plane_id, robots, mode, client_id):
    listed = mode in SPINNING_FRICTION
    p.changeDynamics(plane_id, -1, spinningFriction=1.0 if listed else 0.0, physicsClientId=client_id)
    for r_id in robots:
        for j in range(-1, p.getNumJoints(r_id, physicsClientId=client_id)):
            p.changeDynamics(r_id, j, spinningFriction=SPINNING_FRICTION.get(mode, 0.0), physicsClientId=client_id)


def make_optuna_sampler(mode, num_robots):
    import optuna
    if mode == "cmaes":
        return optuna.samplers.CmaEsSampler(popsize=num_robots) if num_robots >= 2 else optuna.samplers.CmaEsSampler()
    return optuna.samplers.TPESampler()


def hard_reset_robots(robots, start_positions, joint_map, body_height, client_id):
    h_set, th_set, kn_set = solve_leg_ik_3dof(0, 0, body_height)
    for i, r_id in enumerate(robots):
        p.resetBasePositionAndOrientation(r_id, start_positions[i], SPAWN_ORI, physicsClientId=client_id)
        p.resetBaseVelocity(r_id, [0, 0, 0], [0, 0, 0], physicsClientId=client_id)
        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            p.resetJointState(r_id, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=client_id)
        for prefix in LEG_PREFIXES:
            p.resetJointState(r_id, joint_map[f"{prefix}3"], h_set, targetVelocity=0.0, physicsClientId=client_id)
            p.resetJointState(r_id, joint_map[f"{prefix}2"], THIGH_SIGN[prefix] * th_set, targetVelocity=0.0, physicsClientId=client_id)
            p.resetJointState(r_id, joint_map[f"{prefix}1"], KNEE_SIGN[prefix] * kn_set, targetVelocity=0.0, physicsClientId=client_id)


def hold_stance(robots, joint_map, body_height, seconds, client_id):
    h_set, th_set, kn_set = solve_leg_ik_3dof(0, 0, body_height)
    for _ in range(int(seconds * PHYSICS_FREQ)):
        for r_id in robots:
            for prefix in LEG_PREFIXES:
                command_leg(r_id, joint_map, prefix, h_set, th_set, kn_set, client_id)
        p.stepSimulation(physicsClientId=client_id)
        if PYBULLET_CONNECT_MODE == p.GUI:
            time.sleep(1.0 / PHYSICS_FREQ)

# ============================================================
# 6. WORKER (OPTUNA INTEGRATED, MULTI-MODE)
# ============================================================
def run_experiment_worker(args):
    import optuna

    mode, gait_name, target_vel, body_height, step_height, direction, exp_idx, run_stamp = args
    np.random.seed()

    num_robots = 10 if PYBULLET_CONNECT_MODE == p.GUI else NUM_ROBOTS
    gait_offsets = GAIT_LIBRARY[gait_name]["offsets"]
    duty = GAIT_LIBRARY[gait_name]["duty"]
    bounds = GENE_BOUNDS[mode]
    swing_profile = SWING_PROFILE[mode]
    drag_penalty = SWING_DRAG_PENALTY[mode]

    study = optuna.create_study(direction="maximize", sampler=make_optuna_sampler(SAMPLER_MODE, num_robots))

    run_key = (f"{run_stamp}_{SAMPLER_MODE}/{mode}_{gait_name}_dir{direction}_spd{target_vel}"
               f"_h{body_height}_sh{step_height}")

    if PYBULLET_CONNECT_MODE == p.GUI:
        client_id = p.connect(p.GUI, options=gui_options())
        set_camera(client_id)
    else:
        client_id = p.connect(p.DIRECT)

    p.resetSimulation(physicsClientId=client_id)
    p.setTimeStep(1.0 / PHYSICS_FREQ, physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.changeDynamics(plane_id, -1, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)

    urdf_path = os.path.join(PATH, URDF_FILE)
    if not os.path.exists(urdf_path):
        urdf_path = URDF_FILE
    robots = [p.loadURDF(urdf_path, [0, i * SPACING, 0.05], SPAWN_ORI,
                         physicsClientId=client_id,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
              for i in range(num_robots)]

    for r_id in robots:
        p.changeDynamics(r_id, -1, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)
        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            p.changeDynamics(r_id, j, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)

    apply_spinning_friction(plane_id, robots, mode, client_id)

    start_positions = [[0, i * SPACING, body_height + SPAWN_CLEARANCE] for i in range(num_robots)]
    joint_map = {p.getJointInfo(robots[0], j, physicsClientId=client_id)[1].decode(): j
                 for j in range(p.getNumJoints(robots[0], physicsClientId=client_id))}

    foot_link = {pre: joint_map[f"{pre}1"] for pre in LEG_PREFIXES}

    hard_reset_robots(robots, start_positions, joint_map, body_height, client_id)
    hold_stance(robots, joint_map, body_height, INITIAL_SETTLE_SEC, client_id)

    best_overall_score, best_overall_dna = -1e9, None
    best_overall_metrics = None
    servo_interval = int(PHYSICS_FREQ / SERVO_FREQ)

    for gen in range(GAIT_TRIALS):

        trials = [study.ask() for _ in range(num_robots)]
        population = [
            {
                "step_amplitude": trial.suggest_float("step_amplitude", *bounds["step_amplitude"]),
                "frequency":      trial.suggest_float("frequency", *bounds["frequency"]),
                "sway_amp":       trial.suggest_float("sway_amp", *bounds["sway_amp"]) if "sway_amp" in bounds else 0.0,
            }
            for trial in trials
        ]

        hard_reset_robots(robots, start_positions, joint_map, body_height, client_id)
        hold_stance(robots, joint_map, body_height, GEN_SETTLE_SEC, client_id)

        gen_start_positions = []
        gen_start_xy = []
        prev_yaws = [0.0] * num_robots
        accumulated_yaws = [0.0] * num_robots
        swing_checks = [0] * num_robots
        swing_drags = [0] * num_robots
        for i, r_id in enumerate(robots):
            pos, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
            gen_start_positions.append(list(pos))
            gen_start_xy.append(np.array(pos[:2]))
            _, _, yaw = p.getEulerFromQuaternion(orn)
            prev_yaws[i] = yaw

        steps = int(DURATION * PHYSICS_FREQ)
        for s in range(steps):
            t = s / PHYSICS_FREQ

            if s % servo_interval == 0:
                ramp = min(1.0, t / START_RAMP_TIME)
                for i, r_id in enumerate(robots):
                    dna = population[i]
                    cycle_phase = (t * dna["frequency"]) % 1.0

                    sway = ramp * dna["sway_amp"] * math.sin(2.0 * math.pi * cycle_phase)

                    for l_idx, prefix in enumerate(LEG_PREFIXES):
                        phase = (cycle_phase + gait_offsets[l_idx]) % 1.0
                        pm, lift, is_swing, s_phase = gait_curve(phase, duty, swing_profile)
                        phase_multiplier = pm * ramp
                        tz = body_height - LIFT_SIGN[mode] * step_height * ramp * lift

                        if drag_penalty > 0.0 and is_swing and 0.25 < s_phase < 0.75:
                            swing_checks[i] += 1
                            if p.getContactPoints(r_id, plane_id, linkIndexA=foot_link[prefix],
                                                  physicsClientId=client_id):
                                swing_drags[i] += 1

                        tx, ty, tz = foot_target(mode, prefix, phase_multiplier, tz, dna, direction, sway)
                        h, th, kn = solve_leg_ik_3dof(tx, ty, tz)
                        command_leg(r_id, joint_map, prefix, h, th, kn, client_id,
                                    max_vel=GLOBAL_MAX_VELOCITY)

            p.stepSimulation(physicsClientId=client_id)
            if PYBULLET_CONNECT_MODE == p.GUI:
                time.sleep(1.0 / PHYSICS_FREQ)

            if mode == "spin":
                for i, r_id in enumerate(robots):
                    _, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
                    _, _, yaw = p.getEulerFromQuaternion(orn)
                    diff = yaw - prev_yaws[i]
                    while diff > math.pi:  diff -= 2.0 * math.pi
                    while diff < -math.pi: diff += 2.0 * math.pi
                    accumulated_yaws[i] += diff
                    prev_yaws[i] = yaw

        fitnesses = []
        for i, r_id in enumerate(robots):
            pos, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
            roll, pitch, _ = p.getEulerFromQuaternion(orn)

            if mode == "spin":
                score, metrics = compute_spin_score(
                    accumulated_yaws[i], orn, pos, gen_start_xy[i],
                    target_vel, direction, population[i])
            else:
                score, metrics = compute_translational_score(
                    mode, gen_start_positions[i], pos, roll, pitch,
                    target_vel, orn, population[i], direction)

            drag_frac = swing_drags[i] / max(1, swing_checks[i])
            score *= (1.0 - drag_penalty * drag_frac)
            metrics["swing_drag_frac"] = drag_frac

            if pos[2] < (body_height * 0.5) or abs(roll) > 1.2:
                score *= 0.001

            fitnesses.append(score)
            study.tell(trials[i], score)

            if score > best_overall_score:
                best_overall_score = score
                best_overall_dna = copy.deepcopy(population[i])
                best_overall_metrics = copy.deepcopy(metrics)

        if _LOG_SINK is not None:
            _LOG_SINK({
                "key": run_key, "exp": exp_idx, "gen": gen,
                "scores": [float(f) for f in fitnesses],
                "best_score": float(best_overall_score),
                "dna": {k: float(v) for k, v in best_overall_dna.items()},
                "metrics": {k: float(v) for k, v in best_overall_metrics.items()},
            })

        print(f"[{mode} dir{direction} exp {exp_idx}] Gen {gen+1}/{GAIT_TRIALS}  "
              f"best={max(fitnesses):.3f}  mean={np.mean(fitnesses):.3f}  "
              f"all-time-best={best_overall_score:.3f}")

    if PYBULLET_CONNECT_MODE == p.GUI:
        save_camera(client_id)
    p.disconnect(client_id)

    return {
        "firmware_name":  gait_name if mode == "straight" else mode,
        "target_vel":     target_vel,
        "step_amplitude": best_overall_dna["step_amplitude"],
        "frequency":      best_overall_dna["frequency"],
        "sway_amp":       best_overall_dna["sway_amp"],
        "duty":           duty,
        "mode":           mode,
        "exp_id":         exp_idx,
        "gait":           gait_name,
        "target_height":  body_height,
        "step_height":    step_height,
        "direction":      direction,
        "score":          best_overall_score,
        **best_overall_metrics,
    }

# ============================================================
# 7. EXECUTE
# ============================================================
if __name__ == "__main__":
    unknown = [m for m in MODES if m not in GENE_BOUNDS]
    if unknown:
        raise SystemExit(f"Unknown mode(s) {unknown}. Choose from {list(GENE_BOUNDS)}")

    bad_gaits = [(m, g) for m in MODES for g in GAITS.get(m, []) if g not in GAIT_LIBRARY]
    if bad_gaits:
        raise SystemExit(f"Unknown gait(s) in GAITS: {bad_gaits}. Choose from {list(GAIT_LIBRARY)}")

    headless = PYBULLET_CONNECT_MODE == p.DIRECT
    MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    print(f"\n=== Running Optuna optimization for modes: {MODES}  "
          f"(render: {RENDER_MODE}, sampler: {SAMPLER_MODE}, "
          f"environments: {NUM_ROBOTS if headless else 1}) ===")

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    if ENABLE_TENSORBOARD:
        print(f">>> TensorBoard logs: {TENSORBOARD_LOGDIR}/{run_stamp}_{SAMPLER_MODE}/...   "
              f"view with: tensorboard --logdir {TENSORBOARD_LOGDIR}")

    repeats = NUM_EXPERIMENTS if headless else 1
    if not headless and NUM_EXPERIMENTS > 1:
        print(f">>> Windowed mode: running each setting once (NUM_EXPERIMENTS={NUM_EXPERIMENTS} only applies to headless).")

    tasks = []
    skipped = []
    for mode in MODES:
        dirs_for_mode = DIAGONAL_DIRECTIONS if mode == "diagonal" else DIRECTIONS
        if not dirs_for_mode:
            skipped.append(mode)
            continue
        for gait_name in GAITS[mode]:
            for t_vel in TARGET_SPEEDS[mode]:
                for b_height in BODY_HEIGHTS:
                    for step_height in STEP_HEIGHTS:
                        for direction in dirs_for_mode:
                            for i in range(repeats):
                                tasks.append((mode, gait_name, t_vel, b_height, step_height,
                                              direction, i, run_stamp))

    if skipped:
        print(f">>> Skipping (empty direction list): {skipped}")
    print(f">>> Total experiments: {len(tasks)}  |  workers: {MAX_WORKERS if headless else 1}")

    if not tasks:
        raise SystemExit("No experiments to run, check DIRECTIONS / DIAGONAL_DIRECTIONS / MODES.")

    logger = RunLogger(TENSORBOARD_LOGDIR, repeats) if ENABLE_TENSORBOARD else None

    if headless:
        manager = multiprocessing.Manager() if logger else None
        log_queue = manager.Queue() if manager else None
        stop_drain = threading.Event()

        def drain_log_queue():
            while not (stop_drain.is_set() and log_queue.empty()):
                try:
                    logger.put(log_queue.get(timeout=0.2))
                except queue.Empty:
                    continue

        drain_thread = threading.Thread(target=drain_log_queue, daemon=True) if logger else None
        if drain_thread:
            drain_thread.start()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker,
                                 initargs=(log_queue,)) as executor:
            all_results = list(tqdm(executor.map(run_experiment_worker, tasks),
                                    total=len(tasks), desc="Optuna Runs"))
        if drain_thread:
            stop_drain.set()
            drain_thread.join()
            manager.shutdown()
    else:
        _init_worker(logger)
        all_results = [run_experiment_worker(t) for t in tqdm(tasks, desc="Optuna Runs")]

    if logger:
        logger.close()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("opt_results.csv", index=False)
        print("\nAll experiments complete. Results saved to opt_results.csv.")

        best = df.loc[df.groupby(["firmware_name", "target_vel"])["score"].idxmax()]
        best = best.sort_values(["firmware_name", "target_vel"], ascending=[True, False])
        rows = [f'    {{ {chr(34) + r.firmware_name + chr(34) + ",":<11} {r.target_vel:.3f}, {r.step_amplitude:.15g}, '
                f'{r.frequency:.15g}, {r.sway_amp:.15g}, {r.duty} }},' for r in best.itertuples()]
        with open("firmware_rows.txt", "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print("\nBest row per (name, speed), also saved to firmware_rows.txt:")
        print("\n".join(rows))
