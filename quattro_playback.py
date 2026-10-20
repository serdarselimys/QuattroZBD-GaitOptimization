"""
Playback of optimized gaits for the quattro_zbd quadruped.

Reads the CSV written by the optimizer (opt_results.csv) and replays the saved gait
parameters in the pybullet window, one robot at a time, for every mode that was optimized
(straight, sideway, diagonal, spin).

Usage:
    python quattro_playback.py                       # every row in the CSV, one by one
    python quattro_playback.py --best                # the 3 best rows of each mode
    python quattro_playback.py --mode straight spin  # only these modes
    python quattro_playback.py --gait walk --top 3   # the 3 best walk rows overall
    python quattro_playback.py --speed 0.5 --follow  # half speed, camera follows the robot
"""
import argparse
import importlib
import math
import os
import time

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data

# Name of the optimizer file next to this script, without ".py"
OPTIMIZER_MODULE = "quattro_opt"

try:
    opt = importlib.import_module(OPTIMIZER_MODULE)
except ModuleNotFoundError as e:
    raise SystemExit(f"Could not import the optimizer '{OPTIMIZER_MODULE}.py' ({e}). "
                     f"Put this script in the same folder, or change OPTIMIZER_MODULE.")

SERVO_INTERVAL = int(opt.PHYSICS_FREQ / opt.SERVO_FREQ)
CAMERA = dict(distance=0.9, yaw=50, pitch=-25)

# Columns identifying one optimized setting (repeats of it only differ by exp_id)
BEST_PER_MODE = 3   # rows per mode played by --best
SETTING_COLS = ["mode", "gait", "direction", "target_vel", "target_height", "step_height"]


# ============================================================
# 1. PICK WHICH ROWS TO PLAY
# ============================================================
def load_rows(csv_path, modes, gait, top, best):
    df = pd.read_csv(csv_path)

    missing = [c for c in SETTING_COLS + ["step_amplitude", "frequency", "score"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns {missing}. Was it written by the current optimizer?")

    if modes:
        df = df[df["mode"].isin(modes)]
    if gait:
        df = df[df["gait"] == gait]
    if df.empty:
        return df

    if best:  # keep the BEST_PER_MODE highest-scoring rows of each mode
        df = df.sort_values("score", ascending=False).groupby("mode", sort=False).head(BEST_PER_MODE)

    # Play modes in the optimizer's order, fastest-first within each mode
    mode_order = {m: i for i, m in enumerate(opt.GENE_BOUNDS)}
    df = (df.assign(_m=df["mode"].map(mode_order))
            .sort_values(["_m", "gait", "direction", "target_vel", "score"],
                         ascending=[True, True, True, False, False])
            .drop(columns="_m"))

    if top and top > 0:
        df = df.sort_values("score", ascending=False).head(top)
    return df.reset_index(drop=True)


# ============================================================
# 2. WORLD
# ============================================================
def build_world(client_id):
    p.resetSimulation(physicsClientId=client_id)
    p.setTimeStep(1.0 / opt.PHYSICS_FREQ, physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.changeDynamics(plane_id, -1, lateralFriction=opt.LATERAL_FRICTION, physicsClientId=client_id)

    urdf_path = os.path.join(opt.PATH, opt.URDF_FILE)
    if not os.path.exists(urdf_path):
        urdf_path = opt.URDF_FILE
    robot = p.loadURDF(urdf_path, [0, 0, 0.05], opt.SPAWN_ORI, physicsClientId=client_id,
                       flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    p.changeDynamics(robot, -1, lateralFriction=opt.LATERAL_FRICTION, physicsClientId=client_id)
    for j in range(p.getNumJoints(robot, physicsClientId=client_id)):
        p.changeDynamics(robot, j, lateralFriction=opt.LATERAL_FRICTION, physicsClientId=client_id)

    joint_map = {p.getJointInfo(robot, j, physicsClientId=client_id)[1].decode(): j
                 for j in range(p.getNumJoints(robot, physicsClientId=client_id))}
    foot_link = {pre: joint_map[f"{pre}1"] for pre in opt.LEG_PREFIXES}
    return plane_id, robot, joint_map, foot_link


def hold_stance(robot, joint_map, body_height, client_id, seconds, sleep_dt):
    h, th, kn = opt.solve_leg_ik_3dof(0, 0, body_height)
    for _ in range(int(seconds * opt.PHYSICS_FREQ)):
        for prefix in opt.LEG_PREFIXES:
            opt.command_leg(robot, joint_map, prefix, h, th, kn, client_id)
        p.stepSimulation(physicsClientId=client_id)
        if sleep_dt > 0:
            time.sleep(sleep_dt)


# ============================================================
# 3. REPLAY ONE ROW (same loop and same scoring as the optimizer)
# ============================================================
def play_row(row, plane_id, robot, joint_map, foot_link, client_id, sleep_dt, follow):
    mode = row["mode"]
    gait_name = row["gait"]
    gait = opt.GAIT_LIBRARY[gait_name]
    offsets = gait["offsets"]
    duty = float(row["duty"]) if "duty" in row and not pd.isna(row["duty"]) else gait["duty"]

    body_height = float(row["target_height"])
    step_height = float(row["step_height"])
    direction = float(row["direction"])
    target_vel = float(row["target_vel"])
    swing_profile = opt.SWING_PROFILE[mode]
    drag_penalty = opt.SWING_DRAG_PENALTY[mode]

    dna = {
        "step_amplitude": float(row["step_amplitude"]),
        "frequency": float(row["frequency"]),
        "sway_amp": float(row["sway_amp"]) if "sway_amp" in row and not pd.isna(row["sway_amp"]) else 0.0,
    }

    opt.apply_spinning_friction(plane_id, [robot], mode, client_id)

    # Start exactly like a training generation: standing pose, settle, then measure the start.
    start_positions = [[0.0, 0.0, body_height + opt.SPAWN_CLEARANCE]]
    opt.hard_reset_robots([robot], start_positions, joint_map, body_height, client_id)
    hold_stance(robot, joint_map, body_height, client_id, opt.GEN_SETTLE_SEC, sleep_dt)

    start_pos, start_orn = p.getBasePositionAndOrientation(robot, physicsClientId=client_id)
    start_xy = np.array(start_pos[:2])
    prev_yaw = p.getEulerFromQuaternion(start_orn)[2]
    accumulated_yaw = 0.0
    swing_checks = swing_drags = 0
    peak_vel = peak_torque = 0.0
    leg_joint_ids = [joint_map[f"{pre}{n}"] for pre in opt.LEG_PREFIXES for n in (1, 2, 3)]

    for s in range(int(opt.DURATION * opt.PHYSICS_FREQ)):
        t = s / opt.PHYSICS_FREQ

        if s % SERVO_INTERVAL == 0:
            ramp = min(1.0, t / opt.START_RAMP_TIME)
            cycle_phase = (t * dna["frequency"]) % 1.0
            sway = ramp * dna["sway_amp"] * math.sin(2.0 * math.pi * cycle_phase)

            for l_idx, prefix in enumerate(opt.LEG_PREFIXES):
                phase = (cycle_phase + offsets[l_idx]) % 1.0
                pm, lift, is_swing, s_phase = opt.gait_curve(phase, duty, swing_profile)
                tz = body_height - opt.LIFT_SIGN[mode] * step_height * ramp * lift

                if drag_penalty > 0.0 and is_swing and 0.25 < s_phase < 0.75:
                    swing_checks += 1
                    if p.getContactPoints(robot, plane_id, linkIndexA=foot_link[prefix],
                                          physicsClientId=client_id):
                        swing_drags += 1

                tx, ty, tz = opt.foot_target(mode, prefix, pm * ramp, tz, dna, direction, sway)
                h, th, kn = opt.solve_leg_ik_3dof(tx, ty, tz)
                opt.command_leg(robot, joint_map, prefix, h, th, kn, client_id,
                                max_vel=opt.GLOBAL_MAX_VELOCITY)

            states = p.getJointStates(robot, leg_joint_ids, physicsClientId=client_id)
            peak_vel = max(peak_vel, max(abs(st[1]) for st in states))
            peak_torque = max(peak_torque, max(abs(st[3]) for st in states))

            if follow:
                pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client_id)
                p.resetDebugVisualizerCamera(CAMERA["distance"], CAMERA["yaw"], CAMERA["pitch"],
                                             pos, physicsClientId=client_id)

        p.stepSimulation(physicsClientId=client_id)
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        if mode == "spin":  # unwrap the heading every physics step, like the optimizer
            _, orn = p.getBasePositionAndOrientation(robot, physicsClientId=client_id)
            yaw = p.getEulerFromQuaternion(orn)[2]
            diff = yaw - prev_yaw
            while diff > math.pi:
                diff -= 2.0 * math.pi
            while diff < -math.pi:
                diff += 2.0 * math.pi
            accumulated_yaw += diff
            prev_yaw = yaw

    # ---- score it the way the optimizer does ----
    pos, orn = p.getBasePositionAndOrientation(robot, physicsClientId=client_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    if mode == "spin":
        score, metrics = opt.compute_spin_score(accumulated_yaw, orn, pos, start_xy,
                                                target_vel, direction, dna)
    else:
        score, metrics = opt.compute_translational_score(mode, list(start_pos), pos, roll, pitch,
                                                         target_vel, orn, dna, direction)

    drag_frac = swing_drags / max(1, swing_checks)
    score *= (1.0 - drag_penalty * drag_frac)
    fell = pos[2] < (body_height * 0.5) or abs(roll) > 1.2
    if fell:
        score *= 0.001

    metrics.update({"score": score, "swing_drag_frac": drag_frac, "fell": fell,
                    "peak_joint_vel": peak_vel, "peak_joint_torque": peak_torque, "yaw": yaw})
    return metrics


# ============================================================
# 4. MAIN
# ============================================================
def describe(row):
    name = row["firmware_name"] if "firmware_name" in row else row["mode"]
    return (f"{name}  (mode={row['mode']}, gait={row['gait']})   "
            f"target={row['target_vel']}  dir={row['direction']}  "
            f"height={row['target_height']}  step_h={row['step_height']}")


def main():
    ap = argparse.ArgumentParser(description="Play back optimized gaits from the optimizer's CSV.")
    ap.add_argument("--csv", default="opt_results.csv", help="CSV written by the optimizer")
    ap.add_argument("--mode", nargs="+", choices=list(opt.GENE_BOUNDS), default=None,
                    help="only these modes (default: every mode in the CSV)")
    ap.add_argument("--gait", default=None, choices=list(opt.GAIT_LIBRARY), help="only this gait")
    ap.add_argument("--top", type=int, default=0, help="only the N best-scoring rows (0 = no limit)")
    ap.add_argument("--best", action="store_true",
                    help=f"play only the {BEST_PER_MODE} best-scoring rows of each mode (default: every row)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed (1 = real time, 0.5 = half speed, 0 = as fast as possible)")
    ap.add_argument("--follow", action="store_true", help="camera follows the robot")
    args = ap.parse_args()

    df = load_rows(args.csv, args.mode, args.gait, args.top, args.best)
    if df.empty:
        print("No rows to play.")
        return
    print(f"Playing {len(df)} gait(s) from {args.csv}: "
          + ", ".join(f"{m} x{n}" for m, n in df["mode"].value_counts().items()))

    sleep_dt = (1.0 / opt.PHYSICS_FREQ) / args.speed if args.speed > 0 else 0.0

    client_id = p.connect(p.GUI, options=opt.gui_options())
    plane_id, robot, joint_map, foot_link = build_world(client_id)

    try:
        for idx, row in df.iterrows():
            print("\n================================================")
            print(f"{idx + 1}/{len(df)}   {describe(row)}")
            print(f"  amplitude={row['step_amplitude']:.4f}  freq={row['frequency']:.3f}  "
                  f"sway={row.get('sway_amp', 0.0):.4f}  duty={row.get('duty', float('nan')):.2f}   "
                  f"CSV score={row['score']:.3f}")
            print("================================================")

            m = play_row(row, plane_id, robot, joint_map, foot_link, client_id, sleep_dt, args.follow)

            if row["mode"] == "spin":
                result = (f"Rotation speed: {m['actual_rot_speed']:.3f} rad/s   "
                          f"Drift: {m.get('drift_quality', 0):.3f} (drift quality)")
            else:
                result = (f"Distance: {m['distance_traveled']:.4f} m   Avg speed: {m['avg_speed']:.4f} m/s   "
                          f"Lateral drift: {m['lateral_deviation']:.4f} m   Yaw: {math.degrees(m['yaw']):.2f} deg")
            print(result)
            print(f"Replayed score: {m['score']:.3f}   "
                  f"swing drag: {m['swing_drag_frac']:.2%}   "
                  f"peak vel: {m['peak_joint_vel']:.2f}/{opt.GLOBAL_MAX_VELOCITY}   "
                  f"peak torque: {m['peak_joint_torque']:.2f}/{opt.GLOBAL_MAX_TORQUE}"
                  + ("   ** FELL **" if m["fell"] else ""))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        p.disconnect(client_id)
    print("\nPlayback complete.")


if __name__ == "__main__":
    main()
