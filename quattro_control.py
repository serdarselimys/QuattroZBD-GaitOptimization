import pybullet as p
import pybullet_data
import pandas as pd
import math
import random
import time
import numpy as np
import os
import sys
import argparse

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

os.environ.setdefault("SDL_JOYSTICK_WGI", "0")
import pygame
try:
    from pygame._sdl2 import controller as sdl_controller   
except Exception:
    sdl_controller = None

# ============================================================
# GAIT PROFILE DATA
# ============================================================
FILE_PATH = "control_params.csv"   # optimizer output

# Which optimized gait drives straight / reverse motion. Button A toggles it at runtime.
STRAIGHT_GAIT = "trot"

# Smoothing
PARAM_SMOOTH_TAU = 0.20         # s, low-pass on frequency / amplitude / duty / sway / lift direction
GAIT_BLEND_TAU = 0.25           # s, foot cross-fade when the leg timing itself changes (trot <-> walk)
WALK_ENTER_SHARE = 0.60         # walk timing is used only while forward/back weight exceeds this
WALK_EXIT_SHARE = 0.40          # ...and is kept until it drops below this (hysteresis)

SIDE_SCALE = 0.7                # attenuation of the left stick's sideways axis (as before)

KEYBOARD_SPEED = 0.5            # fraction of full stick speed for the arrow keys and Q / E (hold Shift for 100 %)

# Swing-lift convention for the WHOLE mix. This decides which leg pair carries the load, and so the sign of every foot motion.
LIFT_MODE = "up"

DIAGONAL_DEFAULT_DUTY = 0.55

FALLBACK_ROWS = [
    {"gait": "trot", "target_vel": 0.125, "step_amplitude": 0.0435, "frequency": 1.109,
     "sway_amp": 0.0044, "duty": 0.5},
    {"gait": "walk", "target_vel": 0.125, "step_amplitude": 0.0395, "frequency": 1.366,
     "sway_amp": 0.0064, "duty": 0.6},
    {"gait": "sideway", "target_vel": 0.075, "step_amplitude": 0.0172, "frequency": 0.8026},
    {"gait": "diagonal", "target_vel": 0.075, "step_amplitude": 0.0172, "frequency": 1.018},
    {"gait": "spin", "target_vel": 0.3, "step_amplitude": 0.0602, "frequency": 1.025},
]


def load_profiles(path):
    try:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using built-in fallback profiles.")
        df = pd.DataFrame(FALLBACK_ROWS)


    if "firmware_name" in df.columns:
        df["gait"] = df["firmware_name"]
    df["gait"] = df["gait"].astype(str).str.strip().str.lower()


    if "score" in df.columns:
        keys = [c for c in ("gait", "target_vel", "direction") if c in df.columns]
        df = df.loc[df.groupby(keys)["score"].idxmax()]


    if "duty" in df.columns:
        df.loc[~df["gait"].isin(["trot", "walk"]), "duty"] = float("nan")

    profiles = {}
    for gait, g in df.groupby("gait"):
        profiles[gait] = g.sort_values("target_vel").to_dict("records")
    return profiles


def num(row, key, default):
    """Read a numeric CSV cell, treating missing / NaN as `default`."""
    try:
        v = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


PROFILES = load_profiles(FILE_PATH)


def select_row(gait_name, t_norm):
    """Map a 0..1 stick magnitude onto the [min, max] target_vel range of a gait and
    return the closest optimized row."""
    rows = PROFILES.get(gait_name)
    if not rows:
        rows = PROFILES.get("trot") or next(iter(PROFILES.values()))
    speeds = [num(r, "target_vel", 0.0) for r in rows]
    lo, hi = min(speeds), max(speeds)
    mapped = lo + t_norm * (hi - lo)
    best = min(range(len(rows)), key=lambda k: abs(speeds[k] - mapped))
    return rows[best]

# ============================================================
# ROBOT CONSTANTS (identical to the optimization scripts)
# ============================================================
L1, L2 = 0.1000, 0.1000
URDF_X_OFFSET = -0.010
BODY_OFFSET = 0.0544

NEUTRAL_HEIGHT = 0.20
STEP_HEIGHT = 0.010

GLOBAL_MAX_TORQUE = 1.8
GLOBAL_MAX_VELOCITY = 6.0
PHYSICS_FREQ = 240
SERVO_FREQ = 100
DT = 1.0 / PHYSICS_FREQ
SERVO_INTERVAL = int(PHYSICS_FREQ / SERVO_FREQ)

STOP_BLEND_SPEED = 0.08
JOYSTICK_DEADBAND = 0.10

# ---- Stand / sit, body height and pitch trim (ported from the hexapod firmware sim) ----
START_STANDING = False          # False: boot sitting, stand up with L1+R1 (hold), Start, or the S key
STARTUP_DURATION = 2.0          # s, stand / sit ramp (raised cosine)
REQUIRED_HOLD_DURATION = 2.0    # s, L1+R1 must be held this long to toggle stand / sit
SIT_POSE_HEIGHT = 0.12          # m, every foot target collapses toward this while sitting
KNEE_ASSEMBLY_OFFSET = math.pi / 2   # folded knee angle (magnitude) in the sit pose

MIN_HEIGHT = 0.14               # m, body height limits
MAX_HEIGHT = 0.23
HEIGHT_SPEED = 0.06             # m/s while L1 / R1 is held
HEIGHT_FOLLOW = 0.02            # per-step smoothing of the actual height toward the target

PITCH_TRIM_SPEED_DEG_S = 8.0    # deg/s while L2 / R2 is held
PITCH_TRIM_MAX_DEG = 20.0

# ============================================================
# GAMEPAD INPUT  (works with most pads, nothing to edit)
# ============================================================
# How it works
#
# Controls (same on every pad):
#   Left stick  = walk / strafe        Right stick X = spin
#   A           = toggle trot <-> walk Start         = stand / sit
#   L1 / R1     = body height -/+      L2 / R2       = pitch trim -/+
#   L1 + R1 (hold 2 s) = stand / sit
#
# Keyboard (the PyBullet window must have focus; works with or without a gamepad, and is added on top of it):
#   Arrow keys = walk / strafe (Shift = full speed)   Q / E = spin left / right
#   R / F      = body height +/-                       Z / X = pitch trim -/+
#   W          = toggle trot <-> walk                  S     = stand / sit
#   I          = idle motion on / off                  A     = auto-sleep on / off

GAMEPAD_BACKEND = "auto"        # "auto" | "sdl" | "raw"
GAMEPAD_INDEX = None            # None = use whichever connected pad you touch; or an int to lock to one
GAMEPAD_RAW_LAYOUT = None       # None = auto-detect; only used by the raw backend
TRIGGER_THRESHOLD = 0.05        # analog trigger counts as pressed above this (0..1)

# Raw-backend layouts: axis / button numbers as pygame reports them for plain joysticks.
RAW_LAYOUTS = {
    # Xbox-style pad under Linux (xpad): triggers on axes 2 / 5, right stick X on axis 3
    "xbox_linux": dict(lx=0, ly=1, rx=3, l2_axis=2, r2_axis=5, l1=4, r1=5, l2_btn=6, r2_btn=7, a=0, start=7),
    # Xbox-style pad under Windows (XInput): right stick X on axis 2, triggers on axes 4 / 5
    "xbox_win":   dict(lx=0, ly=1, rx=2, l2_axis=4, r2_axis=5, l1=4, r1=5, l2_btn=6, r2_btn=7, a=0, start=7),
    # Logitech DirectInput ("D-mode") style: triggers are buttons
    "dinput":     dict(lx=0, ly=1, rx=2, l2_axis=None, r2_axis=None, l1=4, r1=5, l2_btn=6, r2_btn=7, a=1, start=9),
}
DEFAULT_RAW_LAYOUT = "xbox_linux"   # the layout this script used before (works with the Rampage Hydra)


def pick_raw_layout(name, n_axes):
    low = name.lower()
    logitech = any(k in low for k in ("logitech", "f710", "f310", "f510"))
    xboxish = logitech or any(k in low for k in ("xbox", "xinput", "x-box"))
    if n_axes >= 6 and xboxish and sys.platform.startswith("win"):
        return "xbox_win"
    if n_axes < 6 and logitech:
        return "dinput"
    return DEFAULT_RAW_LAYOUT


# Pads whose SDL mapping misbehaves are read as plain joysticks instead.
PAD_BACKEND_OVERRIDES = {"rampage": "raw", "atari xbox 360": "raw"}


def _device_names(i):
    try:
        if sdl_controller is not None:
            n = sdl_controller.name_forindex(i)
            if n:
                return [n.lower()]
    except Exception:
        pass
    return []


def pad_backend_override(i):
    for name in _device_names(i):
        for key, backend in PAD_BACKEND_OVERRIDES.items():
            if key in name:
                return backend
    return None


_ap = argparse.ArgumentParser(description="Quattro gamepad control")
_ap.add_argument("--backend", choices=("auto", "sdl", "raw"), default=None, help="gamepad backend")
_ap.add_argument("--pad", type=int, default=None, help="lock to this joystick index")
_ap.add_argument("--raw-layout", choices=sorted(RAW_LAYOUTS), default=None, help="layout for the raw backend")
_ap.add_argument("--gamepad-debug", action="store_true", help="print live gamepad values")
_args, _ = _ap.parse_known_args()


class PadState:
    """Normalised pad state, identical for every backend."""
    NAMES = ("lx", "ly", "rx", "l1", "r1", "l2", "r2", "a", "start")

    def __init__(self):
        self.lx = self.ly = self.rx = 0.0          # -1..1, up / left = negative
        self.l1 = self.r1 = self.l2 = self.r2 = self.a = self.start = False

    def vec(self):
        return [float(getattr(self, k)) for k in self.NAMES]

    def text(self):
        btns = " ".join(k.upper() for k in ("l1", "r1", "l2", "r2", "a", "start") if getattr(self, k))
        return f"lx={self.lx:+.2f} ly={self.ly:+.2f} rx={self.rx:+.2f}  [{btns or '-'}]"


def _clip(v):
    return max(-1.0, min(1.0, v))


def _sanitize(s):
    """DirectInput pads (e.g. a Logitech F710 in D-mode) report EVERY stick axis as -1.0 until the first
    event arrives from the pad. Three sticks cannot sit at -1 at once, so treat that as 'not initialised
    yet' and report centred sticks (otherwise the robot would be told to run full speed at start-up)."""
    if s.lx <= -0.999 and s.ly <= -0.999 and s.rx <= -0.999:
        s.lx = s.ly = s.rx = 0.0
    return s


class SdlPad:
    """Pad known to SDL's controller database: standard names on every OS and in X or D mode."""
    kind = "sdl"

    def __init__(self, index):
        self.index = index
        self.dev = sdl_controller.Controller(index)
        try:
            self.name = sdl_controller.name_forindex(index) or f"pad {index}"
        except Exception:
            self.name = f"pad {index}"
        self.baseline = None
        self.engaged_since = None
        self.last_input = 0.0
        self.state = PadState()

    def describe(self):
        return "sdl mapping"

    def _axis(self, a):
        return _clip(self.dev.get_axis(a) / 32767.0)

    def _btn(self, b):
        return bool(self.dev.get_button(b))

    def read(self):
        s = PadState()
        if not self.dev.attached():
            return s
        c = sdl_controller
        s.lx = self._axis(c.CONTROLLER_AXIS_LEFTX)
        s.ly = self._axis(c.CONTROLLER_AXIS_LEFTY)
        s.rx = self._axis(c.CONTROLLER_AXIS_RIGHTX)
        s.l1 = self._btn(c.CONTROLLER_BUTTON_LEFTSHOULDER)
        s.r1 = self._btn(c.CONTROLLER_BUTTON_RIGHTSHOULDER)
        s.l2 = self._axis(c.CONTROLLER_AXIS_TRIGGERLEFT) > TRIGGER_THRESHOLD
        s.r2 = self._axis(c.CONTROLLER_AXIS_TRIGGERRIGHT) > TRIGGER_THRESHOLD
        s.a = self._btn(c.CONTROLLER_BUTTON_A)
        s.start = self._btn(c.CONTROLLER_BUTTON_START)
        return _sanitize(s)

    def raw_text(self):
        # Also show what plain joystick access sees, to tell "SDL mapping problem" from "no data at all"
        try:
            if getattr(self, "_joy", None) is None:
                self._joy = pygame.joystick.Joystick(self.index)
            j = self._joy
            axes = " ".join(f"{j.get_axis(i):+.2f}" for i in range(j.get_numaxes()))
            down = [i for i in range(j.get_numbuttons()) if j.get_button(i)]
            att = self.dev.attached()
            return f"\n      raw axes: [{axes}]  raw buttons down: {down}  attached={att}"
        except Exception as e:
            return f"\n      (raw view unavailable: {e})"


class RawPad:
    """Any other joystick, read through a RAW_LAYOUTS entry."""
    kind = "raw"

    def __init__(self, index, layout_name=None):
        self.index = index
        self.joy = pygame.joystick.Joystick(index)
        self.name = self.joy.get_name()
        self.n_axes = self.joy.get_numaxes()
        self.n_btn = self.joy.get_numbuttons()
        self.layout_name = layout_name or pick_raw_layout(self.name, self.n_axes)
        self.L = RAW_LAYOUTS[self.layout_name]
        self._armed = {}
        self.baseline = None
        self.engaged_since = None
        self.last_input = 0.0
        self.state = PadState()

    def describe(self):
        return f"raw, layout '{self.layout_name}', {self.n_axes} axes, {self.n_btn} buttons"

    def _axis(self, i):
        return self.joy.get_axis(i) if i is not None and 0 <= i < self.n_axes else 0.0

    def _btn(self, i):
        return bool(i is not None and 0 <= i < self.n_btn and self.joy.get_button(i))

    def _trigger(self, key):
        ai = self.L[key + "_axis"]
        if ai is not None and 0 <= ai < self.n_axes:
            v = self.joy.get_axis(ai)

            if v <= -0.9:
                self._armed[key] = True
            return self._armed.get(key, False) and (v + 1.0) / 2.0 > TRIGGER_THRESHOLD
        return self._btn(self.L[key + "_btn"])

    def read(self):
        L, s = self.L, PadState()
        s.lx, s.ly, s.rx = self._axis(L["lx"]), self._axis(L["ly"]), self._axis(L["rx"])
        s.l1, s.r1 = self._btn(L["l1"]), self._btn(L["r1"])
        s.l2, s.r2 = self._trigger("l2"), self._trigger("r2")
        s.a, s.start = self._btn(L["a"]), self._btn(L["start"])
        return _sanitize(s)

    def raw_text(self):
        axes = " ".join(f"{self.joy.get_axis(i):+.2f}" for i in range(self.n_axes))
        down = [i for i in range(self.n_btn) if self.joy.get_button(i)]
        return f"\n      raw axes: [{axes}]  raw buttons down: {down}"


class GamepadManager:
    """Finds every connected pad, follows hot-plug, and exposes the most recently used one."""

    def __init__(self, backend="auto", index=None, raw_layout=None):
        self.backend, self.index, self.raw_layout = backend, index, raw_layout
        self.pads, self.active = [], None
        self._refresh_at = None
        self._ignore_events_until = time.time() + 1.5     # SDL announces already-plugged pads at startup
        self.refresh()

    def toggle_backend(self):
        self.backend = "raw" if self.backend != "raw" else "auto"
        print(f"[gamepad] backend -> {self.backend}")
        self.refresh()

    def request_refresh(self):
        if time.time() >= self._ignore_events_until:
            self._refresh_at = time.time() + 0.5          # let the device settle, then rescan

    def refresh(self):
        self.pads = []
        use_sdl = self.backend in ("auto", "sdl") and sdl_controller is not None
        for i in range(pygame.joystick.get_count()):
            if self.index is not None and i != self.index:
                continue
            pad = None
            forced_raw = self.backend == "auto" and pad_backend_override(i) == "raw"
            if use_sdl and not forced_raw:
                try:
                    if sdl_controller.is_controller(i):
                        pad = SdlPad(i)
                except Exception as e:
                    print(f"[gamepad] SDL mapping failed for #{i}: {e}")
            if pad is None and self.backend in ("auto", "raw"):
                try:
                    pad = RawPad(i, self.raw_layout)
                except Exception as e:
                    print(f"[gamepad] could not open #{i}: {e}")
            if pad is None:
                print(f"[gamepad] #{i} skipped (backend '{self.backend}' cannot use it)")
                continue
            self.pads.append(pad)
            print(f"[gamepad] #{pad.index} {pad.name}  ->  {pad.describe()}")
        self.active = self.pads[0] if self.pads else None
        self.switched_at = time.perf_counter()
        if not self.pads:
            print("[gamepad] No gamepad found -- plug one in any time (hot-plug works), "
                  "or use the keyboard controls listed below.")
        elif len(self.pads) > 1:
            print("[gamepad] Several pads connected: the one you touch last is used.")

    def _demote(self, pad, err):
        """A pad whose read keeps failing: SDL pads fall back to plain joystick reading."""
        if pad.kind != "sdl":
            if not getattr(pad, "_warned", False):
                pad._warned = True
                print(f"[gamepad] #{pad.index} {pad.name}: read failed ({err!r})")
            return None
        print(f"[gamepad] #{pad.index} {pad.name}: SDL read failed ({err!r}) -> reading this pad raw instead")
        try:
            new = RawPad(pad.index, self.raw_layout)
        except Exception as e2:
            print(f"[gamepad] #{pad.index}: raw fallback failed too ({e2!r})")
            return None
        self.pads[self.pads.index(pad)] = new
        if self.active is pad:
            self.active = new
        return new

    def poll(self):
        if self._refresh_at is not None and time.time() >= self._refresh_at:
            self._refresh_at = None
            self.refresh()
        if not self.pads:
            return PadState()
        if sdl_controller is not None:
            try:
                sdl_controller.update()       
            except Exception:
                pass
        now = time.perf_counter()
        for pad in list(self.pads):
            try:
                pad.state = pad.read()
            except Exception as e:
                self._demote(pad, e)
                pad.state = PadState()
                pad.engaged_since = None
                continue
            v = pad.state.vec()
            if pad.baseline is None:
                pad.baseline = v                           
                pad.engaged_since = None
            elif any(abs(x - y) > 0.4 for x, y in zip(v, pad.baseline)):
                if pad.engaged_since is None:
                    pad.engaged_since = now
            else:
                pad.engaged_since = None

        cands = [q for q in self.pads if q.engaged_since is not None
                 and now - q.engaged_since >= 0.25 and q.engaged_since > self.switched_at]
        if cands:
            best = max(cands, key=lambda q: q.engaged_since)
            if best is not self.active:
                self.active = best
                print(f"[gamepad] now using #{best.index} {best.name}")
            self.switched_at = best.engaged_since
        return (self.active or self.pads[0]).state

    def debug_text(self):
        if not self.pads:
            return "[gamepad] none connected"
        return "\n".join(f"[gamepad] {'*' if p is self.active else ' '} #{p.index} {p.name}: "
                         f"{p.state.text()}{' <engaged>' if getattr(p, 'engaged_since', None) is not None else ''}{p.raw_text()}" for p in self.pads)

def key_down(keys, key):
    codes = (ord(key), ord(key.upper())) if isinstance(key, str) else (key,)
    return any(keys.get(c, 0) & p.KEY_IS_DOWN for c in codes)


def keyboard_input(keys):
    """Keyboard -> the same quantities the gamepad gives: (fwd, side, spin, l1, r1, l2, r2, toggle_gait)."""
    fwd = int(key_down(keys, p.B3G_UP_ARROW)) - int(key_down(keys, p.B3G_DOWN_ARROW))
    side = int(key_down(keys, p.B3G_LEFT_ARROW)) - int(key_down(keys, p.B3G_RIGHT_ARROW))
    spin = int(key_down(keys, "e")) - int(key_down(keys, "q"))
    speed = 1.0 if key_down(keys, p.B3G_SHIFT) else KEYBOARD_SPEED
    return (fwd * speed, side * speed / SIDE_SCALE, spin * speed,
            key_down(keys, "f"), key_down(keys, "r"), key_down(keys, "z"), key_down(keys, "x"),
            key_down(keys, "w"))


# ---- Idle "alive" motion, auto-sleep and axis snap  ----
IDLE_ENABLED = True             # key I toggles at runtime
IDLE_START_S = 3.0              # s of no input before the idle motion fades in
AUTO_SLEEP_ENABLED = True       # key A toggles at runtime
AUTO_SLEEP_S = 60.0             # s of no input before the robot sits down
AXIS_SNAP_DEG = 10.0            # stick within this angle of an axis snaps to pure forward/back or pure sideways
AXIS_SNAP_HYST_DEG = 2.0        # extra angle before it un-snaps

LEG_ORDER = ["FL", "RL", "FR", "RR"]
LEG_ROOTS = {
    "FL": [0.12, 0.10], "RL": [-0.12, 0.10],
    "FR": [0.12, -0.10], "RR": [-0.12, -0.10],
}

# Per-leg joint sign convention (from the optimizers)
THIGH_SIGN = {"FL": -1.0, "RL": -1.0, "FR": 1.0, "RR": 1.0}
KNEE_SIGN = {"FL": 1.0, "RL": 1.0, "FR": 1.0, "RR": -1.0}
HIP_SIGN = {"FL": 1.0, "RL": 1.0, "FR": 1.0, "RR": 1.0}

# merged_opt (trot/walk) calibration. FOOT_X_SIGN maps "body forward" to the foot's x travel.
FOOT_X_SIGN = {"FL": -1.0, "RL": -1.0, "FR": -1.0, "RR": -1.0}
LIFT_SIGN = {"FL": 1.0, "RL": 1.0, "FR": 1.0, "RR": 1.0}
HIP_SIDE_SIGN = {"L": 1.0, "R": 1.0}

# Gait library (leg order [FL, RL, FR, RR])
GAIT_LIBRARY = {
    "trot": {"offsets": [0.0, 0.5, 0.5, 0.0], "duty": 0.5},
    "walk": {"offsets": [0.75, 0.0, 0.25, 0.5], "duty": 0.60},
}
TROT_OFFSETS = GAIT_LIBRARY["trot"]["offsets"]

# Optional IMU stabilizer. The new optimizers do not tune these gains, so it is OFF by default.
ENABLE_STABILIZATION = False
KP_ROLL, KD_ROLL = 0.12, 0.02
KP_PITCH, KD_PITCH = 0.12, 0.02

# ============================================================
# ANATOMICAL INVERSE KINEMATICS (clamped version from merged_opt)
# ============================================================
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


def command_leg(robot_id, joint_map, prefix, h, th, kn):
    p.setJointMotorControl2(robot_id, joint_map[f"{prefix}3"], p.POSITION_CONTROL,
                            HIP_SIGN[prefix] * h, force=GLOBAL_MAX_TORQUE,
                            maxVelocity=GLOBAL_MAX_VELOCITY)
    p.setJointMotorControl2(robot_id, joint_map[f"{prefix}2"], p.POSITION_CONTROL,
                            THIGH_SIGN[prefix] * th, force=GLOBAL_MAX_TORQUE,
                            maxVelocity=GLOBAL_MAX_VELOCITY)
    p.setJointMotorControl2(robot_id, joint_map[f"{prefix}1"], p.POSITION_CONTROL,
                            KNEE_SIGN[prefix] * kn, force=GLOBAL_MAX_TORQUE,
                            maxVelocity=GLOBAL_MAX_VELOCITY)

# ============================================================
# FOOT PROFILES
# ============================================================
def quintic_ease(s):
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def cycloid_ease(s):
    return s - math.sin(2.0 * math.pi * s) / (2.0 * math.pi)


def leg_profiles(phase, duty):
    """Normalised foot-travel profiles at a leg phase (all run +1 -> -1 in swing, -1 -> +1 in stance).

    Returns (pm_q, pm_c, lift_q, lift_c):
      pm_q / lift_q : quintic swing + linear stance (merged_opt, used by the straight gaits)
      pm_c / lift_c : cycloid (sideway / diagonal / spin optimizers)
    """
    swing_frac = 1.0 - duty
    if phase < swing_frac:
        s = phase / swing_frac
        q = quintic_ease(s)
        return (1.0 - 2.0 * q, 1.0 - 2.0 * cycloid_ease(s),
                4.0 * q * (1.0 - q), math.sin(math.pi * s))
    s = (phase - swing_frac) / duty
    return (-1.0 + 2.0 * s, -1.0 + 2.0 * cycloid_ease(s), 0.0, 0.0)

# ============================================================
# COMMAND MIXING  (single continuous stick vector, like the hexapod script)
# ============================================================
def build_command(joy_fwd, joy_side, joy_spin, prev_walk):

    t_raw = math.hypot(joy_fwd, joy_side)
    t = min(1.0, t_raw)
    r = abs(joy_spin)
    m_max = max(t, r, 1e-9)

    tw = t / m_max if t > 0.001 else 0.0     # translation share
    rw = r / m_max if r > 0.001 else 0.0     # spin share
    if t_raw > 1e-6:
        c, s = joy_fwd / t_raw, joy_side / t_raw
    else:
        c, s = 1.0, 0.0

    fa, sa = abs(c), abs(s)
    mn = min(fa, sa)
    tot = max(fa + sa, 1e-9)
    w_str, w_sid, w_dia = (fa - mn) / tot, (sa - mn) / tot, 2.0 * mn / tot

    denom = max(tw + rw, 1e-9)
    straight_share = tw * w_str / denom      # 0..1, "how straight is the command overall"
    if STRAIGHT_GAIT == "walk":
        use_walk = straight_share > (WALK_EXIT_SHARE if prev_walk else WALK_ENTER_SHARE)
    else:
        use_walk = False
    timing = "walk" if use_walk else "trot"

    side_name = "sideway" if "sideway" in PROFILES else STRAIGHT_GAIT
    diag_name = "diagonal" if "diagonal" in PROFILES else side_name

    ft = speed_t = duty_t = sway_t = 0.0
    if tw > 0.0:
        r_str = select_row(STRAIGHT_GAIT, t)
        r_sid = select_row(side_name, t)
        r_dia = select_row(diag_name, t)
        straight_duty = (num(r_str, "duty", GAIT_LIBRARY["walk"]["duty"]) if use_walk
                         else GAIT_LIBRARY["trot"]["duty"])
        comps = (
            (w_str, num(r_str, "frequency", 1.0), num(r_str, "step_amplitude", 0.03), straight_duty),
            (w_sid, num(r_sid, "frequency", 1.0), num(r_sid, "step_amplitude", 0.02), 0.5),
            (w_dia, num(r_dia, "frequency", 1.0), num(r_dia, "step_amplitude", 0.02),
             num(r_dia, "duty", DIAGONAL_DEFAULT_DUTY)),
        )
        for w, f_i, a_i, d_i in comps:
            ft += w * f_i
            speed_t += w * a_i * f_i        # amp * freq  ~ stride speed
            duty_t += w * d_i
        sway_t = w_str * num(r_str, "sway_amp", 0.0)

    fr = ar = 0.0
    if rw > 0.0:
        r_spi = select_row("spin", r)
        fr, ar = num(r_spi, "frequency", 1.0), num(r_spi, "step_amplitude", 0.05)

    freq = (tw * ft + rw * fr) / denom
    freq = max(freq, 1e-3)
    duty = (tw * duty_t + rw * 0.5) / denom

    cmd = {
        "freq": freq,
        "duty": duty,
        "amp_t": tw * speed_t / freq,                 # translation amplitude (already scaled by tw)
        "amp_r": rw * ar * fr / freq,                 # spin amplitude (already scaled by rw)
        "sway": tw * sway_t,
        "straight_mix": w_str if tw > 0.0 else 0.0,   # 1 = pure fwd/back: quintic shape
        "c": c, "s": s,
        "spin_dir": -1.0 if joy_spin > 0.0 else 1.0,  # stick right = CW
        "timing": timing,
    }
    return cmd


SMOOTH_KEYS = ("freq", "duty", "amp_t", "amp_r", "sway", "straight_mix")

# ============================================================
# SIMULATION INITIALIZATION
# ============================================================
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(DT)
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.0)

_urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quattro_zbd.urdf")
if not os.path.exists(_urdf_path):
    _urdf_path = "quattro_zbd.urdf"
robot_id = p.loadURDF(_urdf_path, [0, 0, 0.05],
                      flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

joint_map = {
    p.getJointInfo(robot_id, j)[1].decode(): j
    for j in range(p.getNumJoints(robot_id))
}

for j in range(-1, p.getNumJoints(robot_id)):
    p.changeDynamics(robot_id, j, lateralFriction=1.0)

pygame.init()
pygame.joystick.init()
if sdl_controller is not None:
    sdl_controller.init()
else:
    print("[gamepad] pygame._sdl2.controller unavailable (pygame < 2?) -- upgrade with: pip install -U pygame")
gamepads = GamepadManager(_args.backend or GAMEPAD_BACKEND,
                          _args.pad if _args.pad is not None else GAMEPAD_INDEX,
                          _args.raw_layout or GAMEPAD_RAW_LAYOUT)
HOTPLUG_EVENTS = tuple(getattr(pygame, n) for n in ("JOYDEVICEADDED", "JOYDEVICEREMOVED") if hasattr(pygame, n))
gamepad_debug = _args.gamepad_debug
last_debug = 0.0
last_real = time.perf_counter()

print("[keyboard] arrows = walk / strafe (Shift = full speed), Q / E = spin, R / F = height, Z / X = pitch trim, "
      "W = trot <-> walk, S = stand / sit, I = idle, A = auto-sleep")

physics_step = 0
phase_accumulator = 0.0
motion_blend = 0.0
roll = pitch = roll_rate = pitch_rate = 0.0

class IdleLife:
    """Idle 'alive' motion: slow breathing plus a small smoothed random drift in roll / pitch / height.
    Same model as the hexapod firmware. It fades in after IDLE_START_S of no input and out on any input."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.breath_period_s = 4.2
        self.breath_height_m = 0.0062
        self.breath_pitch_deg = 0.40
        self.breath_inhale_frac = 0.42
        self.drift = {"roll": 1.05, "pitch": 0.85, "height": 0.0040}
        self.drift_tau_s, self.smooth_tau_s = 3.5, 0.9
        self.max_roll, self.max_pitch, self.max_h = 4.5, 4.0, 0.020
        self.fade_in_s, self.fade_out_s = 0.8, 0.25
        self._ou = {k: 0.0 for k in self.drift}
        self._sm = {k: 0.0 for k in self.drift}
        self._phase = self._rng.random()
        self.envelope = 0.0

    def _comp(self):
        td, ts = self.drift_tau_s, self.smooth_tau_s
        return math.sqrt((td + ts) / td)

    def _breath(self, ph):
        f = self.breath_inhale_frac
        shaped = (0.5 * ph / f) if ph < f else (0.5 + 0.5 * (ph - f) / (1.0 - f))
        return -math.cos(2.0 * math.pi * shaped)

    def update(self, dt, activity, allowed):
        """Returns (roll_deg, pitch_deg, dz_m) to add to the body attitude / height."""
        target = (1.0 - min(max(activity, 0.0), 1.0)) if allowed else 0.0
        tau = self.fade_in_s if target > self.envelope else self.fade_out_s
        self.envelope += (target - self.envelope) * (1.0 - math.exp(-dt / max(tau, 1e-3)))

        self._phase = (self._phase + dt / self.breath_period_s) % 1.0
        breath = self._breath(self._phase)

        a = math.exp(-dt / self.drift_tau_s)
        ng = math.sqrt(max(0.0, 1.0 - a * a))
        b = math.exp(-dt / self.smooth_tau_s)
        c = self._comp()
        out = {}
        for k in self._ou:
            self._ou[k] = a * self._ou[k] + ng * self._rng.gauss(0.0, 1.0)
            self._sm[k] = b * self._sm[k] + (1.0 - b) * self._ou[k]
            out[k] = self._sm[k] * c

        roll = max(-self.max_roll, min(self.max_roll, out["roll"] * self.drift["roll"]))
        pitch = out["pitch"] * self.drift["pitch"] + breath * self.breath_pitch_deg
        pitch = max(-self.max_pitch, min(self.max_pitch, pitch))
        dz = out["height"] * self.drift["height"] + breath * self.breath_height_m
        dz = max(-self.max_h, min(self.max_h, dz))
        e = self.envelope
        return roll * e, pitch * e, dz * e


def command_leg_blend(robot_id, joint_map, prefix, h, th, kn, sp):
    """Same as command_leg, but blended with the folded 'sleep' pose by sp (0 = sitting, 1 = standing):
    hip and thigh scale toward 0, the knee goes toward its folded assembly angle."""
    knee_fold = KNEE_SIGN[prefix] * KNEE_ASSEMBLY_OFFSET
    targets = (
        (f"{prefix}3", HIP_SIGN[prefix] * h * sp),
        (f"{prefix}2", THIGH_SIGN[prefix] * th * sp),
        (f"{prefix}1", knee_fold + (KNEE_SIGN[prefix] * kn - knee_fold) * sp),
    )
    for name, val in targets:
        p.setJointMotorControl2(robot_id, joint_map[name], p.POSITION_CONTROL, val,
                                force=GLOBAL_MAX_TORQUE, maxVelocity=GLOBAL_MAX_VELOCITY)


# Zero-motion command so the standing pose needs no special case
cmd = {"freq": 1.0, "duty": 0.5, "amp_t": 0.0, "amp_r": 0.0, "sway": 0.0, "straight_mix": 1.0,
       "c": 1.0, "s": 0.0, "spin_dir": 1.0, "timing": "trot"}
sm = {k: cmd[k] for k in SMOOTH_KEYS}      # smoothed numeric parameters
sm_ready = False                           # first real command initialises sm directly
offsets = TROT_OFFSETS      # current leg timing

# Stand / sit + trim state
target_standing = START_STANDING
transition_progress = 1.0 if START_STANDING else 0.0
button_hold_time = 0.0
user_selected_height = NEUTRAL_HEIGHT
current_body_height = NEUTRAL_HEIGHT
manual_pitch_trim = 0.0     # rad, + = nose up
last_print = time.time()

idle = IdleLife()
idle_enabled = IDLE_ENABLED
auto_sleep_enabled = AUTO_SLEEP_ENABLED
last_activity = 0.0         # sim time of the last input
snapped_fwd = snapped_side = False
idle_roll_deg = idle_pitch_deg = idle_dz = 0.0
prev_walk = False
timing_key = "trot"
pending_blend_reset = False
blend_offset = {leg: np.zeros(3) for leg in LEG_ORDER}
last_cmd = {leg: None for leg in LEG_ORDER}
prev_button = False
prev_start = False
PARAM_ALPHA = 1.0 - math.exp(-DT / PARAM_SMOOTH_TAU)
BLEND_ALPHA = 1.0 - math.exp(-(SERVO_INTERVAL * DT) / GAIT_BLEND_TAU)

# ============================================================
# MAIN SIMULATION RUNTIME
# ============================================================
try:
    while True:
        for ev in pygame.event.get():
            if ev.type in HOTPLUG_EVENTS:
                gamepads.request_refresh()
        gs = gamepads.poll()

        # ---------------- inputs ----------------
        keys = p.getKeyboardEvents()
        raw_fwd, raw_side, raw_spin = -gs.ly, -gs.lx, gs.rx
        l1, r1, l2, r2 = gs.l1, gs.r1, gs.l2, gs.r2
        pressed, start_now = gs.a, gs.start

        kb_fwd, kb_side, kb_spin, kb_l1, kb_r1, kb_l2, kb_r2, kb_walk = keyboard_input(keys)
        if kb_fwd or kb_side:
            raw_fwd, raw_side = kb_fwd, kb_side
        if kb_spin:
            raw_spin = kb_spin
        l1, r1, l2, r2 = l1 or kb_l1, r1 or kb_r1, l2 or kb_l2, r2 or kb_r2
        pressed = pressed or kb_walk

        sim_time = physics_step * DT
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            target_standing = not target_standing
            last_activity = sim_time
            print(f"[state] {'STANDING' if target_standing else 'SITTING'}")
        if ord('i') in keys and keys[ord('i')] & p.KEY_WAS_TRIGGERED:
            idle_enabled = not idle_enabled
            print(f"[idle] {'ON' if idle_enabled else 'OFF'}")
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            auto_sleep_enabled = not auto_sleep_enabled
            print(f"[autosleep] {'ON' if auto_sleep_enabled else 'OFF'}")

        if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
            gamepad_debug = not gamepad_debug
            print(f"[gamepad] debug {'ON' if gamepad_debug else 'OFF'}")
        if ord('b') in keys and keys[ord('b')] & p.KEY_WAS_TRIGGERED:
            gamepads.toggle_backend()
        if gamepad_debug and time.time() - last_debug >= 0.25:
            last_debug = time.time()
            print(gamepads.debug_text())

        # Start button: single press toggles stand / sit
        if start_now and not prev_start:
            target_standing = not target_standing
            last_activity = sim_time
            print(f"[state] {'STANDING' if target_standing else 'SITTING'}")
        prev_start = start_now

        # Any stick or button counts as activity (resets the idle / auto-sleep timers)
        if (abs(raw_fwd) > JOYSTICK_DEADBAND or abs(raw_side) > JOYSTICK_DEADBAND or
                abs(raw_spin) > JOYSTICK_DEADBAND or l1 or r1 or l2 or r2 or pressed):
            last_activity = sim_time
        quiet = sim_time - last_activity

        # ---------------- stand / sit: hold L1 + R1 ----------------
        is_mid = ((target_standing and transition_progress < 1.0) or
                  ((not target_standing) and transition_progress > 0.0))
        real_now = time.perf_counter()
        real_dt = min(real_now - last_real, 0.1)
        last_real = real_now
        if l1 and r1 and not is_mid:
            button_hold_time += real_dt
            if button_hold_time >= REQUIRED_HOLD_DURATION:
                target_standing = not target_standing
                button_hold_time = 0.0
                print(f"[state] {'STANDING' if target_standing else 'SITTING'}")
        else:
            button_hold_time = 0.0

        is_fully_active = transition_progress >= 1.0

        # ---------------- height (L1 / R1) and pitch trim (L2 / R2) ----------------
        if is_fully_active and not (l1 and r1):
            if r1:
                user_selected_height += HEIGHT_SPEED * DT
            if l1:
                user_selected_height -= HEIGHT_SPEED * DT
            user_selected_height = float(np.clip(user_selected_height, MIN_HEIGHT, MAX_HEIGHT))

        if is_fully_active:
            trim_step = math.radians(PITCH_TRIM_SPEED_DEG_S) * DT
            trim_lim = math.radians(PITCH_TRIM_MAX_DEG)
            if r2:
                manual_pitch_trim = min(trim_lim, manual_pitch_trim + trim_step)
            if l2:
                manual_pitch_trim = max(-trim_lim, manual_pitch_trim - trim_step)

        # ---------------- auto-sleep ----------------
        if (auto_sleep_enabled and is_fully_active and target_standing and quiet >= AUTO_SLEEP_S):
            target_standing = False
            last_activity = sim_time
            print("[autosleep] inactive -- sitting down")

        # ---------------- stand / sit ramp ----------------
        if target_standing:
            transition_progress = min(1.0, transition_progress + DT / STARTUP_DURATION)
        else:
            transition_progress = max(0.0, transition_progress - DT / STARTUP_DURATION)
        smooth_progress = 0.5 - 0.5 * math.cos(math.pi * transition_progress)
        current_body_height += (user_selected_height - current_body_height) * HEIGHT_FOLLOW

        # ---------------- walking sticks: only once fully standing ----------------
        if is_fully_active:
            joy_fwd = raw_fwd if abs(raw_fwd) > JOYSTICK_DEADBAND else 0.0
            joy_side = raw_side if abs(raw_side) > JOYSTICK_DEADBAND else 0.0
            joy_spin = raw_spin if abs(raw_spin) > JOYSTICK_DEADBAND else 0.0

            if joy_fwd != 0.0 or joy_side != 0.0:
                ang = math.degrees(math.atan2(abs(joy_side), abs(joy_fwd)))
                snap_in, snap_out = AXIS_SNAP_DEG, AXIS_SNAP_DEG + AXIS_SNAP_HYST_DEG
                snapped_fwd = (ang < snap_out) if snapped_fwd else (ang < snap_in)
                snapped_side = (ang > 90.0 - snap_out) if snapped_side else (ang > 90.0 - snap_in)
                if snapped_fwd:
                    joy_side = 0.0
                elif snapped_side:
                    joy_fwd = 0.0
            else:
                snapped_fwd = snapped_side = False
            joy_side *= SIDE_SCALE

            # Button A: toggle trot <-> walk for straight / backward walking
            if pressed and not prev_button:
                STRAIGHT_GAIT = "walk" if STRAIGHT_GAIT == "trot" else "trot"
                print(f"Straight/backward gait -> {STRAIGHT_GAIT}")
        else:
            joy_fwd = joy_side = joy_spin = 0.0
        prev_button = pressed

        active = math.hypot(joy_fwd, joy_side) > 0.001 or abs(joy_spin) > 0.001

        # Idle "alive" motion (only when standing and nothing has been touched for IDLE_START_S)
        idle_allowed = idle_enabled and is_fully_active and quiet >= IDLE_START_S
        idle_roll_deg, idle_pitch_deg, idle_dz = idle.update(
            DT, activity=max(min(1.0, math.hypot(joy_fwd, joy_side)), abs(joy_spin)),
            allowed=idle_allowed)

        # Refresh the mixed command at servo rate while active (held while stopping so it blends out)
        if physics_step % SERVO_INTERVAL == 0 and active:
            cmd = build_command(joy_fwd, joy_side, joy_spin, prev_walk)
            prev_walk = cmd["timing"] == "walk"
            if not sm_ready:
                sm = {k: cmd[k] for k in SMOOTH_KEYS}
                sm_ready = True
            if cmd["timing"] != timing_key:
                timing_key = cmd["timing"]
                offsets = GAIT_LIBRARY[timing_key]["offsets"]
                pending_blend_reset = True

        if active:
            for k in SMOOTH_KEYS:
                sm[k] += (cmd[k] - sm[k]) * PARAM_ALPHA
            phase_accumulator = (phase_accumulator + sm["freq"] * DT) % 1.0
            motion_blend += (1.0 - motion_blend) * STOP_BLEND_SPEED
        else:
            motion_blend += (0.0 - motion_blend) * STOP_BLEND_SPEED

        # --- Optional IMU stabilizer ---
        if ENABLE_STABILIZATION and active:
            _, base_orn = p.getBasePositionAndOrientation(robot_id)
            _, world_ang_vel = p.getBaseVelocity(robot_id)
            inv_rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3).T
            local_ang_vel = np.dot(inv_rot, np.array(world_ang_vel))
            roll_rate, pitch_rate = local_ang_vel[0], local_ang_vel[1]
            q_inv = [-base_orn[0], -base_orn[1], -base_orn[2], base_orn[3]]
            g_local = p.multiplyTransforms([0, 0, 0], q_inv, [0, 0, -1], [0, 0, 0, 1])[0]
            roll = math.atan2(g_local[1], -g_local[2])
            pitch = math.atan2(-g_local[0], math.sqrt(g_local[1]**2 + g_local[2]**2))
        else:
            roll = pitch = roll_rate = pitch_rate = 0.0

        # ========================================================
        # LEG COMMANDS (servo rate)
        # ========================================================
        if physics_step % SERVO_INTERVAL == 0:
            blend = motion_blend
            lift = STEP_HEIGHT * blend
            duty = min(max(sm["duty"], 0.3), 0.9)
            mix_q = sm["straight_mix"]        # quintic vs cycloid shape weight
            sway = blend * sm["sway"] * math.sin(2.0 * math.pi * phase_accumulator)
            c, s = cmd["c"], cmd["s"]
            sp = smooth_progress              # 0 = sitting pose, 1 = standing

            for idx, leg in enumerate(LEG_ORDER):
                rx, ry = LEG_ROOTS[leg]
                phase = (phase_accumulator + offsets[idx]) % 1.0
                pm_q, pm_c, lift_q, lift_c = leg_profiles(phase, duty)

                pm_t = mix_q * pm_q + (1.0 - mix_q) * pm_c
                lift_prof = mix_q * lift_q + (1.0 - mix_q) * lift_c

                if LIFT_MODE == "up":
                    dir_sign = FOOT_X_SIGN[leg]          # -1: body forward = -pm on x
                    lift_term = -LIFT_SIGN[leg] * lift * lift_prof
                else:
                    dir_sign = -FOOT_X_SIGN[leg]
                    lift_term = lift * lift_prof

                # Translation: one vector (c, s) = (forward, left) drives every direction
                tx = dir_sign * sm["amp_t"] * blend * c * pm_t
                ty = dir_sign * sm["amp_t"] * blend * s * pm_t + sway * HIP_SIDE_SIGN[leg[-1]]

                # Spin: signed tangential motion about the body centre, same sign rule
                angle = dir_sign * sm["amp_r"] * blend * pm_c
                tx += -ry * angle * cmd["spin_dir"]
                ty += rx * angle * cmd["spin_dir"]

                # Body height + pitch trim. tz is the hip-to-foot distance, so a larger tz pushes the foot down: with + pitch (nose up) the front feet (rx > 0) reach further down.
                total_pitch = manual_pitch_trim + math.radians(idle_pitch_deg)
                total_roll = math.radians(idle_roll_deg)
                tz = (current_body_height + lift_term
                      + rx * math.sin(total_pitch) - ry * math.sin(total_roll) - idle_dz)

                if ENABLE_STABILIZATION and active:
                    is_front = 1.0 if leg.startswith("F") else -1.0
                    is_left = 1.0 if leg.endswith("L") else -1.0
                    roll_err = KP_ROLL * roll + KD_ROLL * roll_rate
                    pitch_err = KP_PITCH * pitch + KD_PITCH * pitch_rate
                    is_swing = phase < (1.0 - duty)
                    tz += ((-is_left * roll_err) + (is_front * pitch_err)) * (0.05 if is_swing else 1.0)

                # Leg-timing change (trot <-> walk): fade the gap between old and new foot position
                target = np.array([tx, ty, tz])
                if pending_blend_reset and last_cmd[leg] is not None:
                    blend_offset[leg] = last_cmd[leg] - target
                blend_offset[leg] *= (1.0 - BLEND_ALPHA)
                target = target + blend_offset[leg]
                last_cmd[leg] = target.copy()

                # Sit pose: every foot target collapses toward SIT_POSE_HEIGHT as sp -> 0
                z_pose = SIT_POSE_HEIGHT + (target[2] - SIT_POSE_HEIGHT) * sp
                h, th, kn = solve_leg_ik_3dof(target[0], target[1], z_pose)
                command_leg_blend(robot_id, joint_map, leg, h, th, kn, sp)

            pending_blend_reset = False

        # ---------------- telemetry ----------------
        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            state = ("STANDING" if transition_progress >= 1.0 else
                     "SITTING" if transition_progress <= 0.0 else "SHIFTING")
            snap = "FWD" if snapped_fwd else ("SIDE" if snapped_side else "-")
            print(f"[{state:8s}] height {current_body_height:.3f} m  "
                  f"pitch trim {math.degrees(manual_pitch_trim):+5.1f} deg  "
                  f"gait {STRAIGHT_GAIT}  snap {snap:4s}  idle {idle.envelope:.2f}  quiet {quiet:5.1f}s")

        p.stepSimulation()
        physics_step += 1
        time.sleep(DT)

except KeyboardInterrupt:
    p.disconnect()
