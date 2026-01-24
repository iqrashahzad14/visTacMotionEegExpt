"""
Combined PsychoPy experiment: Visual + Tactile (Hand Up / Hand Down) repeated-direction detection

UPGRADES IMPLEMENTED:
1) Next-trial buffer preloading during ITI (tactile blocks):
   - During ITI of trial t, upload seg1+seg2 buffers for trial t+1
   - At segment onset, only call play() (cheap)
2) ctypes preallocation per direction:
   - Build ctypes arrays once for each direction, reuse every time
   - Avoid per-trial ctypes allocation/copy overhead
3) Tactile-block warmup:
   - Before first tactile trial, do a short warmup upload+play to avoid cold-start latency

Everything else unchanged from your working combined script:
- block order depends on subject
- same design, timing frame-locked
- logging includes segment onsets + dropped frames per segment
- response allowed through ITI; ITI press attributed to preceding trial
"""

from psychopy import visual, core, event, gui, data
from psychopy.hardware import keyboard
import numpy as np
import random
import os
import csv
import threading
import ctypes

event.globalKeys.add(key="escape", func=core.quit, name="shutdown")

# =========================
# PARAMETER BLOCK (edit me)
# =========================
PARAMS = {
    "nBlocks": 4,
    "nTrials": 6,

    "segmentDuration": 0.5,
    "isi": 0.05,
    "iti": 1.0,

    "targets_per_block": [1, 2, 1, 2],
    "target_min": 0,
    "target_max": 2,
    "target_balance": True,

    "response_key": "a",

    "alternate_axis_within_block": True,
    "alternate_start_axis": "random",

    "vis_axis_dirs": {"horizontal": ["left", "right"], "vertical": ["up", "down"]},
    "vis_dir_deg": {"right": 0.0, "left": 180.0, "up": 90.0, "down": -90.0},

    "tac_axis_dirs": {
        "pinkyThumb": ["pinky_to_thumb", "thumb_to_pinky"],
        "fingerWrist": ["wrist_to_finger", "finger_to_wrist"]
    },

    "use_external_monitor": False,
    "screen_index_laptop": 0,
    "screen_index_external": 1,
    "win_size": [1536, 960],
    "fullscr": True,
    "monitor": "testMonitor",
    "bg_color": [0, 0, 0],

    "nDots": 50,
    "dotSize": 20,
    "dotSpeed": 4,
    "fieldSize": (800, 800),
    "fieldShape": "circle",
    "dotLife_ms": 200,

    # Haptics
    "fs_hz": 40_000.0,
    "line_length_m": 0.05,
    "scrub_rate_hz": 100.0,
    "drift_distance_m": 0.10,
    "z_height_m": 0.10,
    "intensity_on": 1.0,
    "simulate_only": False,
    "lib_name": "libStreaming_CachedPoint_ctypes.so",

    "flush_at_block_end": True,

    # NEW: tactile warmup parameters
    "tactile_warmup_wait_s": 0.05,
}

# =========================
# Design helpers
# =========================
def validate_params(p: dict) -> dict:
    p = dict(p)

    if p["nBlocks"] < 1 or p["nTrials"] < 1:
        raise ValueError("nBlocks and nTrials must be >= 1")

    if p["segmentDuration"] <= 0:
        raise ValueError("segmentDuration must be > 0")
    if p["isi"] < 0:
        raise ValueError("isi must be >= 0")
    if p["iti"] < 0:
        raise ValueError("iti must be >= 0")

    if p["targets_per_block"] is not None:
        if len(p["targets_per_block"]) != p["nBlocks"]:
            raise ValueError("targets_per_block must have length nBlocks (blocks per modality).")
        for k in p["targets_per_block"]:
            if not (0 <= k <= p["nTrials"]):
                raise ValueError("Each targets_per_block value must be between 0 and nTrials.")

    if p["target_max"] > p["nTrials"]:
        p["target_max"] = p["nTrials"]

    for axis, dirs in p["vis_axis_dirs"].items():
        if len(dirs) != 2:
            raise ValueError(f"vis_axis_dirs['{axis}'] must have exactly 2 directions.")
        for d in dirs:
            if d not in p["vis_dir_deg"]:
                raise ValueError(f"Missing vis_dir_deg for direction '{d}'")

    for axis, dirs in p["tac_axis_dirs"].items():
        if len(dirs) != 2:
            raise ValueError(f"tac_axis_dirs['{axis}'] must have exactly 2 directions.")

    if p.get("dotLife_ms", 0) <= 0:
        raise ValueError("dotLife_ms must be > 0")

    return p

def make_targets_per_block(p: dict, rng: random.Random):
    if p["targets_per_block"] is not None:
        return list(p["targets_per_block"])

    tmin, tmax = p["target_min"], p["target_max"]
    if tmin < 0 or tmax < tmin:
        raise ValueError("Invalid target_min/target_max")

    if not p["target_balance"]:
        return [rng.randint(tmin, tmax) for _ in range(p["nBlocks"])]

    counts = list(range(tmin, tmax + 1)) or [0]
    out = []
    while len(out) < p["nBlocks"]:
        out.extend(counts)
    out = out[:p["nBlocks"]]
    rng.shuffle(out)
    return out

def axis_sequence_for_block(nTrials: int, axes: list, alternate: bool, start_axis: str, rng: random.Random):
    if not alternate:
        return [rng.choice(axes) for _ in range(nTrials)]

    if start_axis == "random":
        start = rng.choice(axes)
    else:
        start = rng.choice(axes) if start_axis not in axes else start_axis

    other = axes[1] if start == axes[0] else axes[0]
    return [start if i % 2 == 0 else other for i in range(nTrials)]

def create_block_trials(p: dict, block_type: str, block_index_within_type: int, rng: random.Random, targets_per_block: list):
    nTrials = p["nTrials"]
    n_targets = targets_per_block[block_index_within_type]

    eligible = list(range(1, nTrials - 1))
    target_idxs = set(rng.sample(eligible, k=n_targets)) if n_targets > 0 else set()

    if block_type == "visual":
        axes = ["horizontal", "vertical"]
        axis_dirs = p["vis_axis_dirs"]
        axes_seq = axis_sequence_for_block(
            nTrials=nTrials,
            axes=axes,
            alternate=p["alternate_axis_within_block"],
            start_axis=p["alternate_start_axis"],
            rng=rng,
        )
        trials = []
        for t in range(nTrials):
            axis = axes_seq[t]
            dirs = axis_dirs[axis]
            dir1 = rng.choice(dirs)
            is_target = 1 if t in target_idxs else 0
            dir2 = dir1 if is_target else (dirs[0] if dir1 == dirs[1] else dirs[1])
            trials.append({
                "axis": axis,
                "dir1_name": dir1,
                "dir2_name": dir2,
                "dir1_deg": p["vis_dir_deg"][dir1],
                "dir2_deg": p["vis_dir_deg"][dir2],
                "is_target": is_target
            })
        return trials

    axes = ["pinkyThumb", "fingerWrist"]
    axis_dirs = p["tac_axis_dirs"]
    axes_seq = axis_sequence_for_block(
        nTrials=nTrials,
        axes=axes,
        alternate=p["alternate_axis_within_block"],
        start_axis=p["alternate_start_axis"],
        rng=rng,
    )

    trials = []
    for t in range(nTrials):
        axis = axes_seq[t]
        dirs = axis_dirs[axis]
        dir1 = rng.choice(dirs)
        is_target = 1 if t in target_idxs else 0
        dir2 = dir1 if is_target else (dirs[0] if dir1 == dirs[1] else dirs[1])
        trials.append({
            "axis": axis,
            "dir1_name": dir1,
            "dir2_name": dir2,
            "is_target": is_target
        })
    return trials

def outcome_label(is_target: int, key_pressed: str, target_key: str) -> str:
    pressed = (key_pressed != "")
    if is_target:
        return "HIT" if (pressed and key_pressed == target_key) else "MISS"
    return "FALSE_ALARM" if pressed else "CORRECT_REJECT"

# =========================
# Haptics: buffer builder + device wrapper
# =========================
def build_haptic_buffer(direction, fs_hz, stim_duration_s, line_length_m, scrub_rate_hz,
                        drift_distance_m, z_height_m, intensity_on):
    t = np.arange(0, stim_duration_s, 1.0 / fs_hz)

    scrub_speed_mps = line_length_m * scrub_rate_hz
    tmp = np.mod(t * scrub_speed_mps, 2.0 * line_length_m)
    pos_along_line = np.where(tmp <= line_length_m, tmp, 2.0 * line_length_m - tmp) - (line_length_m / 2.0)

    drift_speed_mps = drift_distance_m / stim_duration_s
    pos_drift = np.mod(t * drift_speed_mps, drift_distance_m) - (drift_distance_m / 2.0)

    if direction == "wrist_to_finger":
        x = pos_along_line; y = pos_drift
    elif direction == "finger_to_wrist":
        x = pos_along_line; y = -pos_drift
    elif direction == "pinky_to_thumb":
        x = pos_drift; y = pos_along_line
    elif direction == "thumb_to_pinky":
        x = -pos_drift; y = pos_along_line
    else:
        x = np.zeros_like(t); y = np.zeros_like(t)

    z = np.full_like(t, z_height_m)
    i = np.full_like(t, intensity_on)
    return x, y, z, i

class HapticsDevice:
    """
    Updated: supports preallocated ctypes buffers.

    - You can register (x,y,z,i) ctypes arrays per direction once.
    - Then upload_by_name(direction) uses those preallocated arrays (no allocation/copy).
    """
    def __init__(self, lib_name: str, simulate_only: bool):
        self.simulate_only = simulate_only
        self.lib_name = lib_name
        self.prg = None
        self.thread = None

        # direction -> dict of ctypes arrays + n
        self.ctypes_buffers = {}

    def start(self):
        if self.simulate_only:
            return
        self.prg = ctypes.cdll.LoadLibrary(self.lib_name)
        self.thread = threading.Thread(target=self.prg.start_array, args=(), daemon=True)
        self.thread.start()
        core.wait(0.05)

    def stop(self):
        if self.simulate_only:
            return
        try:
            self.prg.stop_array()
        except Exception as e:
            print(f"[WARN] stop_array failed: {e}")

    def register_direction_buffer(self, direction: str, x_np, y_np, z_np, i_np):
        """Create and store ctypes arrays once for a direction."""
        if self.simulate_only:
            self.ctypes_buffers[direction] = {"n": int(len(x_np))}
            return

        n = int(len(x_np))
        arr_t = ctypes.c_double * n

        # Convert once to Python lists to avoid numpy scalar overhead in the *args expansion
        x_list = x_np.tolist()
        y_list = y_np.tolist()
        z_list = z_np.tolist()
        i_list = i_np.tolist()

        self.ctypes_buffers[direction] = {
            "n": n,
            "xc": arr_t(*x_list),
            "yc": arr_t(*y_list),
            "zc": arr_t(*z_list),
            "ic": arr_t(*i_list),
        }

    def upload_by_name(self, direction: str):
        """Upload pre-registered direction buffer to device."""
        if self.simulate_only:
            return
        buf = self.ctypes_buffers[direction]
        n = buf["n"]
        self.prg.set_positions(buf["xc"], buf["yc"], buf["zc"], n)
        self.prg.set_intensities(buf["ic"], n)

    def play(self, idx=0):
        if self.simulate_only:
            return
        self.prg.play_sensation(idx)

# =========================
# Participant GUI
# =========================
PARAMS = validate_params(PARAMS)

info = {"subNb": "", "sessionNb": ""}
dlg = gui.DlgFromDict(info, order=["subNb", "sessionNb"])
if not dlg.OK:
    core.quit()

sub = int(info["subNb"]) if info["subNb"].strip() else 0
ses = int(info["sessionNb"]) if info["sessionNb"].strip() else 0

# =========================
# Paths / TSV name
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

tsv_name = (
    f"sub-{sub:03d}_ses-{ses:03d}_"
    f"task-visTacRepeatDirection_desc-events_"
    f"{data.getDateStr(format='%Y-%m-%d-%H%M')}.tsv"
)
tsv_path = os.path.join(data_dir, tsv_name)

# =========================
# Build block sequence with subject-dependent start
# =========================
cycle = ["visual", "tac_up", "tac_down"]
base_seq = cycle * PARAMS["nBlocks"]

start_idx = (sub - 1) % 3
block_sequence = base_seq[start_idx:] + base_seq[:start_idx]

block_idx_within = {"visual": 0, "tac_up": 0, "tac_down": 0}

rng = random.Random(None)
targets_per_block = make_targets_per_block(PARAMS, rng)

# =========================
# Window & timing
# =========================
screen_index = PARAMS["screen_index_external"] if PARAMS["use_external_monitor"] else PARAMS["screen_index_laptop"]
try:
    win = visual.Window(
        size=PARAMS["win_size"],
        fullscr=PARAMS["fullscr"],
        screen=screen_index,
        winType="pyglet",
        allowGUI=False,
        monitor=PARAMS["monitor"],
        color=PARAMS["bg_color"],
        colorSpace="rgb"
    )
except Exception as e:
    print(f"[WARN] Could not open window on screen={screen_index}. Falling back to screen=0. Error:\n{e}")
    win = visual.Window(
        size=PARAMS["win_size"],
        fullscr=PARAMS["fullscr"],
        screen=0,
        winType="pyglet",
        allowGUI=False,
        monitor=PARAMS["monitor"],
        color=PARAMS["bg_color"],
        colorSpace="rgb"
    )

win.recordFrameIntervals = True
measured_hz = win.getActualFrameRate(nIdentical=20, nMaxFrames=200, nWarmUpFrames=20, threshold=1.0)
if measured_hz is None or measured_hz <= 0:
    measured_hz = 60.0
frame_dur = 1.0 / float(measured_hz)

seg_frames = max(1, int(round(PARAMS["segmentDuration"] / frame_dur)))
isi_frames = max(0, int(round(PARAMS["isi"] / frame_dur)))
iti_frames = max(0, int(round(PARAMS["iti"] / frame_dur)))

seg_ach = seg_frames * frame_dur
isi_ach = isi_frames * frame_dur
iti_ach = iti_frames * frame_dur
trial_ach = (2 * seg_frames + isi_frames + iti_frames) * frame_dur

dotlife_frames = max(1, int(round((PARAMS["dotLife_ms"] / 1000.0) / frame_dur)))

# =========================
# Stimuli
# =========================
fixCross = visual.ShapeStim(
    win, vertices="cross", units="pix", size=(10, 10),
    lineWidth=1, lineColor=[1, 1, 1], fillColor=[1, 1, 1],
    colorSpace="rgb", pos=(0, 0)
)

rdk = visual.DotStim(
    win,
    units="pix",
    nDots=PARAMS["nDots"],
    dotSize=PARAMS["dotSize"],
    speed=PARAMS["dotSpeed"],
    dir=0.0,
    coherence=1.0,
    fieldPos=(0.0, 0.0),
    fieldSize=PARAMS["fieldSize"],
    fieldShape=PARAMS["fieldShape"],
    signalDots="same",
    noiseDots="position",
    dotLife=dotlife_frames,
    color=[1.0, 1.0, 1.0],
    colorSpace="rgb"
)

instr = visual.TextStim(
    win,
    text=(
        "Repeated direction detection\n\n"
        "Each trial has TWO segments.\n"
        "TARGET = the direction REPEATS in segment 2.\n\n"
        "Press 'a' ONLY when there is a TARGET.\n\n"
        "Press SPACE to start."
    ),
    height=0.06, color=[1, 1, 1], wrapWidth=1.4
)

block_msg = visual.TextStim(win, text="", height=0.06, color=[1, 1, 1], wrapWidth=1.4)
blank = visual.TextStim(win, text="")

kb = keyboard.Keyboard()
globalClock = core.Clock()

# =========================
# Haptics device + buffers + ctypes preallocation
# =========================
hdev = HapticsDevice(lib_name=PARAMS["lib_name"], simulate_only=PARAMS["simulate_only"])
hdev.start()

# Precompute tactile buffers as numpy
dir_buffers_np = {}
for d in ["wrist_to_finger", "finger_to_wrist", "pinky_to_thumb", "thumb_to_pinky", "none"]:
    dir_buffers_np[d] = build_haptic_buffer(
        direction=d,
        fs_hz=PARAMS["fs_hz"],
        stim_duration_s=seg_ach,
        line_length_m=PARAMS["line_length_m"],
        scrub_rate_hz=PARAMS["scrub_rate_hz"],
        drift_distance_m=PARAMS["drift_distance_m"],
        z_height_m=PARAMS["z_height_m"],
        intensity_on=PARAMS["intensity_on"],
    )

# Register ctypes arrays ONCE per direction
for d, (x, y, z, i) in dir_buffers_np.items():
    hdev.register_direction_buffer(d, x, y, z, i)

# =========================
# Helper: tactile warmup
# =========================
def tactile_block_warmup():
    """
    Warm up the device/library before first tactile trial of a block.
    Keeps it lightweight but helps avoid first-trial timing hit.
    """
    if PARAMS["simulate_only"]:
        return
    # upload "none" once and play twice like seg1/seg2
    try:
        hdev.upload_by_name("none")
        core.wait(PARAMS["tactile_warmup_wait_s"])
        hdev.play(0)
        core.wait(PARAMS["tactile_warmup_wait_s"])
        hdev.play(0)
        core.wait(PARAMS["tactile_warmup_wait_s"])
    except Exception as e:
        print(f"[WARN] tactile warmup failed: {e}")

# =========================
# TSV logging
# =========================
with open(tsv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "subNb","sessionNb",
        "globalBlockNb","blockType","blockIndexWithinType","trialNb",
        "trialOnset","trialDuration",
        "segment1Onset","segment2Onset",
        "segmentDurationAchieved","isiAchieved","itiAchieved",
        "segmentFrames","isiFrames","itiFrames",
        "axis","dir1","dir2","isTarget",
        "keyPressed","RT","outcome",
        "refreshHz","dotLifeFrames","simulateOnly",
        "droppedFramesSeg1","droppedFramesISI","droppedFramesSeg2","droppedFramesTrial",
    ])
    f.flush()

    instr.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    total_blocks = len(block_sequence)

    for gblock_idx, block_type in enumerate(block_sequence, start=1):
        block_i = block_idx_within[block_type]

        trials = create_block_trials(
            p=PARAMS,
            block_type=block_type,
            block_index_within_type=block_i,
            rng=rng,
            targets_per_block=targets_per_block
        )

        # Block text
        if block_type == "visual":
            block_label = f"Visual block ({block_i+1}/{PARAMS['nBlocks']})"
            reminder = "Press 'a' when direction repeats (visual)."
        elif block_type == "tac_up":
            block_label = f"Tactile HAND UP block ({block_i+1}/{PARAMS['nBlocks']})"
            reminder = "Press 'a' when direction repeats (tactile hand up)."
        else:
            block_label = f"Tactile HAND DOWN block ({block_i+1}/{PARAMS['nBlocks']})"
            reminder = "Press 'a' when direction repeats (tactile hand down)."

        block_msg.setText(
            f"{block_label}\n"
            f"Global block {gblock_idx}/{total_blocks}\n\n"
            "Each trial has TWO segments.\n"
            "TARGET = direction repeats in segment 2.\n\n"
            f"{reminder}\n\n"
            "Press SPACE to begin."
        )
        block_msg.draw()
        win.flip()
        event.waitKeys(keyList=["space"])

        # NEW: warmup at tactile block start (after instruction screen)
        if block_type in ("tac_up", "tac_down"):
            tactile_block_warmup()

        # -----------------------------------------
        # Preload buffers for FIRST tactile trial of this block (before timing starts)
        # -----------------------------------------
        if block_type in ("tac_up", "tac_down"):
            # preload trial 1 buffers now, so seg1 onset is cheap
            first = trials[0]
            hdev.upload_by_name(first["dir1_name"])
            # store "next seg2 buffer" name to load later quickly
            # We still need seg2 uploaded before seg2 onset; we'll do that during ISI (see below).
            # But we CAN also preload seg2 here already, if desired:
            # (We will actually preload seg2 during ISI, to keep device state predictable.)
        # -----------------------------------------

        for t_idx, tr in enumerate(trials, start=1):
            is_tactile = block_type in ("tac_up", "tac_down")

            drop_trial_start = getattr(win, "nDroppedFrames", 0)

            kb.clearEvents()
            key_pressed = ""
            rt = ""

            seg1_onset_box = {"t": None}
            seg2_onset_box = {"t": None}

            def _mark_seg1_onset():
                seg1_onset_box["t"] = globalClock.getTime()

            def _mark_seg2_onset():
                seg2_onset_box["t"] = globalClock.getTime()

            # ============================================================
            # SEGMENT 1
            # ============================================================
            if block_type == "visual":
                rdk.dir = tr["dir1_deg"]
                fixCross.draw(); rdk.draw()
                win.callOnFlip(_mark_seg1_onset)
                win.callOnFlip(kb.clock.reset)
                win.flip()

                for _ in range(max(0, seg_frames - 1)):
                    fixCross.draw(); rdk.draw()
                    win.flip()
                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt

            else:
                # tactile seg1: buffer should already be uploaded (preloaded either before trial or during previous ITI)
                fixCross.draw()
                win.callOnFlip(_mark_seg1_onset)
                win.callOnFlip(kb.clock.reset)
                win.callOnFlip(hdev.play, 0)
                win.flip()

                for _ in range(max(0, seg_frames - 1)):
                    fixCross.draw()
                    win.flip()
                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt

            drop_after_seg1 = getattr(win, "nDroppedFrames", 0)
            dropped_seg1 = max(0, drop_after_seg1 - drop_trial_start)

            # ============================================================
            # ISI
            # ============================================================
            # NEW: for tactile, upload seg2 buffer during ISI (not at seg2 onset)
            if is_tactile:
                hdev.upload_by_name(tr["dir2_name"])

            for _ in range(isi_frames):
                fixCross.draw()
                win.flip()
                if key_pressed == "":
                    keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name
                        rt = keys[0].rt

            drop_after_isi = getattr(win, "nDroppedFrames", 0)
            dropped_isi = max(0, drop_after_isi - drop_after_seg1)

            # ============================================================
            # SEGMENT 2
            # ============================================================
            if block_type == "visual":
                rdk.dir = tr["dir2_deg"]
                fixCross.draw(); rdk.draw()
                win.callOnFlip(_mark_seg2_onset)
                win.flip()

                for _ in range(max(0, seg_frames - 1)):
                    fixCross.draw(); rdk.draw()
                    win.flip()
                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt

            else:
                # tactile seg2: buffer already uploaded during ISI, so onset is cheap
                fixCross.draw()
                win.callOnFlip(_mark_seg2_onset)
                win.callOnFlip(hdev.play, 0)
                win.flip()

                for _ in range(max(0, seg_frames - 1)):
                    fixCross.draw()
                    win.flip()
                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt

            drop_after_seg2 = getattr(win, "nDroppedFrames", 0)
            dropped_seg2 = max(0, drop_after_seg2 - drop_after_isi)

            # ============================================================
            # ITI (blank) — response allowed here too
            # NEW: preload NEXT TRIAL seg1+seg2 buffers during ITI (tactile only)
            # ============================================================
            # Determine next trial (if any) for preloading
            next_tr = trials[t_idx] if (is_tactile and t_idx < len(trials)) else None

            for frame_i in range(iti_frames):
                blank.draw()
                win.flip()

                if key_pressed == "":
                    keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name
                        rt = keys[0].rt

                # Preload once early in ITI to avoid spreading IO
                if is_tactile and next_tr is not None and frame_i == 0:
                    # preload seg1 buffer for next trial immediately
                    hdev.upload_by_name(next_tr["dir1_name"])
                    # (seg2 will be uploaded during next trial's ISI)
                    # If you want, you could also preload seg2 here, but that can overwrite seg1.
                    # So we keep: preload seg1 now, preload seg2 later (during ISI).

            drop_after_iti = getattr(win, "nDroppedFrames", 0)
            dropped_trial = max(0, drop_after_iti - drop_trial_start)

            seg1_onset = seg1_onset_box["t"] if seg1_onset_box["t"] is not None else globalClock.getTime()
            seg2_onset = seg2_onset_box["t"] if seg2_onset_box["t"] is not None else ""

            axis = tr["axis"]
            dir1_name = tr["dir1_name"]
            dir2_name = tr["dir2_name"]
            is_target = tr["is_target"]

            out = outcome_label(is_target, key_pressed, PARAMS["response_key"])

            dotLifeFrames_out = dotlife_frames if block_type == "visual" else ""
            simulate_only_out = int(PARAMS["simulate_only"]) if block_type != "visual" else ""

            writer.writerow([
                info["subNb"], info["sessionNb"],
                gblock_idx, block_type, (block_i + 1), t_idx,
                f"{seg1_onset:.4f}", f"{trial_ach:.4f}",
                f"{seg1_onset:.4f}",
                (f"{seg2_onset:.4f}" if seg2_onset != "" else ""),
                f"{seg_ach:.4f}", f"{isi_ach:.4f}", f"{iti_ach:.4f}",
                seg_frames, isi_frames, iti_frames,
                axis, dir1_name, dir2_name, is_target,
                key_pressed, (f"{rt:.4f}" if rt != "" else ""),
                out,
                f"{measured_hz:.3f}", dotLifeFrames_out, simulate_only_out,
                dropped_seg1, dropped_isi, dropped_seg2, dropped_trial,
            ])

        block_idx_within[block_type] += 1

        if PARAMS.get("flush_at_block_end", True):
            f.flush()

# Cleanup
hdev.stop()
win.close()
core.quit()
