"""
TACTILE repeated-direction detection task
Adds:
- segment1Onset + segment2Onset (true onsets via callOnFlip)
- dropped frames per phase: Seg1 / ISI / Seg2 + total trial
- flush-to-disk only at end of each block (no fsync inside timing loops)

Design/timing remain frame-locked.
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

params = {
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

    "axis_dirs": {
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

    "fs_hz": 40_000.0,
    "line_length_m": 0.05,
    "scrub_rate_hz": 100.0,
    "drift_distance_m": 0.10,
    "z_height_m": 0.10,
    "intensity_on": 1.0,

    "simulate_only": False,
    "lib_name": "libStreaming_CachedPoint_ctypes.so",

    # NEW: reduce disk IO jitter
    "flush_at_block_end": True,
}

# ----------------------------
# Helpers: design generation
# ----------------------------
def validate_params(p: dict) -> dict:
    p = dict(p)
    if p["alternate_start_axis"] not in ["pinkyThumb", "fingerWrist", "random"]:
        raise ValueError("alternate_start_axis must be 'pinkyThumb', 'fingerWrist', or 'random'")
    return p

def make_targets_per_block(p: dict, rng: random.Random):
    if p["targets_per_block"] is not None:
        return list(p["targets_per_block"])
    tmin, tmax = p["target_min"], min(p["target_max"], p["nTrials"])
    return [rng.randint(tmin, tmax) for _ in range(p["nBlocks"])]

def axis_sequence_for_block(p: dict, rng: random.Random):
    if not p["alternate_axis_within_block"]:
        return [rng.choice(["pinkyThumb", "fingerWrist"]) for _ in range(p["nTrials"])]

    if p["alternate_start_axis"] == "random":
        start = rng.choice(["pinkyThumb", "fingerWrist"])
    else:
        start = p["alternate_start_axis"]

    other = "fingerWrist" if start == "pinkyThumb" else "pinkyThumb"
    return [start if i % 2 == 0 else other for i in range(p["nTrials"])]

def create_design(p: dict, seed=None):
    p = validate_params(p)
    rng = random.Random(seed)
    targets_per_block = make_targets_per_block(p, rng)

    blocks = []
    for b in range(p["nBlocks"]):
        n_targets = targets_per_block[b]
        eligible = list(range(1, p["nTrials"] - 1))
        target_idxs = set(rng.sample(eligible, k=n_targets)) if n_targets > 0 else set()

        axes = axis_sequence_for_block(p, rng)
        block_trials = []

        for t in range(p["nTrials"]):
            axis = axes[t]
            dirs = p["axis_dirs"][axis]
            dir1 = rng.choice(dirs)
            is_target = 1 if t in target_idxs else 0
            dir2 = dir1 if is_target else (dirs[0] if dir1 == dirs[1] else dirs[1])

            block_trials.append({
                "axis": axis,
                "dir1_name": dir1,
                "dir2_name": dir2,
                "is_target": is_target
            })
        blocks.append(block_trials)

    return p, blocks, targets_per_block

def outcome_label(is_target: int, key_pressed: str, target_key: str) -> str:
    pressed = (key_pressed != "")
    if is_target:
        return "HIT" if (pressed and key_pressed == target_key) else "MISS"
    return "FALSE_ALARM" if pressed else "CORRECT_REJECT"

# ----------------------------
# HAPTICS: buffer builder
# ----------------------------
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

# ----------------------------
# HAPTICS: device wrapper
# ----------------------------
class HapticsDevice:
    def __init__(self, lib_name: str, simulate_only: bool):
        self.simulate_only = simulate_only
        self.lib_name = lib_name
        self.prg = None
        self.thread = None

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

    def upload_buffer(self, x, y, z, i):
        if self.simulate_only:
            return
        n = len(x)
        arr_t = ctypes.c_double * n
        xc = arr_t(*x); yc = arr_t(*y); zc = arr_t(*z); ic = arr_t(*i)
        self.prg.set_positions(xc, yc, zc, n)
        self.prg.set_intensities(ic, n)

    def play(self, idx=0):
        if self.simulate_only:
            return
        self.prg.play_sensation(idx)

# ----------------------------
# GUI
# ----------------------------
info = {"subNb": "", "sessionNb": ""}
dlg = gui.DlgFromDict(info, order=["subNb", "sessionNb"])
if not dlg.OK:
    core.quit()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

sub = int(info["subNb"]) if info["subNb"].strip() else 0
ses = int(info["sessionNb"]) if info["sessionNb"].strip() else 0
tsv_name = (
    f"sub-{sub:03d}_ses-{ses:03d}_"
    f"task-tacMotionDirRep_desc-events_"
    f"{data.getDateStr(format='%Y-%m-%d-%H%M')}.tsv"
)
tsv_path = os.path.join(data_dir, tsv_name)

# Design
params, design, _ = create_design(params, seed=None)

# Window
screen_index = params["screen_index_external"] if params["use_external_monitor"] else params["screen_index_laptop"]
win = visual.Window(
    size=params["win_size"],
    fullscr=params["fullscr"],
    screen=screen_index,
    winType="pyglet",
    allowGUI=False,
    monitor=params["monitor"],
    color=params["bg_color"],
    colorSpace="rgb"
)

# Timing
win.recordFrameIntervals = True
measured_hz = win.getActualFrameRate(nIdentical=20, nMaxFrames=200, nWarmUpFrames=20, threshold=1.0)
if measured_hz is None or measured_hz <= 0:
    measured_hz = 60.0
frame_dur = 1.0 / float(measured_hz)

seg_frames = max(1, int(round(params["segmentDuration"] / frame_dur)))
isi_frames = max(0, int(round(params["isi"] / frame_dur)))
iti_frames = max(0, int(round(params["iti"] / frame_dur)))

seg_ach = seg_frames * frame_dur
isi_ach = isi_frames * frame_dur
iti_ach = iti_frames * frame_dur
trial_ach = (2 * seg_frames + isi_frames) * frame_dur

# Stimuli
fixCross = visual.ShapeStim(
    win, vertices="cross", units="pix", size=(10, 10),
    lineWidth=2, lineColor=[1, 1, 1], fillColor=[1, 1, 1],
    colorSpace="rgb", pos=(0, 0)
)
instr = visual.TextStim(
    win,
    text=("TACTILE repeated direction detection task\n\n"
          "Each trial has TWO tactile motion segments.\n"
          "TARGET = tactile direction REPEATS in segment 2.\n\n"
          "Press 'a' ONLY when there is a TARGET.\n\n"
          "Press SPACE to start."),
    height=0.06, color=[1, 1, 1], wrapWidth=1.4
)
block_msg = visual.TextStim(win, text="", height=0.06, color=[1, 1, 1], wrapWidth=1.4)
blank = visual.TextStim(win, text="")

kb = keyboard.Keyboard()
globalClock = core.Clock()

# Device
hdev = HapticsDevice(lib_name=params["lib_name"], simulate_only=params["simulate_only"])
hdev.start()

# Precompute buffers once
dir_buffers = {}
for d in ["wrist_to_finger", "finger_to_wrist", "pinky_to_thumb", "thumb_to_pinky", "none"]:
    dir_buffers[d] = build_haptic_buffer(
        direction=d,
        fs_hz=params["fs_hz"],
        stim_duration_s=seg_ach,
        line_length_m=params["line_length_m"],
        scrub_rate_hz=params["scrub_rate_hz"],
        drift_distance_m=params["drift_distance_m"],
        z_height_m=params["z_height_m"],
        intensity_on=params["intensity_on"],
    )

# Run + log
with open(tsv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "subNb","sessionNb","blockNb","trialNb",
        "trialOnset","trialDuration",
        "segment1Onset","segment2Onset",
        "segmentDurationAchieved","isiAchieved","itiAchieved",
        "segmentFrames","isiFrames","itiFrames",
        "axis","dir1","dir2","isTarget",
        "keyPressed","RT","outcome",
        "refreshHz",
        "droppedFramesSeg1","droppedFramesISI","droppedFramesSeg2","droppedFramesTrial",
        "simulateOnly"
    ])
    f.flush()

    instr.draw(); win.flip()
    event.waitKeys(keyList=["space"])

    for b in range(params["nBlocks"]):
        block_msg.setText(
            f"Block {b+1}/{params['nBlocks']}\n"
            "Press 'a' when direction repeats.\n\n"
            "Press SPACE to begin."
        )
        block_msg.draw(); win.flip()
        event.waitKeys(keyList=["space"])

        for t in range(params["nTrials"]):
            tr = design[b][t]

            # dropped-frame baseline at trial start
            drop_trial_start = getattr(win, "nDroppedFrames", 0)

            kb.clearEvents()
            key_pressed = ""
            rt = ""

            dir1 = tr["dir1_name"]
            dir2 = tr["dir2_name"]

            # =========================
            # SEGMENT 1 (onset + frames)
            # =========================
            seg1_onset_box = {"t": None}
            def _mark_seg1_onset():
                seg1_onset_box["t"] = globalClock.getTime()

            x1, y1, z1, i1 = dir_buffers[dir1]
            hdev.upload_buffer(x1, y1, z1, i1)

            fixCross.draw()
            win.callOnFlip(_mark_seg1_onset)    # true seg1 onset
            win.callOnFlip(kb.clock.reset)      # RT relative to seg1 onset
            win.callOnFlip(hdev.play, 0)
            win.flip()

            # check keys on each subsequent frame
            for _ in range(max(0, seg_frames - 1)):
                fixCross.draw(); win.flip()
                if key_pressed == "":
                    k = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if k:
                        key_pressed = k[0].name; rt = k[0].rt

            drop_after_seg1 = getattr(win, "nDroppedFrames", 0)
            dropped_seg1 = max(0, drop_after_seg1 - drop_trial_start)

            # =====
            # ISI
            # =====
            for _ in range(isi_frames):
                fixCross.draw(); win.flip()
                if key_pressed == "":
                    k = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if k:
                        key_pressed = k[0].name; rt = k[0].rt

            drop_after_isi = getattr(win, "nDroppedFrames", 0)
            dropped_isi = max(0, drop_after_isi - drop_after_seg1)

            # =========================
            # SEGMENT 2 (onset + frames)
            # =========================
            seg2_onset_box = {"t": None}
            def _mark_seg2_onset():
                seg2_onset_box["t"] = globalClock.getTime()

            x2, y2, z2, i2 = dir_buffers[dir2]
            hdev.upload_buffer(x2, y2, z2, i2)

            fixCross.draw()
            win.callOnFlip(_mark_seg2_onset)    # true seg2 onset
            win.callOnFlip(hdev.play, 0)
            win.flip()

            for _ in range(max(0, seg_frames - 1)):
                fixCross.draw(); win.flip()
                if key_pressed == "":
                    k = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if k:
                        key_pressed = k[0].name; rt = k[0].rt

            drop_after_seg2 = getattr(win, "nDroppedFrames", 0)
            dropped_seg2 = max(0, drop_after_seg2 - drop_after_isi)

            dropped_trial = max(0, drop_after_seg2 - drop_trial_start)

            seg1_onset = seg1_onset_box["t"] if seg1_onset_box["t"] is not None else globalClock.getTime()
            seg2_onset = seg2_onset_box["t"] if seg2_onset_box["t"] is not None else ""

            out = outcome_label(tr["is_target"], key_pressed, params["response_key"])

            writer.writerow([
                info["subNb"], info["sessionNb"],
                b + 1, t + 1,
                f"{seg1_onset:.4f}", f"{trial_ach:.4f}",
                f"{seg1_onset:.4f}",
                (f"{seg2_onset:.4f}" if seg2_onset != "" else ""),
                f"{seg_ach:.4f}", f"{isi_ach:.4f}", f"{iti_ach:.4f}",
                seg_frames, isi_frames, iti_frames,
                tr["axis"], dir1, dir2, tr["is_target"],
                key_pressed, (f"{rt:.4f}" if rt != "" else ""),
                out,
                f"{measured_hz:.3f}",
                dropped_seg1, dropped_isi, dropped_seg2, dropped_trial,
                int(params["simulate_only"])
            ])

            # ITI (blank), flip-locked
            for _ in range(iti_frames):
                blank.draw(); win.flip()

        # flush outside timing-critical loops
        if params.get("flush_at_block_end", True):
            f.flush()

# Cleanup
hdev.stop()
win.close()
core.quit()
