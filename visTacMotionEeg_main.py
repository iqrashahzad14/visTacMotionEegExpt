"""
Combined PsychoPy experiment: Visual + Tactile (Hand Up / Hand Down) one-back detection
+ EEG serial triggers (callOnFlip + reset-to-zero for stimulus triggers)

NEW DESIGN
----------
- 3 block types: visual, tac_up, tac_down
- 3 randomized repetitions of blockCycle = [visual, tac_up, tac_down]
- Total blocks per subject = 9
- Each block uses 4 stimulus directions/types
- Each stimulus type is repeated nRepTrial times in the block backbone
- Backbone trials within a block are fully randomized with no identical consecutive trials
- One-back targets are created by duplicating selected backbone trials immediately after the original
- Participants press the response key when the CURRENT trial repeats the PREVIOUS trial
- Fixation cross is present throughout the whole block (baseline, trial, ITI)
- Each block starts after SPACE, then a 2 s fixation baseline, then the stimulus + iti

SERIAL PORT
- Set the correct SERIAL_PORT string for your laptop.
- In terminal, type: ls /dev/tty.*
- output in terminal : 
/dev/tty.BLTH
/dev/tty.usbmodem141201  <- choose this
/dev/tty.Bluetooth-Incoming-Port

NOTES ABOUT ITI JITTER:
"1 s average" and with jitter sampled from a normal distribution in [0, 0.5]
iti = 1.0 + truncated_normal_jitter_in_[0, 0.5]
so the actual ITI range is [1.0, 1.5] s.
"""

from psychopy import visual, core, event, gui, data
from psychopy.hardware import keyboard
import numpy as np
import random
import os
import csv
import threading
import ctypes
import serial

event.globalKeys.add(key="escape", func=core.quit, name="shutdown")

# =========================
# PARAMETER BLOCK
# =========================
PARAMS = {
    # -------- New design --------
    "nbStim": 4,   # directions
    "nRepTrial": 40,          # repetitions per stimulus type in the backbone
    "propTarget": 0.10,       # target proportion based on backbone trial count
    "nbBlockReps": 3,         # repetitions of randomized [visual, tac_up, tac_down]
    "baseline_s": 2.0,

    # Trial timing
    "stimDuration": 0.5,      # stimulus duration
    "iti_base": 1.0,
    "iti_jitter_min": 0.0,
    "iti_jitter_max": 0.5,
    "iti_jitter_mu": 0.25,    # truncated normal mean inside jitter interval
    "iti_jitter_sigma": 0.125,

    "response_key": "a",

    # Visual directions
    "vis_dirs": ["left", "right", "up", "down"],
    "vis_dir_deg": {"right": 0.0, "left": 180.0, "up": 90.0, "down": -90.0},

    # Tactile directions
    "tac_dirs": ["wrist_to_finger", "finger_to_wrist", "pinky_to_thumb", "thumb_to_pinky"],

    # Window / display
    "use_external_monitor": True,
    "screen_index_laptop": 0,
    "screen_index_external": 1,
    "win_size": [1280, 1024], #[1536, 960],
    "fullscr": True,
    "monitor": "testMonitor",
    "bg_color": [0, 0, 0],

    # Visual stimulus
    "nDots": 50,
    "dotSize": 20,
    "dotSpeed": 6,
    "fieldSize": (800, 800),
    "fieldShape": "circle",
    "dotLife_ms": 200,

    # Haptics
    "fs_hz": 40_000.0,
    "line_length_m": 0.05,
    "scrub_rate_hz": 100.0,
    "drift_distance_m": 0.06 ,  # movement on the hand, total distance covered across hand
    "z_height_m": 0.10,
    "intensity_on": 1.0,
    "simulate_only": False,
    "lib_name": "libStreaming_CachedPoint_ctypes.so",
    "flush_at_block_end": True,
    "tactile_warmup_wait_s": 0.05,

    # EEG triggers
    "use_eeg_triggers": True,
    "serial_port": "/dev/tty.usbmodem142301",  # <-- CHECK THIS
    "serial_baudrate": 9600,
}

# Trigger map:
# 4 unique triggers for each visual direction
# 4 unique triggers for each tactile direction in tac_up
# 4 unique triggers for each tactile direction in tac_down
# plus target marker and response marker in each block type
TRIG = {
    "visual": {
        "left": 11,
        "right": 12,
        "up": 13,
        "down": 14,
        "target": 15,
        "resp": 16,
    },
    "tac_up": {
        "wrist_to_finger": 21,
        "finger_to_wrist": 22,
        "pinky_to_thumb": 23,
        "thumb_to_pinky": 24,
        "target": 25,
        "resp": 26,
    },
    "tac_down": {
        "wrist_to_finger": 31,
        "finger_to_wrist": 32,
        "pinky_to_thumb": 33,
        "thumb_to_pinky": 34,
        "target": 35,
        "resp": 36,
    },
}

# =========================
# Design helpers
# =========================
def validate_params(p: dict) -> dict:
    p = dict(p)

    if p["nbStim"] != 4:
        raise ValueError("This design expects nbStim = 4.")
    if p["nRepTrial"] < 1:
        raise ValueError("nRepTrial must be >= 1.")
    if not (0.0 <= p["propTarget"] < 1.0):
        raise ValueError("propTarget must be in [0, 1).")
    if p["nbBlockReps"] < 1:
        raise ValueError("nbBlockReps must be >= 1.")
    if p["stimDuration"] <= 0:
        raise ValueError("stimDuration must be > 0.")
    if p["baseline_s"] < 0:
        raise ValueError("baseline_s must be >= 0.")
    if p["iti_base"] < 0:
        raise ValueError("iti_base must be >= 0.")
    if not (0.0 <= p["iti_jitter_min"] <= p["iti_jitter_max"]):
        raise ValueError("Invalid ITI jitter bounds.")
    if p["iti_jitter_sigma"] <= 0:
        raise ValueError("iti_jitter_sigma must be > 0.")

    if len(p["vis_dirs"]) != 4:
        raise ValueError("vis_dirs must contain exactly 4 directions.")
    for d in p["vis_dirs"]:
        if d not in p["vis_dir_deg"]:
            raise ValueError(f"Missing vis_dir_deg for '{d}'.")

    if len(p["tac_dirs"]) != 4:
        raise ValueError("tac_dirs must contain exactly 4 directions.")

    if p.get("dotLife_ms", 0) <= 0:
        raise ValueError("dotLife_ms must be > 0.")

    return p

def outcome_label(is_target: int, key_pressed: str, target_key: str) -> str:
    pressed = (key_pressed != "")
    if is_target:
        return "HIT" if (pressed and key_pressed == target_key) else "MISS"
    return "FALSE_ALARM" if pressed else "CORRECT_REJECT"

def sample_truncated_normal(rng: random.Random, low: float, high: float, mu: float, sigma: float) -> float:
    while True:
        x = rng.gauss(mu, sigma)
        if low <= x <= high:
            return x

def randomize_block_sequence(nb_reps: int, rng: random.Random):
    """Randomize [visual, tac_up, tac_down] inside each cycle, repeated nb_reps times,
    with no identical consecutive blocks across cycle boundaries.
    """
    base = ["visual", "tac_up", "tac_down"]
    out = []
    prev_last = None

    for _ in range(nb_reps):
        perms = []
        for _k in range(100):
            cand = base[:]
            rng.shuffle(cand)
            if prev_last is None or cand[0] != prev_last:
                perms = cand
                break
        if not perms:
            # fallback
            perms = base[:]
            rng.shuffle(perms)
            if prev_last is not None and perms[0] == prev_last:
                perms = perms[1:] + perms[:1]

        out.extend(perms)
        prev_last = perms[-1]

    # final sanity check
    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            raise RuntimeError("Block sequence generation failed: consecutive identical blocks.")
    return out

def build_backbone_no_adjacent(directions: list, n_rep: int, rng: random.Random):
    """Create a randomized backbone with no identical consecutive trials."""
    pool = []
    for d in directions:
        pool.extend([d] * n_rep)

    # Simple randomized greedy sampler
    counts = {d: n_rep for d in directions}
    seq = []
    prev = None

    total = len(pool)
    for _ in range(total):
        available = [d for d in directions if counts[d] > 0 and d != prev]
        if not available:
            raise RuntimeError("Could not build sequence without adjacent identical trials.")
        rng.shuffle(available)
        # mild balancing: choose among highest remaining counts
        max_count = max(counts[d] for d in available)
        candidates = [d for d in available if counts[d] == max_count]
        choice = rng.choice(candidates)
        seq.append(choice)
        counts[choice] -= 1
        prev = choice

    return seq

def choose_nonconsecutive_target_indices(n_backbone: int, nb_targets: int, rng: random.Random):
    """
    Exclude first two and last two backbone trials.
    Choose non-consecutive indices from the remainder.
    Zero-based indices eligible: 2 .. n_backbone-3
    """
    eligible = list(range(2, n_backbone - 2))
    if nb_targets == 0:
        return set()
    if len(eligible) < nb_targets:
        raise ValueError("Not enough eligible trials for requested number of targets.")

    for _ in range(5000):
        sample = sorted(rng.sample(eligible, k=nb_targets))
        if all((b - a) > 1 for a, b in zip(sample[:-1], sample[1:])):
            return set(sample)

    raise RuntimeError("Could not sample non-consecutive target indices.")

def create_presented_trials_for_block(p: dict, block_type: str, rng: random.Random):
    """
    Returns the presented trial list for one block.
    Backbone length = nbStim * nRepTrial
    Target trials are duplicated immediately after selected actual trials.
    """
    if block_type == "visual":
        directions = p["vis_dirs"]
    else:
        directions = p["tac_dirs"]

    nb_backbone = p["nbStim"] * p["nRepTrial"]   # 160
    nb_targets = int(round(p["propTarget"] * nb_backbone))  # 16

    backbone = build_backbone_no_adjacent(directions=directions, n_rep=p["nRepTrial"], rng=rng)
    target_indices = choose_nonconsecutive_target_indices(nb_backbone, nb_targets, rng)

    presented = []
    for base_idx, direction in enumerate(backbone):
        presented.append({
            "baseIndex": base_idx + 1,
            "direction": direction,
            "is_target": 0,
            "origin": "actual",
        })
        if base_idx in target_indices:
            presented.append({
                "baseIndex": base_idx + 1,
                "direction": direction,
                "is_target": 1,   # the duplicate is the target
                "origin": "duplicate",
            })

    return presented, nb_backbone, nb_targets

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
        x = pos_along_line
        y = pos_drift
    elif direction == "finger_to_wrist":
        x = pos_along_line
        y = -pos_drift
    elif direction == "pinky_to_thumb":
        x = pos_drift
        y = pos_along_line
    elif direction == "thumb_to_pinky":
        x = -pos_drift
        y = pos_along_line
    else:
        x = np.zeros_like(t)
        y = np.zeros_like(t)

    z = np.full_like(t, z_height_m)
    i = np.full_like(t, intensity_on)
    return x, y, z, i

class HapticsDevice:
    def __init__(self, lib_name: str, simulate_only: bool):
        self.simulate_only = simulate_only
        self.lib_name = lib_name
        self.prg = None
        self.thread = None
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
        if self.simulate_only:
            self.ctypes_buffers[direction] = {"n": int(len(x_np))}
            return

        n = int(len(x_np))
        arr_t = ctypes.c_double * n

        self.ctypes_buffers[direction] = {
            "n": n,
            "xc": arr_t(*x_np.tolist()),
            "yc": arr_t(*y_np.tolist()),
            "zc": arr_t(*z_np.tolist()),
            "ic": arr_t(*i_np.tolist()),
        }

    def upload_by_name(self, direction: str):
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
# Trigger helpers
# =========================
def make_serial_port(p):
    if not p.get("use_eeg_triggers", True):
        return None
    try:
        s = serial.Serial(port=p["serial_port"], baudrate=p["serial_baudrate"])
        core.wait(0.1)
        s.write(bytes([0]))
        return s
    except Exception as e:
        print(f"[WARN] Could not open serial port for EEG triggers. Triggers disabled. Error:\n{e}")
        return None

def trig_codes_on_flip(win, ser, codes):
    """
    Schedule one or more trigger codes on the upcoming flip.
    Each code is immediately followed by a reset-to-zero, mirroring the original script style.

    NOTE:
    If multiple codes are used on the same flip (e.g., stimulus code + target code),
    they are queued on the same flip callback list in sequence.
    """
    if ser is None:
        return
    for code in codes:
        win.callOnFlip(ser.write, bytes([int(code)]))
        win.callOnFlip(ser.write, bytes([0]))

def trig_response_now(ser, code: int):
    if ser is None:
        return
    try:
        ser.write(bytes([int(code)]))
        ser.write(bytes([0]))
    except Exception as e:
        print(f"[WARN] response trigger write failed: {e}")

# =========================
# Participant GUI
# =========================
PARAMS = validate_params(PARAMS)

info = {"subNb": "", "sessionNb": "1"}  # default sessionNb = 1
dlg = gui.DlgFromDict(info, order=["subNb", "sessionNb"])
if not dlg.OK:
    core.quit()

sub = int(info["subNb"]) if info["subNb"].strip() else 0
ses = int(info["sessionNb"]) if info["sessionNb"].strip() else 1

# =========================
# Paths / TSV name
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# Keeping the same naming style as the original script
tsv_name = (
    f"sub-{sub:03d}_ses-{ses:03d}_"
    f"task-visTacRepeatDirection_desc-events_"
    f"{data.getDateStr(format='%Y-%m-%d-%H%M')}.tsv"
)
tsv_path = os.path.join(data_dir, tsv_name)

# =========================
# Build block sequence
# =========================
rng = random.Random(None)
block_sequence = randomize_block_sequence(PARAMS["nbBlockReps"], rng)

block_idx_within = {"visual": 0, "tac_up": 0, "tac_down": 0}

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

baseline_frames = max(0, int(round(PARAMS["baseline_s"] / frame_dur)))
stim_frames = max(1, int(round(PARAMS["stimDuration"] / frame_dur)))
stim_ach = stim_frames * frame_dur

dotlife_frames = max(1, int(round((PARAMS["dotLife_ms"] / 1000.0) / frame_dur)))

# =========================
# EEG serial setup
# =========================
ser = make_serial_port(PARAMS)

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
        "One-back task\n\n"
        "Press 'a' when the CURRENT trial repeats the PREVIOUS trial.\n\n"
        "There are visual blocks and tactile blocks.\n"
        "Keep looking at the fixation cross.\n\n"
        "Press SPACE to start."
    ),
    height=0.06, color=[1, 1, 1], wrapWidth=1.4
)

block_msg = visual.TextStim(win, text="", height=0.06, color=[1, 1, 1], wrapWidth=1.4)
kb = keyboard.Keyboard()
globalClock = core.Clock()

# =========================
# Haptics device + buffers
# =========================
hdev = HapticsDevice(lib_name=PARAMS["lib_name"], simulate_only=PARAMS["simulate_only"])
hdev.start()

dir_buffers_np = {}
for d in ["wrist_to_finger", "finger_to_wrist", "pinky_to_thumb", "thumb_to_pinky", "none"]:
    dir_buffers_np[d] = build_haptic_buffer(
        direction=d,
        fs_hz=PARAMS["fs_hz"],
        stim_duration_s=stim_ach,
        line_length_m=PARAMS["line_length_m"],
        scrub_rate_hz=PARAMS["scrub_rate_hz"],
        drift_distance_m=PARAMS["drift_distance_m"],
        z_height_m=PARAMS["z_height_m"],
        intensity_on=PARAMS["intensity_on"],
    )

for d, (x, y, z, i) in dir_buffers_np.items():
    hdev.register_direction_buffer(d, x, y, z, i)

def tactile_block_warmup():
    if PARAMS["simulate_only"]:
        return
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
        "subNb", "sessionNb",
        "globalBlockNb", "blockType", "blockIndexWithinType",
        "backboneTrials", "nbTargetsPlanned", "trialNbInBlock",
        "baseIndex", "origin", "direction", "isTarget",
        "trialOnset", "trialDurationPlanned",
        "stimOnset", "stimDurationAchieved",
        "itiPlanned", "itiAchieved",
        "stimFrames", "itiFrames",
        "keyPressed", "RT", "outcome",
        "refreshHz", "dotLifeFrames", "simulateOnly",
        "droppedFramesStim", "droppedFramesITI", "droppedFramesTrial",
    ])
    f.flush()

    instr.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    total_blocks = len(block_sequence)

    for gblock_idx, block_type in enumerate(block_sequence, start=1):
        block_i = block_idx_within[block_type]
        presented_trials, nb_backbone, nb_targets = create_presented_trials_for_block(PARAMS, block_type, rng)

        if block_type == "visual":
            block_label = f"Visual block ({block_i + 1}/{sum(1 for b in block_sequence if b == 'visual')})"
            reminder = "Press 'a' when the current visual trial repeats the previous one."
        elif block_type == "tac_up":
            block_label = f"Tactile HAND UP block ({block_i + 1}/{sum(1 for b in block_sequence if b == 'tac_up')})"
            reminder = "Press 'a' when the current tactile trial repeats the previous one."
        else:
            block_label = f"Tactile HAND DOWN block ({block_i + 1}/{sum(1 for b in block_sequence if b == 'tac_down')})"
            reminder = "Press 'a' when the current tactile trial repeats the previous one."

        block_msg.setText(
            f"{block_label}\n"
            f"Global block {gblock_idx}/{total_blocks}\n\n"
            "One-back task:\n"
            "Respond when the CURRENT trial repeats the PREVIOUS trial.\n\n"
            f"{reminder}\n\n"
            "Press SPACE to begin."
        )
        block_msg.draw()
        win.flip()
        event.waitKeys(keyList=["space"])

        if block_type in ("tac_up", "tac_down"):
            tactile_block_warmup()
            if len(presented_trials) > 0:
                hdev.upload_by_name(presented_trials[0]["direction"])

        # -------- Block baseline (2 s fixation) --------
        for _ in range(baseline_frames):
            fixCross.draw()
            win.flip()

        # -------- Trials --------
        for t_idx, tr in enumerate(presented_trials, start=1):
            is_tactile = block_type in ("tac_up", "tac_down")
            direction = tr["direction"]
            is_target = tr["is_target"]

            iti_jitter = sample_truncated_normal(
                rng=rng,
                low=PARAMS["iti_jitter_min"],
                high=PARAMS["iti_jitter_max"],
                mu=PARAMS["iti_jitter_mu"],
                sigma=PARAMS["iti_jitter_sigma"],
            )
            iti_s = PARAMS["iti_base"] + iti_jitter
            iti_frames = max(0, int(round(iti_s / frame_dur)))
            iti_ach = iti_frames * frame_dur
            trial_ach = stim_ach + iti_ach

            drop_trial_start = getattr(win, "nDroppedFrames", 0)

            kb.clearEvents()
            key_pressed = ""
            rt = ""
            resp_trig_sent = False

            stim_onset_box = {"t": None}

            def _mark_stim_onset():
                stim_onset_box["t"] = globalClock.getTime()

            trig_codes = [TRIG[block_type][direction]]
            if is_target:
                trig_codes.append(TRIG[block_type]["target"])

            # =========================
            # STIMULUS PERIOD
            # =========================
            if block_type == "visual":
                rdk.dir = PARAMS["vis_dir_deg"][direction]
                fixCross.draw()
                rdk.draw()
                win.callOnFlip(_mark_stim_onset)
                win.callOnFlip(kb.clock.reset)
                trig_codes_on_flip(win, ser, trig_codes)
                win.flip()

                for _ in range(max(0, stim_frames - 1)):
                    fixCross.draw()
                    rdk.draw()
                    win.flip()

                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt
                            if not resp_trig_sent:
                                trig_response_now(ser, TRIG[block_type]["resp"])
                                resp_trig_sent = True

            else:
                fixCross.draw()
                win.callOnFlip(_mark_stim_onset)
                win.callOnFlip(kb.clock.reset)
                win.callOnFlip(hdev.play, 0)
                trig_codes_on_flip(win, ser, trig_codes)
                win.flip()

                for _ in range(max(0, stim_frames - 1)):
                    fixCross.draw()
                    win.flip()

                    if key_pressed == "":
                        keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                        if keys:
                            key_pressed = keys[0].name
                            rt = keys[0].rt
                            if not resp_trig_sent:
                                trig_response_now(ser, TRIG[block_type]["resp"])
                                resp_trig_sent = True

            drop_after_stim = getattr(win, "nDroppedFrames", 0)
            dropped_stim = max(0, drop_after_stim - drop_trial_start)

            # =========================
            # ITI (fixation remains visible)
            # =========================
            next_tr = presented_trials[t_idx] if (is_tactile and t_idx < len(presented_trials)) else None

            for frame_i in range(iti_frames):
                fixCross.draw()
                win.flip()

                if key_pressed == "":
                    keys = kb.getKeys(keyList=[PARAMS["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name
                        rt = keys[0].rt
                        if not resp_trig_sent:
                            trig_response_now(ser, TRIG[block_type]["resp"])
                            resp_trig_sent = True

                # pre-upload next tactile buffer early in ITI, as in the original style
                if is_tactile and next_tr is not None and frame_i == 0:
                    hdev.upload_by_name(next_tr["direction"])

            drop_after_iti = getattr(win, "nDroppedFrames", 0)
            dropped_iti = max(0, drop_after_iti - drop_after_stim)
            dropped_trial = max(0, drop_after_iti - drop_trial_start)

            stim_onset = stim_onset_box["t"] if stim_onset_box["t"] is not None else globalClock.getTime()
            out = outcome_label(is_target, key_pressed, PARAMS["response_key"])

            dotLifeFrames_out = dotlife_frames if block_type == "visual" else ""
            simulate_only_out = int(PARAMS["simulate_only"]) if block_type != "visual" else ""

            writer.writerow([
                info["subNb"], info["sessionNb"],
                gblock_idx, block_type, (block_i + 1),
                nb_backbone, nb_targets, t_idx,
                tr["baseIndex"], tr["origin"], direction, is_target,
                f"{stim_onset:.4f}", f"{trial_ach:.4f}",
                f"{stim_onset:.4f}", f"{stim_ach:.4f}",
                f"{iti_s:.4f}", f"{iti_ach:.4f}",
                stim_frames, iti_frames,
                key_pressed, (f"{rt:.4f}" if rt != "" else ""),
                out,
                f"{measured_hz:.3f}", dotLifeFrames_out, simulate_only_out,
                dropped_stim, dropped_iti, dropped_trial,
            ])

        block_idx_within[block_type] += 1

        if PARAMS.get("flush_at_block_end", True):
            f.flush()

# =========================
# Cleanup
# =========================
try:
    hdev.stop()
except Exception:
    pass

try:
    if ser is not None:
        ser.write(bytes([0]))
        ser.close()
except Exception:
    pass

win.close()
core.quit()