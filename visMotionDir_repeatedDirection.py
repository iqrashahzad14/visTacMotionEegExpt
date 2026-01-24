"""
PsychoPy Experiment: Repeated-Direction Target Detection (within-trial repetition)

Adds:
- segment1Onset + segment2Onset (true onsets via callOnFlip)
- dropped frames per phase: Seg1 / ISI / Seg2 + total trial
- Removes os.fsync() from trial loop; flush only at block end
- Segment 2 structured symmetrically with Segment 1:
  prepare first frame -> flip (onset) -> remaining frames

Frame-locked timing based on measured refresh rate.
"""

from psychopy import visual, core, event, gui, data
from psychopy.hardware import keyboard
import random
import os
import csv

event.globalKeys.add(key="escape", func=core.quit, name="shutdown")

PARAMS = {
    "nBlocks": 4,
    "nTrials": 6,

    "segmentDuration": 0.5,
    "isi": 0.05,
    "trialDuration": None,
    "iti": 1.0,

    "targets_per_block": [1, 2, 1, 2],
    "target_min": 0,
    "target_max": 2,
    "target_balance": True,

    "response_key": "a",

    "alternate_axis_within_block": True,
    "alternate_start_axis": "random",

    "axis_dirs": {
        "horizontal": ["left", "right"],
        "vertical": ["up", "down"]
    },
    "dir_deg": {
        "right": 0.0,
        "left": 180.0,
        "up": 90.0,
        "down": -90.0
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

    # NEW: reduce disk IO jitter
    "flush_at_block_end": True,
}

# ----------------------------
# Validation + design helpers
# ----------------------------
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
            raise ValueError("targets_per_block must have length nBlocks.")
        for k in p["targets_per_block"]:
            if not (0 <= k <= p["nTrials"]):
                raise ValueError("Each targets_per_block value must be between 0 and nTrials.")

    if p["target_max"] > p["nTrials"]:
        p["target_max"] = p["nTrials"]

    for axis, dirs in p["axis_dirs"].items():
        if len(dirs) != 2:
            raise ValueError(f"axis_dirs['{axis}'] must have exactly 2 directions.")
        for d in dirs:
            if d not in p["dir_deg"]:
                raise ValueError(f"Missing dir_deg for direction '{d}'")

    if p["alternate_start_axis"] not in ["horizontal", "vertical", "random"]:
        raise ValueError("alternate_start_axis must be 'horizontal', 'vertical', or 'random'")

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

def axis_sequence_for_block(p: dict, rng: random.Random):
    if not p["alternate_axis_within_block"]:
        return [rng.choice(["horizontal", "vertical"]) for _ in range(p["nTrials"])]

    if p["alternate_start_axis"] == "random":
        start = rng.choice(["horizontal", "vertical"])
    else:
        start = p["alternate_start_axis"]

    other = "vertical" if start == "horizontal" else "horizontal"
    return [start if i % 2 == 0 else other for i in range(p["nTrials"])]

def create_design(params: dict, seed=None):
    p = validate_params(params)
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
                "dir1_deg": p["dir_deg"][dir1],
                "dir2_deg": p["dir_deg"][dir2],
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
# Participant GUI
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
    f"task-visMotionDirRep_desc-events_"
    f"{data.getDateStr(format='%Y-%m-%d-%H%M')}.tsv"
)
tsv_path = os.path.join(data_dir, tsv_name)

# Design
params, design, _ = create_design(PARAMS, seed=None)

# Window
screen_index = params["screen_index_external"] if params["use_external_monitor"] else params["screen_index_laptop"]
try:
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
except Exception as e:
    print(f"[WARN] Could not open window on screen={screen_index}. Falling back to screen=0. Error:\n{e}")
    win = visual.Window(
        size=params["win_size"],
        fullscr=params["fullscr"],
        screen=0,
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

params["segmentFrames"] = seg_frames
params["isiFrames"] = isi_frames
params["itiFrames"] = iti_frames

params["segmentDurationAchieved"] = seg_frames * frame_dur
params["isiAchieved"] = isi_frames * frame_dur
params["itiAchieved"] = iti_frames * frame_dur
params["trialDuration"] = (2 * seg_frames + isi_frames) * frame_dur

dotlife_frames = max(1, int(round((params["dotLife_ms"] / 1000.0) / frame_dur)))
params["dotLifeFrames"] = dotlife_frames

# Stimuli
fixCross = visual.ShapeStim(
    win, vertices="cross", units="pix", size=(10, 10),
    lineWidth=1, lineColor=[1, 1, 1], fillColor=[1, 1, 1],
    colorSpace="rgb", pos=(0, 0)
)

rdk = visual.DotStim(
    win,
    units="pix",
    nDots=params["nDots"],
    dotSize=params["dotSize"],
    speed=params["dotSpeed"],
    dir=0.0,
    coherence=1.0,
    fieldPos=(0.0, 0.0),
    fieldSize=params["fieldSize"],
    fieldShape=params["fieldShape"],
    signalDots="same",
    noiseDots="position",
    dotLife=params["dotLifeFrames"],
    color=[1.0, 1.0, 1.0],
    colorSpace="rgb"
)

instr = visual.TextStim(
    win,
    text=(
        "Repeated motion direction detection task\n\n"
        "Each trial has TWO motion segments.\n"
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
        "refreshHz","dotLifeFrames",
        "droppedFramesSeg1","droppedFramesISI","droppedFramesSeg2","droppedFramesTrial"
    ])
    f.flush()

    # Start screen
    instr.draw(); win.flip()
    event.waitKeys(keyList=["space"])

    for b in range(params["nBlocks"]):
        block_msg.setText(
            f"Block {b+1}/{params['nBlocks']}\n"
            "Reminder: press 'a' when direction repeats.\n\n"
            "Press SPACE to begin."
        )
        block_msg.draw(); win.flip()
        event.waitKeys(keyList=["space"])

        for t in range(params["nTrials"]):
            tr = design[b][t]

            drop_trial_start = getattr(win, "nDroppedFrames", 0)
            kb.clearEvents()
            key_pressed = ""
            rt = ""

            # =========================
            # SEGMENT 1 (onset + frames)
            # =========================
            seg1_onset_box = {"t": None}
            def _mark_seg1_onset():
                seg1_onset_box["t"] = globalClock.getTime()

            rdk.dir = tr["dir1_deg"]
            fixCross.draw(); rdk.draw()
            win.callOnFlip(_mark_seg1_onset)
            win.callOnFlip(kb.clock.reset)
            win.flip()

            # collect keys over remaining seg1 frames
            for _ in range(max(0, seg_frames - 1)):
                fixCross.draw(); rdk.draw()
                win.flip()
                if key_pressed == "":
                    keys = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name; rt = keys[0].rt

            drop_after_seg1 = getattr(win, "nDroppedFrames", 0)
            dropped_seg1 = max(0, drop_after_seg1 - drop_trial_start)

            # =====
            # ISI
            # =====
            for _ in range(isi_frames):
                fixCross.draw()
                win.flip()
                if key_pressed == "":
                    keys = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name; rt = keys[0].rt

            drop_after_isi = getattr(win, "nDroppedFrames", 0)
            dropped_isi = max(0, drop_after_isi - drop_after_seg1)

            # =========================
            # SEGMENT 2 (onset + frames)
            # =========================
            seg2_onset_box = {"t": None}
            def _mark_seg2_onset():
                seg2_onset_box["t"] = globalClock.getTime()

            rdk.dir = tr["dir2_deg"]
            fixCross.draw(); rdk.draw()
            win.callOnFlip(_mark_seg2_onset)
            win.flip()

            for _ in range(max(0, seg_frames - 1)):
                fixCross.draw(); rdk.draw()
                win.flip()
                if key_pressed == "":
                    keys = kb.getKeys(keyList=[params["response_key"]], waitRelease=False, clear=False)
                    if keys:
                        key_pressed = keys[0].name; rt = keys[0].rt

            drop_after_seg2 = getattr(win, "nDroppedFrames", 0)
            dropped_seg2 = max(0, drop_after_seg2 - drop_after_isi)

            dropped_trial = max(0, drop_after_seg2 - drop_trial_start)

            seg1_onset = seg1_onset_box["t"] if seg1_onset_box["t"] is not None else globalClock.getTime()
            seg2_onset = seg2_onset_box["t"] if seg2_onset_box["t"] is not None else ""

            out = outcome_label(tr["is_target"], key_pressed, params["response_key"])

            writer.writerow([
                info["subNb"], info["sessionNb"],
                b + 1, t + 1,
                f"{seg1_onset:.4f}", f"{params['trialDuration']:.4f}",
                f"{seg1_onset:.4f}",
                (f"{seg2_onset:.4f}" if seg2_onset != "" else ""),
                f"{params['segmentDurationAchieved']:.4f}",
                f"{params['isiAchieved']:.4f}",
                f"{params['itiAchieved']:.4f}",
                params["segmentFrames"], params["isiFrames"], params["itiFrames"],
                tr["axis"], tr["dir1_name"], tr["dir2_name"], tr["is_target"],
                key_pressed, (f"{rt:.4f}" if rt != "" else ""),
                out,
                f"{measured_hz:.3f}", params["dotLifeFrames"],
                dropped_seg1, dropped_isi, dropped_seg2, dropped_trial
            ])

            # ITI
            for _ in range(iti_frames):
                blank.draw()
                win.flip()

        if params.get("flush_at_block_end", True):
            f.flush()

win.close()
core.quit()
