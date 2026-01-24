# visTacMotionEegExpt
This repository contains Python code for a visual–tactile motion EEG experiment.

## Files
- `visTacMotionEegExpt_main.py` – Main experiment script
- `visTacMotionEegExpt_main_withTriggers.py` – Main script with EEG triggers
- `visMotionDir_repeatedDirection.py` – Only Visual motion condition
- `tacMotionDir_repeatedDirection.py` – Only Tactile motion condition

## visTacMotionEegExpt_main.py

# visTacMotionEegExpt
Combined PsychoPy experiment for **repeated-direction target detection** using:
- **Visual motion** (Random Dot Kinematogram; RDK)
- **Tactile motion** via a haptics device (optional)
## Task overview (Visual + Tactile)

Each trial contains **two motion segments**:
- Segment 1: motion in some direction (e.g., left)
- ISI: short gap
- Segment 2: motion in either:
  - **Same direction** as segment 1 (**TARGET**), or
  - **Opposite direction** within the same axis (**non-target**)

**Participant response**
- Press **`a`** **only** when the direction **repeats** in segment 2 (TARGET).
- Responses are accepted throughout the whole trial (segment 1, ISI, segment 2) and also into the ITI.
  - If a response happens during the ITI, it is attributed to the preceding trial.

Outcome labels per trial:
- **HIT**: Target trial + `a` pressed
- **MISS**: Target trial + no `a`
- **FALSE_ALARM**: Non-target + `a`
- **CORRECT_REJECT**: Non-target + no `a`

## Visual motion stimulus (RDK)
The visual condition uses a Random Dot Kinematogram (RDK):
- Black background, white dots
- Fixation cross centered
- 100% coherence
- Dot lifetime (ms) is converted to frames using measured refresh rate
- Directions: **up/down/left/right**
- Axis per trial: **horizontal** (left/right) or **vertical** (up/down)
- Within each block, trials can alternate axis: H/V/H/V… (or V/H/V/H…)

## Tactile motion stimulus (optional)
Tactile trials play two haptic motion segments (segment 1 and segment 2).
Hardware control is handled via a shared library:
- `libStreaming_CachedPoint_ctypes.so`
When `simulate_only=False`, the script calls:
- `start_array()`
- `set_positions()`
- `set_intensities()`
- `play_sensation(0)`
- `stop_array()`

### Performance upgrades already implemented
- Preallocation of ctypes arrays per direction (avoid per-trial allocation)
- Next-trial buffer preload during ITI (tactile blocks)
- Warmup at tactile block start to avoid cold-start latency

## Requirements
- Python 3.x
- PsychoPy (`psychopy`)
- NumPy

install PsychoPy Builder/Standalone and run this python code from there.
Tested on Macbok Pro 2019 Intel Model 16 inch 
Psychopy Coder v2021.2.3

## How to run (VISUAL-ONLY, no haptics device)
This repository includes a combined script that can run without the haptics hardware.

### Step 1 — Open the script you want to run
Main combined script example:
- `visTacMotionEegExpt_main.py` (or your combined script file)

### Step 2 — Enable simulation mode
In the `PARAMS` dictionary, set:
"simulate_only": True

## visTacMotionEegExpt_main_withTriggers.py
same as visTacMotionEegExpt_main.py but with triggers 
Triggers added and then reset to 0
TRIG = {
    "visual":   {"seg1": 11, "seg2": 12, "resp": 13},
    "tac_up":   {"seg1": 21, "seg2": 22, "resp": 23},
    "tac_down": {"seg1": 31, "seg2": 32, "resp": 33},
}

  
