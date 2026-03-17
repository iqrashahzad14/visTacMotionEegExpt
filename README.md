# visTacMotionEegExpt
This repository contains Python code for a visual–tactile motion EEG experiment.
visTacMotionEeg_main.py
This is the main expeirmental code

## Requirements
- Python 3.x
- PsychoPy (`psychopy`)
- NumPy

install PsychoPy Builder/Standalone and run this python code from there.
Tested on Macbok Pro 2019 Intel Model 16 inch 
Psychopy Coder v2021.2.3

## visTacMotionEeg_main.py
Combined PsychoPy experiment for **repeated-direction target detection** using:
- **Visual motion** (Random Dot Kinematogram; RDK)
- **Tactile motion** via a haptics device (optional)

## Visual motion stimulus (RDK)
The visual condition uses a Random Dot Kinematogram (RDK):
- Black background, white dots
- Fixation cross centered
- 100% coherence
- Dot lifetime (ms) is converted to frames using measured refresh rate

## Tactile motion stimulus (optional)
Hardware control is handled via a shared library:
- `libStreaming_CachedPoint_ctypes.so`
When `simulate_only=False`, the script runs with the device "actual experiment"
- `start_array()`
- `set_positions()`
- `set_intensities()`
- `play_sensation(0)`
- `stop_array()`

## How to run (VISUAL-ONLY, no haptics device)
This repository includes a combined script that can run without the haptics hardware.
`simulate_only=True`

## Performance upgrades already implemented
- Preallocation of ctypes arrays per direction (avoid per-trial allocation)
- Next-trial buffer preload during ITI (tactile blocks)
- Warmup at tactile block start to avoid cold-start latency

## Task overview (Visual + Tactile)
Each trial represents one direction
All trials are in randmoized order
**Participant response**
- Press **`a`** **only** when the direction **repeats**  (TARGET).
- Responses are accepted throughout the whole trial (segment 1, ISI, segment 2) and also into the ITI.
Outcome labels per trial:
- **HIT**: Target trial + `a` pressed
- **MISS**: Target trial + no `a`
- **FALSE_ALARM**: Non-target + `a`
- **CORRECT_REJECT**: Non-target + no `a`

## Files in Old Routine
- `visTacMotionEegExpt_main.py` – Main experiment script
- `visTacMotionEegExpt_main_withTriggers.py` – Main script with EEG triggers
- `visMotionDir_repeatedDirection.py` – Only Visual motion condition
- `tacMotionDir_repeatedDirection.py` – Only Tactile motion condition

