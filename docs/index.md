# physiosync

**Timing corrections for multi-signal physiological recordings.** physiosync finds and removes the timing
artefacts that bedside monitors and recorders add when several signals are captured together: clock drift
between devices, the sawtooth that record stamps carry between clock corrections, and the frame-slip
sawtooth in the delay between ECG and PPG. It also measures how finely each fiducial point is really
resolved, and describes what is left as an error model a downstream estimator can use.

It works on plain arrays of times and sample indices. It reads no file format and assumes no dataset,
monitor, sampling rate or frame rate. A dataset's import code can call it and store its results (corrected
timing plus quality labels); the VitalDB connectors for TimeNet and AtriumDB are one example.

## What it does

| Problem | What physiosync gives you | Start here |
|---|---|---|
| The PPG arrives late and its delay drifts and jumps by one frame (a sawtooth) | A fitted delay model, corrected PPG times, confidence tiers per tooth | [Correct the ECG–PPG delay](how-to/correct-ecg-ppg-delay.md) |
| Record stamps drift with the device clock and step at each correction | One rate error and one offset per stretch; stamps placed on a straight line | [Remove a record-stamp sawtooth](how-to/stamp-clock.md) |
| A fiducial may sit on a coarser grid than one sample (frames, buffers, hum) | The effective resolution, partial locking, the frame clock's rate | [Check fiducial resolution](how-to/fiducial-resolution.md) |
| A PAT estimator needs to know how wrong each observed PAT can be | A JSON measurement record, error draws and flat labels | [Build a measurement record](how-to/measurement-and-labels.md) |
| The ECG–PPG delay is longer than a beat | Beat pairing from the rhythm itself | [Pair beats](how-to/pair-beats.md) |
| A recording's time base was distorted and a fitted correction has been released | Corrected times per lead and Monte Carlo realisations with the stated uncertainty | [Use a correction release](how-to/use-a-release.md) |

## Design rules

- **Nothing dataset-specific is built in.** Frame sizes are estimated unless given; rate priors, tier
  thresholds and the arterial reference are optional parameters.
- **Fiducials come from the caller.** Any detector can be used. The resolution tools check whether a
  detector's points lock to a frame grid, so you can pick a smooth fiducial for timing and a frame-locked
  one for finding the frame.
- **Confidence over coverage.** Corrections are applied only where they are corroborated (tier A or B by
  default). Everything else keeps the offset only and is labelled, so users know where the timing is
  trustworthy.

## Install

```bash
pip install git+https://github.com/metrology-atlas/physiosync      # numpy and scipy only
```

[Beat pairing](how-to/pair-beats.md) also needs the optional `hrvmatch` package.

## Licence and citation

MIT licence, © 2026 Andrew J. Goodwin, University of Sydney. The method builds on Goodwin et al. 2023
(*Physiol. Meas.* 44 085002) for rhythm-based pairing, and generalises the fiducial-resolution test of
Ruffolo et al. 2025 (*Physiol. Meas.* 46 115006).
