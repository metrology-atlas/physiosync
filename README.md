# physiosync

Timing corrections for multi-signal physiological recordings. MIT licence; needs only numpy and scipy.

physiosync works on times and samples you give it. It does not read file formats, and it assumes no
particular dataset, monitor, sampling rate or frame rate. A dataset's import code can call it and store its
results (corrected timing plus quality labels); the TimeNet VitalDB reader is one example. physiosync is part of
the [Metrology Atlas](https://vitaldb.metrology-atlas.org) project.

Paper: Goodwin, A. J. (2026). *The Metrology Atlas: describing recorded physiological data as a measurement system, with VitalDB as a first application*. [arXiv:2610.29001](https://arxiv.org/abs/2610.29001).

## Modules

| Module | What it does |
|---|---|
| `quantisation` | Effective temporal resolution of fiducial points: whole-sample grids and partial locking to any period (`fiducial_resolution`, `locking`, `pool`). |
| `frames` | Finds a hidden frame clock at any rate. It uses the spectrum of the sample-to-sample change (`change_spectrum`, `frame_lines`), fiducial locking (`interval_lock`, `period_lock_scan`) and the frame clock's rate against the sampling clock (`frame_clock`). |
| `ecg_ppg` | ECG-to-PPG delay. One combined offset snaps the PAT to an adopted population mean, and a frame-slip sawtooth is fitted and removed (`fit_ecg_ppg_delay`). It returns confidence tiers, quality intervals and labels, and a timing plan for storage. |
| `measurement` | The PAT measurement model of a recording for downstream estimators such as a maximum-likelihood PAT (`pat_measurement_model`). It records the fiducials' resolution and rounding, the offset, and per-segment sync-error models (corrected, ramp, creep-then-hold where the sawtooth stops, unknown phase), and draws from them (`error_samples`, `simulate_pat`). |
| `labels` | Sawtooth labels: one interval per tooth with its phase, slope, amplitude and duration, the applied correction, the fiducials' time resolution, and a confidence (tier, remaining sync error drawn from the measurement model, rounding error, beat noise, `usable_for_pat`) for downstream PAT estimators (`sawtooth_labels`, `sawtooth_label_tuples`). |
| `chains` | Rate-consistent slip chains: with a recording's rate known, slips are chained one tooth length apart and each chained tooth becomes a label on a separate, lower-confidence track (`chain_slips`, `chain_labels`). |
| `stamps` | Record time stamps that follow a device's sample clock between occasional corrections: fits the rate (from whole correction-to-correction teeth, a line through the stamp staircase, or a given rate) and one offset per stretch between gaps, and places every sample on that line, removing the stamp sawtooth (`stamp_clock`, `corrected_times`). What remains between two devices is a constant offset for a shared reference to supply. |
| `reference` | Published pulse arrival times by study, hardware and clocking, to help choose the adopted mean PAT (`pat_reference`). |

## Design rules

- **Nothing dataset-specific is built in.** The frame size is estimated unless given. Rate priors, the
  plausible rate range, the tier thresholds and the arterial reference are all optional parameters.
- **Fiducials come from the caller.** Any detector can be used. The `quantisation` and `frames` tools
  check whether a detector's points lock to a frame grid, so you can pick a smooth fiducial for timing
  and a frame-locked one for finding the frame.
- **Confidence over coverage.** Corrections are applied only where they are corroborated (tier A or B
  by default). Everything else keeps the offset only and is labelled, so downstream users know where
  timing is uncertain.
- **Errors stay local.** Each tooth of the sawtooth is fitted on its own and balances one frame, so a
  wrong or missed slip distorts only its own tooth.

## Terms

- **Frame:** a block of samples a monitor produces and plays out at once (33.3 ms in VitalDB's PPG).
- **Slip:** the moment the monitor repeats (or drops) a frame because the signal has drifted a whole
  frame ahead of (or behind) its output clock. The PAT jumps by one frame in a single beat.
- **Tooth:** the stretch between two slips, during which the PAT creeps at a steady rate.
