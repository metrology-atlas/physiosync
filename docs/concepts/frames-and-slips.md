# Frames and slips in the PPG

Many pulse oximeters compute their waveform in blocks ("frames") at their own rate, for example 30 Hz, and
the monitor then outputs or re-samples it at its own rate, for example 500 Hz. Two things follow.

**Frame quantisation.** The waveform's fine structure keeps the frame grid. Fiducials taken on raw samples
(a raw peak, second-derivative points) land on frame boundaries, so they are rounded to one frame, 33 ms
at 30 Hz, however fast the output is sampled. Fiducials on a smoothed signal are interpolated between
frames and are much finer. [`fiducial_resolution`](../reference/quantisation.md) and
[`interval_lock`](../reference/frames.md) tell the two apart.

**Frame slips.** The frame clock and the sampling clock are independent crystals. If the frame clock runs
slightly fast, each frame is delivered a little earlier than the last, so the PPG's delay behind the ECG
creeps down at the rate difference (ppm). The monitor's buffer cannot run ahead of real time forever: once
a whole frame has been gained, the output repeats or drops one frame, and the delay jumps back by one
frame. The result is a sawtooth with:

| Property | Value | Example (30 Hz, 30 ppm) |
|---|---|---|
| Amplitude | one frame | 33.3 ms |
| Slope | the rate difference | 30 µs per second |
| Period (tooth) | frame / rate | 1,111 s, about 18.5 min |

A raw PAT series (PPG minus R) therefore shows a slow downward creep of tens of milliseconds and a sharp
upward step of one frame every tooth. Left in, it adds an SD of about frame/√12 (about 10 ms) and
systematic error that varies with the time of the beat in the tooth.

## Correcting it

[`fit_ecg_ppg_delay`](../reference/ecg_ppg.md) finds the slips in the beat-by-beat delay
([`detect_slips`](../reference/ecg_ppg.md)), measures each tooth's rate, inserts slips that were missed
when the tooth length says one was due, and subtracts the fitted sawtooth. Each tooth gets a
[confidence tier](tiers.md).

The sawtooth can also stop: a monitor may stop slipping for a while (a pause), which looks like a tooth
that never ends. [`sawtooth_evidence`](../reference/ecg_ppg.md) tests whether a sawtooth of the expected
rate is actually present over a stretch.
