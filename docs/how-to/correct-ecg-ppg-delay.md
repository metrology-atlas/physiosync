# Correct the ECG–PPG delay

Use this when you have R-peak times and PPG fiducial times from the same recording and want PPG times
free of the monitor's delay and frame sawtooth, plus labels saying where the correction can be trusted.

## 1. Detect fiducials and pair the beats

physiosync does not detect peaks: use any R-peak detector on the ECG and any PPG fiducial (foot, maximum
upstroke, ...) on the PPG. Then pair each R peak with its pulse.

```python
import numpy as np
import physiosync as ps

# synthetic stand-ins for your detector output (s)
rng = np.random.default_rng(1)
r_t = np.cumsum(rng.normal(60 / 70, 0.03, 8400))
frame, rate = 1 / 30, 30e-6
ppg_t = r_t + 0.270 + rng.normal(0, 0.003, r_t.size) + 0.380 + (frame - (rate * r_t) % frame)
ppg_t = ppg_t[rng.random(ppg_t.size) > 0.01]               # a detector misses a few pulses
art_t = r_t + 0.200 + rng.normal(0, 0.002, r_t.size)        # arterial upstroke, if there is a line

i, j, delay = ps.match_beats(r_t, ppg_t)
print(len(i), "pairs; typical delay", round(delay * 1000), "ms")    # 8336 pairs; 674 ms
```

`match_beats` pairs each R peak with the pulse nearest to R plus the typical delay. If the delay is close to
or longer than a beat (fast heart rates, long monitor delays), use [rhythm-based pairing](pair-beats.md).

## 2. Fit

```python
fit = ps.fit_ecg_ppg_delay(
    r_t[i], ppg_t[j],
    adopted_pat_s=0.270,      # population mean PAT for this fiducial (see Concepts)
    matched=True,             # r_t and ppg_t are already paired
    rate_prior_ppm=30.0,      # the setup's creep rate, for teeth too short to measure
    ref_t=art_t[i],           # optional: confirms slips and rejects physiological steps
)
```

Useful options:

| Option | When to use it |
|---|---|
| `frame_s` | The frame is known (e.g. 1/30 s for the SNUADC PLETH in VitalDB). Otherwise it is estimated from the reset sizes, which is less exact. |
| `rate_prior_ppm` | The recording setup's typical creep rate is known from other recordings. |
| `ref_t` | An arterial line is present. Steps that also appear in ECG→ART are physiology and are rejected. |
| `offset_s` | The hardware delay is known for the setup (see step 4). The recording then keeps its own PAT level. |
| `apply_min_tier` | Raise to `"A"` for the strictest correction, lower to `"C"` to correct more. |
| `sawtooth=False` | The device has no frame sawtooth: fit an offset only. |

## 3. Use the result

```python
corrected = fit.correct(ppg_t)                 # every PPG fiducial (or sample) time, corrected
for name, start, end in fit.labels():         # e.g. ('ppg_timing:A', 0.9, 7197.7)
    print(name, round(start), round(end))
print(fit.qc["raw_level_s"])                   # median PAT with the sawtooth removed, before any offset
```

`labels()` and `quality_intervals()` say where the correction is applied. Only use PPG timing for PAT
in intervals of the tiers you accept; see [Confidence tiers](../concepts/tiers.md).

To store a corrected continuous PPG stream, `timing_plan` splits it into one segment per tooth, each with
its own start time and effective sampling rate:

```python
plan = fit.timing_plan(start_t=0.0, n=500 * 7200, fs=500.0)
print(len(plan), "segments; first:", plan[0]["count"], "samples at", round(plan[0]["fs"], 4), "Hz")
```

## 4. Many recordings: one offset per setup

A single recording cannot separate the hardware delay from the patient's PAT. Across many recordings made
with the same equipment, the median recording level minus the adopted PAT estimates the setup's delay:

```python
levels = {"rec1": 0.668, "rec2": 0.659, "rec3": 0.671, "rec4": 0.664}     # fit.qc["raw_level_s"] each
setups = {"rec1": "monitor A", "rec2": "monitor A", "rec3": "monitor A", "rec4": "monitor A"}
off = ps.setup_offsets(levels, setups, adopted_pat_s=0.270)
print(round(off["by_recording"]["rec1"]["offset_s"], 3))    # 0.396 s
```

Refit each recording with `offset_s=off["by_recording"][rec]["offset_s"]` so it keeps its own PAT level.
