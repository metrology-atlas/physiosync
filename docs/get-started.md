# Get started

This page corrects the ECG-to-PPG delay of a simulated two-hour recording in a few lines. Everything runs
on synthetic data, so you can paste it as is.

## Install

```bash
pip install git+https://github.com/meshlab/physiosync
```

## Simulate a recording

A monitor delays its PPG output by a fixed hardware and filter delay, and its 30 Hz frame clock drifts
against the sampling clock. The delay therefore creeps down slowly and jumps back by one frame (33.3 ms)
at each slip: a sawtooth.

```python
import numpy as np
import physiosync as ps

rng = np.random.default_rng(1)
r_t = np.cumsum(rng.normal(60 / 70, 0.03, 8400))          # R peaks, about 70 bpm for two hours (s)

frame, rate = 1 / 30, 30e-6                                 # 30 Hz frames, frame clock off by 30 ppm
saw = frame - (rate * r_t) % frame                          # creeps down, jumps up one frame at a slip
ppg_t = r_t + 0.270 + rng.normal(0, 0.003, r_t.size) + 0.380 + saw   # true PAT 270 ms + 380 ms delay
```

## Fit and correct

```python
fit = ps.fit_ecg_ppg_delay(r_t, ppg_t, adopted_pat_s=0.270, frame_s=frame, matched=True)

print(len(fit.slips_t), "slips")                  # 6 slips
for q in fit.quality_intervals():
    print(q["tier"], round(q["start"]), round(q["end"]), q["rate_ppm"])   # A 1 7198 30.0

pat = fit.correct(ppg_t) - r_t
print(round(np.median(pat) * 1000, 1), round(np.std(pat) * 1000, 1))     # 270.0 3.2 (ms)
```

The raw PAT (PPG minus R) had an SD of about 10 ms from the sawtooth alone; after correction only the
3 ms beat noise is left, and the median sits at the adopted 270 ms.

!!! note "`adopted_pat_s` is a choice, not a measurement"
    A single recording cannot separate the hardware delay from the patient's true PAT. physiosync
    snaps the corrected median to a population value you choose; see
    [Choosing the adopted mean PAT](concepts/adopted-pat.md).

## Next steps

- Real recordings need beat pairing and a confidence check: [Correct the ECG–PPG delay](how-to/correct-ecg-ppg-delay.md).
- Store the result for a downstream estimator: [Build a measurement record](how-to/measurement-and-labels.md).
- Understand what the tiers mean: [Confidence tiers](concepts/tiers.md).
