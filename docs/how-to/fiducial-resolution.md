# Check fiducial resolution and frame locking

A fiducial point is reported at a whole sample, so its nominal resolution is one sample. Its effective
resolution can be coarser: a signal produced upstream at a lower rate and re-sampled, a buffer that
delivers values in blocks, or a detector that only lands on certain samples all put fiducials on a grid.
This matters for PAT: a PPG fiducial locked to 30 Hz frames is rounded to 33 ms, not 2 ms.

## Is there a grid?

```python
import numpy as np
import physiosync as ps

fs = 500.0
rng = np.random.default_rng(1)
frames_t = np.arange(0, 3600, (1 / 30) * (1 + 20e-6))       # a 30 Hz frame clock, 20 ppm slow
beats = np.sort(rng.choice(frames_t.size, 4000, replace=False))
idx = np.round(frames_t[beats] * fs).astype(int)            # fiducials taken on raw samples land on frames

res = ps.fiducial_resolution(idx, fs=fs, ibi_range=(150, 1500))
print(res.effective_samples)          # 1: no whole-sample grid (16.67 samples is not an integer)
print(res.locking[0]["freq_hz"])      # 30.03: but the fiducials lock to a 30 Hz comb
```

`effective_samples` tests whole-sample grids (every k-th sample). `locking` scans non-integer periods,
which catches frame rates that do not divide the sampling rate, and partial locking, where only some
fiducials are pulled onto the comb. Pass `reference_idx` (for example ECG R peaks of the same beats) for a
sharper test against what the interval distribution alone would produce.

## How strongly, and how fast is the frame clock?

```python
print(ps.interval_lock(idx, fs / 30))   # resultant near 1, ratio to control periods well above 1
print(ps.frame_clock(idx, fs / 30))     # ppm 20.0, the same in both halves: a stable frame clock
```

`frame_clock` finds the frame period that keeps the absolute fiducial indices in phase over the whole
recording, which gives the frame clock's rate against the sampling clock to about 1 ppm. That rate is the
sawtooth's creep rate.

## Finding an unknown frame

```python
ppg = np.sin(2 * np.pi * 1.2 * np.arange(0, 600, 1 / fs))   # replace with raw PPG samples
lines = ps.frame_lines(ppg, fs)                               # strongest lines of |x[n+1] - x[n]|
```

A framed signal changes in a pattern that repeats every frame, which shows as spectral lines at the frame
rate and its harmonics whatever the waveform. Mains hum (50/60 Hz families) is flagged.

## Which fiducial to use

Fiducials taken on raw samples (raw peak, second-derivative points) tend to lock to the frame grid:
good for finding the frame clock. Fiducials on a smoothed signal (steepest upstroke after filtering) do not:
better for timing. Use the two together. See [Frames and slips in the PPG](../concepts/frames-and-slips.md).
