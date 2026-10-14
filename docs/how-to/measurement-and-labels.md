# Build a PAT measurement record and labels

After fitting, store what a downstream PAT estimator needs: how each observed PAT relates to the latent
one. The record is a JSON-ready dict (schema `physiosync.pat_measurement/1`, about 4 kB).

## The record

```python
import numpy as np
import physiosync as ps
from physiosync.measurement import dumps

rng = np.random.default_rng(1)
r_t = np.cumsum(rng.normal(60 / 70, 0.03, 8400))
frame, rate = 1 / 30, 30e-6
ppg_t = r_t + 0.270 + rng.normal(0, 0.003, r_t.size) + 0.380 + (frame - (rate * r_t) % frame)
fit = ps.fit_ecg_ppg_delay(r_t, ppg_t, adopted_pat_s=0.270, frame_s=frame, matched=True)

rec = ps.pat_measurement_model(fit, ecg_fs=500.0, ppg_fs=500.0, ecg_fiducial="R peak",
                               ppg_fiducial="foot", r_t=r_t, ppg_t=ppg_t)
text = dumps(rec)            # NaN-free JSON
```

Pass `ppg_resolution_s` or, better, `ppg_error=ps.fiducial_error_model(...)` when the PPG fiducial
locks to frames (see [fiducial resolution](fiducial-resolution.md)), and `adopted_pat_sd_s` for the
uncertainty of the adopted PAT.

## Draw errors, simulate observations

```python
draws = ps.error_samples(rec, r_t[:100], 1000)          # observed minus latent PAT, shape (1000, 100)
obs = ps.simulate_pat(rec, r_t[:100], 0.27, n=10)       # observed PAT for a latent 270 ms
per_beat = ps.error_samples(rec, r_t[:100], 1000, components=("quantisation",))
```

Leave out the shared components (`"sync"`, `"recording"`) to get the per-beat part only, for example to
build a joint likelihood that integrates over the shared ones.

## Labels

```python
labels = ps.sawtooth_labels(rec, max_error_s=0.005)
print(labels[0]["tier"], labels[0]["tooth_s"], labels[0]["usable_for_pat"])   # A 1111.2 True
names = ps.sawtooth_label_tuples(labels)                # ('ppg_sawtooth:A', start, end), ...
offset = ps.offset_label(rec)                           # the hardware delay removed, as one label
```

Each tooth label carries the sawtooth's slope, amplitude, period and phase, the correction applied
(`shift_start_ms`, `shift_slope_ppm`, so a downstream method can undo it and apply its own), the fiducials'
error models, and the remaining sync error. `usable_for_pat` is true when the tier is accepted and the
95th-percentile error is within `max_error_s`. Field lists: `ps.LABEL_FIELDS`, `ps.OFFSET_FIELDS`.

Chained teeth from `ps.chain_labels` are a separate, lower-confidence track; see the
[chains reference](../reference/chains.md).
