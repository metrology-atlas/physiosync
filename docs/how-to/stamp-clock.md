# Remove a record-stamp sawtooth

Many recorders stamp each block of samples with the time its sample count implies, and only now and then
step the stamps back towards the recorder's clock. Between corrections the stamps drift at the device's
rate error; at each correction they jump. Two devices recorded together are then out of step by the
difference of their sawtooths. See [Clocks, drift and sawtooths](../concepts/clocks.md).

## Fit one track

You need, per record, the stamp of its first sample and that sample's index in the track:

```python
import numpy as np
import physiosync as ps

fs = 500.0
starts = np.arange(0, 3_600_000, 100)                    # 2 h of 100-sample records
lag = 30e-6 * starts / fs                                 # a +30 ppm device falls behind the recorder
stamps = starts / fs + 0.01 * np.floor(lag / 0.01)        # stamps follow the count, stepped every 10 ms

clock = ps.stamp_clock(stamps, starts, fs)
print(round(clock["rate_ppm"]), clock["rate_source"], clock["corrections"]["events"])   # 30 teeth 21
```

The result gives the device's rate error (`rate_ppm`, from whole teeth when there are enough, otherwise
from a line fit), the stretches between real gaps (`breaks`) with one offset each, and the correction events.

## Place samples on the fitted line

```python
rec_t = ps.corrected_times(clock, starts)                              # one time per record
sample_t = ps.corrected_times(clock, starts, np.full(starts.size, 100))  # one time per sample
true_t = starts / fs + lag                                     # the recorder's time of each record
print(round(np.ptp(stamps - true_t), 4), round(np.ptp(rec_t - true_t), 4))   # 0.01 s sawtooth -> 0.0 s
```

## Options that matter

- `rate_ppm`: use a known rate (for example the device's rate from other recordings) when a track is too
  short to contain whole teeth.
- `correction_sign`: the direction the device corrects in, if known; steps the other way are then treated
  as real gaps.
- `gap_s`: jumps at least this large start a new stretch with its own offset.

What remains between two devices after this is a constant offset (half of each device's correction
threshold plus any transport delay). A shared reference (shared artefacts, a common signal) has to supply it.
