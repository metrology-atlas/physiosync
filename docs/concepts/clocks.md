# Clocks, drift and sawtooths

Every device in a multi-signal recording has its own clock. A crystal oscillator is off its nominal
frequency by a few to a few hundred parts per million (ppm): 100 ppm is 0.36 s per hour. When signals from
several devices are combined, their timing errors add.

## Two kinds of drift

**Sample clocks.** A device samples at what it believes is, say, 500 Hz. If its crystal runs 30 ppm fast,
each "second" of samples covers 0.99997 s of true time. Placing samples by their count at the nominal rate
then drifts by 30 µs per second against true time.

**Stamp corrections.** A recorder usually stamps each block of samples, and a common design stamps "where
the sample count says this block is" and steps the stamp back towards the recorder's clock whenever the
two disagree by more than a threshold. The stamp error is then a staircase or sawtooth: a ramp at the
device's rate error, and a jump at each correction.

```
stamp error
   ^      /|      /|      /|
   |     / |     / |     / |      ramp: the device's rate error
   |    /  |    /  |    /  |      jump: a correction of about one threshold
   |   /   |   /   |   /   |
   +--------------------------> time
```

[`stamp_clock`](../reference/stamps.md) fits one rate per track and one offset per stretch between real
gaps, which places every sample on a straight line through the middle of the sawtooth. What is left
between two devices is a constant offset, which a shared reference (a shared artefact, a common signal)
has to supply.

## The ECG–PPG sawtooth is different

The frame sawtooth in the ECG–PPG delay comes from inside one monitor: the pulse oximeter produces its
waveform in frames on its own clock, and the monitor re-samples it. See
[Frames and slips in the PPG](frames-and-slips.md).

## Time anchors

A value that summarises a window of signal (a heart rate over 8 s, a frame of PPG) cannot be known before
the window ends. physiosync works with event times (fiducials, samples), not windowed values; when you
build features from its output, stamp each at the end of its window. A value used before that time
carries information from the future.
