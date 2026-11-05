# Changelog

## 0.3.3 (November 2026)

- `method="rebuilt"` measurement records and labels: `pat_measurement_model` and `sawtooth_labels` give one segment per
  tooth (slip to slip, from `tooth_tiers`): A, B and C corrected, with the per-tier residual SD (still the detect
  method's injected benchmark, stated in the record); U and I not corrected, their sync error a sawtooth of the
  recording's rate (or its reported drift) with unknown phase; N none. `EcgPpgDelay.qc` keeps the teeth and the sign.
- `fit_ecg_ppg_delay(method="rebuilt", slip_result=...)` applies a `find_slips` result made earlier (e.g. by a
  whole-release run) to other beats of the same recording on the same clock; `fiducials` and `beat_quality` are then
  not needed.

## 0.3.2 (November 2026)

- `method="rebuilt"`: the correction is rebuilt after an end-to-end check against the arterial line (finger minus
  arterial pulse arrival time, 2,976 VitalDB recordings), where 0.3.1 left a third of the recordings worse than no
  correction. Each tooth between two slips now creeps exactly one frame over its own length (0.3.1: one creep rate for
  the whole recording, so a tooth-length error built up); slips placed only within an interval are corrected as a
  ramp weighted by their probability; unresolved teeth (tier U) are not corrected; recordings whose creep is not
  corrected (isolated slips, a tooth from two slips: tier I) are not corrected at all, since steps without their creep
  turn a sawtooth into a ramp. In sawtooth recordings the spread of finger minus arterial PAT falls from 14.0 ms
  uncorrected to 9.5 ms (method="detect" 11.2), with 7.3% of recordings worse than uncorrected by more than 2 ms
  (0.3.1: 28.9%; detect 2.9%).
- New `delay_at(t, result)`: the delay to remove; `delay_model` gives it as segments (steps may be fractional where an
  unresolved tooth is held at the median). Slip detection is unchanged.

## 0.3.1 (November 2026)

- `method="rebuilt"`: a sawtooth found by the anchored search (a tooth from the spacing of confirmed slips) corrects
  its creep only when three or more slips are confirmed. With two, the steps are corrected and the tooth and its creep
  are reported, not corrected (`fit["creep"] == "reported"`, `EcgPpgDelay.qc["creep"]`; its teeth are tier I). On
  the whole VitalDB release, a tooth from two slips was wrong in 9 of 22 injected sawteeth. The midpoint of the two
  closest slips is judged and reported (`fit["half_check"]`), not used. `SlipOptions.anchor_check="off"` gives 0.3.0.
- Slip detection is unchanged; `method="detect"` is unchanged.

## 0.3.0 (November 2026)

- New module `slips`: the rebuilt frame-slip finder (`find_slips`), with the per-beat quality and pulse shapes it
  needs (`beat_quality`), its options (`SlipOptions`), the tiers of its teeth (`tooth_tiers`) and its delay as
  piecewise-linear teeth (`delay_model`). It uses the foot, upstroke and peak of each pulse, gates beats on ECG and PPG
  quality, fits the breathing in the step test, fits the tooth as a chain of slips, judges each slip on the pulse shape
  and, where there is one, the arterial line, finds both sawtooth directions, and places slips it cannot time within
  an interval with a probability. Only one-frame steps are corrected; sub-frame steps never are; the drift of
  recordings without a sawtooth is reported, not corrected.
- `fit_ecg_ppg_delay(method="rebuilt", matched=True, fiducials=..., beat_quality=..., ref_t=..., slip_options=...)`
  fits the delay from `find_slips`. New tier I (isolated slips, no sawtooth: steps only). New `EcgPpgDelay` fields:
  `method`, `tooth_steps`, `direction`, `slips`, `uncertain`, `drift`.
- `method="detect"` stays the default and gives the same results as 0.2.0 (checked field by field on 324 VitalDB
  recordings). `method="rebuilt"` reproduces vitaldb-bench's prototype exactly (324 recordings and 120 slow
  injections; the delay within 1 microsecond, the same tiers).
