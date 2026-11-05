# Changelog

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
