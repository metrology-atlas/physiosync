# Confidence tiers

Every tooth of a fitted PPG delay gets a tier. Corrections are applied only from `apply_min_tier` up
(default B); every other tooth keeps the offset only and is labelled, so users know where the PPG timing
can be trusted.

| Tier | Meaning | Correction |
|---|---|---|
| **A** | High confidence. The tooth's rate is corroborated: a full tooth within 3% of the recording's median rate (with at least two full teeth), or within 5% of the setup's known rate, no rejected slip-like step inside, beat-to-beat noise at most 6 ms. | Applied |
| **B** | Moderate. Rate within 10% of the recording or setup, noise at most 9 ms; or an A tooth with an inserted slip that the arterial reference confirmed. | Applied |
| **C** | Low. Any other corrected tooth, including teeth bounded by an inserted or predicted slip that only the ECG–PPG series showed. | Not applied by default |
| **N** | No sawtooth: no slip over at least 45 min (or 1.5 tooth lengths), or the sawtooth is shown absent where slips were due. | Offset only |
| **U** | Unresolved: slips but no usable rate, an end tooth too long for its rate, a stretch too short to tell. | Offset only |

The thresholds are the fields of [`TierRules`](../reference/ecg_ppg.md); pass your own to
`fit_ecg_ppg_delay(..., tier_rules=...)`.

**Rate conflict.** When a recording's own tooth rate differs from its setup's known rate by more than
20%, either the setup is wrong for this recording or its slips are. The recording's own rate is then used,
teeth are at most C (B where the arterial reference confirmed the slips), and nothing is labelled N.

**Why inserted slips are capped.** Checked against arterial pressure, about a third of slips inferred
from the ECG–PPG series alone were not real. Slips confirmed by a reference signal (`ref_t`) are trusted
more.
