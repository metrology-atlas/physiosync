# The PAT measurement model

After correction, an observed PAT still differs from the latent (true) PAT. The measurement record from
[`pat_measurement_model`](../reference/measurement.md) describes how, in parts that a likelihood-based
estimator can use:

```text
PAT_observed(t) = PAT_latent(t) + e_sync(t) + e_quant(t) + e_beat(t) + e_rec
```

| Term | Scope | Where it comes from |
|---|---|---|
| `e_sync` | per segment, shared by its beats | the sawtooth left after correction: corrected teeth (small, by tier), ramps where a tooth was not corrected, creep-then-hold where the sawtooth stopped, unknown phase |
| `e_quant` | per beat | rounding of each fiducial to its effective resolution (one sample, or one frame for a frame-locked fiducial), with its rounding rule (nearest, floor, ceil) |
| `e_beat` | per beat | beat-to-beat noise, estimated from the fitted beats |
| `e_rec` | shared by all beats | uncertainty of the adopted mean PAT and of the hardware delay (uniform half-widths) |

[`error_samples`](../reference/measurement.md) draws these terms at any beat times;
[`simulate_pat`](../reference/measurement.md) adds them to a latent PAT, either as stored after correction
or as raw (with the applied shift added back), so a method can be checked end to end. Leaving out the
shared terms gives the per-beat part for a joint likelihood that integrates over the shared ones.

The flat labels from [`sawtooth_labels`](../reference/labels.md) summarise the same record per tooth:
the remaining sync error's mean, SD and 95th percentile, and whether the interval is `usable_for_pat`.
