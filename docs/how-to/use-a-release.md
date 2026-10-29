# Use a timing-correction release

Some recordings carry a distorted time base: a tape replayed at a slowly varying speed, a playback clock that
drifts, signals digitised with a constant skew between channels. When those distortions have been fitted (for
MIT-BIH, by the companion package `mitbih-chain`), the fit is published as a **release**: a folder with a
`manifest.json` holding the SHA-256 of every file, and one JSON file per record. physiosync reads a release,
maps recorded times to corrected times on each lead, and draws Monte Carlo realisations that carry the stated
uncertainty.

## Map recorded times to corrected times

A record's correction is a `TimeMap` plus its uncertainty. Here a small synthetic record stands in for one read
from a release with `ps.load_release(folder).record("100")`:

```python
import numpy as np
import physiosync as ps
from physiosync.release import _example_record, run

rc = ps.RecordCorrection.from_dict(_example_record())   # real use: ps.load_release(folder).record(name)
t = np.arange(10.0, 600.0, 0.8)                         # recorded beat times on lead 0 (s)
tau = rc.to_corrected(t, lead=0)                        # corrected times
print(round(float(np.max(np.abs(tau - t))) * 1000, 1))  # largest correction, ms: 33.7
```

`to_corrected` inverts the map exactly (to 1e-10 s); `rc.map.to_recorded(tau, lead)` goes the other way.
Sample values are never changed: only times move. Every record also states `available_at_s`, the end of the
record: the corrections are fitted on whole records, so they suit offline analysis and must not feed an
algorithm presented as working in real time.

## Propagate the uncertainty

Each realisation combines one bootstrap parameter set, one draw of the unmodelled time-base term and, if you
give `label_sd`, an independent placement draw per time. All times in a record move together, so errors that
are correlated between beats are sampled jointly. `run` applies an analysis to the recorded times, the
corrected times and K realisations:

```python
rmssd = lambda x: float(np.sqrt(np.mean(np.diff(np.diff(x)) ** 2)) * 1000)
res = run(rmssd, rc, t, lead=0, k=30, seed=1)
print(round(float(res["original"]), 1), round(float(res["estimate"]), 1), round(float(res["u"]), 2))   # 0.0 14.7 0.53
```

Evenly spaced recorded times give an RMSSD of zero; once corrected they are not evenly spaced, because the
synthetic record's 6 s warp moved them. The result holds the recorded-time value (`original`), the corrected one, the `estimate` and its standard
uncertainty `u`, a coverage interval, every trial result (`draws`) and the provenance (release checksum,
record, seed, K) from which each realisation can be reproduced exactly.

Measures that grow with noise, such as RMSSD or a band power, come out shifted in the trials, because each
trial adds error draws to times that already contain errors. By default (`adjust=True`) `run` treats that shift
`s` as an uncorrected systematic effect: estimate = corrected − s/2, u = √(sd² + s²/12) (JCGM 100, 4.3.7). For
skewed results such as powers, analyse the logarithm.

## Check the files

`load_release` checks each record file against the manifest's SHA-256 when it is read and raises
`ReleaseError` if a file is missing or altered. The manifest's checksum is carried in every `run` result, so a
reported number names the exact release it came from.
