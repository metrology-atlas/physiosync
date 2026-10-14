# Pair beats when the delay is long

`match_beats` pairs each R peak with the pulse nearest to R plus the most common delay. When the delay is
longer than a beat (fast heart rates and long monitor delays) the most common delay is one beat short, and
every pulse is paired with the next R peak.

`pair_beats` pairs from the beats' own rhythm instead (Goodwin et al. 2023): windows of consecutive PPG
inter-beat intervals are matched to the ECG's, each match gives a beat-index shift, and the running mode of
the good matches pairs every pulse. It needs the optional package hrvmatch:

```bash
pip install --no-deps git+https://github.com/LaussenLabs/hrvmatch
```

```python
# requires hrvmatch
res = ps.pair_beats(r_t, ppg_t, expected_delay_s=(0.45, 0.85))
fit = ps.fit_ecg_ppg_delay(r_t[res["r_index"]], ppg_t[res["p_index"]], adopted_pat_s=0.270, matched=True)
print(res["qc"])     # windows, good and ambiguous matches, beats paired by rhythm or by window
```

A very regular rhythm cannot tell a beat from its neighbours. Those windows are marked ambiguous; give
`expected_delay_s` (for example the setup's known delay ± 0.2 s) so the plausible delay decides them and
pairs beats the rhythm left unpaired.
