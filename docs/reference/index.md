# API reference

Everything below is importable from the top level, e.g. `physiosync.fit_ecg_ppg_delay`. Pages follow the
modules.

| Module | What it does |
|---|---|
| [ecg_ppg](ecg_ppg.md) | ECG–PPG delay: beat matching, slip detection, the fitted delay model and its tiers, setup offsets |
| [stamps](stamps.md) | Record-stamp sawtooths: one rate and one offset per stretch, corrected times |
| [network](network.md) | Networks of clock links: loop misclosures, least-squares adjustment, normalised residuals |
| [frames](frames.md) | Hidden frame clocks: change spectrum, lines, interval locking, the frame clock's rate |
| [quantisation](quantisation.md) | Effective temporal resolution of fiducials: whole-sample grids, partial locking, pooling |
| [measurement](measurement.md) | The PAT measurement record, fiducial error models, error draws and simulation |
| [labels](labels.md) | Flat sawtooth and offset labels for storage |
| [chains](chains.md) | Rate-consistent slip chains: a lower-confidence label track |
| [pairing](pairing.md) | Rhythm-based beat pairing (optional hrvmatch) |
| [reference](reference.md) | Published PAT values for choosing the adopted mean PAT |
| [timemap](timemap.md) | Time maps between recorded and corrected time: periodic, drifting and tabulated components, lead offsets |
| [release](release.md) | Reading a checksummed correction release: corrected times, Monte Carlo realisations, the three-way run |
