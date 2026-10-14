# Choosing the adopted mean PAT

A single recording cannot separate the monitor's hardware and filter delay from the patient's true pulse
arrival time: both are a constant added to every beat. [`fit_ecg_ppg_delay`](../reference/ecg_ppg.md)
therefore needs `adopted_pat_s`, the population mean PAT that corrected data should have, or an
`offset_s` known for the recording setup (see [`setup_offsets`](../reference/ecg_ppg.md)). Use one value
for every dataset you want to combine.

## Rules for choosing it

- **Same fiducial.** The adopted PAT must refer to the PPG fiducial your detector uses. Foot, maximum
  slope and peak differ by tens to hundreds of milliseconds.
- **Same site and population.** Finger PAT is longer than ear PAT and shorter than toe PAT. It shortens
  with age and blood pressure and lengthens under general anaesthesia (by about 40 ms in monitor data,
  Youn et al. 2014).
- **Trust shared clocks.** Only studies where one acquisition unit sampled ECG and PPG (tier A) measure
  PAT without a device delay between the two signals.

## The reference table

`ps.pat_reference()` returns the table that ships with physiosync (`physiosync/data/pat_reference.csv`),
with hardware, clocking, fiducials and notes for each study.

| Tier | Meaning |
|---|---|
| A | ECG and PPG sampled by one acquisition unit (shared clock) |
| B | Research equipment; shared clock likely but not stated |
| C | Clinical monitor or unknown: carries a large, unknown ECG–PPG delay; for context, not adoption |
| C-corrected | Monitor data corrected for profiled delays (Ruffolo et al. 2025, children) |

| Study | Tier | n, population | PPG site, fiducial | Hardware (clock) | PAT (ms) |
|---|---|---|---|---|---|
| [Block et al 2020](https://doi.org/10.1038/s41598-020-73143-8) | A | 32, adults 50% F, 25% treated HTN, 52 ± 17 | finger (right), foot (intersecting tangent) | Biopac MP150 + ECG100C + PPG100C/Nonin 8000 (single DAQ 1 kHz) | 269 ± 6 (SE) |
| [Kortekaas et al 2018](https://doi.org/10.1371/journal.pone.0204105) | A | 20, healthy controls, 27 ± 4 | index finger L/R, foot = max 2nd derivative | Biopac MP100 + ECG100C + PPG100C/TSD200 (single DAQ 2 kHz) | 273.5 ± 8 (L), 7 (R) (Table 3 SD (type unclear)) |
| [Kortekaas et al 2018](https://doi.org/10.1371/journal.pone.0204105) | A | 20, healthy controls, 55 ± 4 | index finger L/R, foot = max 2nd derivative | Biopac MP100 + ECG100C + PPG100C/TSD200 (single DAQ 2 kHz) | 275.5 ± 12 (L), 7 (R) (Table 3 SD (type unclear)) |
| [Kortekaas et al 2018](https://doi.org/10.1371/journal.pone.0204105) | A | 20, CV risk factors (19 HTN), 63 ± 10 | index finger L/R, foot = max 2nd derivative | Biopac MP100 + ECG100C + PPG100C/TSD200 (single DAQ 2 kHz) | 282 ± 16 (L), 12 (R) (Table 3 SD (type unclear)) |
| [Obata et al 2017](https://doi.org/10.3389/fphys.2017.00047) | A | 11, healthy 6 M / 5 F, 30 ± 7 (23-41) | index finger, first positive inflection (onset) | ADInstruments PowerLab 4/26 + FE132 + MLT1020PPG (single DAQ 1 kHz) | 200 ± 10 (SD) |
| [Jose et al 2025](https://doi.org/10.25259/IJPP_149_2024) | A | 21, healthy 10 M / 11 F, 29.6 ± 5.7 | right index finger, foot | ADInstruments PowerLab 4/35 + MLT1020PPG (single DAQ (fs not reported)) | 206.4 ± 3.04 (unclear (likely SE)) |
| [van Velzen et al 2017](https://doi.org/10.1007/s11517-017-1642-x) | A | 1, healthy (n=10 studied), 23-25 | index finger, foot = max 2nd derivative | Biopac ECG100C + PPG100C/TSD200, AcqKnowledge (single system implied 2 kHz) | 282 |
| [Mohammadpoor Faskhodi et al 2024](https://doi.org/10.1371/journal.pone.0298354) | A | 20, healthy 15 M / 5 F, 32.3 ± 11.8 | right middle finger, foot (also 6 others) | Biopac MP36E + SS4LA (single DAQ 5 kHz) | — ± 8.22 (within-recording SD (median)) |
| [Liao et al 2023](https://doi.org/10.3389/fphys.2023.1172150) | A | 100, healthy 48 M / 52 F, 44 ± 14 (20-71) | right index finger, 8 fiducials | Biopac MP160 (single DAQ 2.5 kHz) | — |
| [Martin et al 2016](https://doi.org/10.1038/srep39273) | A | 22, healthy 19 M, 25 ± 3.5 | index finger, foot (intersecting tangent) | Biopac MP150 + BN-EL50 + PPG100C/Nonin (single DAQ 2 kHz) | — |
| [Allen & Murray 2000](https://doi.org/10.1049/ip-smt:20000846) | B | 30, normal subjects, not verified | thumb pad, foot | custom Newcastle multi-site PPG + ECG (one computer; DAQ unnamed) | 199 ± 2.2 (within-subject CV % (value is median)) |
| [Allen & Murray 2002](https://doi.org/10.1038/sj.jhh.1001478) | B | 116, healthy, median 42 (13-72) | finger, not verified | custom Newcastle multi-site PPG + ECG (one computer; DAQ unnamed) | — |
| [Ruffolo et al 2025](https://doi.org/10.1088/1361-6579/ae1b0c) | C-corrected | 7317, children (PICU) | unknown, onset (Kavsaoglu) | Philips MP70 + IntelliBridge X2 + Masimo | 143 (0-6 mo) to 207 (14-18 y), figure-read medians |
| [Lee et al 2019](https://doi.org/10.3390/jcm8111773) | C | 2309, adult surgical patients, not extracted | finger, intersecting tangent | GE Solar 8000M + Vital Recorder, 100 Hz (monitor; uncorrected delay) | 645 ± 26 (SD (between subjects)) |
| [Youn et al 2014](https://www.anesth-pain-med.org/journal/view.php?number=399) | C | 60, adult surgical patients, not extracted | not verified, not verified | patient monitor (not verified) (monitor; uncorrected delay) | 281 ± 4.5 (± reported, likely SE) |

Adults, finger PPG measured to the foot, awake: about 270 ms (tier A studies, 267–282 ms). Under general
anaesthesia a value in the upper half of 250–290 ms is consistent with the monitor-based change.
