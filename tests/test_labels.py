# SPDX-License-Identifier: MIT
import csv
import io
import json

import numpy as np
from test_measurement import FRAME, recording

from physiosync import (
    LABEL_FIELDS,
    TierRules,
    fit_ecg_ppg_delay,
    pat_measurement_model,
    sawtooth_label_tuples,
    sawtooth_labels,
)


def labels_for(**kw):
    r, p, _phys = recording(**kw)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=kw.get("rate_ppm", 50.0),
                          tier_rules=TierRules(inserted_max_tier=None))
    rec = pat_measurement_model(m, ecg_fs=500, ppg_fs=500, ppg_resolution_s=FRAME, ppg_frame_lock=0.1, r_t=r, ppg_t=p)
    return m, sawtooth_labels(rec)


def test_one_interval_per_tooth_with_phase_and_slope():
    m, lab = labels_for()
    assert len(lab) == len(m.tooth_edges) - 1
    assert [d["start"] for d in lab] == m.tooth_edges[:-1]
    inner = lab[1:-1]
    assert all(d["phase_start"] == 0.0 and d["phase_end"] == 1.0 for d in inner)
    assert all(abs(d["slope_ppm"] - 50) < 1.5 for d in inner)
    assert all(abs(d["tooth_s"] - FRAME / 50e-6) < 30 for d in inner)
    # the first tooth ends at a slip: its start phase is how far into a tooth the recording began
    assert 0.0 <= lab[0]["phase_start"] < 1.0 and lab[0]["phase_end"] == 1.0
    assert all(d["amplitude_ms"] == round(FRAME * 1000, 3) for d in lab)


def test_resolution_and_confidence_fields():
    _, lab = labels_for(seed=2)
    d = lab[1]
    assert d["ecg_error_model"] == "rounding" and d["ecg_error_ms"] == 2.0
    assert d["ppg_error_model"] == "rounding" and d["ppg_error_ms"] == round(FRAME * 1000, 3)
    assert d["ppg_sample_ms"] == 2.0 and d["ppg_frame_lock"] == 0.1
    # rounding error of the PAT: uniform over 2 ms and 33.3 ms
    assert abs(d["fiducial_sd_ms"] - np.hypot(2, FRAME * 1000) / np.sqrt(12)) < 0.01
    assert d["tier"] in "AB" and d["usable_for_pat"] and d["sync_sd_ms"] < 2
    assert d["slip_start"] == "detected" and d["beat_noise_ms"] > 1


def test_paused_end_is_flagged_with_large_error():
    _, lab = labels_for(seed=3, minutes=120, stop_min=50)
    last = lab[-1]
    assert last["state"] == "paused" and last["tier"] == "N" and last["phase_end"] is None
    assert last["sync_error_model"] == "creep_then_hold" and last["sync_p95_ms"] > 10
    # the caller can refuse intervals whose error could exceed a limit
    r, p, _ = recording(seed=3, minutes=120, stop_min=50)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    strict = sawtooth_labels(pat_measurement_model(m, ecg_fs=500, ppg_fs=500), max_error_s=0.005)
    assert not strict[-1]["usable_for_pat"]


def test_csv_json_and_tuples():
    _, lab = labels_for(seed=4)
    json.dumps(lab, allow_nan=False)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LABEL_FIELDS)
    w.writeheader()
    w.writerows(lab)
    assert buf.getvalue().count("\n") == len(lab) + 1
    tup = sawtooth_label_tuples(lab)
    assert tup[1][0].startswith("ppg_sawtooth:") and tup[1][1:] == (lab[1]["start"], lab[1]["end"])


def test_per_recording_fiducial_error_model():
    from physiosync import fiducial_error_model
    rng = np.random.default_rng(0)
    fs, fr = 500.0, 500.0 / 30
    beats = np.cumsum(rng.uniform(350, 450, 2000))
    # on the frame grid: rounded to the frame
    locked = np.round(np.round(beats / fr) * fr).astype(int)
    e = fiducial_error_model(locked, fs, frame_samples=fr, pat_noise_s=0.004)
    assert e["rounding"] == "frame" and abs(e["resolution_s"] - 1 / 30) < 1e-9
    # off the grid: jitter from the PAT noise, less the ECG's part
    free = np.round(beats).astype(int)
    e = fiducial_error_model(free, fs, frame_samples=fr, pat_noise_s=0.004, other_sd_s=0.002 / np.sqrt(12))
    assert e["kind"] == "normal" and 0.0039 < e["sd_s"] < 0.004
    # into the labels
    r, p, _ = recording(seed=6)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    lab = sawtooth_labels(pat_measurement_model(m, ecg_fs=500, ppg_fs=500, ppg_fiducial="upstroke", ppg_error=e))
    assert lab[1]["ppg_error_model"] == "jitter" and abs(lab[1]["ppg_error_ms"] - e["sd_s"] * 1000) < 1e-3
    assert lab[1]["ppg_fiducial"] == "upstroke"


def test_setup_offsets_keep_the_recordings_own_level():
    from physiosync import offset_label, setup_offsets
    # three recordings on one setup with different physiology: the setup's delay is removed, differences stay
    levels, setups, data = {}, {}, {}
    for k, extra in enumerate((0.0, 0.02, -0.015)):
        r, p, _ = recording(seed=20 + k, offset=0.38 + extra)
        m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
        levels[k], setups[k], data[k] = m.qc["raw_level_s"], "COM4 step", (r, p)
    so = setup_offsets(levels, setups, adopted_pat_s=0.27)
    off = so["setups"]["COM4 step"]["offset_s"]
    med = []
    for k, (r, p) in data.items():
        m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0, offset_s=off)
        med.append(np.median(m.correct(p) - r))
        lab = offset_label(pat_measurement_model(m, ecg_fs=500, ppg_fs=500), basis="setup COM4 step", n_recordings=3)
        assert lab["offset_ms"] == round(off * 1000, 2) and lab["basis"] == "setup COM4 step"
    assert abs((med[1] - med[0]) - 0.02) < 0.003 and abs((med[2] - med[0]) + 0.015) < 0.003
    # a setup with too few recordings falls back to all recordings
    so2 = setup_offsets({**levels, 9: 0.65}, {**setups, 9: "COM9"}, adopted_pat_s=0.27)
    assert so2["by_recording"][9]["basis"] == "all recordings"
