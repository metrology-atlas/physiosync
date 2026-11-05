# SPDX-License-Identifier: MIT
"""Tests of physiosync.slips and fit_ecg_ppg_delay(method="rebuilt") on synthetic ECG and PPG."""

import numpy as np
import pytest

import physiosync as ps

FRAME = 1 / 30


def recording(hours=1.5, L=900.0, first=400.0, sign=1, slips=True, seed=0, fs=500.0):
    """ECG and PPG waveforms with a frame-slip sawtooth in the PPG; the fiducials, the beat quality and the true delay."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / 0.85)
    r = 5.0 + np.cumsum(0.85 + 0.03 * np.sin(np.arange(n) * 2 * np.pi / 18) + rng.normal(0, 0.01, n))
    t = np.arange(0, r[-1] + 5.0, 1 / fs)
    k = np.where(r < first, 0, np.floor((r - first) / L) + 1) if slips else np.zeros(n)
    D = sign * FRAME * (k - r / L) if slips else np.zeros(n)
    onset = r + 0.20 + D + 0.004 * np.sin(2 * np.pi * r / 60) + rng.normal(0, 0.002, n)
    ecg, ppg = rng.normal(0, 0.02, t.size), rng.normal(0, 0.005, t.size)
    qrs, pulse = np.arange(-int(0.05 * fs), int(0.05 * fs) + 1), np.arange(int(1.2 * fs))
    for rb, ob in zip(r, onset):
        i = round(rb * fs)
        ecg[i + qrs] += np.exp(-0.5 * ((t[i + qrs] - rb) / 0.01) ** 2)
        j = int(np.floor(ob * fs))
        tau = t[j + pulse] - ob
        ppg[j + pulse] += np.where(tau < 0.15, np.sin(np.pi / 2 * np.clip(tau, 0, None) / 0.15) ** 2,
                                   np.exp(-(tau - 0.15) / 0.3))
    fid = {"foot": onset + rng.normal(0, 0.002, n), "up": onset + 0.075 + rng.normal(0, 0.002, n),
           "peak": onset + 0.15 + rng.normal(0, 0.003, n), "down50": onset + 0.15 + 0.3 * np.log(2) + rng.normal(0, 0.003, n)}
    return r, fid, ps.beat_quality(r, 0.0, ecg, ecg, 0.0, ppg, ppg, fs=fs), D


def fit(r, fid, q, **kw):
    return ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, matched=True, method="rebuilt", fiducials=fid,
                                beat_quality=q, **kw)


def err_ms(m, r, D):
    s = m.shift(r)
    return np.abs((s - np.median(s)) - (D - np.median(D))) * 1e3


def test_rebuilt_finds_the_sawtooth_and_removes_it():
    r, fid, q, D = recording()
    m = fit(r, fid, q)
    assert m.method == "rebuilt" and m.qc["sawtooth_fitted"] and m.direction == "up"
    assert m.qc["tooth_s"] == pytest.approx(900, rel=0.01)
    true = 400 + 900 * np.arange(6)
    assert len(m.slips) == 6 and np.all(np.abs(np.array(m.slips_t) - true) < 2.0)
    assert {q_["tier"] for q_ in m.quality_intervals()} == {"A"}
    assert np.percentile(err_ms(m, r, D), 95) < 1.0
    assert m.qc["sawtooth_p2p_corrected_ms"] < 0.2 * m.qc["sawtooth_p2p_raw_ms"]
    assert np.median(fid["up"] - r - m.shift(r)) == pytest.approx(0.25, abs=1e-6)


def test_rebuilt_reversed_sawtooth():
    r, fid, q, D = recording(sign=-1, seed=1)
    m = fit(r, fid, q)
    assert m.direction == "down" and all(abs(s + 1) < 1e-3 for s in m.tooth_steps if abs(s) > 1e-3)
    assert np.percentile(err_ms(m, r, D), 95) < 1.0


def test_rebuilt_without_slips_is_tier_n_and_offset_only():
    r, fid, q, _ = recording(slips=False, seed=2)
    m = fit(r, fid, q)
    assert not m.qc["sawtooth_fitted"] and not m.slips
    assert [x["tier"] for x in m.quality_intervals()] == ["N"]
    assert np.ptp(m.shift(r)) == 0.0


def test_rebuilt_needs_matched_beats_fiducials_and_quality():
    r, fid, q, _ = recording(hours=0.3)
    with pytest.raises(ValueError, match="matched"):
        ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, method="rebuilt", fiducials=fid, beat_quality=q)
    with pytest.raises(ValueError, match="fiducials and beat_quality"):
        ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, matched=True, method="rebuilt")
    with pytest.raises(ValueError, match="frame"):
        fit(r, fid, q, frame_s=0.04)
    with pytest.raises(ValueError, match="method"):
        ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, matched=True, method="other")


def test_detect_stays_the_default():
    r, fid, _, _ = recording(hours=0.5)
    m = ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, matched=True, frame_s=FRAME)
    assert m.method == "detect" and m.tooth_steps == [] and m.slips == [] and "I" not in m.qc["tier_time_s"]


RESULT = {"start": 0.0, "end": 4000.0, "sawtooth": True, "sign": 1, "fit": {"L": 1000.0},
          "slips": [{"t": 500.0, "kind": "slip", "sign": 1}, {"t": 1500.0, "kind": "step", "sign": 1},
                    {"t": 3500.0, "kind": "slip", "sign": 1}],
          "uncertain": [{"q50": 2500.0, "p_slip": 0.7, "q05": 2400.0, "q95": 2600.0, "outcome": "uncertain"}]}


def test_delay_model_steps_creep_ramp_and_tiers():
    dm = ps.delay_model(RESULT)
    assert dm["edges"] == [0.0, 500.0, 1500.0, 2400.0, 2500.0, 2600.0, 3500.0, 4000.0]
    assert dm["steps"] == pytest.approx([0, 1, 1, 0, 0, 0, 1], abs=1e-6)
    r = FRAME / 1000
    assert dm["rates"] == pytest.approx([r, r, r, r - 0.7 * FRAME / 200, r - 0.7 * FRAME / 200, r, r], rel=1e-3)
    assert dm["tiers"] == ["A", "B", "C", "C", "C", "C", "A"]


def test_each_tooth_closes_exactly():
    R = dict(RESULT, end=5000.0, uncertain=[], slips=[{"t": 1000.0, "kind": "slip", "sign": 1},
                                                      {"t": 2050.0, "kind": "slip", "sign": 1},
                                                      {"t": 3000.0, "kind": "slip", "sign": 1}])
    d = ps.delay_at(np.array([1000.0, 2049.999, 2050.0, 2999.999]), R)
    assert d[1] - d[0] == pytest.approx(-FRAME, abs=1e-6)        # a 1,050 s tooth creeps one frame
    assert d[3] - d[2] == pytest.approx(-FRAME, abs=1e-6)        # and a 950 s one


def test_isolated_slips_and_two_slip_teeth_are_not_corrected():
    for R in (dict(RESULT, sawtooth=False, uncertain=[], drift={"ppm": 6.5, "sd_ppm": 2.0}),
              dict(RESULT, fit={"L": 1000.0, "creep": "reported"}, uncertain=[])):
        dm = ps.delay_model(R)
        assert all(x == 0 for x in dm["steps"]) and all(x == 0 for x in dm["rates"])
        assert set(dm["tiers"]) == {"I"} and "not corrected" in dm["reasons"][0]


def test_unresolved_teeth_are_held_flat():
    R = dict(RESULT, start=0.0, end=9000.0, uncertain=[], slips=[{"t": 3000.0, "kind": "slip", "sign": 1},
                                                                  {"t": 4000.0, "kind": "slip", "sign": 1}])
    tiers = {x["tier"] for x in ps.tooth_tiers(R)}
    assert "U" in tiers
    d = ps.delay_at(np.array([100.0, 2500.0, 3100.0, 3900.0, 5000.0, 8000.0]), R)
    assert d[0] == d[1] and d[4] == d[5]                          # the long teeth at both ends: not corrected
    assert d[3] - d[2] == pytest.approx(-FRAME * 0.8, abs=1e-6)   # the tooth between the slips creeps


def test_anchor_check_is_on_by_default():
    assert ps.SlipOptions().anchor_check == "on"


def test_stored_result_gives_the_same_correction():
    r, fid, q, _ = recording(hours=0.6, seed=4)
    m = fit(r, fid, q)
    R = ps.find_slips(r, fid, q)
    m2 = ps.fit_ecg_ppg_delay(r, fid["up"], adopted_pat_s=0.25, method="rebuilt", slip_result=R)
    t = np.linspace(r[0], r[-1], 500)
    assert np.max(np.abs(m.shift(t) - m2.shift(t))) < 1e-9


def test_rebuilt_measurement_record_and_labels():
    r = np.arange(0.0, 4000.0, 0.8)
    R = {"start": 0.0, "end": 3999.0, "sawtooth": True, "sign": 1, "fit": {"L": 1000.0}, "uncertain": [],
         "slips": [{"t": 300.0, "kind": "slip", "sign": 1}, {"t": 3800.0, "kind": "slip", "sign": 1}]}
    m = ps.fit_ecg_ppg_delay(r, r + 0.3, adopted_pat_s=0.25, method="rebuilt", slip_result=R)
    rec = ps.pat_measurement_model(m, ecg_fs=500, ppg_fs=500, ecg_fiducial="R", ppg_fiducial="up")
    labels = ps.sawtooth_labels(rec)
    assert [x["tier"] for x in labels] == ["A", "U", "A"]           # one label per tooth, slip to slip
    assert labels[1]["sync_error_model"] == "sawtooth_unknown_phase" and labels[1]["shift_slope_ppm"] == 0.0
    assert labels[0]["sync_error_model"] == "corrected" and labels[0]["slope_ppm"] == pytest.approx(33.33, abs=0.01)
