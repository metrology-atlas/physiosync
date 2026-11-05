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
    assert m.direction == "down" and all(s == -1 for s in m.tooth_steps if s)
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


def test_delay_model_steps_creep_and_tiers():
    dm = ps.delay_model(RESULT)
    assert dm["edges"] == [0.0, 500.0, 1500.0, 2500.0, 3500.0, 4000.0]
    assert dm["steps"] == [0.0, 1.0, 1.0, 0.0, 1.0]  # the slip placed only within an interval is not a step
    assert dm["rates"] == pytest.approx([FRAME / 1000] * 5)
    assert dm["tiers"] == ["A", "B", "C", "C", "A"]


def test_delay_model_isolated_slips_are_steps_only():
    R = dict(RESULT, sawtooth=False, uncertain=[], drift={"ppm": 6.5, "sd_ppm": 2.0})
    dm = ps.delay_model(R)
    assert dm["steps"] == [0.0, 1.0, 1.0, 1.0] and dm["rates"] == [0.0] * 4
    assert set(dm["tiers"]) == {"I"} and "drift +6.5 ppm" in dm["reasons"][0] and "not corrected" in dm["reasons"][0]


def test_delay_model_creep_limited_to_a_tooth_beyond_the_slips():
    R = dict(RESULT, start=0.0, end=9000.0, slips=[{"t": 3000.0, "kind": "slip", "sign": 1},
                                                   {"t": 4000.0, "kind": "slip", "sign": 1}], uncertain=[])
    dm = ps.delay_model(R)
    assert dm["edges"] == [0.0, 2000.0, 3000.0, 4000.0, 5000.0, 9000.0]
    assert dm["rates"][0] == 0.0 and dm["rates"][-1] == 0.0 and all(x > 0 for x in dm["rates"][1:4])


def test_delay_model_two_slip_tooth_reports_its_creep():
    R = dict(RESULT, start=0.0, end=9000.0, fit={"L": 1000.0, "creep": "reported"}, uncertain=[],
             slips=[{"t": 3000.0, "kind": "slip", "sign": 1}, {"t": 4000.0, "kind": "slip", "sign": 1}])
    dm = ps.delay_model(R)
    assert dm["steps"] == [0.0, 1.0, 1.0] and dm["rates"] == [0.0] * 3
    assert set(dm["tiers"]) == {"I"} and "creep not corrected" in dm["reasons"][0]


def test_anchor_check_is_on_by_default():
    assert ps.SlipOptions().anchor_check == "on"
