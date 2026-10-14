# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from physiosync import pair_beats

pytest.importorskip("peakalignment")


def beats(n=1500, rr=0.5, seed=0, hrv=0.03, delay=0.65):
    rng = np.random.default_rng(seed)
    r = 10 + np.cumsum(rr * (1 + hrv * np.sin(np.arange(n) / 7) + 0.5 * hrv * rng.standard_normal(n)))
    p = r + delay + 0.004 * rng.standard_normal(n)
    return r, p


def test_delay_longer_than_a_beat_is_paired_correctly():
    r, p = beats()                       # PAT 650 ms, R-R about 500 ms
    res = pair_beats(r, p)
    assert abs(np.median(res["delay_s"]) - 0.65) < 0.01
    assert res["p_index"].size > 0.95 * p.size
    assert np.all(res["r_index"] == res["p_index"])


def test_missed_and_extra_beats_change_the_shift_locally():
    r, p = beats(seed=1)
    p = np.delete(p, [300, 900])          # two pulses missed
    p = np.sort(np.r_[p, r[600] + 0.3])   # one false pulse
    res = pair_beats(r, p)
    d = res["delay_s"]
    assert np.mean(np.abs(d - 0.65) < 0.02) > 0.97


def test_regular_rhythm_is_ambiguous_without_an_expected_delay():
    r, p = beats(hrv=0.0, seed=2)
    res = pair_beats(r, p)
    assert res["qc"]["ambiguous"] > 0.5 * res["qc"]["windows"]
    res2 = pair_beats(r, p, expected_delay_s=(0.55, 0.75))
    assert abs(np.median(res2["delay_s"]) - 0.65) < 0.01


def test_window_fallback_pairs_what_the_rhythm_cannot():
    r, p = beats(hrv=0.0, seed=3)
    res = pair_beats(r, p, expected_delay_s=(0.55, 0.75))
    assert res["p_index"].size > 0.98 * p.size and set(res["source"]) <= {"rhythm", "window"}
    assert np.all(np.abs(res["delay_s"] - 0.65) < 0.02)
