# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from physiosync import fiducial_resolution, locking, period_scan, pool

FS = 500.0


def beats(n=3000, seed=0, hr=75, trend=0.05):
    """True (continuous) beat times in samples: HRV plus a slow trend."""
    rng = np.random.default_rng(seed)
    rr = 60 / hr * FS * (1 + trend * np.sin(np.arange(n) / 40) + 0.03 * rng.standard_normal(n))
    return np.cumsum(rr) + 1000.0


def on_grid(t, k, phase=0.0, drift_ppm=0.0):
    """Sample index of each event when the signal only changes every k samples (upstream block of k),
    the grid's phase drifting relative to the sampling clock by drift_ppm."""
    ph = phase + t * drift_ppm * 1e-6
    return (np.ceil((t - ph) / k) * k + np.floor(ph)).astype(np.int64)


def test_no_grid_is_one_sample():
    r = fiducial_resolution(np.ceil(beats()).astype(int), fs=FS)
    assert r.effective_samples == 1
    assert abs(r.strength[2]) < 0.05
    assert r.effective_s == pytest.approx(0.002)


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_fixed_grid(k):
    r = fiducial_resolution(on_grid(beats(), k), fs=FS)
    assert r.effective_samples == k
    assert r.share[k] == 1.0


def test_grid_with_drifting_phase_still_found():
    # 4-sample blocks from a clock 50 ppm off: phase walks one sample every 40 s; IBIs stay multiples of 4
    r = fiducial_resolution(on_grid(beats(), 4, drift_ppm=50), fs=FS)
    assert r.effective_samples == 4


def test_mostly_on_grid():
    rng = np.random.default_rng(1)
    idx = on_grid(beats(), 2)
    off = rng.random(idx.size) < 0.05
    idx[off] += rng.choice([-1, 1], off.sum())
    r = fiducial_resolution(np.sort(idx), fs=FS)
    assert r.effective_samples == 2
    assert 0.75 < r.strength[2] < 0.95


def test_ibi_diff_matches_paper_style():
    r = fiducial_resolution(on_grid(beats(), 2), quantity="ibi_diff")
    assert r.share[2] == 1.0 and r.effective_samples == 2
    r1 = fiducial_resolution(np.ceil(beats()).astype(int), quantity="ibi_diff")
    assert r1.effective_samples == 1


def test_narrow_differences_not_mistaken_for_grid():
    # very regular rhythm: IBI differences within +/-2 samples; large k is untestable, not a grid
    rng = np.random.default_rng(2)
    t = np.cumsum(400 + 0.7 * rng.standard_normal(3000))
    r = fiducial_resolution(np.round(t).astype(int), quantity="ibi_diff")
    assert r.effective_samples == 1
    assert not r.testable[8]


def test_non_integer_period():
    # an upstream rate of 128 Hz sampled at 500 Hz: blocks of 3.906 samples
    p = FS / 128
    idx = (np.ceil(beats() / p) * p).astype(np.int64)
    r = fiducial_resolution(idx)
    periods = np.array(r.periods)
    best = periods[np.argmax(np.where(periods > 2.5, r.resultant, 0))]
    assert abs(best - p) < 0.1
    assert period_scan(np.arange(0, 400, 4), np.array([4.0]))[0] == pytest.approx(1.0)


def test_segments_and_range():
    idx = np.ceil(beats(200)).astype(int)
    seg = np.r_[np.zeros(100, int), np.ones(100, int)]
    r = fiducial_resolution(idx, segments=seg, ibi_range=(100, 1000), min_n=10)
    assert r.n == 198


def test_too_few():
    assert fiducial_resolution(np.arange(0, 4000, 400)).effective_samples == 1


def test_pool_matches_single_run():
    idx = on_grid(beats(4000), 2)
    a = fiducial_resolution(idx[:2000], fs=FS)
    b = fiducial_resolution(idx[2000:], fs=FS)
    p = pool([a, b.to_dict()])
    assert p.n == a.n + b.n
    assert p.effective_samples == 2 and p.share[2] == 1.0
    assert p.effective_s == pytest.approx(0.004)


def test_pool_rejects_mixed_quantities():
    idx = np.ceil(beats()).astype(int)
    with pytest.raises(ValueError):
        pool([fiducial_resolution(idx), fiducial_resolution(idx, quantity="ibi_diff")])


def test_partial_locking_to_hum_is_reported_not_a_grid():
    # 30% of fiducials pulled to the nearest crest of 60 Hz hum (8.33 samples at 500 Hz)
    rng = np.random.default_rng(3)
    t = beats(6000, trend=0.2)  # a heart rate that varies over the recording, as in surgery
    p = FS / 60
    lock = rng.random(t.size) < 0.3
    t[lock] = np.round(t[lock] / p) * p
    r = fiducial_resolution(np.ceil(t).astype(int), fs=FS)
    assert r.effective_samples == 1
    top = r.locking[0]
    # the same with an unlocked detector of the same beats as reference
    ref = np.ceil(beats(6000, trend=0.2) + 3).astype(int)
    rr = fiducial_resolution(np.ceil(t).astype(int), fs=FS, reference_idx=ref)
    assert rr.locking_reference and abs(rr.locking[0]["period_samples"] - p) < 0.1
    assert abs(top["period_samples"] - p) < 0.1 and abs(top["freq_hz"] - 60) < 1
    # an interval sits on the comb only when both of its fiducials are locked: r ~ 0.3**2
    assert 0.06 < top["resultant"] < 0.13


def test_no_locking_without_structure():
    r = fiducial_resolution(np.ceil(beats(6000)).astype(int), fs=FS)
    assert r.locking == []


def test_reference_removes_shape_false_positives():
    # a narrow, lumpy IBI distribution (two rhythms) has r > 0 at long periods from its shape alone;
    # an unlocked reference timing the same beats cancels it
    rng = np.random.default_rng(4)
    rr = np.where(rng.random(3000) < 0.5, 380.0, 410.0) + 2 * rng.standard_normal(3000)
    t = np.cumsum(rr)
    test = fiducial_resolution(np.round(t).astype(int), fs=FS, max_period=40)
    ref = fiducial_resolution(np.round(t + 0.4 * rng.standard_normal(t.size)).astype(int), fs=FS, max_period=40)
    assert locking(test, ref) == []
