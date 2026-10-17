# SPDX-License-Identifier: MIT
import numpy as np

from physiosync import corrected_times, stamp_clock


def _simulate(eps_ppm, *, fs=500.0, rec=16, hours=3.0, thr=0.1, gap_at_h=1.5, gap_s=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n_rec = int(hours * 3600 * fs / rec)
    starts = np.arange(n_rec) * rec
    eps = eps_ppm * 1e-6
    true = starts / fs * (1 + eps)
    g = int(n_rec * gap_at_h / hours)
    true[g:] += gap_s                                   # real gap: no samples for gap_s
    stamps = np.empty(n_rec)
    anchor = 0.0
    for i in range(n_rec):
        if i == g:
            anchor = true[i] - starts[i] / fs           # recorder re-anchors after a gap
        s = anchor + starts[i] / fs
        err = true[i] - s
        if abs(err) > thr:
            anchor += thr * np.sign(err)
            s += thr * np.sign(err)
        stamps[i] = s
    flick = rng.choice(n_rec - 2, 20, replace=False) + 1
    stamps[flick] += 0.03                               # single records stamped late
    return starts, stamps, true


def test_recovers_rate_and_removes_sawtooth():
    for eps, hours in ((-65.0, 3.0), (20.0, 8.0), (50.0, 3.0)):
        starts, stamps, true = _simulate(eps, hours=hours, gap_at_h=hours / 2)
        c = stamp_clock(stamps, starts, 500.0)
        assert c["rate_source"] == "teeth"
        assert abs(c["rate_ppm"] - eps) < 0.5
        assert c["gaps"] == 1
        raw = stamps - true
        fix = corrected_times(c, starts) - true
        assert np.ptp(raw) > 0.09                       # the sawtooth was there
        assert np.ptp(fix) < 0.002                      # one constant offset left


def test_few_corrections_use_given_rate():
    starts, stamps, true = _simulate(20.0)            # 3 h at 20 ppm: no whole tooth
    c = stamp_clock(stamps, starts, 500.0)
    assert c["rate_source"] == "fit"
    g = stamp_clock(stamps, starts, 500.0, rate_ppm=20.0)
    assert g["rate_source"] == "given"
    assert np.ptp(corrected_times(g, starts) - true) < 0.002


def test_between_devices_offset_constant():
    s1, t1, true1 = _simulate(-65.0, rec=16, thr=0.12, seed=1)
    s2, t2, true2 = _simulate(45.0, rec=32, thr=0.2, seed=2, hours=6.0, gap_at_h=3.0)
    c1, c2 = stamp_clock(t1, s1, 500.0), stamp_clock(t2, s2, 500.0)
    # the same true instants, seen by each device
    q = np.linspace(100, 2.9 * 3600, 50)
    def at(c, s, true):
        i = np.searchsorted(true, q) - 1
        return corrected_times(c, s)[i] + (q - true[i])
    raw = lambda s, t, true: t[np.searchsorted(true, q) - 1] + (q - true[np.searchsorted(true, q) - 1])
    d_raw = raw(s1, t1, true1) - raw(s2, t2, true2)
    d_fix = at(c1, s1, true1) - at(c2, s2, true2)
    assert np.std(d_fix) < 0.05 * np.std(d_raw)


def test_per_sample_times():
    starts, stamps, _ = _simulate(30.0, hours=0.2)
    c = stamp_clock(stamps, starts, 500.0)
    per = corrected_times(c, starts, np.full(starts.size, 16))
    assert per.size == starts.size * 16
    assert np.allclose(per[::16], corrected_times(c, starts))


def test_correction_sign_makes_opposite_steps_gaps():
    starts, stamps, true = _simulate(-65.0)
    stamps = stamps.copy()
    stamps[stamps.size // 3:] += 0.05                  # a short real gap (positive) in a device correcting downwards
    true = true.copy()
    true[true.size // 3:] += 0.05
    c = stamp_clock(stamps, starts, 500.0, correction_sign=-1)
    assert c["gaps"] == 2
    assert abs(c["rate_ppm"] + 65.0) < 0.5


def test_samples_lost_at_corrections():
    """A device that drops samples at each re-synchronisation: count time falls behind true time there."""
    fs, rate, loss, thr = 500.0, -120e-6, 0.018, 0.5
    n_rec, size = 90_000, 100                       # 300 min of 100-sample records
    drop = round(loss * fs)                         # 9 samples dropped at each resync
    true_rec, stamps, starts = [], [], []
    t_true, k, anchor_true, anchor_k, dropped = 0.0, 0, 0.0, 0, 0
    for _ in range(n_rec):
        stamp = anchor_true + (k - anchor_k) / fs   # stamps follow the count from the last anchor
        if stamp - t_true >= thr:                   # count has run ahead by the threshold: re-anchor
            t_true += drop / fs * (1 + rate)        # the dropped samples' time passes unrecorded
            anchor_true, anchor_k, stamp = t_true, k, t_true
            dropped += 1
        true_rec.append(t_true); stamps.append(stamp); starts.append(k)
        k += size
        t_true += size / fs * (1 + rate)
    true_rec, stamps, starts = map(np.asarray, (true_rec, stamps, starts))
    assert dropped >= 2
    plain = stamp_clock(stamps, starts, fs, correction_sign=-1)
    fixed = stamp_clock(stamps, starts, fs, correction_sign=-1, event_loss_s=loss)
    err_plain = corrected_times(plain, starts) - true_rec
    err_fixed = corrected_times(fixed, starts) - true_rec
    assert np.ptp(err_plain) > 0.8 * loss                 # without it: the rate is biased and an 18 ms sawtooth stays
    assert np.ptp(err_fixed) < 0.001
    assert abs(fixed["rate_ppm"] - rate * 1e6) < 0.5 < abs(plain["rate_ppm"] - rate * 1e6)
    assert len(fixed["loss_records"]) == dropped
