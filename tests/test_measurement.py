# SPDX-License-Identifier: MIT
import json

import numpy as np

from physiosync import TierRules, error_samples, fit_ecg_ppg_delay, pat_measurement_model, simulate_pat

FRAME = 1 / 30


def recording(minutes=90, rate_ppm=50.0, offset=0.38, seed=0, noise=0.002, phase=0.4, stop_min=None):
    """R and PPG times: physiological PAT + offset + frame-slip sawtooth; the creep (and so the slips)
    stops ``stop_min`` minutes in, if given."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 / 0.8)
    r = 100 + np.cumsum(0.8 * (1 + 0.02 * rng.standard_normal(n)))
    phys = 0.27 + 0.006 * np.sin((r - r[0]) / 500)
    el = r - r[0]
    if stop_min is not None:
        el = np.minimum(el, stop_min * 60)
    creep = rate_ppm * 1e-6 * el + phase * FRAME
    saw = FRAME * np.floor(creep / FRAME) - creep
    p = r + phys + offset + saw + noise * rng.standard_normal(n)
    return r, p, phys


def fitted(**kw):
    r, p, phys = recording(**kw)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=kw.get("rate_ppm", 50.0),
                          tier_rules=TierRules(inserted_max_tier=None))
    rec = pat_measurement_model(m, ecg_fs=500, ppg_fs=500, ecg_fiducial="R", ppg_fiducial="upstroke",
                                adopted_pat_sd_s=0.01, r_t=r, ppg_t=p)
    return r, p, phys, m, rec


def test_record_is_json_and_complete():
    _r, _p, _phys, m, rec = fitted()
    s = json.dumps(rec, allow_nan=False)
    back = json.loads(s)
    assert back["schema"].startswith("physiosync.pat_measurement/")
    assert len(back["segments"]) == len(m.tooth_edges) - 1
    assert all(seg["sync_error"]["kind"] == "corrected" for seg in back["segments"])
    assert back["beat_noise"]["sd_s"] > 0.001
    assert back["fiducials"]["ecg"]["resolution_s"] == 1 / 500


def test_applied_shift_reproduces_the_correction():
    # the record alone gives back the raw PAT: corrected + the applied shift, piecewise linear
    r, _p, _phys, m, rec = fitted(seed=1)
    sim = simulate_pat(rec, r, 0.0, corrected=False, components=())[0]
    assert np.max(np.abs(sim - m.shift(r))) < 1e-6


def test_corrected_segments_error_is_small_and_quantisation_uniform():
    r, _p, _phys, _m, rec = fitted(seed=2)
    e = error_samples(rec, r[:50], 4000, rng=np.random.default_rng(0), components=("quantisation",))
    assert abs(e.mean()) < 1e-4 and abs(e.std() - np.sqrt(2 / 12) * 0.002) < 1e-4
    e = error_samples(rec, r[:50], 2000, rng=np.random.default_rng(0), components=("sync",))
    assert e.std() < 0.001


def test_paused_end_is_modelled_and_covers_the_truth():
    r, p, phys, m, rec = fitted(seed=3, minutes=120, stop_min=50)
    last = rec["segments"][-1]
    assert last["tier"] == "N" and last["state"] == "paused"
    assert last["sync_error"]["kind"] == "creep_then_hold"
    sel = r >= last["start"]
    err = m.correct(p) - r - phys
    # the recording's absolute level is not measured (snapped): errors are relative to the corrected teeth
    ref = np.median(err[(r >= rec["segments"][1]["start"]) & (r < rec["segments"][-2]["end"])])
    actual = (err - ref)[sel]                          # what the stored data's error really is
    sims = error_samples(rec, r[sel], 2000, rng=np.random.default_rng(1), components=("sync",))
    lo, hi = np.quantile(sims, [0.01, 0.99], axis=0)
    # the true error (plus beat noise) lies inside the modelled band for nearly all beats
    assert np.mean((actual > lo - 0.006) & (actual < hi + 0.006)) > 0.95
    # and the model is informative: the band is at most about a frame wide
    assert np.median(hi - lo) <= FRAME + 1e-3


def test_likelihood_from_record_covers_latent_in_a_paused_window():
    # a downstream MLE: window of beats, shared sync error integrated by Monte Carlo, per-beat noise normal
    r, p, phys, m, rec = fitted(seed=4, minutes=120, stop_min=50)
    last = rec["segments"][-1]
    idx = np.flatnonzero(r >= last["start"] + 300)[:150]
    err = m.correct(p) - r - phys
    ref = np.median(err[r < last["start"]])            # the snapped level, common to the recording
    obs = (m.correct(p) - r)[idx] - ref
    latent = float(np.mean(phys[idx]))
    sync = error_samples(rec, r[idx], 3000, rng=np.random.default_rng(2), components=("sync",))
    sd = rec["beat_noise"]["sd_s"]
    grid = np.linspace(latent - 0.05, latent + 0.05, 401)
    ll = np.array([np.logaddexp.reduce(-0.5 * (((obs - L - sync) / sd) ** 2).sum(1)) for L in grid])
    post = np.exp(ll - ll.max())
    post /= post.sum()
    cdf = np.cumsum(post)
    lo, hi = grid[np.searchsorted(cdf, 0.01)], grid[np.searchsorted(cdf, 0.99)]
    assert lo - 0.004 <= latent <= hi + 0.004


def test_offset_only_and_no_sawtooth():
    r, p, _ = recording(seed=5, minutes=30)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, sawtooth=False)
    rec = pat_measurement_model(m, ecg_fs=500, ppg_fs=500)
    assert [s["sync_error"]["kind"] for s in rec["segments"]] == ["none"]
    json.dumps(rec, allow_nan=False)
