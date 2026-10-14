# SPDX-License-Identifier: MIT
import itertools

import numpy as np
import pytest

from physiosync import EcgPpgDelay, TierRules, fit_ecg_ppg_delay, match_beats
from physiosync.ecg_ppg import _fill_rates, _p2p

FRAME = 1 / 30


def recording(minutes=90, rate_ppm=50.0, offset=0.38, true_pat=0.27, seed=0, noise=0.003, phase=0.4, frame=None):
    """R-peak and PPG times with a physiological PAT, a constant offset and a frame-slip sawtooth."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 / 0.8)
    rr = 0.8 * (1 + 0.05 * np.sin(np.arange(n) / 50) + 0.02 * rng.standard_normal(n))
    r = 100 + np.cumsum(rr)
    phys = true_pat + 0.008 * np.sin((r - r[0]) / 600)
    # sawtooth: creeps down by rate, jumps up by one frame when a frame of creep has accumulated
    fr = frame or FRAME
    creep = rate_ppm * 1e-6 * (r - r[0]) + phase * fr
    saw = fr * np.floor(creep / fr) - creep
    p = r + phys + offset + saw + noise * rng.standard_normal(n)
    return r, p, phys, saw


def test_sawtooth_and_offset_removed():
    r, p, phys, _ = recording()
    # (synthetic: the inserted slips are right, so teeth they bound are corrected too)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, tier_rules=TierRules(inserted_max_tier=None))
    rates = np.array(m.tooth_rate) * 1e6
    assert m.qc["n_slips"] >= 7
    assert np.all(np.abs(rates - 50) < 1.5)
    corr = m.correct(p) - r
    assert abs(np.median(corr) - 0.27) < 0.001
    # what is left is physiology + noise: the sawtooth's ~33 ms swing is gone
    assert np.std(corr - phys) < 0.0045
    # slow wander that is not physiology: ~33 ms before, a couple of ms after
    assert _p2p(r, (p - r) - phys) > 25 and _p2p(r, corr - phys) < 3


def test_frame_estimated_when_not_given():
    r, p, _, _ = recording(seed=1)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.25, tier_rules=TierRules(inserted_max_tier=None))
    assert m.frame_s == pytest.approx(FRAME, abs=0.003)


def test_offset_only_snaps_median():
    r, p, _, _ = recording(seed=2)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.30, sawtooth=False)
    assert m.tooth_rate == [] and abs(np.median(m.correct(p) - r) - 0.30) < 1e-9


def test_adopted_pat_is_required():
    r, p, _, _ = recording(minutes=20)
    with pytest.raises((TypeError, ValueError)):
        fit_ecg_ppg_delay(r, p)  # type: ignore[call-arg]


def test_prior_used_when_no_complete_tooth():
    # 8 minutes at 50 ppm: at most one slip, no complete tooth; the setup's known rate fills in
    r, p, _, _ = recording(minutes=8, seed=3, phase=0.95)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    assert set(m.tooth_source) <= {"prior", "median of teeth", "tooth"}
    assert np.allclose(np.array(m.tooth_rate) * 1e6, 50)


def test_grid_inserts_missed_and_rejects_false_slips():
    r, p, _, saw = recording(seed=6)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    # pretend the detector missed one slip and fired on a physiological jump
    m2 = EcgPpgDelay(adopted_pat_s=0.27, offset_s=0.0, frame_s=FRAME, t_start=r[0], t_end=r[-1])
    m2.slips_t = [float(x) for x in np.r_[np.delete(true, 3), true[5] + 250.0]]
    _fill_rates(m2, r, p - r, None, 60.0)
    assert m2.qc["rate_source"] == "teeth"
    assert len(m2.inserted_t) == 1 and np.min(np.abs(true - m2.inserted_t[0])) < 5
    assert len(m2.qc["rejected_t"]) == 1 and abs(m2.qc["rejected_t"][0] - (true[5] + 250.0)) < 16
    assert np.allclose(np.array(m2.tooth_rate) * 1e6, 50, atol=0.5)
    assert len(m.slips_t) >= 7


def test_false_slips_in_real_series_do_not_change_rate():
    r, p, _, _ = recording(seed=5)
    # two physiological jumps of ~30 ms lasting a few minutes
    for t0 in (1500.0, 3900.0):
        sel = (r > t0) & (r < t0 + 200)
        p[sel] += 0.03
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    # a jump that passes as a slip splits a tooth in two; only those teeth change, the rest keep 50 ppm
    rates = np.array(m.tooth_rate[1:-1]) * 1e6
    assert np.sum(np.abs(rates - 50) < 1.5) >= len(rates) - 4


def test_timing_plan_segments():
    r, p, _, _ = recording(minutes=40, seed=4)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    plan = m.timing_plan(start_t=r[0], n=39 * 60 * 500, fs=500.0)
    end = r[0] + 39 * 60
    assert len(plan) == 1 + sum(r[0] < e < end for e in m.tooth_edges[1:-1])
    assert sum(s["count"] for s in plan) == (39 * 60 * 500)
    assert all(abs(s["fs"] - 500 / (1 + s["rate_ppm"] * 1e-6)) < 1e-6 for s in plan)
    # within a segment the planned times equal the model's correction
    s = plan[1]
    k = s["first"] + 1000
    assert s["start_t"] + 1000 / s["fs"] == pytest.approx(m.correct(np.array([r[0] + k / 500]))[0], abs=2e-6)


def test_match_beats_handles_delay_longer_than_a_beat():
    r = np.arange(0, 600, 0.5)                   # 120 bpm
    p = r + 0.65                                 # delay longer than one beat
    i, j, d = match_beats(r, p)
    assert abs(d - 0.65) < 0.004 and np.all(p[j] - r[i] == pytest.approx(0.65))


def test_slip_phase_jump_keeps_rate():
    # one slip comes half a tooth late (the frame clock re-synchronised); later slips keep the new
    # phase. Every slip is still one whole frame and the creep rate does not change.
    r, p, phys, saw = recording(seed=7)
    L = FRAME / 50e-6
    slips = list(r[0] + 300 + L * np.arange(3)) + list(r[0] + 300 + L * (2.5 + np.arange(6)))
    slips = [x for x in slips if x < r[-1]]
    d = FRAME * np.searchsorted(np.array(slips), r) - 50e-6 * (r - r[0])
    p2 = p - saw + d
    m = fit_ecg_ppg_delay(r, p2, adopted_pat_s=0.27, frame_s=FRAME)
    # the short tooth before the late slip gets a balancing rate (wrong there, by up to half a frame);
    # the regular teeth keep 50 ppm and the error stays inside that tooth
    rates = np.array(m.tooth_rate[1:-1]) * 1e6
    assert np.sum(np.abs(rates - 50) < 2) >= len(rates) - 1
    corr = m.correct(p2) - r
    assert _p2p(r, corr - phys) < 0.6 * FRAME * 1000


def test_gradual_change_is_not_a_slip():
    # a 33 ms PAT rise built up over ~40 beats (a physiological change), no sawtooth at all
    r, p, _, saw = recording(seed=8, rate_ppm=1e-9)
    ramp = np.clip((r - (r[0] + 1500)) / 30.0, 0, 1) * FRAME
    m = fit_ecg_ppg_delay(r, p - saw + ramp, adopted_pat_s=0.27, frame_s=FRAME)
    assert m.qc["n_kept"] == 0


def test_slips_without_rate_are_left_alone():
    # 25 minutes at 20 ppm: one slip, no prior, too short for a plausible shared slope -> unresolved
    r, p, _, _ = recording(minutes=25, rate_ppm=20.0, seed=9, phase=0.7)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_range_ppm=(30.0, 100.0))
    assert m.tooth_source == ["unresolved"] and m.qc["unresolved_slips_t"]
    assert np.allclose(m.shift(r), m.offset_s)


def test_measured_teeth_win_over_a_wrong_prior():
    # true creep 15 ppm but the setup prior says 50 ppm: the full tooth between two slips decides
    r, p, _, _ = recording(rate_ppm=15.0, seed=10, phase=0.1)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    assert "prior" not in " ".join(m.tooth_source)
    assert np.all(np.abs(np.array(m.tooth_rate) * 1e6 - 15) < 1)


def test_errors_do_not_accumulate():
    # a false slip in the middle: the corrected series wanders by about a frame at most, locally
    r, p, phys, _ = recording(seed=12)
    sel = r > r[0] + 2500
    p[sel] += FRAME          # a sharp, persistent 33 ms step that is not a frame slip
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    resid = m.correct(p) - r - phys
    assert np.ptp(resid[r > r[0] + 3200]) < 2.5 * FRAME


def test_no_slips_means_offset_only():
    r, p, _, saw = recording(seed=11)
    m = fit_ecg_ppg_delay(r, p - saw, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    assert m.tooth_rate == [0.0] or m.tooth_source == ["unresolved"]


def test_several_missed_slips_reinserted():
    # the detector missed every other slip in the second half: teeth of 1 and 2 units; the basic tooth
    # is still found and the missing ones put back
    r, p, _, saw = recording(seed=13)
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    m = EcgPpgDelay(adopted_pat_s=0.27, offset_s=0.0, frame_s=FRAME, t_start=r[0], t_end=r[-1])
    m.slips_t = [float(x) for i, x in enumerate(true) if i < 4 or i % 2 == 0]
    _fill_rates(m, r, p - r, None, 60.0)
    # the slip missed between two kept ones comes back, and the one missed after the last kept slip is
    # found by prediction
    assert len(m.inserted_t) == 2 and all(np.min(np.abs(true - x)) < 5 for x in m.inserted_t)
    assert len(m.qc["predicted_t"]) == 1
    assert np.allclose(np.array(m.tooth_rate[1:-1]) * 1e6, 50, atol=1.0)


def test_clean_recording_is_mostly_tier_a():
    r, p, _, _ = recording(seed=14)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    q = m.quality_intervals()
    t = m.qc["tier_time_s"]
    assert t["A"] / sum(t.values()) > 0.7
    assert q[0]["start"] == m.t_start and q[-1]["end"] == m.t_end
    assert all(a["end"] == b["start"] for a, b in itertools.pairwise(q))
    assert m.labels()[0][0].startswith("ppg_timing:")


def test_long_stretch_without_slips_is_tier_n_and_short_is_u():
    r, p, _, saw = recording(seed=15)
    m = fit_ecg_ppg_delay(r, p - saw, adopted_pat_s=0.27, frame_s=FRAME)
    assert [q["tier"] for q in m.quality_intervals()] == ["N"]
    r2, p2, _, saw2 = recording(minutes=10, seed=15)
    m2 = fit_ecg_ppg_delay(r2, p2 - saw2, adopted_pat_s=0.27, frame_s=FRAME)
    assert [q["tier"] for q in m2.quality_intervals()] == ["U"]


def test_teeth_below_threshold_keep_offset_only():
    r, p, _, _ = recording(seed=16)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, apply_min_tier="A")
    # pretend the 4th tooth is only tier C: it is labelled U and gets the offset only
    m.tooth_tier[3] = "C"
    a, b = m.tooth_edges[3], m.tooth_edges[4]
    sel = (r > a) & (r < b)
    assert np.allclose(m.shift(r[sel]), m.offset_s)
    assert any(q["tier"] == "U" and q["start"] == a for q in m.quality_intervals())


def test_reference_rejects_physiological_jump_and_keeps_slips():
    # a sharp 33 ms rise of the pre-ejection period moves ECG->ABP and ECG->PPG alike: physiology
    r, p, _, saw = recording(seed=17)
    pep = np.where(r > r[0] + 2500, FRAME, 0.0)
    ref = r + 0.20 + pep                      # arterial upstroke: no sawtooth
    p2 = p + pep
    m0 = fit_ecg_ppg_delay(r, p2, adopted_pat_s=0.27, frame_s=FRAME)
    m1 = fit_ecg_ppg_delay(r, p2, adopted_pat_s=0.27, frame_s=FRAME, ref_t=ref)
    jump = r[0] + 2500
    near = lambda m: any(abs(x - jump) < 30 for x in m.tooth_edges[1:-1])
    assert near(m0) and not near(m1)
    assert any(abs(x - jump) < 30 for x in m1.qc["physiological_t"])
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    assert len(m1.qc["confirmed_by_reference_t"]) + len(m1.inserted_t) >= len(true) - 1
    assert np.all(np.abs(np.array(m1.tooth_rate[1:-1]) * 1e6 - 50) < 1.5)


def test_reference_with_gaps_is_optional_per_beat():
    r, p, _, _ = recording(seed=18)
    ref = r + 0.2
    ref[(r > r[0] + 1000) & (r < r[0] + 2000)] = np.nan   # arterial line flushed / disconnected
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, ref_t=ref)
    assert 0.7 < m.qc["reference_coverage"] < 0.9
    assert np.all(np.abs(np.array(m.tooth_rate[1:-1]) * 1e6 - 50) < 1.5)


def test_due_slips_beyond_the_ends_are_predicted():
    # the detector only saw the middle slips; the ones before and after are found by prediction
    r, p, _, saw = recording(seed=19)
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    m = EcgPpgDelay(adopted_pat_s=0.27, offset_s=0.0, frame_s=FRAME, t_start=r[0], t_end=r[-1])
    m.slips_t = [float(x) for x in true[2:-2]]
    _fill_rates(m, r, p - r, None, 60.0)
    assert len(m.qc["predicted_t"]) == 4
    assert all(np.min(np.abs(true - x)) < 5 for x in m.qc["predicted_t"])
    m2 = EcgPpgDelay(adopted_pat_s=0.27, offset_s=0.0, frame_s=FRAME, t_start=r[0], t_end=r[-1])
    m2.slips_t = [float(x) for x in true[2:-2]]
    _fill_rates(m2, r, p - r, None, 60.0, predict=False)
    assert m2.qc["predicted_t"] == []


def test_prediction_stops_without_evidence():
    # no sawtooth before the first slip seen: nothing is predicted there
    r, p, _, saw = recording(seed=20)
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    p2 = p.copy()
    early = r < true[3] - 10
    p2[early] -= saw[early] - np.median(saw)  # remove the sawtooth before the 4th slip
    m = fit_ecg_ppg_delay(r, p2, adopted_pat_s=0.27, frame_s=FRAME)
    assert not any(x < true[3] - 30 for x in m.qc.get("predicted_t", []))


def test_other_system_frame_estimated_and_corrected():
    # another monitor: 64 ms frames (e.g. a 15.6 Hz output), 30 ppm creep, frame not given to the fit
    r, p, phys, _ = recording(minutes=240, seed=21, rate_ppm=30.0, frame=0.064)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.25, tier_rules=TierRules(inserted_max_tier=None))
    assert abs(m.frame_s - 0.064) < 0.006
    assert np.all(np.abs(np.array(m.tooth_rate[1:-1]) * 1e6 - 30) < 1.5)
    assert _p2p(r, m.correct(p) - r - phys) < 4


def test_stopped_sawtooth_stretch_is_tier_n():
    # the sawtooth only starts at the 4th slip: the long stretch before it has none -> N, not U
    r, p, _, saw = recording(seed=22)
    true = r[np.flatnonzero(np.diff(saw) > 0.02) + 1]
    p2 = p.copy()
    early = r < true[3] - 10
    p2[early] -= saw[early] - np.median(saw)
    m = fit_ecg_ppg_delay(r, p2, adopted_pat_s=0.27, frame_s=FRAME)
    q = m.quality_intervals()
    assert q[0]["tier"] == "N" and "absent" in q[0]["reason"]
    assert q[0]["end"] > true[3] - 120


def test_sawtooth_evidence_present_and_absent():
    from physiosync import sawtooth_evidence

    r, p, _, saw = recording(seed=23)
    yes = sawtooth_evidence(r, p - r, 50e-6, FRAME)
    no = sawtooth_evidence(r, p - r - saw, 50e-6, FRAME)
    assert yes["score"] > 0.5 and yes["score"] > yes["mirror"] + 0.2
    assert no["score"] < -0.1 and no["teeth"] > 7


def test_rate_conflict_with_setup_prevents_false_pause():
    # true rate 27 ppm (teeth ~20.6 min), setup prior 51 ppm (~10.9 min); one full tooth only, so the
    # recording's rate is not corroborated. The last partial tooth (19 min, no slip due at 27 ppm) used to
    # be judged against the prior and called a pause (N).
    r, p, _, _ = recording(minutes=45, rate_ppm=27.0, seed=11, phase=0.75)
    old = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=51.0,
                            tier_rules=TierRules(prior_conflict_tol=None, absent_needs_own_rate=False))
    assert old.tooth_tier[-1] == "N"
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=51.0)
    assert m.qc["rate_conflict"] and set(m.tooth_tier) <= {"C", "U"}
    # with the right prior: no conflict
    m2 = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=27.0)
    assert not m2.qc["rate_conflict"] and "N" not in m2.tooth_tier


def test_end_pause_needs_the_recordings_own_rate():
    # a missed slip leaves no full tooth: the pause call would rest on the prior alone, so it is U
    r, p, _, _ = recording(minutes=45, rate_ppm=27.0, seed=13, phase=0.75)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=51.0)
    assert "N" not in m.tooth_tier


def test_teeth_bounded_by_unconfirmed_inserted_slips_are_capped():
    # the same synthetic recording: by default, teeth next to slips only inferred from ECG->PPG are C
    r, p, _, _ = recording()
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME)
    assert m.inserted_t
    for k in range(1, len(m.tooth_edges) - 2):
        if m.tooth_edges[k] in m.inserted_t or m.tooth_edges[k + 1] in m.inserted_t:
            assert m.tooth_tier[k] == "C"
