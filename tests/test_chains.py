# SPDX-License-Identifier: MIT
import numpy as np
from test_measurement import FRAME, recording

from physiosync import chain_labels, chain_slips, fit_ecg_ppg_delay


def test_chain_finds_every_slip_on_a_clean_recording():
    r, p, _ = recording(minutes=90, seed=1)
    ch = chain_slips(r, p - r, frame_s=FRAME, rate=50e-6)
    assert len(ch) == 1 and len(ch[0]) >= 7
    gaps = np.diff([x for x, _ in ch[0]])
    assert np.all(np.abs(gaps / (FRAME / 50e-6) - 1) < 0.02)


def test_chain_stops_where_the_sawtooth_pauses_and_labels_carry_fields():
    r, p, _ = recording(minutes=120, seed=3, stop_min=50)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, frame_s=FRAME, rate_prior_ppm=50.0)
    lab = chain_labels(m, r, p, rate_ppm=50.0, error_s=(0.0032, 0.0196))
    assert lab and max(d["end"] for d in lab) < r[0] + 52 * 60
    d = lab[0]
    assert abs(d["slope_ppm"] - 50) < 2 and d["error_p90_ms"] == 19.6 and d["usable_for_pat"] is False
    assert 0.0 <= d["main_ab_share"] <= 1.0


def test_no_rate_no_chain():
    r, p, _ = recording(minutes=20, seed=5)
    m = fit_ecg_ppg_delay(r, p, adopted_pat_s=0.27, sawtooth=False)
    assert chain_labels(m, r, p) == []
