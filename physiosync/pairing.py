# SPDX-License-Identifier: MIT
"""Pair each PPG pulse with the R peak that caused it, by matching inter-beat-interval sequences.

``match_beats`` pairs each R peak with the PPG fiducial nearest to R + the most common delay. That fails
when the delay is longer than a beat (fast heart rates and long monitor delays): the most common delay is
then one beat short and every pulse is paired with the next R peak. Here the pairing comes from the beats'
own rhythm, as in Goodwin et al. 2023 (Physiol. Meas. 44 085002): windows of ``window`` consecutive inter-
beat intervals of the PPG are matched to the ECG's by Euclidean distance (hrvmatch's ``get_matches``,
LaussenLabs, Apache-2.0, an optional dependency), and each match gives a beat-index shift (ECG index minus
PPG index). The shift is constant along a stretch of beats except where a beat is missed or extra in one
series, so it is taken as the running mode of the good matches and every pulse is paired with the R peak at
its index plus the shift. A match counts as good when its distance is small and clearly smaller than that of
the one-beat-shifted alternatives (a very regular heart rhythm cannot tell a beat from its neighbours: those
windows are marked ambiguous and, if ``expected_delay_s`` is given, the plausible delay decides).

hrvmatch is installed separately (``pip install --no-deps git+https://github.com/LaussenLabs/hrvmatch``;
its pinned numpy/scipy/scikit-learn versions are not needed for ``get_matches``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["pair_beats"]


def _get_matches():
    try:
        from peakalignment.signal_matching import get_matches
    except ImportError as ex:  # pragma: no cover
        raise ImportError("pair_beats needs hrvmatch (pip install --no-deps "
                          "git+https://github.com/LaussenLabs/hrvmatch)") from ex
    return get_matches


def _ivi_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pair_beats(r_t: np.ndarray, p_t: np.ndarray, *, window: int = 20, delay_range_s: tuple[float, float] = (0.1, 1.6),
               max_rms_s: float = 0.015, margin: float = 1.5, expected_delay_s: tuple[float, float] | None = None,
               mode_halfwidth: int = 30) -> dict:
    """Pair PPG fiducials with ECG R peaks by matching their inter-beat-interval rhythm.

    Windows of ``window`` consecutive PPG intervals are matched to the ECG's by Euclidean distance, giving a
    beat-index shift per window. The shift is constant along a stretch except where a beat is missed or extra,
    so the running mode of the good matches is carried to every PPG beat. A match is good when its RMS
    difference is small and clearly smaller than that of the one-beat-shifted alternatives. Needs the optional
    dependency hrvmatch (see the module docstring).

    Args:
        r_t: R-peak times (s), sorted.
        p_t: PPG fiducial times (s), sorted, on the same clock.
        window: Intervals per matching window.
        delay_range_s: ``(low, high)`` delays (PPG minus R) that are possible at all (s).
        max_rms_s: A window matches if the RMS difference of its intervals is at most this (s).
        margin: ...and if the one-beat-shifted alternatives are at least this many times further.
        expected_delay_s: Optional ``(low, high)`` plausible delay (s), e.g. the setup's known delay
            +-0.2 s. It decides windows the rhythm cannot (too regular), and pairs beats the rhythm left
            unpaired when exactly one R peak gives a delay in range.
        mode_halfwidth: Half-width of the running mode of window shifts (PPG beats).

    Returns:
        A dict with ``r_index`` and ``p_index`` (indices of the paired beats), ``delay_s`` (PPG minus R),
        ``source`` (``"rhythm"`` or ``"window"``: which rule paired each beat), ``shift`` (per PPG beat, NaN
        where unpaired) and ``qc`` (window and pairing counts, median delay, shift histogram, or ``reason``).

    Raises:
        ImportError: hrvmatch is not installed.
    """
    get_matches = _get_matches()
    r_t = np.asarray(r_t, dtype=np.float64)
    p_t = np.asarray(p_t, dtype=np.float64)
    lo, hi = delay_range_s
    n_p = p_t.size
    out = {"r_index": np.empty(0, int), "p_index": np.empty(0, int), "delay_s": np.empty(0), "source": np.empty(0, str),
           "shift": np.full(n_p, np.nan), "qc": {}}
    if r_t.size <= window + 2 or n_p <= window + 2:
        out["qc"] = {"reason": "too few beats"}
        return out
    # hrvmatch works in ms; search windows up to hi after each PPG window (signal A = PPG, B = ECG)
    idx, _dist = get_matches(p_t * 1000.0, r_t * 1000.0, max_offset=hi * 1000.0)
    j_best = idx[:, 0]
    ivi_p = np.diff(p_t)
    ivi_r = np.diff(r_t)
    nw = j_best.size
    shift = np.full(nw, np.iinfo(np.int64).min, dtype=np.int64)
    status = np.zeros(nw, dtype=np.int8)  # 0 none, 1 good, 2 ambiguous
    for i in range(nw):
        j = int(j_best[i])
        if j + window > ivi_r.size or i + window > ivi_p.size:
            continue
        d0 = _ivi_dist(ivi_p[i : i + window], ivi_r[j : j + window])
        delay = p_t[i] - r_t[j]
        if d0 > max_rms_s or not (lo <= delay <= hi):
            continue
        alts = [_ivi_dist(ivi_p[i : i + window], ivi_r[j + s : j + s + window])
                for s in (-1, 1) if 0 <= j + s and j + s + window <= ivi_r.size]
        if alts and min(alts) < margin * max(d0, 1e-4):
            status[i] = 2
            if expected_delay_s is None:
                continue
            # the rhythm cannot decide: take the neighbour whose delay is in the expected range
            cands = [s for s in (-1, 0, 1) if 0 <= j + s < r_t.size
                     and expected_delay_s[0] <= p_t[i] - r_t[j + s] <= expected_delay_s[1]]
            if len(cands) != 1:
                continue
            j += cands[0]
        else:
            status[i] = 1
        shift[i] = j - i
    # running mode of the window shifts, carried to every PPG beat
    none = np.iinfo(np.int64).min
    valid = np.flatnonzero(shift != none)
    beat_shift = np.full(n_p, none, dtype=np.int64)
    for k in range(n_p):
        a, b = np.searchsorted(valid, k - mode_halfwidth), np.searchsorted(valid, k + mode_halfwidth)
        if b <= a:
            continue
        vals, counts = np.unique(shift[valid[a:b]], return_counts=True)
        if counts.max() >= max(3, 0.6 * counts.sum()):
            beat_shift[k] = int(vals[np.argmax(counts)])
    pk = np.flatnonzero(beat_shift != none)
    rk = pk + beat_shift[pk]
    ok = (rk >= 0) & (rk < r_t.size)
    pk, rk = pk[ok], rk[ok]
    d = p_t[pk] - r_t[rk]
    keep = (d >= lo) & (d <= hi)
    if expected_delay_s is not None:
        # the rhythm decides which beat; the expected delay still rules out the impossible (a very regular
        # rhythm can match one beat off as well as the true pairing)
        keep &= (d >= expected_delay_s[0]) & (d <= expected_delay_s[1])
    pk, rk, d = pk[keep], rk[keep], d[keep]
    src = np.full(pk.size, "rhythm", dtype=object)
    # beats the rhythm left unpaired: the R peak whose delay falls in the expected range, if exactly one does
    if expected_delay_s is not None:
        done = np.zeros(n_p, bool)
        done[pk] = True
        extra_p, extra_r = [], []
        for k in np.flatnonzero(~done):
            a_, b_ = np.searchsorted(r_t, p_t[k] - expected_delay_s[1]), np.searchsorted(r_t, p_t[k] - expected_delay_s[0], side="right")
            if b_ - a_ == 1:
                extra_p.append(k)
                extra_r.append(a_)
        if extra_p:
            pk = np.r_[pk, extra_p].astype(np.int64)
            rk = np.r_[rk, extra_r].astype(np.int64)
            d = p_t[pk] - r_t[rk]
            src = np.r_[src, np.full(len(extra_p), "window", dtype=object)]
            order = np.argsort(pk)
            pk, rk, d, src = pk[order], rk[order], d[order], src[order]
    # one PPG pulse per R peak
    _, first = np.unique(rk, return_index=True)
    pk, rk, d, src = pk[first], rk[first], d[first], src[first]
    out.update({"r_index": rk, "p_index": pk, "delay_s": d, "source": src.astype(str),
                "shift": np.where(beat_shift != none, beat_shift, np.nan).astype(np.float64)})
    out["qc"] = {"windows": int(nw), "good": int((status == 1).sum()), "ambiguous": int((status == 2).sum()),
                 "paired": int(pk.size), "paired_by_rhythm": int((src == "rhythm").sum()),
                 "paired_by_window": int((src == "window").sum()), "ppg_beats": int(n_p), "ecg_beats": int(r_t.size),
                 "delay_median_s": float(np.median(d)) if d.size else None,
                 "shifts": {int(v): int(c) for v, c in zip(*np.unique(shift[valid], return_counts=True))} if valid.size else {}}
    return out
