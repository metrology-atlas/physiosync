# SPDX-License-Identifier: MIT
"""Record time stamps that follow a device's own sample clock between occasional corrections.

Many recorders stamp each block of samples as "where the device's sample count says it is", and every so
often step the stamps back towards the recorder's clock. The stamps then carry a sawtooth: within a stretch
they drift at the device's rate error (ppm), and at each correction they jump by the error accumulated.
Two devices recorded together are then out of step by the difference of their sawtooths.

:func:`stamp_clock` fits, for one track, a straight line through the stamp residual (stamp minus the time the
sample count gives at the nominal rate): one rate error for the whole recording, and one offset per
stretch between real gaps (jumps of at least ``gap_s``, or with the opposite sign of the corrections). The
line runs through the middle of the sawtooth; :func:`corrected_times` places every sample on it, which
removes the ramps and the steps. What is left between two devices is a constant offset (half of each
device's correction threshold plus any transport delay), which a shared reference has to supply.

All inputs are plain arrays: the stamp of each record's first sample (s), that sample's index in the track,
and the nominal rate.
"""

from __future__ import annotations

import numpy as np

__all__ = ["corrected_times", "stamp_clock"]


def stamp_clock(record_times: np.ndarray, record_starts: np.ndarray, fs: float, *, gap_s: float = 1.0,
                min_step_s: float = 0.002, cluster_s: float = 300.0, min_teeth: int = 1,
                rate_ppm: float | None = None, correction_sign: int | None = None) -> dict:
    """Fit one track's stamp sawtooth.

    ``record_times``  stamp of each record's first sample (s), in order.
    ``record_starts`` index of that sample in the track (cumulative sample count).
    ``gap_s``         a residual jump at least this large is a real gap (new stretch, own offset); so is a
                      jump (above ``min_step_s``) whose sign is opposite to that of most corrections.
    Single records stamped early or late (a step and its reverse) are neither corrections nor gaps and are
    left out of the fit. ``correction_sign`` (+1/-1) fixes the direction of corrections when it is known
    for the device (otherwise the direction of most small steps); steps the other way are gaps. Corrections within ``cluster_s`` of each other form one correction event.
    The rate comes from whole teeth (event to event: the drift each event corrects over the time since the
    last), or, with fewer than ``min_teeth``, from a least-squares line through the residual staircase, or
    is ``rate_ppm`` when given (e.g. the device's known rate). The offsets put the line through the middle.
    Returns {"t0", "fs", "rate_ppm" (the device's rate error: + = its samples last longer than nominal),
    "breaks" (record indices starting a stretch), "offsets_s" (per stretch), "corrections" (count and median
    size), "gaps" (count), "sawtooth_p2p_s", "residual_mad_s"}."""
    t = np.asarray(record_times, dtype=np.float64)
    k = np.asarray(record_starts, dtype=np.float64)
    if t.size != k.size or t.size < 3:
        raise ValueError("need at least 3 records with matching starts")
    t0 = float(t[0])
    c = k / fs                                   # count time since the first record
    res = (t - t0) - c
    d = np.diff(res)
    step = np.abs(d) >= min_step_s
    # single records stamped early/late: a step followed at once by its reverse; neither is a correction
    fl = np.zeros(d.size, dtype=bool)
    pair = step[:-1] & step[1:] & (np.abs(d[:-1] + d[1:]) < min_step_s)
    fl[:-1] |= pair
    fl[1:] |= pair
    small = step & (np.abs(d) < gap_s) & ~fl
    if correction_sign is not None:
        sign = float(np.sign(correction_sign))
    else:
        sign = np.sign(np.sum(d[small])) if small.any() else 0.0
    gap = ((np.abs(d) >= gap_s) | (small & (np.sign(d) != sign) & (sign != 0))) & ~fl
    breaks = np.r_[0, np.flatnonzero(gap) + 1]
    seg = np.zeros(t.size, dtype=np.int64)
    seg[breaks[1:]] = 1
    seg = np.cumsum(seg)

    use = np.ones(t.size, dtype=bool)
    use[np.flatnonzero(pair) + 1] = False        # the flicked records themselves
    # correction events: corrections within cluster_s of each other are one event (sizes summed)
    ci = np.flatnonzero(small & ~gap)            # correction between record i and i+1
    ev_at, ev_size = [], []
    for i in ci:
        if ev_at and seg[i + 1] == seg[ev_at[-1]] and c[i + 1] - c[ev_at[-1]] <= cluster_s:
            ev_size[-1] += d[i]
        else:
            ev_at.append(i + 1)
            ev_size.append(d[i])
    ev_at, ev_size = np.asarray(ev_at, dtype=np.int64), np.asarray(ev_size)
    # whole teeth: from one event to the next in the same stretch; the second event corrects the drift
    # accumulated over the tooth, so rate = sum(size) / sum(duration)
    tooth = (np.diff(seg[ev_at]) == 0) if ev_at.size > 1 else np.zeros(0, bool)
    dur = np.diff(c[ev_at])[tooth] if ev_at.size > 1 else np.zeros(0)
    if rate_ppm is not None:
        b, source = rate_ppm * 1e-6, "given"
    elif dur.size >= min_teeth:
        b, source = float(np.sum(ev_size[1:][tooth]) / np.sum(dur)), "teeth"
    else:
        n_seg = int(seg.max()) + 1
        cnt = np.bincount(seg[use], minlength=n_seg).astype(float)
        cnt[cnt == 0] = np.nan
        xm = np.bincount(seg[use], c[use], minlength=n_seg) / cnt
        ym = np.bincount(seg[use], res[use], minlength=n_seg) / cnt
        xs, ys = c[use] - xm[seg[use]], res[use] - ym[seg[use]]
        den = float(np.sum(xs * xs))
        b, source = (float(np.sum(xs * ys) / den) if den > 0 else 0.0), "fit"
    n_seg = int(seg.max()) + 1
    cnt = np.bincount(seg[use], minlength=n_seg).astype(float)
    a = np.bincount(seg[use], (res - b * c)[use], minlength=n_seg) / np.where(cnt > 0, cnt, np.nan)
    a = np.where(np.isfinite(a), a, 0.0)
    r = res - (a[seg] + b * c)
    return {"t0": t0, "fs": float(fs), "rate_ppm": b * 1e6, "rate_source": source, "teeth": int(dur.size),
            "breaks": breaks, "offsets_s": a,
            "corrections": {"events": int(ev_at.size), "event_times_s": (t0 + c[ev_at]).tolist(),
                            "median_event_s": float(np.median(ev_size)) if ev_size.size else None,
                            "records": int(ci.size)},
            "gaps": int(breaks.size - 1),
            "sawtooth_p2p_s": float(np.nanpercentile(r[use], 99) - np.nanpercentile(r[use], 1)),
            "residual_mad_s": float(1.4826 * np.nanmedian(np.abs(r[use] - np.nanmedian(r[use]))))}


def corrected_times(clock: dict, record_starts: np.ndarray, sizes: np.ndarray | None = None) -> np.ndarray:
    """Times on the fitted line: per record (``sizes`` None) or for every sample (``sizes`` = samples per
    record, so the result has sum(sizes) entries)."""
    k = np.asarray(record_starts, dtype=np.float64)
    fs, b = clock["fs"], clock["rate_ppm"] * 1e-6
    seg = np.zeros(k.size, dtype=np.int64)
    br = clock["breaks"][1:]
    seg[br[br < k.size]] = 1
    seg = np.cumsum(seg)
    c = k / fs
    rec = clock["t0"] + c * (1 + b) + clock["offsets_s"][seg]
    if sizes is None:
        return rec
    sizes = np.asarray(sizes, dtype=np.int64)
    idx = np.repeat(np.arange(k.size), sizes)
    within = np.arange(idx.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return rec[idx] + within / fs * (1 + b)
