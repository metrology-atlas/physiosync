# SPDX-License-Identifier: MIT
"""Find a hidden frame clock in a recorded signal, at any rate.

Many monitors produce a waveform in blocks ("frames") at their own rate and then output or resample it;
the recorder sees a smooth signal but its fine structure, and fiducials taken on raw samples, keep the
frame grid (e.g. VitalDB's SNUADC PLETH: 30 Hz frames under 500 Hz sampling). Nothing here assumes a dataset,
a sampling rate or a frame rate:

``change_spectrum``  spectrum of the absolute sample-to-sample change, as line-to-background ratios. A
                     framed signal changes in a pattern that repeats every frame, which shows as spectral
                     lines at the frame rate and its harmonics, whatever the waveform.
``frame_lines``      the strongest lines, grouped into fundamentals and harmonics, with mains frequencies
                     (50/60 Hz families) flagged.
``interval_lock``    how strongly fiducials lock to a period, from their inter-beat intervals, against
                     control periods (works even when the frame clock drifts against the sampling clock).
``period_lock_scan`` the same over a range of periods, to find an unknown one.
``frame_clock``      the frame period that keeps the absolute fiducial indices in phase over a recording:
                     the frame clock's rate relative to the sampling clock, in ppm, to about 1 ppm.

Only numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["change_spectrum", "frame_clock", "frame_lines", "interval_lock", "period_lock_scan"]


def change_spectrum(x: np.ndarray, fs: float, *, nperseg: int | None = None, smooth_hz: float = 4.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum of the absolute sample-to-sample change, as a line-to-background ratio.

    A framed signal changes in a pattern that repeats every frame, so ``|x[n+1] - x[n]|`` has spectral lines
    at the frame rate and its harmonics whatever the waveform. The background is a running median of the
    Welch spectrum over ``smooth_hz``.

    Args:
        x: The signal, raw samples (non-finite changes are dropped).
        fs: Sampling rate (Hz).
        nperseg: Welch segment length in samples. Default: 16 s of signal, or the whole signal if shorter.
        smooth_hz: Width of the running median that defines the background (Hz).

    Returns:
        ``(f, ratio)``: frequencies (Hz) and the spectrum divided by its background. A ratio well above 1
        means the signal's changes repeat at that rate (a frame, mains hum, a stimulator).
    """
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    d = d[np.isfinite(d)]
    nper = nperseg or int(min(d.size, 16 * fs))
    f, p = signal.welch(d - d.mean(), fs=fs, nperseg=nper)
    k = max(3, round(smooth_hz / (f[1] - f[0])) | 1)
    pad = np.pad(p, k // 2, mode="edge")
    bg = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    return f, p / np.maximum(bg, 1e-30)


def frame_lines(x: np.ndarray, fs: float, *, min_ratio: float = 3.0, fmin: float = 8.0, max_lines: int = 12,
                mains: tuple[float, ...] = (50.0, 60.0)) -> list[dict]:
    """The strongest lines of :func:`change_spectrum`, with harmonics and mains frequencies marked.

    Below a few hertz the heartbeat's own harmonics dominate, so lines below ``fmin`` are ignored.

    Args:
        x: The signal, raw samples.
        fs: Sampling rate (Hz).
        min_ratio: Smallest line-to-background ratio reported.
        fmin: Lowest frequency considered (Hz).
        max_lines: Most lines returned.
        mains: Mains fundamentals (Hz). A line within 0.5 Hz of a multiple is flagged.

    Returns:
        One dict per line, strongest first: ``hz``, ``ratio``, ``period_samples`` (``fs / hz``), ``mains``
        (bool) and, when the line is a whole multiple of a lower line in the list, ``harmonic_of_hz``.
    """
    f, r = change_spectrum(x, fs)
    sel = (f >= fmin) & (f < fs / 2 - 1)
    pk, _ = signal.find_peaks(np.where(sel, r, 0), height=min_ratio)
    pk = pk[np.argsort(-r[pk])][:max_lines]
    lines = [{"hz": round(float(f[i]), 3), "ratio": round(float(r[i]), 2)} for i in pk]
    for ln in lines:
        ln["period_samples"] = round(fs / ln["hz"], 3)
        ln["mains"] = any(abs(ln["hz"] / m - round(ln["hz"] / m)) * m < 0.5 for m in mains if ln["hz"] >= m * 0.99)
    bases = sorted(lines, key=lambda d: d["hz"])
    for ln in lines:
        for b in bases:
            if b is ln or b["hz"] >= ln["hz"]:
                continue
            m = ln["hz"] / b["hz"]
            if abs(m - round(m)) < 0.02 * round(m):
                ln["harmonic_of_hz"] = b["hz"]
                break
    return lines


def _resultant(v: np.ndarray, period: float) -> float:
    return float(np.abs(np.exp(2j * np.pi * v / period).mean())) if v.size else float("nan")


def interval_lock(idx: np.ndarray, period_samples: float, *, controls: tuple[float, ...] | None = None,
                  ibi_range: tuple[int, int] | None = None) -> dict:
    """How strongly fiducials lock to a period, judged from their inter-beat intervals.

    The resultant is ``|mean(exp(2*pi*i*ibi/period))|`` over the intervals in whole samples: 1 when every
    interval is a whole number of periods, near 0 when they are unrelated. Using intervals rather than
    absolute indices makes the test work even when the frame clock drifts against the sampling clock.

    Args:
        idx: Fiducial sample indices, in order.
        period_samples: The period to test (samples; need not be whole).
        controls: Control periods for the background. Default: 0.82, 0.91, 1.04 and 1.15 times the period.
        ibi_range: Keep only intervals within ``(low, high)`` samples, e.g. to drop missed beats.

    Returns:
        ``n`` (intervals used), ``resultant`` at the period, ``control`` (mean resultant at the control
        periods) and ``ratio`` (resultant / control).
    """
    ibi = np.diff(np.asarray(idx, dtype=np.int64))
    if ibi_range is not None:
        ibi = ibi[(ibi >= ibi_range[0]) & (ibi <= ibi_range[1])]
    ctrl = controls or tuple(period_samples * c for c in (0.82, 0.91, 1.04, 1.15))
    r = _resultant(ibi, period_samples)
    rc = float(np.mean([_resultant(ibi, p) for p in ctrl]))
    return {"n": int(ibi.size), "resultant": round(r, 4), "control": round(rc, 4),
            "ratio": round(r / rc, 2) if rc > 0 else float("inf")}


def period_lock_scan(idx: np.ndarray, periods: np.ndarray, *, ibi_range: tuple[int, int] | None = None
                     ) -> np.ndarray:
    """Interval resultant (see :func:`interval_lock`) at each of many periods.

    Args:
        idx: Fiducial sample indices, in order.
        periods: Periods to test (samples).
        ibi_range: Keep only intervals within ``(low, high)`` samples.

    Returns:
        The resultant at each period. Peaks mark periods the fiducials lock to.
    """
    ibi = np.diff(np.asarray(idx, dtype=np.int64)).astype(np.float64)
    if ibi_range is not None:
        ibi = ibi[(ibi >= ibi_range[0]) & (ibi <= ibi_range[1])]
    out = np.empty(len(periods))
    for k in range(0, len(periods), 256):
        p = np.asarray(periods[k : k + 256], dtype=np.float64)[:, None]
        out[k : k + 256] = np.abs(np.exp(2j * np.pi * ibi[None, :] / p).mean(axis=1))
    return out


def frame_clock(idx: np.ndarray, period_samples: float, *, span_ppm: float = 500.0, step_ppm: float = 1.0) -> dict:
    """The frame period, near a nominal one, that keeps absolute fiducial indices in phase.

    Scans the period's deviation from ``period_samples`` and takes the one with the largest resultant of the
    fiducials' phases. Doing the same on each half of the recording checks that the clock is stable.

    Args:
        idx: Fiducial sample indices over the recording, in order.
        period_samples: Nominal frame period (samples), e.g. ``fs / 30`` for 30 Hz frames.
        span_ppm: Scan deviations from ``-span_ppm`` to ``+span_ppm``.
        step_ppm: Scan step (ppm).

    Returns:
        An empty dict with fewer than 200 fiducials. Otherwise ``ppm`` (the deviation found; positive means
        the frame period is longer, in samples, than nominal, so the frame clock runs slow against the
        sampling clock), ``resultant`` there, ``background`` (median resultant over the scan), ``sharpness``
        (resultant / background) and ``ppm_halves`` (the deviation found in each half; agreement means a
        stable clock).
    """
    idx = np.asarray(idx, dtype=np.float64)
    if idx.size < 200:
        return {}

    def scan(v: np.ndarray) -> tuple[float, float, float]:
        e = np.arange(-span_ppm, span_ppm + step_ppm / 2, step_ppm) * 1e-6
        x = v - v[0]
        r = np.empty(e.size)
        for k in range(0, e.size, 128):
            ph = 2 * np.pi * x[None, :] / (period_samples * (1 + e[k : k + 128, None]))
            r[k : k + 128] = np.abs(np.exp(1j * ph).mean(axis=1))
        i = int(np.argmax(r))
        return float(e[i] * 1e6), float(r[i]), float(np.median(r))

    ppm, rb, rm = scan(idx)
    h = idx.size // 2
    p1, _, _ = scan(idx[:h])
    p2, _, _ = scan(idx[h:])
    return {"ppm": round(ppm, 1), "resultant": round(rb, 4), "background": round(rm, 4),
            "sharpness": round(rb / rm, 1) if rm > 0 else float("inf"), "ppm_halves": [round(p1, 1), round(p2, 1)]}
