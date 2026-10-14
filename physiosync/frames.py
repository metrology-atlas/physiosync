# SPDX-License-Identifier: MIT
"""Find a hidden frame clock in a recorded signal, at any rate.

Many monitors produce a waveform in blocks ("frames") at their own rate and then output or resample it;
the recorder sees a smooth signal but its fine structure, and fiducials taken on raw samples, keep the
frame grid (VitalDB's SNUADC PLETH: 30 Hz frames under 500 Hz sampling). Nothing here assumes a dataset,
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
    """Frequencies and line-to-background ratio of the spectrum of |x[n+1] - x[n]|.

    The background is a running median of the spectrum over ``smooth_hz``; a ratio well above 1 at a
    frequency means the signal's changes repeat at that rate (a frame, mains hum, a stimulator...)."""
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
    """Spectral lines of :func:`change_spectrum` above ``min_ratio`` and ``fmin`` (below a few Hz the
    heartbeat's own harmonics dominate), strongest first. A line at a whole multiple of a lower line in
    the list is marked as its harmonic. Lines on a mains family (multiples of 50 or 60 Hz, within 0.5 Hz)
    are flagged."""
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
    """Resultant of the inter-beat intervals (whole samples) at ``period_samples`` and, as background, the
    mean resultant at control periods (default: 0.82, 0.91, 1.04 and 1.15 times the period)."""
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
    """Interval resultant at each period (samples): peaks mark periods the fiducials lock to."""
    ibi = np.diff(np.asarray(idx, dtype=np.int64)).astype(np.float64)
    if ibi_range is not None:
        ibi = ibi[(ibi >= ibi_range[0]) & (ibi <= ibi_range[1])]
    out = np.empty(len(periods))
    for k in range(0, len(periods), 256):
        p = np.asarray(periods[k : k + 256], dtype=np.float64)[:, None]
        out[k : k + 256] = np.abs(np.exp(2j * np.pi * ibi[None, :] / p).mean(axis=1))
    return out


def frame_clock(idx: np.ndarray, period_samples: float, *, span_ppm: float = 500.0, step_ppm: float = 1.0) -> dict:
    """The frame period (near ``period_samples``) that keeps the absolute fiducial indices in phase.

    Returns the period's deviation in ppm (positive = the frame period is longer, in samples, than
    nominal: the frame clock runs slow against the sampling clock), the resultant there, its median over
    the scan (background) and the deviation found in each half (agreement = a stable clock)."""
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
