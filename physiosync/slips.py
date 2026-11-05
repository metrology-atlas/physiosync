# SPDX-License-Identifier: MIT
"""Frame slips of the ECG-to-PPG delay, found from beats with their evidence (the rebuilt method).

A PPG produced in frames by a clock slightly off the ECG's shows a sawtooth in the ECG-to-PPG delay: it creeps by the
clock difference and steps by one frame (a *slip*) whenever the difference reaches a frame; either way round (a
reversed sawtooth creeps the other way and steps down). :func:`find_slips` finds the slips of one recording from its
beats and says, for each, how it knows:

1. **Delay series** of every foot, upstroke and peak against R (optionally the template-refined R), leaving out beats
   that fail a quality gate (QRS or pulse unlike the local median beat, irregular R-R, clipping); the breathing is
   fitted in the step test.
2. **Candidates** both ways: strong one-frame steps (likelihood ratio, sharpness, persistence).
3. **Period**: the tooth whose chain of slips collects the most evidence (with phase jumps at a cost), within the
   setup's range if one is given, and twice that tooth checked; runs on a shifted grid are fitted run by run.
4. **Judging each predicted slip**: the best one-frame step in its window, then the pulse-shape test (the median
   R-aligned pulse after against before: a slip moves it about a frame with the shape kept), searched at the window's
   likeliest boundaries; then the beats' evidence: with an arterial line, ART-to-PPG stepping a frame with ECG-to-ART flat
   confirms a slip the shape could not judge, and ECG-to-ART carrying the step vetoes one the shape did not confirm; a
   heart-rate change of 20 /min or more vetoes an unconfirmed one. Outcomes: slip, step the shape cannot judge, step
   with the shape unclear, shape only, no slip, unresolved; each with the probability that one slip lies in the
   window and its 5-50-95% times (the fallback).
5. **Elsewhere**: steps off the grid or the other way are kept only when confirmed and not within 120 s of an accepted
   slip; opposite-way pairs within 5 min are a transient.
6. **No sawtooth fitted**: from the confirmed slips, the search outward at the recording's own tooth (their spacing,
   or the measured drift); else the drift is reported (not corrected).

Waveform processing (filtering, fiducial detection) is the caller's: :func:`beat_quality` takes band-passed signals and
the beats' times and returns the per-beat quality and the R-aligned pulse and QRS shapes :func:`find_slips` needs.

:func:`delay_at` turns the result into the delay to remove: each tooth between two slips closes exactly (it creeps one
frame over its own length), uncertain slips are ramps, unresolved teeth and recordings whose creep is not corrected are
left alone; :func:`delay_model` gives the same as segments, with the tiers of :func:`tooth_tiers`.

The method and its evaluation (manual labels, injected sawteeth, a review of the corrections, finger against arterial
pulse arrival time) are described in the metrology-atlas study ppg-sawtooth-correction. Times in seconds on one clock.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter

__all__ = ["SlipOptions", "beat_quality", "delay_at", "delay_model", "find_slips", "tooth_tiers"]

FRAME = 1 / 30          # s; the frame of the PPG (30 Hz in the SNUADC PLETH of VitalDB)
FS = 500.0              # Hz; default sampling rate of the waveforms given to beat_quality
FIDS = ("foot", "up", "peak", "down50")
DET_FIDS = ("foot", "up", "peak")


STEP_W = 15


LOC_W = 3


GUARD = 3


PPB = 3


GLR_W = 15 * PPB


RAMP_HALF = 3 * PPB


LLR_STRONG = 25.0


LLR_GRID = 8.0


SHARP_MIN = 4.0


PERSIST = (15 * PPB, 75 * PPB)


NOISE_GATE = 2.5


PERSIST_SHORT = (15 * PPB, 40 * PPB)


OUTLIER = 0.05


OUT_SPAN = 0.2


PPG_WIN, PPG_FS = (0.1, 1.0), 125.0


ECG_WIN, ECG_FS = (-0.12, 0.12), 250.0


REF_WIN, REF_FS, REF_LAG = 0.06, 1000.0, 10


GOOD_CORR = 0.9


JUMP = 20.0


M_MAX = 8


BROAD = (300.0, 4000.0)


NEAR_SLIP_S = 120.0


TRANSIENT_S = 300.0


ART_SLIP = (0.6, 1.4)


ART_CIRC = 0.3


HR_VETO = 20.0


TEMPERS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


TEMPER = 0.05


TEMPER_TIMES = 0.02


P_NONE = 0.2


P_NONE_GAP = 0.5


ANCHOR_SPREAD = 0.06


DRIFT_SD_PPM = 6.5


def _noise_sd(y: np.ndarray) -> float:
    d = np.diff(y[np.isfinite(y)])
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size > 10 else float("nan")


def _point_series(F: dict, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every foot, upstroke and peak as one time-ordered series of delays (fiducial - base), each type...

    Every foot, upstroke and peak as one time-ordered series of delays (fiducial - base), each type moved onto the
    upstroke's level by the 61-beat running median of its difference from the upstroke (pulse shape changes slowly).
    Returns the times, the delays and each point's beat. A slip inside a pulse is then a clean step in this series,
    between two of its points.
    """
    n = base.size
    ts, ys, bs = [], [], []
    for j in DET_FIDS:
        d = F[j] - F["up"]
        dd = np.where(np.isfinite(d), d, np.nanmedian(d))
        off = median_filter(dd, size=61, mode="nearest")
        ts.append(F[j])
        ys.append(F[j] - base - off)
        bs.append(np.arange(n))
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    b = np.concatenate(bs)
    ok = np.isfinite(t) & np.isfinite(y)
    o = np.argsort(t[ok], kind="stable")
    return t[ok][o], y[ok][o], b[ok][o]


def _clean(y: np.ndarray, sd: float) -> np.ndarray:
    """Replace beats further than 5 noise SD from the 7-beat running median by that median (isolated...

    Replace beats further than 5 noise SD from the 7-beat running median by that median (isolated mis-detections),
    so one bad beat cannot fake or hide a step.
    """
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    m = median_filter(y, size=7, mode="nearest")
    return np.where(np.abs(y - m) > 5 * sd, m, y)


def _long_step(y: np.ndarray, b: int, sign: int, win: tuple[int, int] | None = None) -> float:
    """Persistence at boundary b.

    median of points b+15 .. b+75 beats minus median of points b-75 .. b-15 beats, in the sign's direction (a
    transient bump that returns within 15 beats does not change it). Near the ends of the stretch the windows are
    cut short, down to 15 beats of points.
    """
    a_, b_ = win or PERSIST
    lo, hi = max(b - b_, 0), min(b + b_, y.size)
    if b - a_ - lo < 15 * PPB or hi - (b + a_) < 15 * PPB:
        return float("nan")
    return float(sign * (np.nanmedian(y[b + a_:hi]) - np.nanmedian(y[lo:b - a_])))


def _local_noise(y: np.ndarray, half: int = 30 * PPB) -> np.ndarray:
    """Robust beat-to-beat scatter in +/- ``half`` beats of every beat."""
    d = np.abs(np.diff(y, prepend=y[0]))
    w = 2 * half + 1
    if d.size < w:
        return np.full(d.size, np.nanmedian(d) * 1.4826 / np.sqrt(2))
    med = np.nanmedian(sliding_window_view(d, w), axis=1) * 1.4826 / np.sqrt(2)
    return np.r_[np.full(half, med[0]), med, np.full(d.size - med.size - half, med[-1])]


def _robust_ll(res: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    g = np.exp(-0.5 * ((res - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    return np.log((1 - OUTLIER) * g + OUTLIER / OUT_SPAN)


def _place(b: int, F: dict, r: np.ndarray, sd: dict, sign: int = 1) -> dict:
    """Place the slip at boundary b (between beats b-1 and b) between two fiducials.

    Each fiducial type j has its own level before (beats b-GUARD-STEP_W .. b-GUARD) and after; every fiducial of
    beats b-LOC_W .. b+LOC_W-1, in time order, is pre (residual ~ N(0, sd_j)) or post (~ N(step_j, sd_j)). The
    posterior over the split point gives the most likely position and a 95% bracket. The likelihood allows 5% of
    fiducials to be outliers (uniform over 0.2 s), so one disturbed fiducial cannot move the slip.
    """
    n = r.size
    a0, a1 = b - GUARD - STEP_W, b - GUARD
    c0, c1 = b + GUARD, b + GUARD + STEP_W
    if a0 < 0 or c1 > n:
        return {}
    pts, steps = [], {}
    for j in FIDS:
        y = sign * (F[j] - r)
        bef, aft = np.nanmedian(y[a0:a1]), np.nanmedian(y[c0:c1])
        steps[j] = float(aft - bef)
        for i in range(b - LOC_W, b + LOC_W):
            if np.isfinite(F[j][i]):
                pts.append((float(F[j][i]), j, float(y[i] - bef), i))
    if len(pts) < 8:
        return {}
    pts.sort()
    tt = np.array([p[0] for p in pts])
    res = np.array([p[2] for p in pts])
    st = np.array([steps[p[1]] for p in pts])
    s = np.array([max(sd[p[1]], 1e-4) for p in pts])
    ll_pre = _robust_ll(res, 0.0, s)
    ll_post = _robust_ll(res, st, s)
    # split m: points [0, m) pre, [m, N) post; m from 1 to N-1
    cp = np.cumsum(ll_pre)
    cq = np.cumsum(ll_post[::-1])[::-1]
    N = len(pts)
    ll = np.array([cp[m - 1] + cq[m] for m in range(1, N)])
    post = np.exp(ll - ll.max())
    post /= post.sum()
    m = 1 + int(np.argmax(post))
    order = np.argsort(post)[::-1]
    cum, sel = 0.0, []
    for k in order:
        sel.append(k + 1)
        cum += post[k]
        if cum >= 0.95:
            break
    lo_m, hi_m = min(sel), max(sel)
    frac = res / np.where(np.abs(st) > 1e-4, st, np.nan)
    return {"t": float(0.5 * (tt[m - 1] + tt[m])), "lo": float(tt[lo_m - 1]), "hi": float(tt[hi_m]),
            "best_lo": float(tt[m - 1]), "best_hi": float(tt[m]), "p_best": float(post.max()),
            "between": [pts[m - 1][1], pts[m][1]], "steps_ms": {k: round(v * 1000, 2) for k, v in steps.items()},
            "partial": int(np.sum((frac > 0.25) & (frac < 0.75))), "n_pts": N}


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def _best_shift(before, after, fs, max_ms=80):
    """Shift (ms) of `after` against `before` that maximises the correlation (positive: after is later)."""
    tt = np.arange(before.size) / fs
    fine = np.arange(0, tt[-1], 0.001)
    bb = _z(np.interp(fine, tt, before))
    aa = np.interp(fine, tt, after)
    best, bc, c0 = 0, -2.0, None
    for s in range(-max_ms, max_ms + 1):
        # after(t) ~ before(t - s): compare after shifted back by s
        sh = np.interp(fine + s / 1000.0, fine, aa, left=np.nan, right=np.nan)
        ok = np.isfinite(sh)
        if ok.sum() < 0.6 * fine.size:
            continue
        c = float(np.corrcoef(bb[ok], sh[ok])[0, 1])
        if s == 0:
            c0 = c
        if c > bc:
            best, bc = s, c
    return best, bc, c0


def _call(r: dict, Q: float, I: float, C: float, F: float) -> str:
    g = lambda k: r.get(k) if r.get(k) is not None and np.isfinite(r.get(k)) else None
    wb, ri = g("worst_bad"), g("worst_rr_irregular")
    if wb is None or wb > Q or (ri is not None and ri > I):
        return "cannot judge"
    sh, cb, sp = g("ppg_shift_ms"), g("ppg_corr_best"), g("fid_spread_ms")
    if sh is not None and 20 <= abs(sh) <= 45 and cb is not None and cb >= C and (sp is None or sp <= F):
        return "slip"
    return "no slip"


def _snippets(t0: float, fs_src: float, x: np.ndarray, centres: np.ndarray, a: float, b: float, fs_out: float) -> np.ndarray:
    """The signal x (sampled from t0 at fs_src) at centres + a .. b, fs_out grid, by linear interpolation; NaN rows out of range."""
    g = a + np.arange(int(round((b - a) * fs_out))) / fs_out  # noqa: RUF046
    out = np.full((centres.size, g.size), np.nan, np.float32)
    pos = (centres[:, None] + g[None, :] - t0) * fs_src
    ok = (pos[:, 0] >= 0) & (pos[:, -1] <= x.size - 1)
    if ok.any():
        out[ok] = np.interp(pos[ok].ravel(), np.arange(x.size), x).reshape(-1, g.size)
    return out


def _block_templates(S: np.ndarray, block: int = 30, half: int = 60) -> np.ndarray:
    T = np.full_like(S, np.nan)
    for i in range(0, S.shape[0], block):
        lo, hi = max(0, i - half), min(S.shape[0], i + block + half)
        X = S[lo:hi]
        X = X[np.isfinite(X).all(1)]
        if len(X) >= 5:
            T[i:i + block] = np.median(X, axis=0)
    return T


def _rowcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    d = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (A * B).sum(1) / d, np.nan)


def _shifted_corr(S: np.ndarray, T: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the best correlation of S against T over integer lags +/- max_lag (and that lag)."""
    n = S.shape[1]
    best = np.full(S.shape[0], -2.0)
    lag = np.zeros(S.shape[0])
    for L in range(-max_lag, max_lag + 1):
        c = _rowcorr(S[:, max_lag + L:n - max_lag + L], T[:, max_lag:n - max_lag])
        c = np.where(np.isfinite(c), c, -2.0)
        up = c > best
        best[up], lag[up] = c[up], L
    best[best <= -2.0] = np.nan
    return best, lag


def _refine_r(r: np.ndarray, te0: float, xe: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """R peaks refined against the local QRS template (two passes); returns the refined times and the lag (s)."""
    rr = r.copy()
    pad = REF_LAG / 1000.0
    for _ in range(2):
        S = _snippets(te0, fs, xe, rr, -REF_WIN - pad, REF_WIN + pad, REF_FS)
        T = _block_templates(S)
        best, lag = _shifted_corr(S, T, REF_LAG)
        # sub-sample: parabola through the correlation at lag-1, lag, lag+1
        n = S.shape[1]
        cm, c0, cp = (np.full(r.size, np.nan) for _ in range(3))
        for i_, L in enumerate((-1, 0, 1)):
            Ls = np.clip(lag + L, -REF_LAG, REF_LAG).astype(int)
            vals = np.full(r.size, np.nan)
            for Lv in np.unique(Ls):
                m = Ls == Lv
                vals[m] = _rowcorr(S[m][:, REF_LAG + Lv:n - REF_LAG + Lv], T[m][:, REF_LAG:n - REF_LAG])
            (cm, c0, cp)[i_][:] = vals
        den = cm - 2 * c0 + cp
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.where(np.abs(den) > 1e-12, 0.5 * (cm - cp) / den, 0.0)
        sub = np.clip(np.nan_to_num(sub), -0.5, 0.5)
        shift = (lag + sub) / REF_FS
        shift = np.where(np.isfinite(best) & (best > 0.8), shift, 0.0)
        rr = rr + shift
    return rr, rr - r


def _rr_irregular(r: np.ndarray) -> np.ndarray:
    rr = np.diff(r, prepend=np.nan)
    med = median_filter(np.where(np.isfinite(rr), rr, np.nanmedian(rr)), size=31, mode="nearest")
    return np.abs(rr / med - 1) > 0.2


def _clip_share(t0: float, x: np.ndarray, centres: np.ndarray, a: float, b: float, fs: float = FS) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    rail = ((x == lo) | (x == hi)).astype(np.float32)
    S = _snippets(t0, fs, rail, centres, a, b, fs)
    return np.nanmean(S, axis=1)


def beat_quality(r: np.ndarray, te0: float, xe_band: np.ndarray, xe_raw: np.ndarray, tp0: float, xp_band: np.ndarray,
                 xp_raw: np.ndarray, fs: float = FS) -> dict:
    """Per-beat quality and the pulse and QRS shapes :func:`find_slips` needs, from the caller's filtered signals.

    For every beat: the QRS (+/- 0.12 s at 250 Hz) and the R-aligned pulse (0.1-1.0 s after R at 125 Hz); each
    beat's correlation with the local median beat (blocks of 30 beats, median over +/- 60; the pulse at its best shift
    within +/- 40 ms); irregular R-R (more than 20% from the 31-beat median); clipping (samples at the stretch's
    lowest or highest raw value); and the R peak refined against the local QRS template (+/- 10 ms, two passes).

    Args:
        r: R-peak time of each beat (s).
        te0: Time of the first ECG sample (s).
        xe_band: ECG band-passed (VitalDB use: 1-40 Hz, zero phase).
        xe_raw: ECG as recorded (for clipping).
        tp0: Time of the first PPG sample (s).
        xp_band: PPG band-passed (VitalDB use: 0.5-10 Hz, zero phase).
        xp_raw: PPG as recorded (for clipping).
        fs: Sampling rate of both signals (Hz).

    Returns:
        A dict of per-beat arrays: ``r_ref``, ``r_lag``, ``ecg_corr``, ``ppg_corr``, ``rr_irr``, ``ppg_clip``,
        ``ecg_clip``, ``ppg_snip`` and ``ecg_snip``.
    """
    r_ref, r_lag = _refine_r(r, te0, xe_band, fs)
    Se = _snippets(te0, fs, xe_band, r, ECG_WIN[0], ECG_WIN[1], ECG_FS)
    Sp = _snippets(tp0, fs, xp_band, r, PPG_WIN[0], PPG_WIN[1], PPG_FS)
    ecg_corr = _rowcorr(Se, _block_templates(Se))
    ppg_corr, _ = _shifted_corr(Sp, _block_templates(Sp), 5)
    return {"r_ref": r_ref, "r_lag": r_lag.astype(np.float32), "ecg_corr": ecg_corr.astype(np.float32),
            "ppg_corr": ppg_corr.astype(np.float32), "rr_irr": _rr_irregular(r),
            "ppg_clip": _clip_share(tp0, xp_raw, r, *PPG_WIN, fs).astype(np.float32),
            "ecg_clip": _clip_share(te0, xe_raw, r, *ECG_WIN, fs).astype(np.float32),
            "ppg_snip": Sp.astype(np.float16), "ecg_snip": Se.astype(np.float16)}


def _resp_phase(t: np.ndarray, y: np.ndarray, fs: float = 4.0, win: float = 64.0, step: float = 16.0) -> np.ndarray:
    """Respiratory phase (rad) at each time t from the delay series y.

    the strongest peak 0.1-0.7 Hz in 64 s windows where it stands out (peak power at least 6 times the band's
    median), the frequency interpolated and integrated; NaN where no window has a clear peak.
    """
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 200:
        return np.full(ok.size, np.nan)
    g = np.arange(t[0], t[-1], 1 / fs)
    u = np.interp(g, t, y)
    u = u - median_filter(u, size=int(30 * fs), mode="nearest")
    nw = int(win * fs)
    fr = np.fft.rfftfreq(nw, 1 / fs)
    band_ = (fr >= 0.1) & (fr <= 0.7)
    cen, fq = [], []
    for i in range(0, g.size - nw, int(step * fs)):
        seg = u[i:i + nw] * np.hanning(nw)
        P = np.abs(np.fft.rfft(seg)) ** 2
        pb = P[band_]
        k = int(np.argmax(pb))
        if pb[k] >= 6 * np.median(pb):
            cen.append(g[i + nw // 2])
            fq.append(fr[band_][k])
    if len(cen) < 2:
        return np.full(ok.size, np.nan)
    f_g = np.interp(g, cen, fq)
    phi_g = 2 * np.pi * np.cumsum(f_g) / fs
    # no phase further than one window from a window with a clear peak
    far = np.min(np.abs(g[:, None] - np.array(cen)[None, :]), axis=1) > win if len(cen) < 2000 else np.zeros(g.size, bool)
    phi_g[far] = np.nan
    out = np.full(ok.size, np.nan)
    out[ok] = np.interp(t, g, phi_g)
    return out


def _step_scores(t: np.ndarray, y: np.ndarray, sd: float, sign: int, phi: np.ndarray | None) -> dict:
    """sawtooth_char.step_scores with the breathing fitted too.

    cos and sin of the respiratory phase as two more columns in every model (no step, one-frame step, ramp, free
    step), where the phase is known.
    """
    n = y.size
    W, RH = GLR_W, RAMP_HALF
    out = {k: np.full(n, np.nan) for k in ("llr", "fhat", "sharp", "llr_free")}
    if n < 2 * W + 2:
        return out
    ys = sign * y
    b = np.arange(W, n - W + 1)
    off = np.arange(-W, W)
    Y = ys[b[:, None] + off[None, :]]
    T = t[b[:, None] + off[None, :]] - t[b][:, None]
    T = T / max(np.median(np.abs(T)), 1e-9)
    H = np.broadcast_to((off >= 0).astype(float), Y.shape)
    ramp = np.broadcast_to(np.clip((off + RH + 0.5) / (2 * RH + 1), 0, 1), Y.shape)
    cols = [np.ones_like(Y), T]
    if phi is not None and np.isfinite(phi).any():
        P = phi[b[:, None] + off[None, :]]
        have = np.isfinite(P).all(1)[:, None]
        cols += [np.where(have, np.cos(np.nan_to_num(P)), 0.0), np.where(have, np.sin(np.nan_to_num(P)), 0.0)]

    def rss(X: np.ndarray, Yr: np.ndarray):
        XtX = np.einsum("nki,nkj->nij", X, X)
        XtY = np.einsum("nki,nk->ni", X, Yr)
        beta = np.linalg.solve(XtX + 1e-9 * np.eye(X.shape[2])[None], XtY[..., None])[..., 0]
        r = Yr - np.einsum("nki,ni->nk", X, beta)
        return np.einsum("nk,nk->n", r, r), beta

    X0 = np.stack(cols, axis=2)
    rss0, _ = rss(X0, Y)
    rss1, _ = rss(X0, Y - FRAME * H)
    rssr, _ = rss(X0, Y - FRAME * ramp)
    rss2, beta2 = rss(np.concatenate([X0, H[..., None]], axis=2), Y)
    s2 = 2 * sd ** 2
    out["llr"][b] = (rss0 - rss1) / s2
    out["sharp"][b] = (rssr - rss1) / s2
    out["fhat"][b] = beta2[:, -1]
    out["llr_free"][b] = (rss0 - rss2) / s2
    return out


def _strong_steps(t: np.ndarray, y: np.ndarray, sd: float, sign: int, sc: dict) -> list[int]:
    """sawtooth_char.detect's strong-slip rule on these scores."""
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    quiet = _local_noise(y) <= NOISE_GATE * sd
    ok = np.zeros(y.size, bool)
    for b in np.flatnonzero((llr > LLR_STRONG) & (f > 0.6 * FRAME) & (f < 1.4 * FRAME) & (sh > SHARP_MIN) & quiet):
        pe = _long_step(y, int(b), sign)
        pe2 = _long_step(y, int(b), sign, PERSIST_SHORT)
        ok[b] = (0.5 * FRAME < pe < 1.5 * FRAME) or (0.5 * FRAME < pe2 < 1.5 * FRAME)
    out: list[int] = []
    for b in np.flatnonzero(ok):
        lo, hi = max(b - 30 * PPB, 0), min(b + 30 * PPB + 1, llr.size)
        if llr[b] < np.nanmax(llr[lo:hi]):
            continue
        if out and t[b] - t[out[-1]] < 60:
            if llr[b] > llr[out[-1]]:
                out[-1] = int(b)
            continue
        out.append(int(b))
    return out


def _shape_measures(r: np.ndarray, Sp: np.ndarray, Se: np.ndarray, F: dict, tau: float, half: float = 45.0,
                   guard: float = 3.0) -> dict:
    """label_quality.event_measures on the stored pulses: the keys its rule reads."""
    out: dict = {}
    tpl = {}
    bad, irr = [], []
    steps = {}
    for side, lo, hi in (("before", tau - half, tau - guard), ("after", tau + guard, tau + half)):
        i = np.flatnonzero((r >= lo) & (r <= hi))
        out[side + "_beats"] = int(i.size)
        if i.size >= 3:
            rr = np.diff(r[i])
            irr.append(float(np.mean(np.abs(rr / np.median(rr) - 1) > 0.2)))
        for name, S in (("ppg", Sp), ("ecg", Se)):
            X = np.asarray(S[i], float)
            X = X[np.isfinite(X).all(1)] if X.size else X
            if len(X) >= 5:
                m = np.median(X, axis=0)
                cc = _rowcorr(X, np.broadcast_to(m, X.shape))
                bad.append(float(np.mean(cc < GOOD_CORR)))
                tpl[(side, name)] = m
        for fid in ("foot", "up", "peak"):
            v = (F[fid][i] - r[i]) if i.size else np.empty(0)
            v = v[np.isfinite(v)]
            steps.setdefault(fid, {})[side] = float(np.median(v) * 1000) if v.size >= 3 else None
    st = [steps[f]["after"] - steps[f]["before"] for f in steps
          if steps[f].get("after") is not None and steps[f].get("before") is not None]
    out["fid_step_ms"] = float(np.median(st)) if st else None
    out["fid_spread_ms"] = float(max(st) - min(st)) if len(st) >= 2 else None
    if ("before", "ppg") in tpl and ("after", "ppg") in tpl:
        s, c, c0 = _best_shift(tpl[("before", "ppg")], tpl[("after", "ppg")], PPG_FS)
        out.update(ppg_shift_ms=s, ppg_corr_best=c, ppg_corr_zero=c0)
    out["worst_bad"] = max(bad) if bad else None
    out["worst_rr_irregular"] = max(irr) if irr else None
    return out


def _beat_evidence(tau: float, sign: int, ctx: dict) -> dict:
    """At a step.

    the arterial check and the heart-rate change (medians 3-30 s after minus before, good beats where there are 10
    or more each side). arterial: "slip" (ART-PPG steps 0.6-1.4 frames the sawtooth's way, ECG-ART under 0.3 frame),
    "circulation" (ECG-ART steps 0.3 frame or more), "neither", or None without a clean arterial line.
    """
    r, art, up, good = ctx["r"], ctx.get("art"), ctx["F"]["up"], ctx["good"]
    bef = (r >= tau - 30) & (r <= tau - 3)
    aft = (r >= tau + 3) & (r <= tau + 30)

    def step(y):
        for ok in (good, np.ones(r.size, bool)):
            b_, a_ = y[bef & ok & np.isfinite(y)], y[aft & ok & np.isfinite(y)]
            if b_.size >= 10 and a_.size >= 10:
                return float(np.median(a_) - np.median(b_))
        return None
    hr = 60.0 / np.diff(r, prepend=np.nan)
    hr = np.where((hr > 20) & (hr < 250), hr, np.nan)
    ep = step(up - r)
    out = {"hr_step": step(hr), "arterial": None, "ep_ms": round(ep * 1000, 1) if ep is not None else None}
    if art is not None:
        ea, ap = step(art - r), step(up - art)
        if ea is not None and ap is not None and ep is not None:
            out.update(ea_ms=round(ea * 1000, 1), ap_ms=round(ap * 1000, 1))
            # the circulation: ECG-ART carries the step (same sign as ECG-PPG) and ART-PPG stays within 0.6 frame.
            # An ECG-ART step against ECG-PPG, or with ART-PPG moving more than a frame, is the arterial upstroke
            # being found elsewhere in the pulse (damping, a flush), not the circulation
            if abs(ea) >= ART_CIRC * FRAME and np.sign(ea) == np.sign(ep) and abs(ap) < ART_SLIP[0] * FRAME:
                out["arterial"] = "circulation"
            elif ART_SLIP[0] * FRAME <= sign * ap <= ART_SLIP[1] * FRAME and abs(ea) < ART_CIRC * FRAME:
                out["arterial"] = "slip"
            else:
                out["arterial"] = "neither"
    if out["hr_step"] is not None:
        out["hr_step"] = round(out["hr_step"], 1)
    return out


def _adjudicate(c: str, m: dict, ev: dict, step_like: bool) -> tuple[str, str | None]:
    """The beats' evidence over the shape call.

    The arterial line decides where it can: a circulation change vetoes; a confirmed slip stands for a step the
    shape could not judge (or found unclear), unless the pulse clearly did not move. Without arterial confirmation,
    a heart-rate change of 10 /min or more at the step vetoes it.
    """
    # the circulation vetoes only a step the shape did not confirm: at shape-confirmed slips of a fitted sawtooth,
    # ECG-ART carried the step in 3 % of slips (v19, recordings with an arterial line), mostly in one recording whose
    # arterial upstroke moves with every slip: there the arterial timing, not the slip, is in doubt
    if ev.get("arterial") == "circulation" and c not in ("no slip", "slip"):
        return "no slip", "arterial: the circulation (ECG-ART moves)"
    if ev.get("arterial") == "slip":
        sh_, cb_ = m.get("ppg_shift_ms"), m.get("ppg_corr_best")
        clear_no = sh_ is not None and abs(sh_) < 10 and cb_ is not None and cb_ >= 0.95
        ep = ev.get("ep_ms")
        frame_sized = ep is not None and 0.5 * FRAME * 1000 <= abs(ep) <= 1.5 * FRAME * 1000
        if c in ("cannot judge", "shape only", "step, shape unclear", "unresolved") and (step_like or frame_sized) \
                and not clear_no:
            return "slip", "arterial confirms (shape could not)"
        return c, None
    hs = ev.get("hr_step")
    if hs is not None and abs(hs) >= HR_VETO and c != "no slip":
        return "no slip", f"heart rate changed {hs:+.0f} /min"
    return c, None


def _shape_call(m: dict, sign: int, P: dict) -> str:
    """label_quality.call (the rule fitted on the even-numbered labels), and the shift must go the sawtooth's way."""
    c = _call(m, P["bad_beats_max"], P["rr_irregular_max"], P["match_min"], P["fiducial_spread_max_ms"] or 1e9)
    if c == "slip" and np.sign(m.get("ppg_shift_ms") or 0) != sign:
        return "no slip"
    return c


def _evidence(t: np.ndarray, sc: dict, y: np.ndarray, sign: int, g0: float, g1: float) -> np.ndarray:
    """Per second of the stretch, the best one-frame score of any boundary in it.

    the log likelihood ratio, at least -15, capped at 30 where the step looks like a slip (free step 0.5-1.6 frames,
    sharper than a ramp, the new level holding over 15-40 or 15-75 beats) and at 8 where it does not (at most 0
    where the free step is not 0.4-1.7 frames); -inf where no boundary was scored.
    """
    e = np.full(int(np.ceil(g1 - g0)) + 1, -np.inf)
    llr, f, sh = sc["llr"], sc["fhat"], sc["sharp"]
    ok = np.isfinite(llr)
    s = np.clip(llr, -15, 8)
    s = np.where((f > 0.4 * FRAME) & (f < 1.7 * FRAME), s, np.minimum(s, 0))
    for b in np.flatnonzero(ok & (llr > 8) & (f > 0.5 * FRAME) & (f < 1.6 * FRAME) & (sh > 0)):
        pl = _long_step(y, int(b), sign)
        ps = _long_step(y, int(b), sign, PERSIST_SHORT)
        if (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME):
            s[b] = min(llr[b], 30.0)
    k = (t - g0).astype(int)
    np.maximum.at(e, np.clip(k[ok], 0, e.size - 1), s[ok])
    return e


def _window_max(e: np.ndarray, w: int) -> np.ndarray:
    m = maximum_filter1d(e, size=2 * w + 1, mode="constant", cval=-np.inf)
    return np.where(np.isfinite(m), m, 0.0)


def _win_for(L: float) -> int:
    return int(np.clip(0.02 * L, 10, 30))


def _chain_nodes(e: np.ndarray, w: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Seconds with positive evidence that are the best within +/- w s (one per plateau, its middle).

    the places a slip may sit.
    """
    m = maximum_filter1d(e, size=2 * w + 1, mode="constant", cval=-np.inf)
    k = np.flatnonzero((e > 0) & (e >= m))
    keep, grp = [], []
    for i in k:
        if grp and i - grp[-1] > w:
            keep.append(grp[len(grp) // 2])
            grp = []
        grp.append(i)
    if grp:
        keep.append(grp[len(grp) // 2])
    keep = np.array(keep, int)
    return keep.astype(float), e[keep]


def _chain_fit(T: np.ndarray, v: np.ndarray, Ew: np.ndarray, L: float) -> tuple[float, list[int], list[str]]:
    """The best chain of slips for tooth L: each slip follows the previous one a whole number of teeth...

    The best chain of slips for tooth L: each slip follows the previous one a whole number of teeth later (within
    +/- w x sqrt(teeth)), the slips the grid predicts in between scored on their evidence (clearly no step there
    costs up to 15 each), half a point per tooth; or after a phase jump, at a cost of JUMP. Returns the score, the
    chain (node indices) and each link's kind.
    """
    N = T.size
    if N == 0:
        return 0.0, [], []
    w = _win_for(L)
    G = T[None, :] - T[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.round(G / L)
    grid = (m >= 1) & (m <= M_MAX) & (np.abs(G - m * L) <= w * np.sqrt(np.maximum(m, 1)))
    inter = np.zeros((N, N))
    span = Ew.size - 1
    for j in range(1, M_MAX):
        pos = T[:, None] + j * np.where(m > 0, G / np.maximum(m, 1), 0)
        inter += np.where(grid & (j < m), Ew[np.clip(pos.astype(int), 0, span)], 0.0)
    tr = np.where(grid, np.maximum(inter - 0.5 * m, -JUMP), -JUMP)
    tr = np.where(G > 0.25 * L, tr, -np.inf)          # two slips closer than a quarter tooth cannot both be on it
    best = np.full(N, -np.inf)
    prev = np.full(N, -1)
    for k in range(N):
        cand = best[:k] + tr[:k, k] if k else np.empty(0)
        if cand.size and np.max(cand) > 0:
            j = int(np.argmax(cand))
            best[k], prev[k] = v[k] + cand[j], j
        else:
            best[k] = v[k]
    k = int(np.argmax(best))
    S = float(best[k])
    chain = []
    while k >= 0:
        chain.append(k)
        k = int(prev[k])
    chain = chain[::-1]
    links = ["grid" if grid[a_, b_] and tr[a_, b_] > -JUMP else "jump" for a_, b_ in zip(chain[:-1], chain[1:])]  # noqa: RUF007
    return S, chain, links


def _period_fit(e: np.ndarray, L_lo: float, L_hi: float) -> dict:
    """The tooth length whose best chain scores highest, over L_lo-L_hi (log steps of 0.3%, then 0.05%...

    The tooth length whose best chain scores highest, over L_lo-L_hi (log steps of 0.3%, then 0.05% about the best);
    the chain's score at half and twice that tooth is reported.
    """
    cache: dict = {}

    def ew(L):
        w = _win_for(L)
        if w not in cache:
            cache[w] = _window_max(e, w)
        return cache[w]
    span = e.size - 1
    L_hi = min(L_hi, 0.9 * span)
    T, v = _chain_nodes(e)
    if L_hi <= L_lo or T.size < 2:
        return {}
    Ls = np.exp(np.arange(np.log(L_lo), np.log(L_hi), 0.003))
    sc_ = [_chain_fit(T, v, ew(L), L)[0] for L in Ls]
    k = int(np.argmax(sc_))
    Lf = np.exp(np.arange(np.log(Ls[k]) - 0.003, np.log(Ls[k]) + 0.003, 0.0005))
    scf = [_chain_fit(T, v, ew(L), L)[0] for L in Lf]
    L = float(Lf[int(np.argmax(scf))])
    S, chain, links = _chain_fit(T, v, ew(L), L)
    return {"L": L, "score": S, "chain_t": [float(T[i]) for i in chain], "links": links,
            "score_half": _chain_fit(T, v, ew(L / 2), L / 2)[0] if L / 2 >= 30 else None,
            "score_double": _chain_fit(T, v, ew(2 * L), 2 * L)[0]}


def _predicted(L: float, t_ref: float, g0: float, g1: float) -> np.ndarray:
    k0 = int(np.ceil((g0 - t_ref) / L))
    k1 = int(np.floor((g1 - t_ref) / L))
    return t_ref + L * np.arange(k0, k1 + 1)


def _judge_at(p: float, w: float, sign: int, ctx: dict, gap: bool = False) -> dict:
    """At a predicted time.

    the best one-frame boundary within w and its tests; the shape test there, and, unless that confirms a slip, at
    the five boundaries of the window with the largest free step the sawtooth's way (0.4-1.7 frames) and at the
    prediction itself, keeping the first that confirms one (a boundary that failed the step tests can still be where
    the pulse moved).
    """
    t, sc, y = ctx["t"], ctx["sc"], ctx["y"]
    sel = np.flatnonzero((np.abs(t - p) <= w) & np.isfinite(sc["llr"]))
    row = {"t_pred": round(p, 2), "w": round(float(w), 1)}
    tau = p
    step = False
    bp = None
    if sel.size:
        bp = int(sel[np.argmax(sc["llr"][sel])])
        f, llr, sh = sc["fhat"][bp], sc["llr"][bp], sc["sharp"][bp]
        pl = _long_step(y, bp, sign)
        ps = _long_step(y, bp, sign, PERSIST_SHORT)
        held = (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME)
        edge = not np.isfinite(pl) and not np.isfinite(ps)
        step = bool(llr > LLR_GRID and sh > -10 and ((held and 0.5 * FRAME < f < 1.6 * FRAME) or
                                                          (edge and 0.5 * FRAME < f < 1.8 * FRAME)))
        row.update(llr=round(float(llr), 1), step_ms=round(float(f) * 1000, 1), sharp=round(float(sh), 1), step=step)
        tau = float(0.5 * (t[bp - 1] + t[bp]))

    def judged(tau_):
        m = _shape_measures(ctx["r"], ctx["Sp"], ctx["Se"], ctx["F"], tau_)
        return m, _shape_call(m, sign, ctx["rule"])
    m, c = judged(tau)
    if c == "slip" and not step and not (bp is not None and sc["llr"][bp] > LLR_GRID):
        c = "shape only"
    row["bp"] = bp if step else None
    if c not in ("slip",) and sel.size:
        ff = sc["fhat"][sel] * sign
        cand = sel[(ff > 0.4 * FRAME) & (ff < 1.7 * FRAME)]
        cand = cand[np.argsort(-sc["llr_free"][cand])]
        tried = [tau]
        for b_ in list(cand[:20]) + [None]:
            tb = p if b_ is None else float(0.5 * (t[b_ - 1] + t[b_]))
            if min(abs(tb - x) for x in tried) < 3:
                continue
            tried.append(tb)
            m2, c2 = judged(tb)
            # a boundary the shape prefers must still show a one-frame step more likely than none (llr above the grid
            # threshold): the shape test alone, tried at several places, finds a pulse that moved by chance
            if c2 == "slip" and (b_ is None or not sc["llr"][b_] > LLR_GRID):
                c2 = "shape only"
            if c2 == "slip":
                m, c, tau, row["bp"] = m2, c2, tb, b_
                row["moved_s"] = round(tb - (float(0.5 * (t[bp - 1] + t[bp])) if bp is not None else p), 1)
                break
            if len(tried) > 6:
                break
    row["shape"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                    if k in ("ppg_shift_ms", "ppg_corr_best", "fid_step_ms", "fid_spread_ms", "worst_bad", "worst_rr_irregular")}
    row["prob"] = _slip_probability(p, w, sel, {**ctx, "sign": sign}, P_NONE_GAP if gap else P_NONE)
    if row["prob"]:
        row["p_slip"] = row["prob"][str(TEMPER)][0]
        row["q05"], row["q50"], row["q95"] = row["prob"][str(TEMPER_TIMES)][1:]
    if ctx.get("evidence") == "on":
        ev = _beat_evidence(tau, sign, ctx)
        row["evidence"] = ev
        c, why = _adjudicate(c, m, ev, step or (row.get("bp") is not None) or
                            bool(sel.size and np.nanmax(sc["llr"][sel]) > LLR_GRID))
        if why:
            row["evidence_note"] = why
    row["call"] = c
    # --grid-veto clear: a one-frame step at a predicted slip stands unless the pulse clearly did not move (best shift
    # under 10 ms with the shape matching); --grid-veto rule: the shape rule must confirm it
    if step and c == "no slip" and ctx.get("grid_veto") == "clear" and not row.get("evidence_note"):
        sh_, cb_ = m.get("ppg_shift_ms"), m.get("ppg_corr_best")
        if not (sh_ is not None and abs(sh_) < 10 and cb_ is not None and cb_ >= ctx["rule"]["match_min"]):
            c = "step, shape unclear"
    row["outcome"] = ("slip" if c == "slip" else
                      "step, cannot judge" if step and c == "cannot judge" else
                      "step, shape unclear" if c == "step, shape unclear" else
                      "shape only" if c == "shape only" else
                      "no slip" if c == "no slip" else "unresolved")
    row["t"] = round(tau, 3)
    return row


def _slip_probability(p: float, w: float, sel: np.ndarray, ctx: dict, p_none: float = P_NONE) -> dict:
    """The fallback where the slip cannot be placed or confirmed.

    the probability that one slip lies within the window about the prediction, and where. Over the window's
    boundaries, the prior is Gaussian about the prediction (sd half the window) with P_NONE for no slip at all; each
    boundary's evidence is its one-frame log likelihood ratio, tempered (multiplied by tau) because neighbouring
    boundaries share their beats; the posterior gives P(slip in the window) and its 5th, 50th and 95th centiles in
    time. Only a step that looks like a slip counts as evidence for one. One row per tau, for calibration on the
    injections.
    """
    t, sc = ctx["t"], ctx["sc"]
    if not sel.size:
        return {}
    tb = 0.5 * (t[sel - 1] + t[sel])
    lp = -0.5 * ((tb - p) / (0.5 * w)) ** 2
    lp -= np.log(np.exp(lp - lp.max()).sum()) + lp.max()
    llr = np.clip(sc["llr"][sel], -50, 200)
    f = sc["fhat"][sel] * ctx.get("sign", 1)
    # only a step that looks like a slip is evidence for one: frame-sized, sharper than a ramp, the new level holding
    # (a large or gradual change of the delay also favours one frame over none)
    like = (f > 0.5 * FRAME) & (f < 1.6 * FRAME) & (sc["sharp"][sel] > 0) & (llr > LLR_GRID)
    for i in np.flatnonzero(like):
        b = int(sel[i])
        sg = ctx.get("sign", 1)
        pl, ps = _long_step(ctx["y"], b, sg), _long_step(ctx["y"], b, sg, PERSIST_SHORT)
        like[i] = (0.4 * FRAME < pl < 1.6 * FRAME) or (0.4 * FRAME < ps < 1.6 * FRAME)
    llr = np.where(like, llr, np.minimum(llr, 0))
    out = {}
    for tau in TEMPERS:
        a_ = np.log(1 - p_none) + lp + tau * llr
        z0 = np.log(p_none)
        mx = max(a_.max(), z0)
        wts = np.exp(a_ - mx)
        p_slip = float(wts.sum() / (wts.sum() + np.exp(z0 - mx)))
        cdf = np.cumsum(wts) / wts.sum()
        q = [float(tb[min(np.searchsorted(cdf, x), tb.size - 1)]) for x in (0.05, 0.5, 0.95)]
        out[str(tau)] = [round(p_slip, 4)] + [round(x, 2) for x in q]
    return out


def _place_row(row: dict, sign: int, ctx: dict) -> None:
    if row.get("bp") is None:
        return
    b = int(ctx["beat_pt"][row["bp"]])
    p = _place(b, ctx["F"], ctx["r"], ctx["sd"], sign)
    if p:
        row.update(t=round(p["t"], 3), lo=round(p["lo"], 3), hi=round(p["hi"], 3))


def _runs_of(chain_t: list[float], links: list[str]) -> list[list[float]]:
    runs, cur = [], [chain_t[0]] if chain_t else []
    for t_, l_ in zip(chain_t[1:], links):
        if l_ == "grid":
            cur.append(t_)
        else:
            runs.append(cur)
            cur = [t_]
    if cur:
        runs.append(cur)
    return runs


def _run_slots(runs: list[list[float]], L: float, g0: float, g1: float) -> list[tuple[int, float, bool]]:
    """Predicted slips of every run of two or more slips, over its own stretch and on across the gaps to...

    Predicted slips of every run of two or more slips, over its own stretch and on across the gaps to the
    neighbouring runs' slips (or the ends of the stretch), so in a gap both phases are tried. Each run's phase and
    tooth by least squares on its slips.
    """
    out = []
    for k, ref in enumerate(runs):
        if len(ref) < 2:
            continue
        ref = np.array(ref)
        n_ = np.r_[0, np.cumsum(np.maximum(np.round(np.diff(ref) / L), 1))]
        A = np.column_stack([np.ones_like(n_), n_])
        t0_, L_ = np.linalg.lstsq(A, ref, rcond=None)[0] if len(ref) >= 3 else (ref[0], L)
        if not 0.9 * L < L_ < 1.1 * L:
            t0_, L_ = float(np.median(ref - n_ * L)), L
        before = [x for j, r_ in enumerate(runs) if j < k for x in r_]
        after = [x for j, r_ in enumerate(runs) if j > k for x in r_]
        # across the whole gap to the neighbouring runs: where the phase jumped is then found, not assumed halfway
        lo_ = max(before) + 0.25 * L if before else g0
        hi_ = min(after) - 0.25 * L if after else g1
        # a prediction in a gap between runs (beyond this run's slips, before the next run's) is one of two phases
        # tried there, so no slip at it is likelier
        out += [(k, float(p), bool((before and p < ref[0] - 0.5 * L) or (after and p > ref[-1] + 0.5 * L)))
                for p in _predicted(L_, t0_, lo_, hi_)]
    return out


def _fit_direction(sign: int, ctx: dict, prior: tuple | None, g0: float, g1: float) -> dict:
    t, sc = ctx["t"], ctx["sc_by_sign"][sign]
    ctx = {**ctx, "sc": sc}
    e = _evidence(t, sc, ctx["y"], sign, g0, g1)
    best = None
    span = g1 - g0
    tried = []
    for name, rng_ in ((("prior", prior[:2]),) if prior else ()) + (("broad", (BROAD[0], BROAD[1])),):
        pf = _period_fit(e, *rng_)
        if not pf:
            continue
        tried.append((name, pf))
        if 2 * pf["L"] < 0.9 * span:            # half-period check: the same chain search at twice the tooth
            pd = _period_fit(e, 1.95 * pf["L"], 2.05 * pf["L"])
            if pd:
                tried.append((name + " x2", pd))
    for name, pf in tried:
        chain_abs = [g0 + x for x in pf["chain_t"]]
        runs = _runs_of(chain_abs, pf["links"])
        rows = [{**_judge_at(p, _win_for(pf["L"]), sign, ctx, gap), "run": k, "gap": gap} for k, p, gap in _run_slots(runs, pf["L"], g0, g1)]
        pf.update(source=name, runs=sum(len(r_) >= 2 for r_ in runs), rows=rows,
                  confirmed=sum(r_["outcome"] == "slip" for r_ in rows), chain_abs=chain_abs, runs_t=runs,
                  clear_no=sum(r_["outcome"] == "no slip" for r_ in rows))
    # prior against broad: the most confirmed slips, then the fewest predictions the shape clearly rejects
    base = [pf for name, pf in tried if not name.endswith("x2") and "rows" in pf]
    for pf in base:
        if best is None or (pf["confirmed"], -pf["clear_no"]) > (best["confirmed"], -best["clear_no"]):
            best = pf
    # half-period check: twice the tooth replaces it only if the predictions the doubled grid leaves out (every other
    # one) confirm no slip and the shape clearly rejects at least two of them
    if best is not None:
        for name, pf in tried:
            if name != best["source"] + " x2" or "rows" not in pf:
                continue
            t2 = np.array([r_["t_pred"] for r_ in pf["rows"]])
            w2 = _win_for(best["L"])
            left = [r_ for r_ in best["rows"] if not t2.size or np.min(np.abs(t2 - r_["t_pred"])) > w2]
            best["half_check"] = {"left_out": len(left), "confirmed": sum(r_["outcome"] == "slip" for r_ in left),
                                  "clear_no": sum(r_["outcome"] == "no slip" for r_ in left)}
            if left and best["half_check"]["confirmed"] == 0 and best["half_check"]["clear_no"] >= 2 and pf["confirmed"] >= 2:
                pf["half_check"] = best["half_check"]
                best = pf
    if best is None:
        return {"sign": sign, "confirmed": 0, "rows": [], "e": e}
    # again from the accepted slips: each run's phase and tooth from where its slips were found
    for _ in range(2):
        runs = []
        for k in sorted({r_["run"] for r_ in best["rows"]}):
            acc = [r_["t"] for r_ in best["rows"] if r_["run"] == k and r_["outcome"] in ("slip", "step, cannot judge", "step, shape unclear")]
            if len(acc) >= 2:
                runs.append(acc)
        if not runs:
            break
        rows = [{**_judge_at(p, _win_for(best["L"]), sign, ctx, gap), "run": k, "gap": gap} for k, p, gap in _run_slots(runs, best["L"], g0, g1)]
        best["rows"], best["confirmed"] = rows, sum(r_["outcome"] == "slip" for r_ in rows)
    for r_ in best["rows"]:
        if r_["outcome"] in ("slip", "step, cannot judge", "step, shape unclear"):
            _place_row(r_, sign, ctx)
    # the tooth: from the accepted slips of each run (whole numbers of teeth apart), pooled
    num, den = 0.0, 0.0
    for k in sorted({r_["run"] for r_ in best["rows"]}):
        acc = np.array([r_["t"] for r_ in best["rows"] if r_["run"] == k and r_["outcome"] == "slip"])
        if acc.size >= 2:
            n_ = np.round(np.diff(acc) / best["L"])
            num, den = num + np.sum(np.diff(acc)[n_ >= 1]), den + np.sum(n_[n_ >= 1])
    best["L_search"] = best["L"]
    if den:
        best["L"] = float(num / den)
    best["sign"] = sign
    best["rate_ppm"] = FRAME / best["L"] * 1e6
    best["e"] = e
    return best


def _drift_ppm(t: np.ndarray, y: np.ndarray, slips: list[dict], sign: int) -> float | None:
    """The delay's slope with a free step at each slip (least squares on 15-point medians), as ppm the...

    The delay's slope with a free step at each slip (least squares on 15-point medians), as ppm the sawtooth's way
    (positive: the delay shrinks between up slips, grows between down ones).
    """
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    m = t.size // 15
    if m < 40:
        return None
    tt = np.median(t[: m * 15].reshape(m, 15), 1)
    yy = np.median(y[: m * 15].reshape(m, 15), 1)
    cols = [np.ones_like(tt), tt - tt.mean()] + [(tt >= s["t"]).astype(float) for s in slips if tt[0] < s["t"] < tt[-1]]
    beta = np.linalg.lstsq(np.column_stack(cols), yy, rcond=None)[0]
    return float(-sign * beta[1] * 1e6)


def _recording_tooth(anchors: list[float], creep: float | None, prior: tuple | None, span: float) -> tuple[float, str] | None:
    """The tooth to search with.

    from two or more same-way slips, the spacing (gap / k, k chosen to match the measured drift where there is one);
    else from the drift (frame / rate) when the recording creeps at 3 ppm or more; else the setup's median tooth.
    """
    if len(anchors) >= 2:
        # the spacing itself (one tooth): choosing a fraction by the drift halved the tooth of slow injected
        # sawteeth (the drift is too noisy, sd 6.5 ppm, to tell one tooth from two)
        g = float(np.min(np.diff(sorted(anchors))))
        if g >= 300:
            return g, "slip spacing"
    if creep is not None and creep >= 3.0:
        return float(np.clip(FRAME / (creep * 1e-6), 300.0, 4 * span)), "drift"
    if prior and len(prior) >= 4:
        return float(prior[3]), "setup"
    return None


def _anchored_fit(sign: int, anchors: list[float], ctx: dict, prior: tuple, g0: float, g1: float) -> dict | None:
    """A sawtooth from confirmed slips that the period fit could not join (tier I).

    start from the setup's median tooth and each confirmed slip, step a tooth at a time outward (both ways) and
    judge the next slip where it should fall (window: the usual one, widened by the setup's spread for every tooth
    since the last confirmed slip); every confirmed slip re-anchors the phase, and once two are confirmed the tooth
    is their spacing. Returns a fit like fit_direction's when two or more slips end up confirmed on one period, else
    None.
    """
    L0, L_how = prior[3], prior[4] if len(prior) > 4 else "setup"
    lo_ok, hi_ok = (prior[0], prior[1]) if L_how == "setup" else (0.6 * L0, 1.6 * L0)
    spread = ANCHOR_SPREAD if L_how != "drift" else 0.15
    ctx = {**ctx, "sc": ctx["sc_by_sign"][sign]}
    anchors = sorted(anchors)
    half = None
    if L_how == "slip spacing" and ctx.get("anchor_check", "on") == "on" and len(anchors) >= 2:
        # the midpoint of the two closest slips, judged and reported only: on injected sawteeth it judged a missed
        # slip "no slip" again and could not be judged at gaps, so it neither rejects nor halves the tooth
        i = int(np.argmin(np.diff(anchors)))
        a0, a1 = anchors[i], anchors[i + 1]
        mid = 0.5 * (a0 + a1)
        hr = _judge_at(mid, min(_win_for(0.5 * L0), 0.15 * L0), sign, ctx)
        half = {"t_pred": round(mid, 2), "outcome": hr["outcome"], "spacing_s": round(a1 - a0, 1)}
    conf = list(anchors)
    rows = {}

    def L_now():
        if len(conf) >= 2:
            c = np.array(sorted(conf))
            n = np.round(np.diff(c) / L0)
            n = n[n >= 1]
            Ls = np.diff(c)[np.round(np.diff(c) / L0) >= 1] / n if n.size else []
            Lm = float(np.median(Ls)) if len(Ls) else L0
            return Lm if lo_ok <= Lm <= hi_ok else L0
        return L0
    for direction in (1, -1):
        last = anchors[-1] if direction > 0 else anchors[0]
        k_since = 1
        while True:
            L = L_now()
            p = last + direction * k_since * L
            if not (g0 + 0.25 * L <= p <= g1 - 0.25 * L):
                break
            w = max(_win_for(L), spread * k_since * L)
            w = min(w, 0.3 * L)
            if any(abs(p - a) < 0.5 * L for a in conf if a != last):
                k_since += 1
                continue
            r_ = _judge_at(p, w, sign, ctx)
            r_["run"] = 0
            r_["anchored"] = True
            rows[round(p, 1)] = r_
            if r_["outcome"] == "slip":
                conf.append(r_["t"])
                last, k_since = r_["t"], 1
            else:
                k_since += 1
    for a in anchors:
        rows[round(a, 1)] = {"t_pred": round(a, 2), "t": round(a, 3), "outcome": "slip", "call": "slip", "run": 0,
                             "anchored": True, "anchor": True, "w": 0.0, "shape": {}}
    out_rows = sorted(rows.values(), key=lambda r_: r_["t_pred"])
    n_conf = sum(r_["outcome"] == "slip" for r_ in out_rows)
    # two confirmed slips the same way whose spacing matches the recording's own drift already make a period
    spacing_ok = L_how == "slip spacing" and len(anchors) >= 2
    if n_conf < 2 or (n_conf - len(anchors) < 1 and not spacing_ok):
        return None
    for r_ in out_rows:
        if r_["outcome"] in ("slip", "step, cannot judge", "step, shape unclear") and not r_.get("anchor"):
            _place_row(r_, sign, ctx)
    L = L_now()
    # two confirmed slips give a tooth but not enough to correct the creep: the steps are corrected, the tooth and
    # its creep reported (a tooth from two slips was wrong in 9 of 22 injected recordings)
    creep = "corrected" if n_conf >= 3 or ctx.get("anchor_check", "on") != "on" else "reported"
    return {"sign": sign, "L": L, "L_search": L0, "rate_ppm": FRAME / L * 1e6, "confirmed": n_conf, "rows": out_rows,
            "source": f"anchored ({L_how})", "runs": 1, "clear_no": sum(r_["outcome"] == "no slip" for r_ in out_rows),
            "e": None, "creep": creep, "half_check": half}


def _find_slips(r_scan: np.ndarray, F: dict, Q: dict, pref: str, opts, art: np.ndarray | None = None) -> dict:
    """Everything for one set of beats (real or injected); Q holds that set's quality arrays (no prefix)."""
    r = Q["r_ref"] if opts.r == "template" else r_scan
    good = np.ones(r.size, bool)
    if opts.gate == "on":
        good = (np.nan_to_num(Q["ecg_corr"]) >= GOOD_CORR) & (np.nan_to_num(Q["ppg_corr"]) >= GOOD_CORR) & ~Q["rr_irr"] & \
            (np.nan_to_num(Q["ppg_clip"]) < 0.02) & (np.nan_to_num(Q["ecg_clip"]) < 0.02)
    Fg = {j: np.where(good, F[j], np.nan) for j in FIDS}
    t_pt, y, beat_pt = _point_series(Fg, r)
    gating = opts.gate
    if t_pt.size < 600:
        # too few good beats to fit at all: fit them all, and say so (the shape test still judges each slip)
        good = np.ones(r.size, bool)
        t_pt, y, beat_pt = _point_series(F, r)
        gating = "off: too few good beats"
    if t_pt.size < 600:
        raise ValueError(f"too few beats ({r.size})")
    sd = _noise_sd(y)
    y = _clean(y, sd)
    phi = _resp_phase(t_pt, y) if opts.resp == "on" else None
    ctx = {"t": t_pt, "y": y, "beat_pt": beat_pt, "r": r_scan, "F": F, "Sp": Q["ppg_snip"], "Se": Q["ecg_snip"],
           "rule": opts.rule_params, "sd": {j: _noise_sd(F[j] - r) for j in FIDS}, "grid_veto": opts.grid_veto,
           "good": good, "art": art if (opts.evidence == "on" and art is not None and np.isfinite(art).mean() > 0.3) else None}
    ctx["evidence"] = opts.evidence
    ctx["anchor_check"] = getattr(opts, "anchor_check", "on")
    ctx["sc_by_sign"] = {s: _step_scores(t_pt, y, sd, s, phi) for s in (1, -1)}
    ctx["strong"] = {s: _strong_steps(t_pt, y, sd, s, ctx["sc_by_sign"][s]) for s in (1, -1)}
    g0, g1 = float(t_pt[0]), float(t_pt[-1])
    out = {"beats": int(r.size), "gating": gating, "good_share": round(float(good.mean()), 3), "noise_ms": round(sd * 1000, 2),
           "resp_share": round(float(np.isfinite(phi).mean()), 3) if phi is not None else None,
           "strong_up": len(ctx["strong"][1]), "strong_down": len(ctx["strong"][-1]), "start": g0, "end": g1}
    fits = {s: _fit_direction(s, ctx, opts.prior_by_sign.get(s) if opts.prior == "setup" else None, g0, g1) for s in (1, -1)}
    best = max(fits.values(), key=lambda f: (f.get("confirmed", 0), f["sign"]))
    other = fits[-best["sign"]]
    out["direction"] = "up" if best["sign"] > 0 else "down"
    out["fit"] = {k: best.get(k) for k in ("L", "L_search", "rate_ppm", "clear_no", "half_check", "score", "score_half", "score_double", "source", "confirmed",
                                         "runs", "chain_abs", "links")}
    out["other_direction_confirmed"] = other.get("confirmed", 0)
    rows = best.get("rows", [])
    out["predicted"] = rows
    # a sawtooth: two predicted slips confirmed by the shape, or one and two more one-frame steps on the grid whose
    # shape cannot be judged
    unj = sum(r_["outcome"] in ("step, cannot judge", "step, shape unclear") for r_ in rows)
    if opts.saw_rule == "two confirmed":
        # two slips confirmed (shape or arterial line): one confirmed and two unjudged steps made false sawteeth with
        # short teeth in recordings carrying one slow injected slip
        out["sawtooth"] = bool(best.get("confirmed", 0) >= 2)
    else:
        out["sawtooth"] = bool(best.get("confirmed", 0) >= 2 or (best.get("confirmed", 0) >= 1 and unj >= 2))
    # every other place with slip-like evidence, either way (strong steps, and the seconds whose one-frame step passes
    # the slip tests), not at a predicted slip of the fit: kept when the shape test confirms it
    w = _win_for(best["L"]) if best.get("L") else 10
    pred_t = np.array([r_["t_pred"] for r_ in rows]) if out["sawtooth"] else np.empty(0)
    extra = []
    for s_ in (1, -1):
        sc_s = ctx["sc_by_sign"][s_]
        T, v = _chain_nodes(fits[s_]["e"])
        cands = sorted({float(g0 + x) for x, vv in zip(T, v) if vv > 8} |
                       {float(0.5 * (t_pt[bp - 1] + t_pt[bp])) for bp in ctx["strong"][s_]})
        for tb in cands:
            if s_ == best["sign"] and pred_t.size and np.min(np.abs(pred_t - tb)) <= w:
                continue
            sel = np.flatnonzero((np.abs(t_pt - tb) <= 10) & np.isfinite(sc_s["llr"]))
            bp = int(sel[np.argmax(sc_s["llr"][sel])]) if sel.size else None
            tb_ = float(0.5 * (t_pt[bp - 1] + t_pt[bp])) if bp else tb
            if any(abs(x["t"] - tb_) < 30 and x["sign"] == s_ for x in extra):
                continue
            m = _shape_measures(r_scan, Q["ppg_snip"], Q["ecg_snip"], F, tb_)
            ev, why, cl = None, None, _shape_call(m, s_, opts.rule_params)
            if ctx.get("evidence") == "on":
                ev = _beat_evidence(tb_, s_, ctx)
                cl, why = _adjudicate(cl, m, ev, bool(bp is not None and sc_s["llr"][bp] > LLR_GRID))
            o = {"t": round(tb_, 3), "sign": s_, "bp": bp, "llr": round(float(sc_s["llr"][bp]), 1) if bp else None,
                 "call": cl, "evidence": ev, "evidence_note": why,
                 "shape": {k: (round(vv, 3) if isinstance(vv, float) else vv) for k, vv in m.items()
                           if k in ("ppg_shift_ms", "ppg_corr_best", "worst_bad", "worst_rr_irregular")}}
            if o["call"] == "slip":
                _place_row(o, s_, {**ctx, "sc": sc_s})
            extra.append(o)
    # tier I: confirmed slips but no sawtooth fitted; with the setup's tooth known, search from them for the rest
    out["anchored"] = None
    if not out["sawtooth"] and opts.anchored == "on":
        for s_ in (best["sign"], -best["sign"]):
            pr = opts.prior_by_sign.get(s_) if opts.prior == "setup" else None
            anc = [o["t"] for o in extra if o["call"] == "slip" and o["sign"] == s_]
            if not anc:
                continue
            cr = _drift_ppm(ctx["F"]["up"], ctx["F"]["up"] - ctx["r"], [{"t": o["t"]} for o in extra if o["call"] == "slip"], s_)
            tooth = _recording_tooth(anc, cr, pr, g1 - g0)
            if tooth is None:
                continue
            L_rec, how = tooth
            pr2 = ((pr[0], pr[1]) if pr else (0.6 * L_rec, 1.6 * L_rec)) + ((pr[2],) if pr else (0,)) + (L_rec, how)
            af = _anchored_fit(s_, anc, ctx, pr2, g0, g1)
            if af:
                best = af
                rows = af["rows"]
                out["predicted"] = rows
                out["sawtooth"] = True
                out["anchored"] = {"sign": s_, "anchors": len(anc), "confirmed": af["confirmed"], "L": af["L"], "tooth_from": how,
                                   "drift_ppm": round(cr, 1) if cr is not None else None}
                out["direction"] = "up" if s_ > 0 else "down"
                out["fit"] = {k: af.get(k) for k in ("L", "L_search", "rate_ppm", "confirmed", "source", "runs", "clear_no",
                                                     "creep", "half_check")}
                anc_t = {round(a, 3) for a in anc}
                extra = [o for o in extra if not (o["sign"] == s_ and round(o["t"], 3) in anc_t)]
                break
    out["off_grid"] = extra
    # the slips: confirmed at a prediction, or one-frame steps there whose shape cannot be judged (sawtooth only),
    # and shape-confirmed steps elsewhere (off the grid, or the other way)
    sl = [{"t": r_["t"], "lo": r_.get("lo"), "hi": r_.get("hi"), "kind": r_["outcome"], "t_pred": r_["t_pred"],
           "sign": best["sign"]} for r_ in rows
          if r_["outcome"] == "slip" or (out["sawtooth"] and r_["outcome"] in ("step, cannot judge", "step, shape unclear"))]
    merged = []
    for x in sorted(sl, key=lambda x: x["t"]):
        if merged and x["t"] - merged[-1]["t"] < 30:
            continue
        merged.append(x)
    sl = merged
    grid_t = [x["t"] for x in sl]
    for o in extra:
        if o["call"] != "slip":
            continue
        if any(abs(x - o["t"]) < NEAR_SLIP_S for x in grid_t):
            o["call"] = "slip, near an accepted slip (not kept)"
            continue
        if any(abs(x["t"] - o["t"]) < 30 for x in sl):
            continue
        sl.append({"t": o["t"], "lo": o.get("lo"), "hi": o.get("hi"), "sign": o["sign"],
                   "kind": "off grid" if o["sign"] == best["sign"] else "other way"})
    # the fallback for predicted slips that could not be placed: one slip in [q05, q95] with probability p_slip;
    # only where the period is trusted (three or more confirmed slips)
    # two kept slips the opposite way within 5 min of each other: a transient (the delay stepped and came back), not
    # two slips; in a sawtooth the one against its direction is dropped, otherwise both
    sl.sort(key=lambda x: x["t"])
    drop = set()
    for i, x in enumerate(sl):
        for j in range(i + 1, len(sl)):
            y_ = sl[j]
            if y_["t"] - x["t"] > TRANSIENT_S:
                break
            if x.get("sign", 1) != y_.get("sign", 1):
                if out["sawtooth"]:
                    drop |= {k for k in (i, j) if sl[k].get("sign", 1) != best["sign"]}
                else:
                    drop |= {i, j}
    out["transient"] = [sl[k] for k in sorted(drop)]
    sl = [x for k, x in enumerate(sl) if k not in drop]
    out["uncertain"] = [{k: r_.get(k) for k in ("t_pred", "outcome", "p_slip", "q05", "q50", "q95")} for r_ in rows
                        if r_["outcome"] in ("unresolved", "step, cannot judge", "step, shape unclear", "shape only")
                        and r_.get("p_slip") is not None] if out["sawtooth"] and best.get("confirmed", 0) >= 3 else []
    out["inserted"] = [r_["t_pred"] for r_ in rows if r_["outcome"] == "unresolved"] if out["sawtooth"] else []
    out["slips"] = sorted(sl, key=lambda x: x["t"])
    out["sign"] = best["sign"]
    # the drift, reported and not corrected where no sawtooth is fitted: the delay's slope with a free step at each
    # slip, the sawtooth's way, with the spread of slip-free recordings as its uncertainty
    if not out["sawtooth"] and out["slips"]:
        sg = [x.get("sign", 1) for x in out["slips"]]
        s0 = 1 if sg.count(1) >= sg.count(-1) else -1
        cr = _drift_ppm(F["up"], F["up"] - r_scan, out["slips"], s0)
        if cr is not None:
            hrs = (g1 - g0) / 3600
            out["drift"] = {"ppm": round(cr, 1), "sd_ppm": DRIFT_SD_PPM, "sign": s0,
                            "over_recording_ms": round(cr * 1e-6 * (g1 - g0) * 1000, 1), "hours": round(hrs, 2),
                            "implied_tooth_min": round(FRAME / (cr * 1e-6) / 60, 1) if cr > 0.5 else None,
                            "corrected": False}
    return out

@dataclass
class SlipOptions:
    """Options of :func:`find_slips` (defaults: the method as evaluated).

    Attributes:
        r: ``"template"`` uses the template-refined R peaks for the delay series, ``"peak"`` the R times given.
        gate: ``"on"`` leaves beats that fail the quality gate out of the delay series.
        resp: ``"on"`` fits the breathing in the step test.
        rule_params: The pulse-shape rule (fitted on manual labels): ``bad_beats_max`` (share of beats unlike the
            side's median beat above which the shape cannot be judged), ``rr_irregular_max``, ``match_min`` (the
            shape match after the best shift), ``fiducial_spread_max_ms`` (or None).
        grid_veto: ``"clear"`` keeps a one-frame step at a predicted slip unless the pulse clearly did not move;
            ``"rule"`` needs the shape rule to confirm it.
        evidence: ``"on"`` lets the arterial line and the heart rate confirm or veto.
        prior_by_sign: The setup's tooth by direction, ``{+1: (lo_s, hi_s, n, median_s), -1: ...}``; empty: none.
        anchored: ``"on"`` searches from confirmed slips at the recording's own tooth when no sawtooth is fitted.
        saw_rule: ``"two confirmed"`` needs two confirmed slips for a sawtooth.
        anchor_check: ``"on"``: a sawtooth found by the anchored search corrects its creep only with three or more
            confirmed slips; with two, ``fit["creep"]`` is ``"reported"`` (steps corrected, tooth and creep reported).
            ``"off"``: as in 0.3.0.
    """

    r: str = "template"
    gate: str = "on"
    resp: str = "on"
    rule_params: dict = field(default_factory=lambda: {"bad_beats_max": 0.5, "rr_irregular_max": 0.1,
                                                       "match_min": 0.95, "fiducial_spread_max_ms": None})
    grid_veto: str = "clear"
    evidence: str = "on"
    prior: str = "setup"
    prior_by_sign: dict = field(default_factory=dict)
    anchored: str = "on"
    saw_rule: str = "two confirmed"
    anchor_check: str = "on"


def find_slips(r_t: np.ndarray, fiducials: dict, quality: dict, *, art_t: np.ndarray | None = None,
               opts: SlipOptions | None = None) -> dict:
    """Find the frame slips of one recording's beats.

    Args:
        r_t: R-peak time of each matched beat (s).
        fiducials: Per beat, the PPG ``foot``, ``up`` (steepest upstroke) and ``peak`` times (s), and optionally
            ``down50`` (half height on the downslope); NaN where absent.
        quality: :func:`beat_quality`'s output for the same beats: ``r_ref``, ``ecg_corr``, ``ppg_corr``, ``rr_irr``,
            ``ppg_clip``, ``ecg_clip``, ``ppg_snip`` and ``ecg_snip``.
        art_t: Arterial upstroke time per beat (s), NaN where absent or not physiological; None without.
        opts: :class:`SlipOptions`.

    Returns:
        A dict: ``sawtooth`` (bool), ``direction`` (``"up"`` or ``"down"``), ``sign``, ``fit`` (tooth ``L`` in s,
        ``rate_ppm``, its source), ``slips`` (accepted, each with ``t``, ``kind``, ``sign`` and, where placed, a 95%
        bracket ``lo``-``hi``), ``predicted`` (every predicted slip with its outcome, shape measures, evidence and the
        fallback probability ``p_slip`` with ``q05``-``q95``), ``uncertain``, ``off_grid``, ``transient``,
        ``anchored``, ``drift`` (where no sawtooth is fitted: ppm, sd, not corrected), ``gating``, ``good_share``.
    """
    F = {j: np.asarray(fiducials[j], dtype=np.float64) if j in fiducials else np.full(len(r_t), np.nan) for j in FIDS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _find_slips(np.asarray(r_t, dtype=np.float64), F, quality, "", opts or SlipOptions(),
                           None if art_t is None else np.asarray(art_t, dtype=np.float64))


_CONFIRMED = ("slip", "off grid", "other way", "isolated")


def _tier_events(R: dict) -> list[dict]:
    """The slip events of a :func:`find_slips` result in time order.

    Each with t, sign, kind (confirmed, step, uncertain) and, for uncertain ones, p and the interval.
    """
    ev = []
    for s in R.get("slips", []):
        ev.append({"t": s["t"], "sign": s.get("sign", R.get("sign", 1)),
                   "kind": "confirmed" if s["kind"] in _CONFIRMED else "step", "what": s["kind"]})
    for u in R.get("uncertain", []):
        if any(abs(e["t"] - u["q50"]) < 30 for e in ev):
            continue
        ev.append({"t": u["q50"], "sign": R.get("sign", 1), "kind": "uncertain", "p": u["p_slip"], "q05": u["q05"],
                   "q95": u["q95"], "what": u["outcome"]})
    return sorted(ev, key=lambda e: e["t"])


def tooth_tiers(R: dict) -> list[dict]:
    """Confidence tiers of the teeth of a :func:`find_slips` result.

    A tooth runs from one slip to the next (the first and last from the ends of the stretch). A: both edges confirmed
    slips. B: both edges confirmed or steps the shape cannot judge, at least one such step. C: an edge is a slip placed
    only within an interval (probability 0.5 or more; corrected as a ramp across it, weighted by the probability). U:
    unresolved (a slip with probability under 0.5 inside, a step against the sawtooth at an edge, or longer than 1.5
    teeth, ends included); not corrected. I: not corrected, reported: slips in a recording without a sawtooth (with
    its drift), or a sawtooth whose tooth rests on two confirmed slips (``fit["creep"] == "reported"``, with the tooth).
    N: no slip. See :func:`delay_at` for what is corrected.

    Args:
        R: A :func:`find_slips` result.

    Returns:
        One dict per tooth: ``t0``, ``t1``, ``tier``, ``reason`` and ``edges``.
    """
    g0, g1 = R["start"], R["end"]
    saw = bool(R.get("sawtooth"))
    reported = saw and (R.get("fit") or {}).get("creep") == "reported"
    ev = _tier_events(R)
    L = (R.get("fit") or {}).get("L")
    sign = R.get("sign", 1)
    if not ev:
        return [{"t0": g0, "t1": g1, "tier": "N", "edges": "", "reason": "no slip"}]
    real = [e for e in ev if e["kind"] != "uncertain" or e["p"] >= 0.5]
    weak = [e for e in ev if e["kind"] == "uncertain" and e["p"] < 0.5]
    edges = [{"t": g0, "kind": "end"}] + real + [{"t": g1, "kind": "end"}]
    out = []
    for a_, b_ in pairwise(edges):
        kinds = [e["kind"] for e in (a_, b_) if e["kind"] != "end"]
        if reported:
            tier, reason = "I", (f"two confirmed slips: tooth {L / 60:.1f} min ({FRAME / L * 1e6:.1f} ppm) reported; not corrected "
                                 "(steps without their creep would make a ramp)")
        elif not saw:
            dr = R.get("drift") or {}
            tier, reason = "I", ("isolated slips, no sawtooth; not corrected" if dr.get("ppm") is None else
                                 f"isolated slips, no sawtooth; not corrected; drift {dr['ppm']:+.1f} ppm (sd {dr['sd_ppm']:.1f})")
        elif "uncertain" in kinds:
            tier, reason = "C", "edge uncertain (p {:.2f})".format(min(e.get("p", 1) for e in (a_, b_) if e["kind"] == "uncertain"))
        elif "step" in kinds:
            tier, reason = "B", "edge a step the shape cannot judge"
        else:
            tier, reason = "A", "edges confirmed" if len(kinds) == 2 else "edge confirmed (end of stretch)"
        inside_weak = [e for e in weak if a_["t"] < e["t"] < b_["t"]]
        other = [e for e in (a_, b_) if e["kind"] != "end" and e.get("sign", sign) != sign]
        if reported:
            pass
        elif saw and inside_weak:
            tier, reason = "U", "a slip with p {:.2f} inside".format(max(e["p"] for e in inside_weak))
        elif saw and other:
            tier, reason = "U", "edge a step the other way"
        elif saw and L and b_["t"] - a_["t"] > 1.5 * L:
            tier, reason = "U", ("{:.1f} teeth long" if len(kinds) == 2 else "{:.1f} teeth without a slip at the end").format(
                (b_["t"] - a_["t"]) / L)
        out.append({"t0": a_["t"], "t1": b_["t"], "tier": tier, "reason": reason,
                    "edges": "-".join(e.get("what", "end") for e in (a_, b_))})
    return out


LONG_TOOTH = 1.5  # teeth longer than this many fitted teeth are unresolved (tier U)


def _delay_unmasked(t: np.ndarray, R: dict, frame_s: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    D = np.zeros(t.size)
    fit = R.get("fit") or {}
    L = fit.get("L")
    if not R.get("sawtooth") or not L or fit.get("creep") == "reported":
        return D
    sign = R.get("sign", 1)
    real = [e for e in _tier_events(R) if e["kind"] != "uncertain" or e["p"] >= 0.5]
    if not real:
        return D
    N = np.zeros(t.size)
    for e in real:
        if e["kind"] == "uncertain":
            N += e["p"] * np.clip((t - e["q05"]) / max(e["q95"] - e["q05"], 1e-6), 0, 1)
        else:
            N += (t >= e["t"]) * (e["sign"] * sign)
    # creep, in teeth: one tooth before the first slip and after the last at the fitted tooth; between two slips the
    # sawtooth's way exactly one frame over their spacing; next to other steps at the fitted tooth
    segs = [(real[0]["t"] - L, real[0]["t"], 1 / L)]
    for a_, b_ in pairwise(real):
        g = b_["t"] - a_["t"]
        close = g > 0 and b_["kind"] != "uncertain" and b_["sign"] == sign and (a_["kind"] == "uncertain" or a_["sign"] == sign)
        segs.append((a_["t"], b_["t"], 1 / g if close else 1 / L))
    segs.append((real[-1]["t"], real[-1]["t"] + L, 1 / L))
    run = np.zeros(t.size)
    for s0, s1, rate in segs:
        run += rate * (np.clip(t, s0, s1) - s0)
    return sign * frame_s * (N - run)


def delay_at(t: np.ndarray, R: dict, frame_s: float = FRAME) -> np.ndarray:
    """The sawtooth delay of a :func:`find_slips` result at times ``t`` (s; relative: its level is arbitrary).

    Corrected only where a sawtooth is fitted and its creep is corrected (with two confirmed slips the tooth and creep
    are reported, and without a sawtooth the drift: correcting the steps alone would turn a sawtooth that stays within
    a frame into a ramp). There, each confirmed slip and each one-frame step the shape could not judge steps the delay
    by a frame its own way; a slip placed only within an interval (probability 0.5 or more) steps it by that
    probability, as a ramp across its 5-95% interval; between two slips the sawtooth's way the delay creeps back by
    exactly a frame, at a rate set by their spacing (so each tooth closes and a tooth-length error cannot build up);
    before the first slip and after the last, at frame / the fitted tooth, for one tooth. Teeth of tier U
    (:func:`tooth_tiers`) are not corrected: the delay is held at its median there.

    Args:
        t: Times (s).
        R: A :func:`find_slips` result.
        frame_s: The frame (s).

    Returns:
        The delay (s) at ``t``.
    """
    t = np.asarray(t, dtype=np.float64)
    D = _delay_unmasked(t, R, frame_s)
    U = [x for x in tooth_tiers(R) if x["tier"] == "U"] if R.get("sawtooth") and (R.get("fit") or {}).get("creep") != "reported" else []
    if U:
        med = float(np.median(_delay_unmasked(np.arange(R["start"], R["end"] + 1.0, 1.0), R, frame_s)))
        for x in U:
            D = np.where((t >= x["t0"]) & (t < x["t1"]), med, D)
    return D


def delay_model(R: dict, frame_s: float = FRAME) -> dict:
    """The sawtooth delay of a :func:`find_slips` result (:func:`delay_at`) as piecewise-linear segments, with tiers.

    Args:
        R: A :func:`find_slips` result.
        frame_s: The frame (s).

    Returns:
        ``edges`` (the stretch's start, every place the delay steps or changes slope, its end; s), and per segment
        (between consecutive edges): ``steps`` (the step at the segment's start, in frames, signed; fractional where a
        tier U tooth is held at the median), ``rates`` (s/s; positive: the delay falls), ``tiers`` and ``reasons``.
    """
    g0, g1 = float(R["start"]), float(R["end"])
    fit = R.get("fit") or {}
    L = fit.get("L")
    knots = {g0, g1}
    ev = _tier_events(R)
    for e in ev:
        knots.add(float(e["t"]))
        if e["kind"] == "uncertain":
            knots.update((float(e["q05"]), float(e["q95"])))
    if L and ev:
        knots.update((float(ev[0]["t"] - L), float(ev[-1]["t"] + L)))
    tt = tooth_tiers(R)
    for x in tt:
        knots.update((float(x["t0"]), float(x["t1"])))
    edges = sorted(k for k in knots if g0 <= k <= g1)
    E = np.asarray(edges)
    eps = 1e-4
    right = delay_at(E, R, frame_s)                      # value at (just after) each edge
    left = delay_at(E[1:] - eps, R, frame_s)             # value just before the next edge
    rates = [-(lv - rv) / max(b - a - eps, 1e-9) for a, b, rv, lv in zip(E[:-1], E[1:], right[:-1], left)]
    prev = np.r_[right[0], left]                          # value just before each edge (the first: no step)
    steps = [float((right[k] - prev[k]) / frame_s) if k else 0.0 for k in range(len(edges) - 1)]
    tiers, reasons = [], []
    for a_, b_ in pairwise(edges):
        mid = 0.5 * (a_ + b_)
        x = next((y for y in tt if y["t0"] <= mid <= y["t1"]), tt[-1])
        tiers.append(x["tier"])
        reasons.append(x["reason"])
    return {"edges": [float(x) for x in edges], "steps": steps, "rates": [float(x) for x in rates], "tiers": tiers,
            "reasons": reasons}
