# SPDX-License-Identifier: MIT
"""Effective temporal resolution of fiducial points (Ruffolo et al. 2025, section 3.3, generalised).

A fiducial point (R-peak, pulse onset, ...) is reported at a whole sample, so its nominal resolution is
one sample period. Its *effective* resolution can be coarser: a signal produced upstream at a lower rate
and re-sampled, a buffer that delivers values in blocks, or a detector that only lands on certain samples
all make fiducials fall on a grid of every k-th sample. Then every inter-beat interval (IBI) is a whole
multiple of k samples, and so is every difference between consecutive IBIs.

Ruffolo et al. histogram the differences between consecutive IBIs and report the share that are even
(>95% for their 125 Hz PPG: an effective resolution of 2 samples = 16 ms). Here the same idea is tested
for every grid k = 2..max_k, on two quantities:

``"ibi"``       the intervals themselves. They spread over hundreds of samples, so with no grid every
                remainder modulo k is equally likely (expected share 1/k); this is the better-conditioned test.
``"ibi_diff"``  differences of consecutive intervals, as in the paper. They cluster around 0, which is
                divisible by every k, so the expected share is taken from the histogram smoothed over a
                window of k samples rather than 1/k, and k is only *testable* when the spread is wide enough.

For each k: ``share`` (observed fraction divisible by k), ``expected`` (no-grid fraction), ``strength``
= (share - expected) / (1 - expected), 0 for no grid and 1 for a perfect one, and a binomial z-score. The
effective resolution is the largest k whose strength and z pass the thresholds (1 = the sample period).

``period_scan`` also checks non-integer periods (an upstream rate that does not divide the sampling
rate, mains hum, a block-wise buffer): the mean resultant length r of exp(2*pi*i*interval/P) for P from
2 samples up to ``max_period``. A perfect grid gives r = 1; *partial* locking (only some fiducials pulled
onto a comb, e.g. a raw-sample detector catching 60 Hz hum on a flat pulse top) gives 0 < r < 1 and
barely moves the integer shares. For intervals, r is roughly the square of the fraction of fiducials
locked (both ends of an interval must be locked), so r = 0.09 already means about 30% locked. ``locking`` lists the significant peaks (Rayleigh N*r^2, r above a
floor), only for periods up to half the interquartile range of the values, where the shape of the
distribution itself cannot produce them.

Only numpy. Works on integer sample indices from any detector, for any signal and sampling rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["ResolutionResult", "fiducial_resolution", "indices_from_times", "intervals", "period_scan"]


def indices_from_times(times_s: np.ndarray, fs: float, t0: float | None = None) -> np.ndarray:
    """Fiducial times (s) back to whole-sample indices at ``fs`` (rounded; times must sit on the grid)."""
    t = np.asarray(times_s, dtype=np.float64)
    base = t[0] if t0 is None else t0
    return np.round((t - base) * fs).astype(np.int64)


def intervals(idx: np.ndarray, segments: np.ndarray | None = None, ibi_range: tuple[int, int] | None = None) -> np.ndarray:
    """IBIs (samples) between consecutive fiducials, never across a segment boundary.

    ``segments``: an id per fiducial (e.g. the continuous stretch it lies in); intervals are only taken
    between fiducials with the same id. ``ibi_range``: (lo, hi) in samples; intervals outside it are
    dropped as missed or spurious beats, and are reported as NaN so that differences skip them too.
    """
    idx = np.asarray(idx, dtype=np.int64)
    ibi = np.diff(idx).astype(np.float64)
    if segments is not None:
        seg = np.asarray(segments)
        ibi[seg[1:] != seg[:-1]] = np.nan
    if ibi_range is not None:
        lo, hi = ibi_range
        ibi[(ibi < lo) | (ibi > hi)] = np.nan
    return ibi


def _box_smooth(h: np.ndarray, k: int) -> np.ndarray:
    """Moving average over k bins (for even k, k+1 bins with half weight at both ends): the no-grid
    expectation for each bin when the true distribution is smooth on the scale of k samples."""
    if k % 2:
        w = np.ones(k) / k
    else:
        w = np.r_[0.5, np.ones(k - 1), 0.5] / k
    return np.convolve(h, w, mode="same")


def period_scan(values: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Mean resultant length of exp(2*pi*i*v/P) for each period P (samples). About 1/sqrt(N) with no
    grid; 1 when every value is a whole multiple of P."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.full(len(periods), np.nan)
    ph = 2 * np.pi * v[None, :] / np.asarray(periods, dtype=np.float64)[:, None]
    return np.hypot(np.cos(ph).mean(axis=1), np.sin(ph).mean(axis=1))


@dataclass
class ResolutionResult:
    quantity: str
    n: int
    sample_period_s: float | None
    share: dict[int, float] = field(default_factory=dict)
    expected: dict[int, float] = field(default_factory=dict)
    strength: dict[int, float] = field(default_factory=dict)
    z: dict[int, float] = field(default_factory=dict)
    testable: dict[int, bool] = field(default_factory=dict)
    periods: list[float] = field(default_factory=list)
    resultant: list[float] = field(default_factory=list)
    effective_samples: int = 1
    locking: list[dict] = field(default_factory=list)
    locking_reference: bool = False
    histogram: dict[str, list[int]] = field(default_factory=dict)

    @property
    def effective_s(self) -> float | None:
        return None if self.sample_period_s is None else self.effective_samples * self.sample_period_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["effective_s"] = self.effective_s
        return d


def fiducial_resolution(
    idx: np.ndarray,
    *,
    fs: float | None = None,
    segments: np.ndarray | None = None,
    ibi_range: tuple[int, int] | None = None,
    quantity: str = "ibi",
    max_k: int = 16,
    min_strength: float = 0.5,
    min_z: float = 4.0,
    diff_limit: int | None = None,
    min_n: int = 50,
    max_period: float | None = None,
    reference_idx: np.ndarray | None = None,
    reference_segments: np.ndarray | None = None,
) -> ResolutionResult:
    """Test whether fiducials at whole-sample indices ``idx`` sit on a grid coarser than one sample.

    ``idx``        sorted whole-sample indices of one fiducial type in one signal.
    ``fs``         sampling rate, only to express the result in seconds.
    ``segments``   optional segment id per fiducial (see :func:`intervals`).
    ``ibi_range``  (lo, hi) plausible IBI in samples, e.g. 60/250 bpm .. 60/30 bpm at ``fs``.
    ``quantity``   "ibi" (default) or "ibi_diff" (the paper's histogram).
    ``diff_limit`` for "ibi_diff": keep |difference| <= this many samples (default 4 * max_k).
    ``reference_idx`` fiducials of the same beats from a reference detector or signal at the same rate
                   (e.g. ECG R-peaks when testing PPG onsets): ``locking`` is then tested against it
                   (see :func:`locking`) instead of the stricter stand-alone rule.

    Returns a :class:`ResolutionResult`; ``effective_samples`` is the largest k (2..max_k) that is
    testable, has strength >= ``min_strength`` and z >= ``min_z``, and whose divisors that are tested
    also pass (a grid of 4 implies a grid of 2), else 1.
    """
    ibi = intervals(idx, segments, ibi_range)
    if quantity == "ibi":
        v = ibi[np.isfinite(ibi)].astype(np.int64)
    elif quantity == "ibi_diff":
        dd = np.diff(ibi)
        v = dd[np.isfinite(dd)].astype(np.int64)
        lim = 4 * max_k if diff_limit is None else diff_limit
        v = v[np.abs(v) <= lim]
    else:
        raise ValueError(f"quantity must be 'ibi' or 'ibi_diff', not {quantity!r}")
    if v.size < min_n:
        return ResolutionResult(quantity=quantity, n=int(v.size), sample_period_s=None if fs is None else 1.0 / fs)
    lo = int(v.min()) - max_k
    h = np.bincount(v - lo, minlength=int(v.max()) - lo + max_k + 1)
    res = _from_histogram(h, lo, quantity, fs, max_k, min_strength, min_z, max_period)
    if reference_idx is not None:
        ref = fiducial_resolution(reference_idx, fs=fs, segments=reference_segments, ibi_range=ibi_range,
                                  quantity=quantity, max_k=max_k, diff_limit=diff_limit, min_n=min_n,
                                  max_period=max_period)
        res.locking = locking(res, ref)
        res.locking_reference = True
    return res


def _from_histogram(h: np.ndarray, lo: int, quantity: str, fs: float | None, max_k: int,
                    min_strength: float, min_z: float, max_period: float | None = None,
                    min_rayleigh: float = 25.0, min_resultant: float = 0.03) -> ResolutionResult:
    """The test on a histogram of whole-sample values; bin i holds the value lo + i."""
    h = np.asarray(h, dtype=np.float64)
    d = np.arange(h.size) + lo
    n = h.sum()
    res = ResolutionResult(quantity=quantity, n=int(n), sample_period_s=None if fs is None else 1.0 / fs)
    res.histogram = {"start": [int(lo)], "counts": h.astype(int).tolist()}
    # spread of the values (interquartile range), for the testability of each k
    c = np.cumsum(h) / n
    spread = d[np.searchsorted(c, 0.75)] - d[np.searchsorted(c, 0.25)]
    for k in range(2, max_k + 1):
        on = d % k == 0
        share = float(h[on].sum() / n)
        # no-grid expectation: the histogram smoothed over k samples (about 1/k for broad IBIs)
        exp = float(_box_smooth(h, k)[on].sum() / n)
        exp = min(max(exp, 1e-9), 1 - 1e-9)
        res.share[k] = round(share, 4)
        res.expected[k] = round(exp, 4)
        res.strength[k] = round((share - exp) / (1 - exp), 4)
        res.z[k] = round(float((share - exp) * n / np.sqrt(n * exp * (1 - exp))), 2)
        # the smoothed null is only sound when the distribution is wide compared with k
        res.testable[k] = bool(spread >= 2 * k)
    periods = np.arange(2.0, (2 * max_k if max_period is None else max_period) + 1e-9, 0.05)
    ph = 2 * np.pi * d[None, :] / periods[:, None]
    r = np.hypot((np.cos(ph) * h).sum(axis=1), (np.sin(ph) * h).sum(axis=1)) / n
    res.periods = [round(float(p), 3) for p in periods]
    res.resultant = [round(float(x), 4) for x in r]
    # without a reference the shape of the distribution itself gives r > 0 at long periods, so only
    # periods up to a quarter of the IQR are scanned, and a peak must stand 3x above the median there
    ok = periods <= spread / 4
    floor = 3 * float(np.median(r[ok])) if ok.any() else np.inf
    peaks = [i for i in range(1, r.size - 1) if ok[i] and r[i] >= r[i - 1] and r[i] >= r[i + 1]
             and n * r[i] ** 2 >= min_rayleigh and r[i] >= max(min_resultant, floor)]
    res.locking = _fundamentals([_peak(periods[i], r[i], None, n, fs) for i in sorted(peaks, key=lambda i: -r[i])[:8]])[:5]
    passed = {k for k in res.share if res.testable[k] and res.strength[k] >= min_strength and res.z[k] >= min_z}
    best = 1
    for k in sorted(passed):
        if all(j in passed for j in range(2, k) if k % j == 0 and res.testable[j]):
            best = k
    res.effective_samples = best
    return res


def _peak(period: float, r: float, r_ref: float | None, n: float, fs: float | None) -> dict:
    pk = {"period_samples": round(float(period), 3), "resultant": round(float(r), 4),
          "rayleigh": round(float(n * r**2), 1)}
    if r_ref is not None:
        pk["reference_resultant"] = round(float(r_ref), 4)
    if fs is not None:
        pk["period_s"] = round(float(period) / fs, 6)
        pk["freq_hz"] = round(fs / float(period), 3)
    return pk


def _fundamentals(peaks: list[dict], tol: float = 0.03) -> list[dict]:
    """Drop peaks at P/m (m = 2..6) of a longer-period peak at least ~as strong: a comb of period P
    is also a comb of P/2, P/3, ...; report the fundamental."""
    keep = []
    for a in peaks:
        pa = a["period_samples"]
        dominated = any(b is not a and b["resultant"] >= 0.8 * a["resultant"]
                        and any(abs(b["period_samples"] / m - pa) <= tol * pa for m in range(2, 7))
                        for b in peaks)
        if not dominated:
            keep.append(a)
    return keep


def locking(test: ResolutionResult | dict, reference: ResolutionResult | dict, *, factor: float = 2.0,
            z: float = 5.0, min_resultant: float = 0.03) -> list[dict]:
    """Periods at which the test fiducials lock but the reference fiducials do not.

    The reference should time the same events with a detector of known resolution (ECG R-peaks for PPG
    onsets, or a filtered-signal detector for a raw-sample one): its intervals have the same overall
    shape, so its resultant at each period is the background the shape alone produces. A peak must be a
    local maximum of the excess, at least ``factor`` times the reference, ``z`` / sqrt(N) above it and
    at least ``min_resultant``. Up to 5 peaks, strongest first.
    """
    t = test if isinstance(test, dict) else test.to_dict()
    ref = reference if isinstance(reference, dict) else reference.to_dict()
    if not t.get("periods") or not ref.get("periods"):
        return []
    P = np.asarray(t["periods"], dtype=np.float64)
    rt = np.asarray(t["resultant"], dtype=np.float64)
    rr = np.interp(P, np.asarray(ref["periods"], dtype=np.float64), np.asarray(ref["resultant"], dtype=np.float64))
    n = float(t["n"])
    ex = rt - rr
    sp = t.get("sample_period_s")
    fs = None if sp is None else 1.0 / sp
    pk = [i for i in range(1, P.size - 1) if ex[i] >= ex[i - 1] and ex[i] >= ex[i + 1]
          and rt[i] >= factor * rr[i] and ex[i] >= z / np.sqrt(n) and rt[i] >= min_resultant]
    return _fundamentals([_peak(P[i], rt[i], rr[i], n, fs) for i in sorted(pk, key=lambda i: -ex[i])[:8]])[:5]


def pool(results: list[ResolutionResult | dict], *, max_k: int | None = None, min_strength: float = 0.5,
         min_z: float = 4.0, max_period: float | None = None) -> ResolutionResult:
    """Combine results from several segments or recordings (same quantity and sampling rate) by adding
    their histograms and testing the sum. Accepts results or their ``to_dict()`` forms."""
    rs = [r if isinstance(r, dict) else r.to_dict() for r in results]
    rs = [r for r in rs if r.get("histogram")]
    if not rs:
        raise ValueError("no result with a histogram to pool")
    q = {r["quantity"] for r in rs}
    if len(q) != 1:
        raise ValueError(f"cannot pool different quantities: {q}")
    lo = min(r["histogram"]["start"][0] for r in rs)
    hi = max(r["histogram"]["start"][0] + len(r["histogram"]["counts"]) for r in rs)
    h = np.zeros(hi - lo)
    for r in rs:
        s = r["histogram"]["start"][0] - lo
        h[s : s + len(r["histogram"]["counts"])] += r["histogram"]["counts"]
    sp = rs[0].get("sample_period_s")
    k = max_k if max_k is not None else max(int(x) for x in rs[0]["share"]) if rs[0]["share"] else 16
    return _from_histogram(h, lo, q.pop(), None if sp is None else 1.0 / sp, k, min_strength, min_z, max_period)
