# SPDX-License-Identifier: MIT
"""ECG-to-PPG timing correction: a frame-slip sawtooth plus one combined offset.

The measured pulse arrival time (PAT, R-peak to PPG fiducial) carries, besides physiology:

* a **constant offset**: the monitor's buffering and filtering of the PPG relative to the ECG. Buffer and
  filter delays cannot be separated from recordings alone, so they are one term. It is not estimated
  physiologically: the corrected PAT is shifted so that its median equals a population mean PAT that the
  user adopts (``adopted_pat_s``) for the fiducial used. Datasets with different hardware then all snap
  to the same adopted mean.
* a **sawtooth**: the PPG is produced in frames by a clock that runs at a slightly different rate from
  the ECG path, so the delay creeps by ``rate`` (seconds per second) and is reset by one whole frame
  (a *frame slip*) whenever the accumulated difference reaches a frame. In the VitalDB surgical dataset,
  for example, the frame is 33.3 ms (30 Hz) and the rate 20-55 ppm, fixed per recording setup and stable
  within a recording.

Model of the non-physiological delay of the PPG relative to the ECG at time t:

    D(t) = offset + frame * (slips before t) - integral of rate from t_start to t       (sawtooth part zero-median)

``fit_ecg_ppg_delay`` estimates it from beat-matched fiducial times; :meth:`EcgPpgDelay.shift` gives D(t)
and :meth:`EcgPpgDelay.correct` subtracts it from PPG times. :meth:`EcgPpgDelay.timing_plan` cuts a PPG
sample stream at the slips into segments with their own start time and sampling rate, for storage.

Fitting (see ``_fill_rates``): detected slips are kept if they are one frame and happen between two
beats (physiological PAT changes build up more slowly); bursts are thinned; missed slips are inserted
where a gap has room for whole teeth and the data shows a step. Each tooth between two slips gets
rate = frame / its length, so its creep cancels exactly the frame its slip adds: D(t) returns to the same
level at every slip, and a wrong or missed slip distorts only its own tooth. The partial teeth at the
ends get the median rate of the full teeth (else ``rate_prior_ppm``, the known rate of the recording
setup), capped to one frame of creep. With slips but no rate the sawtooth is left alone. Each tooth
records where its rate came from.

Only numpy. Times in seconds on the ECG's clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np

ORDER = "ABCNU"  # tiers, best first

__all__ = ["EcgPpgDelay", "TierRules", "detect_slips", "fit_ecg_ppg_delay", "match_beats", "sawtooth_evidence",
           "setup_offsets"]


# ----------------------------------------------------------------------------------------------- beats


def match_beats(r_t: np.ndarray, p_t: np.ndarray, *, delay_range: tuple[float, float] = (0.1, 1.6),
                bin_s: float = 0.004, tol_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair each R peak with the PPG fiducial nearest to R plus the typical delay.

    The typical delay is the mode of all PPG-minus-R differences within ``delay_range``, so delays longer
    than a beat are handled. When the delay is close to a beat, use :func:`~physiosync.pair_beats` instead.

    Args:
        r_t: R-peak times (s), sorted.
        p_t: PPG fiducial times (s), sorted, on the same clock.
        delay_range: ``(low, high)`` possible delays (s).
        bin_s: Histogram bin for the mode (s).
        tol_frac: Pairs further than this fraction of the median R-R interval from the typical delay are
            dropped.

    Returns:
        ``(i, j, delay)``: indices into ``r_t``, indices into ``p_t`` and the typical delay (s). Empty arrays
        and NaN when no pair is found.
    """
    r_t = np.asarray(r_t, dtype=np.float64)
    p_t = np.asarray(p_t, dtype=np.float64)
    if r_t.size < 3 or p_t.size < 3:
        return np.empty(0, int), np.empty(0, int), float("nan")
    lo, hi = delay_range
    j0 = np.searchsorted(p_t, r_t + lo)
    n_ahead = int(np.ceil((hi - lo) / max(np.median(np.diff(r_t)), 0.2))) + 2
    d = np.concatenate([p_t[np.clip(j0 + m, 0, p_t.size - 1)] - r_t for m in range(n_ahead)])
    d = d[(d >= lo) & (d < hi)]
    if not d.size:
        return np.empty(0, int), np.empty(0, int), float("nan")
    h, e = np.histogram(d, np.arange(lo, hi + bin_s, bin_s))
    delay = float(e[np.argmax(h)] + bin_s / 2)
    j = np.clip(np.searchsorted(p_t, r_t + delay), 1, p_t.size - 1)
    j = np.where(np.abs(p_t[j - 1] - r_t - delay) < np.abs(p_t[j] - r_t - delay), j - 1, j)
    ok = np.abs(p_t[j] - r_t - delay) < tol_frac * float(np.median(np.diff(r_t)))
    i = np.flatnonzero(ok)
    # one PPG fiducial for one R-peak only
    _, first = np.unique(j[i], return_index=True)
    i = i[first]
    return i, j[i], delay


# ----------------------------------------------------------------------------------------------- slips


def _running_median(x: np.ndarray, k: int) -> np.ndarray:
    h = k // 2
    xp = np.pad(x, h, mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(w, axis=1)


def detect_slips(t: np.ndarray, pat: np.ndarray, *, frame_s: float | None = None, threshold_s: float | None = None,
                 frame_range_s: tuple[float, float] = (0.010, 0.200),
                 smooth: int = 15, consec: int = 5, step_tol: float = 0.3) -> dict:
    """Find frame slips (upward resets of about one frame) in a beat-by-beat PAT series.

    Tracks the running median of ``smooth`` beats with a line through the current tooth. A reset is declared
    when ``consec`` consecutive smoothed beats exceed the line by ``threshold_s``; its size is the median of
    ``smooth`` beats after minus before. Resets within ``step_tol`` of the frame are slips.

    Args:
        t: Beat times (s).
        pat: PAT at each beat (s).
        frame_s: The frame (s). Default: the most common reset size within ``frame_range_s``.
        threshold_s: Reset threshold (s). Default: half the frame when the frame is given, else 16 ms.
        frame_range_s: ``(low, high)`` range searched for the frame (s).
        smooth: Beats in the running median.
        consec: Consecutive beats above the line needed for a reset.
        step_tol: Largest relative difference from the frame for a reset to count as a slip.

    Returns:
        ``t`` (slip times), ``step_s`` (their sizes), ``frame_s`` (the frame used) and ``other_t`` (resets
        that are not slips).
    """
    t = np.asarray(t, dtype=np.float64)
    pat = np.asarray(pat, dtype=np.float64)
    if threshold_s is None:
        threshold_s = 0.5 * frame_s if frame_s else 0.016
    out = {"t": np.empty(0), "step_s": np.empty(0), "frame_s": frame_s, "other_t": np.empty(0)}
    if pat.size < 4 * smooth:
        return out
    ps = _running_median(pat, smooth)
    starts, steps = [], []
    i0, i, above = 0, 1, 0
    while i < ps.size:
        if i - i0 >= 2 * smooth:
            k = np.polyfit(t[i0:i], ps[i0:i], 1)
            pred = np.polyval(k, t[i])
        else:
            pred = np.median(ps[i0:i])
        if ps[i] - pred > threshold_s:
            above += 1
            if above >= consec:
                j = i - consec + 1
                if j >= smooth and j + smooth <= pat.size:
                    starts.append(j)
                    steps.append(float(np.median(pat[j : j + smooth]) - np.median(pat[j - smooth : j])))
                i0, above = j, 0
                i = j + 1
                continue
        else:
            above = 0
        i += 1
    starts, steps = np.array(starts, dtype=int), np.array(steps)
    if frame_s is None:
        # the frame is the most common reset size: densest 20% band of the resets in the plausible range
        cand = np.sort(steps[(steps > frame_range_s[0]) & (steps < frame_range_s[1])])
        if cand.size >= 3:
            best = max(range(cand.size), key=lambda i: np.sum(np.abs(cand / cand[i] - 1) < 0.2))
            frame_s = float(np.median(cand[np.abs(cand / cand[best] - 1) < 0.2]))
        else:
            frame_s = float(np.median(cand)) if cand.size else None
    if frame_s is None or not starts.size:
        out["other_t"] = t[starts] if starts.size else np.empty(0)
        return out
    slip = np.abs(steps - frame_s) <= step_tol * frame_s
    # a slip is at the midpoint between the last beat before and the first beat after it
    st = 0.5 * (t[starts - 1] + t[starts])
    return {"t": st[slip], "step_s": steps[slip], "frame_s": float(frame_s), "other_t": st[~slip]}


# ----------------------------------------------------------------------------------------------- model


@dataclass
class EcgPpgDelay:
    """The fitted non-physiological delay of the PPG relative to the ECG.

    The delay ``D(t)`` is a constant offset (hardware and filter delay) plus a sawtooth: the PPG creeps
    earlier at a tooth's rate and jumps back one frame at each slip. Create it with
    :func:`fit_ecg_ppg_delay`; apply it with :meth:`correct`.

    Attributes:
        adopted_pat_s: The population mean PAT the corrected data should have (s).
        offset_s: Combined hardware and filter offset, the mean of ``D`` (s).
        frame_s: The frame (s), or ``None`` without a sawtooth.
        t_start: Start of the analysed stretch (ECG time, s).
        t_end: End of the analysed stretch (s).
        slips_t: Slip times (s).
        inserted_t: Slips inferred where one was missed (s).
        tooth_edges: ``t_start``, the slips, ``t_end`` (s).
        tooth_rate: Creep rate of each tooth (s/s; positive means the PAT shrinks).
        tooth_source: How each tooth's rate was found.
        sawtooth: Whether a sawtooth is modelled (``False``: offset only).
        tooth_tier: Confidence tier of each tooth: A, B, C (corrected), N (none seen) or U (unresolved).
        tooth_reason: Why each tooth has its tier.
        apply_min_tier: Teeth below this tier keep the offset only.
        qc: Fit diagnostics, e.g. ``raw_level_s`` (median PAT with the sawtooth removed, no offset),
            ``offset_basis`` and ``recording_rate_ppm``.
    """

    adopted_pat_s: float
    offset_s: float                      # combined hardware + filter offset (mean of D)
    frame_s: float | None
    t_start: float
    t_end: float
    slips_t: list[float] = field(default_factory=list)
    inserted_t: list[float] = field(default_factory=list)   # slips inferred where one was missed
    tooth_edges: list[float] = field(default_factory=list)  # t_start, slips..., t_end
    tooth_rate: list[float] = field(default_factory=list)   # s/s, positive = PAT shrinking
    tooth_source: list[str] = field(default_factory=list)
    sawtooth: bool = True
    tooth_tier: list[str] = field(default_factory=list)     # A, B, C (corrected), N (none seen), U (unresolved)
    tooth_reason: list[str] = field(default_factory=list)
    apply_min_tier: str = "B"                                # teeth below this keep the offset only
    qc: dict = field(default_factory=dict)

    # the sawtooth part, before the offset; zero at t_start
    def _saw(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if not self.sawtooth or self.frame_s is None or not self.tooth_rate:
            return np.zeros_like(t)
        edges = np.asarray(self.tooth_edges)
        rate = np.asarray(self.tooth_rate)
        k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, rate.size - 1)
        # accumulated rate integral up to each tooth start
        acc = np.r_[0.0, np.cumsum(rate * np.diff(edges))]
        saw = self.frame_s * k - (acc[k] + rate[k] * (t - edges[k]))
        if self.tooth_tier and "saw_median_s" in self.qc:
            # teeth below the confidence threshold are not corrected: their sawtooth part is replaced by
            # its median level, so they keep the offset only (and are labelled)
            ok = np.array([self.applied(i) for i in range(rate.size)])
            saw = np.where(ok[k], saw, self.qc["saw_median_s"])
        return saw

    def applied(self, k: int) -> bool:
        """Whether tooth ``k``'s sawtooth correction is applied.

        Args:
            k: Tooth index.

        Returns:
            ``True`` when the tooth's tier is at or above :attr:`apply_min_tier`.
        """
        order = {"A": 0, "B": 1, "C": 2}
        t = self.tooth_tier[k] if k < len(self.tooth_tier) else "U"
        return t in order and order[t] <= order.get(self.apply_min_tier, 1)

    def quality_intervals(self) -> list[dict]:
        """Intervals with the confidence of the PPG timing in them, adjacent equal ones merged.

        Tier A, B or C: sawtooth corrected (high, moderate or low confidence). N: no sawtooth seen (offset only).
        U: possibly affected but not corrected (offset only). Tiers below :attr:`apply_min_tier` are reported as U.

        Returns:
            One dict per interval (ECG time, s): ``start``, ``end``, ``tier``, ``applied``, ``teeth``,
            ``rate_ppm`` and ``reason``.
        """
        edges = self.tooth_edges if self.tooth_edges else [self.t_start, self.t_end]
        tiers = self.tooth_tier or ["U"] * (len(edges) - 1)
        out: list[dict] = []
        for k in range(len(edges) - 1):
            tier = tiers[k]
            applied = tier in ("A", "B", "C") and self.applied(k)
            if tier in ("A", "B", "C") and not applied:
                tier, reason = "U", f"below {self.apply_min_tier}: " + (self.tooth_reason[k] if k < len(self.tooth_reason) else "")
            else:
                reason = self.tooth_reason[k] if k < len(self.tooth_reason) else ""
            rate = self.tooth_rate[k] * 1e6 if k < len(self.tooth_rate) else 0.0
            if out and out[-1]["tier"] == tier and out[-1]["applied"] == applied:
                out[-1]["end"] = float(edges[k + 1])
                out[-1]["teeth"] += 1
            else:
                out.append({"start": float(edges[k]), "end": float(edges[k + 1]), "tier": tier, "applied": applied,
                            "teeth": 1, "rate_ppm": round(rate, 2), "reason": reason})
        return out

    def labels(self, prefix: str = "ppg_timing") -> list[tuple[str, float, float]]:
        """Name-only annotations for every quality interval, ready for a connector to write.

        Args:
            prefix: Label name prefix.

        Returns:
            ``(name, start_s, end_s)`` tuples, e.g. ``("ppg_timing:A", 100.0, 740.5)``.
        """
        return [(f"{prefix}:{q['tier']}", q["start"], q["end"]) for q in self.quality_intervals()]

    def shift(self, t: np.ndarray) -> np.ndarray:
        """The delay ``D(t)``: how much later the PPG appears than it should, at ECG time ``t``.

        Args:
            t: Times (s).

        Returns:
            ``D(t)`` (s).
        """
        return self._saw(t) - self._saw_ref + self.offset_s

    @property
    def _saw_ref(self) -> float:
        return float(self.qc.get("saw_median_s", 0.0))

    def correct(self, ppg_t: np.ndarray) -> np.ndarray:
        """Corrected PPG times.

        The shift is evaluated at the PPG time itself; it changes by at most a few microseconds over the PAT,
        far below a sample.

        Args:
            ppg_t: PPG fiducial or sample times (s).

        Returns:
            ``ppg_t - D(ppg_t)`` (s).
        """
        ppg_t = np.asarray(ppg_t, dtype=np.float64)
        return ppg_t - self.shift(ppg_t)

    def timing_plan(self, start_t: float, n: int, fs: float) -> list[dict]:
        """Corrected timing of a continuous PPG stream, one segment per tooth.

        Within a tooth the creep makes the effective sampling rate differ slightly from the nominal one; at a
        slip the timing steps. Storing one segment per tooth keeps the stream exact.

        Args:
            start_t: Time of the first sample (s).
            n: Number of samples.
            fs: Nominal sampling rate (Hz).

        Returns:
            One dict per segment: ``first`` (index of its first sample), ``count``, ``start_t`` (its corrected
            start time, s), ``fs`` (its effective rate, Hz) and ``rate_ppm``.
        """
        t_nom = start_t + np.arange(n) / fs
        cuts = [0]
        if self.sawtooth and self.frame_s is not None:
            for e in self.tooth_edges[1:-1]:
                k = int(np.searchsorted(t_nom, e))
                if 0 < k < n:
                    cuts.append(k)
        cuts.append(n)
        plan = []
        for a, b in pairwise(cuts):
            if b <= a:
                continue
            t0 = float(t_nom[a])
            k = int(np.clip(np.searchsorted(self.tooth_edges, t0, side="right") - 1, 0, max(len(self.tooth_rate) - 1, 0)))
            r = self.tooth_rate[k] if self.sawtooth and self.tooth_rate else 0.0
            # D falls by r per second, so corrected time advances by (1 + r) per nominal second
            plan.append({"first": a, "count": b - a, "start_t": float(self.correct(np.array([t0]))[0]),
                         "fs": fs / (1.0 + r), "rate_ppm": round(r * 1e6, 3)})
        return plan

    def to_dict(self) -> dict:
        """All fields as a dict."""
        return asdict(self)


def fit_ecg_ppg_delay(r_t: np.ndarray, ppg_t: np.ndarray, *, adopted_pat_s: float, sawtooth: bool = True,
                      frame_s: float | None = None, rate_prior_ppm: float | None = None,
                      matched: bool = False, threshold_s: float | None = None, smooth: int = 15,
                      min_tooth_s: float = 60.0, rate_range_ppm: tuple[float, float] = (5.0, 100.0),
                      apply_min_tier: str = "B", tier_rules: TierRules | None = None,
                      ref_t: np.ndarray | None = None, ref_max_step_s: float = 0.010,
                      predict_slips: bool = True, offset_s: float | None = None) -> EcgPpgDelay:
    """Fit the PPG delay model from R-peak and PPG fiducial times on the same clock.

    Pairs the beats (unless ``matched``), detects slips, measures each tooth's creep rate, sets the offset
    and gives every tooth a confidence tier (:class:`TierRules`).

    Args:
        r_t: R-peak times (s).
        ppg_t: PPG fiducial times (s).
        adopted_pat_s: Required. The population mean PAT for this fiducial that the corrected data should
            have; use one value for every dataset to be combined.
        sawtooth: ``False`` fits the offset only.
        frame_s: The frame (s), if known (e.g. 1/30 for the SNUADC PLETH in VitalDB). Default: estimated from the resets.
        rate_prior_ppm: Creep rate of the recording setup, used for teeth that cannot be measured.
        matched: ``r_t`` and ``ppg_t`` are already beat-matched pairs of equal length. Otherwise they are
            paired with :func:`match_beats`.
        threshold_s: Reset threshold for :func:`detect_slips`.
        smooth: Beats in the running median for :func:`detect_slips`.
        min_tooth_s: Shortest tooth used to measure a rate (s).
        rate_range_ppm: Plausible creep rates; teeth whose length implies a rate outside are not used.
        apply_min_tier: ``"A"``, ``"B"`` or ``"C"``. Teeth of a lower tier keep the offset only and are
            labelled U.
        tier_rules: Thresholds of the tiers. Default: :class:`TierRules` defaults.
        ref_t: Optional arrival times of the same beats in a reference signal without the PPG's timing
            artefacts, e.g. the arterial pressure upstroke (one per R peak, NaN where absent). Slips are then
            also looked for in reference-to-PPG, which leaves out the pre-ejection period and is quieter. A step
            that also appears in ECG-to-reference is physiology and is rejected; one that does not is confirmed.
        ref_max_step_s: Largest ECG-to-reference step still compatible with a PPG slip (s).
        predict_slips: Look one tooth beyond the first and last slips for slips that were due, and add them
            where the data shows the step; stop at the first place without one.
        offset_s: The combined hardware and filter delay, if known from outside the recording (e.g. a setup's
            median from :func:`setup_offsets`). The recording then keeps its own PAT level; without it, the
            recording is snapped so that its corrected median equals ``adopted_pat_s``.

    Returns:
        The fitted :class:`EcgPpgDelay`.
    """
    if adopted_pat_s is None or not np.isfinite(adopted_pat_s):
        raise ValueError("adopted_pat_s must be given: the population mean PAT to shift to")
    r_t = np.asarray(r_t, dtype=np.float64)
    ppg_t = np.asarray(ppg_t, dtype=np.float64)
    ref = None if ref_t is None else np.asarray(ref_t, dtype=np.float64)
    if ref is not None and ref.shape != r_t.shape:
        raise ValueError("ref_t needs one value (or NaN) per R-peak of r_t")
    if matched:
        if r_t.shape != ppg_t.shape:
            raise ValueError("matched=True needs r_t and ppg_t of the same length")
        tr, tp, tref = r_t, ppg_t, ref
    else:
        i, j, _ = match_beats(r_t, ppg_t)
        tr, tp = r_t[i], ppg_t[j]
        tref = None if ref is None else ref[i]
    if tr.size < 10:
        raise ValueError("too few matched beats")
    pat = tp - tr
    m = EcgPpgDelay(adopted_pat_s=float(adopted_pat_s), offset_s=0.0, frame_s=frame_s, t_start=float(tr[0]),
                    t_end=float(tr[-1]), sawtooth=sawtooth, apply_min_tier=apply_min_tier)
    m.tooth_edges = [m.t_start, m.t_end]
    if sawtooth:
        s = detect_slips(tr, pat, frame_s=frame_s, threshold_s=threshold_s, smooth=smooth)
        m.frame_s = s["frame_s"]
        slips = list(np.asarray(s["t"], dtype=np.float64))
        refser = None
        if tref is not None and np.isfinite(tref).sum() > 4 * smooth:
            okr = np.isfinite(tref)
            refser = (tp - tref, tref - tr)  # reference->PPG and ECG->reference, NaN without reference
            s2 = detect_slips(tr[okr], (tp - tref)[okr], frame_s=m.frame_s or frame_s, threshold_s=threshold_s,
                              smooth=smooth)
            if m.frame_s is None:
                m.frame_s = s2["frame_s"]
            ref_only = []
            for x in s2["t"]:
                if all(abs(x - y) > 30.0 for y in slips):
                    slips.append(float(x))
                    ref_only.append(float(x))
            m.qc["reference_only_detections_t"] = ref_only
            slips.sort()
            m.qc["reference_coverage"] = float(okr.mean())
        m.slips_t = [float(x) for x in slips]
        if m.frame_s is not None:
            _fill_rates(m, tr, pat, rate_prior_ppm, min_tooth_s, rate_range_ppm, refser=refser,
                        ref_max_step_s=ref_max_step_s, predict=predict_slips)
        _assign_tiers(m, tr, pat, rate_prior_ppm, tier_rules or TierRules(), rate_range_ppm=rate_range_ppm,
                      alt=None if tref is None else (tp - tref))
    saw = m._saw(tr)  # (before saw_median_s exists, no tooth is masked)
    m.qc["saw_median_s"] = float(np.median(saw))
    saw = m._saw(tr)  # now with the teeth below apply_min_tier masked
    resid = pat - (saw - m.qc["saw_median_s"])
    # the single combined offset: everything above the adopted population mean, taken where the timing
    # is known (the corrected teeth) when there are enough of them; else over all beats
    k_of = np.clip(np.searchsorted(m.tooth_edges, tr, side="right") - 1, 0, max(len(m.tooth_edges) - 2, 0))
    in_applied = np.array([m.applied(int(k)) for k in range(max(len(m.tooth_edges) - 1, 1))])[k_of] \
        if m.sawtooth and m.tooth_tier else np.zeros(tr.size, bool)
    use = in_applied if in_applied.sum() >= max(100, 0.2 * tr.size) else np.ones(tr.size, bool)
    # this recording's own level (sawtooth removed), whatever offset is then used
    m.qc["raw_level_s"] = float(np.median(resid[use]))
    if offset_s is not None:
        m.offset_s = float(offset_s)
        m.qc["offset_basis"] = "caller"
    else:
        m.offset_s = float(np.median(resid[use]) - adopted_pat_s)
        m.qc["offset_basis"] = "corrected teeth" if use is in_applied else "all beats"
    m.qc["rate_prior_ppm"] = rate_prior_ppm
    m.qc.update({
        "beats": int(tr.size), "pat_median_raw_s": float(np.median(pat)),
        "pat_median_corrected_s": float(np.median(pat - m.shift(tr))),
        "n_slips": len(m.slips_t), "n_kept": max(len(m.tooth_edges) - 2 - len(m.inserted_t), 0) if m.frame_s else 0,
        "n_inserted": len(m.inserted_t), "n_rejected": len(m.qc.get("rejected_t", [])),
        "sawtooth_p2p_raw_ms": _p2p(tr, pat), "sawtooth_p2p_corrected_ms": _p2p(tr, pat - m.shift(tr)),
        "tier_time_s": {t: float(sum(q["end"] - q["start"] for q in m.quality_intervals() if q["tier"] == t))
                        for t in ("A", "B", "C", "N", "U")},
    })
    return m


def setup_offsets(levels: dict, setups: dict, *, adopted_pat_s: float, min_recordings: int = 3,
                  plausible_s: tuple[float, float] | None = None) -> dict:
    """One hardware and filter delay per recording setup, from many recordings.

    Each setup's delay is the median level of its recordings minus ``adopted_pat_s``. Setups with fewer than
    ``min_recordings`` recordings, and recordings without a setup, get the median over all recordings.

    Args:
        levels: Recording -> its raw PAT level (``qc["raw_level_s"]`` of a fit: median PAT with the sawtooth
            removed, no offset; s).
        setups: Recording -> setup name.
        adopted_pat_s: The adopted population mean PAT (s).
        min_recordings: Fewest recordings for a setup to get its own delay.
        plausible_s: ``(low, high)`` plausible levels (s); others (e.g. beats paired one beat out) are left out.

    Returns:
        ``setups`` (name -> offset, count, basis), ``all`` (the pooled offset), ``by_recording`` (recording ->
        the offset to use) and ``adopted_pat_s``.
    """
    lv = {k: float(v) for k, v in levels.items() if v is not None and np.isfinite(v)
          and (plausible_s is None or plausible_s[0] <= v <= plausible_s[1])}
    allv = np.array(list(lv.values()))
    pooled = {"offset_s": float(np.median(allv) - adopted_pat_s), "n": int(allv.size), "basis": "all recordings",
              "iqr_s": [float(x) for x in np.quantile(allv, [0.25, 0.75])]} if allv.size else None
    groups: dict[str, list[float]] = {}
    for k, v in lv.items():
        if setups.get(k):
            groups.setdefault(setups[k], []).append(v)
    out = {}
    for name, v in groups.items():
        a = np.array(v)
        if a.size >= min_recordings:
            out[name] = {"offset_s": float(np.median(a) - adopted_pat_s), "n": int(a.size), "basis": f"setup {name}",
                         "iqr_s": [float(x) for x in np.quantile(a, [0.25, 0.75])]}
    by_rec = {k: (out.get(setups.get(k, "")) or pooled) for k in levels}
    return {"setups": out, "all": pooled, "by_recording": by_rec, "adopted_pat_s": adopted_pat_s}


def sawtooth_evidence(t: np.ndarray, y: np.ndarray, rate: float, frame: float, *, smooth: int = 15,
                      phases: int = 50) -> dict:
    """Test whether a frame-slip sawtooth of known rate and frame is present in a delay series.

    ``smooth``-beat medians of ``y``, minus a sawtooth of exactly one frame at the best of ``phases`` phases,
    are fitted by a constant plus a slope. Slope-only creep is not a sawtooth.

    Args:
        t: Times (s).
        y: A delay, e.g. PAT, at those times (s).
        rate: The creep rate (s/s).
        frame: The frame (s).
        smooth: Beats per median.
        phases: Phases tried.

    Returns:
        ``score`` (share of the variance, after constant and slope, that the sawtooth explains: near 1 when
        present, negative when subtracting it makes things worse), ``mirror`` (the same with the sawtooth
        reversed, which no slip produces; a control) and ``teeth`` (slips the rate implies over the series).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = y.size // smooth
    out = {"teeth": float((t[-1] - t[0]) * rate / frame) if t.size > 1 else 0.0, "score": float("nan"),
           "mirror": float("nan")}
    if n < 20 or rate <= 0:
        return out
    tt = t[: smooth * n].reshape(n, smooth).mean(1)
    yy = np.median(y[: smooth * n].reshape(n, smooth), 1)
    X0 = np.c_[np.ones_like(tt), tt - tt[0]]
    r0 = yy - X0 @ np.linalg.lstsq(X0, yy, rcond=None)[0]
    ss0 = float(r0 @ r0)
    if ss0 <= 0:
        return out
    for key, sign in (("score", 1.0), ("mirror", -1.0)):
        best = -np.inf
        for ph in np.linspace(0, 1, phases, endpoint=False):
            u = rate * (tt - tt[0]) / frame + ph
            saw = sign * frame * (np.floor(u) - (u - ph))
            y2 = yy - saw
            res = y2 - X0 @ np.linalg.lstsq(X0, y2, rcond=None)[0]
            best = max(best, 1 - float(res @ res) / ss0)
        out[key] = float(best)
    return out


@dataclass
class TierRules:
    """Thresholds of the confidence tiers of a tooth.

    A tooth is the stretch between two slips, or between a slip and the end of the analysed stretch.

    A (high): a tooth whose rate is corroborated: a full tooth (two detected slips) within ``rec_tol_a``
       of the recording's median tooth rate (at least ``min_teeth_a`` full teeth) or, with fewer teeth,
       within ``setup_tol`` of the setup's known rate; no rejected slip-like step inside; beat-to-beat
       PAT noise at most ``noise_max_ms``. An end tooth qualifies the same way if its rate comes from
       such teeth and it is no longer than ``end_max`` tooth lengths (a longer one would have slipped).
    B (moderate): rate within ``rec_tol_b`` of the recording (or of the setup), noise at most 1.5 x the
       limit; or an A tooth with an inserted slip at one end that the reference showed.
    A tooth bounded by an inserted or predicted slip that only the ECG->PPG series showed is at most
    ``inserted_max_tier`` (checked against arterial pressure, a third of such slips were not real).
    C (low): any other corrected tooth.
    N: no slip over at least ``no_slip_min_s`` (or 1.5 tooth lengths at the setup's rate, if longer):
       no sawtooth evident (offset only). Also an end tooth where slips were due (longer than one tooth)
       when :func:`sawtooth_evidence` finds the sawtooth absent: at least ``absent_min_teeth`` teeth
       expected and a score below ``absent_max_score`` (the sawtooth stopped there).
    U: possibly affected but not corrected (offset only): slips but no usable rate, an end tooth too
       long for the rate, a stretch too short to tell.

    Rate conflict: when the recording's own tooth rate (median of its full teeth) differs from the setup's
    known rate by more than ``prior_conflict_tol``, either the setup is wrong for this recording or its
    slips are; the recording's own rate is then used to judge whether a slip was due, a tooth is at most C
    (B where the reference confirmed the slips at its ends), and nothing is labelled N (the pause test depends on knowing the rate): those teeth
    are U instead. ``None`` turns the check off. With ``absent_needs_own_rate`` an end tooth is labelled N
    only if the recording has at least one full tooth of its own (a setup's rate alone can be wrong for a
    recording).
    """

    rec_tol_a: float = 0.03
    rec_tol_b: float = 0.10
    setup_tol: float = 0.05
    min_teeth_a: int = 2
    noise_max_ms: float = 6.0
    no_slip_min_s: float = 2700.0
    end_max: float = 1.1
    absent_min_teeth: float = 1.5
    absent_max_score: float = -0.1
    prior_conflict_tol: float | None = 0.2
    absent_needs_own_rate: bool = True
    inserted_max_tier: str | None = "C"


def _assign_tiers(m: EcgPpgDelay, tr: np.ndarray, pat: np.ndarray, prior_ppm: float | None, rules: TierRules,
                  alt: np.ndarray | None = None,
                  rate_range_ppm: tuple[float, float] = (5.0, 100.0)) -> None:
    edges = m.tooth_edges or [m.t_start, m.t_end]
    n = len(edges) - 1
    src = m.tooth_source or ["none"] * n
    f = m.frame_s or 1 / 30
    if n == 1:
        dur = edges[1] - edges[0]
        need = max(rules.no_slip_min_s, 1.5 * f / (prior_ppm * 1e-6)) if prior_ppm else rules.no_slip_min_s
        if src[0] == "unresolved":
            m.tooth_tier, m.tooth_reason = ["U"], ["slips seen but no rate"]
        # detections that count against "no sawtooth": not those the reference explained as physiology,
        # nor those only the (more sensitive) reference->PPG detector raised and validation then refused
        ref_only = m.qc.get("reference_only_detections_t", [])
        unexplained = [x for x in m.slips_t if x not in m.qc.get("physiological_t", [])
                       and not any(abs(x - y) < 1e-6 for y in ref_only)]
        if not unexplained and dur >= need:
            m.tooth_tier, m.tooth_reason = ["N"], [f"no slip in {dur / 60:.0f} min"]
        elif not unexplained:
            m.tooth_tier, m.tooth_reason = ["U"], [f"no slip, but only {dur / 60:.0f} min (need {need / 60:.0f})"]
        else:
            m.tooth_tier, m.tooth_reason = ["U"], ["detected steps not accepted as slips"]
        return
    rates = np.asarray(m.tooth_rate)
    inserted = set(m.inserted_t)
    by_ref = set(m.qc.get("inserted_by_reference_t", []))
    rejected = np.asarray(m.qc.get("rejected_t", []), dtype=np.float64)
    # full teeth whose length implies a plausible rate (a tooth of a few minutes between two close
    # detections is not a measurement of the rate)
    lo_r, hi_r = rate_range_ppm[0] * 1e-6, rate_range_ppm[1] * 1e-6
    full = [k for k in range(1, n - 1) if src[k] == "tooth" and lo_r <= rates[k] <= hi_r]
    rmed = float(np.median(rates[full])) if full else None
    consistent = [k for k in full if abs(rates[k] / rmed - 1) <= rules.rec_tol_a] if rmed else []
    rec_ok = len(consistent) >= rules.min_teeth_a          # the recording has its own corroborated rate
    conflict = bool(rules.prior_conflict_tol is not None and rmed and prior_ppm
                    and abs(rmed / (prior_ppm * 1e-6) - 1) > rules.prior_conflict_tol)
    ref = rmed if (rec_ok or conflict) else (prior_ppm * 1e-6 if prior_ppm else rmed)
    m.qc["rate_conflict"] = conflict
    m.qc["recording_rate_ppm"] = rmed * 1e6 if rmed else None
    tiers, reasons = [], []
    for k in range(n):
        a, b = edges[k], edges[k + 1]
        sel = (tr >= a) & (tr < b)
        dp = np.diff(pat[sel])
        noise = float(1.4826 * np.median(np.abs(dp - np.median(dp))) / np.sqrt(2) * 1000) if dp.size > 10 else np.inf
        rej_inside = bool(np.any((rejected > a + 5) & (rejected < b - 5)))
        is_full = 0 < k < n - 1
        why = []
        if is_full:
            dev = abs(rates[k] / ref - 1) if ref else np.inf
            corroborated = (rec_ok and dev <= rules.rec_tol_a) or (
                not rec_ok and prior_ppm is not None and abs(rates[k] / (prior_ppm * 1e-6) - 1) <= rules.setup_tol)
            bounds_kept = a not in inserted and b not in inserted
            unconfirmed = [x for x in (a, b) if x in inserted and x not in by_ref]
            if corroborated and not rej_inside and noise <= rules.noise_max_ms:
                tier = "A" if bounds_kept else "B"
            elif dev <= rules.rec_tol_b and noise <= 1.5 * rules.noise_max_ms:
                tier = "B"
            else:
                tier = "C"
            if unconfirmed and rules.inserted_max_tier and ORDER.index(tier) < ORDER.index(rules.inserted_max_tier):
                tier = rules.inserted_max_tier
            if not bounds_kept:
                why.append("inserted slip")
            if not corroborated:
                why.append(f"rate {rates[k] * 1e6:.1f} ppm not corroborated" + (f" (ref {ref * 1e6:.1f})" if ref else ""))
            if rej_inside:
                why.append("unaccepted step inside")
            if noise > rules.noise_max_ms:
                why.append(f"noisy ({noise:.1f} ms)")
            if tier == "A":
                why = [f"rate {rates[k] * 1e6:.1f} ppm corroborated"]
        else:
            # partial end tooth: its rate is borrowed; it is predictable only if no slip was due in it
            L = f / ref if ref else None
            length = b - a
            if L is None:
                tier, why = "C", ["end tooth, no reference rate"]
            elif length > rules.end_max * L:
                # slips were due but none was seen: has the sawtooth stopped, or is it hidden?
                y = pat[sel]
                if alt is not None and np.isfinite(alt[sel]).mean() > 0.7:
                    y = alt[sel]  # the quieter reference->PPG series, where there is one
                ev = sawtooth_evidence(tr[sel], y, ref, f)
                own_ok = rmed is not None or not rules.absent_needs_own_rate
                if own_ok and ev["teeth"] >= rules.absent_min_teeth and ev["score"] < rules.absent_max_score:
                    msg = f"end tooth {length / 60:.0f} min: slips were due but the sawtooth is absent (score {ev['score']:.2f})"
                    tier, why = "N", [msg]
                else:
                    tier, why = "U", [f"end tooth {length / 60:.0f} min > one tooth ({L / 60:.0f} min): a slip was due"]
            elif (rec_ok or prior_ppm is not None) and not rej_inside and noise <= rules.noise_max_ms:
                tier, why = ("A" if rec_ok else "B"), ["end tooth, rate from " + ("consistent teeth" if rec_ok else "setup")]
            else:
                tier, why = "C", ["end tooth, " + src[k]]
        if conflict:
            note = f"recording rate {rmed * 1e6:.1f} ppm disagrees with setup {prior_ppm:.1f} ppm"
            if tier == "N":
                tier, why = "U", [why[0].replace("the sawtooth is absent", "no sawtooth seen"), note]
            elif tier in ("A", "B"):
                # the recording's rate is in doubt: B only where the reference confirmed its slips, else C
                conf = m.qc.get("confirmed_by_reference_t", [])
                ends = [x for j, x in ((k, a), (k + 1, b)) if 0 < j < n]
                if ends and all(any(abs(x - y) < 1e-6 for y in conf) for x in ends):
                    tier, why = "B", [*why, note, "slips confirmed by the reference"]
                else:
                    tier, why = "C", [*why, note]
            else:
                why = [*why, note]
        tiers.append(tier)
        reasons.append("; ".join(why))
    m.tooth_tier, m.tooth_reason = tiers, reasons


def _p2p(t: np.ndarray, x: np.ndarray, win_s: float = 60.0) -> float:
    """Spread (5-95%) of 1-minute medians: how much the slow part of a PAT series wanders, ms."""
    if t.size < 10 or t[-1] - t[0] < 3 * win_s:
        return float("nan")
    b = ((t - t[0]) // win_s).astype(int)
    meds = np.array([np.median(x[b == k]) for k in np.unique(b) if np.sum(b == k) >= 10])
    return float(np.subtract(*np.quantile(meds, [0.95, 0.05])) * 1000) if meds.size >= 3 else float("nan")


def _local_steps(t: np.ndarray, pat: np.ndarray, half: int) -> np.ndarray:
    """Step at each beat boundary i (between beats i-1 and i).

    The median of the next ``half`` beats minus the median of the previous ``half``; NaN where either
    side is short.
    """
    n = pat.size
    s = np.full(n, np.nan)
    if n < 2 * half + 1:
        return s
    w = np.lib.stride_tricks.sliding_window_view(pat, half)
    med = np.median(w, axis=1)  # med[k] = median of pat[k : k + half]
    i = np.arange(half, n - half + 1)
    s[i] = med[i] - med[i - half]
    return s


def _unit_tooth(iv: np.ndarray, tol: float) -> float | None:
    """The basic tooth length.

    The shortest length that most teeth are whole multiples (1-4) of, with at least two single teeth.
    Missed slips make teeth of 2, 3 ... units; a median would pick those.
    """
    best, best_score = None, 0
    for L in sorted(iv):
        k = np.round(iv / L)
        ok = (k >= 1) & (k <= 4) & (np.abs(iv / np.maximum(k, 1) / L - 1) < tol)
        if np.sum(ok & (k == 1)) >= 2 and np.sum(ok) > best_score:
            best, best_score = float(np.median(iv[ok] / k[ok])), int(np.sum(ok))
    return best


def _fill_rates(m: EcgPpgDelay, tr: np.ndarray, pat: np.ndarray, prior_ppm: float | None, min_tooth_s: float,
                rate_range_ppm: tuple[float, float] = (5.0, 100.0), tol: float = 0.05,
                refser: tuple[np.ndarray, np.ndarray] | None = None, ref_max_step_s: float = 0.010,
                predict: bool = True) -> None:
    """Validate the slips, then give every tooth the creep that balances one frame.

    1. Each detected slip is moved to the beat boundary with the largest 15-beat step within 30 s and
       kept if that step is 0.6-1.5 frames and the largest 3-beat step within 3 beats of it is at least
       0.4 frame (a slip happens between two beats; slow physiological changes do not pass). Slips closer than ``min_tooth_s`` to a
       better one are dropped (bursts of detections in noisy stretches).
    2. A gap of k * L (k >= 2; L the basic tooth, the shortest length most teeth are multiples of)
       between kept slips has missed slips where the data shows a step there (>= 0.4 frame); they are
       inserted. Beyond the first and last kept slip, slips that were due one tooth on are predicted
       and added on the same evidence, tooth by tooth, until a predicted place shows no step.
    3. A tooth between two slips gets rate = frame / its length, so that over the tooth the creep cancels
       exactly the frame the slip adds: D(t) comes back to the same level at every slip. A wrong or
       missed slip then distorts only its own tooth, by at most about one frame, and errors never add up.
    4. The partial teeth before the first and after the last slip get the median rate of the full teeth
       (else ``prior_ppm``), capped so that their creep stays within one frame (no slip was seen there).
       With no full tooth and no prior the sawtooth is left alone.

    With a reference (``refser`` = reference->PPG and ECG->reference per beat): a candidate whose
    ECG->reference series steps by more than ``ref_max_step_s`` at the same place is physiology and is
    rejected; a candidate that fails the step tests on ECG->PPG may pass them on reference->PPG; missed
    slips may show their step there too.
    """
    f = m.frame_s
    bound = np.r_[tr[0], 0.5 * (tr[1:] + tr[:-1])]
    s3 = _local_steps(tr, pat, 3)
    s15 = _local_steps(tr, pat, 15)
    s3f = np.where(np.isfinite(s3), s3, -np.inf)

    s15f = np.where(np.isfinite(s15), s15, -np.inf)

    def at(x: float, w: float) -> tuple[int | None, float, float]:
        """The boundary with the largest 15-beat step within w of x.

        Returns it, its 15-beat step, and the largest 3-beat step within 3 beats of it (a frame slip is
        sharp: one beat to the next).
        """
        a_, b_ = np.searchsorted(bound, x - w), np.searchsorted(bound, x + w, side="right")
        if b_ <= a_ or not np.isfinite(s15f[a_:b_]).any():
            return None, float("nan"), float("nan")
        k = a_ + int(np.argmax(s15f[a_:b_]))
        lo_, hi_ = max(k - 3, 0), min(k + 4, s3f.size)
        k3 = lo_ + int(np.argmax(s3f[lo_:hi_]))
        return k3, float(s3[k3]), float(s15[k])

    if refser is not None:
        rp3, rp15 = _local_steps(tr, refser[0], 3), _local_steps(tr, refser[0], 15)
        re15 = _local_steps(tr, refser[1], 15)

    def ref_at(x: float, w: float) -> tuple[int | None, float, float, float]:
        """As ``at`` on reference->PPG, plus the ECG->reference step at that boundary."""
        if refser is None:
            return None, float("nan"), float("nan"), float("nan")
        a_, b_ = np.searchsorted(bound, x - w), np.searchsorted(bound, x + w, side="right")
        v = np.where(np.isfinite(rp15[a_:b_]), rp15[a_:b_], -np.inf)
        if b_ <= a_ or not np.isfinite(v).any():
            return None, float("nan"), float("nan"), float("nan")
        k = a_ + int(np.argmax(v))
        lo_, hi_ = max(k - 3, 0), min(k + 4, rp3.size)
        w3 = np.where(np.isfinite(rp3[lo_:hi_]), rp3[lo_:hi_], -np.inf)
        k3 = lo_ + int(np.argmax(w3))
        return k3, float(rp3[k3]), float(rp15[k]), float(re15[k])

    def passes(st3: float, st15: float) -> bool:
        return np.isfinite(st15) and 0.6 * f <= st15 <= 1.5 * f and np.isfinite(st3) and st3 >= 0.4 * f

    cand, rejected, physio, confirmed = [], [], [], []
    for x in m.slips_t:
        k, st3, st15 = at(x, 30.0)
        kr, rt3, rt15, ref_step = ref_at(x, 30.0)
        if refser is not None and k is not None:
            # the reference is judged where the PPG stepped (the largest 15-beat ECG->PPG step)
            a_, b_ = np.searchsorted(bound, x - 30.0), np.searchsorted(bound, x + 30.0, side="right")
            k15 = a_ + int(np.argmax(s15f[a_:b_]))
            if np.isfinite(re15[k15]):
                ref_step = float(re15[k15])
        if np.isfinite(ref_step) and abs(ref_step) > ref_max_step_s:
            physio.append(float(x))  # the reference moved too: a physiological change, not a slip
            continue
        if k is not None and passes(st3, st15):
            cand.append((float(bound[k]), abs(st15 / f - 1)))
        elif kr is not None and passes(rt3, rt15):
            cand.append((float(bound[kr]), abs(rt15 / f - 1)))
        else:
            rejected.append(float(x))
            continue
        if np.isfinite(ref_step):
            confirmed.append(cand[-1][0])
    # thin bursts: keep the slip whose step is closest to one frame, drop others within min_tooth_s
    kept: list[float] = []
    for x, err in sorted(cand, key=lambda c: c[1]):
        if all(abs(x - y) >= min_tooth_s for y in kept):
            kept.append(x)
        else:
            rejected.append(x)
    kept.sort()
    S = np.asarray(kept)
    inserted: list[float] = []
    by_ref: list[float] = []   # inserted or predicted on the reference's evidence
    L = _unit_tooth(np.diff(S), tol) if S.size >= 3 else None
    if L is not None:
        for a_, b_ in pairwise(S):
            kk = round((b_ - a_) / L)
            if kk >= 2 and abs((b_ - a_) / (kk * L) - 1) < 2 * tol:
                for q in range(1, kk):
                    xq = a_ + q * (b_ - a_) / kk
                    kr, rt3, rt15, ref_step = ref_at(xq, max(15.0, tol * L))
                    if kr is not None and np.isfinite(rt15) and rt15 >= 0.4 * f and abs(ref_step) <= ref_max_step_s:
                        inserted.append(float(bound[kr]))
                        by_ref.append(float(bound[kr]))
                        continue
                    k, st3, st15 = at(xq, max(15.0, tol * L))
                    if k is not None and np.isfinite(st15) and st15 >= 0.4 * f:
                        inserted.append(float(bound[k]))
    # predict the slips that were due beyond the first and last kept slip: one tooth on, look for the
    # step there (allowing for warm-up drift), accept it on evidence, and go on from it; stop at the
    # first place without evidence (never extrapolate blind)
    predicted: list[float] = []
    Lp = L
    if Lp is None and prior_ppm is not None:
        Lp = f / (prior_ppm * 1e-6)
    if Lp is None and S.size == 2 and f / (rate_range_ppm[1] * 1e-6) <= S[1] - S[0] <= f / (rate_range_ppm[0] * 1e-6):
        Lp = float(S[1] - S[0])
    if predict and Lp is not None and S.size:
        w = max(15.0, 2 * tol * Lp)
        for x0, d in ((float(S[0]), -1), (float(S[-1]), 1)):
            y = x0 + d * Lp
            while tr[0] + 15 < y < tr[-1] - 15:
                kr, rt3, rt15, ref_step = ref_at(y, w)
                hit = None
                if kr is not None and np.isfinite(rt15) and 0.4 * f <= rt15 <= 1.6 * f and abs(ref_step) <= ref_max_step_s:
                    hit = float(bound[kr])
                    by_ref.append(hit)
                else:
                    k, st3, st15 = at(y, w)
                    ok_ref = True
                    if refser is not None and k is not None and np.isfinite(re15[k]):
                        ok_ref = abs(re15[k]) <= ref_max_step_s
                    if k is not None and np.isfinite(st15) and 0.4 * f <= st15 <= 1.6 * f and ok_ref:
                        hit = float(bound[k])
                if hit is None or any(abs(hit - z) < min_tooth_s for z in (*kept, *inserted, *predicted)):
                    break
                predicted.append(hit)
                y = hit + d * Lp
    inserted += predicted
    m.qc["predicted_t"] = sorted(predicted)
    m.qc["inserted_by_reference_t"] = sorted(set(by_ref) & set(inserted))
    grid = sorted(set(kept) | set(inserted))
    m.inserted_t = sorted(inserted)
    m.qc["rejected_t"] = sorted(rejected)
    m.qc["physiological_t"] = sorted(physio)
    m.qc["confirmed_by_reference_t"] = sorted(x for x in confirmed if x in kept)
    edges = [m.t_start, *[x for x in grid if m.t_start < x < m.t_end], m.t_end]
    m.tooth_edges = [float(x) for x in edges]
    n_teeth = len(edges) - 1
    if n_teeth == 1:
        # no slip: no evidence of a sawtooth (a prior alone does not say where its teeth are)
        m.tooth_rate, m.tooth_source = [0.0], ["no slips"]
        m.qc["rate_source"] = "no slips"
        m.qc["grid_period_s"] = None
        return
    rates: list[float] = [0.0] * n_teeth
    src = ["none"] * n_teeth
    for k in range(1, n_teeth - 1):
        rates[k], src[k] = f / (edges[k + 1] - edges[k]), "tooth"
    full = [r for r, s_ in zip(rates, src) if s_ == "tooth" and (edges[1] < edges[-2])]
    if full:
        edge_rate, esrc = float(np.median(full)), "median of teeth"
    elif prior_ppm is not None:
        edge_rate, esrc = prior_ppm * 1e-6, "prior"
    else:
        m.qc["unresolved_slips_t"] = [float(x) for x in edges[1:-1]]
        m.tooth_edges = [m.t_start, m.t_end]
        m.tooth_rate, m.tooth_source, m.inserted_t = [0.0], ["unresolved"], []
        m.qc["rate_source"] = "unresolved"
        m.qc["grid_period_s"] = None
        return
    for k in (0, n_teeth - 1):
        length = edges[k + 1] - edges[k]
        cap = f / length if length > 0 else edge_rate
        rates[k], src[k] = (min(edge_rate, cap), esrc + (" (capped)" if cap < edge_rate else ""))
    m.tooth_rate = [float(r) for r in rates]
    m.tooth_source = src
    m.qc["rate_source"] = "teeth" if full else esrc
    m.qc["grid_period_s"] = float(np.median(np.diff(edges[1:-1]))) if n_teeth > 2 else None
