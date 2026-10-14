# SPDX-License-Identifier: MIT
"""Rate-consistent slip chains: a second, lower-confidence sawtooth label track.

When a recording's sawtooth rate is known (its own corroborated rate, or its setup's), the tooth length L is
known to about a percent. A chain starts at a strong slip (a 15-beat step of 0.6-1.5 frames and a 3-beat
step of at least 0.4 frame) and expects the next slip one L on; a step of at least ``min_step`` frames
within ``tol`` x L (at least ``min_window_s``) of it is accepted and the chain re-anchors there (following
slow drift); the chain stops where none is found (a pause or a phase jump). Each pair of consecutive chained
slips is a chained tooth, corrected with rate = frame / its length.

These teeth are a separate label track (``ppg_sawtooth_chain``), not a change to the main fit: checked
against arterial pressure on VitalDB they are of B-to-C quality (see the project's report). Each label
carries the tooth's parameters, the main fit's tier over it and the caller's error estimate, so a
downstream estimator can decide whether to use it.
"""

from __future__ import annotations

import itertools

import numpy as np

from .ecg_ppg import EcgPpgDelay, _local_steps

__all__ = ["CHAIN_FIELDS", "chain_labels", "chain_slips"]

CHAIN_FIELDS = ["start", "end", "duration_s", "slope_ppm", "amplitude_ms", "tooth_s", "chain", "chain_teeth",
                "step_start_ms", "step_end_ms", "main_tiers", "main_ab_share", "error_typ_ms", "error_p90_ms",
                "usable_for_pat"]


def chain_slips(t: np.ndarray, pat: np.ndarray, *, frame_s: float, rate: float, tol: float = 0.01,
                min_step: float = 0.5, min_window_s: float = 10.0, min_sep_s: float = 60.0) -> list[list[tuple[float, float]]]:
    """Chains of (time, 15-beat step) slips in a beat-by-beat delay series (``pat`` at beat times ``t``)."""
    t = np.asarray(t, dtype=np.float64)
    pat = np.asarray(pat, dtype=np.float64)
    if t.size < 60 or rate <= 0:
        return []
    L = frame_s / rate
    bound = np.r_[t[0], 0.5 * (t[1:] + t[:-1])]
    s15 = _local_steps(t, pat, 15)
    s3 = _local_steps(t, pat, 3)
    s15f = np.nan_to_num(s15, nan=-np.inf)
    peaks: list[int] = []
    for i in np.argsort(-s15f):
        if s15f[i] < min_step * frame_s:
            break
        if all(abs(bound[i] - bound[j]) >= min_sep_s for j in peaks):
            peaks.append(int(i))
    peaks.sort()
    strong = [p for p in peaks if 0.6 * frame_s <= s15[p] <= 1.5 * frame_s
              and np.nanmax(np.nan_to_num(s3[max(p - 3, 0) : p + 4], nan=-np.inf)) >= 0.4 * frame_s]
    used: set[int] = set()
    chains = []
    w = max(min_window_s, tol * L)
    for p in sorted(strong, key=lambda i: -s15[i]):
        if p in used:
            continue
        ch = [p]
        for d in (1, -1):
            x = bound[p]
            while t[0] + 30 < x + d * L < t[-1] - 30:
                cand = [q for q in peaks if abs(bound[q] - (x + d * L)) <= w and q not in used and q not in ch]
                if not cand:
                    break
                q = min(cand, key=lambda q: abs(bound[q] - (x + d * L)))
                ch.append(q)
                x = bound[q]
        if len(ch) >= 2:
            used.update(ch)
            chains.append(sorted((float(bound[q]), float(s15[q])) for q in ch))
    return sorted(chains)


def chain_labels(m: EcgPpgDelay, r_t: np.ndarray, ppg_t: np.ndarray, *, rate_ppm: float | None = None,
                 error_s: tuple[float, float] | None = None, usable: bool = False, **kw) -> list[dict]:
    """Chained teeth of a fitted recording as label dicts (fields :data:`CHAIN_FIELDS`).

    ``rate_ppm``  the rate to chain with; default the recording's own corroborated rate (none: no labels).
    ``error_s``   (typical, 90th percentile) remaining sync error of chained teeth, as validated by the caller.
    ``usable``    the ``usable_for_pat`` value to give them (default False: a downstream method opts in).
    ``kw``        passed to :func:`chain_slips` (tol, min_step, ...)."""
    rate = (rate_ppm or m.qc.get("recording_rate_ppm") or 0.0) * 1e-6
    f = m.frame_s
    if not rate or not f:
        return []
    r_t = np.asarray(r_t, dtype=np.float64)
    pat = np.asarray(ppg_t, dtype=np.float64) - r_t
    chains = chain_slips(r_t, pat, frame_s=f, rate=rate, **kw)
    edges = np.asarray(m.tooth_edges)
    tiers = m.tooth_tier or ["U"] * (len(edges) - 1)
    out = []
    for ci, ch in enumerate(chains):
        for (a, sa), (b, sb) in itertools.pairwise(ch):
            # the main fit's tiers over this tooth, by time
            k0 = int(np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(tiers) - 1))
            k1 = int(np.clip(np.searchsorted(edges, b, side="left") - 1, 0, len(tiers) - 1))
            cover = {}
            for k in range(k0, k1 + 1):
                ov = min(b, edges[k + 1]) - max(a, edges[k])
                if ov > 0:
                    cover[tiers[k]] = cover.get(tiers[k], 0.0) + ov
            ab = (cover.get("A", 0.0) + cover.get("B", 0.0)) / (b - a)
            out.append({"start": a, "end": b, "duration_s": round(b - a, 1), "slope_ppm": round(f / (b - a) * 1e6, 2),
                        "amplitude_ms": round(f * 1000, 3), "tooth_s": round(f / rate, 1), "chain": ci,
                        "chain_teeth": len(ch) - 1, "step_start_ms": round(sa * 1000, 1), "step_end_ms": round(sb * 1000, 1),
                        "main_tiers": "+".join(sorted(cover)), "main_ab_share": round(ab, 2),
                        "error_typ_ms": None if error_s is None else round(error_s[0] * 1000, 1),
                        "error_p90_ms": None if error_s is None else round(error_s[1] * 1000, 1),
                        "usable_for_pat": bool(usable)})
    return out
