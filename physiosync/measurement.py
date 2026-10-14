# SPDX-License-Identifier: MIT
"""Measurement model of the pulse arrival time, for downstream estimators (e.g. maximum likelihood).

A downstream estimator of the latent PAT (such as the Monte Carlo likelihood + MLE of Ruffolo et al.)
needs to know how an observed PAT can differ from the latent one at each beat. :func:`pat_measurement_model`
writes that down for one recording, from a fitted :class:`~physiosync.EcgPpgDelay` and what the caller
knows about the fiducials. :func:`error_samples` and :func:`simulate_pat` draw from it, so the record is
checked to be sufficient: a likelihood can be built from it alone.

Observed corrected PAT at a beat at time t:

    pat_obs(t) = latent(t) + sync_error(t) + quant_ppg - quant_ecg [+ beat_noise] [+ recording terms]

* ``sync_error`` is what is left of the PPG's non-physiological delay after the correction that was
  applied. It depends on the segment (one per tooth of the sawtooth) and is **shared by the beats of a
  segment** (one draw per segment, then a deterministic function of t), not independent per beat:

  ``corrected``          sawtooth corrected: N(0, sd) for the segment (sd per tier, see below).
  ``none``               no sawtooth in the segment and no slip on either side: 0. (A constant level is
                         absorbed by the snapped offset.)
  ``ramp``               not corrected, but the sawtooth's course is known: e0 - rate (t - start), plus
                         N(0, sd) (both ends slips, or an end tooth too short for a slip to be due).
  ``creep_then_hold``    after a slip, creep at ``rate`` for an unknown time tau ~ U(0, frame/rate), then
                         hold: e0 - rate min(t - start, tau). Where the sawtooth stops (or a slip was due
                         and was not seen; see ``unresolved_end``).
  ``hold_then_creep``    the mirror image before a slip: e1 + rate min(end - t, tau).
  ``sawtooth_unknown_phase``  a sawtooth of known rate but unknown phase: frame (frac(phi - rate (t -
                         start) / frame) - 1/2), phi ~ U(0, 1).
  ``uniform_beat``       a sawtooth of unknown rate: U(-frame/2, frame/2), independent per beat (the
                         assumption of a method that does not track the teeth).

* ``quant_*``: the fiducial times' rounding, independent per beat: U(-res/2, res/2) for "nearest",
  U(-res, 0) for "floor", U(0, res) for "ceil" (``res`` = the fiducial's effective resolution, e.g. from
  :func:`~physiosync.fiducial_resolution`; a frame-locked PPG fiducial has the frame's resolution).
* ``beat_noise``: optional, estimated from the data (detector jitter + fast physiology), N(0, sd).
* recording terms (shared by all beats): the adopted PAT's uncertainty (the corrected PAT is snapped to
  it, so a recording's absolute PAT level is *not* measured) and any hardware-profiling uncertainties
  the caller has (uniform half-widths).

Only numpy.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from .ecg_ppg import EcgPpgDelay

__all__ = ["DEFAULT_RESIDUAL_FRAC", "SCHEMA", "error_samples", "fiducial_error_model", "pat_measurement_model", "simulate_pat"]

SCHEMA = "physiosync.pat_measurement/1"

# Residual sync error after correction, as a fraction of the frame (SD), per tier: from physiosync's
# injected-sawtooth benchmark (30 Hz frame, 18-52 ppm, ~0.8 s beats): A 0.2 ms, B 1.6 ms, C 5.3 ms RMS.
# A starting point only: benchmark your own data and pass ``residual_sd_s``.
DEFAULT_RESIDUAL_FRAC = {"A": 0.006, "B": 0.048, "C": 0.16}


def _quant(res: float | None, rounding: str) -> dict:
    if not res:
        return {"kind": "none"}
    lo, hi = {"nearest": (-res / 2, res / 2), "floor": (-res, 0.0), "ceil": (0.0, res)}[rounding]
    return {"kind": "uniform", "low_s": lo, "high_s": hi, "resolution_s": res, "rounding": rounding}


def fiducial_error_model(index: np.ndarray | None, fs: float, *, frame_samples: float | None = None,
                         pat_noise_s: float | None = None, other_sd_s: float = 0.0, lock_ratio_min: float = 2.0,
                         ibi_range: tuple[int, int] | None = None) -> dict:
    """The timing error model of one recording's fiducial, from its own data.

    A fiducial that locks to a frame grid is rounded to the frame: uniform, one frame wide. It counts as
    locked when its interval resultant at ``frame_samples`` is at least ``lock_ratio_min`` times that at the
    control periods (see :func:`~physiosync.interval_lock`). Otherwise its error is jitter: normal, with SD
    the beat-to-beat PAT noise less the other fiducial's share (an upper bound, since physiology is in the
    noise too). With neither, one sample, uniform.

    Args:
        index: The fiducial's integer sample indices, or ``None``.
        fs: Sampling rate (Hz).
        frame_samples: The frame (samples), if the signal is framed.
        pat_noise_s: Beat-to-beat PAT noise (SD, s), for the jitter model.
        other_sd_s: SD of the other fiducial's error (s), removed from ``pat_noise_s``.
        lock_ratio_min: Lock ratio from which the fiducial counts as on the frame grid.
        ibi_range: ``(low, high)`` plausible interval (samples) for the lock test.

    Returns:
        An error spec: ``kind`` (``"uniform"`` or ``"normal"``), its width (``low_s``/``high_s``) or ``sd_s``,
        ``resolution_s``, ``rounding`` (``"frame"``, ``"jitter"`` or ``"nearest"``) and ``basis`` (how it
        was decided). Pass it as ``ppg_error`` to :func:`pat_measurement_model`.
    """
    from .frames import interval_lock

    lock = None
    if index is not None and frame_samples and np.asarray(index).size >= 100:
        lock = interval_lock(np.asarray(index), frame_samples, ibi_range=ibi_range)
        if lock["ratio"] >= lock_ratio_min:
            w = frame_samples / fs
            return {"kind": "uniform", "low_s": -w / 2, "high_s": w / 2, "resolution_s": w, "rounding": "frame",
                    "basis": "locks to the frame grid", "frame_lock": lock["resultant"], "lock_ratio": lock["ratio"]}
    if pat_noise_s is not None and np.isfinite(pat_noise_s):
        sd = float(np.sqrt(max(pat_noise_s**2 - other_sd_s**2, 0.0)))
        return {"kind": "normal", "sd_s": sd, "resolution_s": sd * np.sqrt(12), "rounding": "jitter",
                "basis": "beat-to-beat PAT noise (upper bound)", "frame_lock": None if lock is None else lock["resultant"],
                "lock_ratio": None if lock is None else lock["ratio"]}
    return {**_quant(1 / fs, "nearest"), "basis": "one sample (no data)"}


def pat_measurement_model(m: EcgPpgDelay, *, ecg_fs: float, ppg_fs: float,
                          ecg_fiducial: str = "", ppg_fiducial: str = "",
                          ecg_resolution_s: float | None = None, ppg_resolution_s: float | None = None,
                          ecg_rounding: str = "nearest", ppg_rounding: str = "nearest",
                          residual_sd_s: dict[str, float] | None = None,
                          adopted_pat_sd_s: float | None = None, adopted_pat_basis: str = "",
                          hardware_half_widths_s: dict[str, float] | None = None,
                          unresolved_end: str = "hold", r_t: np.ndarray | None = None,
                          ppg_t: np.ndarray | None = None, ppg_frame_clock_ppm: float | None = None,
                          ppg_frame_lock: float | None = None, ppg_error: dict | None = None,
                          extra: dict | None = None) -> dict:
    """The PAT measurement model of one recording, as a JSON-ready record.

    The record holds what a downstream estimator needs to relate an observed PAT to the latent one: the
    fiducials' resolution and rounding, the offset, the applied correction, the slips, and one sync-error
    model per segment (corrected, ramp, creep-then-hold where the sawtooth stops, or unknown phase). Draw
    from it with :func:`error_samples` and :func:`simulate_pat`; write it with :func:`dumps`.

    Args:
        m: The fit from :func:`~physiosync.fit_ecg_ppg_delay`.
        ecg_fs: Nominal ECG sampling rate (Hz).
        ppg_fs: Nominal PPG sampling rate (Hz).
        ecg_fiducial: Name of the ECG fiducial, e.g. ``"R peak"``.
        ppg_fiducial: Name of the PPG fiducial, e.g. ``"max upstroke, 0.5-8 Hz"``.
        ecg_resolution_s: Effective time resolution of the ECG fiducial (s). Default: one sample.
        ppg_resolution_s: Effective time resolution of the PPG fiducial (s). Default: one sample. A fiducial
            taken on raw samples of a framed signal can have the frame's resolution: check with
            :func:`~physiosync.fiducial_resolution` and :func:`~physiosync.locking`.
        ecg_rounding: How the ECG fiducial's time relates to the true event: ``"nearest"``, ``"floor"`` or
            ``"ceil"``.
        ppg_rounding: The same for the PPG fiducial.
        residual_sd_s: SD of the sync error left after correction, per tier (``"A"``, ``"B"``, ``"C"``; s).
            Default: :data:`DEFAULT_RESIDUAL_FRAC` times the frame.
        adopted_pat_sd_s: Uncertainty of the adopted mean PAT the offset snaps to (s), a recording-level term.
        adopted_pat_basis: Where the adopted PAT came from.
        hardware_half_widths_s: Further recording-level uncertainties, each uniform on ``(-w, w)``, e.g. from
            profiling the hardware: ``{"arrival": 0.002, "deviation": 0.008}``.
        unresolved_end: For an end tooth where a slip was due but none was seen and the sawtooth was not shown
            absent: ``"hold"`` (it stopped somewhere in the first tooth length) or ``"sawtooth"`` (it went on,
            with unknown phase).
        r_t: R-peak times the model was fitted on (s), to estimate the beat noise.
        ppg_t: PPG fiducial times of the same beats (s).
        ppg_frame_clock_ppm: The PPG frame clock's rate against the sampling clock, if measured
            (:func:`~physiosync.frame_clock`).
        ppg_frame_lock: How strongly the PPG fiducial locks to the frame grid (resultant 0-1), if measured.
        ppg_error: The PPG fiducial's error model from :func:`fiducial_error_model`. Overrides
            ``ppg_resolution_s`` and ``ppg_rounding``.
        extra: Anything else to keep with the record.

    Returns:
        A dict with schema :data:`SCHEMA` and keys ``schema``, ``t_start``, ``t_end``, ``time_base``,
        ``clocks``, ``fiducials``, ``offset``, ``correction``, ``slips``, ``segments``, ``state_time_s``,
        ``beat_noise``, ``recording_terms`` and ``extra``.
    """
    f = m.frame_s
    res_sd = {k: v * (f or 0.0) for k, v in DEFAULT_RESIDUAL_FRAC.items()}
    res_src = "physiosync injected benchmark (fraction of the frame)"
    if residual_sd_s:
        res_sd.update(residual_sd_s)
        res_src = "caller"
    edges = list(m.tooth_edges) if m.tooth_edges else [m.t_start, m.t_end]
    n = len(edges) - 1
    tiers = m.tooth_tier or ["U"] * n
    reasons = m.tooth_reason or [""] * n
    # the sawtooth as fitted, unmasked: where the delay really is in teeth that were not corrected
    full = replace(m, tooth_tier=[], qc=dict(m.qc))
    full_saw = (lambda t: full._saw(np.atleast_1d(np.asarray(t, dtype=np.float64))))
    saw_med = float(m.qc.get("saw_median_s", 0.0))
    good = [r for r, s in zip(m.tooth_rate, m.tooth_source) if s == "tooth"]
    rec_rate = (m.qc["recording_rate_ppm"] * 1e-6 if m.qc.get("recording_rate_ppm")
                else (float(np.median(good)) if good and "recording_rate_ppm" not in m.qc else None))
    prior = m.qc.get("rate_prior_ppm")
    eps = 1e-6
    conf_ref = set(m.qc.get("confirmed_by_reference_t", [])) | set(m.qc.get("inserted_by_reference_t", []))
    pred = set(m.qc.get("predicted_t", []))

    def evidence(x: float, is_slip: bool) -> str | None:
        """How a slip bounding a tooth is known."""
        if not is_slip:
            return None
        how = "predicted" if x in pred else "inserted" if x in m.inserted_t else "detected"
        return how + ("+reference" if x in conf_ref else "")

    segs = []
    for k in range(n):
        a, b = float(edges[k]), float(edges[k + 1])
        tier = tiers[k]
        applied = tier in ("A", "B", "C") and m.applied(k)
        slip_a, slip_b = k > 0, k < n - 1
        own = float(m.tooth_rate[k]) if k < len(m.tooth_rate) and m.tooth_rate[k] > 0 else None
        # a tooth that was not corrected: the recording's own rate (an end tooth's rate is capped, and
        # means nothing where the sawtooth stopped), else the setup's, else its own
        rate = own if applied else (rec_rate or (prior * 1e-6 if prior else None) or own)
        d0 = float(m.shift(np.array([a + eps]))[0])
        d1 = float(m.shift(np.array([b - eps]))[0])
        seg = {"start": a, "end": b, "tier": tier, "applied": applied, "reason": reasons[k],
               "slip_at_start": slip_a, "slip_at_end": slip_b,
               "inserted_slip_at_start": slip_a and a in m.inserted_t, "inserted_slip_at_end": slip_b and b in m.inserted_t,
               "state": "active" if applied else {"N": "paused", "U": "unknown"}.get(tier, "active"),
               # the correction that was applied: D(t) = d0 - slope (t - start), on top of the raw PAT
               "applied_shift": {"d0_s": d0, "slope": (d0 - d1) / max(b - a - 2 * eps, eps)},
               "sawtooth": {"frame_s": f, "rate": rate, "tooth_s": (f / rate) if f and rate else None,
                            "rate_source": m.tooth_source[k] if k < len(m.tooth_source) else None}}
        top = float(full_saw(a + eps)[0]) - saw_med   # just after a slip at a (relative to the applied level)
        bottom = float(full_saw(b - eps)[0]) - saw_med  # just before a slip at b
        if applied:
            err = {"kind": "corrected", "sd_s": res_sd.get(tier)}
        elif not m.sawtooth or f is None or tier == "N" and not slip_a and not slip_b:
            err = {"kind": "none"}
        elif tier == "N" or (tier == "U" and "slip was due" in reasons[k] and unresolved_end == "hold"):
            if rate is None:
                err = {"kind": "uniform_beat", "half_width_s": f / 2}
            elif slip_a and not slip_b:
                err = {"kind": "creep_then_hold", "e0_s": top, "rate": rate, "tau_max_s": f / rate}
            elif slip_b and not slip_a:
                err = {"kind": "hold_then_creep", "e1_s": bottom, "rate": rate, "tau_max_s": f / rate}
            else:  # paused between two slips: creep from the first, hold, creep to the second (each part unknown)
                err = {"kind": "creep_then_hold", "e0_s": top, "rate": rate, "tau_max_s": f / rate}
        elif tier in ("A", "B", "C"):
            # corrected by the model but below the threshold: the model's ramp, with its tier's error
            err = {"kind": "ramp", "e0_s": top if slip_a else float(full_saw(a + eps)[0]) - saw_med,
                   "rate": float(m.tooth_rate[k]), "sd_s": res_sd.get(tier)}
        elif rate is not None:
            err = {"kind": "sawtooth_unknown_phase", "frame_s": f, "rate": rate}
            if slip_a:   # the phase is known at the slip: it is the top of a tooth
                err = {"kind": "ramp_wrapped", "e0_s": top, "rate": rate, "frame_s": f}
        else:
            err = {"kind": "uniform_beat", "half_width_s": f / 2}
        seg["sync_error"] = err
        seg["slip_evidence"] = {"start": evidence(a, slip_a), "end": evidence(b, slip_b)}
        if r_t is not None and ppg_t is not None:
            sel = (np.asarray(r_t) >= a) & (np.asarray(r_t) < b)
            cp = (m.correct(np.asarray(ppg_t, dtype=np.float64)[sel]) - np.asarray(r_t, dtype=np.float64)[sel])
            dp = np.diff(cp[np.isfinite(cp)])
            seg["beat_noise_s"] = (float(1.4826 * np.median(np.abs(dp - np.median(dp))) / np.sqrt(2))
                                   if dp.size > 10 else None)
            seg["beats"] = int(sel.sum())
        segs.append(seg)
    beat_sd = None
    if r_t is not None and ppg_t is not None:
        r_t = np.asarray(r_t, dtype=np.float64)
        cp = m.correct(np.asarray(ppg_t, dtype=np.float64)) - r_t
        dp = np.diff(cp[np.isfinite(cp)])
        if dp.size > 10:
            beat_sd = float(1.4826 * np.median(np.abs(dp - np.median(dp))) / np.sqrt(2))
    rec_terms = {}
    if adopted_pat_sd_s:
        rec_terms["adopted_pat"] = {"kind": "normal", "sd_s": float(adopted_pat_sd_s)}
    for k, w in (hardware_half_widths_s or {}).items():
        rec_terms[f"hardware_{k}"] = {"kind": "uniform", "low_s": -float(w), "high_s": float(w)}
    dur = {s: sum(g["end"] - g["start"] for g in segs if g["state"] == s) for s in ("active", "paused", "unknown")}
    return {
        "schema": SCHEMA,
        "t_start": m.t_start, "t_end": m.t_end, "time_base": "ECG clock, s",
        "clocks": {"ecg_fs": ecg_fs, "ppg_fs": ppg_fs, "ppg_frame_s": f, "ppg_frame_clock_ppm": ppg_frame_clock_ppm},
        "fiducials": {
            "ecg": {"name": ecg_fiducial, **_quant(ecg_resolution_s or 1 / ecg_fs, ecg_rounding), "sample_s": 1 / ecg_fs},
            "ppg": {"name": ppg_fiducial, **(ppg_error or _quant(ppg_resolution_s or 1 / ppg_fs, ppg_rounding)),
                    "sample_s": 1 / ppg_fs, "frame_s": f,
                    **({} if ppg_error else {"frame_lock": ppg_frame_lock})}},
        "offset": {"offset_s": m.offset_s, "adopted_pat_s": m.adopted_pat_s, "basis": adopted_pat_basis,
                   "note": "one combined hardware + filter delay; snapped per recording (the recording's own level is "
                           "then not kept) unless supplied by the caller, e.g. per setup",
                   "saw_median_s": saw_med, "level_from": m.qc.get("offset_basis", "all beats"),
                   "raw_level_s": m.qc.get("raw_level_s")},
        "correction": {"mode": "per-tooth (each tooth balances one frame)", "apply_min_tier": m.apply_min_tier,
                       "residual_sd_s": res_sd, "residual_source": res_src, "unresolved_end": unresolved_end},
        "slips": {"t": [float(x) for x in m.tooth_edges[1:-1]], "detected_t": m.slips_t, "inserted_t": m.inserted_t,
                  "predicted_t": m.qc.get("predicted_t", []), "confirmed_by_reference_t": m.qc.get("confirmed_by_reference_t", []),
                  "rejected_t": m.qc.get("rejected_t", []), "physiological_t": m.qc.get("physiological_t", [])},
        "segments": segs,
        "state_time_s": dur,
        "beat_noise": {"kind": "normal", "sd_s": beat_sd} if beat_sd is not None else None,
        "recording_terms": rec_terms,
        "extra": extra or {},
    }


def _seg_index(rec: dict, t: np.ndarray) -> np.ndarray:
    starts = np.array([s["start"] for s in rec["segments"]])
    return np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 1)


def _uniform(rng: np.random.Generator, spec: dict, size) -> np.ndarray:
    """A fiducial's error: uniform (rounding) or normal (jitter)."""
    if spec.get("kind") == "uniform":
        return rng.uniform(spec["low_s"], spec["high_s"], size)
    if spec.get("kind") == "normal":
        return rng.normal(0.0, spec["sd_s"], size)
    return np.zeros(size)


def error_samples(rec: dict, t: np.ndarray, n: int, *, rng: np.random.Generator | None = None,
                  components: tuple[str, ...] = ("sync", "quantisation", "recording")) -> np.ndarray:
    """Draw observed-minus-latent corrected PAT at given beat times, from a record.

    Components: ``"sync"`` (per segment, shared by its beats), ``"quantisation"`` (per beat),
    ``"beat_noise"`` (per beat, if the record has it) and ``"recording"`` (shared by all beats). Leave out
    the shared ones to get the per-beat part only, e.g. to build a joint likelihood that integrates over the
    shared ones.

    Args:
        rec: A record from :func:`pat_measurement_model`.
        t: Beat times (s).
        n: Number of draws.
        rng: Random generator. Default: a new unseeded one.
        components: Error components to include.

    Returns:
        An array of shape ``(n, len(t))`` (s).
    """
    rng = rng or np.random.default_rng()
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros((n, t.size))
    if "sync" in components:
        idx = _seg_index(rec, t)
        for k in np.unique(idx):
            sel = idx == k
            seg = rec["segments"][k]
            e = seg["sync_error"]
            kind = e["kind"]
            tt = t[sel][None, :]
            if kind == "corrected":
                out[:, sel] += rng.normal(0.0, e["sd_s"] or 0.0, (n, 1))
            elif kind == "ramp":
                out[:, sel] += e["e0_s"] - e["rate"] * (tt - seg["start"]) + rng.normal(0.0, e["sd_s"] or 0.0, (n, 1))
            elif kind == "creep_then_hold":
                tau = rng.uniform(0.0, e["tau_max_s"], (n, 1))
                out[:, sel] += e["e0_s"] - e["rate"] * np.minimum(tt - seg["start"], tau)
            elif kind == "hold_then_creep":
                tau = rng.uniform(0.0, e["tau_max_s"], (n, 1))
                out[:, sel] += e["e1_s"] + e["rate"] * np.minimum(seg["end"] - tt, tau)
            elif kind in ("sawtooth_unknown_phase", "ramp_wrapped"):
                fr = e["frame_s"]
                phi = rng.uniform(0.0, 1.0, (n, 1)) if kind == "sawtooth_unknown_phase" else 0.5 + e["e0_s"] / fr
                out[:, sel] += fr * (np.mod(phi - e["rate"] * (tt - seg["start"]) / fr, 1.0) - 0.5)
            elif kind == "uniform_beat":
                out[:, sel] += rng.uniform(-e["half_width_s"], e["half_width_s"], (n, int(sel.sum())))
    if "quantisation" in components:
        out += _uniform(rng, rec["fiducials"]["ppg"], out.shape) - _uniform(rng, rec["fiducials"]["ecg"], out.shape)
    if "beat_noise" in components and rec.get("beat_noise"):
        out += rng.normal(0.0, rec["beat_noise"]["sd_s"], out.shape)
    if "recording" in components:
        for spec in rec.get("recording_terms", {}).values():
            if spec["kind"] == "normal":
                out += rng.normal(0.0, spec["sd_s"], (n, 1))
            else:
                out += _uniform(rng, spec, (n, 1))
    return out


def simulate_pat(rec: dict, t: np.ndarray, latent_pat_s: np.ndarray | float, *, n: int = 1, corrected: bool = True,
                 rng: np.random.Generator | None = None,
                 components: tuple[str, ...] = ("sync", "quantisation", "recording")) -> np.ndarray:
    """Simulate the observed PAT for a latent PAT at given beat times.

    Args:
        rec: A record from :func:`pat_measurement_model`.
        t: Beat times (s).
        latent_pat_s: The latent PAT (s), one value or one per beat.
        n: Number of draws.
        corrected: ``True`` for the PAT as stored after physiosync's correction; ``False`` for the raw PAT as
            the recording had it (the applied shift added back).
        rng: Random generator. Default: a new unseeded one.
        components: Error components to include (see :func:`error_samples`).

    Returns:
        Observed PAT, shape ``(n, len(t))`` (s).
    """
    t = np.asarray(t, dtype=np.float64)
    obs = np.asarray(latent_pat_s, dtype=np.float64) + error_samples(rec, t, n, rng=rng, components=components)
    if not corrected:
        idx = _seg_index(rec, t)
        sh = np.array([rec["segments"][k]["applied_shift"]["d0_s"] - rec["segments"][k]["applied_shift"]["slope"]
                       * (x - rec["segments"][k]["start"]) for k, x in zip(idx, t)])
        obs = obs + sh
    return obs


def dumps(rec: dict) -> str:
    """Serialise a record as JSON (NaN-free).

    Args:
        rec: A record from :func:`pat_measurement_model`.

    Returns:
        The JSON text.

    Raises:
        ValueError: The record contains NaN or infinity.
    """
    return json.dumps(rec, allow_nan=False)
