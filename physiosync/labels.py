# SPDX-License-Identifier: MIT
"""Sawtooth labels: one interval per tooth (a new interval wherever the sawtooth's properties change), with
its parameters, the fiducials' time resolution and a confidence a downstream PAT estimator can act on.

:func:`sawtooth_labels` turns a measurement record (:func:`~physiosync.pat_measurement_model`) into a list
of flat, JSON- and CSV-ready dicts. Each interval runs from one slip to the next (or to the start or end of
the analysed stretch) and carries:

* **sawtooth**: ``state`` (active / paused / unknown), ``slope_ppm`` (how fast the PPG delay creeps
  down, ppm = microseconds per second), ``amplitude_ms`` (one frame: the jump at a slip), ``tooth_s``
  (the period, amplitude / slope), ``duration_s``, ``phase_start`` and ``phase_end`` (the share of a tooth
  elapsed since the last slip: 0 just after a slip, 1 at the next; empty where no slip anchors it), how each
  bounding slip is known (detected, inserted, predicted, "+reference" if the arterial line showed it).
* **applied correction**: ``shift_start_ms`` and ``shift_slope_ppm``, so that the PPG delay removed at time
  t in the interval is shift_start - slope (t - start). With these a downstream method can undo physiosync's
  correction and apply its own to the raw PAT (e.g. half the amplitude, as in Ruffolo et al. section 3.8).
* **fiducial error**: for the ECG and the PPG fiducial, its error model in this recording: "rounding"
  (uniform, ``*_error_ms`` wide; e.g. a fiducial that locks to the frame grid) or "jitter" (normal,
  ``*_error_ms`` SD; e.g. an interpolated fiducial), with its basis; the PPG sampling period and frame, and
  how strongly the PPG fiducial locks to the frame grid (resultant and ratio to control periods).
* **confidence**: the tier (A/B/C corrected, N no sawtooth, U unresolved) and its reason; the remaining
  sync error, by drawing from the interval's error model (:func:`~physiosync.error_samples`): mean
  (``sync_bias_ms``), SD and 95th percentile of its size; the SD the two fiducials' errors add to the PAT (``fiducial_sd_ms``); the beat-to-
  beat noise; and ``usable_for_pat``: the tier is one the caller accepts and, if a limit is given, the
  95th-percentile error is within it. A downstream estimator can skip intervals that are not usable.

:func:`sawtooth_label_tuples` gives (name, start, end) for annotation tracks that hold names only.
"""

from __future__ import annotations

import numpy as np

from .measurement import error_samples

__all__ = ["LABEL_FIELDS", "OFFSET_FIELDS", "offset_label", "sawtooth_label_tuples", "sawtooth_labels"]

OFFSET_FIELDS = ["start", "end", "offset_ms", "adopted_pat_ms", "basis", "n_recordings", "setup_iqr_ms",
                 "recording_level_ms", "ecg_fiducial", "ppg_fiducial"]

LABEL_FIELDS = [
    "start", "end", "duration_s", "state", "tier", "usable_for_pat", "reason",
    "slope_ppm", "amplitude_ms", "tooth_s", "phase_start", "phase_end", "slip_start", "slip_end",
    "shift_start_ms", "shift_slope_ppm",
    "sync_error_model", "sync_bias_ms", "sync_sd_ms", "sync_p95_ms", "fiducial_sd_ms", "beat_noise_ms", "beats",
    "ecg_error_model", "ecg_error_ms", "ppg_fiducial", "ppg_error_model", "ppg_error_ms", "ppg_error_basis",
    "ppg_sample_ms", "ppg_frame_ms", "ppg_frame_lock", "ppg_lock_ratio",
]


def _ms(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x) * 1000, nd)


def _rounding_sd(spec: dict) -> float:
    if spec.get("kind") == "uniform":
        return (spec["high_s"] - spec["low_s"]) / np.sqrt(12)
    return float(spec.get("sd_s", 0.0)) if spec.get("kind") == "normal" else 0.0


def _err(spec: dict) -> tuple[str, float | None]:
    """("rounding", width) or ("jitter", SD), in seconds."""
    if spec.get("kind") == "uniform":
        return "rounding", spec["high_s"] - spec["low_s"]
    if spec.get("kind") == "normal":
        return "jitter", spec["sd_s"]
    return "none", None


def sawtooth_labels(rec: dict, *, usable_tiers: tuple[str, ...] = ("A", "B", "N"), max_error_s: float | None = None,
                    draws: int = 400, seed: int = 0) -> list[dict]:
    """One dict per tooth of the record (fields :data:`LABEL_FIELDS`). ``usable_tiers`` and ``max_error_s``
    decide ``usable_for_pat``."""
    rng = np.random.default_rng(seed)
    fe, fp = rec["fiducials"]["ecg"], rec["fiducials"]["ppg"]
    rounding_sd = float(np.hypot(_rounding_sd(fe), _rounding_sd(fp)))
    out = []
    for seg in rec["segments"]:
        a, b = seg["start"], seg["end"]
        saw = seg["sawtooth"]
        f, rate = saw.get("frame_s"), saw.get("rate")
        L = (f / rate) if f and rate else None
        # phase: share of a tooth elapsed since the last slip
        ph0 = ph1 = None
        if L:
            if seg["slip_at_start"]:
                ph0 = 0.0
                ph1 = 1.0 if seg["slip_at_end"] else min((b - a) / L, 1.0)
            elif seg["slip_at_end"]:
                ph1 = 1.0
                ph0 = max(1.0 - (b - a) / L, 0.0)
        if seg["state"] == "paused":
            # the creep stops somewhere in the first tooth; the phase after that is not defined
            ph1 = None if seg["slip_at_start"] else ph1
            ph0 = None if seg["slip_at_end"] else ph0
        # the sync error left in the interval, from its model (shared draws across the interval's beats)
        tt = np.linspace(a, b, 41)
        e = error_samples(rec, tt, draws, rng=rng, components=("sync",))
        sh = seg["applied_shift"]
        usable = seg["tier"] in usable_tiers
        p95 = float(np.quantile(np.abs(e), 0.95))
        if max_error_s is not None and p95 > max_error_s:
            usable = False
        ev = seg.get("slip_evidence", {})
        out.append({
            "start": a, "end": b, "duration_s": round(b - a, 1), "state": seg["state"], "tier": seg["tier"],
            "usable_for_pat": bool(usable), "reason": seg.get("reason", ""),
            "slope_ppm": round(rate * 1e6, 2) if rate else None, "amplitude_ms": _ms(f), "tooth_s": round(L, 1) if L else None,
            "phase_start": None if ph0 is None else round(ph0, 4), "phase_end": None if ph1 is None else round(ph1, 4),
            "slip_start": ev.get("start"), "slip_end": ev.get("end"),
            "shift_start_ms": _ms(sh["d0_s"]), "shift_slope_ppm": round(sh["slope"] * 1e6, 3),
            "sync_error_model": seg["sync_error"]["kind"],
            "sync_bias_ms": _ms(float(e.mean())), "sync_sd_ms": _ms(float(e.std())), "sync_p95_ms": _ms(p95),
            "fiducial_sd_ms": _ms(rounding_sd), "beat_noise_ms": _ms(seg.get("beat_noise_s")), "beats": seg.get("beats"),
            "ecg_error_model": _err(fe)[0], "ecg_error_ms": _ms(_err(fe)[1]),
            "ppg_fiducial": fp.get("name", ""), "ppg_error_model": _err(fp)[0], "ppg_error_ms": _ms(_err(fp)[1]),
            "ppg_error_basis": fp.get("basis", "caller"),
            "ppg_sample_ms": _ms(fp.get("sample_s")), "ppg_frame_ms": _ms(fp.get("frame_s")),
            "ppg_frame_lock": fp.get("frame_lock"), "ppg_lock_ratio": fp.get("lock_ratio"),
        })
    return out


def sawtooth_label_tuples(labels: list[dict], prefix: str = "ppg_sawtooth") -> list[tuple[str, float, float]]:
    """("ppg_sawtooth:A", start, end), ... - name-only annotations; ":x" is added when not usable for PAT."""
    return [(f"{prefix}:{d['tier']}" + ("" if d["usable_for_pat"] else ":x"), d["start"], d["end"]) for d in labels]


def offset_label(rec: dict, *, basis: str = "", n_recordings: int | None = None,
                 setup_iqr_s: list[float] | None = None) -> dict:
    """The combined hardware + filter delay that was removed from this recording's PPG times, as one label
    over the whole analysed stretch (kept apart from the sawtooth labels so that it can be made as granular
    as needed later: per setup, per device, per recording). ``recording_level_ms`` is the recording's own
    median PAT before any offset (sawtooth removed): its corrected median is that minus ``offset_ms``."""
    off = rec["offset"]
    return {"start": rec["t_start"], "end": rec["t_end"], "offset_ms": round(off["offset_s"] * 1000, 2),
            "adopted_pat_ms": round(off["adopted_pat_s"] * 1000, 1), "basis": basis or off.get("level_from", ""),
            "n_recordings": n_recordings,
            "setup_iqr_ms": None if not setup_iqr_s else [round(x * 1000, 1) for x in setup_iqr_s],
            "recording_level_ms": None if off.get("raw_level_s") is None else round(off["raw_level_s"] * 1000, 2),
            "ecg_fiducial": rec["fiducials"]["ecg"].get("name", ""), "ppg_fiducial": rec["fiducials"]["ppg"].get("name", "")}
