# SPDX-License-Identifier: MIT
"""Using a timing-correction release: corrected times, Monte Carlo realisations and the three-way run.

A release is a folder holding ``manifest.json`` (with the SHA-256 of every file) and one JSON file per record;
the MIT-BIH release is built by the companion package ``mitbih-chain``. Each record file carries

- ``map``: the record's fitted ``TimeMap`` (recorded time -> corrected time, per lead);
- ``draws``: K bootstrap parameter sets (complete maps, so all times in a record move together); where the
  lead skew could not be measured the point map has no skew and each draw carries a prior skew draw;
- ``gp``: the unmodelled time-base term (Ornstein-Uhlenbeck: ``sigma_s``, ``ell_s``; optional ``white_s``);
- ``labels``: fitted fiducial times keyed by reference-label sample index, with per-beat SDs;
- ``available_at_s``: the end of the record. Corrections are fitted on whole records, so every corrected
  value depends on later data: use them offline only, never as real-time features.

Building blocks: ``load_release``, ``Release.record``, ``RecordCorrection.to_corrected``,
``RecordCorrection.realisations``. The runner ``run`` applies an analysis to the original times, the
corrected times and K realisations, and reports the change, the standard uncertainty and whether the change
is larger than the spread (a change smaller than the spread should not be interpreted), with the release
checksum, seed and K for reproduction. Every realisation is reproducible from (release, record, k, seed).
Moved from ``adaptivepeaks.correction`` (v0.1.2 of the MIT-BIH release) without changing any computation.

Examples:
    >>> import numpy as np
    >>> rc = RecordCorrection.from_dict(_example_record())
    >>> t = np.arange(10.0, 60.0, 0.8)
    >>> tau = rc.to_corrected(t, lead=0)
    >>> bool(np.all(np.diff(tau) > 0)), rc.available_at_s
    (True, 1805.6)
    >>> res = run(lambda x: float(np.std(np.diff(x))) * 1000, rc, t, lead=0, k=20, seed=1)
    >>> sorted(res)[:4]
    ['available_at_s', 'change', 'change_exceeds_spread', 'corrected']
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .timemap import DriftingPeriodic, PeriodicWarp, TabulatedOffset, TimeMap

__all__ = ["RecordCorrection", "Release", "ReleaseError", "load_release", "run"]

STREAM_GP, STREAM_LABEL = 1, 2
STREAM_WHITE = 4      # white time-base term (release gp.white_s), from v0.1.1


class ReleaseError(ValueError):
    """A release file is missing, altered or inconsistent with its manifest."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rec_int(record):
    digits = "".join(ch for ch in str(record) if ch.isdigit())
    return int(digits) if digits else int(hashlib.sha256(str(record).encode()).hexdigest()[:8], 16)


class RecordCorrection:
    """One record's correction: point map, bootstrap maps, GP term, label sets."""

    def __init__(self, d, checksum=None):
        self.d = d
        self.record = str(d["record"])
        self.map = TimeMap.from_dict(d["map"])
        self.available_at_s = float(d["available_at_s"])
        self.gp = d.get("gp") or {}
        self.checksum = checksum

    @classmethod
    def from_dict(cls, d, checksum=None):
        """Build from a record file's parsed JSON."""
        return cls(d, checksum)

    @property
    def k(self):
        """Number of stored bootstrap draws."""
        return len(self.d.get("draws", []))

    def to_corrected(self, t, lead=None):
        """Corrected times of recorded times ``t`` on ``lead``."""
        return self.map.to_corrected(t, lead)

    def draw_map(self, k):
        """The TimeMap of bootstrap draw ``k``."""
        dr = self.d["draws"][k]
        comps = []
        if self.d["map_components"]["warp"]:
            comps.append(PeriodicWarp(dr["P"], dr["warp_coef"]))
        if dr.get("flutter"):
            f = dr["flutter"]
            comps.append(DriftingPeriodic(f["p0"], f["p1"], f["tc"], f["coef"]))
        # tabulated clock drift: the release table, rescaled by the draw's drift_scale
        for c in self.map.components:
            if isinstance(c, TabulatedOffset):
                comps.append(TabulatedOffset(c.t_s, c.offset_s, dr.get("drift_scale", 1.0)))
        offs = list(self.map.lead_offsets_s)
        # a draw's skew: measured (bootstrap) or, where the skew could not be measured, a draw from the
        # release's prior (``skew_prior``), so cross-lead results carry that uncertainty
        if dr.get("skew_ms") is not None:
            ref = self.d["reference_lead"]
            sk = dr["skew_ms"] / 1000.0
            offs = [0.0] * max(2, len(offs))
            offs[1 - ref] = -sk if ref == 0 else sk
        return TimeMap(comps, offs)

    def realisation(self, k, t, lead=None, label_sd=None, seed=0, gp=True):
        """Corrected times for Monte Carlo realisation ``k``.

        Bootstrap map k, plus a GP realisation (shared by
        all times of the record), plus the release's white time-base term if any (``gp.white_s``: an
        independent N(0, white_s) error per time; part of the unmodelled time base, switched with ``gp``) and, if ``label_sd`` is given (seconds, scalar or per time), an independent
        placement draw per label. Reproducible from (release, record, k, seed).
        """
        t = np.asarray(t, float)
        tau = self.draw_map(k).to_corrected(t, lead)
        rr = _rec_int(self.record)
        if gp and self.gp.get("sigma_s"):
            order = np.argsort(tau)
            g = np.empty(len(tau))
            g[order] = _ou(tau[order], self.gp["sigma_s"], self.gp["ell_s"],
                           np.random.default_rng([rr, k, STREAM_GP, seed]))
            tau = tau - g
        if gp and self.gp.get("white_s"):
            tau = tau + np.random.default_rng([rr, k, STREAM_WHITE, seed]).standard_normal(len(t)) * self.gp["white_s"]
        if label_sd is not None:
            tau = tau + np.random.default_rng([rr, k, STREAM_LABEL, seed]).standard_normal(len(t)) * label_sd
        return tau

    def realisations(self, t, lead=None, k=None, label_sd=None, seed=0, gp=True):
        """Generator over realisations 0..k-1 (default all stored draws)."""
        for j in range(self.k if k is None else int(k)):
            yield self.realisation(j, t, lead, label_sd, seed, gp)

    def timing_labels(self):
        """Fitted fiducial times (the timing set; R apexes for MIT-BIH).

        Returns:
            dict(key=label sample index, t=recorded time s, sd=s, lead).
        """
        L = self.d["labels"]
        return {"key": np.asarray(L["key"]), "t": np.asarray(L["t_s"]), "sd": np.asarray(L["sd_s"]), "lead": L["lead"]}


def _ou(t, sigma, ell, rng):
    g = np.empty(len(t))
    if not len(t):
        return g
    g[0] = sigma * rng.standard_normal()
    a = np.exp(-np.diff(t) / ell)
    z = rng.standard_normal(len(t) - 1) * sigma * np.sqrt(1 - a ** 2)
    for i in range(1, len(t)):
        g[i] = a[i - 1] * g[i - 1] + z[i - 1]
    return g


class Release:
    """A loaded, checksum-verified release folder."""

    def __init__(self, folder, verify=True):
        self.folder = Path(folder)
        mf = self.folder / "manifest.json"
        if not mf.exists():
            raise ReleaseError(f"no manifest.json in {self.folder}")
        self.manifest = json.loads(mf.read_text())
        self.checksum = _sha256(mf)
        self._verify = verify
        self._cache = {}

    @property
    def records(self):
        """Record names listed in the manifest."""
        return sorted(f[:-5] for f in self.manifest["files"])

    def record(self, record):
        """The ``RecordCorrection`` of ``record`` (checked against the manifest on first read; cached)."""
        record = str(record)
        if record not in self._cache:
            name = f"{record}.json"
            if name not in self.manifest["files"]:
                raise ReleaseError(f"record {record} is not in this release")
            path = self.folder / name
            if self._verify and _sha256(path) != self.manifest["files"][name]:
                raise ReleaseError(f"{name} does not match its checksum in manifest.json")
            self._cache[record] = RecordCorrection(json.loads(path.read_text()), checksum=self.checksum)
        return self._cache[record]


def load_release(folder, verify=True):
    """Open a release folder; every record file is checked against the manifest's SHA-256 when read."""
    return Release(folder, verify)


def run(analysis, correction, t, lead=None, k=200, seed=0, label_sd=None, gp=True, coverage=0.95,
        interval="normal", spread=None, debias=False, adjust=True):
    """Run ``analysis(times) -> float or array`` on the original, the corrected and K realised times.

    Returns dict: ``original``, ``corrected``, ``change`` (corrected - original), ``draws`` (K results),
    ``sd`` of the draws (the standard uncertainty of the result), ``interval`` (the coverage interval),
    ``noise_bias`` (median of the draws - corrected), ``change_exceeds_spread`` (|change| larger than half
    the interval width; if False the change should not be interpreted),
    ``available_at_s`` (the record end: offline use only) and ``provenance`` (release checksum, record,
    seed, K, gp, label_sd).

    The coverage interval (``interval="normal"``, the default) is corrected +- z * sd, z = 1.96 for
    ``coverage=0.95``: with K = 200 trials the sample quantiles are too unstable for a 95 % interval
    (JCGM 101 7.2.2 asks for about 1e6 / (1 - p) trials), so the normal approximation of JCGM 101 7.2.3
    Note is used. For a result that is skewed (a power, a variance), analyse its logarithm and transform
    the interval back. ``interval="percentile"`` gives the draws' central ``coverage`` quantiles about their
    median, placed around the corrected result (the v0.1.0 behaviour; needs large K). ``spread`` (deprecated)
    = (lo, hi) percentiles, implies ``interval="percentile"``.

    Adjustment for the shift (``adjust=True``, the default; from v0.1.1): each trial adds error draws to
    times that already contain the unmodelled error, so a noise-sensitive result is shifted in the trials by
    s = median(trials) - corrected, and the corrected result itself by up to s. This uncorrected systematic
    effect is a Type B component, rectangular between corrected - s and corrected (JCGM 100, 4.3.7):
    ``estimate`` = corrected - s/2, ``u`` = sqrt(sd**2 + s**2 / 12); ``interval`` = estimate +- z * u (normal)
    (normal), or the trials' percentile interval moved to the estimate (percentile).
    ``adjust=False`` gives estimate = corrected and u = sd (the v0.1.0 behaviour).

    Why the interval is centred on the corrected result: each realisation adds a fresh error draw to times that already contain
    their own errors, so statistics that grow with noise (RMSSD, SDNN, band powers) come out biased upwards
    in the draws. Their spread measures the uncertainty; their centre does not. ``noise_bias`` reports the
    shift so it can be inspected.

    ``debias=True`` adds ``debiased`` = corrected - noise_bias (a one-step simulation-extrapolation, SIMEX,
    estimate of the error-free result) and ``debiased_note``. It is approximate, and while the release's
    unmodelled time-base term is conservative it over-corrects (noise-sensitive measures come out too low).
    Off by default.
    """
    t = np.asarray(t, float)
    orig = np.asarray(analysis(t), float)
    corr = np.asarray(analysis(correction.to_corrected(t, lead)), float)
    k = min(int(k), correction.k)
    draws = np.array([np.asarray(analysis(x), float)
                      for x in correction.realisations(t, lead, k, label_sd, seed, gp)])
    med = np.median(draws, axis=0)
    sd = draws.std(0, ddof=1)
    if spread is not None:
        interval = "percentile"
    else:
        spread = (50 - 50 * coverage, 50 + 50 * coverage)
    if interval == "normal":
        from scipy.stats import norm
        z = norm.ppf(0.5 + coverage / 2)
        lo, hi = None, None
    elif interval == "percentile":
        lo, hi = np.percentile(draws, spread, axis=0)
    else:
        raise ValueError("interval must be 'normal' or 'percentile'")
    s_shift = med - corr
    if adjust:
        est = corr - s_shift / 2
        u = np.sqrt(sd ** 2 + s_shift ** 2 / 12)
    else:
        est, u = corr, sd
    if interval == "normal":
        lo, hi = est - z * u, est + z * u
    else:
        lo, hi = est + lo - med, est + hi - med
    change = corr - orig
    extra = {}
    if debias:
        extra = {"debiased": corr - (med - corr),
                     "debiased_note": "approximate (one-step SIMEX); over-corrects while the unmodelled term is conservative"}
    return dict(**extra, original=orig, corrected=corr, change=change, draws=draws, sd=sd,
                estimate=est, u=u, interval=(lo, hi), noise_bias=med - corr,
                change_exceeds_spread=np.abs(est - orig) > (hi - lo) / 2,
                available_at_s=correction.available_at_s,
                provenance={"release_checksum": correction.checksum, "record": correction.record, "seed": seed, "k": k,
                                "gp": bool(gp), "label_sd": label_sd, "debias": bool(debias), "adjust": bool(adjust), "interval": interval,
                                "coverage": coverage if interval == "normal" else None,
                                "spread": tuple(spread) if interval == "percentile" else None})


def _example_record():
    """A small synthetic record file (for the examples and tests; not real data)."""
    rng = np.random.default_rng(0)
    draws = [{"P": 6.0 + rng.normal(0, 0.002), "warp_coef": list(np.array([0.03, -0.01, 0.002, 0.001]) +
                                                                rng.normal(0, 0.001, 4)),
                  "flutter": None, "skew_ms": -5.0 + rng.normal(0, 0.8)} for _ in range(30)]
    m = TimeMap([PeriodicWarp(6.0, [0.03, -0.01, 0.002, 0.001])], [0.0, 0.005])
    return {"record": "example", "map": m.to_dict(), "map_components": {"warp": True, "flutter": False, "skew": True},
                "reference_lead": 0, "available_at_s": 1805.6, "gp": {"sigma_s": 0.004, "ell_s": 2.0}, "draws": draws,
                "labels": {"key": [], "t_s": [], "sd_s": [], "lead": 0}}
