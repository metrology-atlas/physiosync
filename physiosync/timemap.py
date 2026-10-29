# SPDX-License-Identifier: MIT
"""Time maps: from recorded time to corrected time, for recordings whose time base was distorted.

A recording chain can stamp events at the wrong times, for example a tape played back at a slowly varying
speed and digitised as if the speed were constant. A ``TimeMap`` describes such a distortion as a smooth,
monotone map between

- **recorded time** ``t``: the time the recording assigns to an instant (for sampled data, ``n / fs`` for
  sample position ``n``), and
- **corrected time** ``tau``: the best estimate of when the instant happened.

The map is

    u = tau + slow(tau),    t = u + periodic(u) + lead_offset[lead]

where ``slow`` is the sum of the tabulated components (``TabulatedOffset``, for example a clock drift measured
from the mains phase), ``periodic`` the sum of the others (``PeriodicWarp``, ``DriftingPeriodic``) and
``lead_offset`` a constant per signal (channel skew; zero for the reference lead). Sample values are never
changed: the map re-times events and samples, nothing else.

The map is fitted elsewhere (for MIT-BIH, by the companion package ``mitbih-chain``) and stored as plain JSON
(``TimeMap.to_dict`` / ``TimeMap.from_dict``), so using a correction needs only this module and NumPy. Moved
from ``adaptivepeaks.timemap`` (v0.1.2 of the MIT-BIH release) without changing any computation.

Examples:
    >>> import numpy as np
    >>> m = TimeMap([PeriodicWarp(period_s=6.0, coef=[0.03, 0.0])], lead_offsets_s=[0.0, -0.01])
    >>> tau = m.to_corrected(np.array([1.5, 10.0]))
    >>> bool(np.allclose(m.to_recorded(tau), [1.5, 10.0]))
    True
    >>> round(m.max_slope(0.0, 60.0), 4)        # |d offset / d tau|, must stay below 1
    0.0314
"""

import json
from dataclasses import dataclass, field

import numpy as np

__all__ = ["COMPONENTS", "DriftingPeriodic", "PeriodicWarp", "TabulatedOffset", "TimeMap"]


def _harmonics(phase, coef):
    """sum_h a_h sin(h phase) + b_h cos(h phase), coef = [a_1, b_1, a_2, b_2, ...]."""
    coef = np.asarray(coef, float)
    out = np.zeros(np.shape(phase))
    for h in range(1, len(coef) // 2 + 1):
        out = out + coef[2 * h - 2] * np.sin(h * phase) + coef[2 * h - 1] * np.cos(h * phase)
    return out


def _harmonics_deriv(phase, dphase, coef):
    coef = np.asarray(coef, float)
    out = np.zeros(np.shape(phase))
    for h in range(1, len(coef) // 2 + 1):
        out = out + h * dphase * (coef[2 * h - 2] * np.cos(h * phase) - coef[2 * h - 1] * np.sin(h * phase))
    return out


@dataclass(frozen=True)
class PeriodicWarp:
    """A strictly periodic time error: offset(tau) = sum_h a_h sin(2 pi h tau / P) + b_h cos(2 pi h tau / P).

    ``coef`` = [a_1, b_1, a_2, b_2, ...] in seconds; phase is referenced to tau = 0.
    """
    period_s: float
    coef: tuple
    kind: str = field(default="periodic", init=False)

    def __post_init__(self):
        object.__setattr__(self, "coef", tuple(float(c) for c in self.coef))
        if len(self.coef) % 2 or self.period_s <= 0:
            raise ValueError("coef needs sin/cos pairs and period_s must be positive")

    def phase(self, tau):
        """Phase (radians) at corrected time ``tau``."""
        return 2 * np.pi * np.asarray(tau, float) / self.period_s

    def __call__(self, tau):
        """Offset (s) at corrected time ``tau``."""
        return _harmonics(self.phase(tau), self.coef)

    def deriv(self, tau):
        """Derivative of the offset with respect to ``tau``."""
        return _harmonics_deriv(self.phase(tau), 2 * np.pi / self.period_s, self.coef)

    def to_dict(self):
        """Plain-JSON form (``TimeMap.from_dict`` reads it)."""
        return {"kind": self.kind, "period_s": self.period_s, "coef": list(self.coef)}


@dataclass(frozen=True)
class DriftingPeriodic:
    """A periodic time error whose period drifts linearly: P(tau) = p0 + p1 (tau - tc).

    Phase = (2 pi / p1) ln(P(tau) / p0) (2 pi (tau - tc) / p0 when p1 = 0), so the instantaneous period is
    P(tau). ``coef`` as for ``PeriodicWarp``. Used for reel-like flutter.
    """
    p0_s: float
    p1: float
    tc_s: float
    coef: tuple
    kind: str = field(default="drifting", init=False)

    def __post_init__(self):
        object.__setattr__(self, "coef", tuple(float(c) for c in self.coef))
        if len(self.coef) % 2 or self.p0_s <= 0:
            raise ValueError("coef needs sin/cos pairs and p0_s must be positive")

    def period(self, tau):
        """Instantaneous period (s) at corrected time ``tau``."""
        return self.p0_s + self.p1 * (np.asarray(tau, float) - self.tc_s)

    def phase(self, tau):
        """Phase (radians) at corrected time ``tau``."""
        u = np.asarray(tau, float) - self.tc_s
        if self.p1 == 0:
            return 2 * np.pi * u / self.p0_s
        P = self.period(tau)
        if np.any(P <= 0):
            raise ValueError("period reaches zero inside the requested range")
        return 2 * np.pi / self.p1 * np.log(P / self.p0_s)

    def __call__(self, tau):
        """Offset (s) at corrected time ``tau``."""
        return _harmonics(self.phase(tau), self.coef)

    def deriv(self, tau):
        """Derivative of the offset with respect to ``tau``."""
        return _harmonics_deriv(self.phase(tau), 2 * np.pi / self.period(tau), self.coef)

    def to_dict(self):
        """Plain-JSON form (``TimeMap.from_dict`` reads it)."""
        return {"kind": self.kind, "p0_s": self.p0_s, "p1": self.p1, "tc_s": self.tc_s, "coef": list(self.coef)}


@dataclass(frozen=True)
class TabulatedOffset:
    """A slowly varying time error tabulated on a grid of corrected times.

    Linear interpolation, held constant beyond the ends: offset(tau) = scale * interp(tau, t_s, offset_s). Used
    for a clock drift, for
    example the playback clock of a tape deck measured from the mains phase. ``scale`` lets a Monte Carlo draw
    rescale the whole table.
    """
    t_s: tuple
    offset_s: tuple
    scale: float = 1.0
    kind: str = field(default="tabulated", init=False)

    def __post_init__(self):
        object.__setattr__(self, "t_s", tuple(float(x) for x in self.t_s))
        object.__setattr__(self, "offset_s", tuple(float(x) for x in self.offset_s))
        object.__setattr__(self, "scale", float(self.scale))
        if len(self.t_s) != len(self.offset_s) or len(self.t_s) < 2 or np.any(np.diff(self.t_s) <= 0):
            raise ValueError("t_s must increase and match offset_s in length (at least 2 points)")

    def __call__(self, tau):
        """Offset (s) at corrected time ``tau``."""
        return self.scale * np.interp(np.asarray(tau, float), self.t_s, self.offset_s)

    def deriv(self, tau):
        """Derivative of the offset with respect to ``tau`` (piecewise constant; zero beyond the ends)."""
        t = np.asarray(self.t_s); o = np.asarray(self.offset_s)
        slope = np.diff(o) / np.diff(t)
        i = np.clip(np.searchsorted(t, np.asarray(tau, float), side="right") - 1, 0, len(slope) - 1)
        inside = (np.asarray(tau, float) >= t[0]) & (np.asarray(tau, float) <= t[-1])
        return self.scale * np.where(inside, slope[i], 0.0)

    def to_dict(self):
        """Plain-JSON form (``TimeMap.from_dict`` reads it)."""
        d = {"kind": self.kind, "t_s": list(self.t_s), "offset_s": list(self.offset_s)}
        if self.scale != 1.0:
            d["scale"] = self.scale
        return d


COMPONENTS = {"periodic": PeriodicWarp, "drifting": DriftingPeriodic, "tabulated": TabulatedOffset}


class TimeMap:
    """Monotone map between recorded time ``t`` and corrected time ``tau`` (see the module docstring).

    Args:
        components: offset components (``PeriodicWarp``, ``DriftingPeriodic``, ``TabulatedOffset``); empty =
            identity.
        lead_offsets_s: constant extra offset per signal (index = signal number); default all zero.
        meta: free-form information carried with the map (record, release, time base label, ...).
    """

    def __init__(self, components=(), lead_offsets_s=None, meta=None):
        self.components = tuple(components)
        self.lead_offsets_s = tuple(float(x) for x in (lead_offsets_s or ()))
        self.meta = dict(meta or {})

    def _lead(self, lead):
        if lead is None:
            return 0.0
        if not self.lead_offsets_s:
            return 0.0
        return self.lead_offsets_s[int(lead)]

    def offset(self, tau):
        """Sum of the components at corrected time tau (seconds); excludes the lead offset."""
        tau = np.asarray(tau, float)
        out = np.zeros(tau.shape)
        for c in self.components:
            out = out + c(tau)
        return out

    def slope(self, tau):
        """D offset / d tau."""
        tau = np.asarray(tau, float)
        out = np.zeros(tau.shape)
        for c in self.components:
            out = out + c.deriv(tau)
        return out

    def _inner(self):
        return [c for c in self.components if getattr(c, "kind", "") != "tabulated"]

    def _outer(self):
        return [c for c in self.components if getattr(c, "kind", "") == "tabulated"]

    def to_recorded(self, tau, lead=None):
        """Recorded time of events at corrected times ``tau`` (on signal ``lead``; None = reference).

        Tabulated components (a slow clock drift) are applied outside the others:
        u = tau + drift(tau), t = u + sum(other components at u) + lead offset. This keeps the periodic components
        on the time axis they were fitted on (the stamped time, up to the slow drift), so the drift changes no
        interval by more than its own slope times the interval.
        """
        tau = np.asarray(tau, float)
        u = tau + sum((c(tau) for c in self._outer()), np.zeros(tau.shape))
        return u + sum((c(u) for c in self._inner()), np.zeros(u.shape)) + self._lead(lead)

    def to_corrected(self, t, lead=None, tol=1e-10, max_iter=50):
        """Corrected time of events recorded at ``t``: inverts ``to_recorded``.

        Fixed-point iterations (converge because |offset'| < 1; ``check_monotone`` verifies that): first the
        periodic components, t - lead = u + sum(c(u)); then the tabulated drift, u = tau + drift(tau).
        """
        y = np.asarray(t, float) - self._lead(lead)
        def solve(y, comps):
            if not comps:
                return y
            tau = y.copy()
            for _ in range(max_iter):
                new = y - sum((c(tau) for c in comps), np.zeros(tau.shape))
                if np.max(np.abs(new - tau), initial=0.0) < tol:
                    return new
                tau = new
            raise RuntimeError("time map inversion did not converge (is the map monotone?)")
        return solve(solve(y, self._inner()), self._outer())

    def max_slope(self, t0, t1, step_s=0.01):
        """Largest |d offset / d tau| on [t0, t1]."""
        grid = np.arange(t0, t1 + step_s, step_s)
        return float(np.max(np.abs(self.slope(grid)), initial=0.0))

    def check_monotone(self, t0, t1, step_s=0.01, limit=0.5):
        """Raise ValueError unless |d offset / d tau| < ``limit`` (< 1 is strictly monotone) on [t0, t1]."""
        s = self.max_slope(t0, t1, step_s)
        if not s < limit:
            raise ValueError(f"time map slope {s:.3f} exceeds {limit} on [{t0}, {t1}] s")
        return s

    def to_dict(self):
        """Plain-JSON form: components, lead offsets and meta."""
        return {"components": [c.to_dict() for c in self.components],
                "lead_offsets_s": list(self.lead_offsets_s), "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        """Build a map from ``to_dict`` output."""
        comps = []
        for c in d.get("components", []):
            c = dict(c)
            kind = c.pop("kind")
            comps.append(COMPONENTS[kind](**c))
        return cls(comps, d.get("lead_offsets_s"), d.get("meta"))

    def to_json(self, **kw):
        """JSON string of ``to_dict`` (keyword arguments go to ``json.dumps``)."""
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s):
        """Build a map from a ``to_json`` string."""
        return cls.from_dict(json.loads(s))

    def __repr__(self):
        kinds = ", ".join(c.kind for c in self.components) or "identity"
        return f"TimeMap({kinds}; lead offsets {self.lead_offsets_s})"
