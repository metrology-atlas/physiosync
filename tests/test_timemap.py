# SPDX-License-Identifier: MIT
import doctest

import numpy as np
import pytest

import physiosync.timemap as tm
from physiosync import DriftingPeriodic, PeriodicWarp, TabulatedOffset, TimeMap


def test_doctests():
    assert doctest.testmod(tm).failed == 0


def _map():
    return TimeMap([PeriodicWarp(6.0, [0.03, -0.01, 0.002, 0.001]),
                    DriftingPeriodic(11.0, 1e-4, 900.0, [0.008, 0.002]),
                    TabulatedOffset([0.0, 600.0, 1200.0, 1800.0], [0.0, 0.05, 0.12, 0.2])],
                   lead_offsets_s=[0.0, 0.017])


@pytest.mark.parametrize("lead", [None, 0, 1])
def test_round_trip(lead):
    m = _map()
    t = np.linspace(1.0, 1790.0, 2001)
    tau = m.to_corrected(t, lead)
    assert np.max(np.abs(m.to_recorded(tau, lead) - t)) < 1e-9
    assert np.all(np.diff(tau) > 0)


def test_tabulated_applied_outside():
    # the periodic components act at u = tau + drift(tau), not at tau
    w, d = PeriodicWarp(6.0, [0.03, 0.0]), TabulatedOffset([0.0, 100.0], [0.0, 1.0])
    m = TimeMap([w, d])
    tau = np.array([10.0, 50.0])
    u = tau + d(tau)
    assert np.allclose(m.to_recorded(tau), u + w(u))


def test_lead_offset_shifts_only():
    m = _map()
    t = np.array([100.0, 200.0])
    assert np.allclose(m.to_corrected(t + 0.017, 1), m.to_corrected(t, 0))


def test_dict_and_json_round_trip():
    m = _map()
    m2 = TimeMap.from_json(m.to_json())
    t = np.linspace(0.0, 1800.0, 50)
    assert np.array_equal(m2.to_corrected(t, 1), m.to_corrected(t, 1))
    assert TabulatedOffset([0.0, 1.0], [0.0, 1.0], 1.2).to_dict()["scale"] == 1.2


def test_derivatives_match_finite_differences():
    for c in _map().components:
        x = np.linspace(5.0, 1700.0, 400)
        h = 1e-5
        fd = (c(x + h) - c(x - h)) / (2 * h)
        assert np.allclose(c.deriv(x), fd, atol=1e-6)


def test_check_monotone_and_errors():
    assert _map().check_monotone(0.0, 1800.0) < 0.5
    with pytest.raises(ValueError):
        TimeMap([PeriodicWarp(1.0, [0.2, 0.0])]).check_monotone(0.0, 10.0)
    with pytest.raises(ValueError):
        PeriodicWarp(6.0, [0.1])
    with pytest.raises(ValueError):
        TabulatedOffset([0.0, 0.0], [0.0, 1.0])
