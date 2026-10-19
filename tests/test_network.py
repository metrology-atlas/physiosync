import numpy as np
import pytest

from physiosync.network import ClockNetwork, Edge


def truth_edges(rng, x, pairs, se, method="m", noise=True):
    out = []
    for k, (a, b) in enumerate(pairs):
        v = x[a] - x[b] + (rng.normal(0, se) if noise else 0.0)
        out.append(Edge(a, b, v, se, method, {f"src{k}"}))
    return out


def test_triangle_closes_and_signs():
    x = {"pc": 0.0, "bis": 25.0, "snu": -63.0}
    e = truth_edges(None, x, [("bis", "pc"), ("snu", "pc"), ("bis", "snu")], 1.0, noise=False)
    cyc = ClockNetwork(e, "pc").cycles()
    assert len(cyc) == 1 and abs(cyc[0].misclosure) < 1e-12
    # reversing an edge's direction (and its value) changes nothing
    e2 = e[:2] + [Edge("snu", "bis", -88.0, 1.0, "m", {"src2"})]
    assert abs(ClockNetwork(e2, "pc").cycles()[0].misclosure) < 1e-12
    e3 = e[:2] + [Edge("bis", "snu", 90.0, 1.0, "m", {"src2"})]
    assert abs(abs(ClockNetwork(e3, "pc").cycles()[0].misclosure) - 2.0) < 1e-12


def test_parallel_edges_form_a_loop():
    e = [Edge("primus", "pc", 104.0, 5.0, "breath", {"awp", "cvp"}), Edge("primus", "pc", 100.0, 10.0, "pip", {"awp", "pip"})]
    c = ClockNetwork(e, "pc").cycles()
    assert len(c) == 1 and abs(abs(c[0].misclosure) - 4.0) < 1e-12
    assert c[0].shared_sources == frozenset({"awp"})
    assert c[0].se == pytest.approx(np.hypot(5, 10))


def test_cycle_count_is_edges_minus_nodes_plus_components():
    rng = np.random.default_rng(0)
    x = {n: rng.normal(0, 50) for n in "pabcd"}
    x["p"] = 0.0
    pairs = [("a", "p"), ("b", "p"), ("a", "b"), ("c", "a"), ("c", "b"), ("d", "c"), ("d", "p")]
    net = ClockNetwork(truth_edges(rng, x, pairs, 1.0, noise=False), "p")
    cyc = net.cycles()
    assert len(cyc) == len(pairs) - 5 + 1
    assert all(abs(c.misclosure) < 1e-9 for c in cyc)


def test_adjustment_recovers_values_and_variance_factor():
    rng = np.random.default_rng(1)
    x = {"p": 0.0, "a": 30.0, "b": -60.0, "c": 100.0}
    pairs = [("a", "p"), ("b", "p"), ("a", "b"), ("c", "p"), ("c", "a")] * 40
    adj = ClockNetwork(truth_edges(rng, x, pairs, 2.0), "p").adjust()
    for n, v in x.items():
        assert adj.values[n] == pytest.approx(v, abs=1.0)
    assert 0.7 < adj.variance_factor < 1.4
    assert adj.dof == len(pairs) - 3


def test_normalised_residual_flags_a_bad_edge():
    rng = np.random.default_rng(2)
    x = {"p": 0.0, "a": 30.0, "b": -60.0}
    e = truth_edges(rng, x, [("a", "p"), ("b", "p"), ("a", "b")] * 5, 1.0)
    e[4] = Edge(e[4].a, e[4].b, e[4].value + 15.0, 1.0, "bad", {"x"})
    adj = ClockNetwork(e, "p").adjust()
    assert np.nanargmax(np.abs(adj.normalised)) == 4
    assert abs(adj.normalised[4]) > 5


def test_disconnected_clock_is_left_out():
    e = [Edge("a", "p", 1.0, 1.0, "m"), Edge("q", "r", 2.0, 1.0, "m")]
    adj = ClockNetwork(e, "p").adjust()
    assert set(adj.values) == {"p", "a"} and np.isnan(adj.residuals[1])


def test_bad_edges_rejected():
    with pytest.raises(ValueError):
        Edge("a", "a", 1.0, 1.0, "m")
    with pytest.raises(ValueError):
        Edge("a", "b", 1.0, 0.0, "m")
