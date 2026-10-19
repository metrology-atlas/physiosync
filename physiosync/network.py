# SPDX-License-Identifier: MIT
"""Reconciling a network of clock links: loop misclosures and a least-squares adjustment.

Clocks are nodes; every measurement that relates two clocks is an edge. An edge from ``a`` to ``b`` measures
``x[a] - x[b]`` for some quantity ``x`` defined per clock against a common reference: a drift rate (ppm), an offset
(s), or anything else that adds along a path. Where measurements form loops, the values around each loop must sum
to zero; the amount they miss by (the misclosure) tests the links in it. With redundancy, all edges can be adjusted
together by weighted least squares (as in survey levelling networks or clock ensembles), giving each clock's value
against the reference and a normalised residual per edge.

Nothing here knows about any dataset or device. Each edge carries the data it was derived from (``sources``), so a
loop whose edges share data can be flagged: such a loop may close by construction and then tests nothing.

Example:
    >>> net = ClockNetwork([
    ...     Edge("bis", "pc", 25.0, 1.0, "correction steps", {"bis stamps"}),
    ...     Edge("snuadc", "pc", -63.0, 0.5, "resets", {"snuadc stamps"}),
    ...     Edge("bis", "snuadc", 87.0, 2.0, "bursts", {"ecg", "eeg"}),
    ... ], reference="pc")
    >>> [round(c.misclosure, 1) for c in net.cycles()]
    [-1.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Adjustment", "ClockNetwork", "Cycle", "Edge"]


@dataclass(frozen=True)
class Edge:
    """One measured link between two clocks.

    Attributes:
        a: The first clock.
        b: The second clock.
        value: The measured ``x[a] - x[b]``.
        se: Its standard error (same unit), greater than zero.
        method: How it was measured; adjustments estimate one variance factor per method.
        sources: The data it was derived from (names); edges sharing a source are not independent.
        label: Optional free text (for example an interval or a recording).
    """

    a: str
    b: str
    value: float
    se: float
    method: str
    sources: frozenset = field(default_factory=frozenset)
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "sources", frozenset(self.sources))
        if not (np.isfinite(self.value) and np.isfinite(self.se) and self.se > 0):
            raise ValueError(f"edge {self.a}-{self.b} ({self.method}) needs a finite value and a positive se")
        if self.a == self.b:
            raise ValueError("an edge must join two different clocks")


@dataclass
class Cycle:
    """One independent loop of the network.

    Attributes:
        edges: Indices of the edges in the loop.
        signs: +1 where the loop runs along the edge (a to b), -1 against it.
        misclosure: The signed sum of the edge values around the loop (zero if they agree).
        se: Its standard error, from the edges' standard errors (taken as independent).
        z: Misclosure divided by its standard error.
        shared_sources: Sources used by more than one edge of the loop; if not empty, the loop is not fully
            independent and may close by construction.
        methods: The methods of its edges, in loop order.
    """

    edges: list
    signs: list
    misclosure: float
    se: float
    z: float
    shared_sources: frozenset
    methods: list


@dataclass
class Adjustment:
    """The least-squares adjustment of a network.

    Attributes:
        values: Each clock's adjusted value against the reference (the reference is 0).
        se: Their standard errors (scaled by the overall variance factor when the network has redundancy).
        residuals: Per edge, measured minus adjusted ``x[a] - x[b]``.
        normalised: Per edge, the residual divided by its own standard error (Baarda's w test); about N(0, 1)
            for an edge that agrees with the rest.
        chi2: Sum of squared weighted residuals.
        dof: Redundancy: edges minus unknowns.
        variance_factor: chi2 / dof (about 1 when the standard errors are right); nan without redundancy.
        method_factors: The same per method, from the residuals of that method's edges.
        method_chi2: Per method, the sum of its squared weighted residuals (to pool over many networks).
        method_redundancy: Per method, its share of the redundancy (the method factor is chi2 / redundancy).
    """

    values: dict
    se: dict
    residuals: np.ndarray
    normalised: np.ndarray
    chi2: float
    dof: int
    variance_factor: float
    method_factors: dict
    method_chi2: dict = field(default_factory=dict)
    method_redundancy: dict = field(default_factory=dict)


class ClockNetwork:
    """A set of clocks joined by measured edges.

    Args:
        edges: The measurements.
        reference: The clock held at 0 (for example the recorder's PC clock). Must appear in an edge.
    """

    def __init__(self, edges, reference: str):
        self.edges = list(edges)
        nodes = sorted({e.a for e in self.edges} | {e.b for e in self.edges})
        if reference not in nodes:
            raise ValueError(f"reference {reference!r} is not in any edge")
        self.reference = reference
        self.nodes = nodes

    def components(self) -> list:
        """Connected groups of clocks (only the one holding the reference can be adjusted)."""
        parent = {n: n for n in self.nodes}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for e in self.edges:
            parent[find(e.a)] = find(e.b)
        groups: dict = {}
        for n in self.nodes:
            groups.setdefault(find(n), []).append(n)
        return list(groups.values())

    def cycles(self) -> list:
        """A fundamental set of independent loops (one per edge outside a spanning tree).

        Returns:
            One :class:`Cycle` per independent loop; parallel edges between the same two clocks form loops of two.
        """
        adj: dict = {n: [] for n in self.nodes}
        for i, e in enumerate(self.edges):
            adj[e.a].append((i, e.b, +1))
            adj[e.b].append((i, e.a, -1))
        parent: dict = {}
        tree = set()
        for root in self.nodes:
            if root in parent:
                continue
            parent[root] = (None, None, 0)
            stack = [root]
            while stack:
                n = stack.pop()
                for i, m, s in adj[n]:
                    if m not in parent:
                        parent[m] = (n, i, s)
                        tree.add(i)
                        stack.append(m)

        def path_to_root(n):
            out = []
            while parent[n][0] is not None:
                p, i, s = parent[n]
                out.append((i, s, n))
                n = p
            return out

        cycles = []
        for i, e in enumerate(self.edges):
            if i in tree:
                continue
            # loop: along edge i from a to b, then from b back to a through the tree
            pa, pb = path_to_root(e.a), path_to_root(e.b)
            # trim the common part above the lowest common ancestor
            while pa and pb and pa[-1][0] == pb[-1][0]:
                pa.pop()
                pb.pop()
            idx, sg = [i], [1]
            # from b up to the ancestor: edge (parent->child) taken child->parent, i.e. against its tree direction
            for j, s, _ in pb:
                idx.append(j)
                sg.append(-s)
            # from the ancestor down to a
            for j, s, _ in reversed(pa):
                idx.append(j)
                sg.append(s)
            # tree sign s = +1 when the edge is stored parent->child: walking child->parent contributes -value,
            # walking parent->child +value
            vals = np.array([self.edges[j].value for j in idx]) * np.array(sg)
            ses = np.array([self.edges[j].se for j in idx])
            mis = float(vals.sum())
            se = float(np.sqrt(np.sum(ses ** 2)))
            seen: dict = {}
            for j in idx:
                for s_ in self.edges[j].sources:
                    seen[s_] = seen.get(s_, 0) + 1
            shared = frozenset(k for k, v in seen.items() if v > 1)
            cycles.append(Cycle(edges=idx, signs=sg, misclosure=mis, se=se, z=mis / se, shared_sources=shared,
                                methods=[self.edges[j].method for j in idx]))
        return cycles

    def adjust(self) -> Adjustment:
        """Weighted least squares with the reference held at 0.

        Returns:
            The :class:`Adjustment`. Clocks not connected to the reference are left out.
        """
        comp = next(c for c in self.components() if self.reference in c)
        unknowns = [n for n in comp if n != self.reference]
        col = {n: k for k, n in enumerate(unknowns)}
        use = [i for i, e in enumerate(self.edges) if e.a in col or e.b in col or {e.a, e.b} <= set(comp)]
        A = np.zeros((len(use), len(unknowns)))
        y = np.array([self.edges[i].value for i in use], float)
        s = np.array([self.edges[i].se for i in use], float)
        for r, i in enumerate(use):
            e = self.edges[i]
            if e.a in col:
                A[r, col[e.a]] += 1.0
            if e.b in col:
                A[r, col[e.b]] -= 1.0
        Wh = 1.0 / s
        Aw, yw = A * Wh[:, None], y * Wh
        N = Aw.T @ Aw
        Ninv = np.linalg.pinv(N)
        x = Ninv @ Aw.T @ yw
        res = y - A @ x
        dof = len(use) - np.linalg.matrix_rank(N)
        chi2 = float(np.sum((res * Wh) ** 2))
        vf = chi2 / dof if dof > 0 else float("nan")
        # residual covariance (unit variance factor): Q_r = Q_y - A N^-1 A^T
        Qr = np.diag(s ** 2) - A @ Ninv @ A.T
        qr = np.clip(np.diag(Qr), 0, None)
        norm = np.where(qr > 1e-12 * s ** 2, res / np.sqrt(np.where(qr > 0, qr, 1)), np.nan)
        scale = np.sqrt(vf) if dof > 0 else 1.0
        values = {self.reference: 0.0, **{n: float(x[k]) for n, k in col.items()}}
        ses = {self.reference: 0.0, **{n: float(np.sqrt(Ninv[k, k]) * scale) for n, k in col.items()}}
        mf, mc, mr = {}, {}, {}
        for m in {self.edges[i].method for i in use}:
            rows = [r for r, i in enumerate(use) if self.edges[i].method == m]
            mr[m] = float(np.sum([qr[r] / s[r] ** 2 for r in rows]))      # this method's share of the redundancy
            mc[m] = float(np.sum((res[rows] / s[rows]) ** 2))
            mf[m] = mc[m] / mr[m] if mr[m] > 1e-9 else float("nan")
        full_res = np.full(len(self.edges), np.nan)
        full_norm = np.full(len(self.edges), np.nan)
        full_res[use], full_norm[use] = res, norm
        return Adjustment(values=values, se=ses, residuals=full_res, normalised=full_norm, chi2=chi2, dof=int(dof),
                          variance_factor=vf, method_factors=mf, method_chi2=mc, method_redundancy=mr)
