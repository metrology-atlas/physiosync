# SPDX-License-Identifier: MIT
import doctest
import hashlib
import json

import numpy as np
import pytest

import physiosync.release as rl
from physiosync import RecordCorrection, ReleaseError, TabulatedOffset, TimeMap, load_release


def test_doctests():
    assert doctest.testmod(rl).failed == 0


def _record():
    d = rl._example_record()
    m = TimeMap.from_dict(d["map"])
    d["map"] = TimeMap(list(m.components) + [TabulatedOffset([0.0, 900.0, 1805.6], [0.0, 0.08, 0.15])],
                       m.lead_offsets_s).to_dict()
    for k, dr in enumerate(d["draws"]):
        dr["drift_scale"] = 1.0 + 0.1 * (k % 3 - 1)
    d["gp"]["white_s"] = 0.0005
    return d


def _write_release(folder, d):
    folder.mkdir()
    name = f"{d['record']}.json"
    (folder / name).write_text(json.dumps(d))
    sha = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps({"files": {name: sha}}))
    return folder


def test_release_reads_and_verifies(tmp_path):
    rel = load_release(_write_release(tmp_path / "rel", _record()))
    assert rel.records == ["example"]
    rc = rel.record("example")
    assert rc.k == 30 and rc.checksum == rel.checksum
    with pytest.raises(ReleaseError):
        rel.record("missing")


def test_tampered_file_is_refused(tmp_path):
    folder = _write_release(tmp_path / "rel", _record())
    (folder / "example.json").write_text(json.dumps(_record()).replace("1805.6", "1805.7"))
    with pytest.raises(ReleaseError):
        load_release(folder).record("example")
    assert load_release(folder, verify=False).record("example").available_at_s == 1805.7
    with pytest.raises(ReleaseError):
        load_release(tmp_path)


def test_realisations_reproducible_and_seeded():
    rc = RecordCorrection.from_dict(_record())
    t = np.arange(5.0, 300.0, 0.8)
    a = rc.realisation(3, t, lead=1, label_sd=0.001, seed=7)
    assert np.array_equal(a, rc.realisation(3, t, lead=1, label_sd=0.001, seed=7))
    assert not np.array_equal(a, rc.realisation(3, t, lead=1, label_sd=0.001, seed=8))
    assert not np.array_equal(a, rc.realisation(4, t, lead=1, label_sd=0.001, seed=7))
    assert len(list(rc.realisations(t, k=5))) == 5


def test_draw_map_carries_drift_scale_and_skew():
    d = _record()
    rc = RecordCorrection.from_dict(d)
    m = rc.draw_map(0)
    tab = [c for c in m.components if isinstance(c, TabulatedOffset)]
    assert len(tab) == 1 and tab[0].scale == d["draws"][0]["drift_scale"]
    assert m.lead_offsets_s[1] == pytest.approx(-d["draws"][0]["skew_ms"] / 1000.0)   # reference lead 0


def test_run_adjusted_estimate():
    rc = RecordCorrection.from_dict(_record())
    t = np.arange(10.0, 600.0, 0.8)
    res = rl.run(lambda x: float(np.std(np.diff(x))) * 1000, rc, t, lead=0, k=30, seed=1)
    s = res["noise_bias"]
    assert res["estimate"] == pytest.approx(res["corrected"] - s / 2)
    assert res["u"] == pytest.approx(np.sqrt(res["sd"] ** 2 + s ** 2 / 12))
    lo, hi = res["interval"]
    assert (lo + hi) / 2 == pytest.approx(res["estimate"])
    assert res["provenance"]["k"] == 30 and res["available_at_s"] == 1805.6
    raw = rl.run(lambda x: float(np.std(np.diff(x))) * 1000, rc, t, lead=0, k=30, seed=1, adjust=False)
    assert raw["estimate"] == raw["corrected"] and raw["u"] == raw["sd"]
