# SPDX-License-Identifier: MIT
from physiosync import pat_reference


def test_reference_table_loads_and_filters():
    rows = pat_reference()
    assert len(rows) >= 10
    a = pat_reference(("A",))
    assert a and all(r["tier"] == "A" for r in a)
    assert any("Block" in r["study"] for r in a)
