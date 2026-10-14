# SPDX-License-Identifier: MIT
"""Run every Python example in docs/ so the documentation cannot drift from the code."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted(p for p in DOCS.rglob("*.md") if "```python" in p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_examples_run(page):
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), flags=re.DOTALL)
    ns: dict = {}
    for code in blocks:
        if code.lstrip().startswith("# requires"):
            continue                      # needs an optional dependency
        exec(compile(code, str(page), "exec"), ns)  # noqa: S102 - our own documentation
