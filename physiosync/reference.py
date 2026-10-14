# SPDX-License-Identifier: MIT
"""Published pulse arrival times, to help choose the adopted mean PAT of :func:`fit_ecg_ppg_delay`.

The table (``data/pat_reference.csv``) lists studies with their acquisition hardware and how ECG and PPG
were clocked. Tier A: ECG and PPG sampled by one acquisition unit (shared clock); B: research
equipment, shared clock likely but not stated; C: clinical monitor or unknown (monitor outputs carry
large, unknown ECG-PPG delays and sawtooth artefacts; listed for context, not for adoption);
C-corrected: monitor data corrected for profiled delays (Ruffolo et al. 2025, children).

The adopted PAT must refer to the same PPG fiducial the user detects (foot, maximum slope and peak
differ by tens to hundreds of ms) and should suit the population (age, posture, anaesthesia).
"""

from __future__ import annotations

import csv
from importlib import resources

__all__ = ["pat_reference"]


def pat_reference(tiers: tuple[str, ...] | None = None) -> list[dict]:
    """Return the published pulse-arrival-time reference table.

    The table ships with the package (``data/pat_reference.csv``). Each row is one study or subgroup:
    population, acquisition hardware, how ECG and PPG were clocked, the ECG and PPG fiducials, the mean PAT
    and its spread. See the module docstring for the tiers.

    Args:
        tiers: Keep only rows of these tiers, e.g. ``("A", "B")``. ``None`` returns every row.

    Returns:
        One dict per row, keyed by the CSV header. All values are strings, as read from the CSV.

    Example:
        >>> rows = pat_reference(tiers=("A",))
        >>> "Block et al" in {r["study"].split(",")[0] for r in rows}
        True
    """
    with resources.files("physiosync").joinpath("data/pat_reference.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if tiers is None or r["tier"] in tiers]
