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
    """Rows of the reference table, optionally only the given tiers (e.g. ("A", "B"))."""
    with resources.files("physiosync").joinpath("data/pat_reference.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if tiers is None or r["tier"] in tiers]
