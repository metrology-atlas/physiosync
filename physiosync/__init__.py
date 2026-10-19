# SPDX-License-Identifier: MIT
"""physiosync: timing corrections for multi-signal physiological recordings (clock drift, ECG-PPG delay).

Format-independent building blocks; connectors call them at import time.
"""

from .chains import CHAIN_FIELDS, chain_labels, chain_slips
from .ecg_ppg import (
    EcgPpgDelay,
    TierRules,
    detect_slips,
    fit_ecg_ppg_delay,
    match_beats,
    sawtooth_evidence,
    setup_offsets,
)
from .frames import change_spectrum, frame_clock, frame_lines, interval_lock, period_lock_scan
from .labels import LABEL_FIELDS, OFFSET_FIELDS, offset_label, sawtooth_label_tuples, sawtooth_labels
from .measurement import error_samples, fiducial_error_model, pat_measurement_model, simulate_pat
from .network import Adjustment, ClockNetwork, Cycle, Edge
from .pairing import pair_beats
from .quantisation import (
    ResolutionResult,
    fiducial_resolution,
    indices_from_times,
    intervals,
    locking,
    period_scan,
    pool,
)
from .reference import pat_reference
from .stamps import corrected_times, stamp_clock

__all__ = ["CHAIN_FIELDS", "LABEL_FIELDS", "OFFSET_FIELDS", "Adjustment", "ClockNetwork", "Cycle", "EcgPpgDelay", "Edge", "ResolutionResult", "TierRules", "chain_labels", "chain_slips", "change_spectrum", "corrected_times", "detect_slips", "error_samples", "fiducial_error_model", "fiducial_resolution", "fit_ecg_ppg_delay", "frame_clock", "frame_lines", "indices_from_times", "interval_lock", "intervals", "locking", "match_beats", "offset_label", "pair_beats", "pat_measurement_model", "pat_reference", "period_lock_scan", "period_scan", "pool", "sawtooth_evidence", "sawtooth_label_tuples", "sawtooth_labels", "setup_offsets", "simulate_pat", "stamp_clock"]
__version__ = "0.0.1"
