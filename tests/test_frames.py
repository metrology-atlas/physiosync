# SPDX-License-Identifier: MIT
import numpy as np
from scipy import signal

from physiosync import change_spectrum, frame_clock, frame_lines, interval_lock

FS = 500.0


def pulse(n, seed):
    rng = np.random.default_rng(seed)
    # heart rate wandering randomly around 72 bpm
    hr = 1.2 + 0.15 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n) + 0.05 * np.sin(np.arange(n) / FS / 7)
    ph = 2 * np.pi * np.cumsum(hr) / FS
    return np.sin(ph) + 0.5 * np.sin(2 * ph + 0.6) + 0.2 * np.sin(3 * ph + 1.1)


def framed(minutes=10, frame_hz=30.0, ppm=40.0, seed=0, frames=True):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    t = np.arange(n) / FS
    x = pulse(n, seed)
    if frames:
        # the waveform is produced in frames (each value held for a frame), smoothed by the analogue
        # output, then sampled
        fr = frame_hz / (1 + ppm * 1e-6)
        held = np.minimum(np.round(np.floor(t * fr) / fr * FS).astype(int), n - 1)
        x = signal.sosfilt(signal.butter(2, 25, fs=FS, output="sos"), x[held])
    return np.round((x + 0.002 * rng.standard_normal(n)) * 2000)  # integer ADC counts


def peaks(x):
    pk, _ = signal.find_peaks(x, distance=int(0.4 * FS))
    return pk


def test_frame_rate_found_in_waveform():
    lines = frame_lines(framed(), FS)
    assert lines and abs(lines[0]["hz"] - 30.0) < 0.2 or any(abs(ln["hz"] - 30) < 0.2 for ln in lines)
    assert not any(abs(ln["hz"] - 30) < 0.2 for ln in frame_lines(framed(frames=False), FS))


def test_spectrum_shape():
    f, r = change_spectrum(framed(), FS)
    assert f.size == r.size and r[np.argmin(np.abs(f - 30))] > 3


def test_fiducials_lock_and_frame_clock_found():
    x = framed(minutes=20, ppm=40.0)
    pk = peaks(x)
    lock = interval_lock(pk, FS / 30)
    assert lock["ratio"] > 1.5 and lock["resultant"] > 0.3
    fc = frame_clock(pk, FS / 30, span_ppm=200)
    assert abs(fc["ppm"] - 40) <= 3 and fc["sharpness"] > 5
    assert abs(fc["ppm_halves"][0] - fc["ppm_halves"][1]) <= 6


def test_no_lock_without_frames():
    pk = peaks(framed(minutes=20, frames=False))
    lock = interval_lock(pk, FS / 30)
    assert lock["ratio"] < 1.2 and lock["resultant"] < 0.05
    assert frame_clock(pk, FS / 30, span_ppm=200)["sharpness"] < 4
