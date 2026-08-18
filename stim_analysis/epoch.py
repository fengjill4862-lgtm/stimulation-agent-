"""Epoch -> blank -> filter, in that order (spec section 3).

The continuous trace is never filtered. Epochs are cut with generous padding,
the artifact window is blanked (linear interpolation) per epoch, each epoch is
filtered on its own, and the padding is trimmed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import signal

from plot_rhs_power_analysis import _blank_and_interpolate, _merge_windows
from stim_analysis.config import AnalysisConfig


# -----------------------------------------------------------------------------
# Epoch extraction
# -----------------------------------------------------------------------------


@dataclass
class EpochSet:
    """Padded raw epochs for one run: ``raw`` is (channels, events, samples)."""

    run_id: str
    channels: list[str]
    sample_rate_hz: float
    event_numbers: np.ndarray  # (E,)
    onset_samples: np.ndarray  # (E,)
    raw: np.ndarray  # (C, E, S) float32, padded
    t_ms: np.ndarray  # (S,) time relative to onset, padded range
    core: slice  # slice of the un-padded epoch inside S
    kept: np.ndarray  # (E,) bool: core epoch fully inside the recording, no gap
    pad_reflected: np.ndarray  # (E,) bool: padding hit a recording edge

    @property
    def n_events(self) -> int:
        return int(self.event_numbers.size)

    @property
    def core_t_ms(self) -> np.ndarray:
        return self.t_ms[self.core]

    def channel(self, name: str) -> np.ndarray:
        return self.raw[self.channels.index(name)]


def _sample_offsets(cfg: AnalysisConfig, sample_rate_hz: float) -> tuple[int, int, int, int]:
    """(padded start, core start, core end, padded end) offsets in samples."""
    to_samples = lambda ms: int(round(ms * 1.0e-3 * sample_rate_hz))  # noqa: E731
    pad = to_samples(cfg.filter_pad_ms)
    core_start = to_samples(cfg.epoch_ms[0])
    core_end = to_samples(cfg.epoch_ms[1])
    return core_start - pad, core_start, core_end, core_end + pad


def gap_starts(timestamps: np.ndarray) -> np.ndarray:
    """Sample indices where the RHS timestamp counter jumps (new segment starts)."""
    if timestamps.size < 2:
        return np.zeros(0, dtype=np.int64)
    return (np.flatnonzero(np.diff(timestamps) != 1) + 1).astype(np.int64)


def extract_epochs(
    raw_uV: np.ndarray,
    sample_rate_hz: float,
    onset_samples: Sequence[int] | np.ndarray,
    event_numbers: Sequence[int] | np.ndarray,
    cfg: AnalysisConfig,
    *,
    run_id: str = "",
    channels: Sequence[str] | None = None,
    gap_sample_starts: np.ndarray | None = None,
) -> EpochSet:
    """Cut padded epochs from a (channels, samples) array around each onset.

    An epoch is dropped (``kept=False``) when its un-padded core leaves the
    recording or contains a timestamp discontinuity. When only the padding
    leaves the recording, the padding is reflect-padded and the epoch kept.
    """
    raw_uV = np.asarray(raw_uV)
    if raw_uV.ndim == 1:
        raw_uV = raw_uV[np.newaxis, :]
    n_channels, n_samples = raw_uV.shape
    channel_names = list(channels) if channels is not None else [f"ch{i}" for i in range(n_channels)]
    onsets = np.asarray(onset_samples, dtype=np.int64)
    numbers = np.asarray(event_numbers, dtype=np.int64)
    if onsets.size != numbers.size:
        raise ValueError("onset_samples and event_numbers must have the same length")

    start_off, core_start, core_end, end_off = _sample_offsets(cfg, sample_rate_hz)
    length = end_off - start_off
    core = slice(core_start - start_off, core_end - start_off)
    t_ms = (np.arange(length) + start_off) * 1.0e3 / sample_rate_hz

    epochs = np.zeros((n_channels, onsets.size, length), dtype=np.float32)
    kept = np.ones(onsets.size, dtype=bool)
    reflected = np.zeros(onsets.size, dtype=bool)
    gaps = np.asarray(gap_sample_starts, dtype=np.int64) if gap_sample_starts is not None else None

    for index, onset in enumerate(onsets):
        c0 = int(onset) + core_start
        c1 = int(onset) + core_end
        if c0 < 0 or c1 > n_samples:
            kept[index] = False
            continue
        if gaps is not None and gaps.size and np.any((gaps > c0) & (gaps < c1)):
            kept[index] = False
            continue
        p0 = int(onset) + start_off
        p1 = int(onset) + end_off
        if p0 >= 0 and p1 <= n_samples:
            epochs[:, index, :] = raw_uV[:, p0:p1]
            continue
        # Padding leaves the recording: reflect the available data into the pad.
        reflected[index] = True
        left = max(0, -p0)
        right = max(0, p1 - n_samples)
        segment = raw_uV[:, max(0, p0) : min(n_samples, p1)]
        if segment.shape[1] < 2:
            kept[index] = False
            continue
        padded = np.pad(segment, ((0, 0), (left, right)), mode="reflect")
        epochs[:, index, :] = padded[:, :length]

    return EpochSet(
        run_id=run_id,
        channels=channel_names,
        sample_rate_hz=float(sample_rate_hz),
        event_numbers=numbers,
        onset_samples=onsets,
        raw=epochs,
        t_ms=t_ms,
        core=core,
        kept=kept,
        pad_reflected=reflected,
    )


def window_slice(t_ms: np.ndarray, start_ms: float, end_ms: float) -> slice:
    """Index slice for ``start_ms <= t < end_ms``."""
    start = int(np.searchsorted(t_ms, start_ms, side="left"))
    end = int(np.searchsorted(t_ms, end_ms, side="left"))
    return slice(start, max(start, end))


def baseline_stats(
    epochs2d: np.ndarray, t_ms: np.ndarray, baseline_ms: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch mean and SD of the raw baseline window. Returns (mean, sd)."""
    window = window_slice(t_ms, *baseline_ms)
    segment = np.asarray(epochs2d, dtype=np.float64)[:, window]
    if segment.shape[1] == 0:
        n = epochs2d.shape[0]
        return np.full(n, np.nan), np.full(n, np.nan)
    return segment.mean(axis=1), segment.std(axis=1, ddof=1 if segment.shape[1] > 1 else 0)


# -----------------------------------------------------------------------------
# Blanking
# -----------------------------------------------------------------------------


def blank_windows(
    recovery_ms: np.ndarray,
    cfg: AnalysisConfig,
    floor_ms: float,
    *,
    post_start_ms: float | None = None,
    epoch_end_ms: float | None = None,
    extra_pulse_windows_ms: Sequence[Sequence[tuple[float, float]]] | None = None,
) -> list[list[tuple[float, float]]]:
    """Per-epoch blank windows in ms.

    ``per_epoch``: [-blank_pre, max(recovery + margin, floor)] for each epoch;
    ``per_condition``: [-blank_pre, post_start_ms] for every epoch. A censored
    (NaN) recovery blanks through the end of the epoch. ``extra_pulse_windows``
    (fake-event epochs) are appended and merged.
    """
    recovery = np.asarray(recovery_ms, dtype=float)
    end_ms = float(epoch_end_ms if epoch_end_ms is not None else cfg.epoch_ms[1])
    windows: list[list[tuple[float, float]]] = []
    for index, value in enumerate(recovery):
        if cfg.blank_mode == "per_condition" and post_start_ms is not None:
            stop = float(post_start_ms)
        elif not np.isfinite(value):
            stop = end_ms
        else:
            stop = max(float(value) + cfg.blank_margin_ms, float(floor_ms))
        items = [(-float(cfg.blank_pre_ms), min(stop, end_ms))]
        if extra_pulse_windows_ms is not None and index < len(extra_pulse_windows_ms):
            items.extend((float(a), float(b)) for a, b in extra_pulse_windows_ms[index])
        windows.append(items)
    return windows


def blank_epochs(
    epochs2d: np.ndarray, t_ms: np.ndarray, windows_ms: Sequence[Sequence[tuple[float, float]]]
) -> np.ndarray:
    """Linear interpolation across each blank window; returns float64 copy."""
    out = np.array(epochs2d, dtype=np.float64, copy=True)
    for index in range(out.shape[0]):
        items = windows_ms[index] if index < len(windows_ms) else []
        sample_windows = []
        for start_ms, end_ms in items:
            sl = window_slice(t_ms, start_ms, end_ms)
            if sl.stop > sl.start:
                sample_windows.append((sl.start, sl.stop))
        if sample_windows:
            out[index] = _blank_and_interpolate(out[index], _merge_windows(sample_windows))
    return out


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterDesign:
    sos: np.ndarray
    description: dict[str, object]

    @property
    def label(self) -> str:
        d = self.description
        return (
            f"{d['type']} {d['btype']} {d['low_hz']:g}-{d['high_hz']:g} Hz, order {d['order']}, "
            f"{d['method']}"
        )


def design_bandpass(
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    zero_phase: bool = True,
    pad_ms: float | None = None,
) -> FilterDesign:
    """Same design as plot_rhs_filtered_wideband.bandpass_filter_wideband."""
    nyquist = 0.5 * sample_rate_hz
    if low_hz <= 0:
        raise ValueError("Bandpass low cutoff must be greater than 0 Hz.")
    if high_hz <= low_hz:
        raise ValueError("Bandpass high cutoff must exceed the low cutoff.")
    if high_hz >= nyquist:
        raise ValueError(f"Bandpass high cutoff must be below Nyquist ({nyquist:g} Hz).")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    description: dict[str, object] = {
        "type": "butterworth",
        "btype": "bandpass",
        "order": int(order),
        "effective_order_zero_phase": int(2 * order) if zero_phase else int(order),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "fs_hz": float(sample_rate_hz),
        "output": "sos",
        "method": "sosfiltfilt zero-phase" if zero_phase else "sosfilt causal",
        "zero_phase": bool(zero_phase),
        "pad_ms": float(pad_ms) if pad_ms is not None else None,
        "sos": np.asarray(sos, dtype=float).tolist(),
    }
    return FilterDesign(sos=np.asarray(sos, dtype=float), description=description)


def filter_epochs(epochs2d: np.ndarray, design: FilterDesign) -> np.ndarray:
    """Filter every epoch (row) independently; returns the padded length."""
    data = np.asarray(epochs2d, dtype=np.float64)
    if data.size == 0:
        return data
    if design.description.get("zero_phase", True):
        return signal.sosfiltfilt(design.sos, data, axis=-1)
    return signal.sosfilt(design.sos, data, axis=-1)


def pseudo_onsets(
    n_samples: int,
    sample_rate_hz: float,
    cfg: AnalysisConfig,
    *,
    spacing_s: float = 1.0,
    jitter_s: float = 0.0,
    rng: np.random.Generator | None = None,
    max_events: int | None = None,
) -> np.ndarray:
    """Evenly spaced onsets that keep the padded epoch inside the recording."""
    start_off, _c0, _c1, end_off = _sample_offsets(cfg, sample_rate_hz)
    first = -start_off
    last = n_samples - end_off
    if last <= first:
        return np.zeros(0, dtype=np.int64)
    step = max(1, int(round(spacing_s * sample_rate_hz)))
    onsets = np.arange(first, last, step, dtype=np.int64)
    if jitter_s > 0 and rng is not None and onsets.size:
        jitter = rng.integers(0, int(round(jitter_s * sample_rate_hz)) + 1, size=onsets.size)
        onsets = np.clip(onsets + jitter, first, last - 1)
    if max_events is not None:
        onsets = onsets[:max_events]
    return onsets


__all__ = [
    "EpochSet",
    "FilterDesign",
    "baseline_stats",
    "blank_epochs",
    "blank_windows",
    "design_bandpass",
    "extract_epochs",
    "filter_epochs",
    "gap_starts",
    "pseudo_onsets",
    "window_slice",
]
