#!/usr/bin/env python3
"""Per-peak analysis of the evoked response (Function 7).

Distinct from ``pulses.py``, which recovers stimulus timing: this module takes
the baseline-centred epochs already measured by ``metrics.evoked_deflection``
and characterises the individual peaks of the mean waveform -- the classical
EP decomposition (N1, P1, ...) -- inside the post-pulse response window.

Windows. The response window for peak detection starts at the *nominal* pulse
width plus a guard, so nothing is silently discarded; but the Keithley often
runs long (a programmed 5 ms pulse measures ~7.5 ms), so any peak landing
within ``edge_flag_ms`` of the *measured* off-edge is flagged ``edge_suspect``
rather than trusted as neural. The judgement stays with the user, as with the
artifact evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .config import EvokedConfig
from .metrics import ChannelEvoked


@dataclass(frozen=True)
class PeakMeasure:
    """One peak of the mean waveform, plus its per-pulse statistics."""

    label: str  # "N1", "P1", "N2", ... by polarity and order of latency
    polarity: int  # -1 negative, +1 positive
    latency_ms: float  # on the mean waveform, from pulse onset
    latency_from_offset_ms: float  # from the nominal pulse offset
    amplitude_uV: float  # signed, baseline-referenced, mean waveform
    width_ms: float  # full width at half prominence
    prominence_uV: float
    edge_suspect: bool  # within edge_flag_ms of the measured off-edge
    amp_median_uV: float  # per-pulse statistics from here down
    amp_iqr_uV: float
    latency_jitter_ms: float  # SD of per-pulse latencies
    present: bool  # |amp_median| > presence_k x single-trial baseline SD
    adaptation_ratio: float  # median(last 10) / median(first 10), NaN if <20
    amp_slope_uV_per_pulse: float


@dataclass(frozen=True)
class ChannelPeaks:
    """Every peak found for one channel in one run."""

    channel: str
    n_peaks: int
    n_peaks_clean: int  # peaks not flagged edge_suspect
    mean_baseline_sd_uV: float  # noise floor of the smoothed mean waveform
    threshold_uV: float  # prominence threshold actually used
    peaks: tuple[PeakMeasure, ...] = ()
    smoothed_mean_uV: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def _smooth(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass; identity when disabled or when fs is too low."""
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2.0 or x.shape[-1] < 30:
        return x
    sos = signal.butter(4, cutoff_hz / (fs / 2.0), btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def _label_peaks(entries: list[tuple[int, int, float, float]]) -> list[str]:
    """N1/P1/N2/P2... by polarity and order of occurrence in latency order."""
    labels: list[str] = []
    counts = {-1: 0, 1: 0}
    for _index, polarity, _prominence, _width in entries:
        counts[polarity] += 1
        labels.append(f"{'N' if polarity < 0 else 'P'}{counts[polarity]}")
    return labels


def analyse_channel_peaks(
    evoked: ChannelEvoked,
    fs: float,
    config: EvokedConfig,
    measured_width_ms: float = float("nan"),
) -> ChannelPeaks:
    """Detect and measure the individual peaks of one channel's mean waveform."""
    time_ms = evoked.waveform_time_ms
    mean = evoked.mean_waveform_uV
    empty = ChannelPeaks(
        channel=evoked.channel,
        n_peaks=0,
        n_peaks_clean=0,
        mean_baseline_sd_uV=float("nan"),
        threshold_uV=float("nan"),
    )
    if mean.size == 0 or time_ms.size != mean.size:
        return empty

    smoothed = _smooth(mean.astype(np.float64), fs, config.peak_lowpass_hz)

    baseline = smoothed[time_ms < 0.0]
    baseline_sd = float(baseline.std()) if baseline.size else float("nan")
    threshold = max(
        config.peak_prominence_k * baseline_sd if np.isfinite(baseline_sd) else 0.0, 2.0
    )

    window_start_ms = config.pulse_width_ms + config.post_pulse_guard_ms
    in_window = (time_ms >= window_start_ms) & (time_ms <= config.response_window_ms)
    if not in_window.any():
        return empty
    window_offset = int(np.argmax(in_window))
    segment = smoothed[in_window]

    # (index into smoothed, polarity, prominence, width_ms)
    candidates: list[tuple[int, int, float, float]] = []
    for polarity in (1, -1):
        indices, properties = signal.find_peaks(
            polarity * segment, prominence=threshold, width=1, rel_height=0.5
        )
        for position, prominence, width_samples in zip(
            indices, properties["prominences"], properties["widths"]
        ):
            candidates.append(
                (
                    window_offset + int(position),
                    polarity,
                    float(prominence),
                    float(width_samples) / fs * 1000.0,
                )
            )

    candidates.sort(key=lambda entry: entry[2], reverse=True)
    kept = sorted(candidates[: config.max_peaks], key=lambda entry: entry[0])
    labels = _label_peaks(kept)

    off_edge_ms = measured_width_ms if np.isfinite(measured_width_ms) else config.pulse_width_ms
    epochs = evoked.epochs_uV
    stack = (
        _smooth(epochs.astype(np.float64), fs, config.peak_lowpass_hz)
        if epochs.size
        else np.zeros((0, 0))
    )
    # Single-trial noise for the presence test; the mean-waveform SD is ~sqrt(n)
    # smaller and would call almost anything "present".
    trial_sd = evoked.baseline_sd_uV if np.isfinite(evoked.baseline_sd_uV) else baseline_sd

    window_last = int(in_window.nonzero()[0][-1])
    half_samples = int(round(config.peak_search_half_ms / 1000.0 * fs))

    measures: list[PeakMeasure] = []
    for (peak_index, polarity, prominence, width_ms), label in zip(kept, labels):
        latency_ms = float(time_ms[peak_index])
        amp_median = amp_iqr = jitter = adaptation = slope = float("nan")
        present = False

        if stack.size:
            low = max(window_offset, peak_index - half_samples)
            high = min(window_last + 1, peak_index + half_samples + 1)
            piece = polarity * stack[:, low:high]
            extremum = np.argmax(piece, axis=1)
            amps = polarity * piece[np.arange(piece.shape[0]), extremum]
            lats = time_ms[low + extremum]

            amp_median = float(np.median(amps))
            amp_iqr = float(np.subtract(*np.percentile(amps, [75, 25])))
            jitter = float(np.std(lats))
            present = bool(
                np.isfinite(trial_sd)
                and trial_sd > 0
                and abs(amp_median) > config.presence_k * trial_sd
            )
            if amps.size >= 20:
                first = float(np.median(amps[:10]))
                last = float(np.median(amps[-10:]))
                if abs(first) > 1e-9:
                    adaptation = last / first
            if amps.size >= 3:
                slope = float(np.polyfit(np.arange(amps.size), amps, 1)[0])

        measures.append(
            PeakMeasure(
                label=label,
                polarity=polarity,
                latency_ms=latency_ms,
                latency_from_offset_ms=latency_ms - config.pulse_width_ms,
                amplitude_uV=float(smoothed[peak_index]),
                width_ms=width_ms,
                prominence_uV=prominence,
                edge_suspect=bool(abs(latency_ms - off_edge_ms) <= config.edge_flag_ms),
                amp_median_uV=amp_median,
                amp_iqr_uV=amp_iqr,
                latency_jitter_ms=jitter,
                present=present,
                adaptation_ratio=adaptation,
                amp_slope_uV_per_pulse=slope,
            )
        )

    return ChannelPeaks(
        channel=evoked.channel,
        n_peaks=len(measures),
        n_peaks_clean=sum(1 for m in measures if not m.edge_suspect),
        mean_baseline_sd_uV=baseline_sd,
        threshold_uV=float(threshold),
        peaks=tuple(measures),
        smoothed_mean_uV=smoothed,
    )


__all__ = ["ChannelPeaks", "PeakMeasure", "analyse_channel_peaks"]
