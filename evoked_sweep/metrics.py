#!/usr/bin/env python3
"""The three outcome measures: evoked deflection, band power, post-train change.

A note on band power, because the obvious approach is wrong here. A 5 ms pulse
repeated at ~3.9 Hz is not a narrowband contaminant: its harmonic comb reaches
past 200 Hz, so pulse-locked energy lands in *every* band and notching the
fundamental plus a couple of harmonics does not remove it. Two estimates are
produced instead, and they check each other:

* **comb-excluded** -- Welch over the whole train window, integrating each band
  only over the frequency bins that are not within a half-width of any harmonic.
  Works for every band, including delta.
* **gap-based** -- Welch computed only from the quiet stretch between pulses,
  blanking the response window after each onset, so pulse energy is excluded by
  construction rather than filtered out. Only valid for bands whose period fits
  at least three times into the gap; anything lower is left blank, and the
  cutoff is reported as ``gap_minimum_hz`` (it depends on the pulse interval).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .config import EvokedConfig
from .load import LoadedRun
from .pulses import PulseTrain


@dataclass(frozen=True)
class ChannelEvoked:
    """Per-pulse evoked deflection for one channel."""

    channel: str
    n_pulses_used: int
    pp_uV_median: float
    pp_uV_iqr: float
    peak_latency_ms_median: float
    gap_baseline_pp_uV: float
    pre_train_pp_uV: float
    snr_vs_gap: float
    post_pulse_fraction: float
    waveform_time_ms: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    mean_waveform_uV: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    # Window-split measures: during = [0, pulse_width + guard), post = the rest
    # of the response window. The legacy pp/latency above span both windows.
    during_pp_uV_median: float = float("nan")
    post_pp_uV_median: float = float("nan")
    post_pp_uV_iqr: float = float("nan")
    post_peak_latency_ms_median: float = float("nan")
    baseline_sd_uV: float = float("nan")
    epochs_uV: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))


@dataclass(frozen=True)
class ChannelBands:
    """Train-vs-baseline band power change for one channel, in dB."""

    channel: str
    pulse_rate_hz: float
    band_db: dict[str, float]
    band_db_gap: dict[str, float]
    comb_bins_excluded: int
    # Gap estimates need >= 3 cycles per gap; bands below this are NaN.
    gap_minimum_hz: float = float("nan")


@dataclass(frozen=True)
class ChannelPostTrain:
    """Change that outlasts the train, for one channel."""

    channel: str
    available: bool
    slow_level_delta_uV: float
    rhythm_hz_pre: float
    rhythm_hz_post: float
    broadband_sd_db: float


def _epoch_indices(onset_s: float, fs: float, pre_ms: float, post_ms: float, n: int):
    start = int(round((onset_s - pre_ms / 1000.0) * fs))
    stop = int(round((onset_s + post_ms / 1000.0) * fs))
    if start < 0 or stop > n:
        return None
    return start, stop


def _highpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz / (fs / 2.0), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def evoked_deflection(
    run: LoadedRun, train: PulseTrain, config: EvokedConfig
) -> list[ChannelEvoked]:
    """Per-pulse peak-to-peak and latency, referenced to the late inter-pulse gap.

    The 2% duty cycle is what makes the gap baseline possible: about 200 ms of
    every 259 ms cycle carries no stimulus, so the signal can be compared with
    itself inside the train instead of only with the pre-train period.
    """
    fs = run.sample_rate_hz
    results: list[ChannelEvoked] = []
    if train.n_pulses == 0:
        return results

    pre_ms = 20.0
    post_ms = min(config.response_window_ms, train.period_s * 1000.0 - 5.0)
    gap_end_ms = min(config.gap_baseline_end_ms, train.period_s * 1000.0 - 5.0)
    gap_start_ms = min(config.gap_baseline_start_ms, gap_end_ms - 10.0)
    # When the response window swallows the configured gap baseline (slow
    # protocols: a 1000 ms window inside a 5 s period), anchor the baseline to
    # the last 100 ms before the next pulse instead of measuring the response.
    if gap_start_ms < config.blank_ms:
        gap_end_ms = train.period_s * 1000.0 - 10.0
        gap_start_ms = max(config.blank_ms + 5.0, gap_end_ms - 100.0)

    for index, channel in enumerate(run.healthy_channels):
        x = run.data.raw_uV[index].astype(np.float64)
        n = x.size

        peaks: list[float] = []
        latencies: list[float] = []
        gaps: list[float] = []
        tails: list[float] = []
        stack: list[np.ndarray] = []
        during_pps: list[float] = []
        post_pps: list[float] = []
        post_latencies: list[float] = []
        baseline_sds: list[float] = []
        guard_samples = int(
            round((config.pulse_width_ms + config.post_pulse_guard_ms) / 1000.0 * fs)
        )

        for onset in train.onsets_s:
            window = _epoch_indices(onset, fs, pre_ms, post_ms, n)
            if window is None:
                continue
            start, stop = window
            epoch = x[start:stop]
            pre_samples = int(round(pre_ms / 1000.0 * fs))
            baseline = epoch[:pre_samples].mean() if pre_samples else 0.0
            centred = epoch - baseline
            response = centred[pre_samples:]
            if response.size == 0:
                continue

            peaks.append(float(response.max() - response.min()))
            peak_index = int(np.argmax(np.abs(response)))
            latencies.append(peak_index / fs * 1000.0)
            stack.append(centred)

            if pre_samples:
                baseline_sds.append(float(epoch[:pre_samples].std()))
            during = response[:guard_samples]
            if during.size:
                during_pps.append(float(during.max() - during.min()))
            post = response[guard_samples:]
            if post.size:
                post_pps.append(float(post.max() - post.min()))
                post_index = guard_samples + int(np.argmax(np.abs(post)))
                post_latencies.append(post_index / fs * 1000.0)

            # Fraction of response energy arriving after the pulse should have
            # ended. Pure coupling stops with the pulse; a response outlasts it.
            width_samples = int(round(config.pulse_width_ms / 1000.0 * fs))
            if response.size > width_samples > 0:
                total = float(np.sum(response**2))
                tail = float(np.sum(response[width_samples:] ** 2))
                if total > 0:
                    tails.append(tail / total)

            gap_window = _epoch_indices(onset, fs, -gap_start_ms, gap_end_ms, n)
            if gap_window is not None:
                gap_start, gap_stop = gap_window
                gap = x[gap_start:gap_stop]
                if gap.size:
                    gaps.append(float(gap.max() - gap.min()))

        if not peaks:
            continue

        pre_train_stop = int(round(train.train_start_s * fs))
        pre_train_start = max(0, pre_train_stop - int(round(2.0 * fs)))
        pre_train = x[pre_train_start:pre_train_stop]
        pre_train_pp = float(pre_train.max() - pre_train.min()) if pre_train.size else float("nan")

        gap_pp = float(np.median(gaps)) if gaps else float("nan")
        pp_median = float(np.median(peaks))
        quartiles = np.percentile(peaks, [25, 75])

        waveform = np.mean(np.vstack(stack), axis=0) if stack else np.zeros(0)
        waveform_time = (np.arange(waveform.size) / fs - pre_ms / 1000.0) * 1000.0

        results.append(
            ChannelEvoked(
                channel=channel,
                n_pulses_used=len(peaks),
                pp_uV_median=pp_median,
                pp_uV_iqr=float(quartiles[1] - quartiles[0]),
                peak_latency_ms_median=float(np.median(latencies)),
                gap_baseline_pp_uV=gap_pp,
                pre_train_pp_uV=pre_train_pp,
                snr_vs_gap=float(pp_median / gap_pp) if gap_pp and np.isfinite(gap_pp) else float("nan"),
                post_pulse_fraction=float(np.median(tails)) if tails else float("nan"),
                waveform_time_ms=waveform_time,
                mean_waveform_uV=waveform,
                during_pp_uV_median=float(np.median(during_pps)) if during_pps else float("nan"),
                post_pp_uV_median=float(np.median(post_pps)) if post_pps else float("nan"),
                post_pp_uV_iqr=(
                    float(np.subtract(*np.percentile(post_pps, [75, 25])))
                    if post_pps
                    else float("nan")
                ),
                post_peak_latency_ms_median=(
                    float(np.median(post_latencies)) if post_latencies else float("nan")
                ),
                baseline_sd_uV=float(np.median(baseline_sds)) if baseline_sds else float("nan"),
                epochs_uV=np.vstack(stack) if stack else np.zeros((0, 0)),
            )
        )

    return results


def _welch(x: np.ndarray, fs: float, nperseg: int):
    nperseg = int(min(nperseg, x.size))
    if nperseg < 32:
        return np.zeros(0), np.zeros(0)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def _band_mean(freqs: np.ndarray, psd: np.ndarray, low: float, high: float, keep: np.ndarray) -> float:
    mask = (freqs >= low) & (freqs < high) & keep
    if not mask.any():
        return float("nan")
    return float(psd[mask].mean())


def band_power(run: LoadedRun, train: PulseTrain, config: EvokedConfig) -> list[ChannelBands]:
    """Band power during the train relative to an equal-length pre-train baseline."""
    fs = run.sample_rate_hz
    results: list[ChannelBands] = []
    if train.n_pulses == 0 or not np.isfinite(train.period_s):
        return results

    train_start = int(round(train.train_start_s * fs))
    train_stop = int(round(train.train_end_s * fs))
    length = train_stop - train_start
    base_stop = train_start
    base_start = max(0, base_stop - length)
    if base_stop - base_start < fs:  # need at least a second of baseline
        return results

    nperseg = int(4 * fs)
    rate = train.rate_hz

    for index, channel in enumerate(run.healthy_channels):
        x = run.data.raw_uV[index].astype(np.float64)
        train_seg = x[train_start:train_stop]
        base_seg = x[base_start:base_stop]

        freqs, psd_train = _welch(train_seg, fs, nperseg)
        _, psd_base = _welch(base_seg, fs, nperseg)
        if freqs.size == 0 or psd_base.size != psd_train.size:
            continue

        # Exclude every bin near a harmonic of the pulse rate.
        keep = np.ones(freqs.size, dtype=bool)
        excluded = 0
        if np.isfinite(rate) and rate > 0:
            df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
            half_width = max(0.5, 2.0 * df)
            harmonic = rate
            while harmonic <= freqs[-1]:
                near = np.abs(freqs - harmonic) <= half_width
                excluded += int(np.count_nonzero(near & keep))
                keep &= ~near
                harmonic += rate

        band_db: dict[str, float] = {}
        for name, low, high in config.bands:
            train_power = _band_mean(freqs, psd_train, low, high, keep)
            base_power = _band_mean(freqs, psd_base, low, high, keep)
            if not np.isfinite(train_power) or not np.isfinite(base_power) or base_power <= 0:
                band_db[name] = float("nan")
            else:
                band_db[name] = float(10.0 * np.log10(train_power / base_power))

        band_db_gap, gap_minimum_hz = _gap_band_power(x, fs, train, config)
        results.append(
            ChannelBands(
                channel=channel,
                pulse_rate_hz=float(rate),
                band_db=band_db,
                band_db_gap=band_db_gap,
                comb_bins_excluded=excluded,
                gap_minimum_hz=gap_minimum_hz,
            )
        )

    return results


def _gap_band_power(
    x: np.ndarray, fs: float, train: PulseTrain, config: EvokedConfig
) -> tuple[dict[str, float], float]:
    """Cross-check using only the quiet stretch between pulses.

    Valid only for bands whose period fits at least three times into the gap;
    lower bands come back NaN rather than as a number that cannot be trusted.
    Returns (band dB dict, minimum valid Hz) so the cutoff can be reported.
    """
    gap_start_ms = config.blank_ms
    gap_stop_ms = train.period_s * 1000.0 - 5.0
    gap_duration_s = (gap_stop_ms - gap_start_ms) / 1000.0
    all_nan = {name: float("nan") for name, _low, _high in config.bands}
    if gap_duration_s <= 0.02:
        return all_nan, float("inf")
    minimum_hz = 3.0 / gap_duration_s

    segments: list[np.ndarray] = []
    for onset in train.onsets_s:
        start = int(round((onset + gap_start_ms / 1000.0) * fs))
        stop = int(round((onset + gap_stop_ms / 1000.0) * fs))
        if start >= 0 and stop <= x.size and stop > start:
            segments.append(x[start:stop])
    if not segments:
        return all_nan, minimum_hz

    segment_length = min(len(s) for s in segments)
    nperseg = int(min(segment_length, 2 ** int(np.floor(np.log2(max(32, segment_length))))))

    def _average_psd(chunks: list[np.ndarray]):
        accumulated = None
        freqs = np.zeros(0)
        count = 0
        for chunk in chunks:
            f, p = _welch(chunk[:segment_length], fs, nperseg)
            if f.size == 0:
                continue
            freqs = f
            accumulated = p if accumulated is None else accumulated + p
            count += 1
        if accumulated is None or count == 0:
            return np.zeros(0), np.zeros(0)
        return freqs, accumulated / count

    train_start = int(round(train.train_start_s * fs))
    base_stop = train_start
    base_chunks: list[np.ndarray] = []
    cursor = base_stop - segment_length
    while cursor >= 0 and len(base_chunks) < len(segments):
        base_chunks.append(x[cursor : cursor + segment_length])
        cursor -= segment_length
    if not base_chunks:
        return all_nan, minimum_hz

    freqs, psd_train = _average_psd(segments)
    _, psd_base = _average_psd(base_chunks)
    if freqs.size == 0 or psd_base.size != psd_train.size:
        return all_nan, minimum_hz

    keep = np.ones(freqs.size, dtype=bool)
    # A band needs at least three cycles inside one gap to be estimable here.
    out: dict[str, float] = {}
    for name, low, high in config.bands:
        if low < minimum_hz:
            out[name] = float("nan")
            continue
        train_power = _band_mean(freqs, psd_train, low, high, keep)
        base_power = _band_mean(freqs, psd_base, low, high, keep)
        if not np.isfinite(train_power) or not np.isfinite(base_power) or base_power <= 0:
            out[name] = float("nan")
        else:
            out[name] = float(10.0 * np.log10(train_power / base_power))
    return out, minimum_hz


def _dominant_rhythm_hz(x: np.ndarray, fs: float) -> float:
    freqs, psd = _welch(x, fs, int(4 * fs))
    if freqs.size == 0:
        return float("nan")
    band = (freqs >= 0.2) & (freqs <= 20.0)
    if not band.any():
        return float("nan")
    return float(freqs[band][int(np.argmax(psd[band]))])


def post_train_change(
    run: LoadedRun, train: PulseTrain, config: EvokedConfig
) -> list[ChannelPostTrain]:
    """Compare an after-the-train window with an equal pre-train window.

    Caveat carried into the output: the DSP high-pass sat at 0.777 Hz and the
    analog corner at 0.09 Hz, so anything slower than about 0.1 Hz -- a true DC
    or vascular drift -- was filtered out before it reached the file and cannot
    be recovered here.
    """
    fs = run.sample_rate_hz
    results: list[ChannelPostTrain] = []

    window = int(round(config.post_train_window_s * fs))
    gap = int(round(config.post_train_gap_s * fs))
    post_start = int(round(train.train_end_s * fs)) + gap
    post_stop = post_start + window
    pre_stop = int(round(train.train_start_s * fs)) - gap
    pre_start = pre_stop - window

    for index, channel in enumerate(run.healthy_channels):
        x = run.data.raw_uV[index].astype(np.float64)
        available = pre_start >= 0 and post_stop <= x.size
        if not available:
            results.append(
                ChannelPostTrain(channel, False, float("nan"), float("nan"), float("nan"), float("nan"))
            )
            continue

        pre = x[pre_start:pre_stop]
        post = x[post_start:post_stop]
        sd_ratio = (
            10.0 * np.log10(post.std() / pre.std()) if pre.std() > 0 and post.std() > 0 else float("nan")
        )
        results.append(
            ChannelPostTrain(
                channel=channel,
                available=True,
                slow_level_delta_uV=float(post.mean() - pre.mean()),
                rhythm_hz_pre=_dominant_rhythm_hz(pre, fs),
                rhythm_hz_post=_dominant_rhythm_hz(post, fs),
                broadband_sd_db=float(sd_ratio),
            )
        )

    return results


__all__ = [
    "ChannelBands",
    "ChannelEvoked",
    "ChannelPostTrain",
    "band_power",
    "evoked_deflection",
    "post_train_change",
]
