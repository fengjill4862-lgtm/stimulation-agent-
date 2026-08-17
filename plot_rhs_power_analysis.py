"""Power analysis for RHS wideband recordings.

Function 5 supports two related analyses:

1. Pre/post neuromodulation power: compare clean recording before the first
   stim sample with clean recording after the last stim sample.
2. Stim-triggered power: epoch each stimulation event, blank pulse artifacts,
   then filter and compute paired baseline-vs-post power changes.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from plot_rhs_filtered_wideband import bandpass_filter_wideband
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    find_pulse_segments,
)
from plot_rhs_stim_triggered_events import build_stim_triggered_events
from rhs_naming import number_token, safe_token


@dataclass(frozen=True)
class PowerBand:
    """One frequency band used for power analysis."""

    name: str
    low_hz: float
    high_hz: float

    @property
    def label(self) -> str:
        return f"{self.name} {self.low_hz:g}-{self.high_hz:g} Hz"


@dataclass(frozen=True)
class TimeWindowMs:
    """A time window in milliseconds relative to a stim trigger."""

    name: str
    start_ms: float
    end_ms: float

    @property
    def label(self) -> str:
        return f"{self.name} {self.start_ms:g} to {self.end_ms:g} ms"


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Generated power-analysis figure, table rows, and output targets."""

    figure: plt.Figure
    rows: list[dict[str, object]]
    png_path: Path
    csv_path: Path
    event_count: int
    stim_channel_name: str


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNSIGNED_NUMBER_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_power_bands(text: str) -> list[PowerBand]:
    """Parse bands from lines like 'theta 4-8' or '4-8 Hz'."""
    bands: list[PowerBand] = []
    for index, raw_part in enumerate(re.split(r"[;\n]+", text), start=1):
        part = raw_part.strip()
        if not part:
            continue
        normalized = _normalize_dashes(part).replace("Hz", "").replace("hz", "").strip()
        numbers = _parse_range_numbers(normalized, allow_signed=False)
        if numbers is None:
            raise ValueError(f"Could not understand power band: {part}")
        low_hz, high_hz = sorted(numbers)
        if low_hz <= 0 or high_hz <= low_hz:
            raise ValueError(f"Power band must be positive and increasing: {part}")
        first_number = re.findall(_UNSIGNED_NUMBER_PATTERN, normalized)[0]
        name_text = normalized[: normalized.find(first_number)].strip(" :-_,")
        name = safe_token(name_text) if name_text else f"band{index}_{low_hz:g}-{high_hz:g}Hz"
        bands.append(PowerBand(name=name, low_hz=low_hz, high_hz=high_hz))

    if not bands:
        raise ValueError("Enter at least one power band, e.g. theta 4-8.")
    return bands


def parse_time_window_ms(text: str, default_name: str = "window") -> TimeWindowMs:
    """Parse one signed ms window, e.g. '-500 to -50' or '20-300'."""
    cleaned = _normalize_dashes(text).replace("ms", "").strip()
    numbers = _parse_range_numbers(cleaned, allow_signed=True)
    if numbers is None:
        raise ValueError(f"Enter {default_name} as two ms values, e.g. -500 to -50.")
    start_ms, end_ms = numbers
    if end_ms <= start_ms:
        raise ValueError(f"{default_name} end must be after start.")
    return TimeWindowMs(name=default_name, start_ms=start_ms, end_ms=end_ms)


def parse_post_windows_ms(text: str) -> list[TimeWindowMs]:
    """Parse one or more post-stim windows separated by semicolons/newlines."""
    windows: list[TimeWindowMs] = []
    for index, raw_part in enumerate(re.split(r"[;\n]+", text), start=1):
        part = raw_part.strip()
        if not part:
            continue
        window = parse_time_window_ms(part, default_name=f"post{index}")
        if window.end_ms <= 0:
            raise ValueError("Post-stim windows must end after the stim trigger.")
        windows.append(window)
    if not windows:
        raise ValueError("Enter at least one post-stim window, e.g. 20 to 300.")
    return windows


def default_prepost_power_output_paths(
    folder: Path,
    channels: Sequence[str],
    bands: Sequence[PowerBand],
    window_s: float,
    step_s: float,
) -> tuple[Path, Path]:
    """Return PNG and CSV paths for pre/post neuromodulation power."""
    channel_label = safe_token(channel_selection_label(channels))
    band_label = _bands_label_for_filename(bands)
    stem = (
        f"power_prepost_{channel_label}_{band_label}_"
        f"win{number_token(window_s)}s_step{number_token(step_s)}s"
    )
    return folder / f"{stem}.png", folder / f"{stem}.csv"


def default_power_output_paths(
    folder: Path,
    channels: Sequence[str],
    bands: Sequence[PowerBand],
    baseline_window: TimeWindowMs,
    post_windows: Sequence[TimeWindowMs],
    blank_window_ms: tuple[float, float] = (-10.0, 50.0),
) -> tuple[Path, Path]:
    """Return PNG and CSV paths for stim-triggered power."""
    channel_label = safe_token(channel_selection_label(channels))
    band_label = _bands_label_for_filename(bands)
    baseline_label = _window_label_for_filename("base", baseline_window)
    post_label = "_".join(
        _window_label_for_filename(f"post{index}", window)
        for index, window in enumerate(post_windows, start=1)
    )
    blank_label = (
        f"blank{number_token(blank_window_ms[0])}to"
        f"{number_token(blank_window_ms[1])}ms"
    )
    stem = (
        f"power_event_{channel_label}_{band_label}_"
        f"{baseline_label}_{post_label}_{blank_label}"
    )
    return folder / f"{stem}.png", folder / f"{stem}.csv"


def analyze_pre_post_power(
    channel_data: Sequence[tuple[str, np.ndarray]],
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    folder: Path,
    bands: Sequence[PowerBand],
    stim_channel_name: str,
    window_s: float = 10.0,
    step_s: float = 5.0,
    guard_s: float = 1.0,
    color_scale_db: float = 3.0,
    bootstrap_iterations: int = 1000,
) -> PowerAnalysisResult:
    """Compare clean pre-stimulation and post-stimulation band power."""
    _validate_channel_and_band_inputs(channel_data, bands)
    if window_s <= 0:
        raise ValueError("Window size must be greater than 0 s.")
    if step_s <= 0:
        raise ValueError("Step size must be greater than 0 s.")
    if guard_s < 0:
        raise ValueError("State guard must be 0 s or greater.")

    stim_samples = np.flatnonzero(stim_uA != 0)
    if stim_samples.size == 0:
        raise ValueError("No nonzero stim_data was found.")

    n_samples = int(channel_data[0][1].size)
    first_stim = int(stim_samples[0])
    last_stim_exclusive = int(stim_samples[-1]) + 1
    guard_samples = int(round(guard_s * sample_rate_hz))
    pre_bounds = (0, max(0, first_stim - guard_samples))
    post_bounds = (min(n_samples, last_stim_exclusive + guard_samples), n_samples)
    min_window_samples = int(round(window_s * sample_rate_hz))
    if pre_bounds[1] - pre_bounds[0] < min_window_samples:
        raise ValueError(
            "Pre-stim recording is shorter than one power window. "
            "Reduce Window size (s) or State guard (s)."
        )
    if post_bounds[1] - post_bounds[0] < min_window_samples:
        raise ValueError(
            "Post-stim recording is shorter than one power window. "
            "Reduce Window size (s) or State guard (s)."
        )

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    channel_names = [channel_name for channel_name, _raw_uV in channel_data]
    for channel_name, raw_uV in channel_data:
        pre_raw = raw_uV[pre_bounds[0] : pre_bounds[1]]
        post_raw = raw_uV[post_bounds[0] : post_bounds[1]]
        for band in bands:
            pre_power = _band_power_trace(pre_raw, sample_rate_hz, band)
            post_power = _band_power_trace(post_raw, sample_rate_hz, band)
            pre_values = _sliding_window_means(pre_power, sample_rate_hz, window_s, step_s)
            post_values = _sliding_window_means(post_power, sample_rate_hz, window_s, step_s)
            if pre_values.size == 0 or post_values.size == 0:
                continue

            pre_mean = float(np.mean(pre_values))
            post_mean = float(np.mean(post_values))
            change_db = _power_change_db(post_mean, pre_mean)
            ci_low, ci_high = _bootstrap_independent_change_ci_db(
                pre_values,
                post_values,
                rng,
                bootstrap_iterations,
            )
            rows.append(
                {
                    "analysis_mode": "pre_post_neuromodulation",
                    "folder": folder.name,
                    "channel": channel_name,
                    "stim_channel": stim_channel_name,
                    "band": band.name,
                    "low_hz": band.low_hz,
                    "high_hz": band.high_hz,
                    "first_stim_s": first_stim / sample_rate_hz,
                    "last_stim_s": last_stim_exclusive / sample_rate_hz,
                    "pre_start_s": pre_bounds[0] / sample_rate_hz,
                    "pre_end_s": pre_bounds[1] / sample_rate_hz,
                    "post_start_s": post_bounds[0] / sample_rate_hz,
                    "post_end_s": post_bounds[1] / sample_rate_hz,
                    "window_s": window_s,
                    "step_s": step_s,
                    "state_guard_s": guard_s,
                    "pre_windows": int(pre_values.size),
                    "post_windows": int(post_values.size),
                    "baseline_power_uV2": pre_mean,
                    "post_power_uV2": post_mean,
                    "change_db": change_db,
                    "change_db_ci_low": ci_low,
                    "change_db_ci_high": ci_high,
                    "percent_change": _percent_change(post_mean, pre_mean),
                }
            )

    if not rows:
        raise ValueError("No valid pre/post power windows were available.")

    png_path, csv_path = default_prepost_power_output_paths(
        folder, channel_names, bands, window_s, step_s
    )
    figure = plot_prepost_power_heatmap(
        rows=rows,
        folder=folder,
        channels=channel_names,
        bands=bands,
        stim_channel_name=stim_channel_name,
        color_scale_db=color_scale_db,
    )
    return PowerAnalysisResult(
        figure=figure,
        rows=rows,
        png_path=png_path,
        csv_path=csv_path,
        event_count=1,
        stim_channel_name=stim_channel_name,
    )


def analyze_event_locked_power(
    channel_data: Sequence[tuple[str, np.ndarray]],
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    folder: Path,
    bands: Sequence[PowerBand],
    baseline_window: TimeWindowMs,
    post_windows: Sequence[TimeWindowMs],
    train_gap_ms: float,
    stim_channel_name: str,
    blank_window_ms: tuple[float, float] = (-10.0, 50.0),
    color_scale_db: float = 3.0,
    filter_padding_ms: float = 500.0,
    bootstrap_iterations: int = 1000,
) -> PowerAnalysisResult:
    """Compute paired event-locked power after blanking each stim pulse."""
    _validate_channel_and_band_inputs(channel_data, bands)
    if blank_window_ms[1] <= blank_window_ms[0]:
        raise ValueError("Blank window end must be after blank window start.")
    if filter_padding_ms < 0:
        raise ValueError("Filter padding must be 0 ms or greater.")

    events = build_stim_triggered_events(
        stim_uA=stim_uA,
        sample_rate_hz=sample_rate_hz,
        time_window=None,
        merge_gap_ms=train_gap_ms,
    )
    if not events:
        raise ValueError("No stim-triggered events were found.")

    pulse_segments = find_pulse_segments(stim_uA)
    if not pulse_segments:
        raise ValueError("No nonzero stim_data pulses were found.")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    channel_names = [channel_name for channel_name, _raw_uV in channel_data]
    min_window_ms = min(
        [baseline_window.start_ms]
        + [window.start_ms for window in post_windows]
        + [blank_window_ms[0]]
    )
    max_window_ms = max(
        [baseline_window.end_ms]
        + [window.end_ms for window in post_windows]
        + [blank_window_ms[1]]
    )
    epoch_start_offset = int(round((min_window_ms - filter_padding_ms) * 1.0e-3 * sample_rate_hz))
    epoch_end_offset = int(round((max_window_ms + filter_padding_ms) * 1.0e-3 * sample_rate_hz))

    for channel_name, raw_uV in channel_data:
        for band in bands:
            for post_window in post_windows:
                baseline_values: list[float] = []
                post_values: list[float] = []
                event_numbers: list[int] = []
                for event in events:
                    onset_sample = int(round(event.onset_s * sample_rate_hz))
                    epoch_start = onset_sample + epoch_start_offset
                    epoch_end = onset_sample + epoch_end_offset
                    if epoch_start < 0 or epoch_end > raw_uV.size or epoch_end <= epoch_start:
                        continue

                    epoch = raw_uV[epoch_start:epoch_end].astype(np.float64, copy=True)
                    pulse_windows = _pulse_blank_windows_for_epoch(
                        pulse_segments=pulse_segments,
                        epoch_start=epoch_start,
                        epoch_end=epoch_end,
                        sample_rate_hz=sample_rate_hz,
                        blank_window_ms=blank_window_ms,
                    )
                    if pulse_windows:
                        epoch = _blank_and_interpolate(epoch, pulse_windows)

                    try:
                        band_epoch = bandpass_filter_wideband(
                            epoch,
                            sample_rate_hz,
                            (band.low_hz, band.high_hz),
                        )
                    except ValueError:
                        raise
                    except Exception as exc:
                        raise ValueError(
                            f"Could not filter {band.label} for event windows. "
                            "Try a wider baseline/post window or remove very low bands."
                        ) from exc

                    power_epoch = np.square(band_epoch, dtype=np.float64)
                    baseline_value = _window_mean_from_epoch(
                        power_epoch, epoch_start, onset_sample, sample_rate_hz, baseline_window
                    )
                    post_value = _window_mean_from_epoch(
                        power_epoch, epoch_start, onset_sample, sample_rate_hz, post_window
                    )
                    if baseline_value is None or post_value is None:
                        continue
                    baseline_values.append(baseline_value)
                    post_values.append(post_value)
                    event_numbers.append(event.event_number)

                baseline_array = np.asarray(baseline_values, dtype=float)
                post_array = np.asarray(post_values, dtype=float)
                event_change_db = _paired_change_db(post_array, baseline_array)
                used = int(event_change_db.size)
                if used == 0:
                    baseline_mean = float("nan")
                    post_mean = float("nan")
                    change_db = float("nan")
                    sem = float("nan")
                    ci_low = float("nan")
                    ci_high = float("nan")
                else:
                    baseline_mean = float(np.mean(baseline_array))
                    post_mean = float(np.mean(post_array))
                    change_db = float(np.mean(event_change_db))
                    sem = _sem(event_change_db)
                    ci_low, ci_high = _bootstrap_paired_change_ci_db(
                        baseline_array,
                        post_array,
                        rng,
                        bootstrap_iterations,
                    )
                rows.append(
                    {
                        "analysis_mode": "stim_triggered_epoch_blank_filter",
                        "folder": folder.name,
                        "channel": channel_name,
                        "stim_channel": stim_channel_name,
                        "band": band.name,
                        "low_hz": band.low_hz,
                        "high_hz": band.high_hz,
                        "baseline_window_ms": f"{baseline_window.start_ms:g} to {baseline_window.end_ms:g}",
                        "post_window": post_window.name,
                        "post_window_ms": f"{post_window.start_ms:g} to {post_window.end_ms:g}",
                        "blank_window_ms": f"{blank_window_ms[0]:g} to {blank_window_ms[1]:g}",
                        "train_gap_ms": train_gap_ms,
                        "events_detected": len(events),
                        "events_used": used,
                        "used_event_numbers": " ".join(str(number) for number in event_numbers),
                        "baseline_power_uV2": baseline_mean,
                        "post_power_uV2": post_mean,
                        "change_db": change_db,
                        "change_db_sem": sem,
                        "change_db_ci_low": ci_low,
                        "change_db_ci_high": ci_high,
                        "percent_change": _percent_change(post_mean, baseline_mean),
                    }
                )

    png_path, csv_path = default_power_output_paths(
        folder, channel_names, bands, baseline_window, post_windows, blank_window_ms
    )
    figure = plot_event_power_heatmap(
        rows=rows,
        folder=folder,
        channels=channel_names,
        bands=bands,
        post_windows=post_windows,
        baseline_window=baseline_window,
        stim_channel_name=stim_channel_name,
        event_count=len(events),
        color_scale_db=color_scale_db,
    )
    return PowerAnalysisResult(
        figure=figure,
        rows=rows,
        png_path=png_path,
        csv_path=csv_path,
        event_count=len(events),
        stim_channel_name=stim_channel_name,
    )


def plot_prepost_power_heatmap(
    rows: Sequence[dict[str, object]],
    folder: Path,
    channels: Sequence[str],
    bands: Sequence[PowerBand],
    stim_channel_name: str,
    color_scale_db: float = 3.0,
) -> plt.Figure:
    """Plot pre/post neuromodulation power change in dB."""
    matrix, annotations = _matrix_from_rows(rows, channels, [(band, None) for band in bands])
    fig_width = max(7.5, 0.9 * len(bands) + 3.0)
    fig_height = max(5.0, 0.45 * len(channels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-abs(color_scale_db),
        vmax=abs(color_scale_db),
    )
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels([f"{band.name}\n{band.low_hz:g}-{band.high_hz:g} Hz" for band in bands])
    _format_power_axis(ax, channels, stim_channel_name)
    _annotate_heatmap(ax, matrix, annotations)
    ax.set_title(
        f"{folder.name}\nPre/post neuromodulation power change "
        f"(post vs pre, fixed scale +/-{abs(color_scale_db):g} dB)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Power change (dB, post vs pre)")
    ax.set_xlabel("Band")
    ax.set_ylabel("Channel")
    fig.tight_layout()
    return fig


def plot_event_power_heatmap(
    rows: Sequence[dict[str, object]],
    folder: Path,
    channels: Sequence[str],
    bands: Sequence[PowerBand],
    post_windows: Sequence[TimeWindowMs],
    baseline_window: TimeWindowMs,
    stim_channel_name: str,
    event_count: int,
    color_scale_db: float = 3.0,
) -> plt.Figure:
    """Plot paired event-locked power change in dB."""
    columns = [(band, window) for band in bands for window in post_windows]
    matrix, annotations = _matrix_from_rows(rows, channels, columns)
    fig_width = max(9.0, 1.1 * len(columns) + 3.0)
    fig_height = max(5.0, 0.45 * len(channels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-abs(color_scale_db),
        vmax=abs(color_scale_db),
    )
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [
            f"{band.name}\n{band.low_hz:g}-{band.high_hz:g} Hz\n{window.start_ms:g}-{window.end_ms:g} ms"
            for band, window in columns
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    _format_power_axis(ax, channels, stim_channel_name)
    _annotate_heatmap(ax, matrix, annotations)
    ax.set_title(
        f"{folder.name}\nStim-triggered power change vs baseline "
        f"{baseline_window.start_ms:g} to {baseline_window.end_ms:g} ms; "
        f"{event_count} stim event(s); fixed scale +/-{abs(color_scale_db):g} dB",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Power change (dB, post vs baseline)")
    ax.set_xlabel("Band and post-stim window")
    ax.set_ylabel("Channel")
    fig.tight_layout()
    return fig


def power_rows_to_csv(rows: Sequence[dict[str, object]]) -> str:
    """Convert power-analysis rows to CSV text."""
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _validate_channel_and_band_inputs(
    channel_data: Sequence[tuple[str, np.ndarray]],
    bands: Sequence[PowerBand],
) -> None:
    if not channel_data:
        raise ValueError("No channel data was provided.")
    if not bands:
        raise ValueError("No power bands were provided.")


def _band_power_trace(raw_uV: np.ndarray, sample_rate_hz: float, band: PowerBand) -> np.ndarray:
    filtered = bandpass_filter_wideband(raw_uV, sample_rate_hz, (band.low_hz, band.high_hz))
    return np.square(filtered, dtype=np.float64)


def _sliding_window_means(
    power: np.ndarray,
    sample_rate_hz: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    window_samples = max(1, int(round(window_s * sample_rate_hz)))
    step_samples = max(1, int(round(step_s * sample_rate_hz)))
    if power.size < window_samples:
        return np.asarray([], dtype=float)
    means = [
        float(np.mean(power[start : start + window_samples]))
        for start in range(0, power.size - window_samples + 1, step_samples)
    ]
    return np.asarray(means, dtype=float)


def _pulse_blank_windows_for_epoch(
    pulse_segments: Sequence[tuple[int, int]],
    epoch_start: int,
    epoch_end: int,
    sample_rate_hz: float,
    blank_window_ms: tuple[float, float],
) -> list[tuple[int, int]]:
    start_offset = int(round(blank_window_ms[0] * 1.0e-3 * sample_rate_hz))
    end_offset = int(round(blank_window_ms[1] * 1.0e-3 * sample_rate_hz))
    windows: list[tuple[int, int]] = []
    for pulse_start, pulse_end in pulse_segments:
        blank_start = pulse_start + start_offset
        blank_end = pulse_end + end_offset
        local_start = max(0, blank_start - epoch_start)
        local_end = min(epoch_end - epoch_start, blank_end - epoch_start)
        if local_end > local_start:
            windows.append((local_start, local_end))
    return _merge_windows(windows)


def _merge_windows(windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    sorted_windows = sorted(windows)
    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted_windows[0]
    for start, end in sorted_windows[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _blank_and_interpolate(y: np.ndarray, windows: Sequence[tuple[int, int]]) -> np.ndarray:
    if not windows:
        return y
    mask = np.zeros(y.size, dtype=bool)
    for start, end in windows:
        mask[max(0, start) : min(y.size, end)] = True
    if not np.any(mask):
        return y
    good = np.flatnonzero(~mask)
    if good.size < 2:
        fill_value = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
        y[mask] = fill_value
        return y
    bad = np.flatnonzero(mask)
    y[bad] = np.interp(bad, good, y[good])
    return y


def _window_mean_from_epoch(
    power_epoch: np.ndarray,
    epoch_start: int,
    onset_sample: int,
    sample_rate_hz: float,
    window_ms: TimeWindowMs,
) -> float | None:
    start = onset_sample + int(round(window_ms.start_ms * 1.0e-3 * sample_rate_hz))
    end = onset_sample + int(round(window_ms.end_ms * 1.0e-3 * sample_rate_hz))
    local_start = start - epoch_start
    local_end = end - epoch_start
    if local_start < 0 or local_end > power_epoch.size or local_end <= local_start:
        return None
    values = power_epoch[local_start:local_end]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _paired_change_db(post_power: np.ndarray, baseline_power: np.ndarray) -> np.ndarray:
    valid = (
        np.isfinite(post_power)
        & np.isfinite(baseline_power)
        & (post_power > 0)
        & (baseline_power > 0)
    )
    if not np.any(valid):
        return np.asarray([], dtype=float)
    return 10.0 * np.log10(post_power[valid] / baseline_power[valid])


def _power_change_db(post_power: float, baseline_power: float) -> float:
    if not np.isfinite(post_power) or not np.isfinite(baseline_power):
        return float("nan")
    if post_power <= 0 or baseline_power <= 0:
        return float("nan")
    return float(10.0 * math.log10(post_power / baseline_power))


def _percent_change(post_power: float, baseline_power: float) -> float:
    if not np.isfinite(post_power) or not np.isfinite(baseline_power) or baseline_power <= 0:
        return float("nan")
    return float((post_power - baseline_power) / baseline_power * 100.0)


def _sem(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _bootstrap_paired_change_ci_db(
    baseline_values: np.ndarray,
    post_values: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    valid = (
        np.isfinite(baseline_values)
        & np.isfinite(post_values)
        & (baseline_values > 0)
        & (post_values > 0)
    )
    baseline = baseline_values[valid]
    post = post_values[valid]
    if baseline.size == 0 or iterations <= 0:
        return float("nan"), float("nan")
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample_index = rng.integers(0, baseline.size, size=baseline.size)
        draws[index] = np.mean(10.0 * np.log10(post[sample_index] / baseline[sample_index]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _bootstrap_independent_change_ci_db(
    baseline_values: np.ndarray,
    post_values: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    baseline = baseline_values[np.isfinite(baseline_values) & (baseline_values > 0)]
    post = post_values[np.isfinite(post_values) & (post_values > 0)]
    if baseline.size == 0 or post.size == 0 or iterations <= 0:
        return float("nan"), float("nan")
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        baseline_sample = baseline[rng.integers(0, baseline.size, size=baseline.size)]
        post_sample = post[rng.integers(0, post.size, size=post.size)]
        draws[index] = _power_change_db(float(np.mean(post_sample)), float(np.mean(baseline_sample)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _matrix_from_rows(
    rows: Sequence[dict[str, object]],
    channels: Sequence[str],
    columns: Sequence[tuple[PowerBand, TimeWindowMs | None]],
) -> tuple[np.ndarray, list[list[str]]]:
    matrix = np.full((len(channels), len(columns)), np.nan, dtype=float)
    annotations = [["" for _column in columns] for _channel in channels]
    for channel_index, channel in enumerate(channels):
        for column_index, (band, window) in enumerate(columns):
            for row in rows:
                if str(row.get("channel")) != channel or str(row.get("band")) != band.name:
                    continue
                if window is not None and str(row.get("post_window")) != window.name:
                    continue
                value = row.get("change_db")
                if value is None:
                    continue
                matrix[channel_index, column_index] = float(value)
                ci_low = row.get("change_db_ci_low")
                ci_high = row.get("change_db_ci_high")
                n_key = "events_used" if "events_used" in row else "post_windows"
                n_value = row.get(n_key, "")
                annotations[channel_index][column_index] = (
                    f"{float(value):.2g}\n"
                    f"n={n_value}\n"
                    f"[{_format_optional_float(ci_low)}, {_format_optional_float(ci_high)}]"
                )
                break
    return matrix, annotations


def _format_power_axis(ax, channels: Sequence[str], stim_channel_name: str) -> None:
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(
        [f"{channel} *" if channel == stim_channel_name else channel for channel in channels],
        fontsize=8,
        fontweight="bold",
    )


def _annotate_heatmap(ax, matrix: np.ndarray, annotations: Sequence[Sequence[str]]) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            if not np.isfinite(matrix[row_index, column_index]):
                continue
            ax.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )


def _format_optional_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.2g}"


def _bands_label_for_filename(bands: Sequence[PowerBand]) -> str:
    band_label = "_".join(
        safe_token(f"{band.name}_{band.low_hz:g}-{band.high_hz:g}Hz") for band in bands
    )
    if len(band_label) > 80:
        band_label = f"{len(bands)}bands"
    return band_label


def _normalize_dashes(text: str) -> str:
    return text.replace("–", "-").replace("—", "-").replace("−", "-")


def _parse_range_numbers(text: str, allow_signed: bool) -> tuple[float, float] | None:
    if allow_signed:
        to_match = re.search(
            rf"({_NUMBER_PATTERN})\s*(?:to|,|:)\s*({_NUMBER_PATTERN})",
            text,
            flags=re.IGNORECASE,
        )
        if to_match:
            return float(to_match.group(1)), float(to_match.group(2))
        positive_dash_match = re.search(
            rf"({_UNSIGNED_NUMBER_PATTERN})\s*-\s*({_UNSIGNED_NUMBER_PATTERN})",
            text,
            flags=re.IGNORECASE,
        )
        if positive_dash_match:
            return float(positive_dash_match.group(1)), float(positive_dash_match.group(2))
        values = re.findall(_NUMBER_PATTERN, text)
        if len(values) == 2:
            return float(values[0]), float(values[1])
        return None

    match = re.search(
        rf"({_UNSIGNED_NUMBER_PATTERN})\s*(?:-|to|,|:)\s*({_UNSIGNED_NUMBER_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    values = re.findall(_UNSIGNED_NUMBER_PATTERN, text)
    if len(values) >= 2:
        return float(values[-2]), float(values[-1])
    return None


def _window_label_for_filename(prefix: str, window: TimeWindowMs) -> str:
    return f"{prefix}_{number_token(window.start_ms)}to{number_token(window.end_ms)}ms"


