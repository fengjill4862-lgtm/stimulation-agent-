"""Bandpass and amplitude-filtered plots for RHS wideband data."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import signal

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib.pyplot as plt

from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    sample_slice_for_time_window,
    time_window_label,
)
from rhs_naming import channel_label_collapsed, signed_number_plain_decimal, signed_number_token


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
DASH_TRANSLATION = str.maketrans({"–": "-", "—": "-", "−": "-"})


def _parse_range_pair(text: str, unit_name: str) -> tuple[float, float]:
    """Parse two numbers separated by '-', 'to', ',', or ':'."""
    normalized = text.translate(DASH_TRANSLATION).strip()
    match = re.search(
        rf"({_NUMBER_PATTERN})\s*(?:-|to|,|:)\s*({_NUMBER_PATTERN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        values = re.findall(_NUMBER_PATTERN, normalized)
        if len(values) == 2:
            return float(values[0]), float(values[1])
        raise ValueError(f"Enter {unit_name} as a range, e.g. 200-400.")

    return float(match.group(1)), float(match.group(2))


def parse_frequency_range(
    text: str, unit_name: str = "bandpass Hz"
) -> tuple[float, float] | None:
    """Parse a positive frequency range, or 'all' for all recorded frequencies."""
    cleaned = text.strip().lower()
    if cleaned in {
        "",
        "all",
        "full",
        "raw",
        "wideband",
        "all frequency",
        "all frequencies",
        "all recorded frequency",
        "all recorded frequencies",
    }:
        return None

    low, high = sorted(_parse_range_pair(text, unit_name))
    if low <= 0 or high <= 0 or low == high:
        raise ValueError(f"{unit_name} range must have two positive, different values.")
    return low, high


def parse_amplitude_range(text: str, unit_name: str = "amplitude uV") -> tuple[float, float]:
    """Parse a signed amplitude window, such as '-100 - 100 uV'."""
    low, high = sorted(_parse_range_pair(text, unit_name))
    if low == high:
        raise ValueError(f"{unit_name} range must have two different values.")
    return low, high


def bandpass_filter_wideband(
    raw_uV: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float] | None,
    order: int = 4,
) -> np.ndarray:
    """Bandpass raw wideband data, or return all recorded frequencies."""
    if band_hz is None:
        return raw_uV.astype(np.float64, copy=False)

    low_hz, high_hz = band_hz
    nyquist_hz = 0.5 * sample_rate_hz
    if low_hz <= 0:
        raise ValueError("Bandpass low cutoff must be greater than 0 Hz.")
    if high_hz >= nyquist_hz:
        raise ValueError(
            f"Bandpass high cutoff must be below Nyquist ({nyquist_hz:g} Hz)."
        )

    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, raw_uV.astype(np.float64, copy=False))


def amplitude_mask(
    filtered_uV: np.ndarray, amplitude_uV: tuple[float, float] | None
) -> np.ndarray:
    """Return a mask for samples inside the signed amplitude window."""
    if amplitude_uV is None:
        return np.ones(filtered_uV.shape, dtype=bool)

    low_uV, high_uV = amplitude_uV
    return (filtered_uV >= low_uV) & (filtered_uV <= high_uV)


def masked_envelope_for_display(
    y: np.ndarray,
    mask: np.ndarray,
    sample_rate_hz: float,
    max_points: int,
    start_time_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build a min/max display envelope using only masked samples."""
    n_samples = y.size
    if max_points <= 0 or n_samples <= max_points:
        x = np.arange(n_samples, dtype=np.float64) / sample_rate_hz + start_time_s
        y_plot = y.copy()
        y_plot[~mask] = np.nan
        return x, y_plot, False

    target_bins = max(2, max_points // 2)
    bin_size = int(math.ceil(n_samples / target_bins))
    n_bins = int(math.ceil(n_samples / bin_size))
    x_values: list[float] = []
    y_values: list[float] = []

    for bin_index in range(n_bins):
        start = bin_index * bin_size
        end = min(start + bin_size, n_samples)
        bin_mask = mask[start:end]
        if not np.any(bin_mask):
            continue
        bin_values = y[start:end][bin_mask]
        x_center = (start + end - 1.0) * 0.5 / sample_rate_hz + start_time_s
        x_values.extend([x_center, x_center])
        y_values.extend([float(np.min(bin_values)), float(np.max(bin_values))])

    return np.asarray(x_values), np.asarray(y_values), True


def default_filtered_output_path(
    folder: Path,
    channel_name: str,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
) -> Path:
    safe_channel = channel_name.replace("-", "").replace(" ", "_")
    band_label = format_bandpass_filename_label(band_hz)
    if amplitude_uV is None:
        amp_label = "allAmp"
    else:
        low_label = signed_number_plain_decimal(amplitude_uV[0])
        high_label = signed_number_plain_decimal(amplitude_uV[1])
        amp_label = f"{low_label}-to-{high_label}uV"
    return folder / f"filtered_wideband_{safe_channel}_{band_label}_{amp_label}.png"


def default_response_output_path(
    folder: Path,
    channel_label: str,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    time_window: tuple[float, float] | None,
) -> Path:
    """Name the Function 4 response-only PNG, without stimulation metadata.

    Function 4 plots recorded responses only, so unlike
    `default_filtered_output_path` this name carries a time-window label and no
    stim information. Kept here rather than in the UI layer so the notebook and
    the batch runner cannot drift apart on filenames.
    """
    safe_channel = channel_label_collapsed(channel_label)
    band_label = format_bandpass_filename_label(band_hz).replace(".", "p")
    if amplitude_uV is None:
        amp_label = "allAmp"
    else:
        amp_label = (
            f"{signed_number_token(amplitude_uV[0])}-to-"
            f"{signed_number_token(amplitude_uV[1])}uV"
        )
    return folder / (
        f"recorded_response_{safe_channel}_{band_label}_{amp_label}_"
        f"{time_window_label(time_window)}.png"
    )


def format_bandpass_filename_label(band_hz: tuple[float, float] | None) -> str:
    if band_hz is None:
        return "allRecordedHz"
    return f"{band_hz[0]:g}-{band_hz[1]:g}Hz"


def format_bandpass_plot_label(band_hz: tuple[float, float] | None) -> str:
    if band_hz is None:
        return "all recorded frequencies"
    return f"bandpass filtered {band_hz[0]:g}-{band_hz[1]:g} Hz"


def format_bandpass_status(band_hz: tuple[float, float] | None) -> str:
    if band_hz is None:
        return "using all recorded frequencies"
    return f"applying {band_hz[0]:g}-{band_hz[1]:g} Hz bandpass"


def bandpass_warning(band_hz: tuple[float, float] | None) -> str | None:
    """Warn when a sub-1 Hz high-pass is applied to the continuous trace.

    A 0.1 Hz high-pass has a ~1.6 s time constant. Applied to a trace that
    saturates at every stim pulse it rings for seconds, and zero-phase
    filtering spreads that ringing backwards into the pre-stim baseline; the
    slow exponential recoveries in stim-triggered plots are then filter
    artifact, not physiology. Use >= 1 Hz for stim epochs, or the session
    analysis (Function 6), which epochs and blanks before filtering.
    """
    if band_hz is None or band_hz[0] >= 1.0:
        return None
    tau_s = 1.0 / (2.0 * 3.141592653589793 * band_hz[0])
    return (
        f"High-pass {band_hz[0]:g} Hz has a ~{tau_s:.1f} s time constant and is applied to the "
        "continuous trace here; saturating stim artifact rings through it into the "
        "post-stim window and (zero-phase) into the baseline. Use >= 1 Hz for stim "
        "epochs, or Function 6 which epochs and blanks before filtering."
    )


# --- Epoch blanking helpers (shared by Functions 3, 5 and stim_analysis) ---
# These live here, below every epoching module in the reload chain, because
# Function 5's numerics import Function 3's and stim_analysis imports Function
# 5's -- any higher home would close an import cycle.


def merge_sample_windows(windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching (start, end) sample windows."""
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


def blank_and_interpolate(y: np.ndarray, windows: Sequence[tuple[int, int]]) -> np.ndarray:
    """Replace the windowed samples with linear interpolation, in place."""
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


def pulse_blank_sample_windows(
    pulse_segments: Sequence[tuple[int, int]],
    epoch_start: int,
    epoch_end: int,
    sample_rate_hz: float,
    blank_window_ms: tuple[float, float],
) -> list[tuple[int, int]]:
    """Blank windows around each pulse, clipped into epoch-local samples."""
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
    return merge_sample_windows(windows)


def plot_filtered_channels(
    channel_data: Sequence[tuple[str, np.ndarray]],
    sample_rate_hz: float,
    folder: Path,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    max_points: int,
    time_window: tuple[float, float] | None = None,
    stim_channel_name: str | None = None,
) -> tuple[plt.Figure, list[dict[str, float | int | str]]]:
    """Plot one or more bandpass-filtered channels as stacked traces."""
    if not channel_data:
        raise ValueError("No channel data was provided.")

    channel_names = [item[0] for item in channel_data]
    display_slice, display_bounds = sample_slice_for_time_window(
        channel_data[0][1].size, sample_rate_hz, time_window
    )
    n_channels = len(channel_data)
    fig_height = max(5.0, 2.25 * n_channels + 2.0)
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(15, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.84, hspace=0.24)

    amp_title = "all amplitudes" if amplitude_uV is None else (
        f"amplitude {amplitude_uV[0]:g} to {amplitude_uV[1]:g} uV"
    )
    time_title = "" if time_window is None else f", time {display_bounds[0]:g}-{display_bounds[1]:g} s"
    fig.suptitle(
        f"{folder.name}\n{channel_selection_label(channel_names)} "
        f"{format_bandpass_plot_label(band_hz)}, {amp_title}{time_title}",
        fontsize=11,
        y=0.96,
    )
    fig.supylabel("filtered amplitude", x=0.035)

    summaries: list[dict[str, float | int | str]] = []
    used_envelope_any = False
    for ax, (channel_name, raw_uV) in zip(axes_flat, channel_data):
        filtered_full_uV = bandpass_filter_wideband(raw_uV, sample_rate_hz, band_hz)
        filtered_uV = filtered_full_uV[display_slice]
        in_range = amplitude_mask(filtered_uV, amplitude_uV)
        x_masked, y_masked, used_envelope = masked_envelope_for_display(
            filtered_uV,
            in_range,
            sample_rate_hz,
            max_points,
            start_time_s=display_bounds[0],
        )
        used_envelope_any = used_envelope_any or used_envelope

        if x_masked.size:
            ax.plot(x_masked, y_masked, color="#d62728", linewidth=0.8)

        if amplitude_uV is not None:
            low_uV, high_uV = amplitude_uV
            ax.axhline(low_uV, color="#d62728", linewidth=0.7, alpha=0.35)
            ax.axhline(high_uV, color="#d62728", linewidth=0.7, alpha=0.35)
            ax.set_ylim(low_uV, high_uV)

        channel_label = f"{channel_name} *" if channel_name == stim_channel_name else channel_name
        ax.text(
            -0.045,
            0.5,
            channel_label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(display_bounds)
        summaries.append(
            {
                "channel_name": channel_name,
                "samples_total": int(filtered_uV.size),
                "samples_in_amplitude_range": int(np.count_nonzero(in_range)),
                "percent_in_amplitude_range": float(np.mean(in_range) * 100.0),
                "filtered_rms_uV": float(np.sqrt(np.mean(filtered_uV**2))),
            }
        )

    axes_flat[-1].set_xlabel("Time (s)")

    if used_envelope_any:
        axes_flat[-1].text(
            0.995,
            0.01,
            "displayed as min/max envelope",
            transform=axes_flat[-1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    return fig, summaries
