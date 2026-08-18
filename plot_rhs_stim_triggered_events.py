"""Stim-triggered bandpass plots for RHS sessions."""

from __future__ import annotations

import re

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib.pyplot as plt

from plot_rhs_filtered_wideband import (
    DASH_TRANSLATION,
    amplitude_mask,
    bandpass_filter_wideband,
    format_bandpass_filename_label,
    format_bandpass_plot_label,
    masked_envelope_for_display,
)
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    sample_slice_for_time_window,
)
from rhs_naming import channel_label_collapsed, ms_label, ms_number, number_token, signed_number_token


@dataclass(frozen=True)


class StimTriggeredEvent:
    event_number: int
    start_sample: int
    end_sample: int
    onset_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.onset_s


@dataclass(frozen=True)


class EventDisplayWindow:
    sample_slice: slice
    data_start_s: float
    axis_bounds_s: tuple[float, float]
    response_blank_s: tuple[float, float] | None


def find_stim_trigger_onsets(
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    merge_gap_ms: float = 10.0,
) -> np.ndarray:
    """Find stim_data onsets, merging nearby pulses within one train."""
    nonzero = np.flatnonzero(stim_uA != 0)
    if nonzero.size == 0:
        return np.asarray([], dtype=np.int64)

    split_after = np.flatnonzero(np.diff(nonzero) > 1)
    starts = np.r_[nonzero[0], nonzero[split_after + 1]].astype(np.int64)
    ends = np.r_[nonzero[split_after] + 1, nonzero[-1] + 1].astype(np.int64)
    merge_gap_samples = max(0, int(round(merge_gap_ms * 1.0e-3 * sample_rate_hz)))

    merged_starts: list[int] = []
    current_start = int(starts[0])
    current_end = int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start_i = int(start)
        end_i = int(end)
        if start_i - current_end <= merge_gap_samples:
            current_end = end_i
        else:
            merged_starts.append(current_start)
            current_start = start_i
            current_end = end_i
    merged_starts.append(current_start)
    return np.asarray(merged_starts, dtype=np.int64)


def build_stim_triggered_events(
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    time_window: tuple[float, float] | None,
    merge_gap_ms: float = 10.0,
) -> list[StimTriggeredEvent]:
    """Build windows from each grouped stim onset to the next grouped onset."""
    trigger_onsets = find_stim_trigger_onsets(
        stim_uA=stim_uA,
        sample_rate_hz=sample_rate_hz,
        merge_gap_ms=merge_gap_ms,
    )
    if trigger_onsets.size == 0:
        return []

    display_slice, _display_bounds = sample_slice_for_time_window(
        stim_uA.size, sample_rate_hz, time_window
    )
    display_start = int(display_slice.start or 0)
    display_stop = int(display_slice.stop or stim_uA.size)

    events: list[StimTriggeredEvent] = []
    for trigger_index, start_sample in enumerate(trigger_onsets):
        start_i = int(start_sample)
        if start_i < display_start or start_i >= display_stop:
            continue

        if trigger_index + 1 < trigger_onsets.size:
            next_start = int(trigger_onsets[trigger_index + 1])
        else:
            next_start = stim_uA.size
        end_i = min(next_start, display_stop)
        if end_i <= start_i:
            continue

        events.append(
            StimTriggeredEvent(
                event_number=len(events) + 1,
                start_sample=start_i,
                end_sample=end_i,
                onset_s=start_i / sample_rate_hz,
                end_s=end_i / sample_rate_hz,
            )
        )
    return events


def filter_channel_data(
    channel_data: Sequence[tuple[str, np.ndarray]],
    sample_rate_hz: float,
    band_hz: tuple[float, float] | None,
) -> list[tuple[str, np.ndarray]]:
    """Apply the selected frequency setting once for event plotting."""
    return [
        (channel_name, bandpass_filter_wideband(raw_uV, sample_rate_hz, band_hz))
        for channel_name, raw_uV in channel_data
    ]


def default_stim_events_grid_output_path(
    folder: Path,
    channel_label: str,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    time_label: str,
    event_count: int,
) -> Path:
    """Name the single combined PNG containing all stim-triggered events."""
    safe_channel = channel_label_collapsed(channel_label)
    band_label = format_bandpass_filename_label(band_hz).replace(".", "p")
    amp_label = "allAmp" if amplitude_uV is None else (
        f"{signed_number_token(amplitude_uV[0])}-to-"
        f"{signed_number_token(amplitude_uV[1])}uV"
    )
    return (
        folder
        / f"stim_events_all_{safe_channel}_{band_label}_{amp_label}_"
        f"{time_label}_{event_count:03d}events.png"
    )


def plot_stim_triggered_events_grid(
    filtered_channel_data: Sequence[tuple[str, np.ndarray]],
    sample_rate_hz: float,
    folder: Path,
    events: Sequence[StimTriggeredEvent],
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    max_points: int,
    events_per_row: int = 3,
    stim_channel_name: str | None = None,
    stim_uA: np.ndarray | None = None,
    event_time_window: tuple[float, float] | None = None,
    pre_time_s: float = 0.0,
    post_time_window_s: tuple[float, float | None] | None = None,
) -> tuple[plt.Figure, list[dict[str, float | int | str]]]:
    """Plot all stim-triggered events in one grid, with three events per row."""
    if not filtered_channel_data:
        raise ValueError("No channel data was provided.")
    if not events:
        raise ValueError("No stim-triggered events were provided.")

    # Never lay out more columns than there are events, or a 1-event run gets
    # two empty columns across the full width.
    events_per_row = min(max(1, int(events_per_row)), len(events))
    channel_names = [item[0] for item in filtered_channel_data]
    n_channels = len(filtered_channel_data)
    has_stim_row = stim_uA is not None
    rows_per_event = n_channels + (1 if has_stim_row else 0)
    n_event_rows = int(np.ceil(len(events) / events_per_row))
    n_plot_rows = n_event_rows * rows_per_event

    # Header/footer are fixed bands measured in inches. Expressing them as
    # figure fractions made them grow with the row count: a 50-event, 9-row
    # figure is ~189 in tall, so top=0.88/bottom=0.07 reserved 22 in of blank
    # above the plots and 13 in below.
    header_in = 0.85
    footer_in = 0.5

    fig_width = max(12.0, 5.2 * events_per_row)
    fig_height = max(4.0, 1.22 * n_plot_rows + header_in + footer_in)
    fig, axes = plt.subplots(
        n_plot_rows,
        events_per_row,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=footer_in / fig_height,
        top=1.0 - header_in / fig_height,
        hspace=0.72,
        wspace=0.25,
    )

    amp_title = "all amplitudes" if amplitude_uV is None else (
        f"amplitude {amplitude_uV[0]:g} to {amplitude_uV[1]:g} uV"
    )
    time_title = _event_window_title(
        event_time_window=event_time_window,
        pre_time_s=pre_time_s,
        post_time_window_s=post_time_window_s,
    )
    fig.suptitle(
        f"{folder.name}\n{len(events)} stim-triggered event(s), "
        f"{events_per_row} per row; {channel_selection_label(channel_names)} "
        f"{format_bandpass_plot_label(band_hz)}, {amp_title}{time_title}",
        fontsize=11,
        y=1.0 - 0.15 / fig_height,
    )
    fig.supylabel("filtered amplitude", x=0.02)

    summaries: list[dict[str, float | int | str]] = []
    used_envelope_any = False
    for event_index, event in enumerate(events):
        event_row = event_index // events_per_row
        event_col = event_index % events_per_row
        window = _event_display_window(
            event,
            sample_rate_hz,
            event_time_window=event_time_window,
            pre_time_s=pre_time_s,
            post_time_window_s=post_time_window_s,
        )

        for channel_index, (channel_name, filtered_uV) in enumerate(filtered_channel_data):
            axis_row = event_row * rows_per_event + channel_index
            ax = axes[axis_row, event_col]
            event_uV = filtered_uV[window.sample_slice]
            in_range = amplitude_mask(event_uV, amplitude_uV)
            hidden_samples = int(event_uV.size - np.count_nonzero(in_range))
            in_range = _apply_response_blank_mask(
                in_range, event_uV.size, sample_rate_hz, window
            )
            x_event, y_event, used_envelope = masked_envelope_for_display(
                event_uV,
                in_range,
                sample_rate_hz,
                max_points,
                start_time_s=window.data_start_s,
            )
            used_envelope_any = used_envelope_any or used_envelope

            if x_event.size:
                ax.plot(x_event, y_event, color="#d62728", linewidth=0.55)

            if amplitude_uV is not None:
                low_uV, high_uV = amplitude_uV
                ax.axhline(low_uV, color="#d62728", linewidth=0.5, alpha=0.3)
                ax.axhline(high_uV, color="#d62728", linewidth=0.5, alpha=0.3)
                ax.set_ylim(low_uV, high_uV)
                if hidden_samples:
                    # Never clip silently: say how many samples lie outside the window.
                    ax.text(
                        0.99,
                        0.92,
                        f"{hidden_samples} samples outside {low_uV:g}..{high_uV:g} uV hidden",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=5.5,
                        color="0.4",
                    )

            if channel_index == 0:
                ax.set_title(
                    f"onset {event.onset_s:g}s, dur {event.duration_s:g}s",
                    fontsize=8,
                )
            channel_label = f"{channel_name} *" if channel_name == stim_channel_name else channel_name
            ax.text(
                0.01,
                0.92,
                channel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.axvline(0.0, color="0.78", linewidth=0.65, zorder=0)
            ax.set_xlim(window.axis_bounds_s)
            if not has_stim_row and event_row == n_event_rows - 1 and channel_index == n_channels - 1:
                ax.set_xlabel("Time from stim trigger (s)", fontsize=7)
            summaries.append(
                {
                    "event_number": event.event_number,
                    "channel_name": channel_name,
                    "samples_total": int(event_uV.size),
                    "samples_in_amplitude_range": int(np.count_nonzero(in_range)),
                    "percent_in_amplitude_range": _percent_true(in_range),
                    "hidden_samples": hidden_samples,
                    "filtered_rms_uV": _rms(event_uV),
                }
            )

        if has_stim_row:
            stim_axis_row = event_row * rows_per_event + n_channels
            ax = axes[stim_axis_row, event_col]
            _plot_stim_row(
                ax,
                stim_uA,
                window.sample_slice,
                sample_rate_hz,
                max_points,
                window,
                stim_channel_name,
            )
            if event_row == n_event_rows - 1:
                ax.set_xlabel("Time from stim trigger (s)", fontsize=7)

    for event_index in range(len(events), n_event_rows * events_per_row):
        event_row = event_index // events_per_row
        event_col = event_index % events_per_row
        for row_index in range(rows_per_event):
            axes[event_row * rows_per_event + row_index, event_col].set_visible(False)

    if used_envelope_any:
        fig.text(
            0.985,
            0.08 / fig_height,
            "displayed as min/max envelope",
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
    return fig, summaries


def _event_display_window(
    event: StimTriggeredEvent,
    sample_rate_hz: float,
    event_time_window: tuple[float, float] | None = None,
    pre_time_s: float = 0.0,
    post_time_window_s: tuple[float, float | None] | None = None,
) -> EventDisplayWindow:
    if post_time_window_s is not None:
        pre_time_s = max(0.0, float(pre_time_s))
        post_start_s = max(0.0, float(post_time_window_s[0]))
        post_stop_value = post_time_window_s[1]
        post_stop_s = (
            event.duration_s
            if post_stop_value is None
            else min(event.duration_s, max(post_start_s, float(post_stop_value)))
        )
        axis_start_s = -pre_time_s
        axis_stop_s = post_stop_s
        response_blank_s = (0.0, post_start_s) if post_start_s > 0.0 else None
    else:
        response_blank_s = None
        if event_time_window is None:
            axis_start_s = 0.0
            axis_stop_s = event.duration_s
        else:
            axis_start_s = max(0.0, float(event_time_window[0]))
            axis_stop_s = min(event.duration_s, float(event_time_window[1]))

    if axis_stop_s <= axis_start_s:
        axis_start_s = min(axis_start_s, event.duration_s)
        axis_stop_s = axis_start_s

    event_sample_count = event.end_sample - event.start_sample
    start_sample = max(0, event.start_sample + int(round(axis_start_s * sample_rate_hz)))
    stop_sample = min(
        event.end_sample,
        event.start_sample + max(0, int(round(axis_stop_s * sample_rate_hz))),
    )
    stop_sample = max(start_sample, stop_sample)
    data_start_s = (start_sample - event.start_sample) / sample_rate_hz
    return EventDisplayWindow(
        sample_slice=slice(start_sample, stop_sample),
        data_start_s=data_start_s,
        axis_bounds_s=(axis_start_s, axis_stop_s),
        response_blank_s=response_blank_s,
    )


def _apply_response_blank_mask(
    mask: np.ndarray,
    n_samples: int,
    sample_rate_hz: float,
    window: EventDisplayWindow,
) -> np.ndarray:
    if window.response_blank_s is None or n_samples == 0:
        return mask

    blank_start_s, blank_stop_s = window.response_blank_s
    if blank_stop_s <= blank_start_s:
        return mask

    relative_time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz + window.data_start_s
    keep = ~((relative_time_s >= blank_start_s) & (relative_time_s < blank_stop_s))
    return mask & keep


def _plot_stim_row(
    ax,
    stim_uA: np.ndarray,
    event_slice: slice,
    sample_rate_hz: float,
    max_points: int,
    window: EventDisplayWindow,
    stim_channel_name: str | None,
) -> None:
    event_stim_uA = stim_uA[event_slice]
    if event_stim_uA.size:
        in_range = np.ones(event_stim_uA.shape, dtype=bool)
        x_stim, y_stim, used_envelope = masked_envelope_for_display(
            event_stim_uA,
            in_range,
            sample_rate_hz,
            max_points,
            start_time_s=window.data_start_s,
        )
        if x_stim.size:
            drawstyle = "default" if used_envelope else "steps-post"
            ax.plot(
                x_stim,
                y_stim,
                color="#d62728",
                linewidth=0.7,
                drawstyle=drawstyle,
                zorder=5,
                clip_on=False,
            )
        max_amp = float(np.nanmax(np.abs(event_stim_uA)))
    else:
        max_amp = 0.0

    y_limit = max(1.0, max_amp * 1.2)
    ax.axhline(0.0, color="0.45", linewidth=0.45, alpha=0.6)
    ax.set_ylim(-y_limit, y_limit)
    ax.axvline(0.0, color="0.78", linewidth=0.65, zorder=0)
    ax.set_xlim(window.axis_bounds_s)
    stim_label = "stim_data" if stim_channel_name is None else f"{stim_channel_name} stim"
    ax.text(
        0.01,
        0.86,
        stim_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_zorder(0)
    ax.spines["bottom"].set_zorder(0)
    ax.tick_params(labelsize=7)


def _event_window_title(
    event_time_window: tuple[float, float] | None,
    pre_time_s: float,
    post_time_window_s: tuple[float, float | None] | None,
) -> str:
    if post_time_window_s is not None:
        post_start_s, post_stop_s = post_time_window_s
        if post_stop_s is None:
            post_label = f"{post_start_s * 1e3:g}ms-end"
        elif post_start_s > 0:
            post_label = f"{post_start_s * 1e3:g}-{post_stop_s * 1e3:g}ms"
        else:
            post_label = f"{post_stop_s * 1e3:g}ms"
        return f", pre {pre_time_s * 1e3:g}ms, post {post_label}"
    if event_time_window is not None:
        return f", event time {event_time_window[0]:g}-{event_time_window[1]:g} s"
    return ""


def _percent_true(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask) * 100.0)


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


# -----------------------------------------------------------------------------
# Event window parsing and filename labels.
#
# These lived in wideband_function3_ui.py, where the batch runner could not
# reach them; batch_run_wideband_main_ui.py imports them from here since
# 2026-08-17 so both front ends name files identically ("pre100ms_post500ms").
# -----------------------------------------------------------------------------

def parse_pre_time_ms(value: float) -> float:
    pre_ms = float(value)
    if pre_ms < 0:
        raise ValueError("Pre time (ms) must be 0 or greater.")
    return pre_ms


def parse_post_time_ms(value: str) -> tuple[float, float | None]:
    cleaned = value.strip().lower().replace("ms", "")
    cleaned = re.sub(r"\s+", "", cleaned).translate(DASH_TRANSLATION)
    if cleaned in {"", "all", "end"}:
        return 0.0, None

    parts = cleaned.split("-", maxsplit=1)
    try:
        if len(parts) == 1:
            stop_ms = float(parts[0])
            if stop_ms <= 0:
                raise ValueError
            return 0.0, stop_ms

        start_ms = float(parts[0])
        stop_ms = float(parts[1])
        if start_ms < 0 or stop_ms <= start_ms:
            raise ValueError
        return start_ms, stop_ms
    except ValueError as exc:
        raise ValueError(
            "Post time (ms) must be a positive value like 500, or a range like 20-300."
        ) from exc


def event_window_label(pre_time_ms: float, post_window_ms: tuple[float, float | None]) -> str:
    pre_label = ms_label(pre_time_ms)
    post_start_ms, post_stop_ms = post_window_ms
    if post_stop_ms is None:
        post_label = f"{ms_number(post_start_ms)}toEnd"
    elif post_start_ms > 0:
        post_label = f"{ms_number(post_start_ms)}to{ms_label(post_stop_ms)}"
    else:
        post_label = ms_label(post_stop_ms)
    return f"pre{pre_label}_post{post_label}"


