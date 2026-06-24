#!/usr/bin/env python3
"""Plot one RHS channel's raw wideband trace with a biphasic stim pulse caption."""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


SAMPLES_PER_DATA_BLOCK = 128
RHS_MAGIC_NUMBER = 0xD69127AC


@dataclass(frozen=True)
class RhsHeader:
    path: Path
    version: tuple[int, int]
    sample_rate_hz: float
    stim_step_size_uA: float
    dc_amp_data_saved: bool
    amplifier_channels: list[str]
    board_adc_channels: list[str]
    board_dac_channels: list[str]
    board_dig_in_channels: list[str]
    board_dig_out_channels: list[str]
    header_bytes: int
    bytes_per_block: int
    num_data_blocks: int
    sample_count: int


@dataclass(frozen=True)
class RhsChannelFile:
    path: Path
    header: RhsHeader
    raw_uV: np.ndarray
    stim_uA: np.ndarray
    timestamp_gaps: int


def _read_exact(fid, n_bytes: int) -> bytes:
    data = fid.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError(f"Unexpected EOF while reading {n_bytes} bytes")
    return data


def _read_scalar(fid, fmt: str):
    return struct.unpack("<" + fmt, _read_exact(fid, struct.calcsize("<" + fmt)))[0]


def _read_qstring(fid) -> str:
    n_bytes = _read_scalar(fid, "I")
    if n_bytes == 0xFFFFFFFF:
        return ""
    if n_bytes == 0:
        return ""
    return _read_exact(fid, n_bytes).decode("utf-16le", errors="replace")


def _read_header(fid, path: Path, file_size: int) -> RhsHeader:
    magic_number = _read_scalar(fid, "I")
    if magic_number != RHS_MAGIC_NUMBER:
        raise ValueError(f"{path.name} is not an Intan RHS2000 file")

    version = (_read_scalar(fid, "h"), _read_scalar(fid, "h"))
    sample_rate_hz = float(_read_scalar(fid, "f"))

    _dsp_enabled = _read_scalar(fid, "h")
    _actual_dsp_cutoff_frequency = _read_scalar(fid, "f")
    _actual_lower_bandwidth = _read_scalar(fid, "f")
    _actual_lower_settle_bandwidth = _read_scalar(fid, "f")
    _actual_upper_bandwidth = _read_scalar(fid, "f")
    _desired_dsp_cutoff_frequency = _read_scalar(fid, "f")
    _desired_lower_bandwidth = _read_scalar(fid, "f")
    _desired_lower_settle_bandwidth = _read_scalar(fid, "f")
    _desired_upper_bandwidth = _read_scalar(fid, "f")
    _notch_filter_mode = _read_scalar(fid, "h")
    _desired_impedance_test_frequency = _read_scalar(fid, "f")
    _actual_impedance_test_frequency = _read_scalar(fid, "f")
    _amp_settle_mode = _read_scalar(fid, "h")
    _charge_recovery_mode = _read_scalar(fid, "h")

    stim_step_size_a = float(_read_scalar(fid, "f"))
    _charge_recovery_current_limit = _read_scalar(fid, "f")
    _charge_recovery_target_voltage = _read_scalar(fid, "f")

    _notes = (_read_qstring(fid), _read_qstring(fid), _read_qstring(fid))
    dc_amp_data_saved = bool(_read_scalar(fid, "h"))
    _board_mode = _read_scalar(fid, "h")
    _reference_channel = _read_qstring(fid)

    amplifier_channels: list[str] = []
    board_adc_channels: list[str] = []
    board_dac_channels: list[str] = []
    board_dig_in_channels: list[str] = []
    board_dig_out_channels: list[str] = []

    number_of_signal_groups = _read_scalar(fid, "h")
    for _signal_group in range(number_of_signal_groups):
        _signal_group_name = _read_qstring(fid)
        _signal_group_prefix = _read_qstring(fid)
        signal_group_enabled = _read_scalar(fid, "h")
        signal_group_num_channels = _read_scalar(fid, "h")
        _signal_group_num_amp_channels = _read_scalar(fid, "h")

        if signal_group_num_channels <= 0 or signal_group_enabled <= 0:
            continue

        for _signal_channel in range(signal_group_num_channels):
            native_channel_name = _read_qstring(fid)
            _custom_channel_name = _read_qstring(fid)
            _native_order = _read_scalar(fid, "h")
            _custom_order = _read_scalar(fid, "h")
            signal_type = _read_scalar(fid, "h")
            channel_enabled = _read_scalar(fid, "h")
            _chip_channel = _read_scalar(fid, "h")
            _command_stream = _read_scalar(fid, "h")
            _board_stream = _read_scalar(fid, "h")
            _voltage_trigger_mode = _read_scalar(fid, "h")
            _voltage_threshold = _read_scalar(fid, "h")
            _digital_trigger_channel = _read_scalar(fid, "h")
            _digital_edge_polarity = _read_scalar(fid, "h")
            _electrode_impedance_magnitude = _read_scalar(fid, "f")
            _electrode_impedance_phase = _read_scalar(fid, "f")

            if not channel_enabled:
                continue
            if signal_type == 0:
                amplifier_channels.append(native_channel_name)
            elif signal_type == 3:
                board_adc_channels.append(native_channel_name)
            elif signal_type == 4:
                board_dac_channels.append(native_channel_name)
            elif signal_type == 5:
                board_dig_in_channels.append(native_channel_name)
            elif signal_type == 6:
                board_dig_out_channels.append(native_channel_name)

    num_amp_channels = len(amplifier_channels)
    bytes_per_block = SAMPLES_PER_DATA_BLOCK * 4
    amp_words_per_block = SAMPLES_PER_DATA_BLOCK * num_amp_channels
    if dc_amp_data_saved:
        bytes_per_block += amp_words_per_block * 2 * 3
    else:
        bytes_per_block += amp_words_per_block * 2 * 2
    bytes_per_block += SAMPLES_PER_DATA_BLOCK * 2 * len(board_adc_channels)
    bytes_per_block += SAMPLES_PER_DATA_BLOCK * 2 * len(board_dac_channels)
    if board_dig_in_channels:
        bytes_per_block += SAMPLES_PER_DATA_BLOCK * 2
    if board_dig_out_channels:
        bytes_per_block += SAMPLES_PER_DATA_BLOCK * 2

    header_bytes = fid.tell()
    bytes_remaining = file_size - header_bytes
    if bytes_remaining < 0 or bytes_remaining % bytes_per_block != 0:
        raise ValueError(
            f"{path.name}: data section is not an even number of RHS data blocks"
        )
    num_data_blocks = bytes_remaining // bytes_per_block
    sample_count = num_data_blocks * SAMPLES_PER_DATA_BLOCK

    return RhsHeader(
        path=path,
        version=version,
        sample_rate_hz=sample_rate_hz,
        stim_step_size_uA=stim_step_size_a / 1.0e-6,
        dc_amp_data_saved=dc_amp_data_saved,
        amplifier_channels=amplifier_channels,
        board_adc_channels=board_adc_channels,
        board_dac_channels=board_dac_channels,
        board_dig_in_channels=board_dig_in_channels,
        board_dig_out_channels=board_dig_out_channels,
        header_bytes=header_bytes,
        bytes_per_block=bytes_per_block,
        num_data_blocks=num_data_blocks,
        sample_count=sample_count,
    )


def _decode_stim_data(stim_raw: np.ndarray, stim_step_size_uA: float) -> np.ndarray:
    stim_raw = stim_raw.astype(np.uint16, copy=False)
    magnitude = (stim_raw & np.uint16(0x00FF)).astype(np.float32)
    is_negative_phase = (stim_raw & np.uint16(0x0100)) != 0
    sign = np.where(is_negative_phase, -1.0, 1.0).astype(np.float32)
    return magnitude * sign * np.float32(stim_step_size_uA)


def read_rhs_channel(path: Path, channel_name: str) -> RhsChannelFile:
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        header = _read_header(fid, path, file_size)
        try:
            channel_index = header.amplifier_channels.index(channel_name)
        except ValueError as exc:
            channels = ", ".join(header.amplifier_channels)
            raise ValueError(
                f"{path.name}: channel {channel_name!r} was not found. "
                f"Available amplifier channels: {channels}"
            ) from exc

        raw_uV = np.empty(header.sample_count, dtype=np.float32)
        stim_uA = np.empty(header.sample_count, dtype=np.float32)

        n_amp_channels = len(header.amplifier_channels)
        amp_matrix_bytes = SAMPLES_PER_DATA_BLOCK * n_amp_channels * 2
        amp_offset = SAMPLES_PER_DATA_BLOCK * 4
        dc_offset = amp_offset + amp_matrix_bytes
        stim_offset = dc_offset + (amp_matrix_bytes if header.dc_amp_data_saved else 0)

        timestamp_gaps = 0
        previous_timestamp: int | None = None

        for block_number in range(header.num_data_blocks):
            block = _read_exact(fid, header.bytes_per_block)
            timestamps = np.frombuffer(
                block, dtype="<i4", count=SAMPLES_PER_DATA_BLOCK, offset=0
            )
            if timestamps.size:
                timestamp_gaps += int(np.count_nonzero(np.diff(timestamps) != 1))
                if previous_timestamp is not None and timestamps[0] != previous_timestamp + 1:
                    timestamp_gaps += 1
                previous_timestamp = int(timestamps[-1])

            sample_start = block_number * SAMPLES_PER_DATA_BLOCK
            sample_end = sample_start + SAMPLES_PER_DATA_BLOCK
            channel_slice = slice(
                channel_index * SAMPLES_PER_DATA_BLOCK,
                (channel_index + 1) * SAMPLES_PER_DATA_BLOCK,
            )

            amp_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=SAMPLES_PER_DATA_BLOCK * n_amp_channels,
                offset=amp_offset,
            )
            amp_raw = amp_flat[channel_slice]
            raw_uV[sample_start:sample_end] = (
                amp_raw.astype(np.float32) - np.float32(32768.0)
            ) * np.float32(0.195)

            stim_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=SAMPLES_PER_DATA_BLOCK * n_amp_channels,
                offset=stim_offset,
            )
            stim_raw = stim_flat[channel_slice]
            stim_uA[sample_start:sample_end] = _decode_stim_data(
                stim_raw, header.stim_step_size_uA
            )

    return RhsChannelFile(
        path=path,
        header=header,
        raw_uV=raw_uV,
        stim_uA=stim_uA,
        timestamp_gaps=timestamp_gaps,
    )


def read_rhs_folder(folder: Path, channel_name: str) -> tuple[np.ndarray, np.ndarray, float, list[RhsChannelFile]]:
    rhs_files = sorted(folder.glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError(f"No .rhs files found in {folder}")

    loaded = [read_rhs_channel(path, channel_name) for path in rhs_files]
    sample_rates = {round(item.header.sample_rate_hz, 9) for item in loaded}
    if len(sample_rates) != 1:
        raise ValueError(f"Cannot concatenate files with different sample rates: {sample_rates}")

    raw_uV = np.concatenate([item.raw_uV for item in loaded])
    stim_uA = np.concatenate([item.stim_uA for item in loaded])
    sample_rate_hz = loaded[0].header.sample_rate_hz
    return raw_uV, stim_uA, sample_rate_hz, loaded


def default_output_path(folder: Path, channel_name: str) -> Path:
    safe_channel = _safe_channel_label(channel_name)
    return folder / f"raw_wideband_{safe_channel}.png"


def _safe_channel_label(channel_text: str) -> str:
    safe = channel_text.strip().replace("-", "").replace(" ", "_")
    safe = safe.replace(",", "_").replace(":", "_").replace("/", "_")
    return safe or "channels"


def parse_channel_selection(text: str) -> list[str]:
    """Parse channel entries like 'A-014', 'A-014-16', or 'A-014, A-016'."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter at least one channel, e.g. A-014.")

    parts = [part.strip() for part in re.split(r"[,;\n]+", cleaned) if part.strip()]
    if len(parts) == 1 and " to " not in parts[0].lower():
        space_parts = [part for part in parts[0].split() if part]
        if len(space_parts) > 1:
            parts = space_parts

    channels: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for channel in _expand_channel_part(part):
            if channel not in seen:
                channels.append(channel)
                seen.add(channel)

    if not channels:
        raise ValueError("Enter at least one channel, e.g. A-014.")
    return channels


def _expand_channel_part(part: str) -> list[str]:
    normalized = part.strip().replace("–", "-").replace("—", "-").replace("−", "-")
    range_match = re.fullmatch(
        r"([A-Za-z]+)-?(\d+)\s*(?:-|:|\bto\b)\s*(?:([A-Za-z]+)-?)?(\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if range_match:
        prefix = range_match.group(1).upper()
        end_prefix = range_match.group(3)
        if end_prefix and end_prefix.upper() != prefix:
            raise ValueError(f"Channel range prefixes must match: {part}")
        start_text = range_match.group(2)
        end_text = range_match.group(4)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        width = max(len(start_text), len(end_text))
        return [f"{prefix}-{number:0{width}d}" for number in range(start, end + step, step)]

    single_match = re.fullmatch(r"([A-Za-z]+)-?(\d+)", normalized)
    if single_match:
        prefix = single_match.group(1).upper()
        number_text = single_match.group(2)
        return [f"{prefix}-{int(number_text):0{len(number_text)}d}"]

    raise ValueError(f"Could not understand channel entry: {part}")


def channel_selection_label(channels: Sequence[str]) -> str:
    if not channels:
        return "channels"
    if len(channels) == 1:
        return channels[0]
    return f"{channels[0]}_to_{channels[-1]}"


def parse_time_window(text: str) -> tuple[float, float] | None:
    """Parse a display time window like '0-10 s', or return None for all."""
    cleaned = text.strip().lower()
    if cleaned in {"", "all", "full", "none"}:
        return None

    normalized = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    match = re.search(
        r"(\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:-|to|,|:)\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Enter time as 'all' or a range, e.g. 0-10 s.")

    start_s, end_s = sorted((float(match.group(1)), float(match.group(2))))
    if start_s < 0 or end_s <= start_s:
        raise ValueError("Time range must start at 0 s or later and have two different values.")
    return start_s, end_s


def sample_slice_for_time_window(
    n_samples: int,
    sample_rate_hz: float,
    time_window: tuple[float, float] | None,
) -> tuple[slice, tuple[float, float]]:
    """Convert a time window to a sample slice and clamped display bounds."""
    if n_samples <= 0:
        raise ValueError("No samples are available to plot.")

    total_duration_s = n_samples / sample_rate_hz
    if time_window is None:
        return slice(0, n_samples), (0.0, total_duration_s)

    start_s, end_s = time_window
    if start_s >= total_duration_s:
        raise ValueError(
            f"Time window starts after the recording ends ({total_duration_s:.3f} s)."
        )

    start_index = max(0, int(math.floor(start_s * sample_rate_hz)))
    end_index = min(n_samples, int(math.ceil(end_s * sample_rate_hz)))
    if end_index <= start_index:
        raise ValueError("Time window contains no samples.")

    return (
        slice(start_index, end_index),
        (start_index / sample_rate_hz, end_index / sample_rate_hz),
    )


def time_window_label(time_window: tuple[float, float] | None) -> str:
    if time_window is None:
        return "allTime"
    return f"{time_window[0]:g}to{time_window[1]:g}s"


def find_pulse_segments(stim_uA: np.ndarray) -> list[tuple[int, int]]:
    nonzero = np.flatnonzero(stim_uA != 0)
    if nonzero.size == 0:
        return []

    split_after = np.flatnonzero(np.diff(nonzero) > 1)
    starts = np.r_[nonzero[0], nonzero[split_after + 1]]
    ends = np.r_[nonzero[split_after] + 1, nonzero[-1] + 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def find_stim_channel_in_data(
    channel_data: Sequence[tuple[str, np.ndarray, np.ndarray]],
    display_slice: slice,
) -> tuple[str, np.ndarray, list[tuple[int, int]]] | None:
    """Return the first selected channel whose displayed stim_data is nonzero."""
    for channel_name, _raw_uV, stim_uA in channel_data:
        display_stim_uA = stim_uA[display_slice]
        pulse_segments = find_pulse_segments(display_stim_uA)
        if pulse_segments:
            return channel_name, display_stim_uA, pulse_segments
    return None


@dataclass(frozen=True)
class StimPulseMetrics:
    amplitude_uA: float
    phase_duration_ms: float
    first_phase_sign: float


def extract_biphasic_pulse_metrics(
    stim_uA: np.ndarray, pulse_start: int, pulse_end: int, sample_rate_hz: float
) -> StimPulseMetrics:
    pulse = stim_uA[pulse_start:pulse_end]
    nonzero = pulse[pulse != 0]
    if nonzero.size == 0:
        return StimPulseMetrics(0.0, 0.0, 1.0)

    signs = np.sign(nonzero)
    phase_sign = float(signs[0]) if signs[0] != 0 else 1.0
    first_phase_samples = int(np.argmax(signs != signs[0]))
    if first_phase_samples == 0:
        first_phase_samples = int(nonzero.size)

    return StimPulseMetrics(
        amplitude_uA=float(np.max(np.abs(nonzero))),
        phase_duration_ms=first_phase_samples / sample_rate_hz * 1.0e3,
        first_phase_sign=phase_sign,
    )


def _format_duration_ms(duration_ms: float) -> str:
    if duration_ms >= 1:
        return f"{duration_ms:.3g}ms"
    return f"{duration_ms:.2g}ms"


def draw_biphasic_pulse_caption(inset, metrics: StimPulseMetrics) -> None:
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    inset.patch.set_edgecolor("white")

    amp = metrics.amplitude_uA
    phase_ms = metrics.phase_duration_ms
    if amp <= 0 or phase_ms <= 0:
        inset.text(0.5, 0.5, "No nonzero\nstim_data found", ha="center", va="center")
        inset.set_axis_off()
        return

    sign = 1.0 if metrics.first_phase_sign >= 0 else -1.0
    phase_width = 0.18
    t0 = 0.0
    t1 = phase_width
    t2 = 2.0 * phase_width
    lead = 0.28
    trail = 0.28

    x = np.array([-lead, t0, t0, t1, t1, t2, t2, t2 + trail])
    y = np.array([0, 0, sign * amp, sign * amp, -sign * amp, -sign * amp, 0, 0])

    red = "#ff0000"
    inset.plot(x, y, color=red, linewidth=1.3, solid_joinstyle="miter")
    inset.set_xlim(-lead * 1.25, t2 + trail * 1.2)
    inset.set_ylim(-amp * 1.55, amp * 1.75)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)

    top_y = sign * amp
    amp_label_x = -0.16
    inset.annotate(
        "",
        xy=(amp_label_x, top_y),
        xytext=(amp_label_x, 0),
        arrowprops={"arrowstyle": "<->", "color": red, "linewidth": 1.0},
    )
    inset.text(
        amp_label_x - 0.05,
        top_y * 0.5,
        f"{amp:.0f}uA",
        color=red,
        fontsize=7,
        fontweight="bold",
        ha="right",
        va="center",
    )

    phase_arrow_y = top_y + sign * amp * 0.32
    inset.annotate(
        "",
        xy=(t0, phase_arrow_y),
        xytext=(t1, phase_arrow_y),
        arrowprops={"arrowstyle": "<->", "color": red, "linewidth": 1.0},
    )
    inset.text(
        (t0 + t1) * 0.5,
        phase_arrow_y + sign * amp * 0.13,
        _format_duration_ms(phase_ms),
        color=red,
        fontsize=7,
        fontweight="bold",
        ha="center",
        va="bottom" if sign >= 0 else "top",
    )


def envelope_for_display(
    y: np.ndarray,
    sample_rate_hz: float,
    max_points: int,
    start_time_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    n_samples = y.size
    if max_points <= 0 or n_samples <= max_points:
        x = np.arange(n_samples, dtype=np.float64) / sample_rate_hz + start_time_s
        return x, y, False

    target_bins = max(2, max_points // 2)
    bin_size = int(math.ceil(n_samples / target_bins))
    n_bins = int(math.ceil(n_samples / bin_size))
    padded_length = n_bins * bin_size
    pad_count = padded_length - n_samples
    if pad_count:
        y_work = np.pad(y, (0, pad_count), mode="edge")
    else:
        y_work = y

    y_bins = y_work.reshape(n_bins, bin_size)
    y_min = y_bins.min(axis=1)
    y_max = y_bins.max(axis=1)
    bin_starts = np.arange(n_bins, dtype=np.float64) * bin_size
    bin_ends = np.minimum(bin_starts + bin_size, n_samples)
    x_centers = (bin_starts + bin_ends - 1.0) * 0.5 / sample_rate_hz + start_time_s
    x_plot = np.repeat(x_centers, 2)
    y_plot = np.column_stack((y_min, y_max)).ravel()
    return x_plot, y_plot, True


def plot_raw_with_stim_pulse(
    raw_uV: np.ndarray,
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    channel_name: str,
    folder: Path,
    output_path: Path,
    max_points: int,
    pulse_number: int,
    add_save_button: bool = True,
    time_window: tuple[float, float] | None = None,
) -> plt.Figure:
    display_slice, display_bounds = sample_slice_for_time_window(
        raw_uV.size, sample_rate_hz, time_window
    )
    display_raw_uV = raw_uV[display_slice]
    display_stim_uA = stim_uA[display_slice]
    x_s, y_uV, used_envelope = envelope_for_display(
        display_raw_uV, sample_rate_hz, max_points, start_time_s=display_bounds[0]
    )
    pulse_segments = find_pulse_segments(display_stim_uA)

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.78)
    ax.plot(x_s, y_uV, color="#1f6ed4", linewidth=0.75, label=f"{channel_name} raw wideband")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("raw wideband (uV)")
    time_title = "" if time_window is None else f", time {display_bounds[0]:g}-{display_bounds[1]:g} s"
    ax.set_title(
        f"{folder.name}\n{channel_name} raw wideband vs time{time_title} "
        "with biphasic stim pulse caption",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(display_bounds)

    if pulse_segments:
        pulse_index = min(max(1, pulse_number), len(pulse_segments)) - 1
        pulse_start, pulse_end = pulse_segments[pulse_index]
        metrics = extract_biphasic_pulse_metrics(
            display_stim_uA, pulse_start, pulse_end, sample_rate_hz
        )
        inset = fig.add_axes([0.82, 0.80, 0.14, 0.13])
        inset.set_zorder(10)
        draw_biphasic_pulse_caption(inset, metrics)
    else:
        inset = fig.add_axes([0.82, 0.80, 0.14, 0.13])
        inset.text(0.5, 0.5, "No nonzero\nstim_data found", ha="center", va="center")
        inset.set_axis_off()

    if used_envelope:
        ax.text(
            0.995,
            0.01,
            "displayed as min/max envelope",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    if add_save_button:
        button_ax = fig.add_axes([0.07, 0.86, 0.08, 0.05])
        save_button = Button(button_ax, "Save PNG")
        status_text = fig.text(
            0.16,
            0.885,
            f"Target: {output_path}",
            ha="left",
            va="center",
            fontsize=8,
            color="0.35",
        )

        def save_png(_event=None) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_name(
                f".{output_path.stem}.tmp{output_path.suffix}"
            )
            button_ax.set_visible(False)
            status_text.set_visible(False)
            try:
                fig.savefig(temp_path, dpi=220)
                os.replace(temp_path, output_path)
            finally:
                button_ax.set_visible(True)
                status_text.set_visible(True)
                if temp_path.exists():
                    temp_path.unlink()
            status_text.set_text(f"Saved: {output_path}")
            status_text.set_color("0.25")
            fig.canvas.draw_idle()
            print(f"Saved PNG: {output_path}")

        save_button.on_clicked(save_png)
        fig._save_png_button = save_button
        fig._save_png_callback = save_png
    return fig


def plot_raw_channels_with_stim_pulse(
    channel_data: Sequence[tuple[str, np.ndarray, np.ndarray]],
    sample_rate_hz: float,
    folder: Path,
    output_path: Path,
    max_points: int,
    pulse_number: int,
    time_window: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot one or more raw wideband channels as stacked traces."""
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
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.78, hspace=0.24)
    time_title = "" if time_window is None else f", time {display_bounds[0]:g}-{display_bounds[1]:g} s"
    fig.suptitle(
        f"{folder.name}\n{channel_selection_label(channel_names)} raw wideband vs time"
        f"{time_title} with biphasic stim pulse caption",
        fontsize=11,
        y=0.96,
    )
    fig.supylabel("raw wideband (uV)", x=0.035)

    stim_channel_info = find_stim_channel_in_data(channel_data, display_slice)
    stim_channel_name = stim_channel_info[0] if stim_channel_info is not None else None

    used_envelope_any = False
    for ax, (channel_name, raw_uV, _stim_uA) in zip(axes_flat, channel_data):
        display_raw_uV = raw_uV[display_slice]
        x_s, y_uV, used_envelope = envelope_for_display(
            display_raw_uV,
            sample_rate_hz,
            max_points,
            start_time_s=display_bounds[0],
        )
        used_envelope_any = used_envelope_any or used_envelope
        ax.plot(x_s, y_uV, color="#1f6ed4", linewidth=0.75)
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

    axes_flat[-1].set_xlabel("Time (s)")

    if stim_channel_info is not None:
        _stim_channel_name, display_stim_uA, pulse_segments = stim_channel_info
        pulse_index = min(max(1, pulse_number), len(pulse_segments)) - 1
        pulse_start, pulse_end = pulse_segments[pulse_index]
        metrics = extract_biphasic_pulse_metrics(
            display_stim_uA, pulse_start, pulse_end, sample_rate_hz
        )
        inset = fig.add_axes([0.82, 0.80, 0.14, 0.13])
        inset.set_zorder(10)
        draw_biphasic_pulse_caption(inset, metrics)
    else:
        inset = fig.add_axes([0.82, 0.80, 0.14, 0.13])
        inset.text(0.5, 0.5, "No nonzero\nstim_data found", ha="center", va="center")
        inset.set_axis_off()

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

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Intan RHS files from a folder and plot raw wideband voltage "
            "for one amplifier channel, plus a caption schematic of one "
            "decoded RHS biphasic stim_data pulse."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .rhs files.",
    )
    parser.add_argument(
        "--channel",
        default="A-014",
        help="Amplifier/stim channel to plot. Default: A-014",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Default: save directly inside the selected RHS data folder.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=600_000,
        help="Maximum points used for the full-trace display envelope. Use 0 to plot every sample.",
    )
    parser.add_argument(
        "--pulse-number",
        type=int,
        default=1,
        help="Which nonzero stim_data pulse to show in the inset. Default: 1",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the interactive figure but do not display it. Useful for quick syntax checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if args.output is None:
        output = default_output_path(folder, args.channel)
    else:
        output = args.output.expanduser().resolve()

    raw_uV, stim_uA, sample_rate_hz, loaded = read_rhs_folder(folder, args.channel)
    fig = plot_raw_with_stim_pulse(
        raw_uV=raw_uV,
        stim_uA=stim_uA,
        sample_rate_hz=sample_rate_hz,
        channel_name=args.channel,
        folder=folder,
        output_path=output,
        max_points=args.max_points,
        pulse_number=args.pulse_number,
    )

    total_timestamp_gaps = sum(item.timestamp_gaps for item in loaded)
    print(f"Read {len(loaded)} RHS file(s) from {folder}")
    print(f"Channel: {args.channel}")
    print(f"Sample rate: {sample_rate_hz:g} Hz")
    print(f"Samples: {raw_uV.size} ({raw_uV.size / sample_rate_hz:.3f} s)")
    print(f"Stim pulses found: {len(find_pulse_segments(stim_uA))}")
    print(f"Timestamp gaps found: {total_timestamp_gaps}")
    print("Interactive plot ready. Click the 'Save PNG' button to write the file.")
    print(f"Manual save target: {output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
