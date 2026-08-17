#!/usr/bin/env python3
"""Conservative stim-triggered analysis for the 2026-06-24 Pt stimulation data."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rhs_raw_wideband_with_stim_legend import (
    SAMPLES_PER_DATA_BLOCK,
    _decode_stim_data,
    _read_exact,
    _read_header,
)
from plot_rhs_stim_triggered_events import find_stim_trigger_onsets


CHANNELS = [f"A-{i:03d}" for i in range(24, 32)]
PRE_S = 0.2
POST_S = 0.8
ARTIFACT_END_S = 0.005
BASELINE_WINDOW_S = (-0.2, -0.02)
EARLY_WINDOW_S = (0.005, 0.05)
LATE_WINDOW_S = (0.05, 0.3)
RECOVERY_WINDOW_S = (0.3, 0.8)


@dataclass(frozen=True)
class RhsAllChannels:
    path: Path
    sample_rate_hz: float
    channels: list[str]
    raw_uV: np.ndarray
    stim_uA: np.ndarray
    timestamp_gaps: int


def read_rhs_all_channels(path: Path, wanted_channels: list[str] | None = None) -> RhsAllChannels:
    """Read enabled amplifier and stim waveforms for selected RHS channels."""
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        header = _read_header(fid, path, file_size)
        wanted_channels = wanted_channels or header.amplifier_channels
        channel_indices = [
            header.amplifier_channels.index(channel)
            for channel in wanted_channels
            if channel in header.amplifier_channels
        ]
        channels = [header.amplifier_channels[i] for i in channel_indices]
        if not channels:
            raise ValueError(f"{path}: none of {wanted_channels} were present")

        n_samples = header.sample_count
        n_selected = len(channels)
        raw_uV = np.empty((n_samples, n_selected), dtype=np.float32)
        stim_uA = np.empty((n_samples, n_selected), dtype=np.float32)

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

            amp_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=SAMPLES_PER_DATA_BLOCK * n_amp_channels,
                offset=amp_offset,
            ).reshape(n_amp_channels, SAMPLES_PER_DATA_BLOCK)
            stim_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=SAMPLES_PER_DATA_BLOCK * n_amp_channels,
                offset=stim_offset,
            ).reshape(n_amp_channels, SAMPLES_PER_DATA_BLOCK)

            raw_block = amp_flat[channel_indices].T.astype(np.float32)
            raw_uV[sample_start:sample_end] = (raw_block - np.float32(32768.0)) * np.float32(0.195)
            stim_block = stim_flat[channel_indices].T
            stim_uA[sample_start:sample_end] = _decode_stim_data(
                stim_block, header.stim_step_size_uA
            )

    return RhsAllChannels(
        path=path,
        sample_rate_hz=header.sample_rate_hz,
        channels=channels,
        raw_uV=raw_uV,
        stim_uA=stim_uA,
        timestamp_gaps=timestamp_gaps,
    )


def read_rhs_folder_all(folder: Path) -> RhsAllChannels:
    parts = [read_rhs_all_channels(path, CHANNELS) for path in sorted(folder.glob("*.rhs"))]
    if not parts:
        raise FileNotFoundError(f"No RHS files found in {folder}")
    sample_rates = {round(part.sample_rate_hz, 9) for part in parts}
    if len(sample_rates) != 1:
        raise ValueError(f"{folder}: mixed sample rates {sample_rates}")
    channels = parts[0].channels
    if any(part.channels != channels for part in parts):
        raise ValueError(f"{folder}: channel list changed across files")
    return RhsAllChannels(
        path=folder,
        sample_rate_hz=parts[0].sample_rate_hz,
        channels=channels,
        raw_uV=np.concatenate([part.raw_uV for part in parts], axis=0),
        stim_uA=np.concatenate([part.stim_uA for part in parts], axis=0),
        timestamp_gaps=sum(part.timestamp_gaps for part in parts),
    )


def parse_folder_metadata(folder_name: str) -> dict[str, str | float | int | None]:
    meta: dict[str, str | float | int | None] = {
        "stim_channel_name": None,
        "folder_polarity": None,
        "folder_current_uA": None,
        "folder_frequency_hz": None,
        "folder_pulses": None,
        "folder_phase_us": None,
    }
    channel = re.search(r"\b(A-\d{3})\b", folder_name)
    if channel:
        meta["stim_channel_name"] = channel.group(1)
    polarity = re.search(r"\b(cathodic|anodic)\s+first\b", folder_name, re.I)
    if polarity:
        meta["folder_polarity"] = polarity.group(1).lower() + " first"
    current = re.search(r"(\d+(?:\.\d+)?)\s*uA", folder_name, re.I)
    if current:
        meta["folder_current_uA"] = float(current.group(1))
    freq = re.search(r"(\d+(?:\.\d+)?)\s*Hz", folder_name, re.I)
    if freq:
        meta["folder_frequency_hz"] = float(freq.group(1))
    pulses = re.search(r"(\d+)\s+pulses", folder_name, re.I)
    if pulses:
        meta["folder_pulses"] = int(pulses.group(1))
    phase = re.search(r"(\d+(?:\.\d+)?)\s*us", folder_name, re.I)
    if phase:
        meta["folder_phase_us"] = float(phase.group(1))
    return meta


def parse_active_stim_from_settings(folder: Path) -> dict[str, str | float | int | None]:
    settings = folder / "settings.xml"
    result: dict[str, str | float | int | None] = {}
    if not settings.exists():
        return result
    try:
        root = ET.parse(settings).getroot()
    except ET.ParseError:
        return result
    for stim_channel in root.findall(".//StimChannel"):
        if stim_channel.attrib.get("StimEnabled") != "True":
            continue
        attr = stim_channel.attrib
        result = {
            "settings_stim_channel": attr.get("NativeChannelName"),
            "settings_polarity": attr.get("Polarity"),
            "settings_first_amp_uA": as_float(attr.get("FirstPhaseAmplitudeMicroAmps")),
            "settings_second_amp_uA": as_float(attr.get("SecondPhaseAmplitudeMicroAmps")),
            "settings_first_phase_us": as_float(attr.get("FirstPhaseDurationMicroseconds")),
            "settings_second_phase_us": as_float(attr.get("SecondPhaseDurationMicroseconds")),
            "settings_interphase_us": as_float(attr.get("InterphaseDelayMicroseconds")),
            "settings_num_pulses": as_int(attr.get("NumberOfStimPulses")),
            "settings_train_period_us": as_float(attr.get("PulseTrainPeriodMicroseconds")),
            "settings_refractory_us": as_float(attr.get("RefractoryPeriodMicroseconds")),
            "settings_post_amp_settle_us": as_float(attr.get("PostStimAmpSettleMicroseconds")),
            "settings_charge_recovery": attr.get("EnableChargeRecovery"),
            "settings_charge_recovery_on_us": as_float(attr.get("PostStimChargeRecovOnMicroseconds")),
            "settings_charge_recovery_off_us": as_float(attr.get("PostStimChargeRecovOffMicroseconds")),
        }
        break
    return result


def as_float(value: str | None) -> float | None:
    try:
        return None if value is None else float(value)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    try:
        return None if value is None else int(float(value))
    except ValueError:
        return None


def identify_stim_channel(stim_uA: np.ndarray, channels: list[str]) -> tuple[str | None, int | None]:
    counts = np.count_nonzero(stim_uA != 0, axis=0)
    if int(np.max(counts)) == 0:
        return None, None
    index = int(np.argmax(counts))
    return channels[index], index


def interpolate_blank_windows(data: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    """Replace artifact windows by linear interpolation before filtering."""
    cleaned = data.astype(np.float64, copy=True)
    n_samples = cleaned.shape[0]
    for start, stop in windows:
        start = max(1, int(start))
        stop = min(n_samples - 2, int(stop))
        if stop <= start:
            continue
        y0 = cleaned[start - 1]
        y1 = cleaned[stop + 1]
        alpha = np.linspace(0.0, 1.0, stop - start, endpoint=False)[:, None]
        cleaned[start:stop] = y0 + (y1 - y0) * alpha
    return cleaned


def make_blank_windows(trigger_onsets: np.ndarray, sample_rate_hz: float, stim_uA: np.ndarray) -> list[tuple[int, int]]:
    """Blank from stimulus onset through 5 ms after the last nonzero sample in a pulse/train."""
    windows: list[tuple[int, int]] = []
    post_pad = int(round(ARTIFACT_END_S * sample_rate_hz))
    nonzero = stim_uA != 0
    n_samples = stim_uA.size
    for onset in trigger_onsets:
        onset = int(onset)
        end = onset
        # If this is a short pulse train, carry the window across all nearby nonzero phases.
        gap_limit = int(round(0.010 * sample_rate_hz))
        last_nonzero = onset
        i = onset
        gap = 0
        while i < n_samples and gap <= gap_limit:
            if nonzero[i]:
                last_nonzero = i
                gap = 0
            else:
                gap += 1
            i += 1
        end = max(last_nonzero + post_pad, onset + post_pad)
        windows.append((onset, min(n_samples, end)))
    return merge_windows(windows)


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for start, stop in windows[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def bandpass(data: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def epoch_data(data: np.ndarray, triggers: np.ndarray, sample_rate_hz: float, pre_s: float, post_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_n = int(round(pre_s * sample_rate_hz))
    post_n = int(round(post_s * sample_rate_hz))
    valid = triggers[(triggers >= pre_n) & (triggers + post_n < data.shape[0])]
    if valid.size == 0:
        return np.empty((0, pre_n + post_n, data.shape[1])), valid, np.empty(0)
    offsets = np.arange(-pre_n, post_n, dtype=np.int64)
    epochs = data[valid[:, None] + offsets[None, :]]
    t = offsets / sample_rate_hz
    return epochs, valid, t


def window_mask(t: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    return (t >= window[0]) & (t < window[1])


def robust_sigma(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    sigma = mad / 0.67448975
    sigma[sigma <= 1e-12] = np.std(x, axis=0)[sigma <= 1e-12]
    sigma[sigma <= 1e-12] = 1.0
    return sigma


def detect_negative_spikes(spike_band: np.ndarray, sample_rate_hz: float, blank_windows: list[tuple[int, int]]) -> list[np.ndarray]:
    sigma = robust_sigma(spike_band)
    thresholds = -4.5 * sigma
    refractory = int(round(0.001 * sample_rate_hz))
    spike_times: list[np.ndarray] = []
    artifact_mask = np.zeros(spike_band.shape[0], dtype=bool)
    for start, stop in blank_windows:
        artifact_mask[start:stop] = True
    for ch in range(spike_band.shape[1]):
        trace = spike_band[:, ch]
        crossing = np.flatnonzero((trace[1:] < thresholds[ch]) & (trace[:-1] >= thresholds[ch])) + 1
        crossing = crossing[~artifact_mask[crossing]]
        if crossing.size == 0:
            spike_times.append(crossing)
            continue
        kept = [int(crossing[0])]
        last = int(crossing[0])
        for idx in crossing[1:]:
            idx = int(idx)
            if idx - last >= refractory:
                kept.append(idx)
                last = idx
            elif trace[idx] < trace[last]:
                kept[-1] = idx
                last = idx
        spike_times.append(np.asarray(kept, dtype=np.int64))
    return spike_times


def count_spikes_around_triggers(spike_times: np.ndarray, triggers: np.ndarray, sample_rate_hz: float, window: tuple[float, float]) -> np.ndarray:
    starts = triggers + int(round(window[0] * sample_rate_hz))
    stops = triggers + int(round(window[1] * sample_rate_hz))
    counts = np.empty(triggers.size, dtype=np.int32)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        counts[i] = int(np.searchsorted(spike_times, stop, side="left") - np.searchsorted(spike_times, start, side="left"))
    return counts


def plot_session_average(
    out_path: Path,
    session_name: str,
    channels: list[str],
    stim_channel: str,
    t: np.ndarray,
    lfp_epochs: np.ndarray,
    spike_counts: dict[str, np.ndarray],
    meta: dict[str, str | float | int | None],
) -> None:
    baseline_mask = window_mask(t, (-0.1, -0.02))
    avg = lfp_epochs - np.mean(lfp_epochs[:, baseline_mask, :], axis=1, keepdims=True)
    mean = np.mean(avg, axis=0)
    sem = np.std(avg, axis=0, ddof=1) / math.sqrt(max(1, avg.shape[0])) if avg.shape[0] > 1 else np.zeros_like(mean)

    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 1.8 * n_channels + 2.2), sharex=True)
    if n_channels == 1:
        axes = [axes]
    fig.suptitle(
        f"{session_name}\n"
        f"stim={stim_channel}, events={avg.shape[0]}, "
        f"{meta.get('folder_polarity') or meta.get('settings_polarity')}, "
        f"{meta.get('folder_current_uA') or meta.get('settings_first_amp_uA')} uA",
        fontsize=10,
    )
    for ch_index, (ax, channel) in enumerate(zip(axes, channels)):
        ax.plot(t * 1000, mean[:, ch_index], color="#2457a6", lw=1.1)
        ax.fill_between(
            t * 1000,
            mean[:, ch_index] - sem[:, ch_index],
            mean[:, ch_index] + sem[:, ch_index],
            color="#2457a6",
            alpha=0.18,
            lw=0,
        )
        ax.axvspan(0, ARTIFACT_END_S * 1000, color="#d9291c", alpha=0.12, lw=0)
        ax.axvline(0, color="#d9291c", lw=0.8)
        ax.axhline(0, color="0.65", lw=0.5)
        label = channel + (" stim" if channel == stim_channel else "")
        early = spike_counts.get(channel, np.array([], dtype=float))
        if early.size:
            label += f" | early spikes/trial {np.mean(early):.2f}"
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlim(-100, 300)
    axes[-1].set_xlabel("Time from stim onset (ms)")
    fig.tight_layout(rect=(0.08, 0.04, 1.0, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, object]], out_path: Path, value_key: str, title: str) -> None:
    if not rows:
        return
    sessions = []
    for row in rows:
        session = str(row["session"])
        if session not in sessions:
            sessions.append(session)
    channels = sorted({str(row["record_channel"]) for row in rows})
    matrix = np.full((len(sessions), len(channels)), np.nan)
    labels = []
    for i, session in enumerate(sessions):
        stim = ""
        current = ""
        polarity = ""
        for row in rows:
            if row["session"] == session:
                stim = str(row.get("stim_channel") or "")
                current = str(row.get("folder_current_uA") or row.get("settings_first_amp_uA") or "")
                polarity = str(row.get("folder_polarity") or row.get("settings_polarity") or "")
                break
        labels.append(f"{session[:15]} | {stim} | {current}uA | {polarity[:8]}")
    for row in rows:
        i = sessions.index(str(row["session"]))
        j = channels.index(str(row["record_channel"]))
        value = row.get(value_key)
        if value not in (None, ""):
            matrix[i, j] = float(value)
    vmax = np.nanpercentile(np.abs(matrix), 95) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(sessions) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(sessions)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(value_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_session(folder: Path, out_dir: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    data = read_rhs_folder_all(folder)
    stim_channel, stim_index = identify_stim_channel(data.stim_uA, data.channels)
    meta = parse_folder_metadata(folder.name)
    meta.update(parse_active_stim_from_settings(folder))
    duration_s = data.raw_uV.shape[0] / data.sample_rate_hz
    if stim_index is None or stim_channel is None:
        return [], {
            "session": folder.name,
            "duration_s": duration_s,
            "status": "no_stim_detected",
            "timestamp_gaps": data.timestamp_gaps,
        }

    stim_trace = data.stim_uA[:, stim_index]
    triggers = find_stim_trigger_onsets(stim_trace, data.sample_rate_hz, merge_gap_ms=10.0)
    if triggers.size == 0:
        return [], {
            "session": folder.name,
            "duration_s": duration_s,
            "status": "no_trigger_onsets",
            "timestamp_gaps": data.timestamp_gaps,
        }

    blank_windows = make_blank_windows(triggers, data.sample_rate_hz, stim_trace)
    clean = interpolate_blank_windows(data.raw_uV, blank_windows)
    lfp = bandpass(clean, data.sample_rate_hz, 0.1, 150.0)
    spike_band = bandpass(clean, data.sample_rate_hz, 250.0, 6000.0)
    spike_times = detect_negative_spikes(spike_band, data.sample_rate_hz, blank_windows)

    lfp_epochs, valid_triggers, t = epoch_data(lfp, triggers, data.sample_rate_hz, PRE_S, POST_S)
    raw_epochs, _, _ = epoch_data(data.raw_uV, triggers, data.sample_rate_hz, PRE_S, POST_S)
    if valid_triggers.size == 0:
        return [], {
            "session": folder.name,
            "duration_s": duration_s,
            "status": "triggers_too_close_to_file_edges",
            "detected_events": int(triggers.size),
            "timestamp_gaps": data.timestamp_gaps,
        }

    base_mask = window_mask(t, BASELINE_WINDOW_S)
    art_mask = window_mask(t, (0.0, ARTIFACT_END_S))
    early_mask = window_mask(t, EARLY_WINDOW_S)
    late_mask = window_mask(t, LATE_WINDOW_S)
    recovery_mask = window_mask(t, RECOVERY_WINDOW_S)

    baseline_mean = np.mean(lfp_epochs[:, base_mask, :], axis=1, keepdims=True)
    lfp_bc = lfp_epochs - baseline_mean
    baseline_sd = np.std(lfp_bc[:, base_mask, :].reshape(-1, lfp_bc.shape[2]), axis=0, ddof=1)
    baseline_sd[baseline_sd <= 1e-12] = 1.0

    spike_counts_for_plot: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for ch_index, channel in enumerate(data.channels):
        pre_counts = count_spikes_around_triggers(spike_times[ch_index], valid_triggers, data.sample_rate_hz, BASELINE_WINDOW_S)
        early_counts = count_spikes_around_triggers(spike_times[ch_index], valid_triggers, data.sample_rate_hz, EARLY_WINDOW_S)
        late_counts = count_spikes_around_triggers(spike_times[ch_index], valid_triggers, data.sample_rate_hz, LATE_WINDOW_S)
        recovery_counts = count_spikes_around_triggers(spike_times[ch_index], valid_triggers, data.sample_rate_hz, RECOVERY_WINDOW_S)
        spike_counts_for_plot[channel] = early_counts

        pre_rate = float(np.sum(pre_counts) / (valid_triggers.size * (BASELINE_WINDOW_S[1] - BASELINE_WINDOW_S[0])))
        early_rate = float(np.sum(early_counts) / (valid_triggers.size * (EARLY_WINDOW_S[1] - EARLY_WINDOW_S[0])))
        late_rate = float(np.sum(late_counts) / (valid_triggers.size * (LATE_WINDOW_S[1] - LATE_WINDOW_S[0])))
        recovery_rate = float(np.sum(recovery_counts) / (valid_triggers.size * (RECOVERY_WINDOW_S[1] - RECOVERY_WINDOW_S[0])))
        expected_early = pre_rate * valid_triggers.size * (EARLY_WINDOW_S[1] - EARLY_WINDOW_S[0])
        expected_late = pre_rate * valid_triggers.size * (LATE_WINDOW_S[1] - LATE_WINDOW_S[0])
        early_z = (float(np.sum(early_counts)) - expected_early) / math.sqrt(max(expected_early, 1.0))
        late_z = (float(np.sum(late_counts)) - expected_late) / math.sqrt(max(expected_late, 1.0))

        early_mean = np.mean(lfp_bc[:, early_mask, ch_index], axis=0)
        late_mean = np.mean(lfp_bc[:, late_mask, ch_index], axis=0)
        early_peak_signed = float(early_mean[np.argmax(np.abs(early_mean))])
        late_peak_signed = float(late_mean[np.argmax(np.abs(late_mean))])
        early_peak_z = early_peak_signed / float(baseline_sd[ch_index])
        late_peak_z = late_peak_signed / float(baseline_sd[ch_index])
        artifact_peak_uV = float(np.max(np.abs(raw_epochs[:, art_mask, ch_index])))
        artifact_tail_uV = float(np.max(np.abs(raw_epochs[:, window_mask(t, (0.005, 0.02)), ch_index]))) if np.any(window_mask(t, (0.005, 0.02))) else float("nan")

        row: dict[str, object] = {
            "session": folder.name,
            "record_channel": channel,
            "stim_channel": stim_channel,
            "duration_s": duration_s,
            "detected_events": int(triggers.size),
            "valid_events": int(valid_triggers.size),
            "timestamp_gaps": int(data.timestamp_gaps),
            "artifact_peak_0_5ms_uV": artifact_peak_uV,
            "artifact_tail_5_20ms_uV": artifact_tail_uV,
            "lfp_peak_5_50ms_uV": early_peak_signed,
            "lfp_peak_5_50ms_z": early_peak_z,
            "lfp_peak_50_300ms_uV": late_peak_signed,
            "lfp_peak_50_300ms_z": late_peak_z,
            "spike_rate_pre_hz": pre_rate,
            "spike_rate_5_50ms_hz": early_rate,
            "spike_rate_50_300ms_hz": late_rate,
            "spike_rate_300_800ms_hz": recovery_rate,
            "spike_delta_5_50ms_hz": early_rate - pre_rate,
            "spike_delta_50_300ms_hz": late_rate - pre_rate,
            "spike_z_5_50ms": early_z,
            "spike_z_50_300ms": late_z,
        }
        row.update(meta)
        rows.append(row)

    avg_dir = out_dir / "session_average_plots"
    avg_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", folder.name).strip("_")
    plot_session_average(
        avg_dir / f"{safe_name}_lfp_average_0p1-150Hz_blank0-5ms.png",
        folder.name,
        data.channels,
        stim_channel,
        t,
        lfp_epochs,
        spike_counts_for_plot,
        meta,
    )

    summary = {
        "session": folder.name,
        "status": "ok",
        "duration_s": duration_s,
        "sample_rate_hz": data.sample_rate_hz,
        "detected_events": int(triggers.size),
        "valid_events": int(valid_triggers.size),
        "stim_channel": stim_channel,
        "timestamp_gaps": int(data.timestamp_gaps),
        "first_trigger_s": float(triggers[0] / data.sample_rate_hz),
        "last_trigger_s": float(triggers[-1] / data.sample_rate_hz),
    }
    summary.update(meta)
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260624 pt stimulation",
        help="Folder containing RHS session subfolders.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max sessions for quick tests.")
    parser.add_argument(
        "--artifact-ms",
        type=float,
        default=5.0,
        help="Blank/interpolate this many milliseconds after each stimulus/train before filtering.",
    )
    args = parser.parse_args()

    global ARTIFACT_END_S
    ARTIFACT_END_S = args.artifact_ms / 1000.0

    root = Path(args.root).expanduser()
    suffix = "" if abs(args.artifact_ms - 5.0) < 1e-9 else f"_blank{args.artifact_ms:g}ms"
    out_dir = root / f"codex_analysis_20260624{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in root.iterdir() if p.is_dir() and list(p.glob("*.rhs"))])
    if args.limit:
        folders = folders[: args.limit]

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for i, folder in enumerate(folders, start=1):
        print(f"[{i}/{len(folders)}] {folder.name}", flush=True)
        try:
            rows, summary = analyze_session(folder, out_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
            summaries.append({"session": folder.name, "status": f"error: {exc}"})
            continue
        all_rows.extend(rows)
        if summary:
            summaries.append(summary)

    write_csv(out_dir / "per_channel_stim_response_metrics.csv", all_rows)
    write_csv(out_dir / "session_summary.csv", summaries)
    plot_heatmap(
        all_rows,
        out_dir / "heatmap_lfp_peak_5_50ms_z.png",
        "lfp_peak_5_50ms_z",
        "LFP response 5-50 ms after artifact blanking (0.1-150 Hz, z vs baseline SD)",
    )
    plot_heatmap(
        all_rows,
        out_dir / "heatmap_spike_z_5_50ms.png",
        "spike_z_5_50ms",
        "Spike-band threshold event change 5-50 ms vs pre-stim baseline",
    )
    plot_heatmap(
        all_rows,
        out_dir / "heatmap_spike_z_50_300ms.png",
        "spike_z_50_300ms",
        "Spike-band threshold event change 50-300 ms vs pre-stim baseline",
    )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
