#!/usr/bin/env python3
"""Rename RHS session folders using the stimulation waveform stored in stim_data."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from plot_rhs_raw_wideband_with_stim_legend import (
    SAMPLES_PER_DATA_BLOCK,
    _decode_stim_data,
    _read_exact,
    _read_header,
)


@dataclass(frozen=True)
class StimWaveformSummary:
    """A concise description of the first stimulation train in one RHS folder."""

    channel_name: str
    polarity: str
    first_phase_duration_us: float
    first_phase_amplitude_uA: float
    pulse_train_frequency_hz: float | None
    pulses_per_train: int
    refractory_period_ms: float | None
    compliance_hit: bool
    suffix: str


@dataclass(frozen=True)
class FolderRenamePlan:
    """One proposed folder rename."""

    source: Path
    target: Path | None
    summary: StimWaveformSummary | None
    status: str
    error: str | None = None

    @property
    def will_rename(self) -> bool:
        return self.status == "rename" and self.target is not None


def find_rhs_session_folders(parent_folder: Path) -> list[Path]:
    """Return RHS-containing folders inside parent_folder.

    If parent_folder itself contains .rhs files, it is treated as one session.
    Otherwise, each immediate child folder containing .rhs files is treated as a
    session. This keeps renaming scoped to the selected folder level.
    """
    parent_folder = parent_folder.expanduser().resolve()
    if not parent_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {parent_folder}")
    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {parent_folder}")

    if list(parent_folder.glob("*.rhs")):
        return [parent_folder]

    return [
        child
        for child in sorted(parent_folder.iterdir())
        if child.is_dir() and list(child.glob("*.rhs"))
    ]


def read_rhs_stim_channel(path: Path, channel_name: str) -> tuple[np.ndarray, float, bool]:
    """Read only stim_data for one amplifier channel from one .rhs file."""
    path = path.expanduser()
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

        stim_uA = np.empty(header.sample_count, dtype=np.float32)
        compliance_hit = False
        n_amp_channels = len(header.amplifier_channels)
        amp_matrix_bytes = SAMPLES_PER_DATA_BLOCK * n_amp_channels * 2
        amp_offset = SAMPLES_PER_DATA_BLOCK * 4
        dc_offset = amp_offset + amp_matrix_bytes
        stim_offset = dc_offset + (amp_matrix_bytes if header.dc_amp_data_saved else 0)

        for block_number in range(header.num_data_blocks):
            block = _read_exact(fid, header.bytes_per_block)
            sample_start = block_number * SAMPLES_PER_DATA_BLOCK
            sample_end = sample_start + SAMPLES_PER_DATA_BLOCK
            channel_slice = slice(
                channel_index * SAMPLES_PER_DATA_BLOCK,
                (channel_index + 1) * SAMPLES_PER_DATA_BLOCK,
            )
            stim_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=SAMPLES_PER_DATA_BLOCK * n_amp_channels,
                offset=stim_offset,
            )
            stim_raw = stim_flat[channel_slice]
            compliance_hit = compliance_hit or bool(
                np.any((stim_raw & np.uint16(0x8000)) != 0)
            )
            stim_uA[sample_start:sample_end] = _decode_stim_data(
                stim_raw, header.stim_step_size_uA
            )

    return stim_uA, header.sample_rate_hz, compliance_hit


def read_rhs_folder_stim_channel(
    folder: Path, channel_name: str
) -> tuple[np.ndarray, float, bool]:
    """Concatenate stim_data for one channel across all .rhs files in a folder."""
    rhs_files = sorted(folder.expanduser().glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError(f"No .rhs files found in {folder}")

    stim_parts: list[np.ndarray] = []
    sample_rates: set[float] = set()
    compliance_hit = False
    for rhs_file in rhs_files:
        stim_uA, sample_rate_hz, file_compliance_hit = read_rhs_stim_channel(
            rhs_file, channel_name
        )
        stim_parts.append(stim_uA)
        sample_rates.add(round(float(sample_rate_hz), 9))
        compliance_hit = compliance_hit or file_compliance_hit

    if len(sample_rates) != 1:
        raise ValueError(
            f"Cannot concatenate files with different sample rates: {sample_rates}"
        )

    return np.concatenate(stim_parts), float(next(iter(sample_rates))), compliance_hit


def summarize_rhs_folder(folder: Path) -> StimWaveformSummary:
    """Find the stim channel in an RHS folder and summarize its waveform."""
    folder = folder.expanduser().resolve()
    rhs_files = sorted(folder.glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError(f"No .rhs files found in {folder}")

    with rhs_files[0].open("rb") as fid:
        header = _read_header(fid, rhs_files[0], rhs_files[0].stat().st_size)

    for channel_name in header.amplifier_channels:
        stim_uA, sample_rate_hz, compliance_hit = read_rhs_folder_stim_channel(
            folder, channel_name
        )
        if np.any(stim_uA != 0):
            return summarize_stim_waveform(
                stim_uA, sample_rate_hz, channel_name, compliance_hit=compliance_hit
            )

    raise ValueError(f"No nonzero stim_data found in {folder.name}")


def summarize_stim_waveform(
    stim_uA: np.ndarray,
    sample_rate_hz: float,
    channel_name: str,
    compliance_hit: bool = False,
) -> StimWaveformSummary:
    """Summarize polarity, first phase, train frequency, pulse count, and RP."""
    nonzero = np.flatnonzero(stim_uA != 0)
    if nonzero.size == 0:
        raise ValueError("No nonzero stim_data found")

    first_index = int(nonzero[0])
    first_sign = float(np.sign(stim_uA[first_index]))
    phase_stop = first_index + 1
    while (
        phase_stop < stim_uA.size
        and stim_uA[phase_stop] != 0
        and float(np.sign(stim_uA[phase_stop])) == first_sign
    ):
        phase_stop += 1

    first_phase = stim_uA[first_index:phase_stop]
    first_phase_duration_us = (phase_stop - first_index) / sample_rate_hz * 1.0e6
    first_phase_amplitude_uA = float(np.max(np.abs(first_phase)))
    polarity = "cathodic first" if first_sign < 0 else "anodic first"

    pulse_starts, pulse_ends = _find_pulse_segments(stim_uA, sample_rate_hz)
    first_train_count = _first_train_count(pulse_starts, sample_rate_hz)
    train_starts = pulse_starts[:first_train_count]
    train_ends = pulse_ends[:first_train_count]
    pulses_per_train = max(1, int(first_train_count))
    pulse_train_frequency_hz = _train_frequency_hz(train_starts, sample_rate_hz)
    refractory_period_ms = _refractory_period_ms(
        train_starts, train_ends, sample_rate_hz
    )

    suffix = format_waveform_suffix(
        polarity=polarity,
        first_phase_duration_us=first_phase_duration_us,
        first_phase_amplitude_uA=first_phase_amplitude_uA,
        pulse_train_frequency_hz=pulse_train_frequency_hz,
        pulses_per_train=pulses_per_train,
        refractory_period_ms=refractory_period_ms,
        compliance_hit=compliance_hit,
    )
    return StimWaveformSummary(
        channel_name=channel_name,
        polarity=polarity,
        first_phase_duration_us=first_phase_duration_us,
        first_phase_amplitude_uA=first_phase_amplitude_uA,
        pulse_train_frequency_hz=pulse_train_frequency_hz,
        pulses_per_train=pulses_per_train,
        refractory_period_ms=refractory_period_ms,
        compliance_hit=compliance_hit,
        suffix=suffix,
    )


def plan_renames(parent_folder: Path) -> list[FolderRenamePlan]:
    """Build a dry-run rename plan for every RHS session folder."""
    session_folders = find_rhs_session_folders(parent_folder)
    if not session_folders:
        return [
            FolderRenamePlan(
                source=parent_folder.expanduser().resolve(),
                target=None,
                summary=None,
                status="skip",
                error="No immediate RHS folders found.",
            )
        ]

    plans: list[FolderRenamePlan] = []
    for folder in session_folders:
        try:
            summary = summarize_rhs_folder(folder)
            target = _target_folder_for_suffix(folder, summary.suffix)
            status = "already named" if target == folder else "rename"
            plans.append(
                FolderRenamePlan(
                    source=folder,
                    target=target,
                    summary=summary,
                    status=status,
                )
            )
        except Exception as exc:
            plans.append(
                FolderRenamePlan(
                    source=folder,
                    target=None,
                    summary=None,
                    status="error",
                    error=str(exc),
                )
            )
    return plans


def apply_rename_plan(plans: Iterable[FolderRenamePlan]) -> list[FolderRenamePlan]:
    """Apply a previewed rename plan and return updated statuses."""
    applied: list[FolderRenamePlan] = []
    for plan in plans:
        if not plan.will_rename:
            applied.append(plan)
            continue

        assert plan.target is not None
        try:
            final_target = _available_target(plan.target)
            plan.source.rename(final_target)
            applied.append(
                FolderRenamePlan(
                    source=final_target,
                    target=final_target,
                    summary=plan.summary,
                    status="renamed",
                )
            )
        except Exception as exc:
            applied.append(
                FolderRenamePlan(
                    source=plan.source,
                    target=plan.target,
                    summary=plan.summary,
                    status="error",
                    error=str(exc),
                )
            )
    return applied


def format_waveform_suffix(
    polarity: str,
    first_phase_duration_us: float,
    first_phase_amplitude_uA: float,
    pulse_train_frequency_hz: float | None,
    pulses_per_train: int,
    refractory_period_ms: float | None = None,
    compliance_hit: bool = False,
) -> str:
    """Format text like 'cathodic first 100us 200uA 1Hz 30 pulses 999ms RP comp'."""
    pieces = [
        polarity,
        _format_value(first_phase_duration_us, "us"),
        _format_value(first_phase_amplitude_uA, "uA"),
    ]
    if pulse_train_frequency_hz is not None and math.isfinite(pulse_train_frequency_hz):
        pieces.append(_format_value(pulse_train_frequency_hz, "Hz"))
    pieces.append(f"{pulses_per_train:g} {_plural('pulse', pulses_per_train)}")
    if refractory_period_ms is not None and math.isfinite(refractory_period_ms):
        pieces.append(f"{_format_value_floor(refractory_period_ms, 'ms')} RP")
    if compliance_hit:
        pieces.append("comp")
    return " ".join(pieces)


def _find_pulse_segments(
    stim_uA: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    active_starts, active_ends = _active_segments(stim_uA)
    if active_starts.size == 0:
        return active_starts, active_ends

    max_interphase_gap = max(1, int(round(sample_rate_hz * 0.00025)))
    pulse_starts = [int(active_starts[0])]
    pulse_ends = []
    previous_end = int(active_ends[0])
    for start, end in zip(active_starts[1:], active_ends[1:]):
        gap = int(start) - previous_end
        if gap > max_interphase_gap:
            pulse_ends.append(previous_end)
            pulse_starts.append(int(start))
        previous_end = int(end)
    pulse_ends.append(previous_end)
    return (
        np.asarray(pulse_starts, dtype=np.int64),
        np.asarray(pulse_ends, dtype=np.int64),
    )


def _active_segments(stim_uA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonzero = np.flatnonzero(stim_uA != 0)
    if nonzero.size == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    split_after = np.flatnonzero(np.diff(nonzero) > 1)
    starts = np.r_[nonzero[0], nonzero[split_after + 1]].astype(np.int64)
    ends = np.r_[nonzero[split_after] + 1, nonzero[-1] + 1].astype(np.int64)
    return starts, ends


def _first_train_count(pulse_starts: np.ndarray, sample_rate_hz: float) -> int:
    if pulse_starts.size <= 1:
        return int(pulse_starts.size)

    diffs_s = np.diff(pulse_starts) / sample_rate_hz
    positive_diffs = diffs_s[diffs_s > 0]
    if positive_diffs.size == 0:
        return 1

    median_diff_s = float(np.median(positive_diffs))
    train_gap_threshold_s = max(0.02, median_diff_s * 3.0)
    first_train_count = 1
    for diff_s in diffs_s:
        if diff_s <= train_gap_threshold_s:
            first_train_count += 1
        else:
            break
    return first_train_count


def _train_frequency_hz(
    train_starts: np.ndarray, sample_rate_hz: float
) -> float | None:
    if train_starts.size <= 1:
        return None

    diffs_s = np.diff(train_starts) / sample_rate_hz
    positive_diffs = diffs_s[diffs_s > 0]
    if positive_diffs.size == 0:
        return None
    return 1.0 / float(np.median(positive_diffs))


def _refractory_period_ms(
    train_starts: np.ndarray, train_ends: np.ndarray, sample_rate_hz: float
) -> float | None:
    if train_starts.size <= 1 or train_ends.size <= 1:
        return None

    silent_samples = train_starts[1:] - train_ends[:-1]
    silent_samples = silent_samples[silent_samples > 0]
    if silent_samples.size == 0:
        return None
    return float(np.median(silent_samples) / sample_rate_hz * 1.0e3)


def _target_folder_for_suffix(folder: Path, suffix: str) -> Path:
    cleaned_suffix = _clean_folder_text(suffix)
    if cleaned_suffix in folder.name:
        return folder
    base_name = _strip_generated_waveform_suffix(folder.name)
    return _available_target(folder.with_name(f"{base_name} {cleaned_suffix}"))


def _available_target(target: Path) -> Path:
    if not target.exists():
        return target

    for index in range(2, 1000):
        candidate = target.with_name(f"{target.name}__{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available folder name for {target}")


def _clean_folder_text(text: str) -> str:
    return " ".join(text.replace("/", "-").replace(":", "-").split())


def _strip_generated_waveform_suffix(folder_name: str) -> str:
    generated_suffix = re.compile(
        r"\s+(?:cathodic|anodic) first "
        r"\d+(?:\.\d+)?us "
        r"\d+(?:\.\d+)?uA"
        r"(?: \d+(?:\.\d+)?Hz)? "
        r"\d+ pulses?"
        r"(?: \d+(?:\.\d+)?ms RP)?"
        r"(?: comp)?$"
    )
    return generated_suffix.sub("", folder_name).strip() or folder_name


def _format_value(value: float, unit: str) -> str:
    value = abs(float(value))
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=0.05):
        return f"{int(round(value))}{unit}"
    if value >= 100:
        return f"{value:.0f}{unit}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".") + unit
    return f"{value:.3g}{unit}"


def _format_value_floor(value: float, unit: str) -> str:
    value = abs(float(value))
    if value >= 1:
        return f"{int(math.floor(value))}{unit}"
    return f"{value:.3g}{unit}"


def _plural(word: str, count: int) -> str:
    return word if int(count) == 1 else f"{word}s"


def _print_plan(plans: Iterable[FolderRenamePlan]) -> None:
    for plan in plans:
        if plan.status == "error":
            print(f"ERROR: {plan.source.name}: {plan.error}")
        elif plan.status == "skip":
            print(f"SKIP: {plan.source}: {plan.error}")
        elif plan.status == "already named":
            assert plan.summary is not None
            print(
                f"OK: {plan.source.name} already includes "
                f"{plan.summary.suffix!r} (stim channel {plan.summary.channel_name})"
            )
        else:
            assert plan.target is not None and plan.summary is not None
            print(
                f"{plan.status.upper()}: {plan.source.name} -> {plan.target.name} "
                f"(stim channel {plan.summary.channel_name})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or apply RHS folder renames using waveform details decoded "
            "from the folder's nonzero stim_data."
        )
    )
    parser.add_argument(
        "parent_folder",
        type=Path,
        help="Folder containing RHS session folders, or one RHS session folder.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename folders. Without this flag, only preview the plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = plan_renames(args.parent_folder)
    _print_plan(plans)
    if args.apply:
        print("\nApplying renames...")
        _print_plan(apply_rename_plan(plans))
    else:
        print("\nDry run only. Re-run with --apply to rename folders.")


if __name__ == "__main__":
    main()
