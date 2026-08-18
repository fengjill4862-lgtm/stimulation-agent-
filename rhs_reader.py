"""Shared multi-channel Intan RHS readers.

Two readers live here:

* ``read_rhs_file_selected_channels`` / ``read_rhs_folder_selected_channels``
  moved verbatim from ``batch_run_wideband_main_ui.py`` so the batch runner and
  the analysis package share one implementation. Their numerical output is
  unchanged (the batch runner's PNG/CSV bytes are the regression check).
* ``read_rhs_file`` / ``read_rhs_run`` -- a faster single-pass reader built on
  a structured dtype that mirrors one RHS data block, returning every selected
  channel as a 2-D array plus the raw stim words (flags included) and the
  timestamps. This is what ``stim_analysis`` uses.

Both rely on the header parser and stim decoder in
``plot_rhs_raw_wideband_with_stim_legend.py`` so there is exactly one place
that knows the RHS header layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_rhs_raw_wideband_with_stim_legend import (
    SAMPLES_PER_DATA_BLOCK,
    RhsHeader,
    _decode_stim_data,
    _read_exact,
    _read_header,
)


# -----------------------------------------------------------------------------
# Batch-runner reader (moved verbatim; do not change numerics)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RhsFolderData:
    """Raw and stim data for all selected channels in one RHS session folder."""

    channels: list[str]
    raw_uV: list[np.ndarray]
    stim_uA: list[np.ndarray]
    sample_rate_hz: float
    rhs_file_count: int
    timestamp_gaps: int


def read_rhs_file_selected_channels(
    path: Path, wanted_channels: list[str]
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], float, int]:
    """Read selected amplifier and stim traces from one RHS file in one pass."""
    path = path.expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        header = _read_header(fid, path, file_size)
        missing = [channel for channel in wanted_channels if channel not in header.amplifier_channels]
        if missing:
            available = ", ".join(header.amplifier_channels)
            raise ValueError(
                f"{path.name}: channels not found: {', '.join(missing)}. "
                f"Available amplifier channels: {available}"
            )

        channel_indices = [header.amplifier_channels.index(channel) for channel in wanted_channels]
        channels = [header.amplifier_channels[index] for index in channel_indices]
        raw_uV = [
            np.empty(header.sample_count, dtype=np.float32)
            for _channel in channels
        ]
        stim_uA = [
            np.empty(header.sample_count, dtype=np.float32)
            for _channel in channels
        ]

        n_amp_channels = len(header.amplifier_channels)
        amp_matrix_words = SAMPLES_PER_DATA_BLOCK * n_amp_channels
        amp_matrix_bytes = amp_matrix_words * 2
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
                count=amp_matrix_words,
                offset=amp_offset,
            ).reshape(n_amp_channels, SAMPLES_PER_DATA_BLOCK)
            stim_flat = np.frombuffer(
                block,
                dtype="<u2",
                count=amp_matrix_words,
                offset=stim_offset,
            ).reshape(n_amp_channels, SAMPLES_PER_DATA_BLOCK)

            for output_index, channel_index in enumerate(channel_indices):
                raw_block = amp_flat[channel_index].astype(np.float32)
                raw_uV[output_index][sample_start:sample_end] = (
                    raw_block - np.float32(32768.0)
                ) * np.float32(0.195)
                stim_uA[output_index][sample_start:sample_end] = _decode_stim_data(
                    stim_flat[channel_index],
                    header.stim_step_size_uA,
                )

    return channels, raw_uV, stim_uA, header.sample_rate_hz, timestamp_gaps


def read_rhs_folder_selected_channels(folder: Path, channels: list[str]) -> RhsFolderData:
    """Read and concatenate all RHS files for selected channels."""
    rhs_files = sorted(folder.expanduser().glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError(f"No .rhs files found in {folder}")

    raw_parts: list[list[np.ndarray]] = [[] for _channel in channels]
    stim_parts: list[list[np.ndarray]] = [[] for _channel in channels]
    sample_rates: set[float] = set()
    timestamp_gaps = 0
    loaded_channels: list[str] | None = None

    for rhs_file in rhs_files:
        file_channels, raw_uV, stim_uA, sample_rate_hz, file_timestamp_gaps = (
            read_rhs_file_selected_channels(rhs_file, channels)
        )
        if loaded_channels is None:
            loaded_channels = file_channels
        elif loaded_channels != file_channels:
            raise ValueError(f"{folder.name}: channel list changed across RHS files.")
        sample_rates.add(round(float(sample_rate_hz), 9))
        timestamp_gaps += file_timestamp_gaps
        for index in range(len(channels)):
            raw_parts[index].append(raw_uV[index])
            stim_parts[index].append(stim_uA[index])

    if len(sample_rates) != 1:
        raise ValueError(f"{folder.name}: mixed sample rates {sample_rates}")

    return RhsFolderData(
        channels=loaded_channels or channels,
        raw_uV=[np.concatenate(parts) for parts in raw_parts],
        stim_uA=[np.concatenate(parts) for parts in stim_parts],
        sample_rate_hz=float(next(iter(sample_rates))),
        rhs_file_count=len(rhs_files),
        timestamp_gaps=timestamp_gaps,
    )


# -----------------------------------------------------------------------------
# Fast single-pass reader (structured dtype)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RhsRunData:
    """Every selected channel of one RHS run (all files concatenated).

    ``raw_uV`` is ``(n_channels, n_samples)`` float32 in microvolts, computed
    exactly like the single-channel reader (``0.195 * (code - 32768)``).
    ``stim_words`` keeps the raw 16-bit stim word per channel so both the
    commanded current and the RHS flag bits (amp settle, charge recovery,
    compliance limit) remain available. ``timestamps`` are the RHS sample
    counters; ``file_boundaries`` gives the sample offset where each file
    starts, and ``timestamp_gaps`` counts every discontinuity, including the
    ones at file boundaries.
    """

    folder: Path
    rhs_files: list[Path]
    header: RhsHeader
    channels: list[str]
    sample_rate_hz: float
    raw_uV: np.ndarray
    stim_words: np.ndarray
    timestamps: np.ndarray
    file_boundaries: list[int]
    timestamp_gaps: int

    @property
    def n_samples(self) -> int:
        return int(self.raw_uV.shape[1]) if self.raw_uV.ndim == 2 else 0

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz if self.sample_rate_hz else 0.0

    def channel_index(self, channel: str) -> int:
        try:
            return self.channels.index(channel)
        except ValueError as exc:
            raise ValueError(
                f"{self.folder.name}: channel {channel!r} was not loaded. "
                f"Loaded channels: {', '.join(self.channels)}"
            ) from exc

    def stim_uA(self, channel: str) -> np.ndarray:
        """Commanded stim current for one channel, in microamps."""
        return stim_uA_from_words(
            self.stim_words[self.channel_index(channel)], self.header.stim_step_size_uA
        )


def stim_uA_from_words(words: np.ndarray, stim_step_size_uA: float) -> np.ndarray:
    """Decode raw stim words into signed microamps (flags ignored)."""
    return _decode_stim_data(np.asarray(words), stim_step_size_uA)


def block_dtype(header: RhsHeader) -> np.dtype:
    """Structured dtype describing one RHS data block for this header."""
    n_amp = len(header.amplifier_channels)
    fields: list[tuple[str, str, tuple[int, ...]]] = [
        ("t", "<i4", (SAMPLES_PER_DATA_BLOCK,)),
        ("amp", "<u2", (n_amp, SAMPLES_PER_DATA_BLOCK)),
    ]
    if header.dc_amp_data_saved:
        fields.append(("dc", "<u2", (n_amp, SAMPLES_PER_DATA_BLOCK)))
    fields.append(("stim", "<u2", (n_amp, SAMPLES_PER_DATA_BLOCK)))
    if header.board_adc_channels:
        fields.append(("adc", "<u2", (len(header.board_adc_channels), SAMPLES_PER_DATA_BLOCK)))
    if header.board_dac_channels:
        fields.append(("dac", "<u2", (len(header.board_dac_channels), SAMPLES_PER_DATA_BLOCK)))
    if header.board_dig_in_channels:
        fields.append(("din", "<u2", (SAMPLES_PER_DATA_BLOCK,)))
    if header.board_dig_out_channels:
        fields.append(("dout", "<u2", (SAMPLES_PER_DATA_BLOCK,)))
    dtype = np.dtype(fields)
    if dtype.itemsize != header.bytes_per_block:
        raise ValueError(
            f"{header.path.name}: block dtype is {dtype.itemsize} bytes but the header "
            f"implies {header.bytes_per_block} bytes per block"
        )
    return dtype


def read_rhs_header(path: Path) -> RhsHeader:
    """Parse only the header of one RHS file."""
    path = Path(path).expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        return _read_header(fid, path, file_size)


def read_rhs_file(path: Path, wanted_channels: list[str] | None = None) -> RhsRunData:
    """Read one RHS file in a single pass.

    ``wanted_channels=None`` loads every enabled amplifier channel, in header
    order. Channel order in the result follows ``wanted_channels`` when given.
    """
    path = Path(path).expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        header = _read_header(fid, path, file_size)
        if wanted_channels is None:
            channels = list(header.amplifier_channels)
        else:
            missing = [channel for channel in wanted_channels if channel not in header.amplifier_channels]
            if missing:
                available = ", ".join(header.amplifier_channels)
                raise ValueError(
                    f"{path.name}: channels not found: {', '.join(missing)}. "
                    f"Available amplifier channels: {available}"
                )
            channels = list(wanted_channels)
        indices = np.asarray(
            [header.amplifier_channels.index(channel) for channel in channels], dtype=np.intp
        )
        dtype = block_dtype(header)
        blocks = np.fromfile(fid, dtype=dtype, count=header.num_data_blocks)

    if blocks.shape[0] != header.num_data_blocks:
        raise EOFError(
            f"{path.name}: expected {header.num_data_blocks} data blocks, read {blocks.shape[0]}"
        )

    timestamps = np.ascontiguousarray(blocks["t"].reshape(-1))
    if timestamps.size:
        timestamp_gaps = int(np.count_nonzero(np.diff(timestamps) != 1))
    else:
        timestamp_gaps = 0

    n_samples = header.sample_count
    # (blocks, n_amp, 128) -> pick channels -> (n_sel, blocks, 128) -> (n_sel, N)
    amp_codes = np.ascontiguousarray(
        blocks["amp"][:, indices, :].transpose(1, 0, 2).reshape(len(channels), n_samples)
    )
    raw_uV = (amp_codes.astype(np.float32) - np.float32(32768.0)) * np.float32(0.195)
    stim_words = np.ascontiguousarray(
        blocks["stim"][:, indices, :].transpose(1, 0, 2).reshape(len(channels), n_samples)
    ).astype(np.uint16, copy=False)
    del blocks

    return RhsRunData(
        folder=path.parent,
        rhs_files=[path],
        header=header,
        channels=channels,
        sample_rate_hz=float(header.sample_rate_hz),
        raw_uV=raw_uV,
        stim_words=stim_words,
        timestamps=timestamps,
        file_boundaries=[0],
        timestamp_gaps=timestamp_gaps,
    )


def read_rhs_run(folder: Path, wanted_channels: list[str] | None = None) -> RhsRunData:
    """Read and concatenate every ``*.rhs`` file in one run folder."""
    folder = Path(folder).expanduser()
    rhs_files = sorted(folder.glob("*.rhs"))
    if not rhs_files:
        raise FileNotFoundError(f"No .rhs files found in {folder}")

    parts = [read_rhs_file(path, wanted_channels) for path in rhs_files]
    first = parts[0]
    for part in parts[1:]:
        if part.channels != first.channels:
            raise ValueError(f"{folder.name}: channel list changed across RHS files.")
        if round(part.sample_rate_hz, 9) != round(first.sample_rate_hz, 9):
            raise ValueError(
                f"{folder.name}: mixed sample rates "
                f"{first.sample_rate_hz} and {part.sample_rate_hz}"
            )

    if len(parts) == 1:
        return RhsRunData(
            folder=folder,
            rhs_files=rhs_files,
            header=first.header,
            channels=first.channels,
            sample_rate_hz=first.sample_rate_hz,
            raw_uV=first.raw_uV,
            stim_words=first.stim_words,
            timestamps=first.timestamps,
            file_boundaries=[0],
            timestamp_gaps=first.timestamp_gaps,
        )

    boundaries: list[int] = []
    offset = 0
    gaps = 0
    previous_last: int | None = None
    for part in parts:
        boundaries.append(offset)
        offset += part.n_samples
        gaps += part.timestamp_gaps
        if previous_last is not None and part.timestamps.size:
            if int(part.timestamps[0]) != previous_last + 1:
                gaps += 1
        if part.timestamps.size:
            previous_last = int(part.timestamps[-1])

    return RhsRunData(
        folder=folder,
        rhs_files=rhs_files,
        header=first.header,
        channels=first.channels,
        sample_rate_hz=first.sample_rate_hz,
        raw_uV=np.concatenate([part.raw_uV for part in parts], axis=1),
        stim_words=np.concatenate([part.stim_words for part in parts], axis=1),
        timestamps=np.concatenate([part.timestamps for part in parts]),
        file_boundaries=boundaries,
        timestamp_gaps=gaps,
    )


__all__ = [
    "RhsFolderData",
    "RhsRunData",
    "block_dtype",
    "read_rhs_file",
    "read_rhs_file_selected_channels",
    "read_rhs_folder_selected_channels",
    "read_rhs_header",
    "read_rhs_run",
    "stim_uA_from_words",
]
