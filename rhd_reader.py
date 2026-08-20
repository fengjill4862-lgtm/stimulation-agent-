#!/usr/bin/env python3
"""Intan RHD2000 reader, the recording-controller counterpart to ``rhs_reader``.

Why this exists
---------------
The 2026-08-19 session was recorded on a ``ControllerRecordUSB3`` (RHD), not on
the stim/record controller (RHS), because the stimulus came from an external
Keithley 2400 rather than from the Intan. Every existing reader in this repo
speaks RHS only, and the two formats differ in more than the magic number:

* RHD has no stim words and no DC amplifier channels; it adds auxiliary inputs
  (sampled at rate/4), supply-voltage and temperature channels (once per block),
  and it has no DAC group.
* The ``signal_type`` codes are shifted. In RHS 3/4/5/6 mean ADC/DAC/DIG-IN/
  DIG-OUT; in RHD 1/2/3/4/5 mean AUX/SUPPLY/ADC/DIG-IN/DIG-OUT. Reusing the RHS
  parser would silently mislabel every non-amplifier channel.
* Block size is 60 samples for file version 1.x and 128 from 2.0 on, and
  timestamps became signed at version 1.2.

The layout below is a direct port of ``read_Intan_RHD2000_file_no_prompt_new.m``
in this repo, which is the reference implementation for these files. The public
shape deliberately mirrors ``rhs_reader``: ``read_rhd_header`` / ``read_rhd_file``
/ ``read_rhd_run``, one structured dtype per data block, one ``np.fromfile`` per
file, ``raw_uV`` as a ``(n_channels, n_samples)`` float32 array in microvolts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


RHD_MAGIC_NUMBER = 0xC6912702

# Amplifier codes are unsigned 16-bit with mid-scale at 32768; 0.195 uV/LSB.
AMPLIFIER_OFFSET = 32768.0
AMPLIFIER_SCALE_uV = 0.195
AUX_SCALE_V = 37.4e-6
SUPPLY_SCALE_V = 74.8e-6
TEMP_SCALE_DEG_C = 0.01

# Board ADC scaling depends on which board wrote the file (eval_board_mode).
ADC_SCALE_V_MODE_1 = 152.59e-6      # unipolar-to-bipolar eval board
ADC_SCALE_V_MODE_13 = 312.5e-6      # Intan Recording Controller
ADC_SCALE_V_DEFAULT = 50.354e-6     # no offset subtraction in this mode


@dataclass(frozen=True)
class RhdHeader:
    """Everything the RHD header says, including fields nothing reads yet.

    Keeping the filter and impedance settings costs nothing and means a run's
    acquisition configuration can be checked against ``settings.xml`` without a
    second parser.
    """

    path: Path
    version: tuple[int, int]
    sample_rate_hz: float
    samples_per_block: int
    timestamps_signed: bool
    eval_board_mode: int
    reference_channel: str

    amplifier_channels: list[str]
    aux_input_channels: list[str]
    supply_voltage_channels: list[str]
    board_adc_channels: list[str]
    board_dig_in_channels: list[str]
    board_dig_out_channels: list[str]
    num_temp_sensor_channels: int

    header_bytes: int
    bytes_per_block: int
    num_data_blocks: int
    sample_count: int

    impedance_magnitude_ohms: dict[str, float] = field(default_factory=dict)
    impedance_phase_deg: dict[str, float] = field(default_factory=dict)
    board_dig_in_orders: dict[str, int] = field(default_factory=dict)
    board_dig_out_orders: dict[str, int] = field(default_factory=dict)

    dsp_enabled: bool = False
    actual_dsp_cutoff_hz: float = float("nan")
    actual_lower_bandwidth_hz: float = float("nan")
    actual_upper_bandwidth_hz: float = float("nan")
    desired_dsp_cutoff_hz: float = float("nan")
    desired_lower_bandwidth_hz: float = float("nan")
    desired_upper_bandwidth_hz: float = float("nan")
    notch_filter_mode: int = 0
    notch_filter_hz: float = 0.0
    desired_impedance_test_frequency_hz: float = float("nan")
    actual_impedance_test_frequency_hz: float = float("nan")
    notes: tuple[str, str, str] = ("", "", "")

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate_hz if self.sample_rate_hz else 0.0

    @property
    def aux_sample_rate_hz(self) -> float:
        return self.sample_rate_hz / 4.0

    @property
    def adc_scale_V(self) -> float:
        if self.eval_board_mode == 1:
            return ADC_SCALE_V_MODE_1
        if self.eval_board_mode == 13:
            return ADC_SCALE_V_MODE_13
        return ADC_SCALE_V_DEFAULT

    @property
    def adc_subtracts_offset(self) -> bool:
        return self.eval_board_mode in (1, 13)


@dataclass(frozen=True)
class RhdRunData:
    """Every selected channel of one RHD run (all files concatenated).

    ``raw_uV`` is ``(n_channels, n_samples)`` float32 microvolts. ``aux_V`` is
    ``(n_aux, n_samples // 4)`` because auxiliary inputs are sampled four times
    slower than the amplifiers -- it is deliberately *not* upsampled to the
    amplifier grid, so callers cannot silently mix the two time bases.
    ``file_boundaries`` gives the sample offset where each file starts, and
    ``timestamp_gaps`` counts every discontinuity, file boundaries included.
    """

    folder: Path
    rhd_files: list[Path]
    header: RhdHeader
    channels: list[str]
    sample_rate_hz: float
    raw_uV: np.ndarray
    timestamps: np.ndarray
    aux_V: np.ndarray
    supply_V: np.ndarray
    temp_deg_C: np.ndarray
    adc_V: np.ndarray
    dig_in_raw: np.ndarray
    dig_out_raw: np.ndarray
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

    def channel(self, channel: str) -> np.ndarray:
        """One channel's trace in microvolts."""
        return self.raw_uV[self.channel_index(channel)]

    def time_s(self) -> np.ndarray:
        """Sample times in seconds, taken from the recorded timestamps.

        Using the timestamps rather than ``arange`` keeps the time base honest
        across a dropped block; check ``timestamp_gaps`` before trusting it as
        uniformly sampled.
        """
        return self.timestamps.astype(np.float64) / self.sample_rate_hz

    def dig_in(self, channel: str) -> np.ndarray:
        """One digital-in line as a boolean array.

        The RHD file stores all digital inputs packed into one 16-bit word per
        sample; the bit position is the channel's ``native_order``.
        """
        bit = self.header.board_dig_in_orders.get(channel)
        if bit is None:
            raise ValueError(
                f"{self.folder.name}: digital input {channel!r} is not in this file. "
                f"Available: {', '.join(self.header.board_dig_in_channels) or 'none'}"
            )
        if self.dig_in_raw.size == 0:
            return np.zeros(0, dtype=bool)
        return (self.dig_in_raw & np.uint16(1 << bit)) != 0


# -----------------------------------------------------------------------------
# Header parsing
# -----------------------------------------------------------------------------


def _read_exact(fid, n_bytes: int) -> bytes:
    data = fid.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError(f"Unexpected EOF while reading {n_bytes} bytes")
    return data


def _read_scalar(fid, fmt: str):
    return struct.unpack("<" + fmt, _read_exact(fid, struct.calcsize("<" + fmt)))[0]


def _read_qstring(fid) -> str:
    n_bytes = _read_scalar(fid, "I")
    if n_bytes in (0, 0xFFFFFFFF):
        return ""
    return _read_exact(fid, n_bytes).decode("utf-16le", errors="replace")


def _read_header(fid, path: Path, file_size: int) -> RhdHeader:
    magic_number = _read_scalar(fid, "I")
    if magic_number != RHD_MAGIC_NUMBER:
        raise ValueError(
            f"{path.name} is not an Intan RHD2000 file "
            f"(magic 0x{magic_number:08x}, expected 0x{RHD_MAGIC_NUMBER:08x})"
        )

    major = _read_scalar(fid, "h")
    minor = _read_scalar(fid, "h")
    version = (major, minor)

    # Block size doubled at version 2.0; timestamps became signed at 1.2.
    samples_per_block = 60 if major == 1 else 128
    timestamps_signed = (major == 1 and minor >= 2) or major > 1

    sample_rate_hz = float(_read_scalar(fid, "f"))
    dsp_enabled = _read_scalar(fid, "h")
    actual_dsp_cutoff = _read_scalar(fid, "f")
    actual_lower_bandwidth = _read_scalar(fid, "f")
    actual_upper_bandwidth = _read_scalar(fid, "f")
    desired_dsp_cutoff = _read_scalar(fid, "f")
    desired_lower_bandwidth = _read_scalar(fid, "f")
    desired_upper_bandwidth = _read_scalar(fid, "f")

    notch_filter_mode = _read_scalar(fid, "h")
    notch_filter_hz = {1: 50.0, 2: 60.0}.get(notch_filter_mode, 0.0)

    desired_impedance_test_frequency = _read_scalar(fid, "f")
    actual_impedance_test_frequency = _read_scalar(fid, "f")

    notes = (_read_qstring(fid), _read_qstring(fid), _read_qstring(fid))

    # Temperature-sensor count appears from GUI v1.1, eval board mode from v1.3,
    # and the digital reference channel from v2.0 (Recording Controller).
    num_temp_sensor_channels = 0
    if (major == 1 and minor >= 1) or major > 1:
        num_temp_sensor_channels = _read_scalar(fid, "h")

    eval_board_mode = 0
    if (major == 1 and minor >= 3) or major > 1:
        eval_board_mode = _read_scalar(fid, "h")

    reference_channel = ""
    if major > 1:
        reference_channel = _read_qstring(fid)

    amplifier_channels: list[str] = []
    aux_input_channels: list[str] = []
    supply_voltage_channels: list[str] = []
    board_adc_channels: list[str] = []
    board_dig_in_channels: list[str] = []
    board_dig_out_channels: list[str] = []
    impedance_magnitude_ohms: dict[str, float] = {}
    impedance_phase_deg: dict[str, float] = {}
    board_dig_in_orders: dict[str, int] = {}
    board_dig_out_orders: dict[str, int] = {}

    number_of_signal_groups = _read_scalar(fid, "h")
    for _signal_group in range(number_of_signal_groups):
        _group_name = _read_qstring(fid)
        _group_prefix = _read_qstring(fid)
        group_enabled = _read_scalar(fid, "h")
        group_num_channels = _read_scalar(fid, "h")
        _group_num_amp_channels = _read_scalar(fid, "h")

        if group_num_channels <= 0 or group_enabled <= 0:
            continue

        for _signal_channel in range(group_num_channels):
            native_channel_name = _read_qstring(fid)
            _custom_channel_name = _read_qstring(fid)
            native_order = _read_scalar(fid, "h")
            _custom_order = _read_scalar(fid, "h")
            signal_type = _read_scalar(fid, "h")
            channel_enabled = _read_scalar(fid, "h")
            _chip_channel = _read_scalar(fid, "h")
            _board_stream = _read_scalar(fid, "h")
            _voltage_trigger_mode = _read_scalar(fid, "h")
            _voltage_threshold = _read_scalar(fid, "h")
            _digital_trigger_channel = _read_scalar(fid, "h")
            _digital_edge_polarity = _read_scalar(fid, "h")
            electrode_impedance_magnitude = _read_scalar(fid, "f")
            electrode_impedance_phase = _read_scalar(fid, "f")

            if not channel_enabled:
                continue
            # RHD signal types: 0 amp, 1 aux, 2 supply, 3 ADC, 4 dig-in, 5 dig-out.
            if signal_type == 0:
                amplifier_channels.append(native_channel_name)
                impedance_magnitude_ohms[native_channel_name] = float(
                    electrode_impedance_magnitude
                )
                impedance_phase_deg[native_channel_name] = float(electrode_impedance_phase)
            elif signal_type == 1:
                aux_input_channels.append(native_channel_name)
            elif signal_type == 2:
                supply_voltage_channels.append(native_channel_name)
            elif signal_type == 3:
                board_adc_channels.append(native_channel_name)
            elif signal_type == 4:
                board_dig_in_channels.append(native_channel_name)
                board_dig_in_orders[native_channel_name] = int(native_order)
            elif signal_type == 5:
                board_dig_out_channels.append(native_channel_name)
                board_dig_out_orders[native_channel_name] = int(native_order)
            else:
                raise ValueError(
                    f"{path.name}: unknown RHD signal type {signal_type} "
                    f"for channel {native_channel_name!r}"
                )

    bytes_per_block = samples_per_block * 4  # timestamps
    bytes_per_block += samples_per_block * 2 * len(amplifier_channels)
    bytes_per_block += (samples_per_block // 4) * 2 * len(aux_input_channels)
    bytes_per_block += 1 * 2 * len(supply_voltage_channels)
    bytes_per_block += 1 * 2 * num_temp_sensor_channels
    bytes_per_block += samples_per_block * 2 * len(board_adc_channels)
    if board_dig_in_channels:
        bytes_per_block += samples_per_block * 2
    if board_dig_out_channels:
        bytes_per_block += samples_per_block * 2

    header_bytes = fid.tell()
    bytes_remaining = file_size - header_bytes
    if bytes_remaining < 0 or bytes_remaining % bytes_per_block != 0:
        raise ValueError(
            f"{path.name}: data section is {bytes_remaining} bytes, not an even "
            f"number of {bytes_per_block}-byte RHD data blocks"
        )
    num_data_blocks = bytes_remaining // bytes_per_block

    return RhdHeader(
        path=path,
        version=version,
        sample_rate_hz=sample_rate_hz,
        samples_per_block=samples_per_block,
        timestamps_signed=timestamps_signed,
        eval_board_mode=eval_board_mode,
        reference_channel=reference_channel,
        amplifier_channels=amplifier_channels,
        aux_input_channels=aux_input_channels,
        supply_voltage_channels=supply_voltage_channels,
        board_adc_channels=board_adc_channels,
        board_dig_in_channels=board_dig_in_channels,
        board_dig_out_channels=board_dig_out_channels,
        num_temp_sensor_channels=num_temp_sensor_channels,
        header_bytes=header_bytes,
        bytes_per_block=bytes_per_block,
        num_data_blocks=num_data_blocks,
        sample_count=num_data_blocks * samples_per_block,
        impedance_magnitude_ohms=impedance_magnitude_ohms,
        impedance_phase_deg=impedance_phase_deg,
        board_dig_in_orders=board_dig_in_orders,
        board_dig_out_orders=board_dig_out_orders,
        dsp_enabled=bool(dsp_enabled),
        actual_dsp_cutoff_hz=float(actual_dsp_cutoff),
        actual_lower_bandwidth_hz=float(actual_lower_bandwidth),
        actual_upper_bandwidth_hz=float(actual_upper_bandwidth),
        desired_dsp_cutoff_hz=float(desired_dsp_cutoff),
        desired_lower_bandwidth_hz=float(desired_lower_bandwidth),
        desired_upper_bandwidth_hz=float(desired_upper_bandwidth),
        notch_filter_mode=int(notch_filter_mode),
        notch_filter_hz=notch_filter_hz,
        desired_impedance_test_frequency_hz=float(desired_impedance_test_frequency),
        actual_impedance_test_frequency_hz=float(actual_impedance_test_frequency),
        notes=notes,
    )


def block_dtype(header: RhdHeader) -> np.dtype:
    """Structured dtype describing one RHD data block for this header.

    Field order is the on-disk order: timestamps, amplifier, aux, supply, temp,
    ADC, digital in, digital out. The itemsize check catches any header field
    this parser got wrong before the numbers reach an analysis.
    """
    spb = header.samples_per_block
    fields: list[tuple[str, str, tuple[int, ...]]] = [
        ("t", "<i4" if header.timestamps_signed else "<u4", (spb,)),
    ]
    if header.amplifier_channels:
        fields.append(("amp", "<u2", (len(header.amplifier_channels), spb)))
    if header.aux_input_channels:
        fields.append(("aux", "<u2", (len(header.aux_input_channels), spb // 4)))
    if header.supply_voltage_channels:
        fields.append(("supply", "<u2", (len(header.supply_voltage_channels),)))
    if header.num_temp_sensor_channels:
        fields.append(("temp", "<i2", (header.num_temp_sensor_channels,)))
    if header.board_adc_channels:
        fields.append(("adc", "<u2", (len(header.board_adc_channels), spb)))
    if header.board_dig_in_channels:
        fields.append(("din", "<u2", (spb,)))
    if header.board_dig_out_channels:
        fields.append(("dout", "<u2", (spb,)))

    dtype = np.dtype(fields)
    if dtype.itemsize != header.bytes_per_block:
        raise ValueError(
            f"{header.path.name}: block dtype is {dtype.itemsize} bytes but the "
            f"header implies {header.bytes_per_block} bytes per block"
        )
    return dtype


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------


def read_rhd_header(path: Path) -> RhdHeader:
    """Parse only the header of one RHD file."""
    path = Path(path).expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        return _read_header(fid, path, file_size)


def _empty(rows: int, cols: int, dtype) -> np.ndarray:
    return np.empty((rows, cols), dtype=dtype)


def read_rhd_file(path: Path, wanted_channels: list[str] | None = None) -> RhdRunData:
    """Read one RHD file in a single pass.

    ``wanted_channels=None`` loads every enabled amplifier channel in header
    order; otherwise the result follows the requested order.
    """
    path = Path(path).expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        header = _read_header(fid, path, file_size)
        if wanted_channels is None:
            channels = list(header.amplifier_channels)
        else:
            missing = [
                channel
                for channel in wanted_channels
                if channel not in header.amplifier_channels
            ]
            if missing:
                available = ", ".join(header.amplifier_channels) or "none"
                raise ValueError(
                    f"{path.name}: channels not found: {', '.join(missing)}. "
                    f"Available amplifier channels: {available}"
                )
            channels = list(wanted_channels)
        indices = np.asarray(
            [header.amplifier_channels.index(channel) for channel in channels],
            dtype=np.intp,
        )
        dtype = block_dtype(header)
        blocks = np.fromfile(fid, dtype=dtype, count=header.num_data_blocks)

    if blocks.shape[0] != header.num_data_blocks:
        raise EOFError(
            f"{path.name}: expected {header.num_data_blocks} data blocks, "
            f"read {blocks.shape[0]}"
        )

    n_samples = header.sample_count
    n_blocks = header.num_data_blocks

    timestamps = np.ascontiguousarray(blocks["t"].reshape(-1))
    timestamp_gaps = (
        int(np.count_nonzero(np.diff(timestamps.astype(np.int64)) != 1))
        if timestamps.size
        else 0
    )

    if channels:
        # (blocks, n_amp, spb) -> select -> (n_sel, blocks, spb) -> (n_sel, N)
        amp_codes = np.ascontiguousarray(
            blocks["amp"][:, indices, :].transpose(1, 0, 2).reshape(len(channels), n_samples)
        )
        raw_uV = (amp_codes.astype(np.float32) - np.float32(AMPLIFIER_OFFSET)) * np.float32(
            AMPLIFIER_SCALE_uV
        )
    else:
        raw_uV = _empty(0, n_samples, np.float32)

    if header.aux_input_channels:
        n_aux = len(header.aux_input_channels)
        aux_codes = blocks["aux"].transpose(1, 0, 2).reshape(n_aux, n_samples // 4)
        aux_V = aux_codes.astype(np.float32) * np.float32(AUX_SCALE_V)
    else:
        aux_V = _empty(0, n_samples // 4, np.float32)

    if header.supply_voltage_channels:
        supply_codes = blocks["supply"].transpose(1, 0)
        supply_V = supply_codes.astype(np.float32) * np.float32(SUPPLY_SCALE_V)
    else:
        supply_V = _empty(0, n_blocks, np.float32)

    if header.num_temp_sensor_channels:
        temp_deg_C = blocks["temp"].transpose(1, 0).astype(np.float32) * np.float32(
            TEMP_SCALE_DEG_C
        )
    else:
        temp_deg_C = _empty(0, n_blocks, np.float32)

    if header.board_adc_channels:
        n_adc = len(header.board_adc_channels)
        adc_codes = blocks["adc"].transpose(1, 0, 2).reshape(n_adc, n_samples)
        adc_V = adc_codes.astype(np.float32)
        if header.adc_subtracts_offset:
            adc_V -= np.float32(AMPLIFIER_OFFSET)
        adc_V *= np.float32(header.adc_scale_V)
    else:
        adc_V = _empty(0, n_samples, np.float32)

    dig_in_raw = (
        np.ascontiguousarray(blocks["din"].reshape(-1))
        if header.board_dig_in_channels
        else np.zeros(0, dtype=np.uint16)
    )
    dig_out_raw = (
        np.ascontiguousarray(blocks["dout"].reshape(-1))
        if header.board_dig_out_channels
        else np.zeros(0, dtype=np.uint16)
    )
    del blocks

    return RhdRunData(
        folder=path.parent,
        rhd_files=[path],
        header=header,
        channels=channels,
        sample_rate_hz=float(header.sample_rate_hz),
        raw_uV=raw_uV,
        timestamps=timestamps,
        aux_V=np.ascontiguousarray(aux_V),
        supply_V=np.ascontiguousarray(supply_V),
        temp_deg_C=np.ascontiguousarray(temp_deg_C),
        adc_V=np.ascontiguousarray(adc_V),
        dig_in_raw=dig_in_raw,
        dig_out_raw=dig_out_raw,
        file_boundaries=[0],
        timestamp_gaps=timestamp_gaps,
    )


def read_rhd_run(folder: Path, wanted_channels: list[str] | None = None) -> RhdRunData:
    """Read and concatenate every ``*.rhd`` file in one run folder.

    Intan splits a long recording into timestamped files; sorting by name is
    chronological because the names carry ``_HHMMSS``. Discontinuities at the
    joins are counted in ``timestamp_gaps`` rather than silently stitched.
    """
    folder = Path(folder).expanduser()
    rhd_files = sorted(folder.glob("*.rhd"))
    if not rhd_files:
        raise FileNotFoundError(f"No .rhd files found in {folder}")

    parts = [read_rhd_file(path, wanted_channels) for path in rhd_files]
    first = parts[0]
    for part in parts[1:]:
        if part.channels != first.channels:
            raise ValueError(f"{folder.name}: channel list changed across RHD files.")
        if round(part.sample_rate_hz, 9) != round(first.sample_rate_hz, 9):
            raise ValueError(
                f"{folder.name}: mixed sample rates "
                f"{first.sample_rate_hz} and {part.sample_rate_hz}"
            )

    if len(parts) == 1:
        return RhdRunData(
            folder=folder,
            rhd_files=rhd_files,
            header=first.header,
            channels=first.channels,
            sample_rate_hz=first.sample_rate_hz,
            raw_uV=first.raw_uV,
            timestamps=first.timestamps,
            aux_V=first.aux_V,
            supply_V=first.supply_V,
            temp_deg_C=first.temp_deg_C,
            adc_V=first.adc_V,
            dig_in_raw=first.dig_in_raw,
            dig_out_raw=first.dig_out_raw,
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

    def _cat(name: str, axis: int) -> np.ndarray:
        arrays = [getattr(part, name) for part in parts]
        return np.concatenate(arrays, axis=axis)

    return RhdRunData(
        folder=folder,
        rhd_files=rhd_files,
        header=first.header,
        channels=first.channels,
        sample_rate_hz=first.sample_rate_hz,
        raw_uV=_cat("raw_uV", 1),
        timestamps=_cat("timestamps", 0),
        aux_V=_cat("aux_V", 1),
        supply_V=_cat("supply_V", 1),
        temp_deg_C=_cat("temp_deg_C", 1),
        adc_V=_cat("adc_V", 1),
        dig_in_raw=_cat("dig_in_raw", 0),
        dig_out_raw=_cat("dig_out_raw", 0),
        file_boundaries=boundaries,
        timestamp_gaps=gaps,
    )


__all__ = [
    "RHD_MAGIC_NUMBER",
    "RhdHeader",
    "RhdRunData",
    "block_dtype",
    "read_rhd_file",
    "read_rhd_header",
    "read_rhd_run",
]
