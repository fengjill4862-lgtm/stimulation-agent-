#!/usr/bin/env python3
"""Round-trip self-test for ``rhd_reader``.

Run with ``python3 rhd_reader_selftest.py``.

There is no reference RHD file small enough to check into the repo, so this
builds synthetic ones byte by byte from the same format description the reader
implements, writes them to a temp directory, and asserts the reader returns the
values that went in. That catches field-order and scaling mistakes, which are
the failure mode that matters: a mis-ordered header field still parses, still
yields plausible-looking microvolts, and silently corrupts every downstream
number.

Covered: version 3.5 (128-sample blocks, signed timestamps) and version 1.0
(60-sample blocks, unsigned timestamps, no temp/eval-board/reference fields),
auxiliary/supply/temperature/ADC/digital channels, disabled channels, the
digital-in bit unpacking, multi-file run concatenation with a deliberate
timestamp gap, and the error paths.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from rhd_reader import (
    ADC_SCALE_V_MODE_13,
    AMPLIFIER_OFFSET,
    AMPLIFIER_SCALE_uV,
    AUX_SCALE_V,
    RHD_MAGIC_NUMBER,
    SUPPLY_SCALE_V,
    TEMP_SCALE_DEG_C,
    read_rhd_file,
    read_rhd_header,
    read_rhd_run,
)


# -----------------------------------------------------------------------------
# Synthetic file writer (mirrors the format the reader parses)
# -----------------------------------------------------------------------------


def _qstring(text: str) -> bytes:
    if text == "":
        return struct.pack("<I", 0xFFFFFFFF)
    encoded = text.encode("utf-16le")
    return struct.pack("<I", len(encoded)) + encoded


def _channel_entry(
    name: str,
    native_order: int,
    signal_type: int,
    enabled: bool,
    impedance_magnitude: float = 0.0,
    impedance_phase: float = 0.0,
) -> bytes:
    return (
        _qstring(name)
        + _qstring(name)
        + struct.pack("<hh", native_order, native_order)
        + struct.pack("<hh", signal_type, 1 if enabled else 0)
        + struct.pack("<hh", 0, 0)          # chip channel, board stream
        + struct.pack("<hhhh", 0, 0, 0, 0)  # spike trigger settings
        + struct.pack("<ff", impedance_magnitude, impedance_phase)
    )


def build_rhd_bytes(
    *,
    version: tuple[int, int] = (3, 5),
    sample_rate_hz: float = 20000.0,
    amp_channels: list[str],
    aux_channels: list[str] = (),
    supply_channels: list[str] = (),
    adc_channels: list[str] = (),
    dig_in_channels: list[tuple[str, int]] = (),
    dig_out_channels: list[tuple[str, int]] = (),
    disabled_channels: list[str] = (),
    num_temp_sensor_channels: int = 0,
    eval_board_mode: int = 13,
    n_blocks: int = 3,
    first_timestamp: int = 0,
    impedances: dict[str, tuple[float, float]] | None = None,
    amp_codes: np.ndarray | None = None,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build one synthetic RHD file and the raw code arrays that went into it.

    ``amp_codes`` supplies the amplifier payload instead of random data, so
    other self-tests can write a signal with known content -- a stimulus train,
    a rhythm -- and check that an analysis recovers it.
    """
    major, minor = version
    spb = 60 if major == 1 else 128
    impedances = impedances or {}

    out = bytearray()
    out += struct.pack("<I", RHD_MAGIC_NUMBER)
    out += struct.pack("<hh", major, minor)
    out += struct.pack("<f", sample_rate_hz)
    out += struct.pack("<h", 1)                    # dsp_enabled
    out += struct.pack("<f", 1.0)                  # actual dsp cutoff
    out += struct.pack("<f", 0.1)                  # actual lower bandwidth
    out += struct.pack("<f", 7500.0)               # actual upper bandwidth
    out += struct.pack("<f", 1.0)                  # desired dsp cutoff
    out += struct.pack("<f", 0.1)                  # desired lower bandwidth
    out += struct.pack("<f", 7500.0)               # desired upper bandwidth
    out += struct.pack("<h", 2)                    # notch filter mode -> 60 Hz
    out += struct.pack("<f", 1000.0)               # desired impedance test freq
    out += struct.pack("<f", 1000.0)               # actual impedance test freq
    out += _qstring("note one") + _qstring("") + _qstring("note three")

    if (major == 1 and minor >= 1) or major > 1:
        out += struct.pack("<h", num_temp_sensor_channels)
    if (major == 1 and minor >= 3) or major > 1:
        out += struct.pack("<h", eval_board_mode)
    if major > 1:
        out += _qstring("")

    entries = bytearray()
    n_entries = 0
    for index, name in enumerate(amp_channels):
        magnitude, phase = impedances.get(name, (0.0, 0.0))
        entries += _channel_entry(name, index, 0, True, magnitude, phase)
        n_entries += 1
    for name in disabled_channels:
        # A disabled amplifier channel must be skipped without consuming block bytes.
        entries += _channel_entry(name, 0, 0, False)
        n_entries += 1
    for index, name in enumerate(aux_channels):
        entries += _channel_entry(name, index, 1, True)
        n_entries += 1
    for index, name in enumerate(supply_channels):
        entries += _channel_entry(name, index, 2, True)
        n_entries += 1
    for index, name in enumerate(adc_channels):
        entries += _channel_entry(name, index, 3, True)
        n_entries += 1
    for name, bit in dig_in_channels:
        entries += _channel_entry(name, bit, 4, True)
        n_entries += 1
    for name, bit in dig_out_channels:
        entries += _channel_entry(name, bit, 5, True)
        n_entries += 1

    out += struct.pack("<h", 1)                    # one signal group
    out += _qstring("Port B") + _qstring("B")
    out += struct.pack("<hhh", 1, n_entries, len(amp_channels))
    out += entries

    # Deterministic but non-trivial payload: every stream gets a distinct ramp.
    rng = np.random.default_rng(20260819)
    n_amp = len(amp_channels)
    n_aux = len(aux_channels)
    n_supply = len(supply_channels)
    n_adc = len(adc_channels)
    n_samples = spb * n_blocks

    if amp_codes is None:
        amp_codes = rng.integers(0, 65536, size=(n_amp, n_samples), dtype=np.uint16)
    else:
        amp_codes = np.asarray(amp_codes, dtype=np.uint16)
        if amp_codes.shape != (n_amp, n_samples):
            raise ValueError(
                f"amp_codes must be {(n_amp, n_samples)}, got {amp_codes.shape}"
            )
    aux_codes = rng.integers(0, 65536, size=(n_aux, n_samples // 4), dtype=np.uint16)
    supply_codes = rng.integers(0, 65536, size=(n_supply, n_blocks), dtype=np.uint16)
    temp_codes = rng.integers(
        -2000, 4000, size=(num_temp_sensor_channels, n_blocks)
    ).astype(np.int16)
    adc_codes = rng.integers(0, 65536, size=(n_adc, n_samples), dtype=np.uint16)
    din_codes = rng.integers(0, 65536, size=n_samples, dtype=np.uint16)
    dout_codes = rng.integers(0, 65536, size=n_samples, dtype=np.uint16)
    timestamps = np.arange(first_timestamp, first_timestamp + n_samples, dtype=np.int64)

    ts_dtype = "<i4" if ((major == 1 and minor >= 2) or major > 1) else "<u4"
    for block in range(n_blocks):
        lo, hi = block * spb, (block + 1) * spb
        out += timestamps[lo:hi].astype(ts_dtype).tobytes()
        if n_amp:
            out += amp_codes[:, lo:hi].astype("<u2").tobytes()
        if n_aux:
            out += aux_codes[:, block * (spb // 4) : (block + 1) * (spb // 4)].astype(
                "<u2"
            ).tobytes()
        if n_supply:
            out += supply_codes[:, block].astype("<u2").tobytes()
        if num_temp_sensor_channels:
            out += temp_codes[:, block].astype("<i2").tobytes()
        if n_adc:
            out += adc_codes[:, lo:hi].astype("<u2").tobytes()
        if dig_in_channels:
            out += din_codes[lo:hi].astype("<u2").tobytes()
        if dig_out_channels:
            out += dout_codes[lo:hi].astype("<u2").tobytes()

    truth = {
        "amp_codes": amp_codes,
        "aux_codes": aux_codes,
        "supply_codes": supply_codes,
        "temp_codes": temp_codes,
        "adc_codes": adc_codes,
        "din_codes": din_codes,
        "dout_codes": dout_codes,
        "timestamps": timestamps,
    }
    return bytes(out), truth


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------


_failures: list[str] = []
_checks = 0


def check(condition: bool, label: str) -> None:
    global _checks
    _checks += 1
    if not condition:
        _failures.append(label)
        print(f"  FAIL  {label}")


def check_close(actual, expected, label: str, tol: float = 1e-6) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        check(False, f"{label} (shape {actual.shape} != {expected.shape})")
        return
    if actual.size == 0:
        check(True, label)
        return
    worst = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    check(worst <= tol, f"{label} (max abs diff {worst:.3g} > {tol:.3g})")


def test_full_featured_v35(tmp: Path) -> None:
    print("version 3.5, all stream types")
    payload, truth = build_rhd_bytes(
        amp_channels=["B-016", "B-017", "B-018"],
        aux_channels=["B-AUX1", "B-AUX2"],
        supply_channels=["B-VDD"],
        adc_channels=["ANALOG-IN-1"],
        dig_in_channels=[("DIGITAL-IN-01", 0), ("DIGITAL-IN-04", 3)],
        dig_out_channels=[("DIGITAL-OUT-01", 0)],
        disabled_channels=["B-019"],
        num_temp_sensor_channels=1,
        n_blocks=4,
        impedances={"B-017": (2.11e5, -45.0)},
    )
    path = tmp / "full_260819_120000.rhd"
    path.write_bytes(payload)

    header = read_rhd_header(path)
    check(header.version == (3, 5), "version parsed")
    check(header.samples_per_block == 128, "128 samples per block")
    check(header.timestamps_signed, "timestamps signed at v3.5")
    check(header.sample_rate_hz == 20000.0, "sample rate")
    check(header.amplifier_channels == ["B-016", "B-017", "B-018"], "amplifier channels")
    check(header.aux_input_channels == ["B-AUX1", "B-AUX2"], "aux channels")
    check(header.supply_voltage_channels == ["B-VDD"], "supply channels")
    check(header.board_adc_channels == ["ANALOG-IN-1"], "adc channels")
    check(
        header.board_dig_in_channels == ["DIGITAL-IN-01", "DIGITAL-IN-04"],
        "dig-in channels",
    )
    check(header.board_dig_out_channels == ["DIGITAL-OUT-01"], "dig-out channels")
    check(header.num_temp_sensor_channels == 1, "temp sensor count")
    check(header.notch_filter_hz == 60.0, "notch filter decoded as 60 Hz")
    check(header.notes == ("note one", "", "note three"), "notes round-trip")
    check(header.num_data_blocks == 4, "block count from file size")
    check(header.sample_count == 512, "sample count")
    check(
        abs(header.impedance_magnitude_ohms["B-017"] - 2.11e5) < 1.0,
        "impedance magnitude kept",
    )
    check(header.eval_board_mode == 13, "eval board mode")
    check(header.adc_scale_V == ADC_SCALE_V_MODE_13, "adc scale for mode 13")

    run = read_rhd_file(path)
    check(run.channels == ["B-016", "B-017", "B-018"], "all channels loaded by default")
    check(run.timestamp_gaps == 0, "no timestamp gaps")
    check(run.n_samples == 512, "n_samples")
    check(abs(run.duration_s - 512 / 20000.0) < 1e-12, "duration")

    expected_uV = (truth["amp_codes"].astype(np.float64) - AMPLIFIER_OFFSET) * AMPLIFIER_SCALE_uV
    check_close(run.raw_uV, expected_uV, "amplifier microvolts", tol=1e-2)
    check_close(run.aux_V, truth["aux_codes"] * AUX_SCALE_V, "aux volts", tol=1e-6)
    check_close(
        run.supply_V, truth["supply_codes"] * SUPPLY_SCALE_V, "supply volts", tol=1e-5
    )
    check_close(
        run.temp_deg_C, truth["temp_codes"] * TEMP_SCALE_DEG_C, "temperature degC", tol=1e-4
    )
    expected_adc = (truth["adc_codes"].astype(np.float64) - AMPLIFIER_OFFSET) * ADC_SCALE_V_MODE_13
    check_close(run.adc_V, expected_adc, "adc volts", tol=1e-4)
    check_close(run.timestamps, truth["timestamps"], "timestamps", tol=0)

    # Aux is a quarter-rate stream and must stay on its own grid.
    check(run.aux_V.shape == (2, 128), "aux stays at rate/4")
    check(abs(header.aux_sample_rate_hz - 5000.0) < 1e-9, "aux sample rate")

    expected_bit0 = (truth["din_codes"] & 1) != 0
    expected_bit3 = (truth["din_codes"] & (1 << 3)) != 0
    check(np.array_equal(run.dig_in("DIGITAL-IN-01"), expected_bit0), "dig-in bit 0")
    check(np.array_equal(run.dig_in("DIGITAL-IN-04"), expected_bit3), "dig-in bit 3")

    # Channel selection and ordering.
    subset = read_rhd_file(path, ["B-018", "B-016"])
    check(subset.channels == ["B-018", "B-016"], "requested channel order honoured")
    check_close(
        subset.raw_uV[0], expected_uV[2], "selected channel maps to right data", tol=1e-2
    )
    check_close(subset.channel("B-016"), expected_uV[0], "channel() accessor", tol=1e-2)


def test_v1_layout(tmp: Path) -> None:
    print("version 1.0, 60-sample blocks")
    payload, truth = build_rhd_bytes(
        version=(1, 0),
        amp_channels=["A-000", "A-001"],
        n_blocks=5,
    )
    path = tmp / "old_260819_120000.rhd"
    path.write_bytes(payload)

    header = read_rhd_header(path)
    check(header.samples_per_block == 60, "60 samples per block at v1.0")
    check(not header.timestamps_signed, "timestamps unsigned before v1.2")
    check(header.num_temp_sensor_channels == 0, "no temp field at v1.0")
    check(header.eval_board_mode == 0, "no eval board mode at v1.0")
    check(header.reference_channel == "", "no reference channel at v1.0")
    check(header.sample_count == 300, "sample count at v1.0")

    run = read_rhd_file(path)
    expected_uV = (truth["amp_codes"].astype(np.float64) - AMPLIFIER_OFFSET) * AMPLIFIER_SCALE_uV
    check_close(run.raw_uV, expected_uV, "v1.0 amplifier microvolts", tol=1e-2)


def test_run_concatenation(tmp: Path) -> None:
    print("multi-file run concatenation")
    folder = tmp / "run_260819_120000"
    folder.mkdir()

    payload_a, truth_a = build_rhd_bytes(
        amp_channels=["B-016", "B-017"], n_blocks=2, first_timestamp=0
    )
    # Second file starts 1000 samples later: one deliberate boundary gap.
    payload_b, truth_b = build_rhd_bytes(
        amp_channels=["B-016", "B-017"], n_blocks=3, first_timestamp=256 + 1000
    )
    (folder / "run_260819_120000.rhd").write_bytes(payload_a)
    (folder / "run_260819_120100.rhd").write_bytes(payload_b)

    run = read_rhd_run(folder)
    check(len(run.rhd_files) == 2, "both files picked up")
    check(run.n_samples == 256 + 384, "samples concatenated")
    check(run.file_boundaries == [0, 256], "file boundary offsets")
    check(run.timestamp_gaps == 1, "boundary gap counted")

    expected = np.concatenate(
        [
            (truth_a["amp_codes"].astype(np.float64) - AMPLIFIER_OFFSET) * AMPLIFIER_SCALE_uV,
            (truth_b["amp_codes"].astype(np.float64) - AMPLIFIER_OFFSET) * AMPLIFIER_SCALE_uV,
        ],
        axis=1,
    )
    check_close(run.raw_uV, expected, "concatenated microvolts", tol=1e-2)

    contiguous = tmp / "contig_260819_120000"
    contiguous.mkdir()
    (contiguous / "a_260819_120000.rhd").write_bytes(
        build_rhd_bytes(amp_channels=["B-016"], n_blocks=2, first_timestamp=0)[0]
    )
    (contiguous / "a_260819_120100.rhd").write_bytes(
        build_rhd_bytes(amp_channels=["B-016"], n_blocks=2, first_timestamp=256)[0]
    )
    check(read_rhd_run(contiguous).timestamp_gaps == 0, "contiguous run reports no gap")


def test_error_paths(tmp: Path) -> None:
    print("error paths")
    not_rhd = tmp / "not_intan.rhd"
    not_rhd.write_bytes(b"\x00\x01\x02\x03" + b"\x00" * 64)
    try:
        read_rhd_header(not_rhd)
        check(False, "wrong magic number rejected")
    except ValueError as exc:
        check("not an Intan RHD2000 file" in str(exc), "wrong magic number rejected")

    payload, _truth = build_rhd_bytes(amp_channels=["B-016"], n_blocks=3)
    truncated = tmp / "truncated_260819_120000.rhd"
    truncated.write_bytes(payload[:-100])
    try:
        read_rhd_header(truncated)
        check(False, "partial data block rejected")
    except ValueError as exc:
        check("not an even" in str(exc), "partial data block rejected")

    good = tmp / "good_260819_120000.rhd"
    good.write_bytes(payload)
    try:
        read_rhd_file(good, ["B-099"])
        check(False, "missing channel rejected")
    except ValueError as exc:
        check("channels not found" in str(exc), "missing channel rejected")

    empty = tmp / "empty_run"
    empty.mkdir()
    try:
        read_rhd_run(empty)
        check(False, "empty run folder rejected")
    except FileNotFoundError:
        check(True, "empty run folder rejected")

    try:
        read_rhd_file(good).dig_in("DIGITAL-IN-01")
        check(False, "dig_in on a file without digital inputs rejected")
    except ValueError as exc:
        check(
            "is not in this file" in str(exc),
            "dig_in on a file without digital inputs rejected",
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rhd_selftest_") as tmp_name:
        tmp = Path(tmp_name)
        test_full_featured_v35(tmp)
        test_v1_layout(tmp)
        test_run_concatenation(tmp)
        test_error_paths(tmp)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} of {_checks} checks")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print(f"OK: all {_checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
