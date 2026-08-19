"""Synthetic-data self-tests for stim_analysis.

    /usr/local/bin/python3 -m stim_analysis.selftest

Writes only under a temporary directory. Every check is an assertion with a
message; the script prints one line per check and exits 1 on the first
failure. No real recordings are needed.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from plot_rhs_raw_wideband_with_stim_legend import RHS_MAGIC_NUMBER, read_rhs_channel
from plot_rhs_filtered_wideband import bandpass_filter_wideband
from rhs_reader import read_rhs_file, read_rhs_run
from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import (
    baseline_stats,
    blank_epochs,
    blank_windows,
    design_bandpass,
    extract_epochs,
    filter_epochs,
    window_slice,
)
from stim_analysis.load_rhs import (
    StimSettings,
    detect_stim_events,
    hardware_floor_ms,
    parse_run_folder_name,
    parse_stim_settings,
)
from stim_analysis.recovery import compute_recovery, condition_windows, mark_retained
from stim_analysis.validate import estimate_rail, railed_mask

UV_PER_CODE = 0.195
CODE_OFFSET = 32768


# -----------------------------------------------------------------------------
# Synthetic RHS writer (mirrors _read_header field for field)
# -----------------------------------------------------------------------------


def _qstring(text: str) -> bytes:
    if not text:
        return struct.pack("<I", 0)
    data = text.encode("utf-16le")
    return struct.pack("<I", len(data)) + data


def uv_to_code(uv: np.ndarray) -> np.ndarray:
    codes = np.round(np.asarray(uv, dtype=np.float64) / UV_PER_CODE + CODE_OFFSET)
    return np.clip(codes, 0, 65535).astype(np.uint16)


def write_synthetic_rhs(
    path: Path,
    *,
    sample_rate_hz: float,
    channel_names: list[str],
    amp_codes: np.ndarray,
    stim_words: np.ndarray,
    impedances_ohms: list[float],
    stim_step_uA: float = 2.0,
    with_dig_in: bool = True,
    dsp_enabled: bool = True,
    dsp_cutoff_hz: float = 1.0,
    lower_bw_hz: float = 0.1,
    upper_bw_hz: float = 7500.0,
) -> None:
    n_amp, n_samples = amp_codes.shape
    assert n_samples % 128 == 0, "synthetic data must be a whole number of 128-sample blocks"
    n_blocks = n_samples // 128
    parts: list[bytes] = [struct.pack("<I", RHS_MAGIC_NUMBER), struct.pack("<hh", 3, 3), struct.pack("<f", sample_rate_hz)]
    parts.append(struct.pack("<hffff", 1 if dsp_enabled else 0, dsp_cutoff_hz, lower_bw_hz, 1000.0, upper_bw_hz))  # dsp enabled + actual bws
    parts.append(struct.pack("<ffff", dsp_cutoff_hz, lower_bw_hz, 1000.0, upper_bw_hz))  # desired bws
    parts.append(struct.pack("<h", 0))  # notch
    parts.append(struct.pack("<ff", 1000.0, 1000.0))  # impedance test freq desired/actual
    parts.append(struct.pack("<hh", 0, 0))  # amp settle mode, charge recovery mode
    parts.append(struct.pack("<fff", stim_step_uA * 1e-6, 0.01e-6, 0.0))  # step size (A), cr limit, cr voltage
    parts.extend([_qstring(""), _qstring(""), _qstring("")])  # notes
    parts.append(struct.pack("<hh", 0, 0))  # dc amp saved, board mode
    parts.append(_qstring(""))  # reference channel
    n_groups = 2 if with_dig_in else 1
    parts.append(struct.pack("<h", n_groups))
    # Port A amplifier group
    parts.append(_qstring("Port A") + _qstring("A") + struct.pack("<hhh", 1, n_amp, n_amp))
    for index, name in enumerate(channel_names):
        parts.append(_qstring(name) + _qstring(name))
        parts.append(struct.pack("<hhhhhhhhhhh", index, index, 0, 1, index, 0, 0, 0, 0, 0, 0))
        parts.append(struct.pack("<ff", impedances_ohms[index], -60.0))
    if with_dig_in:
        parts.append(_qstring("Board Digital Inputs") + _qstring("DIN") + struct.pack("<hhh", 1, 1, 0))
        parts.append(_qstring("DIGITAL-IN-01") + _qstring("DIGITAL-IN-01"))
        parts.append(struct.pack("<hhhhhhhhhhh", 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0))
        parts.append(struct.pack("<ff", 0.0, 0.0))
    header = b"".join(parts)

    fields = [("t", "<i4", (128,)), ("amp", "<u2", (n_amp, 128)), ("stim", "<u2", (n_amp, 128))]
    if with_dig_in:
        fields.append(("din", "<u2", (128,)))
    blocks = np.zeros(n_blocks, dtype=np.dtype(fields))
    blocks["t"] = np.arange(n_samples, dtype=np.int32).reshape(n_blocks, 128)
    blocks["amp"] = amp_codes.reshape(n_amp, n_blocks, 128).transpose(1, 0, 2)
    blocks["stim"] = stim_words.reshape(n_amp, n_blocks, 128).transpose(1, 0, 2)
    with Path(path).open("wb") as fh:
        fh.write(header)
        blocks.tofile(fh)


def biphasic_words(
    n_samples: int,
    sample_rate_hz: float,
    onsets: np.ndarray,
    *,
    amplitude_uA: float,
    phase_us: float,
    interphase_us: float = 100.0,
    step_uA: float = 2.0,
    settle_ms: float = 1.0,
    compliance_on: set[int] | None = None,
) -> np.ndarray:
    words = np.zeros(n_samples, dtype=np.uint16)
    n_phase = int(round(phase_us * 1e-6 * sample_rate_hz))
    n_gap = int(round(interphase_us * 1e-6 * sample_rate_hz))
    n_settle = int(round(settle_ms * 1e-3 * sample_rate_hz))
    magnitude = int(round(amplitude_uA / step_uA))
    for index, onset in enumerate(onsets):
        o = int(onset)
        words[o : o + n_phase] = magnitude | 0x0100  # cathodic (negative) first
        words[o + n_phase + n_gap : o + 2 * n_phase + n_gap] = magnitude
        end = o + 2 * n_phase + n_gap
        words[o : end + n_settle] |= 0x2000  # amp settle through pulse + settle
        if compliance_on and index in compliance_on:
            words[o : o + n_phase] |= 0x8000
    return words


def _settings_xml(stim_channel: str, amplitude_uA: float, phase_us: float, n_pulses: int, sample_rate_hz: float) -> str:
    channels = "".join(
        f'        <Channel NativeChannelName="{name}" CustomChannelName="{name}" Enabled="True"/>\n'
        for name in ("A-024", "A-025", "A-026", "A-027")
    )
    stim_enabled = "True" if n_pulses > 0 else "False"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<IntanRHX SampleRateHertz="{sample_rate_hz:g}" Type="ControllerStimRecord" StimStepSizeMicroAmps="2" Version="3.5.0">\n'
        '    <GeneralConfig DSPEnabled="True" DesiredDSPCutoffFreqHertz="1" DesiredLowerBandwidthHertz="0.1" '
        'DesiredUpperBandwidthHertz="7500" SaveDCAmplifierWaveforms="False" DesiredLowerSettleBandwidthHertz="1000"/>\n'
        '    <SignalGroup Name="Port A" Prefix="A" Enabled="True">\n' + channels + '    </SignalGroup>\n'
        '    <StimParameters>\n'
        f'        <StimChannel NativeChannelName="{stim_channel}" StimEnabled="{stim_enabled}" Shape="Biphasic" Polarity="NegativeFirst" '
        f'FirstPhaseAmplitudeMicroAmps="{amplitude_uA:g}" SecondPhaseAmplitudeMicroAmps="{amplitude_uA:g}" '
        f'FirstPhaseDurationMicroseconds="{phase_us:g}" SecondPhaseDurationMicroseconds="{phase_us:g}" InterphaseDelayMicroseconds="100" '
        f'NumberOfStimPulses="{max(n_pulses, 1)}" PulseTrainPeriodMicroseconds="999990" RefractoryPeriodMicroseconds="1000" '
        'PostStimAmpSettleMicroseconds="1000" EnableAmpSettle="True" EnableChargeRecovery="True" PulseOrTrain="PulseTrain" Source="KeyPressF1"/>\n'
        '    </StimParameters>\n'
        '</IntanRHX>\n'
    )


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------


class Check:
    def __init__(self) -> None:
        self.n = 0

    def ok(self, condition: bool, message: str) -> None:
        self.n += 1
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {message}")
        if not condition:
            raise AssertionError(message)


def check_reader(tmp: Path, c: Check) -> None:
    fs = 30000.0
    n = 128 * 400  # 1.7 s
    rng = np.random.default_rng(0)
    names = ["A-024", "A-025", "A-026", "A-027"]
    uv = rng.normal(0, 30, size=(4, n))
    uv[1, 1000:1200] = 6389.565  # top rail on channel 1
    codes = uv_to_code(uv)
    onsets = np.arange(3000, n - 3000, 30000)
    words = np.zeros((4, n), dtype=np.uint16)
    words[3] = biphasic_words(n, fs, onsets, amplitude_uA=50, phase_us=200)
    path = tmp / "synthetic_260101_120000.rhs"
    write_synthetic_rhs(path, sample_rate_hz=fs, channel_names=names, amp_codes=codes, stim_words=words, impedances_ohms=[29e3, 240e3, 164e3, 3.8e6])
    run = read_rhs_file(path)
    c.ok(run.channels == names and run.sample_rate_hz == fs and run.n_samples == n, "fast reader: header/channels/samples")
    c.ok(np.array_equal(run.raw_uV, ((codes.astype(np.float32) - np.float32(32768.0)) * np.float32(0.195))), "fast reader: microvolt conversion")
    for index, name in enumerate(names):
        legacy = read_rhs_channel(path, name)
        c.ok(np.array_equal(legacy.raw_uV, run.raw_uV[index]) and np.array_equal(legacy.stim_uA, run.stim_uA(name)), f"fast reader == legacy per-channel reader ({name})")
    c.ok(abs(run.header.impedance_magnitude_ohms["A-025"] - 240e3) < 1 and abs(run.header.impedance_phase_deg["A-025"] + 60) < 1e-3, "header impedance round-trips")
    c.ok(run.header.board_dig_in_channels == ["DIGITAL-IN-01"] and run.header.dsp_enabled, "header digital-in + dsp fields")
    c.ok(run.timestamp_gaps == 0 and run.file_boundaries == [0], "fast reader: timestamps contiguous")
    # two-file run: concatenation + boundary gap counting
    path2 = tmp / "synthetic_260101_120100.rhs"
    write_synthetic_rhs(path2, sample_rate_hz=fs, channel_names=names, amp_codes=codes, stim_words=words, impedances_ohms=[29e3, 240e3, 164e3, 3.8e6])
    both = read_rhs_run(tmp)
    c.ok(both.n_samples == 2 * n and both.file_boundaries == [0, n] and both.timestamp_gaps == 1, "two-file run: concatenated, one boundary gap")


def check_events(c: Check) -> None:
    fs = 30000.0
    cfg = AnalysisConfig()
    n = 128 * 12000  # 51.2 s
    onsets = 30000 + np.arange(50) * 30001
    words = biphasic_words(n, fs, onsets, amplitude_uA=50, phase_us=200, settle_ms=1.0)
    stim_uA = ((words & 0x00FF).astype(np.float32)) * np.where((words & 0x0100) != 0, -1.0, 1.0).astype(np.float32) * 2.0
    events = detect_stim_events(stim_uA, words, fs, cfg, train_period_s=1.0)
    c.ok(events.n_events == 50 and events.n_trains == 1, f"events: 50 pulses detected in one train (got {events.n_events}, {events.n_trains})")
    c.ok(np.allclose(events.amplitude_uA, 50) and np.allclose(events.first_phase_us, 200) and np.all(events.first_phase_sign == -1), "events: amplitude 50 uA, phase 200 us, cathodic first")
    c.ok(events.amp_settle_bit_any and not events.compliance_bit_any, "events: amp-settle flag seen, no compliance")
    floor = hardware_floor_ms(events, None, fs)
    c.ok(abs(floor - 1.5) < 0.05, f"hardware floor = pulse 0.5 ms + settle 1.0 ms (got {floor:.3f})")
    # compliance: 46 pulses delivered, bit set on the last delivered ones
    words46 = biphasic_words(n, fs, onsets[:46], amplitude_uA=250, phase_us=500, compliance_on={44, 45})
    stim46 = ((words46 & 0x00FF).astype(np.float32)) * np.where((words46 & 0x0100) != 0, -1.0, 1.0).astype(np.float32) * 2.0
    ev46 = detect_stim_events(stim46, words46, fs, cfg, train_period_s=1.0)
    c.ok(ev46.n_events == 46 and ev46.compliance_bit_any and int(ev46.compliance_flag.sum()) == 2, "events: 46 pulses + compliance bits detected")
    # paired pulses -> trains split correctly (2 pulses 30 ms apart, trains 1 s apart)
    paired = np.sort(np.concatenate([onsets[:10], onsets[:10] + 900]))
    wp = biphasic_words(n, fs, paired, amplitude_uA=250, phase_us=300)
    sp = ((wp & 0x00FF).astype(np.float32)) * np.where((wp & 0x0100) != 0, -1.0, 1.0).astype(np.float32) * 2.0
    evp = detect_stim_events(sp, wp, fs, cfg, train_period_s=0.03)
    c.ok(evp.n_events == 20 and evp.n_trains == 10, f"events: paired pulses -> 20 pulses in 10 trains (got {evp.n_events}, {evp.n_trains})")
    meta = parse_run_folder_name("step1_260817_141336 A-030 stim cathodic first 200us 50uA 1Hz 50 pulses 999ms RP comp")
    c.ok(meta.run_id == "260817_141336" and meta.label == "step1" and meta.stim_channel == "A-030" and meta.amplitude_uA == 50 and meta.phase_us == 200 and meta.pulses == 50 and meta.rp_ms == 999 and meta.comp_token, "folder-name parser")


def check_rail(c: Check) -> None:
    fs = 30000.0
    cfg = AnalysisConfig()
    rng = np.random.default_rng(1)
    x = rng.normal(0, 30, size=60000).astype(np.float32)
    x[10000:10150] = 6389.565  # 5 ms at the top rail
    x[20000:20150] = -6389.565  # 5 ms at the bottom rail
    rail = estimate_rail(x, fs, cfg, "A-000")
    c.ok(rail.is_railed and rail.at_adc_full_scale, "rail: 5 ms plateaus at +/-6389.6 uV detected as ADC rail")
    c.ok(abs(rail.pos_level_uV - 6389.565) < 0.01 and abs(rail.neg_level_uV + 6389.565) < 0.01, "rail: levels exact")
    c.ok(abs(rail.pct_railed - 100.0 * 300 / 60000) < 1e-6 and rail.n_episodes == 2, f"rail: pct {rail.pct_railed:.3f} %, 2 episodes")
    mask = railed_mask(x, rail, cfg, fs)
    c.ok(int(mask.sum()) == 300 and mask[10000] and mask[20149] and not mask[10150], "rail: mask marks exactly the plateau samples")
    y = rng.normal(0, 30, size=60000).astype(np.float32)
    y[5000:5020] = 1500.0  # 20 samples (0.67 ms) at 1500 uV: railed but not ADC
    rail2 = estimate_rail(y, fs, cfg, "A-001")
    c.ok(rail2.is_railed and not rail2.at_adc_full_scale and abs(rail2.pos_level_uV - 1500) < 0.01 and not np.isfinite(rail2.neg_level_uV), "rail: non-ADC plateau at 1500 uV, negative side clean")
    z = rng.normal(0, 30, size=60000).astype(np.float32)
    rail3 = estimate_rail(z, fs, cfg, "A-002")
    c.ok(not rail3.is_railed and rail3.pct_railed == 0.0, "rail: pure noise is not railed")


def check_recovery(c: Check) -> None:
    fs = 30000.0
    cfg = AnalysisConfig()
    rng = np.random.default_rng(2)
    n = 128 * 24000  # 102 s
    onsets = 30000 + np.arange(50) * 30001
    sd = 10.0
    raw = rng.normal(0, sd, size=n)
    t = np.arange(n) / fs
    amplitude, tau = 5000.0, 0.030
    for onset in onsets:
        seg = slice(int(onset), int(onset) + 30000)
        tt = t[seg] - t[int(onset)]
        raw[seg] += amplitude * np.exp(-tt / tau)
    epochs = extract_epochs(raw[np.newaxis, :], fs, onsets, np.arange(1, 51), cfg, channels=["A-000"])
    c.ok(bool(epochs.kept.all()) and epochs.raw.shape == (1, 50, int(round(2.5 * fs))), "epochs: 50 padded epochs of 2.5 s")
    ep = epochs.raw[0]
    mean, bsd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
    c.ok(abs(float(np.median(bsd)) - sd) < 1.5, f"baseline SD ~ {sd} uV (got {np.median(bsd):.2f})")
    frame = compute_recovery(ep - mean[:, None], epochs.t_ms, fs, bsd, cfg, core=epochs.core, event_numbers=epochs.event_numbers)
    threshold = np.maximum(3 * bsd, cfg.threshold_floor_uV)
    analytic_ms = tau * np.log(amplitude / threshold) * 1e3
    err = frame["recovery_ms"].to_numpy() - analytic_ms
    # With 10 uV white noise at 30 kHz the *last* threshold crossing is jittered
    # late by rare noise excursions (spec definition); the bulk must still sit
    # on the analytic crossing and never before it.
    c.ok(bool((err > -1.0).all()) and float(np.median(np.abs(err))) < 15.0 and float(err.max()) < 40.0 and not frame["censored"].any(), f"recovery: never before the analytic crossing; median lag {np.median(err):.1f} ms, max {err.max():.1f} ms from late noise excursions (10 uV noise, 100 uV threshold)")
    # Low noise: crossing must be sample-precise (within 2 ms)
    quiet_raw = rng.normal(0, 1.0, size=n)
    for onset in onsets:
        seg = slice(int(onset), int(onset) + 30000)
        tt = t[seg] - t[int(onset)]
        quiet_raw[seg] += amplitude * np.exp(-tt / tau)
    q_epochs = extract_epochs(quiet_raw[np.newaxis, :], fs, onsets, np.arange(1, 51), cfg, channels=["A-000"])
    q_mean, q_sd = baseline_stats(q_epochs.raw[0], q_epochs.t_ms, cfg.baseline_ms)
    q_frame = compute_recovery(q_epochs.raw[0] - q_mean[:, None], q_epochs.t_ms, fs, q_sd, cfg, core=q_epochs.core)
    q_err = np.abs(q_frame["recovery_ms"].to_numpy() - tau * np.log(amplitude / cfg.threshold_floor_uV) * 1e3)
    c.ok(bool((q_err < 2.0).all()), f"recovery within 2 ms of analytic crossing at low noise (max err {q_err.max():.2f} ms)")
    # zero case
    zero = rng.normal(0, sd, size=(5, ep.shape[1]))
    fz = compute_recovery(zero, epochs.t_ms, fs, np.full(5, sd), cfg, core=epochs.core)
    c.ok(bool((fz["recovery_ms"] == 0).all()) and not fz["censored"].any(), "recovery: no artifact -> 0 ms, not censored")
    # censored case: never quiet
    huge = np.zeros((3, ep.shape[1]))
    tt = epochs.t_ms / 1e3
    huge[:, :] = 5000.0 * np.exp(-np.clip(tt, 0, None) / 10.0) * (tt >= 0)
    fc = compute_recovery(huge, epochs.t_ms, fs, np.full(3, sd), cfg, core=epochs.core)
    c.ok(bool(fc["censored"].all()) and bool((fc["recovery_ms"] >= cfg.epoch_ms[1] - 1).all()), "recovery: 10 s time constant -> censored at epoch end")
    # condition windows / P90 rule
    frame.insert(0, "channel", "A-000")
    frame.insert(0, "run_id", "r1")
    windows = condition_windows(frame, cfg, {"r1": 1.5})
    row = windows.iloc[0]
    expected = float(np.quantile(frame["recovery_ms"], cfg.recovery_quantile)) + cfg.blank_margin_ms
    c.ok(abs(row["post_start_ms"] - expected) < 1e-9 and row["n_retained_early"] >= 45, f"condition window: post start = P90 + margin ({row['post_start_ms']:.1f} ms), retained {row['n_retained_early']}")
    c.ok(row["verdict"] == "late_only" and bool(row["early_possible"]), "verdict late_only for ~120 ms recovery")
    marked = mark_retained(frame, windows, cfg)
    c.ok(int(marked["retained_early"].sum()) == int(row["n_retained_early"]), "mark_retained agrees with condition_windows")


def check_filter_and_blank(c: Check) -> None:
    fs = 30000.0
    rng = np.random.default_rng(3)
    x = rng.normal(0, 20, size=(4, 90000))
    design = design_bandpass(fs, 1.0, 150.0, 4, True, pad_ms=500)
    ours = filter_epochs(x, design)
    ref = np.stack([bandpass_filter_wideband(row, fs, (1.0, 150.0)) for row in x])
    c.ok(np.allclose(ours, ref, atol=1e-9), "filter design == plot_rhs_filtered_wideband.bandpass_filter_wideband (zero-phase)")
    c.ok(design.description["method"].startswith("sosfiltfilt") and design.description["low_hz"] == 1.0, "filter design metadata")
    t_ms = (np.arange(90000) - 30000) * 1e3 / fs
    windows = blank_windows(np.array([100.0, np.nan, 20.0, 0.0]), AnalysisConfig(), 1.5)
    c.ok(windows[0][0] == (-1.0, 105.0) and windows[1][0][1] == 900.0 and windows[2][0] == (-1.0, 25.0) and windows[3][0] == (-1.0, 5.0), "blank windows: per-epoch recovery + margin, censored -> epoch end, zero recovery -> margin")
    windows_floor = blank_windows(np.array([0.0]), AnalysisConfig(), 12.0)
    c.ok(windows_floor[0][0] == (-1.0, 12.0), "blank windows: hardware floor wins when larger than recovery + margin")
    blanked = blank_epochs(x, t_ms, windows)
    sl = window_slice(t_ms, -1.0, 105.0)
    inside = blanked[0, sl]
    c.ok(bool(np.all(np.abs(np.diff(inside, 2)) < 1e-6)), "blanked window is a straight line (linear interpolation)")
    c.ok(np.array_equal(blanked[3, : sl.start], x[3, : sl.start]), "samples outside the blank are untouched")


def check_pipeline(tmp: Path, c: Check) -> None:
    from stim_analysis.pipeline import render_outputs, run_session, write_outputs

    fs = 30000.0
    parent = tmp / "session"
    parent.mkdir()
    names = ["A-024", "A-025", "A-026", "A-027"]
    rng = np.random.default_rng(4)
    n = 128 * 6000  # 25.6 s
    onsets = 30000 + np.arange(20) * 30001

    def make_run(folder_name: str, amplitude: float, phase: float, n_pulses: int, artifact_uV: float, tau_s: float, n_commanded: int | None = None) -> None:
        folder = parent / folder_name
        folder.mkdir()
        uv = rng.normal(0, 15, size=(4, n))
        words = np.zeros((4, n), dtype=np.uint16)
        if n_pulses:
            words[3] = biphasic_words(n, fs, onsets[:n_pulses], amplitude_uA=amplitude, phase_us=phase)
            t = np.arange(n) / fs
            for onset in onsets[:n_pulses]:
                seg = slice(int(onset), min(n, int(onset) + 30000))
                tt = t[seg] - t[int(onset)]
                for ch in range(4):
                    uv[ch, seg] += -artifact_uV * (0.4 + 0.2 * ch) * np.exp(-tt / tau_s)
        codes = uv_to_code(uv)
        write_synthetic_rhs(folder / f"{folder_name.split(' ')[0]}.rhs", sample_rate_hz=fs, channel_names=names, amp_codes=codes, stim_words=words, impedances_ohms=[29e3, 240e3, 164e3, 88e3])
        (folder / "settings.xml").write_text(_settings_xml("A-027", amplitude, phase, n_pulses if n_commanded is None else n_commanded, fs))

    make_run("recording_260101_100000", 0, 0, 0, 0, 0.03)
    make_run("step1_260101_100100 A-027 stim cathodic first 200us 10uA 1Hz 20 pulses 999ms RP", 10, 200, 20, 3000, 0.02)
    make_run("step1_260101_100200 A-027 stim cathodic first 200us 20uA 1Hz 20 pulses 999ms RP", 20, 200, 20, 8000, 0.03)
    make_run("step2_260101_100300 A-027 stim cathodic first 500us 20uA 1Hz 16 pulses 999ms RP comp", 20, 500, 16, 8000, 0.03, n_commanded=20)

    cfg = dataclasses.replace(AnalysisConfig(), trace_amplitudes_uA=(10.0, 20.0), min_trials=10)
    seen: dict[str, object] = {}
    result = run_session(parent, cfg, stage="recovery", on_validation=lambda frame: seen.setdefault("validation", frame))
    c.ok("validation" in seen, "pipeline: on_validation called before analysis")
    v = result.validation.set_index("run_id")
    c.ok(list(v["block"]) == ["baseline", "block1", "block1", "block2"], f"pipeline: blocks {list(v['block'])}")
    c.ok(bool(v.loc["260101_100300", "compliance_flag"]) and not bool(v.loc["260101_100300", "included"]) and v.loc["260101_100300", "exclusion_reason"] == "compliance", "pipeline: 16/20 pulses -> compliance, excluded")
    c.ok(result.baseline_run_id == "260101_100000", "pipeline: baseline auto-selected")
    c.ok(result.exit_code == 1, "pipeline: exit code 1 when a run is excluded")
    c.ok(set(result.figures) >= {"fig01_recovery_vs_amplitude", "fig02_recovery_vs_impedance", "fig02b_recovery_vs_distance", "fig03_raw_traces_low_amplitudes", "fig03b_raw_traces_low_amplitudes_zoom", "figS5_recovery_all_runs_incl_excluded"}, f"pipeline: recovery figures present ({len(result.figures)})")
    trials = result.tables["trials_recovery"]
    c.ok(len(trials) == 4 * (20 + 20 + 16), f"pipeline: {len(trials)} recovery trials")
    t10 = trials[(trials["run_id"] == "260101_100100") & (trials["channel"] == "A-027")]["recovery_ms"].median()
    t20 = trials[(trials["run_id"] == "260101_100200") & (trials["channel"] == "A-027")]["recovery_ms"].median()
    c.ok(t20 > t10 > 20, f"pipeline: recovery grows with artifact (10 uA {t10:.1f} ms, 20 uA {t20:.1f} ms)")
    c.ok(all(caption.startswith("n trials retained") and "Blanking:" in caption and "Filter:" in caption for caption in result.captions.values()), "pipeline: every caption states n / blanking / filter")
    out_dir = tmp / "out"
    result.output_dir = out_dir
    items = render_outputs(result)
    stems = {Path(path).name for path, _ in items}
    c.ok({"table01_validation.csv", "table02_impedance_per_run.csv", "table03_recovery_summary.csv", "trials_recovery.csv", "stim_analysis_metadata.json", "figures_index.csv", "run_log.txt"} <= stems, "pipeline: table/metadata outputs rendered")
    per_run = [p for p, _ in items if p.parent != out_dir]
    c.ok(any("stim_analysis_260101_100100_recovery_trials.csv" == p.name for p in per_run), "pipeline: per-run CSVs go into run folders")
    written = write_outputs(result)
    c.ok(all(p.exists() for p in written) and not any(p.name.startswith(".") for p in out_dir.iterdir()), f"pipeline: {len(written)} files written atomically")
    val_only = run_session(parent, cfg, stage="validate")
    c.ok(not val_only.figures and "trials_recovery" not in val_only.tables, "pipeline: stage=validate stops before analysis")


def main() -> int:
    started = time.time()
    c = Check()
    with tempfile.TemporaryDirectory(prefix="stim_analysis_selftest_") as tmpdir:
        tmp = Path(tmpdir)
        try:
            print("reader"); check_reader(tmp / "reader", c) if (tmp / "reader").mkdir() is None else None
            print("events"); check_events(c)
            print("rail"); check_rail(c)
            print("recovery"); check_recovery(c)
            print("filter/blank"); check_filter_and_blank(c)
            print("pipeline"); (tmp / "pipe").mkdir(); check_pipeline(tmp / "pipe", c)
            try:
                from stim_analysis import selftest_secondary  # phase 3 checks, when present
            except ImportError:
                selftest_secondary = None
            if selftest_secondary is not None:
                print("secondary"); selftest_secondary.run(tmp / "secondary", c)
        except AssertionError:
            print(f"\nSELFTEST FAILED after {c.n} checks")
            return 1
        except Exception:
            traceback.print_exc()
            print(f"\nSELFTEST ERROR after {c.n} checks")
            return 1
    print(f"\nall {c.n} checks passed in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
