#!/usr/bin/env python3
"""Synthetic end-to-end checks for the evoked-response sweep.

Run with ``python3 -m evoked_sweep.selftest``. Writes only under a temp dir.

The point of these checks is that the pulse recovery has no ground truth in the
real data -- nothing recorded the trigger -- so the only way to know it works is
to build recordings whose stimulus times are known by construction and confirm
they come back. The synthetic runs carry the things that made naive detection
fail on the real files: an ongoing rhythm comparable to the stimulus, a low
duty cycle, and amplitudes ranging from obvious down to invisible.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from rhd_reader import AMPLIFIER_OFFSET, AMPLIFIER_SCALE_uV
from rhd_reader_selftest import build_rhd_bytes

from .artifact import decade_mislabel_evidence, run_evidence, sweep_evidence
from .config import EvokedConfig
from .load import load_run
from .metrics import band_power, evoked_deflection
from .naming import (
    contact_index,
    contact_positions_um,
    discover_runs,
    parse_amplitude,
    parse_protocol_name,
    parse_wiring,
)
from .peaks import analyse_channel_peaks
from .pipeline import _apply_decade_flags, _apply_wiring_filter, run_single
from .pulses import PulseTrain, recover_pulses

FS = 5000.0
SAMPLES_PER_BLOCK = 128

_failures: list[str] = []
_checks = 0


def check(condition: bool, label: str) -> None:
    global _checks
    _checks += 1
    if not condition:
        _failures.append(label)
        print(f"  FAIL  {label}")


def _codes_from_uV(signal_uV: np.ndarray) -> np.ndarray:
    codes = np.round(signal_uV / AMPLIFIER_SCALE_uV + AMPLIFIER_OFFSET)
    return np.clip(codes, 0, 65535).astype(np.uint16)


def make_run(
    folder: Path,
    *,
    amplitude_uV: float,
    period_s: float = 0.259,
    n_pulses: int = 50,
    first_onset_s: float = 10.0,
    duration_s: float = 40.0,
    pulse_width_ms: float = 5.0,
    rhythm_uV: float = 60.0,
    rhythm_hz: float = 4.0,
    noise_uV: float = 20.0,
    shape: str = "coupling",
    gamma_boost_uV: float = 0.0,
    seed: int = 7,
) -> np.ndarray:
    """Write one synthetic run folder; returns the true onset times in seconds.

    ``shape='coupling'`` puts a rectangle exactly under the pulse -- what direct
    current injection looks like. ``shape='response'`` puts a delayed decaying
    transient after it -- what a synaptic response looks like. The artifact
    criteria have to tell these apart. ``shape='ep'`` is the full menagerie the
    peak analysis must sort out: the coupling rectangle, an onset spike, an
    off-edge transient at 6.5 ms, then two genuine components -- N at 9 ms
    (-150 uV, sigma 1 ms) and P at 15 ms (+300 uV, sigma 2.5 ms).
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(duration_s * FS / SAMPLES_PER_BLOCK)
    n_samples = n_blocks * SAMPLES_PER_BLOCK
    time = np.arange(n_samples) / FS

    signal = rhythm_uV * np.sin(2 * np.pi * rhythm_hz * time)
    signal += rng.normal(0.0, noise_uV, size=n_samples)

    onsets = first_onset_s + np.arange(n_pulses) * period_s
    width = int(round(pulse_width_ms / 1000.0 * FS))

    if shape == "ep":
        length = int(round(0.060 * FS))
        t_ms = np.arange(length) / FS * 1000.0
        template = np.zeros(length)
        template[t_ms < pulse_width_ms] += amplitude_uV  # coupling rectangle
        template[t_ms < 0.6] -= 400.0  # onset spike
        template -= 120.0 * np.exp(-0.5 * ((t_ms - 6.5) / 0.4) ** 2)  # off-edge
        template -= 150.0 * np.exp(-0.5 * ((t_ms - 9.0) / 1.0) ** 2)  # N component
        template += 300.0 * np.exp(-0.5 * ((t_ms - 15.0) / 2.5) ** 2)  # P component

    if shape == "biphasic":
        # The slow Keithley waveform: 0.2 s cathodic-analog phase, 0.1 s
        # interphase pause, 0.2 s opposite phase. Total width = pulse_width_ms.
        phase = int(round(0.2 * FS))
        pause = int(round(0.1 * FS))
        template = np.zeros(2 * phase + pause)
        template[:phase] += amplitude_uV
        template[phase + pause :] -= amplitude_uV

    for onset in onsets:
        start = int(round(onset * FS))
        if shape in ("ep", "biphasic"):
            stop = min(n_samples, start + template.size)
            signal[start:stop] += template[: stop - start]
        elif shape == "coupling":
            stop = min(n_samples, start + width)
            signal[start:stop] += amplitude_uV
        else:
            # 3 ms delay, then a 15 ms decay: too slow and too late to be coupling.
            delay = int(round(0.003 * FS))
            length = int(round(0.040 * FS))
            begin = start + delay
            stop = min(n_samples, begin + length)
            if stop > begin:
                tau = np.arange(stop - begin) / FS
                signal[begin:stop] += amplitude_uV * np.exp(-tau / 0.015)
        if gamma_boost_uV > 0:
            # Band-limited noise, not a single tone: a real band-power change is
            # broadband, and one spectral line would be diluted by the width of
            # the band it sits in.
            stop = min(n_samples, start + int(round(period_s * FS)))
            length = stop - start
            if length > 8:
                white = rng.normal(0.0, 1.0, size=length)
                spectrum = np.fft.rfft(white)
                frequencies = np.fft.rfftfreq(length, d=1.0 / FS)
                spectrum[(frequencies < 30.0) | (frequencies > 100.0)] = 0.0
                shaped = np.fft.irfft(spectrum, n=length)
                if shaped.std() > 0:
                    signal[start:stop] += gamma_boost_uV * shaped / shaped.std()

    codes = _codes_from_uV(np.vstack([signal, signal * 0.8]))
    payload, _truth = build_rhd_bytes(
        version=(3, 5),
        sample_rate_hz=FS,
        amp_channels=["B-017", "B-018"],
        n_blocks=n_blocks,
        impedances={"B-017": (2.0e5, -45.0), "B-018": (2.1e5, -45.0)},
        amp_codes=codes,
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{folder.name}.rhd").write_bytes(payload)
    return onsets


def _config(session: Path, **overrides) -> EvokedConfig:
    defaults = dict(session_folder=session, expected_pulses=50, pulse_width_ms=5.0)
    defaults.update(overrides)
    return EvokedConfig(**defaults)


def test_naming() -> None:
    print("folder-name parsing")
    cases = {
        "-0_02mA": -0.02,
        "+0_2mA": 0.2,
        "-02mA": -0.2,
        "-08mA": -0.8,
        "+1mA": 1.0,
        "-05 1 mA": -0.5,
        "-0_01 unsure mA": -0.01,
    }
    for text, expected in cases.items():
        value, _assumed = parse_amplitude(text)
        check(value is not None and abs(value - expected) < 1e-9, f"{text} -> {expected} mA")

    value, assumed = parse_amplitude("0_1mA")
    check(value == 0.1 and assumed, "unsigned name parses positive and is flagged")
    check(parse_amplitude("1_260819_171913")[0] is None, "baseline recording has no amplitude")

    wiring = parse_wiring("stim 1 stim ground 2 recording ground 3")
    check(
        (wiring.stim_wire, wiring.stim_ground_wire, wiring.recording_ground_wire) == (1, 2, 3),
        "wiring parsed",
    )
    check(not wiring.user_marked_artifact, "clean config not marked artifact")
    check(
        parse_wiring("stim 1 recording ground 3 common ground - artifact").common_ground,
        "common ground detected",
    )

    protocol_name = "0.001mA_-0.001mA_pulsewidth0.3s_interval4.5s_pulsenumber25"
    for text in (protocol_name, protocol_name + ".rhd", protocol_name + ".onset.txt"):
        info = parse_protocol_name(text)
        check(
            info is not None
            and (info.amplitude_mA, info.second_phase_mA) == (0.001, -0.001)
            and (info.pulse_width_s, info.interval_s, info.pulse_number) == (0.3, 4.5, 25),
            f"protocol name parses ({text[-12:]})",
        )
    cathodic = parse_protocol_name("-0.02mA_0.02mA_pulsewidth0.3s_interval4.5s_pulsenumber25")
    check(
        cathodic is not None and cathodic.amplitude_mA == -0.02,
        "cathodic-leading protocol keeps the leading sign",
    )
    check(
        parse_amplitude(protocol_name) == (None, False),
        "legacy parser is shielded from protocol names (no 0.0 mA misparse)",
    )

    check(contact_index("B-017") == 17, "contact index from trailing digits")
    positions = contact_positions_um(("B-017", "B-018", "B-021"), 500.0)
    check(
        positions == {"B-017": 0.0, "B-018": 500.0, "B-021": 2000.0},
        f"relative positions by index x pitch (got {positions})",
    )
    ordered = contact_positions_um(("B-021", "B-017"), 500.0, order=("B-021", "B-017"))
    check(
        ordered == {"B-021": 0.0, "B-017": 500.0},
        f"explicit contact order overrides indices (got {ordered})",
    )
    unknown = contact_positions_um(("B-017", "weird"), 500.0)
    check(
        unknown["B-017"] == 0.0 and unknown["weird"] != unknown["weird"],
        "channel without trailing digits gets NaN position",
    )


def test_pulse_recovery(tmp: Path) -> None:
    print("pulse recovery against known onsets")
    session = tmp / "recovery"
    folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_120000"
    truth = make_run(folder, amplitude_uV=800.0)

    config = _config(session)
    conditions = {c.raw_name: c for c in discover_runs(session)}
    loaded = load_run(conditions["+0_5mA_260819_120000"], config)
    train = recover_pulses(loaded, config)

    check(train.n_pulses == 50, f"recovered 50 pulses (got {train.n_pulses})")
    check(abs(train.period_s - 0.259) < 0.005, f"period 259 ms (got {train.period_s * 1000:.1f})")
    check(train.ok, f"timing reported ok (issues: {train.issues})")
    if train.n_pulses == truth.size:
        error_ms = np.abs(train.onsets_s - truth) * 1000.0
        check(float(error_ms.max()) < 5.0, f"every onset within 5 ms (worst {error_ms.max():.2f})")
    check(loaded.healthy_channels == ("B-017", "B-018"), "both contacts pass the impedance test")


def test_recovery_survives_strong_rhythm(tmp: Path) -> None:
    print("recovery with a rhythm as large as the stimulus")
    session = tmp / "rhythm"
    folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_05mA_260819_120000"
    truth = make_run(folder, amplitude_uV=120.0, rhythm_uV=120.0, rhythm_hz=3.9, seed=11)

    config = _config(session)
    conditions = {c.raw_name: c for c in discover_runs(session)}
    train = recover_pulses(load_run(conditions["+0_05mA_260819_120000"], config), config)

    check(train.n_pulses == 50, f"50 pulses despite the rhythm (got {train.n_pulses})")
    if train.n_pulses == truth.size:
        error_ms = np.abs(train.onsets_s - truth) * 1000.0
        check(float(np.median(error_ms)) < 5.0, f"median onset error under 5 ms ({np.median(error_ms):.2f})")


def test_invisible_stimulus_is_flagged(tmp: Path) -> None:
    print("a stimulus too small to see declares itself")
    session = tmp / "invisible"
    folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_01mA_260819_120000"
    make_run(folder, amplitude_uV=0.0, rhythm_uV=80.0, noise_uV=30.0, seed=3)

    config = _config(session)
    conditions = {c.raw_name: c for c in discover_runs(session)}
    train = recover_pulses(load_run(conditions["+0_01mA_260819_120000"], config), config)
    check(not train.ok, "pure noise is not reported as a confident train")
    check(
        any("comb fit" in issue for issue in train.issues),
        f"the reason names the weak comb fit (issues: {train.issues})",
    )


def test_response_scales_and_artifact_separates(tmp: Path) -> None:
    print("dose-response and coupling-vs-response separation")
    session = tmp / "sweep"
    config = _config(session)

    amplitudes_uV = {0.1: 200.0, 0.2: 400.0, 0.5: 1000.0, 1.0: 2000.0}
    measured: list[float] = []
    currents: list[float] = []
    for current, amplitude in amplitudes_uV.items():
        name = f"+{str(current).replace('.', '_')}mA_260819_1200{int(current * 10):02d}"
        folder = session / "cfg stim 1 stim ground 2 recording ground 3" / name
        make_run(folder, amplitude_uV=amplitude, shape="coupling", seed=int(current * 100))

    conditions = discover_runs(session)
    for condition in conditions:
        loaded = load_run(condition, config)
        train = recover_pulses(loaded, config)
        evoked = evoked_deflection(loaded, train, config)
        if evoked and condition.amplitude_mA is not None:
            measured.append(evoked[0].pp_uV_median)
            currents.append(condition.amplitude_mA)

    check(len(measured) == 4, f"all four amplitudes measured (got {len(measured)})")
    order = np.argsort(currents)
    values = np.array(measured)[order]
    check(bool(np.all(np.diff(values) > 0)), "measured deflection increases with current")

    evidence = sweep_evidence("cfg", "B-017", np.array(currents), np.array(measured))
    check(evidence is not None, "sweep evidence produced")
    if evidence is not None:
        check(evidence.r_squared > 0.95, f"rectangular coupling is linear (r2={evidence.r_squared:.3f})")
        check(evidence.linear_through_origin, "rectangular coupling reads as through-origin linear")


def test_artifact_criteria_separate_shapes(tmp: Path) -> None:
    print("artifact criteria on known coupling and known response")
    session = tmp / "shapes"
    config = _config(session)

    coupling_folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_120000"
    response_folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_130000"
    make_run(coupling_folder, amplitude_uV=800.0, shape="coupling", seed=21)
    make_run(response_folder, amplitude_uV=800.0, shape="response", seed=22)

    results = {}
    for condition in discover_runs(session):
        loaded = load_run(condition, config)
        train = recover_pulses(loaded, config)
        evoked = evoked_deflection(loaded, train, config)
        if evoked:
            results[condition.run_folder.name] = run_evidence(
                evoked[0].channel, evoked[0].peak_latency_ms_median, evoked[0].post_pulse_fraction
            )

    coupling = results.get(coupling_folder.name)
    response = results.get(response_folder.name)
    check(coupling is not None and response is not None, "both shapes measured")
    if coupling and response:
        check(coupling.stops_with_pulse, "rectangular coupling reads as stopping with the pulse")
        check(not response.stops_with_pulse, "a decaying response reads as outlasting the pulse")
        check(
            coupling.suspicion > response.suspicion,
            f"coupling is more suspicious ({coupling.suspicion} vs {response.suspicion})",
        )


def test_peak_windows_and_detection(tmp: Path) -> None:
    print("per-peak analysis on a known multi-peak EP")
    session = tmp / "peaks"
    config = _config(session)

    ep_folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_140000"
    coupling_folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_150000"
    truths = {
        ep_folder.name: make_run(ep_folder, amplitude_uV=800.0, shape="ep", seed=41),
        coupling_folder.name: make_run(coupling_folder, amplitude_uV=800.0, shape="coupling", seed=42),
    }

    # Epoch on the true onsets: the peak measures are being tested here, not the
    # comb fit, whose own accuracy has its own tests above.
    evoked_by_name = {}
    for condition in discover_runs(session):
        loaded = load_run(condition, config)
        truth = truths[condition.run_folder.name]
        train = PulseTrain(
            onsets_s=truth,
            period_s=0.259,
            train_start_s=float(truth[0]),
            train_end_s=float(truth[-1] + 0.005),
            width_ms=5.0,
            jitter_ms=0.0,
            n_detected=truth.size,
            ok=True,
        )
        evoked = evoked_deflection(loaded, train, config)
        if evoked:
            evoked_by_name[condition.run_folder.name] = evoked[0]

    ep = evoked_by_name.get(ep_folder.name)
    coupling = evoked_by_name.get(coupling_folder.name)
    check(ep is not None and coupling is not None, "both synthetic runs measured")
    if ep is None or coupling is None:
        return

    # Window split: the rectangle plus onset spike dominates the during window,
    # the N/P components the post window, and the post latency must ignore the
    # onset spike entirely.
    check(ep.during_pp_uV_median > 700.0, f"during-pulse p-p sees the rectangle ({ep.during_pp_uV_median:.0f} uV)")
    check(
        250.0 < ep.post_pp_uV_median < 700.0,
        f"post-pulse p-p sees the components, not the rectangle ({ep.post_pp_uV_median:.0f} uV)",
    )
    check(
        abs(ep.post_peak_latency_ms_median - 15.0) < 1.5,
        f"post-window latency lands on the P component ({ep.post_peak_latency_ms_median:.2f} ms)",
    )
    check(np.isfinite(ep.baseline_sd_uV) and ep.baseline_sd_uV > 0, "single-trial baseline SD measured")

    # Detection with a deterministic measured off-edge: the 6.5 ms transient is
    # flagged, the two genuine components are not.
    # Four extrema live after the pulse: the off-edge transient, the rebound
    # bump between the two negative components, then the genuine N and P.
    found = analyse_channel_peaks(ep, FS, config, measured_width_ms=6.5)
    check(found.n_peaks == 4, f"four post-pulse peaks found (got {found.n_peaks})")
    if found.n_peaks == 4:
        edge, _bump, n_component, p_component = found.peaks
        check(edge.edge_suspect, "off-edge transient flagged edge_suspect")
        check(abs(edge.latency_ms - 6.5) < 1.0, f"off-edge transient at 6.5 ms (got {edge.latency_ms:.2f})")
        check(not n_component.edge_suspect and not p_component.edge_suspect, "genuine components not flagged")
        check(
            (edge.label, n_component.label, p_component.label) == ("N1", "N2", "P2"),
            f"labels by polarity and order (got {[p.label for p in found.peaks]})",
        )
        check(abs(n_component.latency_ms - 9.0) < 1.0, f"N component at 9 ms (got {n_component.latency_ms:.2f})")
        check(abs(p_component.latency_ms - 15.0) < 1.0, f"P component at 15 ms (got {p_component.latency_ms:.2f})")
        # The P tail reaches back under the N component (+17 uV at 9 ms), so
        # the composite trough is shallower than the -150 uV component alone.
        check(
            abs(n_component.amplitude_uV + 150.0) < 40.0,
            f"N amplitude near -150 uV (got {n_component.amplitude_uV:.1f})",
        )
        check(
            abs(p_component.amplitude_uV - 300.0) < 0.15 * 300.0,
            f"P amplitude within 15% (got {p_component.amplitude_uV:.1f})",
        )
        check(0.5 < n_component.width_ms < 5.0, f"N width sensible ({n_component.width_ms:.2f} ms)")
        check(2.0 < p_component.width_ms < 10.0, f"P width sensible ({p_component.width_ms:.2f} ms)")
        check(n_component.present and p_component.present, "both components pass the presence test")
        check(p_component.latency_jitter_ms < 2.0, f"P latency jitter small ({p_component.latency_jitter_ms:.2f} ms)")
        check(
            0.7 < p_component.adaptation_ratio < 1.3,
            f"no adaptation in a stationary train ({p_component.adaptation_ratio:.2f})",
        )

    evidence = run_evidence(
        ep.channel,
        ep.peak_latency_ms_median,
        ep.post_pulse_fraction,
        post_peak_latency_ms=ep.post_peak_latency_ms_median,
        during_pp_uV=ep.during_pp_uV_median,
        post_pp_uV=ep.post_pp_uV_median,
        n_post_peaks=found.n_peaks_clean,
        pulse_width_ms=config.pulse_width_ms,
    )
    check(evidence.post_response_detected, "EP run reports a detected post-pulse response")
    check(not evidence.fast_latency_post, "post-window latency is not pulse-locked")
    check(
        np.isfinite(evidence.coupling_ratio) and evidence.coupling_ratio > 1.0,
        f"during/post ratio finite and > 1 ({evidence.coupling_ratio:.2f})",
    )

    bare = analyse_channel_peaks(coupling, FS, config, measured_width_ms=5.0)
    check(bare.n_peaks_clean == 0, f"pure coupling yields no clean post-pulse peaks (got {bare.n_peaks_clean})")

    # Full pipeline regression: the legacy columns survive and the new ones appear.
    single = run_single(EvokedConfig(session_folder=ep_folder, single_run=True))
    check(bool(single.rows), "single-run pipeline produced rows")
    if single.rows:
        row = single.rows[0]
        legacy = (
            "run", "wiring", "channel", "amplitude_mA", "evoked_pp_uV", "evoked_pp_iqr_uV",
            "peak_latency_ms", "gap_baseline_pp_uV", "pre_train_pp_uV", "snr_vs_gap",
            "post_pulse_fraction", "artifact_fast_latency", "artifact_suspicion",
        )
        missing = [key for key in legacy if key not in row]
        check(not missing, f"legacy runs.csv columns intact (missing: {missing})")
        added = (
            "evoked_pp_during_uV", "evoked_pp_post_uV", "post_peak_latency_ms",
            "baseline_sd_uV", "n_peaks", "n_peaks_clean", "artifact_fast_latency_post",
            "artifact_coupling_ratio", "artifact_post_response_detected",
            "contact_position_um", "band_gap_minimum_hz",
            "decade_suspect", "decade_suspect_note",
            "timing_source", "clock_offset_s", "scope_capture", "scope_align_z",
            "scope_lead", "scope_phase1_s", "scope_ipd_s", "scope_phase2_s",
            "expected_pulses_run", "pulse_width_s_run", "interval_s_run",
        )
        missing_new = [key for key in added if key not in row]
        check(not missing_new, f"new runs.csv columns present (missing: {missing_new})")
        b017 = next((r for r in single.rows if r["channel"] == "B-017"), None)
        check(
            b017 is not None and b017["contact_position_um"] == 0.0,
            "B-017 sits at relative position 0",
        )
    check(bool(single.peak_rows), "single-run pipeline produced peak rows")
    if single.peak_rows:
        check(
            any(r["peak_label"].startswith("P") for r in single.peak_rows),
            "pipeline peak rows include a positive component",
        )
        check(
            "contact_position_um" in single.peak_rows[0],
            "peak rows carry the contact position",
        )


def test_band_power_excludes_the_comb(tmp: Path) -> None:
    print("band power sees injected gamma, not the pulse comb")
    session = tmp / "bands"
    config = _config(session)

    plain = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_2mA_260819_120000"
    boosted = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_2mA_260819_130000"
    make_run(plain, amplitude_uV=600.0, shape="coupling", seed=31)
    make_run(boosted, amplitude_uV=600.0, shape="coupling", gamma_boost_uV=150.0, seed=31)

    values = {}
    for condition in discover_runs(session):
        loaded = load_run(condition, config)
        train = recover_pulses(loaded, config)
        bands = band_power(loaded, train, config)
        if bands:
            values[condition.run_folder.name] = bands[0].band_db

    plain_bands = values.get(plain.name)
    boosted_bands = values.get(boosted.name)
    check(plain_bands is not None and boosted_bands is not None, "band power computed for both")
    if plain_bands and boosted_bands:
        check(
            boosted_bands["gamma"] > plain_bands["gamma"] + 3.0,
            f"injected 60 Hz raises gamma ({plain_bands['gamma']:.1f} -> {boosted_bands['gamma']:.1f} dB)",
        )
        check(
            abs(plain_bands["delta"]) < 6.0,
            f"delta stays near zero without a real delta change ({plain_bands['delta']:.1f} dB)",
        )


def test_decade_mislabel() -> None:
    print("decade-mislabel detection on a known bad label")
    amplitudes = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    clean_responses = 2000.0 * np.abs(amplitudes)
    names = tuple(f"run{i}" for i in range(amplitudes.size))

    points = decade_mislabel_evidence(amplitudes, clean_responses, names)
    check(not any(p.suspect for p in points), "clean linear sweep raises no suspects")

    # The -0_02 / -02 style mistake: the 0.02 run actually delivered 0.2 mA.
    bad = clean_responses.copy()
    bad[1] = 2000.0 * 0.2
    points = decade_mislabel_evidence(amplitudes, bad, names)
    suspects = [p for p in points if p.suspect]
    check(
        len(suspects) == 1 and suspects[0].run == "run1" and suspects[0].shift == 1,
        f"exactly the mislabelled point flagged with shift +1 "
        f"(got {[(p.run, p.shift) for p in suspects]})",
    )

    few = decade_mislabel_evidence(amplitudes[:3], bad[:3], names[:3])
    check(not any(p.suspect for p in few), "fewer than 4 points never flags")

    # Run-level aggregation: both channels flag the same run -> run flagged.
    rows = []
    for channel in ("B-017", "B-018"):
        for amplitude, response, name in zip(amplitudes, bad, names):
            rows.append(
                {
                    "run": name,
                    "wiring": "w",
                    "channel": channel,
                    "amplitude_mA": float(amplitude),
                    "evoked_pp_uV": float(response),
                }
            )
    _apply_decade_flags(rows)
    flagged = {r["run"] for r in rows if r["decade_suspect"]}
    check(flagged == {"run1"}, f"run-level flag set on the mislabelled run (got {flagged})")
    check(
        all("10x higher" in r["decade_suspect_note"] for r in rows if r["decade_suspect"]),
        "note names the direction",
    )
    clean_rows = [dict(r, evoked_pp_uV=2000.0 * abs(r["amplitude_mA"])) for r in rows]
    _apply_decade_flags(clean_rows)
    check(
        all(r["decade_suspect"] is False and r["decade_suspect_note"] == "" for r in clean_rows),
        "clean rows keep both columns present and unset",
    )


def test_gap_band_power_validity(tmp: Path) -> None:
    print("gap band power reports its validity cutoff")
    session = tmp / "gapbands"
    folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_2mA_260819_140000"
    make_run(
        folder,
        amplitude_uV=600.0,
        shape="coupling",
        period_s=1.2,
        n_pulses=20,
        duration_s=45.0,
        gamma_boost_uV=150.0,
        seed=33,
    )

    config = _config(session)
    condition = discover_runs(session)[0]
    loaded = load_run(condition, config)
    train = recover_pulses(loaded, config)
    bands = band_power(loaded, train, config)
    check(bool(bands), "band power computed for the long-period run")
    if bands:
        gap_duration_s = (train.period_s * 1000.0 - 5.0 - config.blank_ms) / 1000.0
        expected_minimum = 3.0 / gap_duration_s
        check(
            abs(bands[0].gap_minimum_hz - expected_minimum) < 0.05 * expected_minimum,
            f"gap_minimum_hz ~= 3/gap ({bands[0].gap_minimum_hz:.2f} vs {expected_minimum:.2f})",
        )
        check(
            np.isfinite(bands[0].band_db_gap["gamma"]),
            "gamma gap estimate finite at a 1.2 s period",
        )
        check(
            not np.isfinite(bands[0].band_db_gap["delta"]),
            "delta gap estimate still NaN at a 1.2 s period",
        )


def test_slow_protocol(tmp: Path) -> None:
    print("slow biphasic protocol: 0.5 s pulses at 5 s period")
    session = tmp / "slow"
    folder = session / "cfg stim 1 stim ground 2 recording ground 3" / "+0_05mA_260819_150000"
    truth = make_run(
        folder,
        amplitude_uV=800.0,
        shape="biphasic",
        period_s=5.0,
        n_pulses=10,
        pulse_width_ms=500.0,
        duration_s=70.0,
        seed=35,
    )

    # envelope_ms raised: 0.5 ms bins over 70 s make the quadratic
    # autocorrelation in the comb search painfully slow; 2 ms bins still
    # resolve the 200 ms phases easily.
    config = _config(
        session,
        expected_pulses=10,
        pulse_width_ms=500.0,
        max_period_s=6.0,
        response_window_ms=1000.0,
        envelope_ms=2.0,
    )
    condition = discover_runs(session)[0]
    loaded = load_run(condition, config)
    train = recover_pulses(loaded, config)
    check(train.n_pulses == 10, f"10 slow pulses recovered (got {train.n_pulses})")
    if train.n_pulses == 10:
        check(abs(train.period_s - 5.0) < 0.05, f"period 5 s (got {train.period_s:.3f})")
        error_ms = np.abs(train.onsets_s - truth) * 1000.0
        check(float(np.median(error_ms)) < 10.0, f"median onset error {np.median(error_ms):.1f} ms")

        evoked = evoked_deflection(loaded, train, config)
        check(bool(evoked), "evoked measured for the slow run")
        if evoked:
            first = evoked[0]
            check(
                first.during_pp_uV_median > 1200.0,
                f"during-pulse p-p sees the biphasic swing ({first.during_pp_uV_median:.0f} uV)",
            )
            check(
                first.post_pp_uV_median < first.during_pp_uV_median / 3.0,
                "post window is quiet next to the pulse",
            )
            check(
                np.isfinite(first.gap_baseline_pp_uV)
                and first.gap_baseline_pp_uV < first.pp_uV_median,
                "auto-relocated gap baseline is finite and below the evoked p-p",
            )

        bands = band_power(loaded, train, config)
        check(bool(bands), "band power computed for the slow run")
        if bands:
            check(
                bands[0].gap_minimum_hz < 1.0,
                f"gap valid below delta ({bands[0].gap_minimum_hz:.2f} Hz)",
            )
            check(
                np.isfinite(bands[0].band_db_gap["delta"]),
                "delta gap estimate finite at a 5 s period",
            )

    from .config import config_from_text_fields

    parsed = config_from_text_fields(
        session_folder=str(session),
        response_window_ms="1000",
        max_period_s="6",
        contact_pitch_um="500",
        contact_order="B-017, B-018",
    )
    check(
        parsed.response_window_ms == 1000.0
        and parsed.max_period_s == 6.0
        and parsed.contact_pitch_um == 500.0
        and parsed.contact_order == ("B-017", "B-018"),
        "config_from_text_fields round-trips the four new fields",
    )


def test_wiring_scope_and_filter(tmp: Path) -> None:
    print("wiring-folder sessions and the wiring filter")
    # Pointing "whole session" directly at ONE wiring-condition folder: the
    # flat runs and the nested-amplitude runs must all belong to that wiring.
    wcfg = tmp / "scope" / "stim 1 stim ground 2 recording ground 3"
    make_run(wcfg / "-0_01mA_260819_170000", amplitude_uV=200.0, n_pulses=1, duration_s=5.0, first_onset_s=2.0)
    make_run(wcfg / "-0_05mA" / "-0_260819_171000", amplitude_uV=200.0, n_pulses=1, duration_s=5.0, first_onset_s=2.0)

    conditions = discover_runs(wcfg)
    check(len(conditions) == 2, f"both runs found under the wiring folder (got {len(conditions)})")
    check(
        all(c.wiring.label == "stim 1 / gnd 2 / ref 3" for c in conditions),
        f"every run carries the folder's wiring (got {[c.wiring.label for c in conditions]})",
    )
    check(
        all("baseline_recording" not in c.flags for c in conditions),
        "no run misread as a session-root baseline",
    )
    nested = next(c for c in conditions if "amplitude_from_parent" in c.flags)
    check(
        nested.amplitude_mA is not None and abs(nested.amplitude_mA + 0.05) < 1e-9,
        "nested run still inherits its amplitude from the -0_05mA folder",
    )

    # The include/exclude filter, on a two-config session.
    session = tmp / "filter"
    make_run(
        session / "stim 1 stim ground 2 recording ground 3" / "-0_01mA_260819_170000",
        amplitude_uV=200.0, n_pulses=1, duration_s=5.0, first_onset_s=2.0,
    )
    make_run(
        session / "stim 1 recording ground 3 common ground - artifact" / "-0_01mA_260819_171500",
        amplitude_uV=200.0, n_pulses=1, duration_s=5.0, first_onset_s=2.0,
    )
    conditions = discover_runs(session)
    check(len(conditions) == 2, "two configs discovered")

    kept, note = _apply_wiring_filter(
        conditions, _config(session, wiring_exclude=("artifact",))
    )
    check(
        [c.wiring.label for c in kept] == ["stim 1 / gnd 2 / ref 3"] and note is not None,
        f"exclude=artifact drops the artifact config (kept {[c.wiring.label for c in kept]})",
    )
    kept, _note = _apply_wiring_filter(
        conditions, _config(session, wiring_include=("stim ground 2",))
    )
    check(
        [c.wiring.label for c in kept] == ["stim 1 / gnd 2 / ref 3"],
        "include filter keeps only the matching config",
    )
    kept, note = _apply_wiring_filter(conditions, _config(session))
    check(len(kept) == 2 and note is None, "no filter terms means no filtering")

    from .config import config_from_text_fields

    parsed = config_from_text_fields(
        session_folder=str(session),
        wiring_include="stim ground 2, stim 2",
        wiring_exclude="Artifact",
    )
    check(
        parsed.wiring_include == ("stim ground 2", "stim 2")
        and parsed.wiring_exclude == ("artifact",),
        "filter fields parse as lowercase comma-separated terms",
    )


def _write_fake_capture(
    path: Path, pulse_epochs, *, phase_s: float = 0.2, pause_s: float = 0.1,
    lead_sign: int = 1, seed: int = 9
) -> None:
    """A small scope capture: bursty timestamps, arbitrary probe scale."""
    rng = np.random.default_rng(seed)
    start = float(pulse_epochs[0]) - 20.0
    stop = float(pulse_epochs[-1]) + 8.0
    lines = ["timestamp time voltage1 current_mA1"]
    t = start
    envelope_s = 2.0 * phase_s + pause_s
    while t < stop:
        value = -19.5 + rng.normal(0.0, 0.2)
        for onset in pulse_epochs:
            offset = t - onset
            # Piecewise-linear RC-ish electrode voltage: ramp up under the
            # first current phase, sag slightly in the pause, ramp down
            # through the second phase, then a decaying discharge tail.
            if 0.0 <= offset < phase_s:
                value += 20.0 * lead_sign * (offset / phase_s)
                break
            if phase_s <= offset < phase_s + pause_s:
                value += lead_sign * (20.0 - 2.0 * (offset - phase_s) / pause_s)
                break
            if phase_s + pause_s <= offset < envelope_s:
                progress = (offset - phase_s - pause_s) / phase_s
                value += lead_sign * (18.0 - 38.0 * progress)
                break
            if envelope_s <= offset < envelope_s + 0.6:
                value += -20.0 * lead_sign * (1.0 - (offset - envelope_s) / 0.6)
                break
        lines.append(f"{t:.8f} {t - start:.8f} {value / 200.0:.8f} {value:.8f}")
        # Bursty grid: ~1 ms typical, occasional 30 ms dropouts.
        t += 0.03 if rng.random() < 0.005 else float(rng.lognormal(np.log(0.001), 0.4))
    path.write_text("\r\n".join(lines) + "\r\n")


def test_scope_sync(tmp: Path) -> None:
    print("oscilloscope timing: onset file + capture + clock-offset fit")
    from datetime import datetime

    from . import scope_sync
    from .pipeline import run_session

    session = tmp / "20990821 flat session"
    run_folder = session / "1_260821_120000"
    truth = make_run(
        run_folder,
        amplitude_uV=800.0,
        shape="biphasic",
        period_s=5.55,
        n_pulses=10,
        pulse_width_ms=500.0,
        first_onset_s=10.0,
        duration_s=80.0,
        seed=71,
    )

    start_epoch = datetime(2026, 8, 21, 12, 0, 0).timestamp()
    planted_offset = 1.7  # stim-host clock ahead of the Intan folder clock
    onset_epoch = start_epoch + float(truth[0]) + planted_offset
    protocol = "0.05mA_-0.05mA_pulsewidth0.5s_interval5.05s_pulsenumber10"
    (run_folder / f"{protocol}.onset.txt").write_text(
        f"pulse_onset_epoch_s={onset_epoch:.7f}\n"
        f"pulse_onset_local_time=2026-08-21 12:00:00.000000\n"
    )

    scope_dir = session / "oscilloscope"
    scope_dir.mkdir(parents=True, exist_ok=True)
    pulse_epochs = onset_epoch + (truth - truth[0])
    capture = scope_dir / f"{int(onset_epoch - 15)}.txt"
    _write_fake_capture(capture, pulse_epochs, lead_sign=-1)
    (scope_dir / f"{int(onset_epoch - 300)}.txt").write_text("")  # 0-byte decoy

    check(scope_sync.read_onset_epoch(run_folder) == float(f"{onset_epoch:.7f}"), "onset epoch read back")
    check(
        scope_sync.run_start_epoch(run_folder) == start_epoch,
        "run start epoch from the folder name",
    )
    span = scope_sync.capture_span(capture)
    check(span is not None and span[0] <= onset_epoch <= span[1], "capture span brackets the onset")
    found = scope_sync.find_scope_capture(scope_dir, onset_epoch)
    check(found == capture, f"capture found, decoy skipped (got {found and found.name})")

    pulses = scope_sync.read_scope_pulse_epochs(capture, onset_epoch, 10, 0.5, 5.05)
    check(pulses.epochs_s.size == 10, f"10 scope pulses detected (got {pulses.epochs_s.size})")
    if pulses.epochs_s.size == 10:
        edge_error_ms = np.abs(pulses.epochs_s - pulse_epochs) * 1000.0
        # A ramp base in noise is genuinely ~10 ms fuzzy; the amplifier
        # alignment absorbs the shared bias, so onsets still land within 5 ms.
        check(
            float(edge_error_ms.max()) < 15.0,
            f"scope edges within 15 ms (worst {edge_error_ms.max():.2f})",
        )
        check(abs(pulses.period_s - 5.55) < 0.05, f"measured period 5.55 s (got {pulses.period_s:.3f})")
        check(pulses.lead_sign == -1, f"leading phase sign from the slope (got {pulses.lead_sign})")
        check(
            abs(pulses.phase1_s - 0.2) < 0.05
            and abs(pulses.ipd_s - 0.1) < 0.06
            and abs(pulses.phase2_s - 0.2) < 0.06,
            f"phase segmentation ~0.2/0.1/0.2 s (got {pulses.phase1_s:.2f}/{pulses.ipd_s:.2f}/{pulses.phase2_s:.2f})",
        )
        check(abs(pulses.envelope_s - 0.5) < 0.08, f"envelope ~0.5 s (got {pulses.envelope_s:.2f})")

    # End-to-end through the session pipeline (scope dir auto-detected).
    result = run_session(_config(session, wiring_label="test wiring"), progress=lambda m: None)
    scoped = next((r for r in result.runs if r.condition.run_folder == run_folder), None)
    check(scoped is not None and scoped.ok, "scope-timed run analysed")
    if scoped is not None and scoped.train is not None:
        train = scoped.train
        check(train.source == "scope", f"timing source is scope (got {train.source})")
        check(train.n_pulses == 10, f"per-run pulsenumber overrides config (got {train.n_pulses})")
        if train.n_pulses == 10:
            onset_error_ms = np.abs(train.onsets_s - truth) * 1000.0
            check(
                float(onset_error_ms.max()) < 5.0,
                f"onsets within 5 ms of truth (worst {onset_error_ms.max():.2f})",
            )
        check(
            abs(train.clock_offset_s - planted_offset) < 0.02,
            f"planted clock offset recovered ({train.clock_offset_s:+.3f} vs +1.700)",
        )
        check(train.align_z >= 5.0, f"alignment confident (z={train.align_z:.1f})")
    rows = [r for r in result.rows if r["run"] == run_folder.name]
    check(
        bool(rows)
        and rows[0]["timing_source"] == "scope"
        and rows[0]["scope_capture"] == capture.name
        and rows[0]["wiring"] == "test wiring"
        and rows[0]["expected_pulses_run"] == 10,
        "rows carry the scope columns and the flat-session wiring label",
    )
    check(
        bool(rows) and abs(rows[0]["amplitude_mA"] + 0.05) < 1e-9 and rows[0]["polarity"] == "cathodic",
        "polarity corrected from the scope against the anodic-first label",
    )
    check(
        bool(rows)
        and "polarity_from_scope" in rows[0]["flags"]
        and rows[0]["scope_lead"] == "cathodic"
        and abs(rows[0]["scope_phase1_s"] - 0.2) < 0.05,
        "rows carry the scope waveform columns and the correction flag",
    )

    # Fallback: a protocol-named run without onset.txt uses the comb fit.
    fallback_folder = session / "1_260821_130000"
    make_run(
        fallback_folder,
        amplitude_uV=800.0,
        shape="biphasic",
        period_s=5.55,
        n_pulses=10,
        pulse_width_ms=500.0,
        first_onset_s=10.0,
        duration_s=80.0,
        seed=72,
    )
    (fallback_folder / f"{fallback_folder.name}.rhd").rename(fallback_folder / f"{protocol}.rhd")
    result = run_session(_config(session, wiring_label="test wiring"), progress=lambda m: None)
    fallback = next((r for r in result.runs if r.condition.run_folder == fallback_folder), None)
    check(fallback is not None and fallback.train is not None, "fallback run analysed")
    if fallback is not None and fallback.train is not None:
        check(fallback.train.source == "comb", f"fallback uses comb fit (got {fallback.train.source})")
        check(
            any("scope timing unavailable" in issue for issue in fallback.train.issues),
            f"fallback reason recorded (issues: {fallback.train.issues})",
        )
    baselines = [r for r in result.runs if "baseline_recording" in r.condition.flags]
    check(not baselines, "no protocol run misread as baseline")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="evoked_selftest_") as name:
        tmp = Path(name)
        test_naming()
        test_pulse_recovery(tmp)
        test_recovery_survives_strong_rhythm(tmp)
        test_invisible_stimulus_is_flagged(tmp)
        test_response_scales_and_artifact_separates(tmp)
        test_artifact_criteria_separate_shapes(tmp)
        test_peak_windows_and_detection(tmp)
        test_band_power_excludes_the_comb(tmp)
        test_decade_mislabel()
        test_gap_band_power_validity(tmp)
        test_slow_protocol(tmp)
        test_wiring_scope_and_filter(tmp)
        test_scope_sync(tmp)

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
