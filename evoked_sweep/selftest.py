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

from .artifact import run_evidence, sweep_evidence
from .config import EvokedConfig
from .load import load_run
from .metrics import band_power, evoked_deflection
from .naming import discover_runs, parse_amplitude, parse_wiring
from .peaks import analyse_channel_peaks
from .pipeline import run_single
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

    for onset in onsets:
        start = int(round(onset * FS))
        if shape == "ep":
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
        )
        missing_new = [key for key in added if key not in row]
        check(not missing_new, f"new runs.csv columns present (missing: {missing_new})")
    check(bool(single.peak_rows), "single-run pipeline produced peak rows")
    if single.peak_rows:
        check(
            any(r["peak_label"].startswith("P") for r in single.peak_rows),
            "pipeline peak rows include a positive component",
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
