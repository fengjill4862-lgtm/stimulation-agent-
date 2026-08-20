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
from .pulses import recover_pulses

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
    criteria have to tell these apart.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(duration_s * FS / SAMPLES_PER_BLOCK)
    n_samples = n_blocks * SAMPLES_PER_BLOCK
    time = np.arange(n_samples) / FS

    signal = rhythm_uV * np.sin(2 * np.pi * rhythm_hz * time)
    signal += rng.normal(0.0, noise_uV, size=n_samples)

    onsets = first_onset_s + np.arange(n_pulses) * period_s
    width = int(round(pulse_width_ms / 1000.0 * FS))

    for onset in onsets:
        start = int(round(onset * FS))
        if shape == "coupling":
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
