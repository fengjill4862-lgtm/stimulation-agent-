"""Test C: synthetic step through the instrument chain, then the UNCHANGED recovery algorithm.

Chain (physical order): input -> analog HP (0.0945 Hz) -> analog LP -> ADC clip
(+/-6389.6 uV) -> Intan DSP high-pass (k from the header cutoff) -> 16-bit clamp.
The spec's order (DSP first, then clip) is available as ``order="spec"``.

Two input families:
  * "step": zero, then a step of amplitude V for duration d, then a small residual
  * "step_tail": the same step followed by an exponential input tail (tau_in) --
    an electrode-polarisation / analog-recovery stand-in that keeps its sign

The recovery time is computed by ``stim_analysis.recovery.compute_recovery``
imported unchanged, on epochs built on the same time axis as the real ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import baseline_stats
from stim_analysis.recovery import compute_recovery
from filter_diag.common import (
    ADC_FULL_SCALE_UV,
    analog_highpass,
    analog_lowpass,
    clip_adc,
    dsp_tau_s,
    intan_dsp_highpass,
    intan_dsp_highpass_freeze,
)


@dataclass(frozen=True)
class SyntheticEpoch:
    t_ms: np.ndarray
    core: slice
    trace: np.ndarray  # recorded-like output (E=1)
    stages: dict[str, np.ndarray] = field(default_factory=dict)


def epoch_axis(cfg: AnalysisConfig, sample_rate_hz: float) -> tuple[np.ndarray, slice]:
    """Same padded axis as stim_analysis.epoch.extract_epochs."""
    to_s = lambda ms: int(round(ms * 1e-3 * sample_rate_hz))  # noqa: E731
    pad = to_s(cfg.filter_pad_ms)
    c0, c1 = to_s(cfg.epoch_ms[0]), to_s(cfg.epoch_ms[1])
    start, end = c0 - pad, c1 + pad
    t_ms = (np.arange(end - start) + start) * 1e3 / sample_rate_hz
    return t_ms, slice(c0 - start, c1 - start)


def synthetic_input(
    t_ms: np.ndarray,
    *,
    amplitude_uV: float,
    duration_ms: float,
    residual_uV: float = 0.0,
    tail_amplitude_uV: float = 0.0,
    tail_tau_ms: float = 100.0,
    sign: float = -1.0,
) -> np.ndarray:
    """Step of ``amplitude_uV`` for ``duration_ms`` at t=0, then residual (+ optional exponential tail)."""
    x = np.zeros(t_ms.size)
    during = (t_ms >= 0.0) & (t_ms < duration_ms)
    after = t_ms >= duration_ms
    x[during] = sign * amplitude_uV
    x[after] = sign * residual_uV
    if tail_amplitude_uV:
        x[after] += sign * tail_amplitude_uV * np.exp(-(t_ms[after] - duration_ms) / tail_tau_ms)
    return x


RAIL_MODES = ("freeze", "track", "spec")


def instrument_chain(
    x: np.ndarray,
    sample_rate_hz: float,
    *,
    k: int = 12,
    rail_mode: str = "freeze",
    analog_hp_hz: float | None = 0.0945,
    analog_lp_hz: float | None = 7600.0,
    full_scale_uV: float = ADC_FULL_SCALE_UV,
    order: str | None = None,
) -> dict[str, np.ndarray]:
    """Return every stage; 'recorded' is what the file would contain.

    rail_mode -- how the DSP behaves when the ADC saturates:
      "freeze": analog HP/LP -> ADC clip -> DSP with the running mean FROZEN while
                the input is at the rail (reproduces the flat 8-31 ms rails seen
                in the recordings) -> clamp
      "track":  same chain but the running mean keeps updating during the rail
                (the output lifts off the rail within a sample)
      "spec":   DSP applied to the un-clipped input, then clip (the spec's order)
    """
    if order is not None:  # backwards compatibility with the first draft
        rail_mode = "spec" if order == "spec" else "track"
    stages: dict[str, np.ndarray] = {"input": np.asarray(x, dtype=np.float64)}
    if rail_mode in ("freeze", "track"):
        y = stages["input"]
        if analog_hp_hz:
            y = analog_highpass(y, sample_rate_hz, analog_hp_hz)
        if analog_lp_hz:
            y = analog_lowpass(y, sample_rate_hz, analog_lp_hz)
        stages["analog"] = y
        y = clip_adc(y, full_scale_uV)
        stages["adc"] = y
        if rail_mode == "freeze":
            y = intan_dsp_highpass_freeze(y, sample_rate_hz, k=k, full_scale_uV=full_scale_uV, initial_mean=float(y[0]))
        else:
            y = intan_dsp_highpass(y, sample_rate_hz, k=k, initial_mean=float(y[0]))
        stages["dsp"] = y
        y = clip_adc(y, full_scale_uV)  # 16-bit output clamp
        stages["recorded"] = y
    elif rail_mode == "spec":
        y = intan_dsp_highpass(stages["input"], sample_rate_hz, k=k, initial_mean=float(stages["input"][0]))
        stages["dsp"] = y
        y = clip_adc(y, full_scale_uV)
        stages["recorded"] = y
    else:
        raise ValueError(rail_mode)
    return stages


def synthetic_recovery(
    cfg: AnalysisConfig,
    sample_rate_hz: float,
    *,
    k: int,
    amplitude_uV: float,
    duration_ms: float,
    noise_sd_uV: float,
    n_trials: int = 20,
    rng: np.random.Generator | None = None,
    rail_mode: str = "freeze",
    residual_uV: float = 0.0,
    tail_amplitude_uV: float = 0.0,
    tail_tau_ms: float = 100.0,
    analog_hp_hz: float | None = 0.0945,
    return_traces: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, slice]:
    """n_trials synthetic epochs (independent noise) -> compute_recovery rows."""
    rng = rng if rng is not None else np.random.default_rng(0)
    t_ms, core = epoch_axis(cfg, sample_rate_hz)
    x = synthetic_input(t_ms, amplitude_uV=amplitude_uV, duration_ms=duration_ms, residual_uV=residual_uV, tail_amplitude_uV=tail_amplitude_uV, tail_tau_ms=tail_tau_ms)
    epochs = np.empty((n_trials, t_ms.size))
    for i in range(n_trials):
        noisy = x + rng.normal(0.0, noise_sd_uV, size=t_ms.size)
        epochs[i] = instrument_chain(noisy, sample_rate_hz, k=k, rail_mode=rail_mode, analog_hp_hz=analog_hp_hz)["recorded"]
    mean, sd = baseline_stats(epochs, t_ms, cfg.baseline_ms)
    centred = epochs - mean[:, None]
    rows = compute_recovery(centred, t_ms, sample_rate_hz, sd, cfg, core=core)
    rows["amplitude_uV"] = amplitude_uV
    rows["duration_ms"] = duration_ms
    rows["noise_sd_uV"] = noise_sd_uV
    rows["rail_mode"] = rail_mode
    rows["tail_amplitude_uV"] = tail_amplitude_uV
    rows["tail_tau_ms"] = tail_tau_ms
    rows["k"] = k
    rows["sample_rate_hz"] = sample_rate_hz
    rows["tau_dsp_ms"] = dsp_tau_s(sample_rate_hz, k) * 1e3
    rows["clipped_input"] = bool(amplitude_uV > ADC_FULL_SCALE_UV)
    return rows, (centred if return_traces else np.zeros((0, 0))), t_ms, core


def run_test_c(
    cfg: AnalysisConfig,
    *,
    sample_rates: tuple[float, ...] = (30000.0, 20000.0),
    k_by_rate: dict[float, int] | None = None,
    durations_ms: tuple[float, ...] = (0.2, 1.0, 5.0, 10.0, 30.0),
    amplitude_multipliers: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0),
    base_amplitude_uV: float = 20000.0,
    noise_levels_uV: tuple[float, ...] = (10.0, 50.0, 150.0),
    tail_taus_ms: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0, 500.0, 1700.0, 1.0e6),
    tail_amplitude_uV: float = 3000.0,
    n_trials: int = 20,
    seed: int = 0,
    progress=None,
) -> dict[str, object]:
    """All synthetic sweeps. Returns tables and example traces for the figures."""
    rng = np.random.default_rng(seed)
    k_by_rate = k_by_rate or {30000.0: 12, 20000.0: 11}
    rows: list[pd.DataFrame] = []
    examples: dict[str, tuple[np.ndarray, np.ndarray, slice]] = {}
    for fs in sample_rates:
        k = k_by_rate.get(fs, 12)
        # sub-rail amplitude sweep (log law) + supra-rail (plateau): multipliers of 400 uV up to base*25
        amps = sorted({400.0 * m for m in (1, 2, 5, 10)} | {base_amplitude_uV * m for m in amplitude_multipliers})
        for rail_mode in RAIL_MODES:
            for d in durations_ms:
                for amp in amps:
                    for noise in noise_levels_uV:
                        if progress:
                            progress(f"test C: fs {fs:g} {rail_mode} d {d:g} ms amp {amp:g} uV noise {noise:g}")
                        r, tr, t_ms, core = synthetic_recovery(cfg, fs, k=k, amplitude_uV=amp, duration_ms=d, noise_sd_uV=noise, n_trials=n_trials, rng=rng, rail_mode=rail_mode, return_traces=(rail_mode == "freeze" and noise == noise_levels_uV[0] and amp == base_amplitude_uV))
                        r["family"] = "step"
                        rows.append(r)
                        if tr.size:
                            examples[f"step_fs{fs:g}_d{d:g}"] = (tr[:3], t_ms, core)
        # step + same-sign exponential input tail (electrode / analog recovery stand-in), freeze mode
        for tail_tau in tail_taus_ms:
            for d in (1.0, 10.0):
                for noise in noise_levels_uV[:2]:
                    r, tr, t_ms, core = synthetic_recovery(cfg, fs, k=k, amplitude_uV=base_amplitude_uV, duration_ms=d, noise_sd_uV=noise, n_trials=n_trials, rng=rng, rail_mode="freeze", tail_amplitude_uV=tail_amplitude_uV, tail_tau_ms=tail_tau, return_traces=(noise == noise_levels_uV[0] and d == 10.0))
                    r["family"] = "step_tail"
                    rows.append(r)
                    if tr.size:
                        examples[f"tail_fs{fs:g}_tau{tail_tau:g}_d{d:g}"] = (tr[:3], t_ms, core)
    table = pd.concat(rows, ignore_index=True)
    return {"table": table, "examples": examples}


__all__ = ["RAIL_MODES", "SyntheticEpoch", "epoch_axis", "instrument_chain", "run_test_c", "synthetic_input", "synthetic_recovery"]
