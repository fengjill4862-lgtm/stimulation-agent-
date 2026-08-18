"""Shared pieces for the filter diagnosis.

* Intan RHS DSP offset-removal high-pass: first-order IIR
      m[n] = m[n-1] + (x[n] - m[n-1]) * 2**-k,   y[n] = x[n] - m[n]
  with cutoff f_c = fs * ln(2**k / (2**k - 1)) / (2 pi) and time constant
  tau ~= 2**k / fs (k = 12 at 30 kHz -> 1.166 Hz, 136.5 ms), matching the
  ``actual_dsp_cutoff`` stored in the RHS header. Forward and exact inverse.
* Analog first-order high-pass (the 0.0945 Hz stage) for the synthetic chain.
* Session loading with the agreed exclusions (compliance-flagged runs, paired
  pulses, A-031; stim contacts flagged), epoching, and the unchanged
  ``compute_recovery`` from stim_analysis.
* Robust single-exponential fit of the post-rail tail.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import least_squares

from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import EpochSet, baseline_stats, extract_epochs, gap_starts, window_slice
from stim_analysis.load_rhs import RunRecord, contact_distance_um, discover_run_folders, hardware_floor_ms, load_run, parse_run_folder_name
from stim_analysis.recovery import compute_recovery
from stim_analysis.validate import RunValidation, assign_blocks, estimate_rail, railed_mask, validate_run

DEFAULT_LIVE = Path("/Users/jf/SynologyDrive/Research/Stimulation/20260817 re stim Noah 1")
DEFAULT_DEAD = Path("/Users/jf/SynologyDrive/Research/Stimulation/20260816 re stim Yun dead rat")
DEFAULT_OUTPUT = DEFAULT_LIVE / "filter_diagnosis"
EXCLUDED_CHANNELS = ("A-031",)
OPEN_CHANNEL_KOHM = 2000.0  # header impedance above this = disconnected contact, excluded
ADC_FULL_SCALE_UV = 0.195 * 32768.0


# -----------------------------------------------------------------------------
# Intan DSP high-pass model
# -----------------------------------------------------------------------------


def dsp_k_from_cutoff(sample_rate_hz: float, cutoff_hz: float) -> int:
    """Register value k whose cutoff fs*ln(2^k/(2^k-1))/(2pi) is closest to cutoff_hz."""
    best_k, best_err = 12, float("inf")
    for k in range(1, 17):
        fc = sample_rate_hz * math.log((2**k) / (2**k - 1)) / (2 * math.pi)
        err = abs(math.log(fc / cutoff_hz))
        if err < best_err:
            best_k, best_err = k, err
    return best_k


def dsp_cutoff_hz(sample_rate_hz: float, k: int) -> float:
    return sample_rate_hz * math.log((2**k) / (2**k - 1)) / (2 * math.pi)


def dsp_tau_s(sample_rate_hz: float, k: int) -> float:
    """Exact time constant of the pole (1 - 2^-k): -1 / (fs * ln(1 - 2^-k))."""
    a = 2.0**-k
    return -1.0 / (sample_rate_hz * math.log(1.0 - a))


def dsp_coefficients(k: int) -> tuple[np.ndarray, np.ndarray]:
    """(b, a) of Y/X = (1-a)(1 - z^-1) / (1 - (1-a) z^-1), a = 2^-k."""
    a = 2.0**-k
    r = 1.0 - a
    return np.array([r, -r]), np.array([1.0, -r])


def intan_dsp_highpass(x: np.ndarray, sample_rate_hz: float, *, k: int | None = None, cutoff_hz: float | None = None, initial_mean: float | None = None) -> np.ndarray:
    """Forward model of the on-chip DSP high-pass (offset removal)."""
    if k is None:
        if cutoff_hz is None:
            raise ValueError("give k or cutoff_hz")
        k = dsp_k_from_cutoff(sample_rate_hz, cutoff_hz)
    x = np.asarray(x, dtype=np.float64)
    b, a = dsp_coefficients(k)
    if initial_mean is None:
        initial_mean = float(x[0]) if x.size else 0.0
    # zi so that the running mean starts at initial_mean (steady state for a constant input)
    zi = signal.lfilter_zi(b, a) * initial_mean
    y, _ = signal.lfilter(b, a, x, zi=zi)
    return y


def intan_dsp_highpass_freeze(x: np.ndarray, sample_rate_hz: float, *, k: int, full_scale_uV: float = ADC_FULL_SCALE_UV, initial_mean: float | None = None) -> np.ndarray:
    """DSP high-pass whose running mean is FROZEN while the input sits at the ADC rail.

    Motivated by the recordings: rails are flat at the exact extreme code for
    8-31 ms, which a mean that keeps tracking during saturation cannot produce
    (it lifts the output off the rail within a sample). Implemented segment by
    segment with lfilter state carried across the non-saturated stretches.
    """
    x = np.asarray(x, dtype=np.float64)
    b, a = dsp_coefficients(k)
    m = float(x[0]) if initial_mean is None else float(initial_mean)
    y = np.empty_like(x)
    saturated = np.abs(x) >= full_scale_uV - 1e-9
    n = x.size
    i = 0
    zi = signal.lfilter_zi(b, a) * m
    while i < n:
        if saturated[i]:
            j = i
            while j < n and saturated[j]:
                j += 1
            y[i:j] = x[i:j] - m  # frozen mean; output clamps at the rail below
            i = j
            zi = signal.lfilter_zi(b, a) * m
        else:
            j = i
            while j < n and not saturated[j]:
                j += 1
            seg, zf = signal.lfilter(b, a, x[i:j], zi=zi)
            y[i:j] = seg
            # recover the running mean at the end of the segment: m = x - y
            m = float(x[j - 1] - seg[-1])
            zi = zf
            i = j
    return y


def intan_dsp_inverse(y: np.ndarray, sample_rate_hz: float, *, k: int, initial_mean: float = 0.0, freeze_at_full_scale: bool = False, full_scale_uV: float = ADC_FULL_SCALE_UV) -> np.ndarray:
    """Exact inverse: x[n] = y[n] + m[n], m[n] = m[n-1] + y[n] * a/(1-a).

    Restores the DC / slow drift the DSP removed (an integrator), so follow it
    with a gentle high-pass. Exact only where the recorded y was not clamped;
    with ``freeze_at_full_scale`` the mean is held while |y| is at the rail
    (the inverse of ``intan_dsp_highpass_freeze``).
    """
    a = 2.0**-k
    r = 1.0 - a
    y = np.asarray(y, dtype=np.float64)
    if freeze_at_full_scale:
        inc = np.where(np.abs(y) >= full_scale_uV - 1e-6, 0.0, y * (a / r))
        m = initial_mean + np.cumsum(inc)
    else:
        m = initial_mean + np.cumsum(y) * (a / r)
    return y + m


def gentle_highpass(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 0.1, order: int = 1) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def analog_highpass(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 0.0945) -> np.ndarray:
    """First-order analog HP (bilinear), causal -- the 0.09 Hz coupling stage."""
    b, a = signal.butter(1, cutoff_hz, btype="highpass", fs=sample_rate_hz)
    zi = signal.lfilter_zi(b, a) * float(np.asarray(x)[0])
    y, _ = signal.lfilter(b, a, np.asarray(x, dtype=np.float64), zi=zi)
    return y


def analog_lowpass(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 7600.0) -> np.ndarray:
    b, a = signal.butter(1, cutoff_hz, btype="lowpass", fs=sample_rate_hz)
    y, _ = signal.lfilter(b, a, np.asarray(x, dtype=np.float64), zi=signal.lfilter_zi(b, a) * float(np.asarray(x)[0]))
    return y


def clip_adc(x: np.ndarray, full_scale_uV: float = ADC_FULL_SCALE_UV) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), -full_scale_uV, full_scale_uV)


def verify_dsp_model(sample_rate_hz: float = 30000.0, k: int = 12) -> dict[str, float]:
    """Step-response check: fitted tau must equal 2^k/fs and the inverse must round-trip."""
    n = int(3.0 * sample_rate_hz)
    x = np.zeros(n)
    x[int(0.5 * sample_rate_hz):] = 1000.0
    y = intan_dsp_highpass(x, sample_rate_hz, k=k, initial_mean=0.0)
    t = np.arange(n) / sample_rate_hz - 0.5
    post = t > 0.001
    slope, intercept = np.polyfit(t[post], np.log(y[post] / 1000.0), 1)
    tau_fit = -1.0 / slope
    x_back = intan_dsp_inverse(y, sample_rate_hz, k=k, initial_mean=0.0)
    return {
        "k": k,
        "cutoff_hz": dsp_cutoff_hz(sample_rate_hz, k),
        "tau_ms_expected": dsp_tau_s(sample_rate_hz, k) * 1e3,
        "tau_ms_step_fit": tau_fit * 1e3,
        "inverse_max_abs_error_uV": float(np.max(np.abs(x_back - x))),
        "amplitude_at_step_uV": float(y[int(0.5 * sample_rate_hz)]),
    }


# -----------------------------------------------------------------------------
# Session loading with the agreed exclusions
# -----------------------------------------------------------------------------


@dataclass
class DiagRun:
    """One included run: metadata, its epochs (all analysis channels) and per-trial recovery."""

    session: str
    run_id: str
    folder: Path
    record: RunRecord
    validation: RunValidation
    channels: list[str]  # analysis channels (A-031 excluded), stim contact included but flagged
    stim_channel: str | None
    epochs: EpochSet
    trials: pd.DataFrame  # compute_recovery output + channel/run columns
    rails: dict[str, object]
    baseline_sd: dict[str, np.ndarray]
    baseline_mean: dict[str, np.ndarray]
    floor_ms: float

    @property
    def amplitude_uA(self) -> float:
        return float(self.validation.amplitude_uA_data)

    @property
    def phase_us(self) -> float:
        return float(self.validation.phase_us_data)

    def channel_epochs(self, channel: str, *, centred: bool = True) -> np.ndarray:
        idx = self.epochs.channels.index(channel)
        ep = self.epochs.raw[idx][self.epochs.kept].astype(np.float64)
        if centred:
            ep = ep - self.baseline_mean[channel][:, None]
        return ep

    def release(self) -> None:
        self.epochs.raw = np.zeros((0, 0, 0), dtype=np.float32)


@dataclass
class SessionSelection:
    session: str
    parent: Path
    included: list[RunValidation]
    excluded: list[RunValidation]
    all_validations: list[RunValidation]
    channels: list[str]
    records: dict[str, RunRecord]
    open_channels: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        inc = ", ".join(f"{v.run_id}({v.amplitude_uA_data:g}uA)" for v in self.included)
        exc = ", ".join(f"{v.run_id}[{v.exclusion_reason or v.block}]" for v in self.excluded)
        opens = ", ".join(f"{c} ({z / 1e3:.1f} MOhm)" for c, z in sorted(self.open_channels.items()))
        return (f"{self.session}: included {len(self.included)} [{inc}]; excluded {len(self.excluded)} [{exc}]; "
                f"channels {', '.join(self.channels)}" + (f"; open channels excluded: {opens}" if opens else ""))


def _is_single_pulse_ladder(v: RunValidation) -> bool:
    """Block 1 style: single pulses (>= 0.3 s apart), any amplitude, phase = ladder phase."""
    return v.status == "ok" and v.n_detected > 0 and v.block in ("block1", "block2") and (not np.isfinite(v.ipi_median_s) or v.ipi_median_s >= 0.3)


def select_session(parent: Path, session: str, cfg: AnalysisConfig, *, progress: Callable[[str], None] | None = None) -> SessionSelection:
    """Load + validate every run; keep clean single-pulse runs (Block 1/2), drop compliance/paired/no-stim."""
    validations: list[RunValidation] = []
    records: dict[str, RunRecord] = {}
    for folder in discover_run_folders(parent):
        if progress:
            progress(f"{session}: validate {folder.name}")
        try:
            record = load_run(folder, cfg, None)
            v = validate_run(record, cfg)
            record.release_data()
            records[record.run_id] = record
        except Exception as exc:
            meta = parse_run_folder_name(folder.name)
            v = RunValidation(run_id=meta.run_id, run_folder=str(folder), label=meta.label, status="error")
            v.error = f"{type(exc).__name__}: {exc}"
            v.included = False
            v.exclusion_reason = "error"
        validations.append(v)
    assign_blocks(validations)
    included, excluded = [], []
    for v in validations:
        keep = v.included and not v.compliance_flag and _is_single_pulse_ladder(v)
        (included if keep else excluded).append(v)
        if not keep and not v.exclusion_reason:
            v.exclusion_reason = "no_stim" if v.n_detected == 0 else ("paired_or_train" if not _is_single_pulse_ladder(v) else "excluded")
    channels: list[str] = []
    open_channels: dict[str, float] = {}
    for r in records.values():
        for c in r.channels:
            z_kohm = r.impedance_ohms.get(c, float("nan")) / 1e3
            if np.isfinite(z_kohm) and z_kohm > OPEN_CHANNEL_KOHM:
                open_channels[c] = max(open_channels.get(c, 0.0), z_kohm)
                continue
            if c not in channels and c not in EXCLUDED_CHANNELS:
                channels.append(c)
    channels = [c for c in channels if c not in open_channels]
    channels.sort(key=lambda c: int(c.split("-")[-1]))
    included.sort(key=lambda v: (v.amplitude_uA_data, v.run_id))
    selection = SessionSelection(session, parent, included, excluded, validations, channels, records)
    selection.open_channels = open_channels
    return selection


def load_diag_run(selection: SessionSelection, v: RunValidation, cfg: AnalysisConfig, *, keep_data: bool = False) -> DiagRun:
    """Reload one run, epoch it, compute per-trial recovery with the unchanged algorithm."""
    record = load_run(Path(v.run_folder), cfg, selection.channels)
    assert record.data is not None and record.events is not None
    events = record.events
    epochs = extract_epochs(record.data.raw_uV, record.sample_rate_hz, events.onset_samples, events.event_numbers, cfg, run_id=v.run_id, channels=record.channels, gap_sample_starts=gap_starts(record.data.timestamps))
    floor = hardware_floor_ms(events, record.settings, record.sample_rate_hz)
    rails = {rail.channel: rail for rail in v.rails}
    frames = []
    baseline_sd: dict[str, np.ndarray] = {}
    baseline_mean: dict[str, np.ndarray] = {}
    for c_index, channel in enumerate(record.channels):
        ep = epochs.raw[c_index][epochs.kept]
        if ep.shape[0] == 0:
            continue
        mean, sd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
        baseline_mean[channel] = mean
        baseline_sd[channel] = sd
        rail = rails.get(channel)
        railed = np.stack([railed_mask(row, rail, cfg, record.sample_rate_hz) for row in ep]) if (rail is not None and rail.is_railed) else None
        frame = compute_recovery(ep - mean[:, None], epochs.t_ms, record.sample_rate_hz, sd, cfg, railed=railed, core=epochs.core, event_numbers=epochs.event_numbers[epochs.kept])
        frame.insert(0, "channel", channel)
        frame.insert(0, "run_id", v.run_id)
        frame.insert(0, "session", selection.session)
        frame["is_stim_contact"] = channel == record.stim_channel
        frame["impedance_kohm"] = record.impedance_ohms.get(channel, float("nan")) / 1e3
        frame["distance_um"] = contact_distance_um(channel, record.stim_channel, cfg)
        frame["amplitude_uA"] = v.amplitude_uA_data
        frame["phase_us"] = v.phase_us_data
        frame["baseline_mean_uV"] = mean
        frames.append(frame)
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not keep_data:
        record.release_data()
    return DiagRun(selection.session, v.run_id, Path(v.run_folder), record, v, list(record.channels), record.stim_channel, epochs, trials, rails, baseline_sd, baseline_mean, floor)


# -----------------------------------------------------------------------------
# Tail fitting (Test B)
# -----------------------------------------------------------------------------


@dataclass
class TailFit:
    A_uV: float = float("nan")
    tau_ms: float = float("nan")
    C_uV: float = float("nan")
    r2: float = float("nan")
    n_points: int = 0
    exit_ms: float = float("nan")
    tail_sign: int = 0  # sign of the tail (A) relative to the rail excursion: +1 same, -1 opposite
    converged: bool = False


def _exp_model(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return p[0] * np.exp(-t / p[1]) + p[2]


def fit_exponential_tail(
    x: np.ndarray,
    t_ms: np.ndarray,
    *,
    exit_ms: float,
    excursion_sign: float,
    start_offset_ms: float = 2.0,
    end_ms: float = 800.0,
    tau_bounds_ms: tuple[float, float] = (2.0, 5000.0),
    decimate: int = 10,
) -> TailFit:
    """Robust (soft-L1) single-exponential fit V(t) = A exp(-t/tau) + C on [exit+2 ms, end]."""
    sl = window_slice(t_ms, exit_ms + start_offset_ms, end_ms)
    tt = t_ms[sl][::decimate] - (exit_ms + start_offset_ms)
    yy = np.asarray(x[sl][::decimate], dtype=np.float64)
    if tt.size < 30:
        return TailFit(exit_ms=exit_ms)
    a0 = float(yy[0] - yy[-1])
    p0 = np.array([a0 if a0 != 0 else 1.0, 150.0, float(yy[-1])])
    lo = [-1e6, tau_bounds_ms[0], -1e5]
    hi = [1e6, tau_bounds_ms[1], 1e5]
    scale = max(1.0, float(np.std(yy)))
    try:
        res = least_squares(lambda p: (_exp_model(p, tt) - yy) / scale, p0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=2000)
    except Exception:
        return TailFit(exit_ms=exit_ms)
    pred = _exp_model(res.x, tt)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    tail_sign = int(np.sign(res.x[0]) * np.sign(excursion_sign)) if excursion_sign else 0
    return TailFit(float(res.x[0]), float(res.x[1]), float(res.x[2]), r2, int(tt.size), exit_ms, tail_sign, bool(res.success))


def rail_exit_ms(x: np.ndarray, t_ms: np.ndarray, railed_row: np.ndarray | None, core: slice) -> tuple[float, float, bool]:
    """(exit time, excursion sign, was_railed): last railed sample after 0, else the |peak| time."""
    t_core = t_ms[core]
    xc = x[core]
    post = t_core > 0
    if railed_row is not None and np.any(railed_row[core] & post):
        idx = np.flatnonzero(railed_row[core] & post)[-1]
        return float(t_core[idx]), float(np.sign(xc[idx])), True
    idx = int(np.argmax(np.abs(xc[post])))
    t_post = t_core[post]
    return float(t_post[idx]), float(np.sign(xc[post][idx])), False


def bootstrap_median_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    from stim_analysis.stats import bootstrap_ci

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(v, np.median, n, rng)
    return float(np.median(v)), lo, hi


def config_for_diag() -> AnalysisConfig:
    """Same recovery parameters as Function 6 (spec section 4)."""
    return AnalysisConfig()


__all__ = [
    "ADC_FULL_SCALE_UV",
    "DEFAULT_DEAD",
    "DEFAULT_LIVE",
    "DEFAULT_OUTPUT",
    "DiagRun",
    "OPEN_CHANNEL_KOHM",
    "SessionSelection",
    "TailFit",
    "analog_highpass",
    "analog_lowpass",
    "bootstrap_median_ci",
    "clip_adc",
    "config_for_diag",
    "dsp_coefficients",
    "dsp_cutoff_hz",
    "dsp_k_from_cutoff",
    "dsp_tau_s",
    "fit_exponential_tail",
    "gentle_highpass",
    "intan_dsp_highpass",
    "intan_dsp_highpass_freeze",
    "intan_dsp_inverse",
    "load_diag_run",
    "rail_exit_ms",
    "select_session",
    "verify_dsp_model",
]
