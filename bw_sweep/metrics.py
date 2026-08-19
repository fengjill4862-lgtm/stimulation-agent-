"""Per epoch x channel metrics for one sweep run.

Reuses, unchanged: ``load_run`` (data + events), ``validate_run`` (empirical
rail estimate, pulse counts), ``extract_epochs`` / ``baseline_stats``,
``compute_recovery`` (recovery time with the FIXED threshold config),
``mark_baseline_contamination``, ``rail_exit_ms`` and
``fit_exponential_tail`` from ``filter_diag.common``.

New here: the spec rail mask (|raw| >= 6389 uV), the 0-5 ms peak, the clean
pre-train noise floor, and the per-run time-constant columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from filter_diag.common import fit_exponential_tail, rail_exit_ms
from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import baseline_stats, extract_epochs, gap_starts, window_slice
from stim_analysis.load_rhs import hardware_floor_ms, load_run
from stim_analysis.recovery import compute_recovery, mark_baseline_contamination
from stim_analysis.validate import railed_mask, validate_run
from bw_sweep.config import SweepConfig
from bw_sweep.load import SweepRun

TRACE_DECIMATE = 10


@dataclass
class RunMetrics:
    run: SweepRun
    trials: pd.DataFrame  # one row per channel x kept epoch
    prestim_sd_uV: dict[str, float]
    prestim_seconds: float
    median_trace_uV: dict[str, np.ndarray]  # channel -> median centred trace over the trace window (decimated)
    trace_t_ms: np.ndarray
    floor_ms: float
    n_events: int
    n_kept: int
    compliance_flag: bool
    rail_levels: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def spec_rail_mask(raw_uV: np.ndarray, level_uV: float) -> np.ndarray:
    """Samples at the ADC rail: |raw| >= level (6388.9 uV -> codes <= 4 / >= 65532)."""
    return np.abs(np.asarray(raw_uV, dtype=np.float32)) >= np.float32(level_uV)


def local_centre(epochs2d: np.ndarray, t_ms: np.ndarray, window_ms: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per epoch: mean of the raw signal over the local pre-pulse window and its linear drift across it.

    Centring on a window that ends just before the pulse minimises the previous
    pulse's tail inside the reference level (with a 1 s IPI and long tau the
    spec's -500..-50 ms baseline sits on that tail).  The drift (slope x window
    length, uV) says how much the reference level was still moving.
    """
    sl = window_slice(t_ms, *window_ms)
    seg = np.asarray(epochs2d, dtype=np.float64)[:, sl]
    n = seg.shape[1]
    if n < 2:
        e = epochs2d.shape[0]
        return np.full(e, np.nan), np.full(e, np.nan)
    offset = seg.mean(axis=1)
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    slope_per_sample = (seg - offset[:, None]) @ x / float(np.sum(x * x))
    return offset, slope_per_sample * (n - 1)


def prestim_noise_sd(raw_uV: np.ndarray, sample_rate_hz: float, first_onset_sample: int, cfg: SweepConfig) -> tuple[float, float]:
    """(SD, seconds used) of the raw trace between the recording start and the first pulse."""
    start = int(round(cfg.prestim_skip_start_s * sample_rate_hz))
    end = int(first_onset_sample - round(cfg.prestim_gap_before_first_s * sample_rate_hz))
    minimum = int(round(cfg.prestim_min_s * sample_rate_hz))
    if end - start < minimum:
        # short lead-in: use the latest prestim_min_s before the gap (closest to the pulse, past the start-up transient)
        start = max(0, end - minimum)
    if end - start < minimum:
        return float("nan"), 0.0
    segment = np.asarray(raw_uV[start:end], dtype=np.float64)
    return float(segment.std(ddof=1)), (end - start) / sample_rate_hz


def per_run_metrics(run: SweepRun, cfg: SweepConfig, acfg: AnalysisConfig | None = None) -> RunMetrics:
    """Load one run and compute every per-epoch metric on every channel."""
    acfg = acfg or cfg.analysis_config()
    record = load_run(run.folder, acfg, None)
    warnings: list[str] = list(record.stim_channel_warnings)
    if record.data is None or record.events is None or record.events.n_events == 0:
        record.release_data()
        raise ValueError(f"{run.folder.name}: no stimulation events detected")
    validation = validate_run(record, acfg)
    events = record.events
    fs = record.sample_rate_hz
    epochs = extract_epochs(
        record.data.raw_uV, fs, events.onset_samples, events.event_numbers, acfg,
        run_id=run.run_id, channels=record.channels, gap_sample_starts=gap_starts(record.data.timestamps),
    )
    floor = hardware_floor_ms(events, record.settings, fs)
    onset_s_by_event = {int(n): float(s) for n, s in zip(events.event_numbers, events.onset_s)}
    rails = {rail.channel: rail for rail in validation.rails}
    kept = epochs.kept
    kept_numbers = epochs.event_numbers[kept]
    t_ms = epochs.t_ms
    core = epochs.core
    t_core = t_ms[core]
    post = window_slice(t_core, 0.0, float("inf"))
    post_abs = slice(core.start + post.start, core.start + post.stop)
    peak_win = window_slice(t_ms, 0.0, cfg.peak_window_ms)
    trace_win = window_slice(t_ms, cfg.trace_window_ms[0], cfg.trace_window_ms[1])
    trace_t = t_ms[trace_win][::TRACE_DECIMATE]
    first_onset = int(events.onset_samples.min())

    frames: list[pd.DataFrame] = []
    prestim: dict[str, float] = {}
    prestim_seconds = 0.0
    traces: dict[str, np.ndarray] = {}
    rail_levels: dict[str, tuple[float, float]] = {}
    for c_index, channel in enumerate(record.channels):
        ep = epochs.raw[c_index][kept]  # (E, S) float32, raw
        if ep.shape[0] == 0:
            warnings.append(f"{channel}: no kept epochs")
            continue
        mean, sd = baseline_stats(ep, t_ms, acfg.baseline_ms)  # spec window: baseline_sd_uV and the spec baseline mean
        offset, local_drift = local_centre(ep, t_ms, cfg.centre_window_ms)
        centred = ep.astype(np.float64) - offset[:, None]
        railed_spec = spec_rail_mask(ep, cfg.rail_level_uV)
        rail = rails.get(channel)
        if rail is not None and rail.is_railed:
            railed_emp = np.stack([railed_mask(row, rail, acfg, fs) for row in ep])
            rail_levels[channel] = (rail.neg_level_uV, rail.pos_level_uV)
        else:
            railed_emp = None
        frame = compute_recovery(centred, t_ms, fs, sd, acfg, railed=railed_emp, core=core, event_numbers=kept_numbers)
        if not np.allclose(frame["threshold_uV"].to_numpy(dtype=float), cfg.threshold_uV):
            raise AssertionError(f"{run.folder.name} {channel}: threshold is not fixed at {cfg.threshold_uV} uV")
        frame = frame.rename(columns={"rail_ms": "rail_emp_ms"})
        # secondary: the same recovery with the spec's -500..-50 ms baseline mean as the reference level
        spec_frame = compute_recovery(ep.astype(np.float64) - mean[:, None], t_ms, fs, sd, acfg, railed=railed_emp, core=core, event_numbers=kept_numbers)
        frame["recovery_spec_centred_ms"] = spec_frame["recovery_ms"].to_numpy()
        frame["censored_spec_centred"] = spec_frame["censored"].to_numpy()
        frame["rail_fs_ms"] = railed_spec[:, post_abs].sum(axis=1) * 1e3 / fs
        frame["peak_uV"] = np.abs(centred[:, peak_win]).max(axis=1) if peak_win.stop > peak_win.start else np.nan
        frame["peak_raw_uV"] = np.abs(ep[:, peak_win].astype(np.float64)).max(axis=1) if peak_win.stop > peak_win.start else np.nan
        railed_any = railed_spec if railed_emp is None else (railed_spec | railed_emp)
        exits, signs, was_railed = [], [], []
        fits = []
        for i in range(centred.shape[0]):
            exit_ms, sign, railed_flag = rail_exit_ms(centred[i], t_ms, railed_any[i], core)
            fit = fit_exponential_tail(
                centred[i], t_ms, exit_ms=exit_ms, excursion_sign=sign,
                start_offset_ms=cfg.fit_start_offset_ms, end_ms=cfg.fit_end_ms, tau_bounds_ms=cfg.fit_tau_bounds_ms,
            )
            exits.append(exit_ms)
            signs.append(sign)
            was_railed.append(railed_flag)
            fits.append(fit)
        frame["exit_ms"] = exits
        frame["excursion_sign"] = signs
        frame["was_railed"] = was_railed
        frame["A_uV"] = [f.A_uV for f in fits]
        frame["tau_fit_ms"] = [f.tau_ms for f in fits]
        frame["C_uV"] = [f.C_uV for f in fits]
        frame["r2"] = [f.r2 for f in fits]
        frame["fit_n_points"] = [f.n_points for f in fits]
        frame["fit_converged"] = [f.converged for f in fits]
        frame["tail_sign"] = [f.tail_sign for f in fits]
        frame["same_sign_tail"] = frame["tail_sign"] > 0
        frame = mark_baseline_contamination(frame, onset_s_by_event, acfg)
        frame.insert(0, "channel", channel)
        frame.insert(0, "run_id", run.run_id)
        frame["folder"] = run.folder.name
        frame["is_stim_contact"] = channel == record.stim_channel
        frame["impedance_kohm"] = run.impedance_kohm.get(channel, float("nan"))
        frame["baseline_mean_uV"] = mean  # spec window -500..-50 ms
        frame["centre_offset_uV"] = offset  # local window (centre_window_ms): what the epoch is centred on
        frame["baseline_drift_uV"] = offset - mean  # previous-tail decay between the two windows (signed)
        frame["local_drift_uV"] = local_drift  # linear drift of raw across the centre window
        frames.append(frame)
        prestim[channel], prestim_seconds = prestim_noise_sd(record.data.raw_uV[c_index], fs, first_onset, cfg)
        traces[channel] = np.median(centred[:, trace_win], axis=0)[::TRACE_DECIMATE]

    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not trials.empty:
        trials["arms"] = "+".join(run.arms)
        trials["lower_hz"] = run.lower_hz
        trials["dsp_enabled"] = run.dsp_enabled
        trials["dsp_hz"] = run.dsp_hz
        trials["upper_hz"] = run.upper_hz
        trials["fc_hz"] = run.fc_hz
        trials["tau_nominal_ms"] = run.tau_nominal_ms
        trials["tau_second_ms"] = run.tau_second_ms
        trials["fit_informative"] = run.tau_nominal_ms >= cfg.fit_informative_tau_ms
        trials["r2_below_min"] = trials["r2"] < cfg.r2_min
        trials["prestim_sd_uV"] = trials["channel"].map(prestim)
        trials["prestim_seconds"] = prestim_seconds
        trials["floor_ms"] = floor
        for arm in ("A", "B", "C"):
            trials[f"arm_{arm}_label"] = run.arm_label(arm)
    stim_channel = record.stim_channel
    if stim_channel != run.stim_channel and run.stim_channel is not None:
        warnings.append(f"stim channel from data {stim_channel} != settings {run.stim_channel}")
    record.release_data()
    return RunMetrics(
        run=run,
        trials=trials,
        prestim_sd_uV=prestim,
        prestim_seconds=prestim_seconds,
        median_trace_uV=traces,
        trace_t_ms=trace_t,
        floor_ms=floor,
        n_events=int(events.n_events),
        n_kept=int(np.count_nonzero(kept)),
        compliance_flag=bool(validation.compliance_flag),
        rail_levels=rail_levels,
        warnings=warnings,
    )


__all__ = ["RunMetrics", "TRACE_DECIMATE", "local_centre", "per_run_metrics", "prestim_noise_sd", "spec_rail_mask"]
