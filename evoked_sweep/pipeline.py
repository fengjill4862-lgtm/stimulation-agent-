#!/usr/bin/env python3
"""Orchestration for Function 7.

``run_session`` computes and returns; it never writes. ``render_outputs`` turns
a result into payloads, and ``write_outputs`` puts them on disk atomically.
Output paths are decided here rather than in widget code, so the UI can stay a
thin display layer and the headless CLI produces identical files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from rhs_files import atomic_write_all

from . import artifact as artifact_module
from .config import EvokedConfig
from .load import LoadedRun, load_run, run_size_bytes
from .metrics import (
    ChannelBands,
    ChannelEvoked,
    ChannelPostTrain,
    band_power,
    evoked_deflection,
    post_train_change,
)
from .naming import RunCondition, contact_positions_um, discover_runs
from .peaks import ChannelPeaks, analyse_channel_peaks
from .pulses import PulseTrain, recover_pulses
from .scope_sync import scope_train_for_run

ProgressCallback = Callable[[str], None]


@dataclass
class RunResult:
    """Everything computed for one run folder."""

    condition: RunCondition
    train: PulseTrain | None = None
    evoked: list[ChannelEvoked] = field(default_factory=list)
    peaks: list[ChannelPeaks] = field(default_factory=list)
    bands: list[ChannelBands] = field(default_factory=list)
    post_train: list[ChannelPostTrain] = field(default_factory=list)
    evidence: list[artifact_module.RunArtifactEvidence] = field(default_factory=list)
    healthy_channels: tuple[str, ...] = ()
    impedance_ohms: dict[str, float] = field(default_factory=dict)
    duration_s: float = float("nan")
    sample_rate_hz: float = float("nan")
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.train is not None and self.train.n_pulses > 0


@dataclass
class SessionResult:
    """Every run in a session, plus the across-amplitude evidence."""

    config: EvokedConfig
    runs: list[RunResult] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    peak_rows: list[dict] = field(default_factory=list)
    sweep: list[artifact_module.SweepArtifactEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def analysed(self) -> list[RunResult]:
        return [r for r in self.runs if r.ok]

    @property
    def failed(self) -> list[RunResult]:
        return [r for r in self.runs if not r.ok]


def _effective_config(condition: RunCondition, config: EvokedConfig) -> EvokedConfig:
    """Per-run protocol overrides from a protocol-named run, else the config.

    The 2x period headroom covers the Keithley's serial overhead (5.55 s
    measured against a 4.8 s label on the 20260821 session).
    """
    if condition.expected_pulses_run is None:
        return config
    width_s = condition.pulse_width_s_run or config.pulse_width_ms / 1000.0
    interval_s = condition.interval_s_run or 0.0
    # Wide pulses tolerate a coarser envelope, and the comb fallback's
    # quadratic autocorrelation needs one to stay affordable on long periods.
    envelope_ms = max(config.envelope_ms, 2.0) if width_s >= 0.05 else config.envelope_ms
    return replace(
        config,
        expected_pulses=condition.expected_pulses_run,
        pulse_width_ms=width_s * 1000.0,
        max_period_s=max(config.max_period_s, 2.0 * (interval_s + width_s)),
        envelope_ms=envelope_ms,
    )


def _resolve_scope_dir(config: EvokedConfig) -> Path | None:
    """Explicit scope folder, else <session>/oscilloscope, else the parent's."""
    if config.scope_dir is not None:
        return config.scope_dir
    candidates = [config.session_folder / "oscilloscope"]
    if config.single_run:
        candidates.append(config.session_folder.parent / "oscilloscope")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def analyse_run(
    condition: RunCondition,
    config: EvokedConfig,
    prior_periods: tuple[float, ...] = (),
    scope_dir: Path | None = None,
) -> RunResult:
    """Load and analyse one run folder. Never raises; failures land in `error`."""
    result = RunResult(condition=condition)
    try:
        loaded: LoadedRun = load_run(condition, config)
    except Exception as exc:  # noqa: BLE001 - a bad run must not stop the sweep
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.healthy_channels = loaded.healthy_channels
    result.impedance_ohms = loaded.impedance_ohms
    result.duration_s = loaded.duration_s
    result.sample_rate_hz = loaded.sample_rate_hz

    effective = _effective_config(condition, config)
    try:
        train = None
        if scope_dir is not None:
            train, reason = scope_train_for_run(loaded, effective, scope_dir)
        if train is None:
            train = recover_pulses(loaded, effective, prior_periods)
            if scope_dir is not None:
                train = replace(
                    train,
                    issues=train.issues + (f"scope timing unavailable ({reason}); comb fit used",),
                )
        result.train = train
        if train.n_pulses == 0:
            result.error = "; ".join(train.issues) or "no pulses recovered"
            return result

        result.evoked = evoked_deflection(loaded, train, effective)
        result.peaks = [
            analyse_channel_peaks(e, loaded.sample_rate_hz, effective, train.width_ms)
            for e in result.evoked
        ]
        result.bands = band_power(loaded, train, effective)
        result.post_train = post_train_change(loaded, train, effective)
        peaks_by_channel = {p.channel: p for p in result.peaks}
        result.evidence = [
            artifact_module.run_evidence(
                e.channel,
                e.peak_latency_ms_median,
                e.post_pulse_fraction,
                post_peak_latency_ms=e.post_peak_latency_ms_median,
                during_pp_uV=e.during_pp_uV_median,
                post_pp_uV=e.post_pp_uV_median,
                n_post_peaks=(
                    peaks_by_channel[e.channel].n_peaks_clean
                    if e.channel in peaks_by_channel
                    else 0
                ),
                pulse_width_ms=effective.pulse_width_ms,
            )
            for e in result.evoked
        ]
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _rows_for(
    result: RunResult, config: EvokedConfig, positions: dict[str, float] | None = None
) -> list[dict]:
    """Flatten one run into one row per channel."""
    condition = result.condition
    train = result.train
    bands_by_channel = {b.channel: b for b in result.bands}
    post_by_channel = {p.channel: p for p in result.post_train}
    evidence_by_channel = {e.channel: e for e in result.evidence}
    peaks_by_channel = {p.channel: p for p in result.peaks}

    rows: list[dict] = []
    for evoked in result.evoked:
        bands = bands_by_channel.get(evoked.channel)
        post = post_by_channel.get(evoked.channel)
        evidence = evidence_by_channel.get(evoked.channel)
        peaks = peaks_by_channel.get(evoked.channel)

        row = {
            "run": condition.raw_name,
            "wiring": condition.wiring.label,
            "wiring_raw": condition.wiring.raw_name,
            "channel": evoked.channel,
            "amplitude_mA": condition.amplitude_mA,
            "abs_amplitude_mA": condition.abs_amplitude_mA,
            "polarity": condition.polarity,
            "replicate": condition.replicate,
            "impedance_ohms": result.impedance_ohms.get(evoked.channel, float("nan")),
            "duration_s": result.duration_s,
            "n_pulses": train.n_pulses if train else 0,
            "pulse_period_ms": train.period_s * 1000.0 if train else float("nan"),
            "pulse_rate_hz": train.rate_hz if train else float("nan"),
            "pulse_width_ms": train.width_ms if train else float("nan"),
            "pulse_jitter_ms": train.jitter_ms if train else float("nan"),
            "train_start_s": train.train_start_s if train else float("nan"),
            "train_end_s": train.train_end_s if train else float("nan"),
            "timing_ok": bool(train.ok) if train else False,
            "timing_issues": "; ".join(train.issues) if train else "",
            "evoked_pp_uV": evoked.pp_uV_median,
            "evoked_pp_iqr_uV": evoked.pp_uV_iqr,
            "peak_latency_ms": evoked.peak_latency_ms_median,
            "gap_baseline_pp_uV": evoked.gap_baseline_pp_uV,
            "pre_train_pp_uV": evoked.pre_train_pp_uV,
            "snr_vs_gap": evoked.snr_vs_gap,
            "post_pulse_fraction": evoked.post_pulse_fraction,
            "flags": ",".join(condition.flags),
        }
        for name, _low, _high in config.bands:
            row[f"band_{name}_dB"] = bands.band_db.get(name, float("nan")) if bands else float("nan")
            row[f"band_{name}_dB_gap"] = (
                bands.band_db_gap.get(name, float("nan")) if bands else float("nan")
            )
        if post is not None:
            row["post_available"] = post.available
            row["post_slow_level_delta_uV"] = post.slow_level_delta_uV
            row["post_rhythm_hz_pre"] = post.rhythm_hz_pre
            row["post_rhythm_hz_post"] = post.rhythm_hz_post
            row["post_broadband_sd_dB"] = post.broadband_sd_db
        if evidence is not None:
            row["artifact_fast_latency"] = evidence.fast_latency
            row["artifact_stops_with_pulse"] = evidence.stops_with_pulse
            row["artifact_suspicion"] = evidence.suspicion
            row["artifact_reasons"] = "; ".join(evidence.reasons)
        # Window-split and per-peak summary columns, appended after the legacy
        # set so existing runs.csv consumers keep their column meanings.
        row["evoked_pp_during_uV"] = evoked.during_pp_uV_median
        row["evoked_pp_post_uV"] = evoked.post_pp_uV_median
        row["evoked_pp_post_iqr_uV"] = evoked.post_pp_uV_iqr
        row["post_peak_latency_ms"] = evoked.post_peak_latency_ms_median
        row["baseline_sd_uV"] = evoked.baseline_sd_uV
        row["n_peaks"] = peaks.n_peaks if peaks else 0
        row["n_peaks_clean"] = peaks.n_peaks_clean if peaks else 0
        if evidence is not None:
            row["artifact_fast_latency_post"] = evidence.fast_latency_post
            row["artifact_coupling_ratio"] = evidence.coupling_ratio
            row["artifact_post_response_detected"] = evidence.post_response_detected
        row["contact_position_um"] = (positions or {}).get(evoked.channel, float("nan"))
        row["band_gap_minimum_hz"] = bands.gap_minimum_hz if bands else float("nan")
        row["timing_source"] = train.source if train else ""
        row["clock_offset_s"] = train.clock_offset_s if train else float("nan")
        row["scope_capture"] = train.scope_capture if train else ""
        row["scope_align_z"] = train.align_z if train else float("nan")
        row["expected_pulses_run"] = condition.expected_pulses_run
        row["pulse_width_s_run"] = condition.pulse_width_s_run
        row["interval_s_run"] = condition.interval_s_run
        rows.append(row)

    return rows


def _peak_rows_for(result: RunResult, positions: dict[str, float] | None = None) -> list[dict]:
    """Flatten one run into one row per channel and detected peak (long format)."""
    condition = result.condition
    rows: list[dict] = []
    for channel_peaks in result.peaks:
        for peak in channel_peaks.peaks:
            rows.append(
                {
                    "run": condition.raw_name,
                    "wiring": condition.wiring.label,
                    "channel": channel_peaks.channel,
                    "amplitude_mA": condition.amplitude_mA,
                    "abs_amplitude_mA": condition.abs_amplitude_mA,
                    "polarity": condition.polarity,
                    "peak_label": peak.label,
                    "peak_polarity": peak.polarity,
                    "latency_ms": peak.latency_ms,
                    "latency_from_offset_ms": peak.latency_from_offset_ms,
                    "edge_suspect": peak.edge_suspect,
                    "amplitude_uV": peak.amplitude_uV,
                    "width_ms": peak.width_ms,
                    "prominence_uV": peak.prominence_uV,
                    "amp_median_uV": peak.amp_median_uV,
                    "amp_iqr_uV": peak.amp_iqr_uV,
                    "latency_jitter_ms": peak.latency_jitter_ms,
                    "present": peak.present,
                    "adaptation_ratio": peak.adaptation_ratio,
                    "amp_slope_uV_per_pulse": peak.amp_slope_uV_per_pulse,
                    "contact_position_um": (positions or {}).get(
                        channel_peaks.channel, float("nan")
                    ),
                }
            )
    return rows


def _sweep_evidence(rows: list[dict]) -> list[artifact_module.SweepArtifactEvidence]:
    """Fit response against current for every wiring configuration and channel."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("amplitude_mA") is None:
            continue
        grouped.setdefault((row["wiring"], row["channel"]), []).append(row)

    out: list[artifact_module.SweepArtifactEvidence] = []
    for (wiring, channel), group in sorted(grouped.items()):
        amplitudes = np.array([r["amplitude_mA"] for r in group], dtype=np.float64)
        responses = np.array([r["evoked_pp_uV"] for r in group], dtype=np.float64)
        evidence = artifact_module.sweep_evidence(wiring, channel, amplitudes, responses)
        if evidence is not None:
            out.append(evidence)
    return out


def _apply_wiring_filter(
    conditions: list[RunCondition], config: EvokedConfig
) -> tuple[list[RunCondition], str | None]:
    """Keep runs whose wiring folder name passes the include/exclude terms.

    Terms are case-insensitive substrings of the config folder name. Include
    (when given) is applied first, then exclude -- so "artifact" in the
    exclude field drops the configs the user marked as artifact-generating.
    Returns (kept runs, a note describing what was dropped, or None).
    """
    if not config.wiring_include and not config.wiring_exclude:
        return conditions, None

    def _keep(condition: RunCondition) -> bool:
        name = condition.wiring.raw_name.lower()
        if config.wiring_include and not any(term in name for term in config.wiring_include):
            return False
        return not any(term in name for term in config.wiring_exclude)

    kept = [c for c in conditions if _keep(c)]
    dropped = [c for c in conditions if not _keep(c)]
    if not dropped:
        return kept, None
    dropped_configs = sorted({c.wiring.raw_name for c in dropped})
    parts = []
    if config.wiring_include:
        parts.append(f"include={', '.join(config.wiring_include)}")
    if config.wiring_exclude:
        parts.append(f"exclude={', '.join(config.wiring_exclude)}")
    note = (
        f"{len(dropped)} run(s) skipped by the wiring filter ({'; '.join(parts)}): "
        + "; ".join(dropped_configs)
    )
    return kept, note


def _apply_decade_flags(rows: list[dict]) -> None:
    """Flag runs whose response fits a current 10x off its label. Mutates rows.

    Detection is per (wiring, channel); a mislabel is a property of the run's
    folder name, so a run is flagged only when at least half of its tested
    channels agree on the same decade shift. Flag only, never correct.
    """
    for row in rows:
        row["decade_suspect"] = False
        row["decade_suspect_note"] = ""

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("amplitude_mA") is None:
            continue
        grouped.setdefault((row["wiring"], row["channel"]), []).append(row)

    votes: dict[str, dict[int, set[str]]] = {}
    tested: dict[str, set[str]] = {}
    for (_wiring, channel), group in grouped.items():
        amplitudes = np.array([r["amplitude_mA"] for r in group], dtype=np.float64)
        responses = np.array([r["evoked_pp_uV"] for r in group], dtype=np.float64)
        names = tuple(str(r["run"]) for r in group)
        points = artifact_module.decade_mislabel_evidence(amplitudes, responses, names)
        if len(points) < len(group):
            continue
        finite = any(np.isfinite(p.log_residual) for p in points)
        for point in points:
            if finite:
                tested.setdefault(point.run, set()).add(channel)
            if point.suspect:
                votes.setdefault(point.run, {}).setdefault(point.shift, set()).add(channel)

    for run_name, shifts in votes.items():
        shift, channels = max(shifts.items(), key=lambda item: len(item[1]))
        if 2 * len(channels) < len(tested.get(run_name, channels)):
            continue
        direction = "higher" if shift > 0 else "lower"
        note = f"response fits 10x {direction} current"
        for row in rows:
            if str(row.get("run")) == run_name:
                row["decade_suspect"] = True
                row["decade_suspect_note"] = note


def _comb_z(result: RunResult) -> float:
    if result.train is None or not np.isfinite(result.train.comb_z):
        return -np.inf
    return float(result.train.comb_z)


def _session_periods(runs: list[RunResult], config: EvokedConfig) -> tuple[float, ...]:
    """Distinct pulse periods established by the high-confidence runs.

    Returned as a small set rather than one number because this session used at
    least two interval settings -- the runs labelled "0.5s interval" repeat about
    every 609 ms, the rest about every 259 ms.
    """
    confident = [
        r.train.period_s
        for r in runs
        if r.ok
        and r.train is not None
        and r.train.ok
        and (r.train.comb_z >= config.min_comb_z or r.train.source == "scope")
    ]
    if not confident:
        return ()

    clusters: dict[int, list[float]] = {}
    for period in confident:
        clusters.setdefault(int(round(period * 1000.0 / 20.0)), []).append(period)
    # A single confident run is not enough to call a setting; require two.
    return tuple(
        sorted(float(np.median(values)) for values in clusters.values() if len(values) >= 2)
    )


def run_session(config: EvokedConfig, progress: ProgressCallback | None = None) -> SessionResult:
    """Analyse every run below the session folder. Writes nothing."""
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    conditions = discover_runs(config.session_folder, config.wiring_label)
    scope_dir = _resolve_scope_dir(config)
    stim_runs = [c for c in conditions if "baseline_recording" not in c.flags]
    skipped = len(conditions) - len(stim_runs)

    result = SessionResult(config=config)
    if skipped:
        result.notes.append(f"{skipped} baseline recording(s) found and not analysed as stimulus runs.")

    stim_runs, filter_note = _apply_wiring_filter(stim_runs, config)
    if filter_note:
        result.notes.append(filter_note)

    selected = stim_runs[: config.max_runs] if config.max_runs else stim_runs
    total_bytes = sum(run_size_bytes(c) for c in selected)
    report(f"{len(selected)} run(s) to analyse, {total_bytes / 1e9:.2f} GB")

    for index, condition in enumerate(selected, start=1):
        report(f"[{index}/{len(selected)}] {condition.raw_name}")
        result.runs.append(analyse_run(condition, config, scope_dir=scope_dir))

    # Second pass. The Keithley interval did not change between consecutive
    # recordings, so periods established by the runs where the stimulus is
    # unmistakable are used to re-time the runs where it is not. Only the
    # flagged minority is re-read, so this costs a fraction of a second pass.
    priors = _session_periods(result.runs, config)
    if priors:
        result.notes.append(
            "Session pulse periods established from confident runs: "
            + ", ".join(f"{p * 1000:.0f} ms" for p in priors)
        )
        weak = [
            i
            for i, r in enumerate(result.runs)
            if not (r.ok and r.train and r.train.ok)
            and not (r.train is not None and r.train.source == "scope")
        ]
        for count, index in enumerate(weak, start=1):
            condition = result.runs[index].condition
            report(f"[retime {count}/{len(weak)}] {condition.raw_name}")
            retimed = analyse_run(condition, config, priors, scope_dir=scope_dir)
            if retimed.ok and retimed.train and retimed.train.comb_z > _comb_z(result.runs[index]):
                result.runs[index] = retimed

    all_channels = sorted({ch for r in result.runs for ch in r.healthy_channels})
    positions = contact_positions_um(all_channels, config.contact_pitch_um, config.contact_order)
    for run_result in result.runs:
        result.rows.extend(_rows_for(run_result, config, positions))
        result.peak_rows.extend(_peak_rows_for(run_result, positions))

    result.sweep = _sweep_evidence(result.rows)
    _apply_decade_flags(result.rows)

    unknown = [r for r in selected if r.amplitude_mA is None]
    if unknown:
        result.notes.append(
            f"{len(unknown)} run(s) have no parseable amplitude and are excluded from the "
            "dose-response: " + ", ".join(r.raw_name for r in unknown)
        )
    report("done")
    return result


def run_single(config: EvokedConfig, progress: ProgressCallback | None = None) -> SessionResult:
    """Analyse one run folder on its own. Writes nothing.

    A whole-session pass reads about 1.9 GB, so this exists to let one run be
    checked -- does the timing come back, does the waveform look like a response
    -- before committing to the full sweep.
    """
    from .naming import Wiring, parse_run, parse_wiring

    folder = config.session_folder
    if not any(folder.glob("*.rhd")):
        raise ValueError(f"No .rhd files directly in {folder}. Point at a single run folder.")

    # The user's wiring label always wins; otherwise the wiring lives in the
    # configuration folder -- the parent for a flat run, the grandparent for a
    # run nested under an amplitude folder.
    wiring = None
    if config.wiring_label:
        wiring = Wiring(raw_name=config.wiring_label)
    if wiring is None:
        for candidate in (folder.parent, folder.parent.parent):
            if "stim" in candidate.name.lower():
                wiring = parse_wiring(candidate.name)
                break
    if wiring is None:
        wiring = parse_wiring(folder.parent.name)

    condition = parse_run(folder, wiring, folder.parent.name)
    scope_dir = _resolve_scope_dir(config)
    if progress is not None:
        progress(f"analysing {condition.raw_name}")

    result = SessionResult(config=config)
    run_result = analyse_run(condition, config, scope_dir=scope_dir)
    result.runs.append(run_result)
    positions = contact_positions_um(
        sorted(run_result.healthy_channels), config.contact_pitch_um, config.contact_order
    )
    result.rows.extend(_rows_for(run_result, config, positions))
    result.peak_rows.extend(_peak_rows_for(run_result, positions))
    _apply_decade_flags(result.rows)
    result.notes.append("Single-run mode: no session-wide period prior was available.")
    if progress is not None:
        progress("done")
    return result


def render_outputs(result: SessionResult) -> list[tuple[Path, bytes | str]]:
    """Build every output payload without touching the filesystem."""
    from . import figures, summary

    output_dir = result.config.output_dir
    payloads: list[tuple[Path, bytes | str]] = [
        (output_dir / "verdict.txt", summary.verdict_text(result)),
        (output_dir / "runs.csv", summary.runs_csv(result)),
        (output_dir / "conditions.csv", summary.conditions_csv(result)),
        (output_dir / "peaks.csv", summary.peaks_csv(result)),
    ]
    payloads.extend(figures.render_all(result, output_dir))
    return payloads


def write_outputs(result: SessionResult) -> list[Path]:
    """Write the rendered outputs atomically into the session folder."""
    payloads = render_outputs(result)
    atomic_write_all([(path, payload) for path, payload in payloads])
    return [path for path, _payload in payloads]


__all__ = [
    "RunResult",
    "SessionResult",
    "analyse_run",
    "render_outputs",
    "run_session",
    "run_single",
    "write_outputs",
]
