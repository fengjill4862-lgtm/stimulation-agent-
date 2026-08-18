"""Session orchestration: validate -> recovery -> (secondary) with stage gating.

``run_session`` never writes. It returns a ``SessionResult`` holding every
table (DataFrame), figure (PNG bytes), caption and the metadata dictionary;
``render_outputs`` decides every output path and ``write_outputs`` writes them
atomically. The validation table is produced (and handed to ``on_validation``)
before any analysis runs.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rhs_files import atomic_write_all
from stim_analysis import __version__ as PACKAGE_VERSION
from stim_analysis.config import AnalysisConfig, config_to_dict
from stim_analysis.epoch import EpochSet, baseline_stats, extract_epochs, gap_starts, window_slice
from stim_analysis.figures import (
    CaptionContext,
    TraceSnapshot,
    figure_to_png_bytes,
    plot_raw_traces_grid,
    plot_recovery_all_runs,
    plot_recovery_vs_amplitude,
    plot_recovery_vs_distance,
    plot_recovery_vs_impedance,
)
from stim_analysis.load_rhs import (
    RunRecord,
    contact_distance_um,
    contact_index,
    discover_run_folders,
    hardware_floor_ms,
    load_run,
    parse_run_folder_name,
    read_intan_impedance_csv,
)
from stim_analysis.recovery import (
    compute_recovery,
    condition_windows,
    mark_baseline_contamination,
    mark_retained,
    recovery_summary_table,
)
from stim_analysis.validate import (
    RunValidation,
    assign_blocks,
    exclusion_frame,
    format_validation_ascii,
    rail_long_frame,
    railed_mask,
    read_manifest,
    validate_run,
    validation_frame,
)

STAGES = ("validate", "recovery", "all")
ProgressFn = Callable[[str], None]


@dataclass
class SessionResult:
    parent: Path
    output_dir: Path
    stage: str
    cfg: AnalysisConfig
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    rail_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    exclusions: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_run_id: str | None = None
    channels: list[str] = field(default_factory=list)
    stim_channel: str | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    per_run_tables: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    run_folders: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)
    figure_titles: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exit_code: int = 0
    elapsed_s: float = 0.0
    write_per_run: bool = True

    @property
    def n_outputs(self) -> int:
        return len(render_outputs(self))

    def summary_lines(self) -> list[str]:
        lines = [
            f"stage: {self.stage}",
            f"runs: {len(self.validation)} ({int(self.validation['included'].sum()) if not self.validation.empty else 0} included)",
            f"baseline run: {self.baseline_run_id or 'none'}",
            f"tables: {len(self.tables)}, figures: {len(self.figures)}, per-run tables: {sum(len(v) for v in self.per_run_tables.values())}",
            f"exit code: {self.exit_code}",
        ]
        return lines


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _log(result: SessionResult, progress: ProgressFn | None, message: str) -> None:
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    result.log.append(line)
    if progress is not None:
        try:
            progress(message)
        except Exception:  # pragma: no cover - UI callbacks must not break the run
            pass


def _git_commit(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "stim_analysis": PACKAGE_VERSION,
    }
    for name in ("numpy", "scipy", "matplotlib", "pandas", "statsmodels"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = None
    return versions


def _matches_filter(folder: Path, run_id: str, run_filter: list[str] | None) -> bool:
    if not run_filter:
        return True
    haystack = f"{folder.name} {run_id}".lower()
    return any(token.lower() in haystack for token in run_filter)


def _error_validation(folder: Path, error: str) -> RunValidation:
    meta = parse_run_folder_name(folder.name)
    v = RunValidation(run_id=meta.run_id, run_folder=str(folder), label=meta.label, status="error")
    v.error = error
    v.included = False
    v.exclusion_reason = "error"
    v.stim_channel_folder = meta.stim_channel or ""
    return v


def _choose_baseline(validations: list[RunValidation], baseline_run: str | None) -> str | None:
    if baseline_run:
        token = baseline_run.lower()
        for v in validations:
            if token in v.run_id.lower() or token in Path(v.run_folder).name.lower():
                return v.run_id
    candidates = [v for v in validations if v.block == "baseline" and v.status in ("ok", "no_stim")]
    if not candidates:
        return None
    recordings = [v for v in candidates if v.label.lower().startswith("recording")]
    pool = recordings or candidates
    return sorted(pool, key=lambda v: v.run_id)[0].run_id


def _impedance_table(records: list[RunRecord], stim_channel: str | None, cfg: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = sorted(records, key=lambda r: r.run_id)
    first: dict[str, float] = {}
    for record in ordered:
        for channel in record.channels:
            z = record.impedance_ohms.get(channel, float("nan"))
            if channel not in first and np.isfinite(z):
                first[channel] = z
            rows.append(
                {
                    "run_id": record.run_id,
                    "run_time": record.meta.time,
                    "channel": channel,
                    "contact_index": contact_index(channel),
                    "distance_um": contact_distance_um(channel, stim_channel, cfg),
                    "impedance_kohm": z / 1e3 if np.isfinite(z) else float("nan"),
                    "phase_deg": record.impedance_phase_deg.get(channel, float("nan")),
                    "source": record.impedance_source,
                    "pct_change_vs_first_run": (
                        100.0 * (z - first[channel]) / first[channel]
                        if channel in first and np.isfinite(z) and first[channel] > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _channel_info(impedance: pd.DataFrame) -> dict[str, tuple[float, float]]:
    info: dict[str, tuple[float, float]] = {}
    if impedance.empty:
        return info
    for channel, group in impedance.groupby("channel"):
        info[channel] = (float(group["impedance_kohm"].median()), float(group["distance_um"].median()))
    return info


# -----------------------------------------------------------------------------
# main entry point
# -----------------------------------------------------------------------------


def run_session(
    parent: Path,
    cfg: AnalysisConfig,
    *,
    stage: str = "all",
    baseline_run: str | None = None,
    output_dir: Path | None = None,
    run_filter: list[str] | None = None,
    impedance_csv: Path | None = None,
    manifest: Path | None = None,
    channels: list[str] | None = None,
    progress: ProgressFn | None = None,
    on_validation: Callable[[pd.DataFrame], None] | None = None,
) -> SessionResult:
    """Run the pipeline up to ``stage`` on every run below ``parent``. Never writes."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    cfg.validate()
    started = time.time()
    parent = Path(parent).expanduser().resolve()
    output_dir = Path(output_dir).expanduser() if output_dir else parent / cfg.output_subdir
    result = SessionResult(parent=parent, output_dir=output_dir, stage=stage, cfg=cfg)
    _log(result, progress, f"session {parent.name}: stage {stage}")

    try:
        folders = discover_run_folders(parent)
    except (FileNotFoundError, NotADirectoryError) as exc:
        result.exit_code = 2
        result.warnings.append(str(exc))
        _log(result, progress, f"ERROR {exc}")
        return result
    if not folders:
        result.exit_code = 2
        result.warnings.append(f"no RHS run folders below {parent}")
        _log(result, progress, "ERROR no RHS run folders found")
        return result

    impedance_override = None
    if impedance_csv is not None:
        try:
            impedance_override = read_intan_impedance_csv(Path(impedance_csv))
            _log(result, progress, f"impedance override from {Path(impedance_csv).name}: {len(impedance_override)} channels")
        except Exception as exc:
            result.warnings.append(f"impedance CSV ignored: {exc}")
    manifest_map = None
    if manifest is not None:
        try:
            manifest_map = read_manifest(Path(manifest))
        except Exception as exc:
            result.warnings.append(f"manifest ignored: {exc}")

    # ---- pass 1: load + validate every run -----------------------------------
    validations: list[RunValidation] = []
    records: dict[str, RunRecord] = {}
    for folder in folders:
        meta = parse_run_folder_name(folder.name)
        if not _matches_filter(folder, meta.run_id, run_filter):
            continue
        _log(result, progress, f"validate {folder.name}")
        try:
            record = load_run(folder, cfg, channels, impedance_override)
            validation = validate_run(record, cfg)
            record.release_data()
            records[record.run_id] = record
        except Exception as exc:  # keep going; the table reports the failure
            validation = _error_validation(folder, f"{type(exc).__name__}: {exc}")
            result.warnings.append(f"{folder.name}: {exc}")
        validations.append(validation)
        result.run_folders[validation.run_id] = folder

    if not validations:
        result.exit_code = 2
        result.warnings.append("no runs matched the filter")
        _log(result, progress, "ERROR no runs matched")
        return result

    assign_blocks(validations, manifest_map)
    all_channels: list[str] = []
    for record in records.values():
        for channel in record.channels:
            if channel not in all_channels:
                all_channels.append(channel)
    all_channels.sort(key=contact_index)
    result.channels = all_channels
    stim_counts = pd.Series([r.stim_channel for r in records.values() if r.stim_channel]).value_counts()
    result.stim_channel = str(stim_counts.index[0]) if not stim_counts.empty else None

    result.validation = validation_frame(validations, all_channels)
    result.rail_long = rail_long_frame(validations)
    result.exclusions = exclusion_frame(validations)
    result.baseline_run_id = _choose_baseline(validations, baseline_run)
    result.tables["table01_validation"] = result.validation
    result.tables["table01_validation_rail_long"] = result.rail_long
    result.tables["exclusions"] = result.exclusions
    impedance = _impedance_table(list(records.values()), result.stim_channel, cfg)
    result.tables["table02_impedance_per_run"] = impedance
    for v in validations:
        result.per_run_tables.setdefault(v.run_id, {})["validation"] = result.validation[result.validation["run_id"] == v.run_id]
        rails = result.rail_long[result.rail_long["run_id"] == v.run_id] if not result.rail_long.empty else pd.DataFrame()
        result.per_run_tables[v.run_id]["rail"] = rails
    _log(result, progress, "validation table:\n" + format_validation_ascii(result.validation))
    if on_validation is not None:
        try:
            on_validation(result.validation)
        except Exception as exc:  # pragma: no cover
            result.warnings.append(f"on_validation callback failed: {exc}")
    n_excluded = int((~result.validation["included"]).sum())
    if n_excluded:
        result.exit_code = 1
        _log(result, progress, f"{n_excluded} run(s) excluded from quantitative analysis")
    if result.baseline_run_id is None:
        result.warnings.append("no no-stim baseline run found; block-vs-baseline comparison unavailable")

    result.metadata = _base_metadata(result, records, validations)
    if stage == "validate":
        result.elapsed_s = time.time() - started
        return result

    # ---- pass 2: recovery (spec section 4) ------------------------------------
    _recovery_stage(result, records, validations, progress)
    if stage == "recovery":
        result.metadata = _base_metadata(result, records, validations)
        result.elapsed_s = time.time() - started
        return result

    # ---- pass 3: secondary analyses --------------------------------------------
    try:
        from stim_analysis.secondary import run_secondary_stage  # implemented in phase 3
    except ImportError:
        run_secondary_stage = None
    if run_secondary_stage is None:
        _log(result, progress, "secondary stage unavailable in this build; stopping after recovery")
    else:
        run_secondary_stage(result, records, validations, progress)
    result.metadata = _base_metadata(result, records, validations)
    result.elapsed_s = time.time() - started
    return result


# -----------------------------------------------------------------------------
# recovery stage
# -----------------------------------------------------------------------------


def _hardware_note(record: RunRecord) -> str:
    h = record.header
    parts = []
    if np.isfinite(h.actual_lower_bandwidth_hz):
        parts.append(f"analog HP {h.actual_lower_bandwidth_hz:.2g} Hz")
    if h.dsp_enabled and np.isfinite(h.actual_dsp_cutoff_hz):
        parts.append(f"DSP HP {h.actual_dsp_cutoff_hz:.2g} Hz")
    if np.isfinite(h.actual_upper_bandwidth_hz):
        parts.append(f"LP {h.actual_upper_bandwidth_hz:.0f} Hz")
    return ", ".join(parts)


def _recovery_stage(
    result: SessionResult,
    records: dict[str, RunRecord],
    validations: list[RunValidation],
    progress: ProgressFn | None,
) -> None:
    cfg = result.cfg
    by_id = {v.run_id: v for v in validations}
    trials_frames: list[pd.DataFrame] = []
    floor_by_run: dict[str, float] = {}
    snapshots: dict[float, TraceSnapshot] = {}
    dropped_epochs: dict[str, int] = {}
    hardware_notes: set[str] = set()
    ladder_phase = _ladder_phase(validations)

    for v in sorted(validations, key=lambda item: (item.block, item.amplitude_uA_data if np.isfinite(item.amplitude_uA_data) else 1e9, item.phase_us_data if np.isfinite(item.phase_us_data) else 1e9)):
        if v.status != "ok" or v.n_detected == 0:
            continue
        folder = result.run_folders[v.run_id]
        _log(result, progress, f"recovery {folder.name}")
        try:
            record = load_run(folder, cfg, result.channels or None)
        except Exception as exc:
            result.warnings.append(f"{folder.name}: reload failed: {exc}")
            continue
        assert record.data is not None and record.events is not None
        events = record.events
        floor = hardware_floor_ms(events, record.settings, record.sample_rate_hz)
        floor_by_run[v.run_id] = floor
        hardware_notes.add(_hardware_note(record))
        epochs = extract_epochs(
            record.data.raw_uV,
            record.sample_rate_hz,
            events.onset_samples,
            events.event_numbers,
            cfg,
            run_id=v.run_id,
            channels=record.channels,
            gap_sample_starts=gap_starts(record.data.timestamps),
        )
        dropped_epochs[v.run_id] = int(np.count_nonzero(~epochs.kept))
        onset_by_event = dict(zip(events.event_numbers.tolist(), events.onset_s.tolist()))
        rails = {rail.channel: rail for rail in v.rails}
        run_frames: list[pd.DataFrame] = []
        for c_index, channel in enumerate(record.channels):
            ep = epochs.raw[c_index][epochs.kept]
            if ep.shape[0] == 0:
                continue
            mean, sd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
            centred = ep - mean[:, None]
            rail = rails.get(channel)
            railed = None
            if rail is not None and rail.is_railed:
                railed = np.stack([railed_mask(row, rail, cfg, record.sample_rate_hz) for row in ep])
            frame = compute_recovery(
                centred, epochs.t_ms, record.sample_rate_hz, sd, cfg,
                railed=railed, core=epochs.core, event_numbers=epochs.event_numbers[epochs.kept],
            )
            frame = mark_baseline_contamination(frame, onset_by_event, cfg)
            frame.insert(0, "channel", channel)
            frame.insert(0, "run_id", v.run_id)
            frame["onset_s"] = frame["event_number"].map(onset_by_event)
            frame["baseline_mean_uV"] = mean
            frame["impedance_kohm"] = record.impedance_ohms.get(channel, float("nan")) / 1e3
            frame["distance_um"] = contact_distance_um(channel, record.stim_channel, cfg)
            run_frames.append(frame)
        if not run_frames:
            continue
        run_trials = pd.concat(run_frames, ignore_index=True)
        run_trials["block"] = v.block
        run_trials["included"] = v.included
        run_trials["amplitude_uA"] = v.amplitude_uA_data
        run_trials["phase_us"] = v.phase_us_data
        run_trials["charge_nC_per_phase"] = v.charge_nC_per_phase
        run_trials["hardware_floor_ms"] = floor
        trials_frames.append(run_trials)

        # Figure 3 snapshots: raw traces at the low amplitudes on identical axes
        if v.included and ladder_phase is not None and abs(v.phase_us_data - ladder_phase) <= 10.0:
            for target in cfg.trace_amplitudes_uA:
                if abs(v.amplitude_uA_data - target) <= 2.0 and target not in snapshots:
                    win = window_slice(epochs.t_ms, *cfg.trace_window_ms)
                    kept = epochs.kept
                    # Truly raw: no baseline subtraction, so the rail lines are exact.
                    traces = np.ascontiguousarray(epochs.raw[:, kept, win]).astype(np.float32)
                    levels = {
                        channel: (rails[channel].neg_level_uV, rails[channel].pos_level_uV)
                        if channel in rails else (float("nan"), float("nan"))
                        for channel in record.channels
                    }
                    snapshots[target] = TraceSnapshot(
                        amplitude_uA=float(v.amplitude_uA_data), run_id=v.run_id,
                        channels=list(record.channels), traces=traces, t_ms=epochs.t_ms[win],
                        rail_levels=levels, n_events=int(kept.sum()),
                    )
        record.release_data()

    if not trials_frames:
        result.warnings.append("no stim runs to analyze")
        return
    trials = pd.concat(trials_frames, ignore_index=True)
    windows = condition_windows(trials, cfg, floor_by_run)
    trials = mark_retained(trials, windows, cfg)
    result.tables["trials_recovery"] = trials
    result.tables["condition_windows"] = windows
    result.tables["table03_recovery_summary"] = recovery_summary_table(trials, windows, result.validation)
    for run_id, group in trials.groupby("run_id"):
        result.per_run_tables.setdefault(run_id, {})["recovery_trials"] = group
        result.per_run_tables[run_id]["condition_windows"] = windows[windows["run_id"] == run_id]

    # ---- figures 1, 2, 2b, 3, 3b, S5 ------------------------------------------
    hardware = "; ".join(sorted(note for note in hardware_notes if note))
    included_ladder = trials[(trials["included"]) & (trials["block"] == "block1")]
    n_dropped = sum(dropped_epochs.get(run_id, 0) for run_id in included_ladder["run_id"].unique())
    ctx = CaptionContext(
        n_retained=int(len(included_ladder)),
        n_rejected=int(n_dropped),
        reject_reasons={"epoch_at_recording_edge_or_gap": int(n_dropped)},
        blank_desc=(
            f"none for the recovery measurement; derived per-epoch blank = recovery + {cfg.blank_margin_ms:g} ms, "
            f"floor = pulse + amp settle (hardware); condition post-window start = P{int(cfg.recovery_quantile * 100)}(recovery) + {cfg.blank_margin_ms:g} ms"
        ),
        filter_desc=f"none (raw wideband, per-trial baseline mean subtracted; hardware {hardware})" if hardware else "none (raw wideband)",
        epoch_desc=(
            f"{cfg.epoch_ms[0]:g} to {cfg.epoch_ms[1]:g} ms; baseline {cfg.baseline_ms[0]:g} to {cfg.baseline_ms[1]:g} ms; "
            f"threshold = max({cfg.threshold_k:g} x baseline SD, {cfg.threshold_floor_uV:g} uV); quiet {cfg.quiet_ms:g} ms"
        ),
        note=(
            f"Block 1 amplitude ladder, included runs only; censored trials shown at the epoch end. "
            f"n censored = {int(included_ladder['censored'].sum())}."
        ),
    )
    info = _channel_info(result.tables["table02_impedance_per_run"])
    if not included_ladder.empty:
        _add_figure(result, "fig01_recovery_vs_amplitude", "Recovery time vs stimulus current, per channel",
                    plot_recovery_vs_amplitude(included_ladder, windows, cfg, ctx, channel_info=info), ctx)
        _add_figure(result, "fig02_recovery_vs_impedance", "Recovery time vs channel impedance",
                    plot_recovery_vs_impedance(included_ladder, cfg, ctx), ctx)
        _add_figure(result, "fig02b_recovery_vs_distance", "Recovery time vs distance from the stim contact",
                    plot_recovery_vs_distance(included_ladder, cfg, ctx), ctx)
    else:
        result.warnings.append("no included Block 1 runs: figures 1-2 skipped")
    if snapshots:
        snaps = [snapshots[k] for k in sorted(snapshots)]
        n_tr = sum(s.n_events for s in snaps)
        ctx3 = CaptionContext(
            n_retained=n_tr, n_rejected=sum(dropped_epochs.get(s.run_id, 0) for s in snaps),
            reject_reasons={"epoch_at_recording_edge_or_gap": sum(dropped_epochs.get(s.run_id, 0) for s in snaps)},
            blank_desc="none", filter_desc=f"none (raw wideband, no baseline subtraction; hardware {hardware})" if hardware else "none (raw wideband)",
            epoch_desc=f"shown {cfg.trace_window_ms[0]:g} to {cfg.trace_window_ms[1]:g} ms of the {cfg.epoch_ms[0]:g} to {cfg.epoch_ms[1]:g} ms epoch",
            note="Identical axes across amplitudes; dashed red = empirical rail level per channel.",
        )
        amps = ", ".join(f"{s.amplitude_uA:g}" for s in snaps)
        _add_figure(result, "fig03_raw_traces_low_amplitudes", f"Raw stim-triggered traces at {amps} uA (full scale)",
                    plot_raw_traces_grid(snaps, cfg, ctx3), ctx3)
        _add_figure(result, "fig03b_raw_traces_low_amplitudes_zoom", f"Raw stim-triggered traces at {amps} uA (zoom, off-scale marked)",
                    plot_raw_traces_grid(snaps, cfg, ctx3, ylim=cfg.lim_trace_zoom_uV,
                                         title="Raw stim-triggered traces, zoomed y-axis (off-scale samples marked in red)"), ctx3)
    else:
        result.warnings.append("no included runs at the requested trace amplitudes: figure 3 skipped")
    ctx_all = CaptionContext(
        n_retained=int(len(trials)), n_rejected=int(sum(dropped_epochs.values())),
        reject_reasons={"epoch_at_recording_edge_or_gap": int(sum(dropped_epochs.values()))},
        blank_desc=ctx.blank_desc, filter_desc=ctx.filter_desc, epoch_desc=ctx.epoch_desc,
        note="Every run with stim pulses, including excluded ones (shaded). Artifact-shape inspection only for excluded runs.",
    )
    _add_figure(result, "figS5_recovery_all_runs_incl_excluded", "Recovery time for every run (excluded shaded)",
                plot_recovery_all_runs(trials, result.validation, cfg, ctx_all), ctx_all)
    _log(result, progress, f"recovery stage done: {len(trials)} trials, {len(windows)} conditions")


def _ladder_phase(validations: list[RunValidation]) -> float | None:
    phases = [v.phase_us_data for v in validations if v.block == "block1" and np.isfinite(v.phase_us_data)]
    if not phases:
        return None
    return float(np.median(phases))


def _add_figure(result: SessionResult, stem: str, title: str, fig, ctx: CaptionContext) -> None:
    from stim_analysis.figures import build_caption

    result.figures[stem] = figure_to_png_bytes(fig, dpi=result.cfg.dpi)
    result.captions[stem] = build_caption(ctx)
    result.figure_titles[stem] = title


# -----------------------------------------------------------------------------
# metadata / outputs
# -----------------------------------------------------------------------------


def _base_metadata(result: SessionResult, records: dict[str, RunRecord], validations: list[RunValidation]) -> dict[str, object]:
    meta = dict(result.metadata)  # keep anything a later stage added
    meta.update(
        {
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "package_version": PACKAGE_VERSION,
            "stage": result.stage,
            "parent": str(result.parent),
            "output_dir": str(result.output_dir),
            "config": config_to_dict(result.cfg),
            "versions": _versions(),
            "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
            "argv": sys.argv,
            "channels": result.channels,
            "stim_channel": result.stim_channel,
            "baseline_run_id": result.baseline_run_id,
            "runs": [
                {
                    "run_id": v.run_id,
                    "folder": Path(v.run_folder).name,
                    "block": v.block,
                    "included": v.included,
                    "exclusion_reason": v.exclusion_reason,
                    "status": v.status,
                    "n_detected": v.n_detected,
                    "n_commanded_total": v.n_commanded_total,
                    "amplitude_uA": v.amplitude_uA_data,
                    "phase_us": v.phase_us_data,
                    "hardware_floor_ms": v.hardware_floor_ms,
                    "warnings": v.warnings,
                    "header": _header_meta(records.get(v.run_id)),
                }
                for v in validations
            ],
            "warnings": result.warnings,
        }
    )
    if "condition_windows" in result.tables and not result.tables["condition_windows"].empty:
        meta["condition_windows"] = json.loads(result.tables["condition_windows"].to_json(orient="records"))
    return meta


def _header_meta(record: RunRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    h = record.header
    return {
        "sample_rate_hz": h.sample_rate_hz,
        "stim_step_size_uA": h.stim_step_size_uA,
        "dc_amp_data_saved": h.dc_amp_data_saved,
        "dsp_enabled": h.dsp_enabled,
        "actual_dsp_cutoff_hz": h.actual_dsp_cutoff_hz,
        "actual_lower_bandwidth_hz": h.actual_lower_bandwidth_hz,
        "actual_upper_bandwidth_hz": h.actual_upper_bandwidth_hz,
        "actual_lower_settle_bandwidth_hz": h.actual_lower_settle_bandwidth_hz,
        "amp_settle_mode": h.amp_settle_mode,
        "charge_recovery_mode": h.charge_recovery_mode,
        "impedance_source": record.impedance_source,
        "n_files": record.n_files,
        "settings": None if record.settings is None else {
            "source": record.settings.source,
            "pulse_or_train": record.settings.pulse_or_train,
            "num_pulses": record.settings.num_pulses,
            "train_period_us": record.settings.train_period_us,
            "refractory_us": record.settings.refractory_us,
            "first_amp_uA": record.settings.first_amp_uA,
            "first_phase_us": record.settings.first_phase_us,
            "post_amp_settle_us": record.settings.post_amp_settle_us,
            "enable_charge_recovery": record.settings.enable_charge_recovery,
            "sample_rate_hz": record.settings.sample_rate_hz,
        },
    }


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def render_outputs(result: SessionResult) -> list[tuple[Path, bytes | str]]:
    """Every (path, payload) the Save step would write. Paths are decided here."""
    out = result.output_dir
    items: list[tuple[Path, bytes | str]] = []
    for stem, frame in result.tables.items():
        items.append((out / f"{stem}.csv", frame.to_csv(index=False)))
    for stem, png in result.figures.items():
        items.append((out / f"{stem}.png", png))
    if result.figures:
        index = pd.DataFrame(
            [
                {"file": f"{stem}.png", "title": result.figure_titles.get(stem, ""), "caption": result.captions.get(stem, "")}
                for stem in result.figures
            ]
        )
        items.append((out / "figures_index.csv", index.to_csv(index=False)))
    items.append((out / "stim_analysis_metadata.json", json.dumps(result.metadata, indent=2, default=_json_default)))
    items.append((out / "run_log.txt", "\n".join(result.log) + "\n"))
    for run_id, tables in (result.per_run_tables.items() if result.write_per_run else []):
        folder = result.run_folders.get(run_id)
        if folder is None:
            continue
        for stem, frame in tables.items():
            if frame is None or frame.empty:
                continue
            items.append((folder / f"stim_analysis_{run_id}_{stem}.csv", frame.to_csv(index=False)))
    return items


def write_outputs(result: SessionResult) -> list[Path]:
    """Atomically write everything from render_outputs. Creates the output folder."""
    items = render_outputs(result)
    result.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_all(items)
    return [path for path, _payload in items]


def format_exception() -> str:
    return traceback.format_exc()


__all__ = [
    "STAGES",
    "SessionResult",
    "render_outputs",
    "run_session",
    "write_outputs",
]
