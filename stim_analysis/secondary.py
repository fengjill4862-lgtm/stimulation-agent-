"""Secondary analyses (spec sections 5-8) on the conditions that survive gating.

Runs after the recovery stage inside ``pipeline.run_session``:

* per included stim run: epoch -> blank (per-epoch recovery) -> filter ->
  per-trial response amplitude and band power in baseline / early / late
  windows -> paired dB keyed by event; within-block drift; within-run shuffle
* the no-stim baseline run: pseudo-epochs -> same metrics (noise floor,
  block-vs-baseline reference, baseline-run shuffle)
* cross-run: comparisons (a) within-epoch, (b) block vs baseline, (c) across
  amplitude; drift; charge dependence; spatial decay; compliance; amplitude-
  response models; channel model; log-normal checks; figures 4-8 + S1-S4
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import (
    FilterDesign,
    baseline_stats,
    blank_epochs,
    blank_windows,
    design_bandpass,
    extract_epochs,
    filter_epochs,
    gap_starts,
    pseudo_onsets,
)
from stim_analysis.figures import (
    CaptionContext,
    build_caption,
    figure_to_png_bytes,
    plot_bandpower_heatmap,
    plot_bandpower_vs_amplitude,
    plot_compliance,
    plot_drift,
    plot_impedance_drift,
    plot_model_fits,
    plot_noise_floor,
    plot_qq_grid,
    plot_spatial_decay,
)
from stim_analysis.load_rhs import RunRecord, contact_distance_um, hardware_floor_ms, load_run
from stim_analysis.metrics import (
    band_power_per_trial,
    first_vs_last,
    noise_floor_table,
    paired_change_db,
    response_amplitude_per_trial,
)
from stim_analysis.models import fit_amplitude_response
from stim_analysis.stats import (
    HAS_STATSMODELS,
    bootstrap_independent_db_ci,
    bootstrap_paired_db_ci,
    clean_intervals,
    draw_fake_onsets,
    fit_cross_channel_model,
    lognormal_check,
    median_with_ci,
    qq_points,
    spearman_ci,
)
from stim_analysis.validate import RunValidation

ProgressFn = Callable[[str], None]
GENERIC_EARLY_MS = (50.0, 350.0)


def _log(result, progress: ProgressFn | None, message: str) -> None:
    from stim_analysis.pipeline import _log as pipeline_log

    pipeline_log(result, progress, message)


def _designs(cfg: AnalysisConfig, sample_rate_hz: float, cache: dict[float, dict[str, FilterDesign]]) -> dict[str, FilterDesign]:
    key = round(float(sample_rate_hz), 6)
    if key not in cache:
        designs = {
            "broadband": design_bandpass(sample_rate_hz, cfg.highpass_hz, cfg.lowpass_hz, cfg.filter_order, cfg.zero_phase, cfg.filter_pad_ms)
        }
        for band in cfg.bands:
            designs[band.name] = design_bandpass(sample_rate_hz, band.low_hz, band.high_hz, cfg.filter_order, cfg.zero_phase, cfg.filter_pad_ms)
        cache[key] = designs
    return cache[key]


def _equal_length_pair(cfg: AnalysisConfig, post: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Crop the post window and the END of the baseline window to a common length.

    Log-power estimates from windows of different length have different degrees
    of freedom, so E[10 log10(post/base)] is biased on pure noise (about -0.6 dB
    for alpha with 300 vs 450 ms). Equal lengths remove that bias; the shuffle
    control on the no-stim run verifies it.
    """
    base_start, base_end = float(cfg.baseline_ms[0]), float(cfg.baseline_ms[1])
    length = min(post[1] - post[0], base_end - base_start)
    return (float(post[0]), float(post[0] + length)), (float(base_end - length), base_end)


def _windows_for(cfg: AnalysisConfig, post_start: float, post_end: float) -> dict[str, tuple[float, float]]:
    """Analysis windows for one condition; every post window has an equal-length baseline partner."""
    windows: dict[str, tuple[float, float]] = {"baseline": tuple(cfg.baseline_ms)}
    late, base_late = _equal_length_pair(cfg, tuple(cfg.late_ms))
    windows["late"] = late
    windows["baseline_late"] = base_late
    if np.isfinite(post_start) and np.isfinite(post_end) and post_end > post_start:
        early, base_early = _equal_length_pair(cfg, (float(post_start), float(post_end)))
        windows["early"] = early
        windows["baseline_early"] = base_early
    return windows


def _add_figure(result, stem: str, title: str, fig, ctx: CaptionContext) -> None:
    result.figures[stem] = figure_to_png_bytes(fig, dpi=result.cfg.dpi)
    result.captions[stem] = build_caption(ctx)
    result.figure_titles[stem] = title


# -----------------------------------------------------------------------------
# main entry
# -----------------------------------------------------------------------------


def run_secondary_stage(result, records: dict[str, RunRecord], validations: list[RunValidation], progress: ProgressFn | None) -> None:
    cfg = result.cfg
    trials_recovery = result.tables.get("trials_recovery")
    windows_table = result.tables.get("condition_windows")
    if trials_recovery is None or trials_recovery.empty or windows_table is None or windows_table.empty:
        result.warnings.append("secondary stage skipped: no recovery trials")
        return
    started = time.time()
    rng = np.random.default_rng(cfg.seed)
    design_cache: dict[float, dict[str, FilterDesign]] = {}
    by_id = {v.run_id: v for v in validations}
    cond = windows_table.set_index(["run_id", "channel"])
    recovery_lookup = trials_recovery.set_index(["run_id", "channel", "event_number"])["recovery_ms"]

    amp_frames: list[pd.DataFrame] = []
    pow_frames: list[pd.DataFrame] = []
    db_frames: list[pd.DataFrame] = []
    shuffle_frames: list[pd.DataFrame] = []
    baseline_amp = pd.DataFrame()
    baseline_pow = pd.DataFrame()
    baseline_impedance: dict[str, float] = {}
    n_dropped_epochs = 0
    shuffle_dropped = 0

    # ---- baseline (no-stim) run --------------------------------------------------
    if result.baseline_run_id and result.baseline_run_id in result.run_folders:
        folder = result.run_folders[result.baseline_run_id]
        _log(result, progress, f"secondary: baseline run {folder.name}")
        try:
            record = load_run(folder, cfg, result.channels or None)
            assert record.data is not None
            designs = _designs(cfg, record.sample_rate_hz, design_cache)
            # jittered so periodic components (line noise, synthetic sines) do not share a phase across pseudo-epochs
            onsets = pseudo_onsets(record.n_samples, record.sample_rate_hz, cfg, spacing_s=1.0, jitter_s=0.2, rng=rng)
            numbers = np.arange(1, onsets.size + 1)
            epochs = extract_epochs(record.data.raw_uV, record.sample_rate_hz, onsets, numbers, cfg, run_id=record.run_id, channels=record.channels, gap_sample_starts=gap_starts(record.data.timestamps))
            baseline_impedance = {c: record.impedance_ohms.get(c, float("nan")) / 1e3 for c in record.channels}
            a_frames, p_frames = [], []
            for c_index, channel in enumerate(record.channels):
                ep = epochs.raw[c_index][epochs.kept]
                if ep.shape[0] == 0:
                    continue
                mean, _sd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
                centred = np.asarray(ep, dtype=np.float64) - mean[:, None]
                # windows: baseline, late, generic early, and every condition's early window on this channel
                windows = _windows_for(cfg, float("nan"), float("nan"))
                windows["early_generic"], windows["baseline_early_generic"] = _equal_length_pair(cfg, GENERIC_EARLY_MS)
                for (run_id, ch), row in cond.iterrows():
                    if ch == channel and np.isfinite(row["post_start_ms"]) and row["post_end_ms"] > row["post_start_ms"]:
                        windows[f"early_{run_id}"], windows[f"baseline_early_{run_id}"] = _equal_length_pair(cfg, (float(row["post_start_ms"]), float(row["post_end_ms"])))
                broadband = filter_epochs(centred, designs["broadband"])
                amp = response_amplitude_per_trial(broadband, epochs.t_ms, windows, event_numbers=epochs.event_numbers[epochs.kept], core=epochs.core)
                pw = band_power_per_trial(centred, epochs.t_ms, cfg.bands, windows, designs, event_numbers=epochs.event_numbers[epochs.kept], core=epochs.core)
                for frame in (amp, pw):
                    frame.insert(0, "channel", channel)
                    frame.insert(0, "run_id", record.run_id)
                a_frames.append(amp)
                p_frames.append(pw)
            if a_frames:
                baseline_amp = pd.concat(a_frames, ignore_index=True)
                baseline_pow = pd.concat(p_frames, ignore_index=True)
                # baseline-run shuffle: generic early vs baseline should be ~0 dB
                sh = paired_change_db(baseline_pow, post="early_generic", base="baseline_early_generic", keys=("run_id", "channel", "event_number", "band"))
                sh["mode"] = "baseline_run"
                sh["window"] = "early_generic"
                shuffle_frames.append(sh)
            record.release_data()
        except Exception as exc:
            result.warnings.append(f"baseline run failed in secondary stage: {exc}")
    else:
        result.warnings.append("no baseline run: comparison (b), noise floor and the baseline-run shuffle are unavailable")

    # ---- included stim runs -------------------------------------------------------
    stim_runs = [v for v in validations if v.included and v.status == "ok" and v.n_detected > 0]
    stim_runs.sort(key=lambda v: (v.block, v.amplitude_uA_data if np.isfinite(v.amplitude_uA_data) else 1e9, v.phase_us_data if np.isfinite(v.phase_us_data) else 1e9))
    for v in stim_runs:
        folder = result.run_folders[v.run_id]
        _log(result, progress, f"secondary: {folder.name}")
        try:
            record = load_run(folder, cfg, result.channels or None)
        except Exception as exc:
            result.warnings.append(f"{folder.name}: reload failed in secondary stage: {exc}")
            continue
        assert record.data is not None and record.events is not None
        events = record.events
        designs = _designs(cfg, record.sample_rate_hz, design_cache)
        floor = hardware_floor_ms(events, record.settings, record.sample_rate_hz)
        gaps = gap_starts(record.data.timestamps)
        epochs = extract_epochs(record.data.raw_uV, record.sample_rate_hz, events.onset_samples, events.event_numbers, cfg, run_id=v.run_id, channels=record.channels, gap_sample_starts=gaps)
        n_dropped_epochs += int(np.count_nonzero(~epochs.kept))
        kept_numbers = epochs.event_numbers[epochs.kept]
        # fake onsets for the within-run shuffle (shared across channels)
        fake_epochs = None
        fake_pulse_windows_by_channel: dict[str, list[list[tuple[float, float]]]] = {}
        if cfg.shuffle_enabled:
            post_starts = [cond.loc[(v.run_id, ch), "post_start_ms"] for ch in record.channels if (v.run_id, ch) in cond.index]
            blank_end_ms = float(np.nanmax(post_starts)) if post_starts and np.isfinite(np.nanmax(post_starts)) else float(cfg.epoch_ms[1])
            blank_end_samples = events.onset_samples + int(round(blank_end_ms * 1e-3 * record.sample_rate_hz))
            intervals = clean_intervals(events.onset_samples, blank_end_samples, record.n_samples, guard_samples=int(round(0.02 * record.sample_rate_hz)))
            fake = draw_fake_onsets(intervals, min(cfg.shuffle_n_events, events.n_events), rng, min_separation=int(round(0.5 * record.sample_rate_hz)))
            if fake.size:
                fake_epochs = extract_epochs(record.data.raw_uV, record.sample_rate_hz, fake, np.arange(1, fake.size + 1), cfg, run_id=v.run_id, channels=record.channels, gap_sample_starts=gaps)
        for c_index, channel in enumerate(record.channels):
            if (v.run_id, channel) not in cond.index:
                continue
            crow = cond.loc[(v.run_id, channel)]
            post_start, post_end = float(crow["post_start_ms"]), float(crow["post_end_ms"])
            windows = _windows_for(cfg, post_start, post_end)
            ep = epochs.raw[c_index][epochs.kept]
            if ep.shape[0] == 0:
                continue
            mean, _sd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
            centred = np.asarray(ep, dtype=np.float64) - mean[:, None]
            recovery = np.array([recovery_lookup.get((v.run_id, channel, int(n)), np.nan) for n in kept_numbers], dtype=float)
            # Neighbouring pulses (at +/-1 s) fall inside the filter padding; blank them
            # through their own measured recovery so filtfilt cannot ring their
            # artifact into the baseline / late windows of this epoch.
            span_lo = cfg.epoch_ms[0] - cfg.filter_pad_ms
            span_hi = cfg.epoch_ms[1] + cfg.filter_pad_ms
            fallback_end = post_start if np.isfinite(post_start) else float(cfg.epoch_ms[1])
            neighbours: list[list[tuple[float, float]]] = []
            for own_number, own_onset in zip(kept_numbers, epochs.onset_samples[epochs.kept]):
                items: list[tuple[float, float]] = []
                rel_all = (events.onset_samples - int(own_onset)) * 1e3 / record.sample_rate_hz
                for other_number, rel in zip(events.event_numbers, rel_all):
                    if other_number == own_number or rel <= span_lo - fallback_end or rel >= span_hi:
                        continue
                    other_recovery = recovery_lookup.get((v.run_id, channel, int(other_number)), np.nan)
                    end = max(float(other_recovery) + cfg.blank_margin_ms, floor) if np.isfinite(other_recovery) else fallback_end
                    items.append((float(rel) - cfg.blank_pre_ms, float(rel) + end))
                neighbours.append(items)
            bw = blank_windows(recovery, cfg, floor, post_start_ms=post_start, extra_pulse_windows_ms=neighbours)
            blanked = blank_epochs(centred, epochs.t_ms, bw)
            broadband = filter_epochs(blanked, designs["broadband"])
            amp = response_amplitude_per_trial(broadband, epochs.t_ms, windows, event_numbers=kept_numbers, core=epochs.core)
            pw = band_power_per_trial(blanked, epochs.t_ms, cfg.bands, windows, designs, event_numbers=kept_numbers, core=epochs.core)
            for frame in (amp, pw):
                frame.insert(0, "channel", channel)
                frame.insert(0, "run_id", v.run_id)
            amp_frames.append(amp)
            pow_frames.append(pw)
            for window_name in ("early", "late"):
                if window_name not in windows:
                    continue
                db = paired_change_db(pw, post=window_name, base=f"baseline_{window_name}", keys=("run_id", "channel", "event_number", "band"))
                db["window"] = window_name
                db_frames.append(db)
            # within-run shuffle for this channel: blank the real pulses inside each fake epoch
            if fake_epochs is not None:
                fep = fake_epochs.raw[c_index][fake_epochs.kept]
                if fep.shape[0]:
                    fake_on = fake_epochs.onset_samples[fake_epochs.kept]
                    extra: list[list[tuple[float, float]]] = []
                    span_ms = (cfg.epoch_ms[0] - cfg.filter_pad_ms, cfg.epoch_ms[1] + cfg.filter_pad_ms)
                    for f in fake_on:
                        rel_ms = (events.onset_samples - int(f)) * 1e3 / record.sample_rate_hz
                        inside = rel_ms[(rel_ms > span_ms[0] - post_start) & (rel_ms < span_ms[1])]
                        extra.append([(float(r) - cfg.blank_pre_ms, float(r) + max(post_start if np.isfinite(post_start) else float(cfg.epoch_ms[1]), floor)) for r in inside])
                    f_mean, _ = baseline_stats(fep, fake_epochs.t_ms, cfg.baseline_ms)
                    f_centred = np.asarray(fep, dtype=np.float64) - f_mean[:, None]
                    f_blanked = blank_epochs(f_centred, fake_epochs.t_ms, extra)
                    f_pw = band_power_per_trial(f_blanked, fake_epochs.t_ms, cfg.bands, windows, designs, event_numbers=fake_epochs.event_numbers[fake_epochs.kept], core=fake_epochs.core)
                    f_pw.insert(0, "channel", channel)
                    f_pw.insert(0, "run_id", v.run_id)
                    for window_name in ("early", "late"):
                        if window_name not in windows:
                            continue
                        sh = paired_change_db(f_pw, post=window_name, base=f"baseline_{window_name}", keys=("run_id", "channel", "event_number", "band"))
                        # a fake epoch counts only when neither window overlaps a blanked real pulse
                        overlap = []
                        for f, items in zip(fake_on, extra):
                            w0, w1 = windows[window_name]
                            b0, b1 = windows[f"baseline_{window_name}"]
                            bad = any((a < w1 and b > w0) or (a < b1 and b > b0) for a, b in items)
                            overlap.append(bad)
                        clean_events = set(fake_epochs.event_numbers[fake_epochs.kept][~np.asarray(overlap, dtype=bool)].tolist())
                        if window_name == "early":
                            shuffle_dropped += int(np.count_nonzero(overlap))
                        sh = sh[sh["event_number"].isin(clean_events)]
                        sh["mode"] = "within_run"
                        sh["window"] = window_name
                        shuffle_frames.append(sh)
        record.release_data()

    if not amp_frames:
        result.warnings.append("secondary stage: no included stim runs with usable epochs")
        return

    # ---- assemble per-trial tables ---------------------------------------------------
    run_info = result.validation.set_index("run_id")
    def annotate(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["block"] = out["run_id"].map(run_info["block"])
        out["amplitude_uA"] = out["run_id"].map(run_info["amplitude_uA_data"])
        out["phase_us"] = out["run_id"].map(run_info["phase_us_data"])
        out["charge_nC_per_phase"] = out["run_id"].map(run_info["charge_nC_per_phase"])
        out["included"] = out["run_id"].map(run_info["included"])
        return out

    retained_early = trials_recovery.set_index(["run_id", "channel", "event_number"])["retained_early"]
    retained_late = trials_recovery.set_index(["run_id", "channel", "event_number"])["retained_late"]
    def add_retained(frame: pd.DataFrame, window_col: str = "window") -> pd.DataFrame:
        if frame.empty:
            return frame
        keys = list(zip(frame["run_id"], frame["channel"], frame["event_number"]))
        early = np.array([bool(retained_early.get(k, False)) for k in keys])
        late = np.array([bool(retained_late.get(k, False)) for k in keys])
        w = frame[window_col].to_numpy()
        frame = frame.copy()
        frame["retained"] = np.where(w == "early", early, np.where(w == "late", late, True))
        return frame

    amplitude = add_retained(annotate(pd.concat(amp_frames, ignore_index=True)))
    power = add_retained(annotate(pd.concat(pow_frames, ignore_index=True)))
    paired = add_retained(annotate(pd.concat(db_frames, ignore_index=True))) if db_frames else pd.DataFrame()
    trial_info = trials_recovery.set_index(["run_id", "channel", "event_number"])
    for frame in (amplitude, power, paired):
        if frame.empty:
            continue
        keys = list(zip(frame["run_id"], frame["channel"], frame["event_number"]))
        frame["impedance_kohm"] = [trial_info["impedance_kohm"].get(k, np.nan) for k in keys]
        frame["distance_um"] = [trial_info["distance_um"].get(k, np.nan) for k in keys]
    result.tables["trials_response_amplitude"] = amplitude
    result.tables["trials_bandpower"] = power
    result.tables["trials_paired_db"] = paired
    if not baseline_amp.empty:
        result.tables["baseline_run_response_amplitude"] = baseline_amp
        result.tables["baseline_run_bandpower"] = baseline_pow

    channel_info = _channel_info(result)
    filter_desc = cfg.filter_label + "; per band the same design at the band edges"
    blank_desc = (
        f"per epoch -{cfg.blank_pre_ms:g} ms to recovery + {cfg.blank_margin_ms:g} ms (floor = pulse + amp settle), linear interpolation, neighbouring pulses inside the filter pad blanked the same way; "
        f"early window per channel x run = P{int(cfg.recovery_quantile * 100)}(recovery)+{cfg.blank_margin_ms:g} ms for {cfg.post_length_ms:g} ms; late window from {cfg.late_ms[0]:g} ms; each post window is paired with an equal-length slice at the end of the {cfg.baseline_ms[0]:g}..{cfg.baseline_ms[1]:g} ms baseline"
    )
    epoch_desc = f"{cfg.epoch_ms[0]:g} to {cfg.epoch_ms[1]:g} ms (+/- {cfg.filter_pad_ms:g} ms filter pad); baseline {cfg.baseline_ms[0]:g} to {cfg.baseline_ms[1]:g} ms"

    # ---- (a) within-epoch comparison --------------------------------------------------
    rows = []
    for (run_id, channel, band, window), group in (paired.groupby(["run_id", "channel", "band", "window"]) if not paired.empty else []):
        used = group[group["retained"]]
        base = used["base_value"].to_numpy(dtype=float)
        post = used["post_value"].to_numpy(dtype=float)
        lo, hi = bootstrap_paired_db_ci(base, post, cfg.bootstrap_n, rng)
        crow = cond.loc[(run_id, channel)] if (run_id, channel) in cond.index else None
        rows.append(
            {
                "run_id": run_id, "block": run_info.loc[run_id, "block"], "amplitude_uA": run_info.loc[run_id, "amplitude_uA_data"], "phase_us": run_info.loc[run_id, "phase_us_data"],
                "channel": channel, "band": band, "window": window,
                "n_pairs": int(len(used)), "n_dropped": int(len(group) - len(used)),
                "median_db": float(np.nanmedian(used["db"])) if len(used) else float("nan"),
                "mean_db": float(np.nanmean(used["db"])) if len(used) else float("nan"),
                "ci_low_db": lo, "ci_high_db": hi,
                "post_window_ms": (f"{crow['post_start_ms']:.1f} to {crow['post_end_ms']:.1f}" if (crow is not None and window == "early") else f"{cfg.late_ms[0]:g} to {cfg.late_ms[1]:g}"),
                "cycles_in_window": float(used["cycles_in_window"].iloc[0]) if "cycles_in_window" in used and len(used) else float("nan"),
                "note": "within-epoch: sub-second effects only; baseline of pulse n is the tail of pulse n-1 (1 Hz confound)",
            }
        )
    within = pd.DataFrame(rows)
    result.tables["comparisons_within_epoch"] = within

    # ---- (b) block vs no-stim baseline ---------------------------------------------------
    rows = []
    if not baseline_pow.empty:
        base_late = baseline_pow[baseline_pow["window"] == "late"]
        for (run_id, channel, band), group in power[(power["window"] == "late")].groupby(["run_id", "channel", "band"]):
            used = group[group["retained"]]["power_uV2"].to_numpy(dtype=float)
            ref_all = base_late[(base_late["channel"] == channel) & (base_late["band"] == band)]["power_uV2"].to_numpy(dtype=float)
            if used.size == 0 or ref_all.size == 0:
                continue
            k = min(used.size, ref_all.size)
            ref = rng.choice(ref_all, size=k, replace=False) if ref_all.size > k else ref_all
            change = 10.0 * np.log10(np.mean(used) / np.mean(ref)) if np.mean(ref) > 0 else float("nan")
            lo, hi = bootstrap_independent_db_ci(ref, used, cfg.bootstrap_n, rng)
            rows.append(
                {
                    "run_id": run_id, "block": run_info.loc[run_id, "block"], "amplitude_uA": run_info.loc[run_id, "amplitude_uA_data"], "phase_us": run_info.loc[run_id, "phase_us_data"],
                    "channel": channel, "band": band, "n_block_segments": int(used.size), "n_baseline_segments": int(len(ref)),
                    "block_power_uV2": float(np.mean(used)), "baseline_power_uV2": float(np.mean(ref)),
                    "change_db": float(change), "ci_low_db": lo, "ci_high_db": hi,
                    "segment_window_ms": f"{cfg.late_ms[0]:g} to {cfg.late_ms[1]:g} after each pulse (clean inter-pulse) vs the same window after pseudo-onsets in the no-stim run",
                    "baseline_run_id": result.baseline_run_id,
                }
            )
    result.tables["comparisons_block_vs_baseline"] = pd.DataFrame(rows)

    # ---- (c) across amplitude ------------------------------------------------------------
    rows = []
    early_amp = amplitude[(amplitude["window"] == "early") & (amplitude["retained"]) & (amplitude["block"] == "block1")]
    early_db = paired[(paired["window"] == "early") & (paired["retained"]) & (paired["block"] == "block1")] if not paired.empty else pd.DataFrame()
    for (channel, amp), group in early_amp.groupby(["channel", "amplitude_uA"]):
        med, lo, hi, n = median_with_ci(group["rms_uV"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
        rows.append({"channel": channel, "metric": "early_rms_uV", "band": "", "amplitude_uA": amp, "n": n, "median": med, "ci_low": lo, "ci_high": hi, "distance_um": float(group["distance_um"].iloc[0]), "impedance_kohm": float(group["impedance_kohm"].iloc[0])})
    for (channel, band, amp), group in (early_db.groupby(["channel", "band", "amplitude_uA"]) if not early_db.empty else []):
        med, lo, hi, n = median_with_ci(group["db"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
        rows.append({"channel": channel, "metric": "early_db", "band": band, "amplitude_uA": amp, "n": n, "median": med, "ci_low": lo, "ci_high": hi, "distance_um": float(group["distance_um"].iloc[0]), "impedance_kohm": float(group["impedance_kohm"].iloc[0])})
    across = pd.DataFrame(rows)
    result.tables["comparisons_across_amplitude"] = across

    # ---- drift: first vs last n trials -----------------------------------------------------
    drift_rows = []
    if not paired.empty:
        for (run_id, channel, band), group in paired[paired["retained"]].groupby(["run_id", "channel", "band"]):
            window = "early" if (group["window"] == "early").any() else "late"
            sub = group[group["window"] == window]
            fl = first_vs_last(sub, "db", cfg.drift_n)
            if fl.empty:
                continue
            row = fl.iloc[0].to_dict()
            row.update({"run_id": run_id, "block": run_info.loc[run_id, "block"], "amplitude_uA": run_info.loc[run_id, "amplitude_uA_data"], "channel": channel, "band": band, "window": window})
            drift_rows.append(row)
    drift = pd.DataFrame(drift_rows)
    result.tables["drift_first_vs_last"] = drift

    # ---- 7.3 charge dependence (200 vs 300 us at the sweep amplitude) ----------------------
    rows = []
    sweep = amplitude[(amplitude["window"] == "early") & (amplitude["retained"]) & (amplitude["run_id"].map(run_info["in_block2"]).fillna(False).astype(bool))]
    for (run_id, channel), group in sweep.groupby(["run_id", "channel"]):
        med, lo, hi, n = median_with_ci(group["rms_uV"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
        rows.append({"run_id": run_id, "amplitude_uA": run_info.loc[run_id, "amplitude_uA_data"], "phase_us": run_info.loc[run_id, "phase_us_data"], "charge_nC_per_phase": run_info.loc[run_id, "charge_nC_per_phase"], "channel": channel, "metric": "early_rms_uV", "n": n, "median": med, "ci_low": lo, "ci_high": hi, "note": "two points are not a strength-duration curve; preliminary observation only"})
    result.tables["charge_dependence"] = pd.DataFrame(rows)

    # ---- 7.4 spatial decay ---------------------------------------------------------------------
    rows = []
    spatial = across[across["metric"] == "early_rms_uV"] if not across.empty else pd.DataFrame()
    if not spatial.empty:
        for amp, group in spatial.groupby("amplitude_uA"):
            rho = spearman_ci(group["distance_um"].to_numpy(dtype=float), group["median"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
            for _, r in group.iterrows():
                rows.append({**r.to_dict(), "spearman_rho_vs_distance": rho["rho"], "spearman_ci_low": rho["ci_low"], "spearman_ci_high": rho["ci_high"], "spearman_p": rho["p_value"]})
    spatial_table = pd.DataFrame(rows)
    result.tables["spatial_decay"] = spatial_table

    # ---- 7.5 compliance characterisation --------------------------------------------------------
    comp = result.validation[result.validation["n_detected"] > 0][["run_id", "block", "amplitude_uA_data", "phase_us_data", "charge_nC_per_phase", "n_commanded_total", "n_detected", "compliance_flag", "compliance_bit_seen", "included"]].copy()
    comp = comp.rename(columns={"amplitude_uA_data": "amplitude_uA", "phase_us_data": "phase_us"})
    comp["delivered_fraction"] = comp["n_detected"] / comp["n_commanded_total"]
    result.tables["compliance_characterisation"] = comp

    # ---- 7.2 amplitude-response models (Block 1, retained early trials) --------------------------
    wide = _wide_metrics(amplitude, power, cfg)
    metrics = ["early_rms_uV"] + [f"early_power_{band.name}_uV2" for band in cfg.bands]
    fits = []
    t0 = time.time()
    _log(result, progress, f"secondary: amplitude-response fits ({len(metrics)} metrics x {wide['channel'].nunique() if not wide.empty else 0} channels, bootstrap {cfg.bootstrap_n})")
    for metric in metrics:
        if metric in wide:
            fits.append(fit_amplitude_response(wide[wide["block"] == "block1"], metric, bootstrap_n=cfg.bootstrap_n, rng=rng))
    models_table = pd.concat([f for f in fits if not f.empty], ignore_index=True) if any(not f.empty for f in fits) else pd.DataFrame()
    result.tables["models_amplitude_response"] = models_table
    _log(result, progress, f"secondary: fits done in {time.time() - t0:.1f} s")

    # ---- cross-channel model: log response ~ log amplitude + log impedance + (1|channel) -------------
    rows = []
    if not wide.empty:
        w1 = wide[(wide["block"] == "block1")].copy()
        w1 = w1[(w1["amplitude_uA"] > 0) & (w1["impedance_kohm"] > 0)]
        if not w1.empty:
            w1["log_amplitude"] = np.log10(w1["amplitude_uA"])
            w1["log_impedance"] = np.log10(w1["impedance_kohm"])
            for metric in metrics:
                if metric not in w1:
                    continue
                sub = w1[w1[metric] > 0].copy()
                sub["log_response"] = np.log10(sub[metric])
                fit = fit_cross_channel_model(sub, "log_response", ("log_amplitude", "log_impedance"), "channel", use_statsmodels=cfg.use_statsmodels, bootstrap_n=min(cfg.bootstrap_n, 300), rng=rng)
                if not fit.empty:
                    fit.insert(0, "metric", metric)
                    rows.append(fit)
    channel_model = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not channel_model.empty:
        channel_model["note"] = "impedance is nearly collinear with channel (one value per channel per run); coefficient weakly identified"
    result.tables["channel_model"] = channel_model

    # ---- log-normal checks -----------------------------------------------------------------------
    checks = []
    qq_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    candidates: dict[str, np.ndarray] = {
        "recovery_ms": trials_recovery["recovery_ms"].to_numpy(dtype=float),
        "impedance_kohm": result.tables["table02_impedance_per_run"]["impedance_kohm"].to_numpy(dtype=float) if "table02_impedance_per_run" in result.tables else np.zeros(0),
    }
    if not wide.empty:
        for metric in metrics:
            if metric in wide:
                candidates[metric] = wide[metric].to_numpy(dtype=float)
    for name, values in candidates.items():
        checks.append(lognormal_check(values, name))
        if len(qq_data) < 8:
            qq_data[name] = (*qq_points(values, log=False), *qq_points(values, log=True))
    result.tables["lognormal_checks"] = pd.DataFrame(checks)

    # ---- noise floor ------------------------------------------------------------------------------
    noise = noise_floor_table(baseline_amp, baseline_pow, baseline_impedance) if not baseline_amp.empty else pd.DataFrame()
    result.tables["noise_floor"] = noise

    # ---- shuffle control ---------------------------------------------------------------------------
    shuffle = pd.concat(shuffle_frames, ignore_index=True) if shuffle_frames else pd.DataFrame()
    shuffle_summary_rows = []
    if not shuffle.empty:
        shuffle = annotate(shuffle) if "run_id" in shuffle else shuffle
        for (mode, run_id, channel, band, window), group in shuffle.groupby(["mode", "run_id", "channel", "band", "window"]):
            base = group["base_value"].to_numpy(dtype=float)
            post = group["post_value"].to_numpy(dtype=float)
            lo, hi = bootstrap_paired_db_ci(base, post, cfg.bootstrap_n, rng)
            shuffle_summary_rows.append({"mode": mode, "run_id": run_id, "channel": channel, "band": band, "window": window, "n_fake_retained": int(len(group)), "median_db": float(np.nanmedian(group["db"])), "ci_low_db": lo, "ci_high_db": hi, "ci_contains_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi)})
    shuffle_summary = pd.DataFrame(shuffle_summary_rows)
    result.tables["shuffle_control"] = shuffle_summary
    result.tables["shuffle_trials"] = shuffle

    # ---- figures ---------------------------------------------------------------------------------
    n_early_retained = int(paired[(paired["window"] == "early") & (paired["retained"])]["event_number"].groupby([paired["run_id"], paired["channel"]]).count().sum()) if not paired.empty else 0
    n_early_total = int(len(paired[(paired["window"] == "early")].groupby(["run_id", "channel", "event_number"]))) if not paired.empty else 0
    early_block1 = paired[(paired["window"] == "early") & (paired["retained"]) & (paired["block"] == "block1")] if not paired.empty else pd.DataFrame()
    n_ret_b1 = int(len(early_block1.groupby(["run_id", "channel", "event_number"]))) if not early_block1.empty else 0
    n_all_b1 = int(len(paired[(paired["window"] == "early") & (paired["block"] == "block1")].groupby(["run_id", "channel", "event_number"]))) if not paired.empty else 0
    ctx4 = CaptionContext(
        n_retained=n_ret_b1, n_rejected=n_all_b1 - n_ret_b1 + n_dropped_epochs,
        reject_reasons={"recovery_after_post_start_or_censored": n_all_b1 - n_ret_b1, "epoch_at_recording_edge_or_gap": n_dropped_epochs},
        blank_desc=blank_desc, filter_desc=filter_desc, epoch_desc=epoch_desc,
        note="Comparison (a), within-epoch: valid only for sub-second effects because the -500..-50 ms baseline of pulse n is the +500..+950 ms tail of pulse n-1. Delta/theta have < 1 cycle in a 300 ms window (see cycles_in_window).",
    )
    summary_db = across[across["metric"] == "early_db"] if not across.empty else pd.DataFrame()
    _add_figure(result, "fig04_bandpower_vs_amplitude", "Band power change vs current (early vs baseline), per channel", plot_bandpower_vs_amplitude(early_block1, cfg, ctx4, channel_info=channel_info, summary=summary_db), ctx4)
    _add_figure(result, "fig04b_bandpower_heatmap", "Median band-power change heatmap (channel x current)", plot_bandpower_heatmap(summary_db, cfg, ctx4), ctx4)
    wide_b1 = wide[wide["block"] == "block1"] if not wide.empty else wide
    ctx5 = ctx4.with_note("Linear through the origin winning (or within 2 AIC of the best) is the artifact signature; a sigmoid with a bounded I50 is the candidate neural response. Fits per channel; impedance covariate in channel_model.csv.")
    _add_figure(result, "fig05_linear_vs_sigmoid", "Amplitude-response fits (linear vs sigmoid, AIC)", plot_model_fits(wide_b1, models_table, "early_rms_uV", cfg, ctx5), ctx5)
    ctx6 = ctx4.with_note("Graded decay is expected from volume conduction; a response on exactly one channel is suspicious. Contacts also sample different A-P levels along the ACA.")
    _add_figure(result, "fig06_spatial_decay", "Spatial decay of the early-window RMS", plot_spatial_decay(spatial_table if not spatial_table.empty else spatial, cfg, ctx6), ctx6)
    ctx7 = CaptionContext(n_retained=int(comp["included"].sum()) if not comp.empty else 0, n_rejected=int((~comp["included"]).sum()) if not comp.empty else 0, reject_reasons={"excluded_runs": int((~comp["included"]).sum()) if not comp.empty else 0}, blank_desc="n/a", filter_desc="n/a (pulse counts from the stim marker)", note="Delivered/commanded pulses from the stim marker vs commanded charge per phase; excluded runs included here on purpose (methods result).")
    _add_figure(result, "fig07_compliance_pulses_vs_charge", "Compliance: delivered pulses vs commanded charge", plot_compliance(comp, cfg, ctx7), ctx7)
    # figure 8: shuffle control version of figure 4
    shuffle_mode = ""
    if not shuffle.empty:
        within_run = shuffle[(shuffle["mode"] == "within_run") & (shuffle["window"] == "early")]
        per_cond = within_run.groupby(["run_id", "channel"])["event_number"].nunique() if not within_run.empty else pd.Series(dtype=int)
        n_within = int(within_run.groupby(["run_id", "channel", "event_number"]).ngroups) if not within_run.empty else 0
        if not within_run.empty and (per_cond >= cfg.min_trials).all():
            shuffle_mode = "within_run"
            fig8_trials = within_run[within_run["block"] == "block1"] if "block" in within_run else within_run
            n_used = n_within
        else:
            shuffle_mode = "baseline_run"
            fig8_trials = shuffle[shuffle["mode"] == "baseline_run"].copy()
            n_used = int(fig8_trials.groupby(["run_id", "channel", "event_number"]).ngroups) if not fig8_trials.empty else 0
        how = (
            "drawn from clean inter-pulse intervals of the same run, real pulses inside the fake epoch blanked"
            if shuffle_mode == "within_run"
            else f"placed 1 s apart in the no-stim baseline run (within-run fake epochs starve at a 1 s IPI: {n_within} survived, {shuffle_dropped} overlapped a real pulse); x position is arbitrary"
        )
        ctx8 = CaptionContext(
            n_retained=n_used, n_rejected=shuffle_dropped if shuffle_mode == "within_run" else 0,
            reject_reasons={"fake_epoch_window_overlaps_real_pulse": shuffle_dropped} if shuffle_mode == "within_run" else {},
            blank_desc=blank_desc, filter_desc=filter_desc, epoch_desc=epoch_desc,
            note=f"Shuffle control ({shuffle_mode}): fake event times {how} and pushed through the identical pipeline (paired dB, generic early window {GENERIC_EARLY_MS[0]:g}-{GENERIC_EARLY_MS[1]:g} ms when on the no-stim run). Any effect surviving here means the pipeline is broken.",
        )
        if shuffle_mode == "baseline_run" and not fig8_trials.empty:
            fig8_trials = fig8_trials.assign(amplitude_uA=cfg.lim_amplitude_uA[0] * 1.5)
        _add_figure(result, "fig08_shuffle_control_bandpower_vs_amplitude", f"Shuffle control ({shuffle_mode}) of figure 4", plot_bandpower_vs_amplitude(fig8_trials, cfg, ctx8, channel_info=channel_info, title=f"SHUFFLE CONTROL ({shuffle_mode}): band power change for fake events"), ctx8)
    ctx_s1 = CaptionContext(n_retained=int(len(trials_recovery)), n_rejected=0, blank_desc=blank_desc, filter_desc=filter_desc, note="Model on the log scale when the log column is closer to normal (lognormal_checks.csv).")
    _add_figure(result, "figS1_lognormal_qq", "Log-normality QQ plots", plot_qq_grid(qq_data, cfg, ctx_s1), ctx_s1)
    ctx_s2 = ctx4.with_note(f"Within-block drift: median dB of the last {cfg.drift_n} trials minus the first {cfg.drift_n} (early window, late when early is impossible).")
    _add_figure(result, "figS2_within_block_drift", "Within-block drift (first vs last trials)", plot_drift(drift, cfg, ctx_s2, value_first="db_first_median", value_last="db_last_median", ylabel="last - first median dB", title="Within-block drift per run: last minus first trials (band power dB)") if not drift.empty else plot_drift(drift, cfg, ctx_s2, value_first="db_first_median", value_last="db_last_median", ylabel="dB", title="Within-block drift"), ctx_s2)
    ctx_s3 = CaptionContext(n_retained=int(result.tables["table02_impedance_per_run"]["impedance_kohm"].notna().sum()) if "table02_impedance_per_run" in result.tables else 0, blank_desc="n/a", filter_desc="n/a", note="Impedance is the RHS header value stored at recording start (last measurement in Intan RHX).")
    _add_figure(result, "figS3_impedance_drift", "Impedance per channel per run", plot_impedance_drift(result.tables.get("table02_impedance_per_run", pd.DataFrame()), cfg, ctx_s3), ctx_s3)
    ctx_s4 = CaptionContext(n_retained=int(len(baseline_amp[baseline_amp["window"] == "late"])) if not baseline_amp.empty else 0, blank_desc="none (no-stim run)", filter_desc=filter_desc, note="Noise floor = broadband RMS in the late window of pseudo-epochs of the no-stim run; thermal noise scales with sqrt(R).")
    _add_figure(result, "figS4_noise_floor", "Noise floor per channel from the no-stim run", plot_noise_floor(noise, cfg, ctx_s4), ctx_s4)

    # ---- metadata ---------------------------------------------------------------------------------
    result.metadata["filter_designs"] = {fs: {name: d.description for name, d in designs.items()} for fs, designs in design_cache.items()}
    result.metadata["shuffle_mode_used"] = shuffle_mode
    result.metadata["has_statsmodels"] = HAS_STATSMODELS
    result.metadata["secondary_elapsed_s"] = time.time() - started
    _log(result, progress, f"secondary stage done in {time.time() - started:.1f} s ({len(result.figures)} figures, {len(result.tables)} tables)")


def _wide_metrics(amplitude: pd.DataFrame, power: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """One row per (run, channel, event) with early-window metrics for retained trials."""
    if amplitude.empty:
        return pd.DataFrame()
    early = amplitude[(amplitude["window"] == "early") & (amplitude["retained"])]
    if early.empty:
        return pd.DataFrame()
    wide = early[["run_id", "block", "amplitude_uA", "phase_us", "channel", "event_number", "impedance_kohm", "distance_um", "rms_uV", "peak_abs_uV"]].rename(columns={"rms_uV": "early_rms_uV", "peak_abs_uV": "early_peak_abs_uV"})
    if not power.empty:
        pe = power[(power["window"] == "early") & (power["retained"])]
        for band in cfg.bands:
            col = pe[pe["band"] == band.name][["run_id", "channel", "event_number", "power_uV2"]].rename(columns={"power_uV2": f"early_power_{band.name}_uV2"})
            wide = wide.merge(col, on=["run_id", "channel", "event_number"], how="left")
    return wide


def _channel_info(result) -> dict[str, tuple[float, float]]:
    table = result.tables.get("table02_impedance_per_run")
    info: dict[str, tuple[float, float]] = {}
    if table is None or table.empty:
        return info
    for channel, group in table.groupby("channel"):
        info[channel] = (float(group["impedance_kohm"].median()), float(group["distance_um"].median()))
    return info


__all__ = ["run_secondary_stage"]
