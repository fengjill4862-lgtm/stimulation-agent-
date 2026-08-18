"""Run the whole filter diagnosis (Tests A-F) over the live and post-mortem sessions.

    /usr/local/bin/python3 -m filter_diag.run_all [--live DIR] [--dead DIR] [--out DIR]
        [--runs SUBSTR ...] [--fast] [--dry-run] [--no-f2]

Nothing is written until the end (atomic); ``--dry-run`` computes and writes nothing.
The verdict line is printed and stored in ``verdict.txt`` / ``metadata.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rhs_files import atomic_write_all
from stim_analysis.config import AnalysisConfig, config_to_dict
from stim_analysis.figures import CaptionContext, build_caption, figure_to_png_bytes
from stim_analysis.load_rhs import load_run
from filter_diag import __version__
from filter_diag.common import (
    DEFAULT_DEAD,
    DEFAULT_LIVE,
    DEFAULT_OUTPUT,
    DiagRun,
    SessionSelection,
    config_for_diag,
    dsp_k_from_cutoff,
    dsp_tau_s,
    load_diag_run,
    select_session,
    verify_dsp_model,
)
from filter_diag.filter_diag_A import settings_summary, settings_table
from filter_diag.filter_diag_B import figure_tau, fit_run_tails, invariance_model, tau_summary
from filter_diag.filter_diag_C import figure_log_law, figure_synthetic_vs_real, synthetic_summary
from filter_diag.filter_diag_D import figure_live_dead, live_vs_dead
from filter_diag.filter_diag_E import decompose, decomposition_summary, figure_decomposition
from filter_diag.filter_diag_F import before_after_table, f1_subtract, f2_inverse, figure_examples
from filter_diag.synthetic_step import run_test_c


@dataclass
class DiagResult:
    output_dir: Path
    cfg: AnalysisConfig
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    verdict: str = ""

    def outputs(self) -> list[tuple[Path, bytes | str]]:
        items: list[tuple[Path, bytes | str]] = []
        for stem, frame in self.tables.items():
            items.append((self.output_dir / f"{stem}.csv", frame.to_csv(index=False)))
        for stem, png in self.figures.items():
            items.append((self.output_dir / f"{stem}.png", png))
        items.append((self.output_dir / "figures_index.csv", pd.DataFrame([{"file": f"{s}.png", "caption": self.captions.get(s, "")} for s in self.figures]).to_csv(index=False)))
        items.append((self.output_dir / "metadata.json", json.dumps(self.metadata, indent=2, default=_json_default)))
        items.append((self.output_dir / "verdict.txt", self.verdict + "\n"))
        items.append((self.output_dir / "run_log.txt", "\n".join(self.log) + "\n"))
        return items


def _json_default(value):
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


def _log(result: DiagResult, message: str, quiet: bool) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {message}"
    result.log.append(line)
    if not quiet:
        print(message, flush=True)


def _add_figure(result: DiagResult, stem: str, fig, ctx: CaptionContext) -> None:
    result.figures[stem] = figure_to_png_bytes(fig, dpi=result.cfg.dpi)
    result.captions[stem] = build_caption(ctx)


def run_diagnosis(
    *,
    live: Path = DEFAULT_LIVE,
    dead: Path | None = DEFAULT_DEAD,
    output_dir: Path = DEFAULT_OUTPUT,
    run_filter: list[str] | None = None,
    fast: bool = False,
    do_f2: bool = True,
    quiet: bool = False,
    cfg: AnalysisConfig | None = None,
) -> DiagResult:
    started = time.time()
    cfg = cfg or config_for_diag()
    if fast:
        cfg = AnalysisConfig(**{**config_to_dict_plain(cfg), "bootstrap_n": 200})
    rng = np.random.default_rng(cfg.seed)
    result = DiagResult(output_dir=Path(output_dir), cfg=cfg)
    _log(result, f"filter diagnosis v{__version__}: live={live} dead={dead}", quiet)
    dsp_check = verify_dsp_model(30000.0, 12)
    result.metadata["dsp_model_check"] = dsp_check
    _log(result, f"DSP model check: cutoff {dsp_check['cutoff_hz']:.4f} Hz, tau {dsp_check['tau_ms_step_fit']:.2f} ms (expected {dsp_check['tau_ms_expected']:.2f}), inverse error {dsp_check['inverse_max_abs_error_uV']:.1e} uV", quiet)

    # ---- select runs ---------------------------------------------------------------------
    sessions: list[tuple[str, Path]] = [("live", Path(live))]
    if dead is not None:
        sessions.append(("dead", Path(dead)))
    selections: list[SessionSelection] = []
    for name, parent in sessions:
        sel = select_session(parent, name, cfg, progress=(None if quiet else lambda m: print(m, flush=True)))
        if run_filter:
            sel.included = [v for v in sel.included if any(tok.lower() in (v.run_id + " " + Path(v.run_folder).name).lower() for tok in run_filter)]
        _log(result, sel.summary(), quiet)
        selections.append(sel)
    result.metadata["open_channels_excluded"] = {s.session: {c: z for c, z in s.open_channels.items()} for s in selections}

    # ---- Test A -----------------------------------------------------------------------------
    table_a = settings_table(selections)
    summary_a = settings_summary(table_a)
    result.tables["A_filter_settings_per_run"] = table_a
    result.metadata["test_A"] = summary_a
    _log(result, f"Test A: {summary_a['natural_experiment']}; DSP tau {summary_a['tau_dsp_ms']:.1f} ms; analog lower bandwidth {summary_a['analog_lower_bandwidth_hz']} Hz", quiet)
    tau_dsp_ms = float(summary_a["tau_dsp_ms"]) if np.isfinite(summary_a["tau_dsp_ms"]) else dsp_tau_s(30000.0, 12) * 1e3

    # ---- one pass over the included runs: B, E rows, F ---------------------------------------
    fits_frames: list[pd.DataFrame] = []
    trials_frames: list[pd.DataFrame] = []
    f1_frames: list[pd.DataFrame] = []
    f2_frames: list[pd.DataFrame] = []
    examples: dict[str, dict] = {}
    for sel in selections:
        for v in sel.included:
            _log(result, f"{sel.session}: {Path(v.run_folder).name}", quiet)
            run = load_diag_run(sel, v, cfg, keep_data=do_f2)
            fits = fit_run_tails(run, cfg)
            fits_frames.append(fits)
            trials_frames.append(run.trials.assign(is_stim_contact=run.trials["channel"] == run.stim_channel))
            key = f"{sel.session}:{v.run_id}"
            want_example = v.amplitude_uA_data in (10.0, 20.0, 50.0, 100.0, 250.0)
            f1_frames.append(f1_subtract(run, fits, cfg, examples=examples if want_example else None, example_key=key))
            if do_f2 and run.record.data is not None:
                data = run.record.data
                idx = [data.channel_index(c) for c in run.channels]
                k = dsp_k_from_cutoff(data.sample_rate_hz, run.record.header.actual_dsp_cutoff_hz)
                f2_frames.append(f2_inverse(run, data.raw_uV[idx], run.channels, run.record.events.onset_samples, run.record.events.event_numbers, data.timestamps, cfg, k=k, examples=examples if want_example else None, example_key=key))
                run.record.release_data()
            run.release()
    if not fits_frames:
        result.verdict = "no included runs -- nothing to diagnose"
        return result
    fits = pd.concat(fits_frames, ignore_index=True)
    trials = pd.concat(trials_frames, ignore_index=True)
    trials["amplitude_uA"] = trials["amplitude_uA"].astype(float)
    result.tables["B_tail_fits_per_epoch"] = fits
    result.tables["recovery_trials"] = trials

    # ---- Test B -----------------------------------------------------------------------------
    tau_tab = tau_summary(fits, cfg, rng, tau_dsp_ms)
    inv = invariance_model(fits)
    result.tables["B_tau_summary"] = tau_tab
    result.tables["B_invariance_model"] = inv
    pooled = tau_tab[tau_tab["grouping"] == "pooled"].iloc[0]
    good = fits[fits["fit_ok"] & ~fits["is_stim_contact"]]
    same_sign_frac = float((good["tail_sign"] > 0).mean()) if len(good) else float("nan")
    reject_rate = float((~fits[~fits["is_stim_contact"]]["fit_ok"]).mean()) if len(fits) else float("nan")
    _log(result, f"Test B: median tau_fit {pooled['median_tau_ms']:.1f} ms [{pooled['ci_low']:.1f}, {pooled['ci_high']:.1f}] vs DSP {tau_dsp_ms:.1f} ms (ratio {pooled['ratio_to_tau_dsp']:.2f}); same-sign tails {same_sign_frac:.0%}; fits rejected {reject_rate:.0%}", quiet)
    ctx_b = CaptionContext(n_retained=int(len(good)), n_rejected=int(len(fits) - len(good)), reject_reasons={"r2_below_0.9_or_stim_contact": int(len(fits) - len(good))}, blank_desc="none (raw recorded epochs)", filter_desc="none offline; hardware analog HP 0.0945 Hz + DSP HP 1.166 Hz (k=12) as recorded", epoch_desc="fit window rail exit + 2 ms to +800 ms, robust soft-L1 least squares, decimated x10", note=f"Included runs: clean single-pulse runs only (compliance-flagged, paired and no-stim runs excluded); A-031 excluded; stim contacts excluded from the pooled statistics. Predicted DSP tau = {tau_dsp_ms:.1f} ms.")
    _add_figure(result, "fig1_tau_fit_distributions", figure_tau(fits, tau_dsp_ms, cfg, ctx_b), ctx_b)

    # ---- Test C -----------------------------------------------------------------------------
    _log(result, "Test C: synthetic sweeps", quiet)
    c = run_test_c(cfg, n_trials=8 if fast else 20, seed=cfg.seed, progress=None)
    synth = c["table"]
    result.tables["C_synthetic_recovery_per_trial"] = synth
    result.tables["C_synthetic_summary"] = synthetic_summary(synth)
    real_recovery = trials[~trials["is_stim_contact"]][["session", "recovery_ms", "censored", "rail_ms"]]
    ctx_c = CaptionContext(n_retained=int(len(synth)), n_rejected=0, blank_desc="none", filter_desc="synthetic chain: analog HP 0.0945 Hz -> LP 7.6 kHz -> ADC clip +/-6389.6 uV -> Intan DSP HP (k=12 at 30 kHz, k=11 at 20 kHz) -> clamp; rail modes: freeze (DSP mean held while saturated), track (mean keeps updating), spec (DSP before clip)", epoch_desc="same padded epoch axis and the unchanged compute_recovery (threshold max(3 SD, 100 uV), 20 ms quiet run)", note="Step families: rectangular step (duration 0.2-30 ms, amplitude 400 uV to 500 mV) and step + same-sign exponential input tail (20 ms to flat).")
    _add_figure(result, "fig2_synthetic_vs_real_recovery", figure_synthetic_vs_real(synth, real_recovery, cfg, tau_dsp_ms, ctx_c, examples=c["examples"]), ctx_c)
    _add_figure(result, "fig3_synthetic_recovery_vs_amplitude", figure_log_law(synth, cfg, tau_dsp_ms, ctx_c), ctx_c)
    step_rail = synth[(synth["family"] == "step") & (synth["rail_mode"] == "freeze") & (synth["sample_rate_hz"] == 30000.0) & synth["clipped_input"]]
    synth_range = (float(step_rail["recovery_ms"].quantile(0.05)), float(step_rail["recovery_ms"].quantile(0.95))) if len(step_rail) else (float("nan"), float("nan"))
    obs_med = float(np.median(real_recovery["recovery_ms"]))
    obs_in_synth = bool(synth_range[0] <= obs_med <= synth_range[1]) if np.isfinite(synth_range[0]) else False
    _log(result, f"Test C: synthetic rail-level recovery 5-95% {synth_range[0]:.0f}-{synth_range[1]:.0f} ms (physical order); observed median {obs_med:.0f} ms -> {'inside' if obs_in_synth else 'outside'} the synthetic range", quiet)

    # ---- Test D -----------------------------------------------------------------------------
    have_dead = any(s.session == "dead" for s in selections) and (fits["session"] == "dead").any()
    if have_dead:
        table_d = live_vs_dead(fits, cfg, rng)
        result.tables["D_live_vs_dead"] = table_d
        ctx_d = CaptionContext(n_retained=int(len(good)), n_rejected=0, blank_desc="none", filter_desc="none offline (recorded data)", note="Matched current and phase width, single pulses; unpaired (different animals; impedances differ, see table D).")
        _add_figure(result, "fig4_live_vs_dead", figure_live_dead(table_d, fits, cfg, ctx_d), ctx_d)
        rec = table_d[table_d["metric"] == "recovery_ms"]
        _log(result, f"Test D: {len(rec)} matched conditions; recovery dead/live median ratio {rec['ratio_dead_over_live'].median():.2f}; indistinguishable (95% CI covers 0) in {int(rec['indistinguishable_95ci'].sum())}/{len(rec)}", quiet)
    else:
        table_d = pd.DataFrame()
        _log(result, "Test D: no post-mortem runs available", quiet)

    # ---- Test E -----------------------------------------------------------------------------
    rows_e = decompose(fits, tau_dsp_ms)
    summ_e = decomposition_summary(rows_e[~rows_e["is_stim_contact"]])
    result.tables["E_decomposition_per_epoch"] = rows_e
    result.tables["E_decomposition_summary"] = summ_e
    ctx_e = CaptionContext(n_retained=int((~rows_e["censored"] & ~rows_e["is_stim_contact"]).sum()), n_rejected=int((rows_e["censored"] | rows_e["is_stim_contact"]).sum()), reject_reasons={"censored_or_stim_contact": int((rows_e["censored"] | rows_e["is_stim_contact"]).sum())}, blank_desc="none", filter_desc="none offline", note="Predicted = rail exit + tau ln(V_ref / threshold) with the per-epoch threshold; V_ref = ADC rail for railed epochs, |peak| otherwise.")
    _add_figure(result, "fig5_predicted_vs_observed", figure_decomposition(rows_e, cfg, tau_dsp_ms, ctx_e), ctx_e)
    e_spec = summ_e[summ_e["model"] == "pred_spec_ms"].iloc[0] if not summ_e.empty else None
    if e_spec is not None:
        _log(result, f"Test E: spec model R^2 {e_spec.get('r2', float('nan')):.2f}, median residual {e_spec.get('median_resid_ms', float('nan')):.0f} ms; residual corr with current {e_spec.get('resid_corr_amplitude_uA', float('nan')):.2f}, impedance {e_spec.get('resid_corr_impedance_kohm', float('nan')):.2f}, distance {e_spec.get('resid_corr_distance_um', float('nan')):.2f}", quiet)

    # ---- Test F -----------------------------------------------------------------------------
    f1 = pd.concat([f for f in f1_frames if not f.empty], ignore_index=True) if any(not f.empty for f in f1_frames) else pd.DataFrame()
    f2 = pd.concat([f for f in f2_frames if not f.empty], ignore_index=True) if any(not f.empty for f in f2_frames) else pd.DataFrame()
    before = trials.copy()
    table_f = before_after_table(before, f1, f2, cfg, rng)
    result.tables["F_recovery_before_after"] = table_f
    if not f1.empty:
        result.tables["F1_recovery_per_epoch"] = f1
    if not f2.empty:
        result.tables["F2_recovery_per_epoch"] = f2
    ctx_f = CaptionContext(n_retained=int(len(before[~before["is_stim_contact"]])), n_rejected=0, blank_desc="none", filter_desc="F1: fitted exponential subtracted from rail exit + 2 ms; F2: exact inverse of the DSP recursion on the continuous channel + causal 0.1 Hz HP, re-epoched", note="F2 is exact only for epochs that never touched the ADC rail; railed epochs are reported separately (lower bound).")
    _add_figure(result, "fig6_example_epochs_filter_removed", figure_examples(examples, cfg, ctx_f), ctx_f)
    tf = table_f[~table_f["is_stim_contact"]]
    f1_frac = float(np.nanmedian(tf["f1_fraction_of_before"])) if "f1_fraction_of_before" in tf else float("nan")
    f2_frac = float(np.nanmedian(tf["f2_fraction_of_before"])) if "f2_fraction_of_before" in tf else float("nan")
    f2_nonrailed = float(np.nanmedian(tf["f2_nonrailed_median_ms"])) if "f2_nonrailed_median_ms" in tf else float("nan")
    _log(result, f"Test F: median recovery after F1 = {f1_frac:.0%} of before; after F2 = {f2_frac:.0%} of before (non-railed epochs median {f2_nonrailed:.0f} ms)", quiet)

    # ---- verdict ---------------------------------------------------------------------------
    verdict, evidence = _verdict(pooled, tau_dsp_ms, inv, same_sign_frac, obs_in_synth, synth_range, obs_med, table_d, e_spec, f1_frac, f2_frac, f2_nonrailed, tf, tau_tab=tau_tab)
    result.verdict = verdict
    result.metadata.update({
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "config": config_to_dict(cfg),
        "sessions": {s.session: {"parent": str(s.parent), "included": [Path(v.run_folder).name for v in s.included], "excluded": {Path(v.run_folder).name: (v.exclusion_reason or v.block) for v in s.excluded}} for s in selections},
        "tau_dsp_ms": tau_dsp_ms,
        "test_B": {"median_tau_fit_ms": float(pooled["median_tau_ms"]), "ci": [float(pooled["ci_low"]), float(pooled["ci_high"])], "ratio_to_dsp": float(pooled["ratio_to_tau_dsp"]), "same_sign_fraction": same_sign_frac, "reject_rate": reject_rate},
        "test_C": {"synthetic_rail_level_recovery_5_95_ms": synth_range, "observed_median_ms": obs_med, "observed_inside_synthetic": obs_in_synth},
        "test_E": (e_spec.to_dict() if e_spec is not None else None),
        "test_F": {"f1_fraction_of_before": f1_frac, "f2_fraction_of_before": f2_frac, "f2_nonrailed_median_ms": f2_nonrailed},
        "evidence": evidence,
        "verdict": verdict,
        "elapsed_s": time.time() - started,
    })
    _log(result, "VERDICT: " + verdict, quiet)
    return result


def config_to_dict_plain(cfg: AnalysisConfig) -> dict:
    import dataclasses

    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _verdict(pooled, tau_dsp_ms, inv, same_sign_frac, obs_in_synth, synth_range, obs_med, table_d, e_spec, f1_frac, f2_frac, f2_nonrailed, tf, tau_tab=None) -> tuple[str, dict]:
    ratio = float(pooled["ratio_to_tau_dsp"])
    tau_match = bool(np.isfinite(ratio) and abs(ratio - 1.0) <= 0.2)
    shifts = {}
    if not inv.empty and "term" in inv and "pct_shift_over_observed_range" in inv:
        ols = inv[inv["method"] == "ols_hc1"] if "method" in inv else inv
        for term in ("log_current", "log_impedance", "distance_mm"):
            r = ols[ols["term"] == term]
            if not r.empty and np.isfinite(r["pct_shift_over_observed_range"].iloc[0]):
                shifts[term] = float(r["pct_shift_over_observed_range"].iloc[0])
    invariant_regression = all(abs(v) <= 20.0 for v in shifts.values()) if shifts else False
    # Spec decision rule: does any covariate group shift the MEDIAN tau by more than ~20% from pooled?
    group_dev = {}
    if tau_tab is not None and not tau_tab.empty:
        pooled_med = float(tau_tab[tau_tab["grouping"] == "pooled"]["median_tau_ms"].iloc[0])
        for grouping in ("session", "session_channel", "session_amplitude_uA", "impedance_tertile", "distance_um"):
            sub = tau_tab[(tau_tab["grouping"] == grouping) & (tau_tab["n_fits"] >= 10)]
            if not sub.empty and pooled_med > 0:
                group_dev[grouping] = float(np.nanmax(np.abs(sub["median_tau_ms"] / pooled_med - 1.0)) * 100.0)
    invariant = all(v <= 20.0 for v in group_dev.values()) if group_dev else invariant_regression
    live_dead_same = None
    dead_shorter = None
    dead_ratio = float("nan")
    if table_d is not None and not table_d.empty:
        rec = table_d[table_d["metric"] == "recovery_ms"]
        if len(rec):
            live_dead_same = bool(rec["indistinguishable_95ci"].mean() >= 0.5)
            dead_ratio = float(np.nanmedian(rec["ratio_dead_over_live"]))
            dead_shorter = bool(np.isfinite(dead_ratio) and dead_ratio < 0.8 and (rec["diff_ci_high"] < 0).mean() >= 0.5)
    collapse_f1 = bool(np.isfinite(f1_frac) and f1_frac <= 0.25)
    collapse_f2 = bool(np.isfinite(f2_frac) and f2_frac <= 0.25)
    collapse = collapse_f1 or collapse_f2
    dsp_signature = same_sign_frac < 0.5  # DSP/HP overshoot after a brief pulse is opposite-sign
    f2_lengthens = bool(np.isfinite(f2_frac) and f2_frac > 1.1)
    amp_dependent = bool(shifts.get("log_current") is not None and abs(shifts["log_current"]) > 20) or bool(group_dev.get("session_amplitude_uA", 0.0) > 20.0)
    z_dependent = bool(group_dev.get("impedance_tertile", 0.0) > 20.0)
    evidence = {
        "tau_fit_ratio_to_dsp": ratio, "tau_within_20pct": tau_match, "covariate_pct_shifts_over_range (regression, collinear covariates -- supplementary)": shifts, "max_group_median_deviation_pct": group_dev, "tau_invariant": invariant,
        "same_sign_tail_fraction": same_sign_frac, "opposite_sign_tail_dominates (DSP signature)": dsp_signature,
        "observed_inside_synthetic_range": obs_in_synth, "synthetic_range_ms": synth_range, "observed_median_ms": obs_med,
        "live_dead_indistinguishable": live_dead_same, "dead_over_live_recovery_ratio": dead_ratio, "dead_shorter_than_live": dead_shorter,
        "F1_fraction_of_before": f1_frac, "F2_fraction_of_before": f2_frac, "F2_nonrailed_median_ms": f2_nonrailed, "recovery_collapses_after_removal": collapse,
        "F2_lengthens_recovery (DSP shortens an analog tail)": f2_lengthens, "amplitude_dependent": amp_dependent, "impedance_dependent": z_dependent,
    }
    # Fractions: what the fitted exponential tail accounts for (whatever its origin), and what the DSP itself
    # accounts for (F2 inverse; negative = the DSP is SHORTENING the recorded tail).
    tail_frac = float(1.0 - f1_frac) if np.isfinite(f1_frac) else float("nan")
    dsp_frac = float(1.0 - f2_frac) if np.isfinite(f2_frac) else float("nan")
    evidence["exponential_tail_fraction_of_recovery (F1)"] = tail_frac
    evidence["dsp_attributable_fraction_of_recovery (F2; negative = DSP shortens)"] = dsp_frac
    if tau_match and invariant and dsp_signature and collapse_f1:
        label = "FILTER-DOMINATED"
    elif dsp_signature and tau_match and not invariant:
        label = "MIXED (DSP-shaped opposite-sign tail, but tau varies with condition)"
    elif not dsp_signature and (dead_shorter is True):
        label = "MIXED (same-sign tail; post-mortem shorter -> a biological contribution exists)"
    elif not dsp_signature:
        qual = []
        if amp_dependent:
            qual.append("current-dependent")
        if z_dependent:
            qual.append("impedance-dependent")
        if f2_lengthens:
            qual.append("DSP shortens rather than causes it")
        label = "ELECTRODE / ANALOG FRONT-END-DOMINATED (same-sign tail" + (", " + ", ".join(qual) if qual else "") + ")"
    elif z_dependent and not amp_dependent:
        label = "ELECTRODE-DOMINATED"
    else:
        label = "MIXED"
    text = (
        f"{label}: tau_fit/tau_DSP = {ratio:.2f} (within 20%: {tau_match}); max group-median deviation "
        + ", ".join(f"{k} {v:.0f}%" for k, v in group_dev.items())
        + f"; same-sign tails {same_sign_frac:.0%}; observed median {obs_med:.0f} ms vs synthetic rail-level {synth_range[0]:.0f}-{synth_range[1]:.0f} ms"
        + (f"; live vs dead: dead/live recovery {dead_ratio:.2f}, indistinguishable {live_dead_same}, dead shorter {dead_shorter}" if live_dead_same is not None else "; no post-mortem data")
        + f"; recovery after removal F1 {f1_frac:.0%} / F2 {f2_frac:.0%} of before"
        + (f"; exponential-tail fraction {tail_frac:.0%}, DSP-attributable fraction {dsp_frac:+.0%}" if np.isfinite(tail_frac) else "")
    )
    return text, evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", default=str(DEFAULT_LIVE))
    parser.add_argument("--dead", default=str(DEFAULT_DEAD), help="post-mortem session parent; 'none' to skip")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", nargs="*", default=None, help="only runs whose folder/run_id contains one of these")
    parser.add_argument("--fast", action="store_true", help="fewer synthetic trials and bootstrap resamples")
    parser.add_argument("--no-f2", action="store_true", help="skip the inverse-filter test (needs continuous data)")
    parser.add_argument("--dry-run", action="store_true", help="compute, write nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_diagnosis(live=Path(args.live), dead=None if args.dead.lower() == "none" else Path(args.dead), output_dir=Path(args.out), run_filter=args.runs, fast=args.fast, do_f2=not args.no_f2, quiet=args.quiet)
    print(result.verdict)
    if args.dry_run:
        print(f"dry run: {len(result.outputs())} files would be written to {result.output_dir}")
        return 0
    items = result.outputs()
    result.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_all(items)
    print(f"wrote {len(items)} files to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
