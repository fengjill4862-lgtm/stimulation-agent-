"""Verdict lines (one per arm) and the setting recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bw_sweep.config import SweepConfig, format_hz
from bw_sweep.load import SweepSet
from bw_sweep.stats import SlopeResult
from bw_sweep.summary import arm_table


@dataclass
class Verdict:
    lines: list[str] = field(default_factory=list)
    recommendation: str = ""
    pareto: pd.DataFrame = field(default_factory=pd.DataFrame)
    arm_A: str = ""
    arm_B: str = ""
    arm_C: str = ""

    def text(self) -> str:
        return "\n".join(self.lines + [self.recommendation])


def _slope_line(arm: str, slope: SlopeResult, meaning: dict[str, str]) -> str:
    if slope.n_runs_used < 2 or not np.isfinite(slope.slope):
        return f"Arm {arm}: slope not estimable ({slope.n_runs_used} usable runs; excluded {slope.excluded or 'none'}) -> no verdict"
    core = f"Arm {arm}: log-log slope {slope.slope:.2f} [{slope.ci_low:.2f}, {slope.ci_high:.2f}] ({slope.n_runs_used} runs, {slope.n_trials_used} trials"
    if slope.excluded:
        core += "; excluded " + ", ".join(f"{k} {v}" for k, v in slope.excluded.items())
    core += ")"
    return f"{core} -> {slope.verdict}: {meaning.get(slope.verdict, 'inconclusive')}"


def arm_c_line(table1: pd.DataFrame, per_channel: pd.DataFrame, cfg: SweepConfig) -> tuple[str, dict[str, float | None]]:
    """Per channel: the highest upper bandwidth at which >= zero_rail_fraction of trials have zero rail."""
    rows = arm_table(per_channel, "C")
    reach: dict[str, float | None] = {}
    if rows.empty:
        return "Arm C: no data", reach
    for channel, g in rows.groupby("channel", sort=True):
        g = g.sort_values("upper_hz", ascending=False)
        # walk from the widest bandwidth down; report the widest bandwidth at which the criterion holds
        first_hz = None
        for _, r in g.iterrows():
            if float(r["frac_zero_rail"]) >= cfg.zero_rail_fraction:
                first_hz = float(r["upper_hz"])
                break
        # require it to hold for every narrower bandwidth too (monotone reading)
        holds_below = all(float(r["frac_zero_rail"]) >= cfg.zero_rail_fraction for _, r in g.iterrows() if first_hz is not None and float(r["upper_hz"]) <= first_hz)
        reach[str(channel)] = first_hz if (first_hz is not None and holds_below) else (first_hz if first_hz is not None else None)
    always = [c for c, hz in reach.items() if hz is not None and hz >= 7000]
    reached = {c: hz for c, hz in reach.items() if hz is not None and hz < 7000}
    never = [c for c, hz in reach.items() if hz is None]
    parts = []
    if always:
        parts.append(f"already zero at 7500 Hz on {', '.join(always)}")
    if reached:
        parts.append("reaches zero at <= " + ", ".join(f"{format_hz(hz)} Hz on {c}" for c, hz in reached.items()))
    if never:
        parts.append(f"never reaches zero on {', '.join(never)}")
    stim_rows = table1[(table1["contacts"] == "stim") & table1["arms"].astype(str).str.split("+").apply(lambda p: "C" in p)] if not table1.empty else table1
    peak = ""
    rec_rows = arm_table(table1[table1["contacts"] == "recording"], "C") if not table1.empty else table1
    if not rec_rows.empty:
        wide = rec_rows.iloc[rec_rows["upper_hz"].to_numpy().argmax()]
        narrow = rec_rows.iloc[rec_rows["upper_hz"].to_numpy().argmin()]
        peak = f"; recording-contact 0-5 ms peak {wide['median_peak_uV']:.0f} uV at {format_hz(wide['upper_hz'])} Hz -> {narrow['median_peak_uV']:.0f} uV at {format_hz(narrow['upper_hz'])} Hz"
        if not stim_rows.empty:
            sw = stim_rows.iloc[stim_rows["upper_hz"].to_numpy().argmax()]
            sn = stim_rows.iloc[stim_rows["upper_hz"].to_numpy().argmin()]
            peak += f"; stim contact rail {sw['median_rail_fs_ms']:.2f} ms -> {sn['median_rail_fs_ms']:.2f} ms"
    line = f"Arm C: rail duration (>= {100 * cfg.zero_rail_fraction:.0f}% of trials zero) " + ("; ".join(parts) if parts else "no channel evaluated") + peak
    if reached or (always and not never):
        line += " -> reducing upper bandwidth prevents ADC saturation" if reached else " -> recording contacts do not saturate at any tested upper bandwidth"
    elif never:
        line += " -> reducing upper bandwidth alone does not remove saturation on " + ", ".join(never)
    return line, reach


def recommend(table1: pd.DataFrame, cfg: SweepConfig) -> tuple[str, pd.DataFrame]:
    """Lowest median recovery on the recording contacts, upper >= min_upper_hz, noise SD <= factor x the reference setting (in-vivo setting when present)."""
    rec = table1[table1["contacts"] == "recording"].copy() if not table1.empty else table1
    if rec.empty:
        return "Recommendation: no data", pd.DataFrame()
    rec = rec[np.isfinite(rec["median_recovery_ms"])]
    noise = rec["median_prestim_sd_uV"].where(np.isfinite(rec["median_prestim_sd_uV"]), rec["median_baseline_sd_uV"])
    rec = rec.assign(noise_sd_uV=noise)
    # reference noise: the setting in use (in vivo) when it is part of the sweep, else the sweep median
    ref_lower, ref_dsp, ref_upper = cfg.reference_setting
    is_ref = (
        (np.abs(np.log(rec["lower_hz"] / ref_lower)) <= cfg.knob_match_log_tol)
        & rec["dsp_enabled"].astype(bool)
        & (np.abs(np.log(rec["dsp_hz"].where(rec["dsp_hz"] > 0, np.nan) / ref_dsp)) <= cfg.knob_match_log_tol)
        & (np.abs(np.log(rec["upper_hz"] / ref_upper)) <= cfg.knob_match_log_tol)
    )
    if is_ref.any():
        ref_noise = float(rec.loc[is_ref, "noise_sd_uV"].iloc[0])
        ref_text = f"reference = {rec.loc[is_ref, 'folder'].iloc[0]} (in-vivo setting) {ref_noise:.2f} uV"
    else:
        ref_noise = float(np.nanmedian(rec["noise_sd_uV"])) if len(rec) else float("nan")
        ref_text = f"reference = sweep median {ref_noise:.2f} uV (in-vivo setting not in sweep)"
    rec = rec.assign(is_reference=is_ref, noise_ok=rec["noise_sd_uV"] <= cfg.noise_factor * ref_noise, upper_ok=rec["upper_hz"] >= cfg.min_upper_hz, not_censored=rec["frac_censored"] < 0.5)
    cols = ["run_id", "folder", "arms", "lower_hz", "dsp_enabled", "dsp_hz", "upper_hz", "tau_nominal_ms", "median_recovery_ms", "recovery_ci_low", "recovery_ci_high", "frac_censored", "median_rail_fs_ms", "median_peak_uV", "noise_sd_uV", "median_baseline_sd_uV", "median_prestim_sd_uV", "is_reference", "noise_ok", "upper_ok", "not_censored"]
    pareto = rec[cols].sort_values(["median_recovery_ms", "noise_sd_uV"]).reset_index(drop=True)
    eligible = pareto[pareto["noise_ok"] & pareto["upper_ok"] & pareto["not_censored"]]
    if eligible.empty:
        return (f"Recommendation: no setting satisfies upper >= {cfg.min_upper_hz:g} Hz and noise SD <= {cfg.noise_factor:g} x {ref_text}; see the Pareto table", pareto)
    best = eligible.iloc[0]
    dsp = f"DSP {format_hz(best['dsp_hz'])} Hz" if bool(best["dsp_enabled"]) else "DSP off"
    text = (
        f"Recommendation: {best['folder']} -- analog lower {format_hz(best['lower_hz'])} Hz, {dsp}, upper {format_hz(best['upper_hz'])} Hz "
        f"(tau_nominal {best['tau_nominal_ms']:.1f} ms): median recovery {best['median_recovery_ms']:.1f} ms "
        f"[{best['recovery_ci_low']:.1f}, {best['recovery_ci_high']:.1f}] on the recording contacts, noise SD {best['noise_sd_uV']:.2f} uV "
        f"({ref_text}, limit {cfg.noise_factor:g}x), upper >= {cfg.min_upper_hz:g} Hz. "
        f"Runner-up: " + (f"{eligible.iloc[1]['folder']} ({eligible.iloc[1]['median_recovery_ms']:.1f} ms, {eligible.iloc[1]['noise_sd_uV']:.2f} uV)" if len(eligible) > 1 else "none")
    )
    return text, pareto


def build_verdict(sweep: SweepSet, table1: pd.DataFrame, per_channel: pd.DataFrame, slopes: dict[str, SlopeResult], cfg: SweepConfig) -> Verdict:
    v = Verdict()
    v.arm_A = _slope_line("A", slopes["A"], {
        "~1": "the low-frequency analog pole sets recovery; the fix is a higher analog cutoff",
        "flat": "nonlinear saturation recovery, independent of the nominal cutoff",
        "intermediate": "recovery scales with the analog pole but weaker than 1:1 (partly nonlinear)",
    })
    v.arm_B = _slope_line("B", slopes["B"], {
        "flat": "DSP definitively exonerated (recovery does not follow the DSP cutoff)",
        "~1": "the DSP cutoff sets recovery",
        "intermediate": "recovery scales with the DSP cutoff but weaker than 1:1",
    })
    v.arm_C, _ = arm_c_line(table1, per_channel, cfg)
    v.lines = [v.arm_A, v.arm_B, v.arm_C]
    v.recommendation, v.pareto = recommend(table1, cfg)
    return v


__all__ = ["Verdict", "arm_c_line", "build_verdict", "recommend"]
