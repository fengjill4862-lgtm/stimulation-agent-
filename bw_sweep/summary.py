"""table1 (per run x contact group) and the per-run x channel table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bw_sweep.config import SweepConfig
from bw_sweep.load import SweepRun, SweepSet
from bw_sweep.stats import fraction_ci, summarize

CONTACT_GROUPS = ("recording", "stim")


def in_arm(trials: pd.DataFrame, arm: str) -> pd.Series:
    """Boolean mask: the trial's run belongs to ``arm`` (runs may sit in two arms)."""
    if trials.empty:
        return pd.Series(dtype=bool)
    return trials["arms"].astype(str).str.split("+").apply(lambda parts: arm in parts)


def contact_mask(trials: pd.DataFrame, group: str) -> pd.Series:
    if group == "stim":
        return trials["is_stim_contact"].astype(bool)
    return ~trials["is_stim_contact"].astype(bool)


def _run_columns(run: SweepRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "folder": run.folder.name,
        "arms": "+".join(run.arms),
        "arm_A_label": run.arm_label("A"),
        "arm_B_label": run.arm_label("B"),
        "arm_C_label": run.arm_label("C"),
        "lower_hz": run.lower_hz,
        "dsp_enabled": run.dsp_enabled,
        "dsp_hz": run.dsp_hz,
        "dsp_k": run.dsp_k if run.dsp_k is not None else "",
        "upper_hz": run.upper_hz,
        "fc_hz": run.fc_hz,
        "dominant_pole": run.dominant_pole,
        "tau_nominal_ms": run.tau_nominal_ms,
        "tau_second_pole_ms": run.tau_second_ms,
    }


def _group_summary(group: pd.DataFrame, cfg: SweepConfig, rng: np.random.Generator) -> dict[str, object]:
    out: dict[str, object] = {"n_trials": int(len(group)), "n_channels": int(group["channel"].nunique()) if len(group) else 0}
    if group.empty:
        return out
    rec = summarize(group["recovery_ms"].to_numpy(), cfg.bootstrap_n, rng)
    out.update({
        "median_recovery_ms": rec["median"], "recovery_ci_low": rec["ci_low"], "recovery_ci_high": rec["ci_high"],
        "recovery_q25": rec["q25"], "recovery_q75": rec["q75"],
        "n_censored": int(group["censored"].sum()), "frac_censored": float(group["censored"].mean()),
        "n_baseline_contaminated": int(group["baseline_contaminated"].sum()),
        "frac_baseline_contaminated": float(group["baseline_contaminated"].mean()),
        "median_recovery_spec_centred_ms": float(np.median(group["recovery_spec_centred_ms"])),
        "frac_censored_spec_centred": float(group["censored_spec_centred"].mean()),
        "median_baseline_drift_uV": float(np.nanmedian(group["baseline_drift_uV"])),
        "median_abs_local_drift_uV": float(np.nanmedian(group["local_drift_uV"].abs())),
        "frac_local_drift_above_max": float((group["local_drift_uV"].abs() > cfg.local_drift_max_uV).mean()),
    })
    for name, col in (("rail_fs_ms", "rail_fs_ms"), ("rail_emp_ms", "rail_emp_ms"), ("peak_uV", "peak_uV"), ("baseline_sd_uV", "baseline_sd_uV")):
        s = summarize(group[col].to_numpy(), cfg.bootstrap_n, rng)
        out[f"median_{name}"] = s["median"]
        out[f"{name}_ci_low"] = s["ci_low"]
        out[f"{name}_ci_high"] = s["ci_high"]
    zero, zlo, zhi = fraction_ci((group["rail_fs_ms"] <= 0).to_numpy(), cfg.bootstrap_n, rng)
    out.update({"frac_zero_rail": zero, "frac_zero_rail_ci_low": zlo, "frac_zero_rail_ci_high": zhi})
    informative = group[group["fit_informative"] & group["fit_converged"]]
    tau = summarize(informative["tau_fit_ms"].to_numpy(), cfg.bootstrap_n, rng)
    out.update({"median_tau_fit_ms": tau["median"], "tau_fit_ci_low": tau["ci_low"], "tau_fit_ci_high": tau["ci_high"], "n_fits_informative": int(len(informative))})
    tau_nom = float(group["tau_nominal_ms"].iloc[0])
    out["tau_fit_over_tau_nominal"] = tau["median"] / tau_nom if np.isfinite(tau["median"]) and tau_nom > 0 else float("nan")
    r2_all = fraction_ci(group["r2_below_min"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
    out.update({"pct_r2_below_0p9_all": 100 * r2_all[0], "pct_r2_below_0p9_all_ci_low": 100 * r2_all[1], "pct_r2_below_0p9_all_ci_high": 100 * r2_all[2]})
    if len(informative):
        r2_inf = fraction_ci(informative["r2_below_min"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
    else:
        r2_inf = (float("nan"),) * 3
    out.update({"pct_r2_below_0p9_informative": 100 * r2_inf[0], "pct_r2_below_0p9_informative_ci_low": 100 * r2_inf[1], "pct_r2_below_0p9_informative_ci_high": 100 * r2_inf[2]})
    same = fraction_ci(group["same_sign_tail"].to_numpy(dtype=float), cfg.bootstrap_n, rng)
    out.update({"frac_same_sign_tail": same[0], "same_sign_ci_low": same[1], "same_sign_ci_high": same[2]})
    out["median_r2"] = float(np.nanmedian(group["r2"])) if group["r2"].notna().any() else float("nan")
    ps = group.groupby("channel")["prestim_sd_uV"].first()
    out["median_prestim_sd_uV"] = float(np.nanmedian(ps.to_numpy(dtype=float))) if len(ps) else float("nan")
    out["prestim_sd_uV_by_channel"] = "; ".join(f"{c} {v:.2f}" for c, v in ps.items())
    out["prestim_seconds"] = float(group["prestim_seconds"].iloc[0]) if "prestim_seconds" in group else float("nan")
    if "clean_sd_uV" in group:
        cs = group.groupby("channel")["clean_sd_uV"].first()
        ch = group.groupby("channel")["clean_sd_gt5hz_uV"].first()
        out["median_clean_sd_uV"] = float(np.nanmedian(cs.to_numpy(dtype=float))) if len(cs) else float("nan")
        out["median_clean_sd_gt5hz_uV"] = float(np.nanmedian(ch.to_numpy(dtype=float))) if len(ch) else float("nan")
        out["clean_sd_uV_by_channel"] = "; ".join(f"{c} {v:.2f}" for c, v in cs.items())
        out["clean_sd_seconds"] = float(group["clean_sd_seconds"].iloc[0])
        out["clean_sd_source"] = str(group["clean_sd_source"].iloc[0])
    out["floor_ms"] = float(group["floor_ms"].iloc[0])
    return out


def table1(trials: pd.DataFrame, sweep: SweepSet, cfg: SweepConfig, rng: np.random.Generator) -> pd.DataFrame:
    """One row per run x contact group (recording contacts pooled; stim contact separate)."""
    rows: list[dict[str, object]] = []
    for run in sweep.runs:
        sub = trials[trials["run_id"] == run.run_id] if not trials.empty else trials
        for group in CONTACT_GROUPS:
            g = sub[contact_mask(sub, group)] if not sub.empty else sub
            row = _run_columns(run)
            row["contacts"] = group
            row["channels"] = " ".join(sorted(g["channel"].unique())) if not g.empty else ""
            row.update(_group_summary(g, cfg, rng))
            rows.append(row)
    return pd.DataFrame(rows)


def table_per_channel(trials: pd.DataFrame, sweep: SweepSet, cfg: SweepConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in sweep.runs:
        sub = trials[trials["run_id"] == run.run_id] if not trials.empty else trials
        if sub.empty:
            continue
        for channel, g in sub.groupby("channel", sort=True):
            row = _run_columns(run)
            row["channel"] = channel
            row["is_stim_contact"] = bool(g["is_stim_contact"].iloc[0])
            row["impedance_kohm"] = float(g["impedance_kohm"].iloc[0])
            row.update(_group_summary(g, cfg, rng))
            rows.append(row)
    return pd.DataFrame(rows)


def arm_table(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Rows of table1 / per-channel table that belong to ``arm``, sorted by its knob."""
    if table.empty:
        return table
    mask = table["arms"].astype(str).str.split("+").apply(lambda parts: arm in parts)
    knob = {"A": "lower_hz", "B": "dsp_hz", "C": "upper_hz"}[arm]
    return table[mask].sort_values([knob, "run_id"]).reset_index(drop=True)


__all__ = ["CONTACT_GROUPS", "arm_table", "contact_mask", "in_arm", "table1", "table_per_channel"]
