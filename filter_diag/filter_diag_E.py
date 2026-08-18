"""Test E: decompose recovery = t_rail_exit + tau * ln(V_ref / V_threshold); residual structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from filter_diag.common import ADC_FULL_SCALE_UV


def decompose(fits: pd.DataFrame, tau_dsp_ms: float) -> pd.DataFrame:
    """Per epoch predictions of the additive model in two variants.

    * ``pred_spec``: exit + tau * ln(V_rail / thr)   (V_peak for non-railed epochs)
    * ``pred_dsp_state``: exit + tau * ln(V_rail * (1 - exp(-d/tau)) / thr) -- the amplitude
      a first-order high-pass actually leaves after a rail of duration d
    * ``pred_fit``: exit + tau_fit * ln(|A_fit| / thr)   (per-epoch fitted exponential)
    """
    f = fits.copy()
    thr = f["threshold_uV"].to_numpy(dtype=float)
    v_ref = np.where(f["was_railed"], ADC_FULL_SCALE_UV, f["peak_abs_uV"].to_numpy(dtype=float))
    d = f["rail_ms"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["pred_spec_ms"] = f["exit_ms"] + tau_dsp_ms * np.log(np.maximum(v_ref / thr, 1.0))
        left = ADC_FULL_SCALE_UV * (1.0 - np.exp(-np.maximum(d, 0.0333) / tau_dsp_ms))
        v_state = np.where(f["was_railed"], left, f["peak_abs_uV"].to_numpy(dtype=float))
        f["pred_dsp_state_ms"] = f["exit_ms"] + tau_dsp_ms * np.log(np.maximum(v_state / thr, 1.0))
        a_fit = np.abs(f["A_uV"].to_numpy(dtype=float))
        f["pred_fit_ms"] = np.where(f["fit_ok"], f["exit_ms"] + f["tau_ms"] * np.log(np.maximum(a_fit / thr, 1.0)), np.nan)
    for name in ("pred_spec_ms", "pred_dsp_state_ms", "pred_fit_ms"):
        f[name.replace("pred_", "resid_")] = f["recovery_ms"] - f[name]
    return f


def decomposition_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    obs = rows["recovery_ms"].to_numpy(dtype=float)
    for name in ("pred_spec_ms", "pred_dsp_state_ms", "pred_fit_ms"):
        pred = rows[name].to_numpy(dtype=float)
        ok = np.isfinite(pred) & np.isfinite(obs) & ~rows["censored"].to_numpy(dtype=bool)
        if ok.sum() < 5:
            out.append({"model": name, "n": int(ok.sum())})
            continue
        resid = obs[ok] - pred[ok]
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((obs[ok] - obs[ok].mean()) ** 2))
        row = {"model": name, "n": int(ok.sum()), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "median_resid_ms": float(np.median(resid)), "iqr_resid_ms": float(np.percentile(resid, 75) - np.percentile(resid, 25)), "median_abs_resid_ms": float(np.median(np.abs(resid)))}
        for cov in ("amplitude_uA", "impedance_kohm", "distance_um"):
            x = rows[cov].to_numpy(dtype=float)[ok]
            good = np.isfinite(x) & (x > 0 if cov != "distance_um" else np.isfinite(x))
            if good.sum() > 10 and np.unique(x[good]).size > 1:
                xx = np.log(x[good]) if cov != "distance_um" else x[good]
                r = np.corrcoef(xx, resid[good])[0, 1]
                row[f"resid_corr_{cov}"] = float(r)
                slope = np.polyfit(xx, resid[good], 1)[0]
                row[f"resid_slope_{cov}"] = float(slope)
        out.append(row)
    return pd.DataFrame(out)


def figure_decomposition(rows: pd.DataFrame, cfg: AnalysisConfig, tau_dsp_ms: float, ctx) -> object:
    from stim_analysis.figures import FOOTER_IN, _style_axes, build_caption, finish_figure, plt

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4 + FOOTER_IN))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=(FOOTER_IN + 0.5) / (8.4 + FOOTER_IN), hspace=0.42, wspace=0.3)
    lim = cfg.lim_recovery_ms
    ok = rows[~rows["censored"] & ~rows["is_stim_contact"]]
    for ax, name, title in zip(axes[0], ("pred_spec_ms", "pred_dsp_state_ms", "pred_fit_ms"), ("exit + tau_dsp ln(V_rail/thr)", "exit + tau_dsp ln(V_rail(1-e^-d/tau)/thr)", "exit + tau_fit ln(|A_fit|/thr)")):
        sub = ok[np.isfinite(ok[name])]
        for session, s2 in sub.groupby("session"):
            ax.scatter(np.clip(s2[name], *lim), np.clip(s2["recovery_ms"], *lim), s=5, alpha=0.25, linewidths=0, label=session)
        ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
        pred = sub[name].to_numpy(dtype=float); obs = sub["recovery_ms"].to_numpy(dtype=float)
        r2 = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2) if len(sub) > 2 else float("nan")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("predicted recovery (ms, log)"); ax.set_ylabel("observed recovery (ms, log)"); ax.set_title(f"{title}\nR^2 = {r2:.2f}, n = {len(sub)}", fontsize=8.5); ax.legend(fontsize=6.5, frameon=False)
    for ax, cov, log in zip(axes[1], ("amplitude_uA", "impedance_kohm", "distance_um"), (True, True, False)):
        sub = ok[np.isfinite(ok["resid_spec_ms"])]
        for session, s2 in sub.groupby("session"):
            x = s2[cov].to_numpy(dtype=float)
            ax.scatter(x * (1 + np.random.default_rng(1).uniform(-0.04, 0.04, x.size)) if log else x + np.random.default_rng(1).uniform(-50, 50, x.size), np.clip(s2["resid_spec_ms"], -800, 800), s=5, alpha=0.25, linewidths=0, label=session)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        if log:
            ax.set_xscale("log")
        ax.set_ylim(-800, 800); ax.set_xlabel(cov); ax.set_ylabel("observed - predicted (spec model), ms"); ax.set_title(f"residual vs {cov}", fontsize=8.5); ax.legend(fontsize=6.5, frameon=False)
    for a in axes.ravel():
        _style_axes(a)
    fig.suptitle(f"Test E: additive decomposition of recovery time (tau_dsp = {tau_dsp_ms:.1f} ms; per-epoch threshold; stim contacts and censored epochs excluded)", fontsize=10)
    finish_figure(fig, build_caption(ctx))
    return fig


__all__ = ["decompose", "decomposition_summary", "figure_decomposition"]
