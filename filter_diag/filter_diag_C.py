"""Test C figures: synthetic vs real recovery distributions; recovery vs step amplitude."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig


def synthetic_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Median recovery per synthetic condition."""
    keys = ["family", "sample_rate_hz", "rail_mode", "duration_ms", "amplitude_uV", "noise_sd_uV", "tail_tau_ms", "tail_amplitude_uV", "clipped_input", "tau_dsp_ms"]
    g = table.groupby(keys)
    out = g["recovery_ms"].median().rename("median_recovery_ms").reset_index()
    out["q25_recovery_ms"] = g["recovery_ms"].quantile(0.25).to_numpy()
    out["q75_recovery_ms"] = g["recovery_ms"].quantile(0.75).to_numpy()
    out["median_rail_ms"] = g["rail_ms"].median().to_numpy()
    out["censored_fraction"] = g["censored"].mean().to_numpy()
    out["median_threshold_uV"] = g["threshold_uV"].median().to_numpy()
    out["n"] = g.size().to_numpy()
    return out


def figure_synthetic_vs_real(synth: pd.DataFrame, real: pd.DataFrame, cfg: AnalysisConfig, tau_dsp_ms: float, ctx, examples: dict | None = None) -> object:
    """Figure 2: recovery distributions, synthetic (physical / spec order, step / step+tail) vs observed."""
    from stim_analysis.figures import FOOTER_IN, _style_axes, build_caption, finish_figure, plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0 + FOOTER_IN))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=(FOOTER_IN + 0.5) / (5.0 + FOOTER_IN), wspace=0.28)
    lim = cfg.lim_recovery_ms
    bins = np.geomspace(lim[0], lim[1], 50)
    ax = axes[0]
    for session, sub in real.groupby("session"):
        ax.hist(np.clip(sub["recovery_ms"], *lim), bins=bins, alpha=0.45, density=True, label=f"observed {session} (n={len(sub)}, median {np.median(sub['recovery_ms']):.0f} ms)")
    fs30 = synth[(synth["sample_rate_hz"] == 30000.0)]
    for (family, mode), sub in fs30[fs30["clipped_input"] & fs30["noise_sd_uV"].isin([50.0])].groupby(["family", "rail_mode"]):
        ax.hist(np.clip(sub["recovery_ms"], *lim), bins=bins, histtype="step", lw=1.3, density=True, label=f"synthetic {family} (rail mode {mode}), rail-level input, noise 50 uV: median {np.median(sub['recovery_ms']):.0f} ms")
    ax.axvline(tau_dsp_ms * np.log(6389.6 / 100.0), color="#d62728", ls="--", lw=1.0, label=f"tau*ln(rail/100 uV) = {tau_dsp_ms * np.log(6389.6 / 100.0):.0f} ms")
    ax.set_xscale("log"); ax.set_xlim(lim); ax.set_xlabel("recovery time (ms, log)"); ax.set_ylabel("density"); ax.set_title("observed vs synthetic recovery distributions", fontsize=9); ax.legend(fontsize=5.8, frameon=False)
    ax = axes[1]
    step = fs30[(fs30["family"] == "step") & (fs30["rail_mode"] == "freeze") & (fs30["noise_sd_uV"] == 50.0)]
    for d, sub in step.groupby("duration_ms"):
        med = sub.groupby("amplitude_uV")["recovery_ms"].median()
        ax.plot(med.index, np.clip(med.values, *lim), "-o", ms=3.5, lw=1.0, label=f"step d = {d:g} ms")
    ax.axvline(6389.6, color="0.5", ls=":", lw=0.9)
    ax.text(6389.6, lim[1] * 0.7, " ADC rail", fontsize=6.5, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlabel("input step amplitude (uV, log)"); ax.set_ylabel("median recovery (ms, log)"); ax.set_title("synthetic recovery vs step amplitude (freeze mode, noise 50 uV)", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[2]
    if examples:
        for key, (traces, t_ms, core) in list(examples.items())[:6]:
            tr = traces[0][core]
            tt = t_ms[core]
            w = (tt >= -50) & (tt <= 600)
            ax.plot(tt[w], np.clip(tr[w], -7000, 7000), lw=0.8, label=key)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlim(-50, 600); ax.set_ylim(-7000, 7000); ax.set_xlabel("time from step (ms)"); ax.set_ylabel("recorded (uV)"); ax.set_title("example synthetic recorded traces (freeze mode)", fontsize=9); ax.legend(fontsize=5.5, frameon=False)
    for a in axes:
        _style_axes(a)
    fig.suptitle("Test C: synthetic step through analog HP -> ADC clip -> Intan DSP high-pass (rail modes freeze / track / spec) -> clamp, then the unchanged recovery algorithm", fontsize=10)
    finish_figure(fig, build_caption(ctx))
    return fig


def figure_log_law(synth: pd.DataFrame, cfg: AnalysisConfig, tau_dsp_ms: float, ctx) -> object:
    """Figure 3: recovery vs amplitude on log axes for sub-rail steps (log law) and the clipping plateau; plus the tail family."""
    from stim_analysis.figures import FOOTER_IN, _style_axes, build_caption, finish_figure, plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0 + FOOTER_IN))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=(FOOTER_IN + 0.5) / (5.0 + FOOTER_IN), wspace=0.28)
    lim = cfg.lim_recovery_ms
    ax = axes[0]
    for (fs, mode), sub in synth[(synth["family"] == "step") & (synth["duration_ms"] == 10.0) & (synth["noise_sd_uV"] == 10.0)].groupby(["sample_rate_hz", "rail_mode"]):
        med = sub.groupby("amplitude_uV")["recovery_ms"].median()
        ax.plot(med.index, np.clip(med.values, *lim), "-o", ms=3.5, lw=1.0, label=f"fs {fs / 1000:g} kHz, rail mode {mode}")
    amps = np.geomspace(100, 6389.6, 50)
    ax.plot(amps, tau_dsp_ms * np.log(np.maximum(amps / 100.0, 1.0)), color="0.4", ls="--", lw=1.0, label=f"tau*ln(V/100 uV), tau = {tau_dsp_ms:.0f} ms")
    ax.axvline(6389.6, color="0.5", ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlabel("input step amplitude (uV, log)"); ax.set_ylabel("median recovery (ms, log)"); ax.set_title("log law below the rail, plateau above (10 ms step, noise 10 uV)", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    for noise, sub in synth[(synth["family"] == "step") & (synth["rail_mode"] == "freeze") & (synth["sample_rate_hz"] == 30000.0) & (synth["amplitude_uV"] == 20000.0)].groupby("noise_sd_uV"):
        med = sub.groupby("duration_ms")["recovery_ms"].median()
        ax.plot(med.index, np.clip(med.values, *lim), "-o", ms=3.5, lw=1.0, label=f"noise SD {noise:g} uV (threshold {max(3 * noise, 100):g} uV)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlabel("rail duration d (ms, log)"); ax.set_ylabel("median recovery (ms, log)"); ax.set_title("recovery vs rail duration (20 mV step, freeze mode)", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[2]
    tail = synth[(synth["family"] == "step_tail") & (synth["sample_rate_hz"] == 30000.0) & (synth["duration_ms"] == 10.0) & (synth["noise_sd_uV"] == 10.0)]
    if not tail.empty:
        med = tail.groupby("tail_tau_ms")["recovery_ms"].median()
        x = np.minimum(med.index.to_numpy(dtype=float), 3000.0)
        ax.plot(x, np.clip(med.values, *lim), "-s", ms=4, lw=1.0, color="#2ca02c", label="step + same-sign input tail (3 mV)")
        ax.axhline(float(synth[(synth["family"] == "step") & (synth["rail_mode"] == "freeze") & (synth["sample_rate_hz"] == 30000.0) & (synth["duration_ms"] == 10.0) & (synth["noise_sd_uV"] == 10.0) & (synth["amplitude_uV"] == 20000.0)]["recovery_ms"].median()), color="0.4", ls="--", lw=1.0, label="pure step (no input tail)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlabel("input tail time constant (ms, log; 3000 = flat offset)"); ax.set_ylabel("median recovery (ms, log)"); ax.set_title("recovery when the INPUT has a slow same-sign tail", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    for a in axes:
        _style_axes(a)
    fig.suptitle("Test C: amplitude / duration / input-tail dependence of the recovery time through the instrument model", fontsize=10)
    finish_figure(fig, build_caption(ctx))
    return fig


__all__ = ["figure_log_law", "figure_synthetic_vs_real", "synthetic_summary"]
