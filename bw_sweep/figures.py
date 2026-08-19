"""Figures 0-6 of the bandwidth sweep. Every axis limit comes from SweepConfig."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from stim_analysis.figures import _style_axes, channel_palette  # noqa: E402
from bw_sweep.config import ARM_BY_NAME, SweepConfig, format_hz  # noqa: E402
from bw_sweep.load import SweepRun, SweepSet  # noqa: E402
from bw_sweep.stats import SlopeResult  # noqa: E402
from bw_sweep.summary import in_arm  # noqa: E402

DOT_ALPHA = 0.35
DOT_SIZE = 8.0
ARM_COLOURS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}


def _clip_log(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), limits[0], limits[1])


def _jitter_log(rng: np.random.Generator, n: int, scale: float = 0.03) -> np.ndarray:
    return 10 ** rng.uniform(-scale, scale, size=n)


def _knob_colours(runs: list[SweepRun], arm: str) -> dict[str, tuple[float, float, float, float]]:
    knob = ARM_BY_NAME[arm].knob
    ordered = sorted(runs, key=lambda r: (r.knob_values[knob], r.run_id))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(ordered) - 1)
    return {r.run_id: cmap(0.05 + 0.9 * i / n) for i, r in enumerate(ordered)}


def _arm_x(run: SweepRun, arm: str) -> float:
    return run.upper_hz if arm == "C" else run.fc_hz


def _arm_xlim(cfg: SweepConfig, arm: str) -> tuple[float, float]:
    return cfg.lim_upper_hz if arm == "C" else cfg.lim_fc_hz


def _arm_xlabel(arm: str) -> str:
    return "analog upper bandwidth (Hz)" if arm == "C" else "effective high-pass corner f_c (Hz)"


# -----------------------------------------------------------------------------
# fig0: median traces per run
# -----------------------------------------------------------------------------


def fig_traces(sweep: SweepSet, traces: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]], stim_channel: str | None, cfg: SweepConfig) -> Figure:
    """Rows = arms, columns = full scale / zoom. One line per run (median of the recording contacts), stim contact dashed."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), squeeze=False)
    for row, arm in enumerate(("A", "B", "C")):
        runs = sweep.arm_runs(arm)
        colours = _knob_colours(runs, arm)
        for col, ylim in enumerate((cfg.lim_trace_uV, cfg.lim_trace_zoom_uV)):
            ax = axes[row][col]
            _style_axes(ax)
            for run in runs:
                if run.run_id not in traces:
                    continue
                t, per_channel = traces[run.run_id]
                rec = [v for c, v in per_channel.items() if c != stim_channel]
                if rec:
                    ax.plot(t, np.median(np.stack(rec), axis=0), color=colours[run.run_id], lw=1.0, label=f"{run.arm_label(arm)}")
                if stim_channel in per_channel:
                    ax.plot(t, per_channel[stim_channel], color=colours[run.run_id], lw=0.7, ls="--")
            ax.axhline(cfg.threshold_uV, color="0.6", lw=0.5, ls=":")
            ax.axhline(-cfg.threshold_uV, color="0.6", lw=0.5, ls=":")
            if col == 0:
                ax.axhline(cfg.rail_level_uV, color="0.3", lw=0.5, ls="--")
                ax.axhline(-cfg.rail_level_uV, color="0.3", lw=0.5, ls="--")
            ax.set_xlim(*cfg.trace_window_ms)
            ax.set_ylim(*ylim)
            ax.set_xlabel("time from pulse onset (ms)", fontsize=8)
            ax.set_ylabel("median centred trace (uV)", fontsize=8)
            ax.set_title(f"Arm {arm}: {ARM_BY_NAME[arm].title}" + (" (zoom)" if col else ""), fontsize=9)
            if col == 0:
                ax.legend(fontsize=6.5, ncol=2, frameon=False, title=ARM_BY_NAME[arm].knob_label, title_fontsize=6.5)
    fig.suptitle("Bandwidth sweep: median artifact trace per run (solid = median of recording contacts, dashed = stim contact); dotted = +-100 uV threshold, dashed grey = ADC rail", fontsize=9)
    fig.subplots_adjust(top=0.94, hspace=0.42, wspace=0.28)
    return fig


# -----------------------------------------------------------------------------
# fig1 / fig2: recovery vs tau_nominal
# -----------------------------------------------------------------------------


def fig_recovery_vs_tau(trials: pd.DataFrame, sweep: SweepSet, arm: str, table: pd.DataFrame, slope: SlopeResult, cfg: SweepConfig, rng: np.random.Generator) -> Figure:
    runs = sweep.arm_runs(arm)
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    _style_axes(ax)
    sub = trials[in_arm(trials, arm)] if not trials.empty else trials
    channels = sorted(sub["channel"].unique()) if not sub.empty else []
    palette = channel_palette(channels)
    xlim, ylim = cfg.lim_tau_ms, cfg.lim_recovery_ms
    # per-trial dots
    for channel in channels:
        g = sub[sub["channel"] == channel]
        stim = bool(g["is_stim_contact"].iloc[0])
        x = _clip_log(g["tau_nominal_ms"].to_numpy() * _jitter_log(rng, len(g)), xlim)
        y = _clip_log(g["recovery_ms"].to_numpy(), ylim)
        if stim:
            ax.scatter(x, y, s=DOT_SIZE + 4, marker="^", facecolors="none", edgecolors=palette[channel], alpha=0.5, lw=0.6, label=f"{channel} (stim contact)")
        else:
            ax.scatter(x, y, s=DOT_SIZE, color=palette[channel], alpha=DOT_ALPHA, lw=0, label=channel)
    # per-run medians (recording contacts pooled) with IQR + CI
    rec = table[(table["contacts"] == "recording")] if not table.empty else table
    for run in runs:
        row = rec[rec["run_id"] == run.run_id]
        if row.empty or not np.isfinite(row["median_recovery_ms"].iloc[0]):
            continue
        r = row.iloc[0]
        x = float(np.clip(run.tau_nominal_ms, *xlim))
        med = float(r["median_recovery_ms"])
        censored = med >= cfg.censor_ms
        y = float(np.clip(med, *ylim))
        if censored:
            ax.annotate("", xy=(x, ylim[1] * 0.98), xytext=(x, ylim[1] * 0.6), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
            ax.text(x, ylim[1] * 0.55, "censored", ha="center", va="top", fontsize=6.5)
            continue
        excluded = run.run_id in slope.excluded
        ax.plot([x, x], [np.clip(r["recovery_q25"], *ylim), np.clip(r["recovery_q75"], *ylim)], color="k", lw=0.8)
        ax.plot([x, x], [np.clip(r["recovery_ci_low"], *ylim), np.clip(r["recovery_ci_high"], *ylim)], color="k", lw=2.4, alpha=0.6)
        right_half = math.log10(x) > 0.5 * (math.log10(xlim[0]) + math.log10(xlim[1]))
        ha = "right" if right_half else "left"
        pad = "  "
        if excluded:
            ax.scatter([x], [y], s=46, facecolors="w", edgecolors="k", zorder=5, marker="o", lw=1.2)
            reason = slope.excluded[run.run_id].split(" (")[0]
            label = f"{run.arm_label(arm)} [not in fit: {reason}]"
            ax.text(x, y, (label + pad) if right_half else (pad + label), fontsize=6.0, va="center", ha=ha, color="0.3")
        else:
            ax.scatter([x], [y], s=46, color="k", zorder=5, marker="o", edgecolors="w", lw=0.6)
            label = run.arm_label(arm)
            ax.text(x, y, (label + pad) if right_half else (pad + label), fontsize=6.5, va="center", ha=ha)
    xs = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), 200)
    ax.plot(xs, xs, color="0.5", lw=0.8, ls="-", label="identity: recovery = tau")
    ax.plot(xs, xs * math.log(cfg.rail_level_uV / cfg.threshold_uV), color="0.5", lw=0.8, ls=":", label=f"linear filter from rail: tau*ln({cfg.rail_level_uV:.0f}/{cfg.threshold_uV:.0f}) = {math.log(cfg.rail_level_uV / cfg.threshold_uV):.2f} tau")
    if np.isfinite(slope.slope1_ratio):
        ax.plot(xs, xs * slope.slope1_ratio, color=ARM_COLOURS[arm], lw=0.9, ls="--", label=f"slope-1 fit: recovery = {slope.slope1_ratio:.2f} tau")
    if np.isfinite(slope.slope) and slope.n_runs_used >= 2:
        used = trials[trials["run_id"].isin(slope.used_run_ids)]
        lo, hi = float(used["tau_nominal_ms"].min()), float(used["tau_nominal_ms"].max())
        xf = np.logspace(math.log10(max(lo, xlim[0]) / 1.5), math.log10(min(hi, xlim[1]) * 1.5), 50)
        ax.plot(xf, 10 ** (slope.intercept + slope.slope * np.log10(xf)), color=ARM_COLOURS[arm], lw=1.6, label=f"OLS slope {slope.slope:.2f} [{slope.ci_low:.2f}, {slope.ci_high:.2f}] -> {slope.verdict}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("tau_nominal = 1/(2 pi f_c) of the higher high-pass pole (ms)", fontsize=8.5)
    ax.set_ylabel(f"recovery time, fixed {cfg.threshold_uV:.0f} uV threshold (ms)", fontsize=8.5)
    ax.set_title(f"Arm {arm}: {ARM_BY_NAME[arm].title} -- recovery vs nominal time constant", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.subplots_adjust(top=0.93)
    return fig


# -----------------------------------------------------------------------------
# fig3: Arm C rail duration and peak vs upper bandwidth
# -----------------------------------------------------------------------------


def fig_arm_c(trials: pd.DataFrame, sweep: SweepSet, per_channel: pd.DataFrame, cfg: SweepConfig, rng: np.random.Generator) -> Figure:
    runs = sweep.arm_runs("C")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    sub = trials[in_arm(trials, "C")] if not trials.empty else trials
    channels = sorted(sub["channel"].unique()) if not sub.empty else []
    palette = channel_palette(channels)
    pc = per_channel[per_channel["arms"].astype(str).str.split("+").apply(lambda p: "C" in p)] if not per_channel.empty else per_channel
    for k, (col, ylim, log, ylabel) in enumerate((("rail_fs_ms", cfg.lim_rail_ms, False, "rail duration at +-6389 uV after onset (ms)"), ("peak_uV", cfg.lim_peak_uV, True, "peak |centred| in 0-5 ms (uV)"))):
        ax = axes[k]
        _style_axes(ax)
        for j, channel in enumerate(channels):
            g = sub[sub["channel"] == channel]
            stim = bool(g["is_stim_contact"].iloc[0])
            offset = 10 ** (-0.045 + 0.03 * j)
            x = g["upper_hz"].to_numpy() * offset * _jitter_log(rng, len(g), 0.01)
            y = g[col].to_numpy(dtype=float)
            y = _clip_log(y, ylim) if log else np.clip(y, *ylim)
            marker = "^" if stim else "o"
            ax.scatter(np.clip(x, *cfg.lim_upper_hz), y, s=DOT_SIZE, marker=marker, color=palette[channel], alpha=DOT_ALPHA, lw=0)
            rows = pc[pc["channel"] == channel] if not pc.empty else pc
            for run in runs:
                r = rows[rows["run_id"] == run.run_id]
                if r.empty:
                    continue
                r = r.iloc[0]
                xm = float(np.clip(run.upper_hz * offset, *cfg.lim_upper_hz))
                med_raw = float(r[f"median_{col}"])
                med, lo, hi = (float(np.clip(v, *ylim)) for v in (med_raw, float(r[f"{col}_ci_low"]), float(r[f"{col}_ci_high"])))
                label = f"{channel}{' (stim contact)' if stim else ''}" if run is runs[0] else None
                if med_raw > ylim[1]:  # off the fixed scale: arrow at the top with the value
                    ax.annotate("", xy=(xm, ylim[1]), xytext=(xm, ylim[1] - 0.12 * (ylim[1] - ylim[0]) if not log else ylim[1] / 1.6), arrowprops=dict(arrowstyle="->", color=palette[channel], lw=1.2))
                    ax.text(xm, ylim[1] - 0.13 * (ylim[1] - ylim[0]) if not log else ylim[1] / 1.7, f"{med_raw:.0f}", fontsize=5.5, ha="center", va="top", color=palette[channel])
                    if label:
                        ax.scatter([], [], s=48, marker=marker, color=palette[channel], edgecolors="k", lw=0.6, label=label)
                    continue
                ax.plot([xm, xm], [lo, hi], color=palette[channel], lw=2.0, alpha=0.8)
                ax.scatter([xm], [med], s=48, marker=marker, color=palette[channel], edgecolors="k", lw=0.6, zorder=5, label=label)
                if col == "rail_fs_ms" and float(r["frac_zero_rail"]) < 1.0:
                    ax.text(xm, med + 0.02 * (ylim[1] - ylim[0]) + 0.6 * j, f"{100 * float(r['frac_zero_rail']):.0f}% zero", fontsize=5.5, ha="center", va="bottom", color=palette[channel])
        ax.set_xscale("log")
        ax.set_xlim(*cfg.lim_upper_hz)
        if log:
            ax.set_yscale("log")
            ax.axhline(cfg.rail_level_uV, color="0.3", lw=0.8, ls="--")
            ax.text(cfg.lim_upper_hz[0] * 1.1, cfg.rail_level_uV * 1.05, "ADC rail +-6389 uV", fontsize=7, va="bottom")
        ax.set_ylim(*ylim)
        ax.set_xlabel("analog upper bandwidth (Hz)  [7500 Hz point = shared Arm A run; two 500 Hz replicates]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left" if k == 0 else "lower left")
    axes[0].set_title("Arm C: rail duration vs upper bandwidth (median, 95% CI, per-trial dots; % zero shown when < 100%)", fontsize=9)
    axes[1].set_title("Arm C: 0-5 ms peak vs upper bandwidth", fontsize=9)
    fig.subplots_adjust(top=0.9, wspace=0.28)
    return fig


# -----------------------------------------------------------------------------
# fig4: tau_fit vs tau_nominal
# -----------------------------------------------------------------------------


def fig_tau_fit(trials: pd.DataFrame, sweep: SweepSet, table: pd.DataFrame, cfg: SweepConfig, rng: np.random.Generator) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    _style_axes(ax)
    lim = cfg.lim_tau_ms
    for arm in ("A", "B", "C"):
        sub = trials[in_arm(trials, arm) & ~trials["is_stim_contact"]] if not trials.empty else trials
        if sub.empty:
            continue
        conv = sub[sub["fit_converged"]]
        for informative, marker_kw in ((True, dict(color=ARM_COLOURS[arm], alpha=DOT_ALPHA, lw=0)), (False, dict(facecolors="none", edgecolors=ARM_COLOURS[arm], alpha=0.35, lw=0.5))):
            g = conv[conv["fit_informative"] == informative]
            if g.empty:
                continue
            x = _clip_log(g["tau_nominal_ms"].to_numpy() * _jitter_log(rng, len(g)), lim)
            y = _clip_log(g["tau_fit_ms"].to_numpy(), lim)
            ax.scatter(x, y, s=DOT_SIZE, label=f"arm {arm} {'informative' if informative else 'uninformative (tau_nominal < %g ms)' % cfg.fit_informative_tau_ms}", **marker_kw)
        rec = table[(table["contacts"] == "recording") & table["arms"].astype(str).str.split("+").apply(lambda p: arm in p)]
        for _, r in rec.iterrows():
            if not np.isfinite(r["median_tau_fit_ms"]):
                continue
            x = float(np.clip(r["tau_nominal_ms"], *lim))
            ax.plot([x, x], [np.clip(r["tau_fit_ci_low"], *lim), np.clip(r["tau_fit_ci_high"], *lim)], color=ARM_COLOURS[arm], lw=2.0, alpha=0.8)
            ax.scatter([x], [np.clip(r["median_tau_fit_ms"], *lim)], s=44, color=ARM_COLOURS[arm], edgecolors="k", lw=0.6, zorder=5)
    xs = np.logspace(math.log10(lim[0]), math.log10(lim[1]), 100)
    ax.plot(xs, xs, color="0.5", lw=0.8, label="identity")
    ax.axhspan(lim[0], cfg.fit_tau_bounds_ms[0], color="0.9", zorder=0)
    ax.axhspan(cfg.fit_tau_bounds_ms[1], lim[1], color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("tau_nominal (ms)", fontsize=8.5)
    ax.set_ylabel(f"tau_fit, single exponential on [rail exit + {cfg.fit_start_offset_ms:g} ms, {cfg.fit_end_ms:g} ms] (ms)", fontsize=8.5)
    ax.set_title("tau_fit vs tau_nominal, all arms (recording contacts; grey bands = fit bounds)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.subplots_adjust(top=0.93)
    return fig


# -----------------------------------------------------------------------------
# fig5: R2 distributions
# -----------------------------------------------------------------------------


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(0), np.zeros(0)
    return v, np.arange(1, v.size + 1) / v.size


def fig_r2(trials: pd.DataFrame, sweep: SweepSet, cfg: SweepConfig) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for k, arm in enumerate(("A", "B", "C")):
        ax = axes[k]
        _style_axes(ax)
        runs = sweep.arm_runs(arm)
        colours = _knob_colours(runs, arm)
        sub = trials[in_arm(trials, arm) & ~trials["is_stim_contact"] & trials["fit_converged"]] if not trials.empty else trials
        pooled = sub["r2"].to_numpy() if not sub.empty else np.zeros(0)
        for run in runs:
            g = sub[sub["run_id"] == run.run_id]
            if g.empty:
                continue
            x, y = _ecdf(np.clip(g["r2"].to_numpy(dtype=float), -0.05, 1.0))
            ls = "-" if bool(g["fit_informative"].iloc[0]) else ":"
            frac = float((g["r2"] < cfg.r2_min).mean())
            ax.step(x, y, where="post", color=colours[run.run_id], lw=1.0, ls=ls, label=f"{run.arm_label(arm)}: {100 * frac:.0f}% < {cfg.r2_min:g}")
        if pooled.size:
            x, y = _ecdf(np.clip(pooled, -0.05, 1.0))
            ax.step(x, y, where="post", color="k", lw=1.8, label=f"pooled: {100 * float((pooled < cfg.r2_min).mean()):.0f}% < {cfg.r2_min:g}")
        ax.axvline(cfg.r2_min, color="0.4", lw=0.8, ls="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("R2 of single-exponential tail fit", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("ECDF (fraction of trials)", fontsize=8.5)
        ax.set_title(f"Arm {arm}: {ARM_BY_NAME[arm].title}", fontsize=8.5)
        ax.legend(fontsize=6, frameon=False, loc="upper left", title="dotted = uninformative fits", title_fontsize=6)
    fig.suptitle("R2 distributions per arm (recording contacts). A linear filter gives near-perfect single exponentials; widespread R2 < 0.9 points to a nonlinear mechanism.", fontsize=9)
    fig.subplots_adjust(top=0.85, wspace=0.12)
    return fig


# -----------------------------------------------------------------------------
# fig6: noise floor vs bandwidth
# -----------------------------------------------------------------------------


def fig_noise(per_channel: pd.DataFrame, sweep: SweepSet, cfg: SweepConfig) -> Figure:
    """Baseline SD (spec window), clean-segment SD (pre-/post-train) and its > 5 Hz component, per channel and setting."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True)
    for k, arm in enumerate(("A", "B", "C")):
        ax = axes[k]
        _style_axes(ax)
        runs = sweep.arm_runs(arm)
        rows = per_channel[per_channel["arms"].astype(str).str.split("+").apply(lambda p: arm in p)] if not per_channel.empty else per_channel
        channels = sorted(rows["channel"].unique()) if not rows.empty else []
        palette = channel_palette(channels)
        stim_flags: dict[str, bool] = {}
        for j, channel in enumerate(channels):
            offset = 10 ** (-0.045 + 0.03 * j)
            g = rows[rows["channel"] == channel]
            stim = bool(g["is_stim_contact"].iloc[0]) if not g.empty else False
            stim_flags[channel] = stim
            for run in runs:
                r = g[g["run_id"] == run.run_id]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = float(np.clip(_arm_x(run, arm) * offset, *_arm_xlim(cfg, arm)))
                med = float(np.clip(r["median_baseline_sd_uV"], *cfg.lim_sd_uV))
                lo, hi = (float(np.clip(r[c], *cfg.lim_sd_uV)) for c in ("baseline_sd_uV_ci_low", "baseline_sd_uV_ci_high"))
                ax.plot([x, x], [lo, hi], color=palette[channel], lw=1.6, alpha=0.8)
                ax.scatter([x], [med], s=34, marker="^" if stim else "o", color=palette[channel], edgecolors="k", lw=0.5, zorder=5)
                cs = float(r["median_clean_sd_uV"]) if "median_clean_sd_uV" in r else float("nan")
                if np.isfinite(cs):
                    ax.scatter([x], [float(np.clip(cs, *cfg.lim_sd_uV))], s=34, marker="x", color=palette[channel], lw=1.1, zorder=6)
                hf = float(r["median_clean_sd_gt5hz_uV"]) if "median_clean_sd_gt5hz_uV" in r else float("nan")
                if np.isfinite(hf):
                    ax.scatter([x], [float(np.clip(hf, *cfg.lim_sd_uV))], s=30, marker="+", color=palette[channel], lw=1.1, zorder=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*_arm_xlim(cfg, arm))
        ax.set_ylim(*cfg.lim_sd_uV)
        ax.set_xlabel(_arm_xlabel(arm), fontsize=8.5)
        if k == 0:
            ax.set_ylabel("noise SD (uV)", fontsize=8.5)
        ax.set_title(f"Arm {arm}: {ARM_BY_NAME[arm].title}", fontsize=8.5)
        handles = [Line2D([], [], marker="^" if stim_flags[c] else "o", color=palette[c], ls="", markeredgecolor="k", markersize=5, label=f"{c}{' (stim contact)' if stim_flags[c] else ''}") for c in channels]
        legend1 = ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left", title="channel", title_fontsize=6)
        ax.add_artist(legend1)
        markers = [
            Line2D([], [], marker="o", color="0.3", ls="", markeredgecolor="k", markersize=5, label="baseline SD, -500..-50 ms (median, 95% CI)"),
            Line2D([], [], marker="x", color="0.3", ls="", markersize=6, label="clean segment SD (pre-/post-train)"),
            Line2D([], [], marker="+", color="0.3", ls="", markersize=6, label=f"clean segment SD > {cfg.clean_sd_highpass_hz:g} Hz (broadband noise)"),
        ]
        ax.legend(handles=markers, fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Noise floor cost of each setting: baseline SD (includes the previous pulse's tail at long tau), the clean stimulation-free segment, and its > 5 Hz component (sub-5 Hz drift is what a low high-pass corner lets through)", fontsize=8.5)
    fig.subplots_adjust(top=0.85, wspace=0.12)
    return fig


__all__ = ["fig_arm_c", "fig_noise", "fig_r2", "fig_recovery_vs_tau", "fig_tau_fit", "fig_traces"]
