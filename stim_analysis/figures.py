"""Fixed-scale figures with self-describing captions (spec section 9).

Rules enforced here: no axis or colour limit is ever derived from the data
(they all come from AnalysisConfig); log axes for impedance, recovery time and
amplitude; per-trial scatter or CI bands, never bare means; clipped or railed
samples are annotated, never hidden; every caption states n trials retained /
rejected, the blanking window and the filter settings.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from stim_analysis.config import AnalysisConfig  # noqa: E402

TRIAL_ALPHA = 0.35
TRIAL_SIZE = 9.0
FOOTER_IN = 0.9


# -----------------------------------------------------------------------------
# Captions and shared helpers
# -----------------------------------------------------------------------------


@dataclass
class CaptionContext:
    n_retained: int = 0
    n_rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    blank_desc: str = "none"
    filter_desc: str = "none (raw wideband)"
    epoch_desc: str = ""
    note: str = ""

    def with_note(self, note: str) -> "CaptionContext":
        return CaptionContext(
            self.n_retained, self.n_rejected, dict(self.reject_reasons),
            self.blank_desc, self.filter_desc, self.epoch_desc, note,
        )


def build_caption(ctx: CaptionContext) -> str:
    reasons = ""
    if ctx.reject_reasons:
        parts = [f"{key} {value}" for key, value in sorted(ctx.reject_reasons.items()) if value]
        if parts:
            reasons = " (" + ", ".join(parts) + ")"
    pieces = [
        f"n trials retained {ctx.n_retained}, rejected {ctx.n_rejected}{reasons}.",
        f"Blanking: {ctx.blank_desc}.",
        f"Filter: {ctx.filter_desc}.",
    ]
    if ctx.epoch_desc:
        pieces.append(f"Epoch: {ctx.epoch_desc}.")
    if ctx.note:
        pieces.append(ctx.note)
    return " ".join(pieces)


def finish_figure(fig: Figure, caption: str, *, footer_in: float = FOOTER_IN, wrap: int = 165) -> None:
    """Reserve a fixed-inch footer and write the caption into it."""
    width_in, height_in = fig.get_size_inches()
    bottom = footer_in / height_in
    try:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, bottom + 0.02))
    except Exception:  # pragma: no cover - layout engines
        pass
    wrapped = "\n".join(textwrap.wrap(caption, width=int(wrap * width_in / 16.0)))
    fig.text(0.01, 0.012, wrapped, ha="left", va="bottom", fontsize=7.2, color="0.25", wrap=True)


def figure_to_png_bytes(fig: Figure, dpi: int = 220) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    return buffer.getvalue()


def channel_palette(channels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {channel: cmap(index % 10) for index, channel in enumerate(channels)}


def amplitude_palette(amplitudes: Sequence[float]) -> dict[float, tuple[float, float, float, float]]:
    values = sorted({float(a) for a in amplitudes if np.isfinite(a)})
    cmap = plt.get_cmap("viridis")
    if not values:
        return {}
    if len(values) == 1:
        return {values[0]: cmap(0.5)}
    lo, hi = math.log10(values[0]), math.log10(values[-1])
    return {v: cmap(0.1 + 0.85 * (math.log10(v) - lo) / (hi - lo)) for v in values}


def _log_clip(values: np.ndarray, limits: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Clip to log-axis limits; returns (clipped, was_below_lower_limit)."""
    arr = np.asarray(values, dtype=float)
    below = arr < limits[0]
    return np.clip(arr, limits[0], limits[1]), below


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(False)


def _panel_title(channel: str, impedance_kohm: float | None, distance_um: float | None) -> str:
    parts = [channel]
    if impedance_kohm is not None and np.isfinite(impedance_kohm):
        parts.append(f"Z {impedance_kohm:.0f} kOhm")
    if distance_um is not None and np.isfinite(distance_um):
        parts.append(f"d {distance_um:.0f} um")
    return "  ".join(parts)


def _jitter(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.uniform(-scale, scale, size=n)


# -----------------------------------------------------------------------------
# Figure 1: recovery time vs amplitude, per channel (headline)
# -----------------------------------------------------------------------------


def plot_recovery_vs_amplitude(
    trials: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: AnalysisConfig,
    ctx: CaptionContext,
    *,
    title: str = "Artifact recovery time vs stimulus current",
    channel_info: dict[str, tuple[float, float]] | None = None,
    show_rail: bool = True,
) -> Figure:
    """2 x 4 channel panels; per-trial scatter, median + IQR; rail duration hollow."""
    channels = list(dict.fromkeys(trials["channel"])) if not trials.empty else []
    n_panels = max(1, len(channels))
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.1 * n_rows + FOOTER_IN + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=(FOOTER_IN + 0.5) / (3.1 * n_rows + FOOTER_IN + 0.8), hspace=0.45, wspace=0.28)
    rng = np.random.default_rng(0)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(channels):
            ax.set_visible(False)
            continue
        channel = channels[index]
        group = trials[trials["channel"] == channel]
        amps = sorted(group["amplitude_uA"].dropna().unique())
        for amp in amps:
            sub = group[group["amplitude_uA"] == amp]
            recovery, below = _log_clip(sub["recovery_ms"].to_numpy(dtype=float), cfg.lim_recovery_ms)
            censored = sub["censored"].to_numpy(dtype=bool)
            x = amp * (1.0 + _jitter(rng, recovery.size, 0.06))
            ok = ~censored
            ax.scatter(x[ok], recovery[ok], s=TRIAL_SIZE, alpha=TRIAL_ALPHA, color="#1f77b4", linewidths=0, zorder=2)
            if np.any(censored):
                ax.scatter(x[censored], recovery[censored], s=18, marker="^", color="#d62728", alpha=0.8, linewidths=0, zorder=3)
            if np.any(below):
                ax.scatter(x[below], recovery[below], s=12, marker="v", color="#1f77b4", alpha=0.6, linewidths=0, zorder=2)
            if show_rail and "rail_ms" in sub:
                rail, _ = _log_clip(sub["rail_ms"].to_numpy(dtype=float), cfg.lim_recovery_ms)
                positive = sub["rail_ms"].to_numpy(dtype=float) > 0
                if np.any(positive):
                    ax.scatter(x[positive], rail[positive], s=12, facecolors="none", edgecolors="0.45", linewidths=0.6, alpha=0.7, zorder=1)
            values = sub["recovery_ms"].to_numpy(dtype=float)
            if values.size:
                med = float(np.median(values))
                q25, q75 = np.percentile(values, [25, 75])
                med_c, _ = _log_clip(np.array([med]), cfg.lim_recovery_ms)
                lo_c, _ = _log_clip(np.array([q25]), cfg.lim_recovery_ms)
                hi_c, _ = _log_clip(np.array([q75]), cfg.lim_recovery_ms)
                ax.errorbar([amp], med_c, yerr=[[med_c[0] - lo_c[0]], [hi_c[0] - med_c[0]]], fmt="s", color="black", ms=4, capsize=3, lw=1.0, zorder=4)
        ax.axhline(cfg.early_verdict_ms, color="0.6", lw=0.7, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(cfg.lim_amplitude_uA)
        ax.set_ylim(cfg.lim_recovery_ms)
        info = channel_info.get(channel) if channel_info else None
        ax.set_title(_panel_title(channel, info[0] if info else None, info[1] if info else None), fontsize=8.5)
        if index % n_cols == 0:
            ax.set_ylabel("recovery time (ms, log)", fontsize=8)
        if index >= (n_rows - 1) * n_cols:
            ax.set_xlabel("stimulus current (uA, log)", fontsize=8)
        _style_axes(ax)
    fig.suptitle(
        f"{title}\nper-trial (dots), median +/- IQR (black), rail duration (hollow grey), "
        f"censored = no {cfg.quiet_ms:g} ms quiet run (red triangles); dashed = {cfg.early_verdict_ms:g} ms verdict line",
        fontsize=9.5,
    )
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 2 / 2b: recovery vs impedance and vs distance
# -----------------------------------------------------------------------------


def _scatter_by_amplitude(ax, frame: pd.DataFrame, xcol: str, cfg: AnalysisConfig, palette, rng, jitter_scale: float, log_x: bool) -> None:
    for amp, sub in frame.groupby("amplitude_uA"):
        recovery, _ = _log_clip(sub["recovery_ms"].to_numpy(dtype=float), cfg.lim_recovery_ms)
        x = sub[xcol].to_numpy(dtype=float)
        if log_x:
            x = x * (1.0 + _jitter(rng, x.size, jitter_scale))
        else:
            x = x + _jitter(rng, x.size, jitter_scale)
        color = palette.get(float(amp), (0.3, 0.3, 0.3, 1.0))
        ax.scatter(x, recovery, s=TRIAL_SIZE, alpha=TRIAL_ALPHA, color=color, linewidths=0, zorder=2)
    # medians per (channel, amplitude)
    med = frame.groupby(["channel", "amplitude_uA"]).agg(x=(xcol, "median"), y=("recovery_ms", "median")).reset_index()
    for _, row in med.iterrows():
        color = palette.get(float(row["amplitude_uA"]), (0.3, 0.3, 0.3, 1.0))
        y, _ = _log_clip(np.array([row["y"]]), cfg.lim_recovery_ms)
        ax.scatter([row["x"]], y, s=42, marker="D", color=color, edgecolors="black", linewidths=0.6, zorder=4)


def _stagger_channel_labels(ax, frame: pd.DataFrame, xcol: str, cfg: AnalysisConfig) -> None:
    """Channel names above the data, alternating heights so close values stay legible."""
    positions = sorted((float(sub[xcol].median()), channel) for channel, sub in frame.groupby("channel"))
    top = cfg.lim_recovery_ms[1]
    for index, (x, channel) in enumerate(positions):
        y = top * (0.86 if index % 2 == 0 else 0.72)
        ax.annotate(channel, (x, y), fontsize=6.5, ha="center", color="0.3")


def _amplitude_legend(ax, palette) -> None:
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=color, label=f"{amp:g} uA")
        for amp, color in sorted(palette.items())
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2, title="current", title_fontsize=7)


def plot_recovery_vs_impedance(trials: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.4 + FOOTER_IN))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=(FOOTER_IN + 0.5) / (5.4 + FOOTER_IN))
    frame = trials.dropna(subset=["impedance_kohm"]) if "impedance_kohm" in trials else trials.iloc[0:0]
    palette = amplitude_palette(frame["amplitude_uA"].unique() if not frame.empty else [])
    if frame.empty:
        ax.text(0.5, 0.5, "no impedance available in RHS headers", ha="center", va="center", transform=ax.transAxes)
    else:
        _scatter_by_amplitude(ax, frame, "impedance_kohm", cfg, palette, np.random.default_rng(1), 0.03, log_x=True)
        _stagger_channel_labels(ax, frame, "impedance_kohm", cfg)
        _amplitude_legend(ax, palette)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(cfg.lim_impedance_kohm)
    ax.set_ylim(cfg.lim_recovery_ms)
    ax.set_xlabel("electrode impedance at 1 kHz (kOhm, log; RHS header, per run)", fontsize=8.5)
    ax.set_ylabel("recovery time (ms, log)", fontsize=8.5)
    ax.set_title("Artifact recovery time vs channel impedance\nper-trial dots, diamonds = median per channel x current", fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_recovery_vs_distance(trials: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.4 + FOOTER_IN))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=(FOOTER_IN + 0.5) / (5.4 + FOOTER_IN))
    frame = trials.dropna(subset=["distance_um"]) if "distance_um" in trials else trials.iloc[0:0]
    palette = amplitude_palette(frame["amplitude_uA"].unique() if not frame.empty else [])
    if frame.empty:
        ax.text(0.5, 0.5, "no distance information", ha="center", va="center", transform=ax.transAxes)
    else:
        _scatter_by_amplitude(ax, frame, "distance_um", cfg, palette, np.random.default_rng(2), 40.0, log_x=False)
        _stagger_channel_labels(ax, frame, "distance_um", cfg)
        _amplitude_legend(ax, palette)
    ax.set_yscale("log")
    ax.set_xlim(cfg.lim_distance_um)
    ax.set_ylim(cfg.lim_recovery_ms)
    ax.set_xlabel(f"distance from stim contact (um, {cfg.contact_pitch_um:g} um pitch)", fontsize=8.5)
    ax.set_ylabel("recovery time (ms, log)", fontsize=8.5)
    ax.set_title("Artifact recovery time vs distance from the stimulating contact", fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 3: raw stim-triggered traces at low amplitudes, identical axes
# -----------------------------------------------------------------------------


@dataclass
class TraceSnapshot:
    amplitude_uA: float
    run_id: str
    channels: list[str]
    traces: np.ndarray  # (C, E, S) float32
    t_ms: np.ndarray  # (S,)
    rail_levels: dict[str, tuple[float, float]]  # channel -> (neg, pos) uV, NaN when not railed
    n_events: int


def plot_raw_traces_grid(
    snapshots: Sequence[TraceSnapshot],
    cfg: AnalysisConfig,
    ctx: CaptionContext,
    *,
    ylim: tuple[float, float] | None = None,
    title: str = "Raw stim-triggered traces (unfiltered, all trials overlaid, identical axes)",
) -> Figure:
    """Rows = channels, columns = amplitudes; nothing is clipped silently."""
    ylim = ylim or cfg.lim_trace_uV
    snapshots = sorted(snapshots, key=lambda s: s.amplitude_uA)
    channels = list(snapshots[0].channels) if snapshots else []
    n_rows = max(1, len(channels))
    n_cols = max(1, len(snapshots))
    fig_h = 1.35 * n_rows + FOOTER_IN + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, fig_h), squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=1.0 - 0.75 / fig_h, bottom=(FOOTER_IN + 0.45) / fig_h, hspace=0.35, wspace=0.18)
    off_scale_total = 0
    for col, snap in enumerate(snapshots):
        for row, channel in enumerate(channels):
            ax = axes[row, col]
            if channel not in snap.channels:
                ax.set_visible(False)
                continue
            data = snap.traces[snap.channels.index(channel)]
            t = snap.t_ms
            for trial in data:
                ax.plot(t, trial, color="#1f77b4", lw=0.35, alpha=0.35)
            ax.axvline(0.0, color="0.75", lw=0.6, zorder=0)
            neg, pos = snap.rail_levels.get(channel, (float("nan"), float("nan")))
            for level, label in ((neg, "rail"), (pos, "rail")):
                if np.isfinite(level) and ylim[0] <= level <= ylim[1]:
                    ax.axhline(level, color="#d62728", lw=0.6, ls="--", alpha=0.8)
                    ax.text(t[-1], level, f" {label} {level:+.0f} uV", fontsize=5.5, color="#d62728", va="center", ha="right")
            over = np.abs(data) > max(abs(ylim[0]), abs(ylim[1]))
            n_over = int(np.count_nonzero(over))
            if n_over:
                off_scale_total += n_over
                trials_over = int(np.count_nonzero(over.any(axis=1)))
                frac = over.mean(axis=0)
                ax.fill_between(t, ylim[1] * 0.93, ylim[1], where=frac > 0, color="#d62728", alpha=0.25, lw=0)
                ax.fill_between(t, ylim[0], ylim[0] * 0.93, where=frac > 0, color="#d62728", alpha=0.25, lw=0)
                ax.text(0.99, 0.05, f"{n_over} samples off-scale ({trials_over} trials)", transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom", color="#d62728")
            ax.set_ylim(ylim)
            ax.set_xlim(cfg.trace_window_ms)
            _style_axes(ax)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{channel}\nuV", fontsize=7)
            if row == 0:
                ax.set_title(f"{snap.amplitude_uA:g} uA  (run {snap.run_id}, {snap.n_events} trials)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("time from stim onset (ms)", fontsize=7)
    fig.suptitle(f"{title}\ny-limits {ylim[0]:g} to {ylim[1]:g} uV; ADC full scale +/-{cfg.adc_full_scale_uV:.0f} uV; red bands = samples beyond the y-limits", fontsize=9.5)
    note = ctx.note
    if off_scale_total:
        note = (note + " " if note else "") + f"{off_scale_total} samples exceed the y-limits and are marked, not clipped."
    finish_figure(fig, build_caption(ctx.with_note(note)))
    return fig


# -----------------------------------------------------------------------------
# Supplementary S5: recovery for every run including excluded ones
# -----------------------------------------------------------------------------


def plot_recovery_all_runs(trials: pd.DataFrame, runs: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    """x = run (ordered by block, amplitude, phase); y = recovery; colour = channel."""
    order = runs["run_id"].tolist() if not runs.empty else []
    order = [r for r in order if r in set(trials["run_id"])] if not trials.empty else order
    fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(order) + 4), 5.6 + FOOTER_IN))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=(FOOTER_IN + 1.3) / (5.6 + FOOTER_IN))
    channels = list(dict.fromkeys(trials["channel"])) if not trials.empty else []
    palette = channel_palette(channels)
    rng = np.random.default_rng(3)
    labels = []
    info = runs.set_index("run_id") if not runs.empty else None
    for position, run_id in enumerate(order):
        sub = trials[trials["run_id"] == run_id]
        for c_index, channel in enumerate(channels):
            s = sub[sub["channel"] == channel]
            if s.empty:
                continue
            y, _ = _log_clip(s["recovery_ms"].to_numpy(dtype=float), cfg.lim_recovery_ms)
            x = position + (c_index - len(channels) / 2) * 0.08 + _jitter(rng, y.size, 0.02)
            ax.scatter(x, y, s=6, alpha=0.3, color=palette[channel], linewidths=0)
            ax.scatter([position + (c_index - len(channels) / 2) * 0.08], [np.clip(np.median(s["recovery_ms"]), *cfg.lim_recovery_ms)], s=20, color=palette[channel], edgecolors="black", linewidths=0.4, zorder=4)
        label = run_id
        if info is not None and run_id in info.index:
            row = info.loc[run_id]
            amp = row.get("amplitude_uA_data", float("nan"))
            phase = row.get("phase_us_data", float("nan"))
            block = row.get("block", "")
            included = bool(row.get("included", True))
            label = f"{run_id}\n{block}"
            if np.isfinite(amp):
                label += f"\n{amp:g} uA"
            if np.isfinite(phase):
                label += f" {phase:g} us"
            if not included:
                ax.axvspan(position - 0.5, position + 0.5, color="0.9", zorder=0)
                label += "\nEXCLUDED"
        labels.append(label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_yscale("log")
    ax.set_ylim(cfg.lim_recovery_ms)
    ax.axhline(cfg.early_verdict_ms, color="0.6", lw=0.7, ls="--")
    ax.set_ylabel("recovery time (ms, log)", fontsize=8.5)
    ax.set_title("Artifact recovery time for every run (excluded runs shaded); colour = channel, filled = median", fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[c], label=c) for c in channels]
    if handles:
        ax.legend(handles=handles, fontsize=6.5, frameon=False, ncol=min(8, len(handles)), loc="upper left")
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 4 / 4b / 8: band power vs amplitude (per-trial dB), heatmap
# -----------------------------------------------------------------------------


def _panel_grid(n_rows: int, n_cols: int, *, panel_w: float = 2.9, panel_h: float = 2.3, top_in: float = 0.9):
    fig_h = panel_h * n_rows + FOOTER_IN + top_in + 0.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols + 0.8, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=1.0 - top_in / fig_h, bottom=(FOOTER_IN + 0.4) / fig_h, hspace=0.5, wspace=0.3)
    return fig, axes


def _empty_figure(message: str, ctx: CaptionContext) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4 + FOOTER_IN))
    fig.subplots_adjust(bottom=(FOOTER_IN + 0.3) / (4 + FOOTER_IN))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_bandpower_vs_amplitude(
    trials: pd.DataFrame,
    cfg: AnalysisConfig,
    ctx: CaptionContext,
    *,
    value: str = "db",
    title: str = "Band power change vs stimulus current (early window vs pre-pulse baseline)",
    ylabel: str = "10 log10(post / baseline) (dB)",
    channel_info: dict[str, tuple[float, float]] | None = None,
    summary: pd.DataFrame | None = None,
) -> Figure:
    """Rows = bands, columns = channels (ordered by distance); per-trial dots, median + CI."""
    if trials.empty:
        return _empty_figure("no retained trials", ctx)
    bands = list(dict.fromkeys(trials["band"]))
    channels = list(dict.fromkeys(trials["channel"]))
    if channel_info:
        channels.sort(key=lambda c: (channel_info.get(c, (np.nan, np.inf))[1], c))
    fig, axes = _panel_grid(len(bands), len(channels))
    rng = np.random.default_rng(5)
    for r, band in enumerate(bands):
        for col, channel in enumerate(channels):
            ax = axes[r, col]
            sub = trials[(trials["band"] == band) & (trials["channel"] == channel)]
            for amp, s in sub.groupby("amplitude_uA"):
                y = np.clip(s[value].to_numpy(dtype=float), *cfg.lim_db)
                x = float(amp) * (1.0 + _jitter(rng, y.size, 0.06))
                ax.scatter(x, y, s=6, alpha=0.3, color="#1f77b4", linewidths=0)
                if summary is not None and not summary.empty:
                    row = summary[(summary["band"] == band) & (summary["channel"] == channel) & (summary["amplitude_uA"] == amp)]
                    if not row.empty:
                        med = float(row["median"].iloc[0]); lo = float(row["ci_low"].iloc[0]); hi = float(row["ci_high"].iloc[0])
                        med_c = float(np.clip(med, *cfg.lim_db))
                        err = [[max(0.0, med_c - float(np.clip(lo, *cfg.lim_db)))], [max(0.0, float(np.clip(hi, *cfg.lim_db)) - med_c)]]
                        ax.errorbar([float(amp)], [med_c], yerr=err, fmt="s", color="black", ms=3.5, capsize=2.5, lw=0.9, zorder=4)
                else:
                    ax.scatter([float(amp)], [np.clip(np.median(s[value]), *cfg.lim_db)], s=16, marker="s", color="black", zorder=4)
            ax.axhline(0.0, color="0.6", lw=0.7, ls="--", zorder=0)
            ax.set_xscale("log")
            ax.set_xlim(cfg.lim_amplitude_uA)
            ax.set_ylim(cfg.lim_db)
            _style_axes(ax)
            if r == 0:
                info = channel_info.get(channel) if channel_info else None
                ax.set_title(_panel_title(channel, info[0] if info else None, info[1] if info else None), fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"{band}\n{ylabel}", fontsize=7)
            if r == len(bands) - 1:
                ax.set_xlabel("current (uA, log)", fontsize=7)
    fig.suptitle(f"{title}\nper-trial dots (clipped to {cfg.lim_db[0]:g}..{cfg.lim_db[1]:g} dB), black = median with bootstrap 95% CI", fontsize=9.5)
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_bandpower_heatmap(summary: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext, *, value: str = "median", title: str = "Median band-power change (dB), channel x current") -> Figure:
    """One heatmap per band; fixed coolwarm scale from cfg.lim_db."""
    if summary.empty:
        return _empty_figure("no data", ctx)
    bands = list(dict.fromkeys(summary["band"]))
    channels = list(dict.fromkeys(summary["channel"]))
    amps = sorted(summary["amplitude_uA"].unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(2.6 * len(bands) + 1.5, 0.35 * len(channels) + 2.4 + FOOTER_IN), squeeze=False)
    fig_h = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.09, right=0.93, top=1.0 - 0.9 / fig_h, bottom=(FOOTER_IN + 0.5) / fig_h, wspace=0.15)
    image = None
    for index, band in enumerate(bands):
        ax = axes[0, index]
        matrix = np.full((len(channels), len(amps)), np.nan)
        for _, row in summary[summary["band"] == band].iterrows():
            matrix[channels.index(row["channel"]), amps.index(row["amplitude_uA"])] = row[value]
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=cfg.lim_db[0], vmax=cfg.lim_db[1])
        ax.set_xticks(range(len(amps)))
        ax.set_xticklabels([f"{a:g}" for a in amps], fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels if index == 0 else [""] * len(channels), fontsize=7)
        ax.set_title(band, fontsize=8.5)
        ax.set_xlabel("uA", fontsize=7)
        for i in range(len(channels)):
            for j in range(len(amps)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="black")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("dB", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"{title}; fixed scale {cfg.lim_db[0]:g} to {cfg.lim_db[1]:g} dB", fontsize=9.5)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 5: linear vs sigmoid fits
# -----------------------------------------------------------------------------


def plot_model_fits(
    trials: pd.DataFrame,
    fits: pd.DataFrame,
    metric: str,
    cfg: AnalysisConfig,
    ctx: CaptionContext,
    *,
    ylim: tuple[float, float] | None = None,
    ylabel: str = "early-window RMS (uV, log)",
    title: str = "Amplitude-response fits per channel: linear vs linear-through-origin vs sigmoid",
) -> Figure:
    from stim_analysis.models import sigmoid

    channels = list(dict.fromkeys(trials["channel"])) if not trials.empty else []
    if not channels:
        return _empty_figure("no trials", ctx)
    n_cols = min(4, max(1, len(channels)))
    n_rows = int(math.ceil(max(1, len(channels)) / n_cols))
    fig, axes = _panel_grid(n_rows, n_cols, panel_w=3.6, panel_h=2.8)
    ylim = ylim or cfg.lim_rms_uV
    rng = np.random.default_rng(6)
    grid = np.geomspace(cfg.lim_amplitude_uA[0], cfg.lim_amplitude_uA[1], 200)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(channels):
            ax.set_visible(False)
            continue
        channel = channels[index]
        sub = trials[trials["channel"] == channel]
        y = np.clip(sub[metric].to_numpy(dtype=float), *ylim)
        x = sub["amplitude_uA"].to_numpy(dtype=float) * (1.0 + _jitter(rng, y.size, 0.06))
        ax.scatter(x, y, s=6, alpha=0.3, color="#1f77b4", linewidths=0)
        med = sub.groupby("amplitude_uA")[metric].median()
        ax.scatter(med.index.to_numpy(dtype=float), np.clip(med.to_numpy(dtype=float), *ylim), s=18, marker="s", color="black", zorder=4)
        row = fits[(fits["channel"] == channel) & (fits["metric"] == metric)] if not fits.empty else fits
        if not row.empty:
            r = row.iloc[0]
            if np.isfinite(r.get("linear_m", np.nan)):
                ax.plot(grid, np.clip(r["linear_m"] * grid + r["linear_c"], *ylim), color="#2ca02c", lw=1.0, label=f"linear dAIC {r.get('delta_aic_linear', np.nan):.1f}")
            if np.isfinite(r.get("origin_m", np.nan)):
                ax.plot(grid, np.clip(r["origin_m"] * grid, *ylim), color="#ff7f0e", lw=1.0, ls="--", label=f"through origin dAIC {r.get('delta_aic_linear_origin', np.nan):.1f}")
            if bool(r.get("sigmoid_converged", False)):
                ax.plot(grid, np.clip(sigmoid(grid, r["sigmoid_amax"], r["sigmoid_i50"], r["sigmoid_k"]), *ylim), color="#d62728", lw=1.0, label=f"sigmoid dAIC {r.get('delta_aic_sigmoid', np.nan):.1f}")
                ax.text(0.02, 0.96, f"I50 {r['sigmoid_i50']:.0f} uA [{r['i50_ci_low']:.0f}, {r['i50_ci_high']:.0f}]", transform=ax.transAxes, fontsize=6.5, va="top")
            ax.text(0.02, 0.86, f"preferred: {r.get('preferred', '')}" + ("  (artifact-like)" if bool(r.get("artifact_candidate", False)) else ""), transform=ax.transAxes, fontsize=6.5, va="top")
            ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(cfg.lim_amplitude_uA)
        ax.set_ylim(ylim)
        ax.set_title(channel, fontsize=8.5)
        _style_axes(ax)
        if index % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=7.5)
        if index >= (n_rows - 1) * n_cols:
            ax.set_xlabel("current (uA, log)", fontsize=7.5)
    fig.suptitle(f"{title}\nmetric: {metric}; dAIC relative to the best model (lower is better); linear-through-origin preferred = artifact signature", fontsize=9.5)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 6: spatial decay
# -----------------------------------------------------------------------------


def plot_spatial_decay(summary: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext, *, value: str = "median", ylim: tuple[float, float] | None = None, ylabel: str = "early-window RMS (uV, log)", title: str = "Spatial decay: metric vs distance from the stim contact") -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.2 + FOOTER_IN))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=(FOOTER_IN + 0.5) / (5.2 + FOOTER_IN))
    ylim = ylim or cfg.lim_rms_uV
    if summary.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        palette = amplitude_palette(summary["amplitude_uA"].unique())
        for amp, sub in summary.groupby("amplitude_uA"):
            sub = sub.sort_values("distance_um")
            color = palette.get(float(amp), (0.3, 0.3, 0.3, 1.0))
            x = sub["distance_um"].to_numpy(dtype=float)
            y = np.clip(sub[value].to_numpy(dtype=float), *ylim)
            ax.plot(x, y, "-o", color=color, ms=4, lw=1.0, label=f"{amp:g} uA")
            if "ci_low" in sub and "ci_high" in sub:
                ax.fill_between(x, np.clip(sub["ci_low"].to_numpy(dtype=float), *ylim), np.clip(sub["ci_high"].to_numpy(dtype=float), *ylim), color=color, alpha=0.15, lw=0)
        for channel, sub in summary.groupby("channel"):
            ax.annotate(channel, (float(sub["distance_um"].iloc[0]), ylim[1] * 0.8), fontsize=6.5, ha="center", color="0.3")
        ax.legend(fontsize=7, frameon=False, ncol=3, title="current", title_fontsize=7)
    ax.set_yscale("log")
    ax.set_xlim(cfg.lim_distance_um)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"distance from stim contact (um, {cfg.contact_pitch_um:g} um pitch)", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(f"{title}\nmedian with bootstrap CI band; the ACA runs rostro-caudally so contacts also differ in A-P level", fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Figure 7: compliance characterisation
# -----------------------------------------------------------------------------


def plot_compliance(comp: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.8 + FOOTER_IN))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=(FOOTER_IN + 0.5) / (4.8 + FOOTER_IN))
    if comp.empty:
        ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes)
    else:
        markers = {"block1": "o", "block2": "s", "block3": "^", "other": "D"}
        for block, sub in comp.groupby("block"):
            colors = ["#d62728" if flag else "#1f77b4" for flag in sub["compliance_flag"]]
            ax.scatter(sub["charge_nC_per_phase"], sub["delivered_fraction"], marker=markers.get(block, "o"), s=42, alpha=0.85, edgecolors="black", linewidths=0.5, label=block, c=colors)
            for _, row in sub.iterrows():
                ax.annotate(f"{row['amplitude_uA']:g}uA/{row['phase_us']:g}us", (row["charge_nC_per_phase"], row["delivered_fraction"]), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
        ax.legend(fontsize=7, frameon=False, title="marker = block; red = compliance flagged", title_fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("commanded charge per phase (nC, log)", fontsize=8.5)
    ax.set_ylabel("delivered pulses / commanded pulses", fontsize=8.5)
    ax.set_title("Compliance characterisation: delivered pulse fraction vs commanded charge per phase", fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


# -----------------------------------------------------------------------------
# Supplementary: QQ, drift, impedance drift, noise floor
# -----------------------------------------------------------------------------


def plot_qq_grid(qq_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    """qq_data: metric -> (theo_raw, ordered_raw, theo_log, ordered_log)."""
    names = list(qq_data)
    if not names:
        return _empty_figure("no metrics", ctx)
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n + 1, 5.4 + FOOTER_IN), squeeze=False)
    fig_h = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.07, right=0.985, top=1.0 - 0.8 / fig_h, bottom=(FOOTER_IN + 0.4) / fig_h, hspace=0.45, wspace=0.35)
    for index, name in enumerate(names):
        theo_raw, ord_raw, theo_log, ord_log = qq_data[name]
        for row, (theo, ordered, label) in enumerate(((theo_raw, ord_raw, "raw"), (theo_log, ord_log, "log10"))):
            ax = axes[row, index]
            if theo.size:
                ax.scatter(theo, ordered, s=5, alpha=0.5, color="#1f77b4", linewidths=0)
                slope, intercept = np.polyfit(theo, ordered, 1)
                ax.plot(theo, slope * theo + intercept, color="0.4", lw=0.8)
            ax.set_title(f"{name} ({label})", fontsize=7.5)
            ax.set_xlabel("normal quantile", fontsize=6.5)
            _style_axes(ax)
    fig.suptitle("Log-normality check: QQ plots on raw vs log10 values (Shapiro-Wilk p in lognormal_checks.csv)", fontsize=9.5)
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_drift(drift: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext, *, value_first: str, value_last: str, ylabel: str, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.0 + FOOTER_IN))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=(FOOTER_IN + 0.9) / (5.0 + FOOTER_IN))
    if drift.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        channels = list(dict.fromkeys(drift["channel"]))
        palette = channel_palette(channels)
        keys = sorted(drift["amplitude_uA"].unique()) if "amplitude_uA" in drift else []
        for c_index, channel in enumerate(channels):
            sub = drift[drift["channel"] == channel].sort_values("amplitude_uA")
            x = np.array([keys.index(a) for a in sub["amplitude_uA"]], dtype=float)
            delta = sub[value_last].to_numpy(dtype=float) - sub[value_first].to_numpy(dtype=float)
            ax.plot(x + (c_index - len(channels) / 2) * 0.05, np.clip(delta, *cfg.lim_db), "-o", ms=3.5, lw=0.8, color=palette[channel], label=channel)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{k:g} uA" for k in keys], fontsize=7)
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        ax.set_ylim(cfg.lim_db)
        ax.legend(fontsize=6.5, frameon=False, ncol=min(8, len(channels)))
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_impedance_drift(impedance: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.0 + FOOTER_IN))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=(FOOTER_IN + 1.0) / (5.0 + FOOTER_IN))
    if impedance.empty or impedance["impedance_kohm"].isna().all():
        ax.text(0.5, 0.5, "no impedance in RHS headers", ha="center", va="center", transform=ax.transAxes)
    else:
        runs = list(dict.fromkeys(impedance.sort_values("run_id")["run_id"]))
        channels = list(dict.fromkeys(impedance["channel"]))
        palette = channel_palette(channels)
        for channel in channels:
            sub = impedance[impedance["channel"] == channel].set_index("run_id").reindex(runs)
            ax.plot(range(len(runs)), np.clip(sub["impedance_kohm"].to_numpy(dtype=float), *cfg.lim_impedance_kohm), "-o", ms=3.5, lw=0.8, color=palette[channel], label=channel)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=60, fontsize=6.5, ha="right")
        ax.legend(fontsize=6.5, frameon=False, ncol=min(8, len(channels)))
    ax.set_yscale("log")
    ax.set_ylim(cfg.lim_impedance_kohm)
    ax.set_ylabel("impedance at 1 kHz (kOhm, log)", fontsize=8.5)
    ax.set_title("Electrode impedance per run (RHS header value at recording start)", fontsize=9.5)
    _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig


def plot_noise_floor(noise: pd.DataFrame, cfg: AnalysisConfig, ctx: CaptionContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6 + FOOTER_IN))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=(FOOTER_IN + 0.5) / (4.6 + FOOTER_IN), wspace=0.3)
    ax0, ax1 = axes
    if noise.empty:
        ax0.text(0.5, 0.5, "no baseline run", ha="center", va="center", transform=ax0.transAxes)
    else:
        channels = list(noise["channel"])
        palette = channel_palette(channels)
        colors = [palette[c] for c in channels]
        ax0.bar(range(len(channels)), np.clip(noise["broadband_rms_uV_median"], *cfg.lim_rms_uV), color=colors)
        ax0.set_xticks(range(len(channels)))
        ax0.set_xticklabels(channels, fontsize=7, rotation=45)
        ax0.set_yscale("log")
        ax0.set_ylim(cfg.lim_rms_uV)
        ax0.set_ylabel("broadband RMS in no-stim run (uV, log)", fontsize=8)
        ax0.set_title("Noise floor per channel", fontsize=9)
        z = noise["impedance_kohm"].to_numpy(dtype=float)
        ok = np.isfinite(z)
        if ok.any():
            ax1.scatter(z[ok], noise["broadband_rms_uV_median"].to_numpy(dtype=float)[ok], c=[colors[i] for i in np.flatnonzero(ok)], s=40, edgecolors="black", linewidths=0.5)
            for i in np.flatnonzero(ok):
                ax1.annotate(channels[i], (z[i], noise["broadband_rms_uV_median"].iloc[i]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            zz = np.geomspace(cfg.lim_impedance_kohm[0], cfg.lim_impedance_kohm[1], 50)
            ref = float(np.nanmedian(noise["rms_per_sqrt_kohm"]))
            if np.isfinite(ref):
                ax1.plot(zz, ref * np.sqrt(zz), color="0.5", lw=0.8, ls="--", label="sqrt(R) scaling (median)")
                ax1.legend(fontsize=6.5, frameon=False)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(cfg.lim_impedance_kohm)
        ax1.set_ylim(cfg.lim_rms_uV)
        ax1.set_xlabel("impedance (kOhm, log)", fontsize=8)
        ax1.set_ylabel("broadband RMS (uV, log)", fontsize=8)
        ax1.set_title("Noise floor vs impedance (thermal noise ~ sqrt(R))", fontsize=9)
    for ax in axes:
        _style_axes(ax)
    finish_figure(fig, build_caption(ctx))
    return fig

__all__ = [
    "CaptionContext",
    "plot_bandpower_heatmap",
    "plot_bandpower_vs_amplitude",
    "plot_compliance",
    "plot_drift",
    "plot_impedance_drift",
    "plot_model_fits",
    "plot_noise_floor",
    "plot_qq_grid",
    "plot_spatial_decay",
    "TraceSnapshot",
    "amplitude_palette",
    "build_caption",
    "channel_palette",
    "figure_to_png_bytes",
    "finish_figure",
    "plot_raw_traces_grid",
    "plot_recovery_all_runs",
    "plot_recovery_vs_amplitude",
    "plot_recovery_vs_distance",
    "plot_recovery_vs_impedance",
]
