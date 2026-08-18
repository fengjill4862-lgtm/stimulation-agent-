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


__all__ = [
    "CaptionContext",
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
