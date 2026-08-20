#!/usr/bin/env python3
"""Figures for Function 7.

Rendered to PNG bytes in memory so the pipeline can hand them to the UI as a
preview without writing anything, and to the atomic writer unchanged on save.
Matplotlib runs headless here; this module never imports ipywidgets.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import SessionResult

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _to_png(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _by_wiring_channel(rows: list[dict], value_key: str):
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        amplitude = row.get("amplitude_mA")
        value = row.get(value_key)
        if amplitude is None or value is None:
            continue
        if not np.isfinite(value):
            continue
        grouped[(row["wiring"], row["channel"])].append((float(amplitude), float(value)))
    return grouped


def dose_response_figure(result: SessionResult) -> bytes:
    """Evoked deflection against current, one panel per wiring configuration."""
    grouped = _by_wiring_channel(result.rows, "evoked_pp_uV")
    wirings = sorted({key[0] for key in grouped})
    if not wirings:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no dose-response points", ha="center", va="center")
        ax.axis("off")
        return _to_png(fig)

    columns = min(3, len(wirings))
    rows_count = int(np.ceil(len(wirings) / columns))
    fig, axes = plt.subplots(
        rows_count, columns, figsize=(5.2 * columns, 3.8 * rows_count), squeeze=False
    )

    for index, wiring in enumerate(wirings):
        ax = axes[index // columns][index % columns]
        channels = sorted({key[1] for key in grouped if key[0] == wiring})
        for channel_index, channel in enumerate(channels):
            points = sorted(grouped[(wiring, channel)])
            amplitudes = np.array([p[0] for p in points])
            values = np.array([p[1] for p in points])
            color = _COLORS[channel_index % len(_COLORS)]
            for polarity, marker in ((1, "o"), (-1, "s")):
                mask = np.sign(amplitudes) == polarity
                if mask.any():
                    ax.plot(
                        np.abs(amplitudes[mask]),
                        values[mask],
                        marker,
                        color=color,
                        label=f"{channel} {'anodic' if polarity > 0 else 'cathodic'}",
                        markersize=5,
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|current| (mA)")
        ax.set_ylabel("evoked p-p (uV)")
        ax.set_title(wiring, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=6, ncol=2)

    for index in range(len(wirings), rows_count * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(
        "Evoked deflection vs current. Circles anodic, squares cathodic. "
        "A straight line through the origin on linear axes is what coupling looks like.",
        fontsize=9,
    )
    fig.tight_layout()
    return _to_png(fig)


def waveform_figure(result: SessionResult) -> bytes:
    """Mean per-pulse waveform, a few amplitudes per configuration."""
    runs = [r for r in result.analysed if r.evoked and r.condition.amplitude_mA is not None]
    if not runs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no waveforms", ha="center", va="center")
        ax.axis("off")
        return _to_png(fig)

    by_wiring: dict[str, list] = defaultdict(list)
    for run in runs:
        by_wiring[run.condition.wiring.label].append(run)

    wirings = sorted(by_wiring)
    fig, axes = plt.subplots(len(wirings), 1, figsize=(7.5, 3.0 * len(wirings)), squeeze=False)

    for index, wiring in enumerate(wirings):
        ax = axes[index][0]
        selected = sorted(by_wiring[wiring], key=lambda r: abs(r.condition.amplitude_mA))
        step = max(1, len(selected) // 5)
        for color_index, run in enumerate(selected[::step][:5]):
            evoked = run.evoked[0]
            if evoked.mean_waveform_uV.size == 0:
                continue
            ax.plot(
                evoked.waveform_time_ms,
                evoked.mean_waveform_uV,
                linewidth=1.0,
                color=_COLORS[color_index % len(_COLORS)],
                label=f"{run.condition.amplitude_label} ({evoked.channel})",
            )
        ax.axvline(0.0, color="k", linewidth=0.6)
        ax.axvspan(0.0, result.config.pulse_width_ms, color="k", alpha=0.08)
        ax.set_xlabel("time from pulse onset (ms)")
        ax.set_ylabel("uV")
        ax.set_title(f"{wiring} -- mean of the pulses in each train", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    fig.suptitle(
        "Shaded band is the 5 ms pulse. Coupling stops with the pulse; a response outlasts it.",
        fontsize=9,
    )
    fig.tight_layout()
    return _to_png(fig)


def band_power_figure(result: SessionResult) -> bytes:
    """Band power change against current, comb-excluded estimate."""
    band_names = [name for name, _low, _high in result.config.bands]
    fig, axes = plt.subplots(1, len(band_names), figsize=(3.0 * len(band_names), 3.4), squeeze=False)

    for index, name in enumerate(band_names):
        ax = axes[0][index]
        grouped = _by_wiring_channel(result.rows, f"band_{name}_dB")
        wirings = sorted({key[0] for key in grouped})
        for wiring_index, wiring in enumerate(wirings):
            amplitudes: list[float] = []
            values: list[float] = []
            for (group_wiring, _channel), points in grouped.items():
                if group_wiring != wiring:
                    continue
                for amplitude, value in points:
                    amplitudes.append(abs(amplitude))
                    values.append(value)
            if amplitudes:
                ax.plot(
                    amplitudes,
                    values,
                    "o",
                    markersize=3.5,
                    color=_COLORS[wiring_index % len(_COLORS)],
                    label=wiring if index == 0 else None,
                )
        ax.axhline(0.0, color="k", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("|current| (mA)")
        if index == 0:
            ax.set_ylabel("train vs baseline (dB)")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=3)
    fig.suptitle("Band power change, pulse-harmonic bins excluded before integrating", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _to_png(fig)


def artifact_figure(result: SessionResult) -> bytes:
    """Latency against tail fraction: the two within-run coupling indicators."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    by_wiring: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in result.rows:
        latency = row.get("peak_latency_ms")
        tail = row.get("post_pulse_fraction")
        if latency is None or tail is None:
            continue
        if not (np.isfinite(latency) and np.isfinite(tail)):
            continue
        by_wiring[row["wiring"]].append((float(latency), float(tail)))

    for index, (wiring, points) in enumerate(sorted(by_wiring.items())):
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            "o",
            markersize=4,
            alpha=0.75,
            color=_COLORS[index % len(_COLORS)],
            label=wiring,
        )

    from .artifact import COUPLING_LATENCY_MS, COUPLING_TAIL_FRACTION

    ax.axvline(COUPLING_LATENCY_MS, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(COUPLING_TAIL_FRACTION, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("peak latency (ms)")
    ax.set_ylabel("fraction of energy after the pulse ends")
    ax.set_title(
        "Bottom-left quadrant = fast and stops with the pulse = coupling-like", fontsize=9
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    return _to_png(fig)


def render_all(result: SessionResult, output_dir: Path) -> list[tuple[Path, bytes]]:
    """Every figure, as (path, PNG bytes)."""
    return [
        (output_dir / "fig1_dose_response.png", dose_response_figure(result)),
        (output_dir / "fig2_pulse_waveforms.png", waveform_figure(result)),
        (output_dir / "fig3_band_power.png", band_power_figure(result)),
        (output_dir / "fig4_artifact_evidence.png", artifact_figure(result)),
    ]


__all__ = [
    "artifact_figure",
    "band_power_figure",
    "dose_response_figure",
    "render_all",
    "waveform_figure",
]
