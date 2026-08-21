#!/usr/bin/env python3
"""Text and CSV outputs for Function 7."""

from __future__ import annotations

import csv
import io
from collections import defaultdict

import numpy as np

from .pipeline import SessionResult


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _ordered_fieldnames(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def runs_csv(result: SessionResult) -> str:
    """One row per run and channel, every flag and the raw folder name included."""
    rows = result.rows
    if not rows:
        return "no rows\n"
    buffer = io.StringIO()
    fieldnames = _ordered_fieldnames(rows)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _format(row.get(key)) for key in fieldnames})
    return buffer.getvalue()


def peaks_csv(result: SessionResult) -> str:
    """One row per run, channel and detected peak (long format)."""
    rows = result.peak_rows
    if not rows:
        return "no peaks detected\n"
    buffer = io.StringIO()
    fieldnames = _ordered_fieldnames(rows)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _format(row.get(key)) for key in fieldnames})
    return buffer.getvalue()


def conditions_csv(result: SessionResult) -> str:
    """One row per wiring configuration and channel: the dose-response fit."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "wiring",
            "channel",
            "n_points",
            "slope_uV_per_mA",
            "intercept_uV",
            "r_squared",
            "linear_through_origin",
            "polarity_asymmetry",
        ]
    )
    for evidence in result.sweep:
        writer.writerow(
            [
                evidence.wiring_label,
                evidence.channel,
                evidence.n_points,
                _format(evidence.slope_uV_per_mA),
                _format(evidence.intercept_uV),
                _format(evidence.r_squared),
                evidence.linear_through_origin,
                _format(evidence.polarity_asymmetry),
            ]
        )
    return buffer.getvalue()


def verdict_text(result: SessionResult) -> str:
    """A readable summary of what the sweep found."""
    config = result.config
    lines: list[str] = []
    lines.append("Evoked-response sweep -- Function 7")
    lines.append("=" * 70)
    lines.append(f"Session folder : {config.session_folder}")
    lines.append(f"Runs analysed  : {len(result.analysed)} of {len(result.runs)}")
    lines.append(
        f"Protocol       : {config.expected_pulses} pulses, {config.pulse_width_ms:.0f} ms wide, "
        f"search from {config.search_start_s:.0f} s"
    )
    lines.append("")

    lines.append("Stimulus timing recovered from the amplifier trace")
    lines.append("-" * 70)
    lines.append("Nothing recorded the trigger (ANALOG-IN and DIGITAL-IN were disabled and the")
    lines.append("Keithley ran over serial), so onsets are fitted to the signal and checked")
    lines.append("against the known protocol.")
    periods = [r.train.period_s * 1000.0 for r in result.analysed if r.train]
    widths = [r.train.width_ms for r in result.analysed if r.train and np.isfinite(r.train.width_ms)]
    counts = [r.train.n_pulses for r in result.analysed if r.train]
    if periods:
        lines.append(
            f"  period  : median {np.median(periods):.1f} ms "
            f"({1000.0 / np.median(periods):.2f} Hz), range {min(periods):.1f}-{max(periods):.1f} ms"
        )
    if widths:
        lines.append(f"  width   : median {np.median(widths):.2f} ms (expected {config.pulse_width_ms:.0f} ms)")
    if counts:
        exact = sum(1 for c in counts if c == config.expected_pulses)
        lines.append(f"  pulses  : {exact}/{len(counts)} runs recovered exactly {config.expected_pulses}")
    lines.append("")

    timing_problems = [r for r in result.analysed if r.train and not r.train.ok]
    if timing_problems:
        lines.append(f"Runs with timing warnings ({len(timing_problems)}):")
        for run in timing_problems[:20]:
            lines.append(f"  {run.condition.raw_name}: {'; '.join(run.train.issues)}")
        lines.append("")

    if result.failed:
        lines.append(f"Runs not analysed ({len(result.failed)}):")
        for run in result.failed[:20]:
            lines.append(f"  {run.condition.raw_name}: {run.error}")
        lines.append("")

    lines.append("Dose-response and coupling evidence, per configuration and channel")
    lines.append("-" * 70)
    lines.append("A linear fit through the origin is what pure resistive coupling looks like;")
    lines.append("a neural response has a threshold and saturates. This reports the evidence,")
    lines.append("it does not decide.")
    for evidence in result.sweep:
        marker = "coupling-like" if evidence.linear_through_origin else "not linear"
        lines.append(
            f"  {evidence.wiring_label:34s} {evidence.channel:6s} n={evidence.n_points:2d} "
            f"slope={evidence.slope_uV_per_mA:9.1f} uV/mA r2={evidence.r_squared:5.3f} "
            f"intercept={evidence.intercept_uV:8.1f} uV  {marker}"
        )
    lines.append("")

    decade_suspects = sorted(
        {
            (str(row["run"]), row["decade_suspect_note"])
            for row in result.rows
            if row.get("decade_suspect")
        }
    )
    if decade_suspects:
        lines.append("Possible decade mislabels (flagged from the dose-response, never corrected)")
        lines.append("-" * 70)
        for run_name, note in decade_suspects:
            lines.append(f"  {run_name}: {note}")
        lines.append("")

    suspicion = defaultdict(list)
    for row in result.rows:
        if row.get("artifact_suspicion") is not None:
            suspicion[row["wiring"]].append(row["artifact_suspicion"])
    if suspicion:
        lines.append("Within-run coupling indicators (mean of two criteria, 1.0 = both met)")
        lines.append("-" * 70)
        for wiring, values in sorted(suspicion.items()):
            lines.append(f"  {wiring:40s} {np.mean(values):.2f}  (n={len(values)})")
        lines.append("")

    if result.peak_rows:
        lines.append("Post-pulse peaks, per wiring configuration")
        lines.append("-" * 70)
        lines.append("Peaks are detected after the nominal pulse plus a guard; a peak within")
        lines.append(f"{config.edge_flag_ms:.1f} ms of the measured off-edge is marked (edge?) because it may")
        lines.append("be the stimulator turn-off, not tissue. The legacy latency column still")
        lines.append("spans the whole response window and can land on the onset spike by design.")
        by_wiring_label: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in result.peak_rows:
            by_wiring_label[(row["wiring"], row["peak_label"])].append(row)
        for (wiring, label), group in sorted(
            by_wiring_label.items(), key=lambda item: (item[0][0], np.median([g["latency_ms"] for g in item[1]]))
        ):
            latencies = [g["latency_ms"] for g in group]
            amplitudes = [g["amp_median_uV"] for g in group if np.isfinite(g["amp_median_uV"])]
            present = sum(1 for g in group if g["present"])
            edge = sum(1 for g in group if g["edge_suspect"])
            amp_text = f"{np.median(amplitudes):+8.1f} uV" if amplitudes else "     n/a"
            edge_text = f"  ({edge} edge?)" if edge else ""
            lines.append(
                f"  {wiring:34s} {label:3s} at {np.median(latencies):5.1f} ms {amp_text} "
                f"present {present}/{len(group)}{edge_text}"
            )
        lines.append("")

    lines.append("Caveats carried from the recording settings")
    lines.append("-" * 70)
    lines.append("  The DSP high-pass sat at 0.777 Hz and the analog corner at 0.09 Hz, so a")
    lines.append("  true DC or vascular drift was removed before it reached the file. The")
    lines.append("  post-train measure only sees changes faster than about 0.1 Hz.")
    lines.append("  Band power is reported twice: comb-excluded (valid for every band) and")
    minimums = [
        row["band_gap_minimum_hz"]
        for row in result.rows
        if np.isfinite(row.get("band_gap_minimum_hz", float("nan")))
    ]
    if minimums:
        cutoff = max(minimums)
        blank = [
            name
            for name, low, _high in config.bands
            if not any(
                np.isfinite(row.get(f"band_{name}_dB_gap", float("nan"))) for row in result.rows
            )
        ]
        blank_text = ", ".join(blank) if blank else "no band"
        lines.append(
            f"  gap-based (needs >=3 cycles per gap: invalid below {cutoff:.1f} Hz at this"
        )
        lines.append(f"  session's intervals, leaving {blank_text} blank).")
    else:
        lines.append("  gap-based (blank for any band below three cycles per gap).")
    lines.append("  Amplitudes are parsed literally from folder names; suspected magnitude")
    lines.append("  mislabels are flagged, never corrected.")
    lines.append("")

    for note in result.notes:
        lines.append(f"Note: {note}")

    return "\n".join(lines) + "\n"


__all__ = ["conditions_csv", "peaks_csv", "runs_csv", "verdict_text"]
