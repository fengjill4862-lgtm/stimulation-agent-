#!/usr/bin/env python3
"""Stimulus timing from oscilloscope captures with absolute timestamps.

The 20260821-style sessions record ground-truth timing outside the Intan:
each run folder holds a ``*.onset.txt`` with the absolute Unix epoch of the
first pulse command, and the session's ``oscilloscope/`` folder holds one
ASCII capture per run whose first column is the absolute epoch of every
sample. This module turns those into a ``PulseTrain`` so Function 7 does not
have to guess the train blindly from the amplifier trace.

Three clocks meet here, each doing what it is best at:

* ``onset.txt`` anchors the train absolutely (stim-host clock);
* the scope capture supplies every pulse's leading edge on that same clock --
  essential because the Keithley's serial overhead stretches the real period
  well past the labelled one (5.55 s measured vs 4.8 s labelled), so a
  label-based comb would drift by seconds across the train;
* the amplifier trace closes the last gap: the Intan folder name is only
  1 s precise and its PC clock can drift, so a single per-run clock offset is
  fitted by sliding the scope-derived onsets against the recording's envelope.

The scope's ``current_mA1`` column has an arbitrary probe scale (it read
~20 units on a 1 uA run). Only its TIMING is used, never its amplitude.
The captures are up to 256 MB with bursty sampling (dt from microseconds to
tens of ms), so parsing streams line-by-line, seeks by bisection on the
monotone timestamps, and never assumes a uniform grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import EvokedConfig
from .load import LoadedRun
from .pulses import PulseTrain, _rms_envelope, _robust_z

_RUN_START = re.compile(r"_(\d{6})_(\d{6})$")

# Rising-edge contrast half-window for the fine alignment: the envelope is
# high throughout a 0.3-0.5 s pulse, so scoring plain envelope height leaves
# the offset ambiguous across the whole pulse width. Scoring the CONTRAST
# across the putative edge (after minus before) peaks only at the edge.
_EDGE_CONTRAST_S = 0.1


@dataclass(frozen=True)
class ScopePulses:
    """Pulse timing extracted from one oscilloscope capture."""

    epochs_s: np.ndarray  # absolute epoch of each pulse's leading edge
    widths_s: np.ndarray  # above-threshold envelope duration per pulse
    period_s: float
    capture: Path
    issues: tuple[str, ...]
    # Waveform shape from slope segmentation (medians across pulses). The
    # scope trace is the voltage across the electrode: a current square
    # becomes a voltage ramp, so the phases live in the DERIVATIVE -- steady
    # positive slope = positive phase, decay = interphase, steady negative
    # slope = negative phase. lead_sign +1 = anodic-leading.
    lead_sign: int = 0
    phase1_s: float = float("nan")
    ipd_s: float = float("nan")
    phase2_s: float = float("nan")
    envelope_s: float = float("nan")  # onset to the second phase's extremum


def read_onset_epoch(run_folder: Path) -> float | None:
    """The absolute epoch of the first pulse command, from ``*.onset.txt``."""
    for path in sorted(run_folder.glob("*.onset.txt")):
        try:
            for line in path.read_text().splitlines():
                if line.startswith("pulse_onset_epoch_s="):
                    return float(line.split("=", 1)[1])
        except (OSError, ValueError):
            continue
    return None


def run_start_epoch(run_folder: Path) -> float | None:
    """The Intan recording start from the folder name, 1 s resolution."""
    match = _RUN_START.search(run_folder.name)
    if match is None:
        return None
    date_text, time_text = match.groups()
    try:
        started = datetime(
            2000 + int(date_text[0:2]), int(date_text[2:4]), int(date_text[4:6]),
            int(time_text[0:2]), int(time_text[2:4]), int(time_text[4:6]),
        )
    except ValueError:
        return None
    return started.timestamp()


def _line_epoch(line: str) -> float | None:
    parts = line.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def capture_span(path: Path) -> tuple[float, float] | None:
    """(start, end) epoch of a capture: filename stem plus a tail-read.

    Never reads the whole file; the last complete line is found by seeking
    close to the end.
    """
    if not path.stem.isdigit():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    start = float(path.stem)
    with path.open("rb") as handle:
        handle.seek(max(0, size - 8192))
        tail = handle.read().decode("ascii", errors="replace").splitlines()
    for line in reversed(tail):
        epoch = _line_epoch(line)
        if epoch is not None:
            return start, epoch
    return None


def find_scope_capture(scope_dir: Path, onset_epoch: float) -> Path | None:
    """The capture whose time span contains the onset; None when none does."""
    best: tuple[float, Path] | None = None
    for path in sorted(scope_dir.glob("*.txt")):
        span = capture_span(path)
        if span is None:
            continue
        start, end = span
        if start - 2.0 <= onset_epoch <= end + 2.0 and (best is None or start > best[0]):
            if start <= onset_epoch + 2.0:
                best = (start, path)
    return best[1] if best else None


def _seek_to_epoch(handle, size: int, target: float) -> None:
    """Position a text handle at a line boundary shortly before ``target``.

    Bisection on byte offsets, reading one line per probe -- the timestamps
    are monotone, so this replaces reading hundreds of MB with ~40 probes.
    """
    lo, hi = 0, size
    for _step in range(40):
        if hi - lo < 1 << 16:
            break
        mid = (lo + hi) // 2
        handle.seek(mid)
        handle.readline()  # discard the partial line
        epoch = _line_epoch(handle.readline())
        if epoch is None or epoch < target:
            lo = mid
        else:
            hi = mid
    handle.seek(lo)
    if lo:
        handle.readline()


def _segment_pulses(
    time_array: np.ndarray, values: np.ndarray, edges: list[float], pulse_width_s: float
) -> tuple[int, float, float, float, float]:
    """Median (lead_sign, phase1, ipd, phase2, envelope) across pulses.

    Works on the electrode-voltage trace: resample to a uniform 5 ms grid,
    smooth, and read phase boundaries from the voltage extrema and the last
    opposite-slope span. Returns lead_sign 0 and NaNs when indeterminate.
    """
    if time_array.size < 20 or not edges:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    grid = np.arange(time_array[0], time_array[-1], 0.005)
    if grid.size < 20:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    u_full = np.convolve(np.interp(grid, time_array, values), np.ones(9) / 9, mode="same")
    slope_full = np.gradient(u_full, grid)

    leads: list[int] = []
    phase1s: list[float] = []
    ipds: list[float] = []
    phase2s: list[float] = []
    envelopes: list[float] = []
    span_s = max(1.5, 6.0 * pulse_width_s)
    for edge in edges:
        window = (grid >= edge - 0.3) & (grid <= edge + span_s)
        if not window.any():
            continue
        g = grid[window]
        pre = window & (grid < edge)
        baseline = float(np.median(u_full[pre])) if pre.any() else float(u_full[window][0])
        u = u_full[window] - baseline
        s = slope_full[window]

        t_max = float(g[np.argmax(u)])
        t_min = float(g[np.argmin(u)])
        lead = 1 if t_max < t_min else -1
        t_first = min(t_max, t_min)
        t_second = max(t_max, t_min)
        if t_first <= edge - 0.05:
            continue

        # Second phase = the last span of opposite-sign slope ending near the
        # second extremum.
        opposite = -lead * s
        threshold = 0.35 * float(np.percentile(np.abs(s), 99))
        active = np.flatnonzero((opposite > threshold) & (g > t_first) & (g <= t_second + 0.05))
        phase2_start = float(g[active[0]]) if active.size else float("nan")

        leads.append(lead)
        phase1s.append(t_first - edge)
        envelopes.append(t_second - edge)
        if np.isfinite(phase2_start):
            ipds.append(phase2_start - t_first)
            phase2s.append(t_second - phase2_start)

    if not leads:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    lead_sign = 1 if sum(leads) > 0 else -1 if sum(leads) < 0 else 0
    return (
        lead_sign,
        float(np.median(phase1s)) if phase1s else float("nan"),
        float(np.median(ipds)) if ipds else float("nan"),
        float(np.median(phase2s)) if phase2s else float("nan"),
        float(np.median(envelopes)) if envelopes else float("nan"),
    )


def read_scope_pulse_epochs(
    path: Path,
    onset_epoch: float,
    expected_pulses: int,
    pulse_width_s: float,
    interval_s: float,
    *,
    pre_s: float = 5.0,
    threshold_ratio: float = 0.5,
    merge_gap_s: float = 1.0,
) -> ScopePulses:
    """Stream one capture and return the leading edge of every pulse."""
    window_start = onset_epoch - pre_s
    window_stop = onset_epoch + expected_pulses * (pulse_width_s + interval_s) * 1.5 + 10.0

    times: list[float] = []
    values: list[float] = []
    size = path.stat().st_size
    with path.open("r", errors="replace") as handle:
        _seek_to_epoch(handle, size, window_start)
        for line in handle:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                epoch = float(parts[0])
                value = float(parts[3])
            except ValueError:
                continue
            if epoch < window_start:
                continue
            if epoch > window_stop:
                break
            times.append(epoch)
            values.append(value)

    empty = np.zeros(0)
    if len(times) < 10:
        return ScopePulses(empty, empty, float("nan"), path, ("capture window is empty",))

    time_array = np.asarray(times)
    deviation = np.abs(np.asarray(values) - np.median(values))
    # Half of the near-maximum deviation, floored at 6 robust SD so a
    # noise-only capture yields zero events rather than garbage edges.
    mad_sd = 1.4826 * np.median(np.abs(deviation - np.median(deviation)))
    threshold = max(threshold_ratio * float(np.percentile(deviation, 99.5)), 6.0 * mad_sd)
    active = np.flatnonzero(deviation > threshold)
    if active.size == 0:
        return ScopePulses(empty, empty, float("nan"), path, ("no pulses above threshold",))

    # Group by TIME gaps (the grid is bursty; sample counts mean nothing).
    active_times = time_array[active]
    breaks = np.flatnonzero(np.diff(active_times) > merge_gap_s)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, active_times.size - 1]

    # The electrode voltage RAMPS under a current step, so the half-max
    # crossing sits well inside the pulse. Walk each event's start backwards
    # while the deviation stays above a low floor to reach the ramp's base.
    low_floor = max(3.0 * mad_sd, 0.05 * float(np.percentile(deviation, 99.5)))
    edges: list[float] = []
    widths: list[float] = []
    for lo, hi in zip(starts, stops):
        if hi - lo + 1 < 3:
            continue
        raw = int(active[lo])
        while (
            raw > 0
            and deviation[raw - 1] > low_floor
            and time_array[raw] - time_array[raw - 1] < merge_gap_s
        ):
            raw -= 1
        edge = float(time_array[raw])
        width = float(active_times[hi]) - edge
        if width < pulse_width_s / 10.0:
            continue
        edges.append(edge)
        widths.append(width)

    issues: list[str] = []
    if len(edges) != expected_pulses:
        issues.append(f"scope shows {len(edges)} pulses, label says {expected_pulses}")
    period = float(np.median(np.diff(edges))) if len(edges) >= 2 else float("nan")
    labelled_period = pulse_width_s + interval_s
    if np.isfinite(period) and labelled_period > 0 and abs(period - labelled_period) > 0.2 * labelled_period:
        issues.append(
            f"measured period {period:.2f} s vs labelled {labelled_period:.2f} s "
            "(Keithley overhead; measured timing wins)"
        )
    lead_sign, phase1_s, ipd_s, phase2_s, envelope_s = _segment_pulses(
        time_array, np.asarray(values), edges, pulse_width_s
    )
    return ScopePulses(
        np.asarray(edges),
        np.asarray(widths),
        period,
        path,
        tuple(issues),
        lead_sign=lead_sign,
        phase1_s=phase1_s,
        ipd_s=ipd_s,
        phase2_s=phase2_s,
        envelope_s=envelope_s,
    )


def scope_train_for_run(
    loaded: LoadedRun, config: EvokedConfig, scope_dir: Path
) -> tuple[PulseTrain | None, str]:
    """Build a PulseTrain from the scope; (None, reason) means fall back."""
    run_folder = loaded.condition.run_folder
    onset_epoch = read_onset_epoch(run_folder)
    if onset_epoch is None:
        return None, "no onset.txt"
    start_epoch = run_start_epoch(run_folder)
    if start_epoch is None:
        return None, "no parseable start time in folder name"
    capture = find_scope_capture(scope_dir, onset_epoch)
    if capture is None:
        return None, "no capture spans the onset"

    expected = loaded.condition.expected_pulses_run or config.expected_pulses
    width_s = loaded.condition.pulse_width_s_run or config.pulse_width_ms / 1000.0
    interval_s = loaded.condition.interval_s_run or max(0.0, config.max_period_s - width_s)
    pulses = read_scope_pulse_epochs(capture, onset_epoch, expected, width_s, interval_s)
    if pulses.epochs_s.size == 0:
        return None, "capture empty or no pulses detected"

    # Naive run-relative onsets, then one scalar clock-offset fit against the
    # amplifier envelope: score the CONTRAST across each putative edge.
    naive = pulses.epochs_s - start_epoch
    fs = loaded.sample_rate_hz
    bin_s = max(config.envelope_ms, 2.0) / 1000.0
    bin_width = max(1, int(round(fs * bin_s)))
    z_stack = [
        _robust_z(_rms_envelope(loaded.data.raw_uV[index].astype(np.float64), bin_width))
        for index in range(len(loaded.healthy_channels))
    ]
    z_stack = [z for z in z_stack if z.size]
    if not z_stack:
        return None, "no usable channel for alignment"
    z = np.max(np.vstack(z_stack), axis=0)

    # Window-averaged contrast across each putative edge: mean envelope z in
    # [+20, +180] ms minus [-180, -20] ms around the onset, via prefix sums so
    # every candidate offset costs O(1). Single-bin samples are far too noisy.
    inner = max(1, int(round(0.2 * _EDGE_CONTRAST_S / bin_s)))
    outer = max(inner + 1, int(round(1.8 * _EDGE_CONTRAST_S / bin_s)))
    window_bins = outer - inner
    prefix = np.concatenate([[0.0], np.cumsum(z)])

    deltas = np.arange(-config.scope_align_window_s, config.scope_align_window_s + bin_s, bin_s)
    onset_bins = np.round((naive[None, :] + deltas[:, None]) / bin_s).astype(int)
    valid = (onset_bins >= outer) & (onset_bins + outer < z.size)
    safe = np.clip(onset_bins, outer, max(outer, z.size - outer - 1))
    after_mean = (prefix[safe + outer] - prefix[safe + inner]) / window_bins
    before_mean = (prefix[safe - inner] - prefix[safe - outer]) / window_bins
    contrast = np.where(valid, after_mean - before_mean, 0.0)
    counts = np.maximum(valid.sum(axis=1), 1)
    scores = contrast.sum(axis=1) / counts

    best_index = int(np.argmax(scores))
    # Confidence against the curve OUTSIDE the peak's own neighbourhood, so a
    # sharp genuine peak is not penalised for raising the curve's variance.
    neighbourhood = np.abs(deltas - deltas[best_index]) <= max(0.5, width_s)
    rest = scores[~neighbourhood]
    if rest.size < 10 or not np.isfinite(rest.std()) or rest.std() <= 0:
        return None, "alignment inconclusive (flat score)"
    align_z = float((scores[best_index] - rest.mean()) / rest.std())
    if align_z < config.min_comb_z:
        return None, f"alignment inconclusive (z={align_z:.1f})"
    delta = float(deltas[best_index])

    duration_s = loaded.duration_s
    onsets = naive + delta
    inside = (onsets >= 0.0) & (onsets < duration_s)
    issues = list(pulses.issues)
    if not inside.all():
        issues.append(f"{int((~inside).sum())} scope pulse(s) fall outside the recording")
    onsets = onsets[inside]
    if onsets.size == 0:
        return None, "every scope pulse falls outside the recording"

    # The pulse envelope (onset to the second phase's extremum) is the honest
    # "width" for a biphasic pulse; the half-max estimate stands in when the
    # slope segmentation is indeterminate.
    if np.isfinite(pulses.envelope_s):
        width_ms = float(pulses.envelope_s * 1000.0)
    else:
        width_ms = float(np.median(pulses.widths_s) * 1000.0)
    if np.isfinite(width_ms) and not (
        0.2 * width_s * 1000.0 <= width_ms <= 6.0 * width_s * 1000.0
    ):
        issues.append(f"scope envelope {width_ms:.0f} ms, label says {width_s * 1000:.0f} ms")
    if onsets.size != expected:
        issues.append(f"{onsets.size} pulses inside the recording, expected {expected}")

    period_s = pulses.period_s if np.isfinite(pulses.period_s) else float("nan")
    jitter_ms = (
        float(np.std(np.diff(pulses.epochs_s)) * 1000.0) if pulses.epochs_s.size >= 2 else float("nan")
    )
    train = PulseTrain(
        onsets_s=onsets,
        period_s=period_s,
        train_start_s=float(onsets[0]),
        train_end_s=float(onsets[-1] + (period_s if np.isfinite(period_s) else 0.0)),
        width_ms=width_ms,
        jitter_ms=jitter_ms,
        n_detected=int(pulses.epochs_s.size),
        ok=not issues,
        issues=tuple(issues),
        comb_z=float("nan"),
        source="scope",
        # Host epoch of Intan sample t = start_epoch + t + clock_offset_s;
        # positive = the stim-host clock runs ahead of the Intan folder clock
        # (the 1 s filename quantization is included).
        clock_offset_s=-delta,
        scope_capture=capture.name,
        align_z=align_z,
        lead_sign=pulses.lead_sign,
        phase1_s=pulses.phase1_s,
        ipd_s=pulses.ipd_s,
        phase2_s=pulses.phase2_s,
    )
    return train, ""


__all__ = [
    "ScopePulses",
    "capture_span",
    "find_scope_capture",
    "read_onset_epoch",
    "read_scope_pulse_epochs",
    "run_start_epoch",
    "scope_train_for_run",
]
