#!/usr/bin/env python3
"""Recovering stim pulse times from the amplifier trace.

Nothing recorded the trigger: ANALOG-IN and DIGITAL-IN were both disabled and
the Keithley talks over serial. Pulse times therefore have to come out of the
signal itself.

A plain amplitude threshold does not work. The recordings carry a strong
periodic rhythm (about 3.9 Hz in some runs, near 1 Hz in others) that a
threshold locks onto instead of the stimulus. What makes recovery tractable is
the known protocol -- 50 pulses, 5 ms wide, starting around 10 s in -- which
turns this from open-ended detection into a constrained fit whose answer can be
checked: if 50 onsets do not come back at a consistent period, the recovery
failed and says so rather than handing back plausible-looking wrong epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EvokedConfig
from .load import LoadedRun


@dataclass(frozen=True)
class PulseTrain:
    """Recovered stimulus timing for one run."""

    onsets_s: np.ndarray
    period_s: float
    train_start_s: float
    train_end_s: float
    width_ms: float
    jitter_ms: float
    n_detected: int
    ok: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    # How many SD the winning comb alignment beats every other alignment by.
    comb_z: float = float("nan")
    # Timing provenance. "comb" = fitted to the amplifier trace alone;
    # "scope" = oscilloscope capture + onset file (see scope_sync).
    source: str = "comb"
    clock_offset_s: float = float("nan")  # stim-host clock minus Intan folder clock
    scope_capture: str = ""
    align_z: float = float("nan")  # scope-template alignment confidence

    @property
    def n_pulses(self) -> int:
        return int(self.onsets_s.size)

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.period_s if self.period_s > 0 else float("nan")


def _rms_envelope(x: np.ndarray, width: int) -> np.ndarray:
    """Non-overlapping RMS in `width`-sample bins."""
    usable = (x.size // width) * width
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt((x[:usable].reshape(-1, width) ** 2).mean(axis=1))


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = values.std() or 1.0
    return (values - median) / scale


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Start and end index (inclusive) of the longest run of True."""
    if not mask.any():
        return None
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best] - 1)


def _candidate_periods(
    z: np.ndarray, bin_s: float, config: EvokedConfig, n_candidates: int = 3
) -> list[float]:
    """Top autocorrelation peaks of the envelope, as candidate pulse periods.

    Autocorrelation is used rather than differencing detected peaks because a
    single missed pulse doubles one inter-peak interval and corrupts the median,
    while it barely perturbs the correlogram. Several candidates are returned
    because a strong physiological rhythm can outrank the stimulus here; the
    comb fit downstream decides between them on how well 50 evenly spaced
    pulses actually line up.
    """
    signal = z - z.mean()
    if signal.size < 8:
        return []
    correlation = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
    if correlation[0] <= 0:
        return []
    correlation = correlation / correlation[0]

    min_lag = max(1, int(round(config.min_period_s / bin_s)))
    max_lag = min(correlation.size - 1, int(round(config.max_period_s / bin_s)))
    if max_lag <= min_lag:
        return []

    window = correlation[min_lag : max_lag + 1]
    # Local maxima only, so one broad peak does not fill every slot.
    interior = np.flatnonzero(
        (window[1:-1] >= window[:-2]) & (window[1:-1] >= window[2:])
    ) + 1
    if interior.size == 0:
        return [(int(np.argmax(window)) + min_lag) * bin_s]
    ranked = interior[np.argsort(window[interior])[::-1]][:n_candidates]
    return [float((int(lag) + min_lag) * bin_s) for lag in ranked]


def _fit_comb(
    z: np.ndarray,
    bin_s: float,
    candidates: list[float],
    config: EvokedConfig,
    spread_bins: float = 10.0,
) -> tuple[float, float, float, float]:
    """Fit a comb of `expected_pulses` evenly spaced onsets to the envelope.

    Returns (period_s, first_onset_s, score). The search is done on a coarse
    5 ms grid -- fine enough to place a 5 ms pulse, coarse enough that the whole
    (period, start) plane can be scored vectorised -- and the result is snapped
    back to the fine envelope afterwards.
    """
    coarse_bin_s = 0.005
    pool = max(1, int(round(coarse_bin_s / bin_s)))
    usable = (z.size // pool) * pool
    if usable == 0:
        return float("nan"), float("nan"), -np.inf
    coarse = z[:usable].reshape(-1, pool).max(axis=1)
    coarse_bin_s = pool * bin_s

    best = (float("nan"), float("nan"), -np.inf, float("nan"))
    first_allowed = int(config.search_start_s / coarse_bin_s)

    for candidate in candidates:
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        # The period grid has to be fine, not wide. Over 49 gaps a period wrong
        # by one step drifts by 49 steps, so a step of 2 ms puts the far end of
        # the comb 100 ms out and no alignment can hit every pulse -- which is
        # how the fit ends up a whole period off station. Autocorrelation
        # already pins the period to about one envelope bin, so search a narrow
        # window around it at a fraction of a bin.
        step = bin_s / 4.0
        spread = spread_bins * bin_s
        for period_s in np.arange(candidate - spread, candidate + spread + step / 2, step):
            if not (config.min_period_s <= period_s <= config.max_period_s):
                continue
            period_bins = period_s / coarse_bin_s
            if period_bins < 2:
                continue
            span = int(round((config.expected_pulses - 1) * period_bins))
            last_start = coarse.size - span - 1
            if last_start <= first_allowed:
                continue

            starts = np.arange(first_allowed, last_start, dtype=np.int64)
            offsets = np.round(np.arange(config.expected_pulses) * period_bins).astype(np.int64)
            indices = starts[:, None] + offsets[None, :]
            score = coarse[indices].mean(axis=1)
            position = int(np.argmax(score))
            if score[position] > best[2]:
                # How far the winning placement stands above every other
                # placement of the same comb. This is the check that separates a
                # real train from a comb that merely landed on ongoing rhythm:
                # a stimulus has one sharply preferred alignment, noise does not.
                spread = float(score.std())
                comb_z = (score[position] - float(score.mean())) / spread if spread > 0 else float("nan")
                best = (
                    float(period_s),
                    float(starts[position] * coarse_bin_s),
                    float(score[position]),
                    float(comb_z),
                )

    return best


def _snap_onsets(
    z: np.ndarray, bin_s: float, first_onset_s: float, period_s: float, config: EvokedConfig
) -> np.ndarray:
    """Snap each comb tooth to the nearest envelope peak at fine resolution.

    Snapping is bounded to a quarter period, so a pulse the detector missed
    cannot pull its neighbour off station, and jitter from the software-timed
    serial link is absorbed without the comb drifting.

    For wide pulses (>= 50 ms, the slow biphasic protocol) the envelope is a
    plateau and its argmax can sit hundreds of ms into the pulse, so the tooth
    snaps to the leading edge instead: the first bin in the search window to
    cross half the window's maximum. Narrow pulses keep the classic peak snap
    unchanged.
    """
    period_bins = period_s / bin_s
    if not np.isfinite(period_bins) or period_bins < 1:
        return np.zeros(0, dtype=np.float64)

    tolerance = max(1, int(round(period_bins / 4.0)))
    first_bin = first_onset_s / bin_s
    wide_pulse = config.pulse_width_ms >= 50.0

    onsets: list[float] = []
    for index in range(config.expected_pulses):
        centre = int(round(first_bin + index * period_bins))
        lo = max(0, centre - tolerance)
        hi = min(z.size - 1, centre + tolerance)
        if lo > hi or lo >= z.size:
            break
        window = z[lo : hi + 1]
        if wide_pulse:
            above = np.flatnonzero(window >= float(window.max()) / 2.0)
            peak = lo + int(above[0]) if above.size else lo + int(np.argmax(window))
        else:
            peak = lo + int(np.argmax(window))
        onsets.append(peak * bin_s)

    return np.asarray(onsets, dtype=np.float64)


def _pulse_width_ms(z: np.ndarray, bin_s: float, onsets_s: np.ndarray) -> float:
    """Median half-maximum duration of the detected events."""
    if onsets_s.size == 0:
        return float("nan")
    widths: list[float] = []
    for onset in onsets_s:
        centre = int(round(onset / bin_s))
        if centre >= z.size:
            continue
        peak = z[centre]
        if peak <= 0:
            continue
        half = peak / 2.0
        left = centre
        while left > 0 and z[left - 1] > half:
            left -= 1
        right = centre
        while right < z.size - 1 and z[right + 1] > half:
            right += 1
        widths.append((right - left + 1) * bin_s * 1000.0)
    return float(np.median(widths)) if widths else float("nan")


def recover_pulses(
    run: LoadedRun, config: EvokedConfig, prior_periods: tuple[float, ...] = ()
) -> PulseTrain:
    """Recover the stimulus train from one loaded run.

    ``prior_periods`` constrains the search to periods already established from
    high-confidence runs in the same session. The Keithley settings did not
    change between consecutive recordings, so a run too weak to reveal its own
    period can still be timed from its neighbours' -- which matters precisely at
    the low amplitudes where the threshold question lives. Runs recovered this
    way are flagged, never silently promoted.
    """
    fs = run.sample_rate_hz
    bin_width = max(1, int(round(fs * config.envelope_ms / 1000.0)))
    bin_s = bin_width / fs
    issues: list[str] = []

    # Combine channels by taking the strongest robust z at each bin: a stimulus
    # appears on every connected contact, so the max is robust to one bad one.
    z_stack = []
    for index in range(len(run.healthy_channels)):
        envelope = _rms_envelope(run.data.raw_uV[index].astype(np.float64), bin_width)
        if envelope.size:
            z_stack.append(_robust_z(envelope))
    if not z_stack:
        return PulseTrain(
            np.zeros(0), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), 0, False, ("no usable channel",),
        )
    z = np.max(np.vstack(z_stack), axis=0)

    if z.size * bin_s < config.search_start_s + 2.0:
        return PulseTrain(
            np.zeros(0), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), 0, False, ("recording too short",),
        )

    # Fit the whole 50-pulse comb rather than hunting for a "loud" stretch. At
    # 0.01 mA the per-pulse deflection is comparable to the ongoing rhythm, so a
    # contrast threshold finds nothing; 50 evenly spaced events still line up.
    notes: list[str] = []
    if prior_periods:
        candidates = [p for p in prior_periods if np.isfinite(p) and p > 0]
        spread_bins = 30.0
        notes.append("period taken from other runs in this session")
    else:
        candidates = _candidate_periods(z, bin_s, config)
        spread_bins = 10.0
    if not candidates:
        return PulseTrain(
            np.zeros(0), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), 0, False, ("no periodic structure found",),
        )

    period_s, first_onset_s, score, comb_z = _fit_comb(
        z, bin_s, candidates, config, spread_bins
    )
    if not np.isfinite(period_s) or period_s <= 0:
        return PulseTrain(
            np.zeros(0), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), 0, False, ("comb fit failed",),
        )

    onsets_s = _snap_onsets(z, bin_s, first_onset_s, period_s, config)

    # The comb grid is far too coarse on its own: a period wrong by one grid
    # step accumulates into a drift of over 100 ms across 49 gaps, which can
    # leave the whole comb sitting a pulse or two off station. Refit period and
    # start by least squares through the snapped onsets, then re-snap, until it
    # settles. Two or three passes is enough to reach sub-millisecond alignment.
    for _iteration in range(3):
        if onsets_s.size < 3:
            break
        index = np.arange(onsets_s.size, dtype=np.float64)
        design = np.vstack([index, np.ones_like(index)]).T
        slope, intercept = np.linalg.lstsq(design, onsets_s, rcond=None)[0]
        if not (np.isfinite(slope) and slope > 0):
            break
        refined = _snap_onsets(z, bin_s, float(intercept), float(slope), config)
        if refined.size == 0:
            break
        if refined.size == onsets_s.size and np.allclose(refined, onsets_s, atol=bin_s):
            onsets_s = refined
            period_s = float(slope)
            break
        onsets_s = refined
        period_s = float(slope)

    width_ms = _pulse_width_ms(z, bin_s, onsets_s)
    train_start_s = float(onsets_s[0]) if onsets_s.size else float("nan")
    train_end_s = float(onsets_s[-1] + period_s) if onsets_s.size else float("nan")

    if np.isfinite(comb_z) and comb_z < config.min_comb_z:
        issues.append(
            f"comb fit only {comb_z:.1f} SD above other alignments; timing unreliable, "
            "which usually means no stimulus is visible at this amplitude"
        )

    if onsets_s.size >= 2:
        intervals = np.diff(onsets_s)
        jitter_ms = float(np.std(intervals) * 1000.0)
        period_s = float(np.median(intervals))
    else:
        jitter_ms = float("nan")

    # Checks against the known protocol. These do not silently repair anything;
    # they mark the run so a timing failure is visible in the output.
    if onsets_s.size != config.expected_pulses:
        issues.append(f"recovered {onsets_s.size} pulses, expected {config.expected_pulses}")
    if np.isfinite(width_ms) and not (0.2 * config.pulse_width_ms <= width_ms <= 6.0 * config.pulse_width_ms):
        issues.append(f"recovered width {width_ms:.1f} ms, expected about {config.pulse_width_ms:.0f} ms")
    if train_start_s < config.search_start_s + 1.0:
        issues.append(f"train starts at {train_start_s:.0f} s, close to the search floor")

    return PulseTrain(
        onsets_s=onsets_s,
        period_s=float(period_s),
        train_start_s=train_start_s,
        train_end_s=train_end_s,
        width_ms=float(width_ms),
        jitter_ms=jitter_ms,
        n_detected=int(onsets_s.size),
        ok=not issues,
        issues=tuple(issues + notes),
        comb_z=float(comb_z),
    )


__all__ = ["PulseTrain", "recover_pulses"]
