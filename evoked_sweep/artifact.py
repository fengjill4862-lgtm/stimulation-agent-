#!/usr/bin/env python3
"""Evidence about whether a deflection is a response or stimulus coupling.

At +1 mA in a configuration the user considers clean, the per-pulse deflection
is about 3.7 mV against a 400 uV baseline. That is large enough that resistive
coupling has to be tested for rather than assumed absent -- but the test reports
evidence and never overrules the measurement. Every run is measured the same
way; these flags sit beside the numbers so the judgement stays with the user.

Criteria, and what each one distinguishes:

* **latency** -- coupling is locked to the pulse edge; a synaptic response takes
  several ms to arrive.
* **tail** -- coupling stops when the 5 ms pulse stops; a response outlasts it.
* **linearity** -- coupling scales linearly with current through the origin; a
  neural response has a threshold and saturates.
* **polarity symmetry** -- coupling mirrors exactly when the current reverses;
  a response usually does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# A deflection peaking within this time of the pulse edge is too fast to be
# synaptic, so it looks like direct coupling.
COUPLING_LATENCY_MS = 1.5
# If this little of the response energy arrives after the pulse has ended, the
# deflection is essentially the pulse itself.
COUPLING_TAIL_FRACTION = 0.30


@dataclass(frozen=True)
class RunArtifactEvidence:
    """Per-run, per-channel artifact indicators."""

    channel: str
    fast_latency: bool
    stops_with_pulse: bool
    suspicion: float
    reasons: tuple[str, ...]
    # Window-split indicators. fast_latency above always trips on the onset
    # coupling spike; fast_latency_post asks the same question of the post-pulse
    # window only, which the spike cannot win.
    fast_latency_post: bool = False
    coupling_ratio: float = float("nan")  # during-pulse p-p / post-pulse p-p
    post_response_detected: bool = False  # any non-edge-suspect peak found


@dataclass(frozen=True)
class SweepArtifactEvidence:
    """Across-amplitude indicators for one wiring configuration and channel."""

    wiring_label: str
    channel: str
    n_points: int
    slope_uV_per_mA: float
    intercept_uV: float
    r_squared: float
    linear_through_origin: bool
    polarity_asymmetry: float


def run_evidence(
    channel: str,
    peak_latency_ms: float,
    post_pulse_fraction: float,
    post_peak_latency_ms: float = float("nan"),
    during_pp_uV: float = float("nan"),
    post_pp_uV: float = float("nan"),
    n_post_peaks: int = 0,
    pulse_width_ms: float = 5.0,
) -> RunArtifactEvidence:
    """Score one run/channel on the indicators available within a run."""
    reasons: list[str] = []

    fast = bool(np.isfinite(peak_latency_ms) and peak_latency_ms <= COUPLING_LATENCY_MS)
    if fast:
        reasons.append(f"peak at {peak_latency_ms:.2f} ms, too fast to be synaptic")

    stops = bool(np.isfinite(post_pulse_fraction) and post_pulse_fraction <= COUPLING_TAIL_FRACTION)
    if stops:
        reasons.append(f"only {post_pulse_fraction * 100:.0f}% of energy outlasts the pulse")

    fast_post = bool(
        np.isfinite(post_peak_latency_ms)
        and (post_peak_latency_ms - pulse_width_ms) <= COUPLING_LATENCY_MS
    )
    if fast_post:
        reasons.append(
            f"post-pulse peak at {post_peak_latency_ms:.2f} ms, still locked to the pulse edge"
        )

    ratio = float("nan")
    if np.isfinite(during_pp_uV) and np.isfinite(post_pp_uV) and post_pp_uV > 0:
        ratio = during_pp_uV / post_pp_uV

    detected = bool(n_post_peaks > 0)
    if detected:
        reasons.append(f"{n_post_peaks} distinct post-pulse peak(s) beyond the off-edge")

    return RunArtifactEvidence(
        channel=channel,
        fast_latency=fast,
        stops_with_pulse=stops,
        suspicion=float(fast + stops) / 2.0,
        reasons=tuple(reasons),
        fast_latency_post=fast_post,
        coupling_ratio=float(ratio),
        post_response_detected=detected,
    )


def sweep_evidence(
    wiring_label: str, channel: str, amplitudes_mA: np.ndarray, responses_uV: np.ndarray
) -> SweepArtifactEvidence | None:
    """Fit response against absolute current for one configuration and channel."""
    amplitudes = np.asarray(amplitudes_mA, dtype=np.float64)
    responses = np.asarray(responses_uV, dtype=np.float64)
    good = np.isfinite(amplitudes) & np.isfinite(responses)
    amplitudes, responses = amplitudes[good], responses[good]
    if amplitudes.size < 3:
        return None

    magnitude = np.abs(amplitudes)
    slope, intercept = np.polyfit(magnitude, responses, 1)
    predicted = slope * magnitude + intercept
    total = float(np.sum((responses - responses.mean()) ** 2))
    residual = float(np.sum((responses - predicted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")

    # "Through the origin" means the intercept is small next to the range the
    # response actually covers, not merely small in absolute terms.
    span = float(responses.max() - responses.min())
    through_origin = bool(np.isfinite(r_squared) and r_squared > 0.95 and span > 0 and abs(intercept) < 0.15 * span)

    positive = responses[amplitudes > 0]
    negative = responses[amplitudes < 0]
    if positive.size and negative.size:
        mean_positive, mean_negative = float(positive.mean()), float(negative.mean())
        denominator = 0.5 * (mean_positive + mean_negative)
        asymmetry = abs(mean_positive - mean_negative) / denominator if denominator > 0 else float("nan")
    else:
        asymmetry = float("nan")

    return SweepArtifactEvidence(
        wiring_label=wiring_label,
        channel=channel,
        n_points=int(amplitudes.size),
        slope_uV_per_mA=float(slope),
        intercept_uV=float(intercept),
        r_squared=float(r_squared),
        linear_through_origin=through_origin,
        polarity_asymmetry=float(asymmetry),
    )


# Decade-mislabel detection. The session mixed two folder-name conventions a
# decade apart (-02 = -0.2 mA vs -0_02 = -0.02 mA), so a run's label can be off
# by exactly 10x. The fit runs in log-log space with Theil-Sen (median of
# pairwise slopes) so a single mislabelled point cannot drag the fit toward
# itself the way it dominates an ordinary least-squares line. Flag only, never
# correct: the folder name stays the record.
DECADE_MIN_POINTS = 4
DECADE_MIN_LEVELS = 3  # distinct |I| values required for a stable slope
DECADE_TAU_LO = 0.15  # log10 units the shifted point must land within
DECADE_TAU_HI_FLOOR = 0.3  # log10 units: minimum residual to be suspect at all
DECADE_TAU_HI_SLOPE_FRACTION = 0.5


@dataclass(frozen=True)
class DecadeMislabelPoint:
    """One dose-response point's decade-mislabel evidence."""

    run: str
    log_residual: float
    shift: int  # +1 = fits 10x higher current, -1 = 10x lower, 0 = clean
    suspect: bool


def decade_mislabel_evidence(
    amplitudes_mA: np.ndarray,
    responses_uV: np.ndarray,
    run_names: tuple[str, ...],
) -> list[DecadeMislabelPoint]:
    """Score each point of one wiring/channel sweep for a 10x label error.

    A point is suspect when it sits far off the robust log-log trend AND lands
    on it when its current is shifted by exactly one decade. Returns one entry
    per input point, aligned with the inputs; unusable points come back clean.
    """
    amplitudes = np.asarray(amplitudes_mA, dtype=np.float64)
    responses = np.asarray(responses_uV, dtype=np.float64)
    clean = [
        DecadeMislabelPoint(run=name, log_residual=float("nan"), shift=0, suspect=False)
        for name in run_names
    ]

    usable = np.isfinite(amplitudes) & (np.abs(amplitudes) > 0)
    usable &= np.isfinite(responses) & (responses > 0)
    indices = np.flatnonzero(usable)
    if indices.size < DECADE_MIN_POINTS:
        return clean
    x = np.log10(np.abs(amplitudes[indices]))
    y = np.log10(responses[indices])
    if np.unique(np.round(x, 6)).size < DECADE_MIN_LEVELS:
        return clean

    # Theil-Sen: median slope over all pairs with distinct x, median intercept.
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(x.size)
        for j in range(i + 1, x.size)
        if x[j] != x[i]
    ]
    if not slopes:
        return clean
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))

    tau_hi = max(DECADE_TAU_HI_FLOOR, DECADE_TAU_HI_SLOPE_FRACTION * abs(slope))
    out = list(clean)
    for position, index in enumerate(indices):
        residual = float(y[position] - (intercept + slope * x[position]))
        shift = 0
        suspect = False
        if abs(residual) > tau_hi:
            candidates = [
                (abs(residual - s * slope), s)
                for s in (1, -1)
                if abs(residual - s * slope) < DECADE_TAU_LO
            ]
            if candidates:
                shift = min(candidates)[1]
                suspect = True
        out[index] = DecadeMislabelPoint(
            run=run_names[index], log_residual=residual, shift=shift, suspect=suspect
        )
    return out


__all__ = [
    "COUPLING_LATENCY_MS",
    "COUPLING_TAIL_FRACTION",
    "DECADE_MIN_LEVELS",
    "DECADE_MIN_POINTS",
    "DECADE_TAU_HI_FLOOR",
    "DECADE_TAU_HI_SLOPE_FRACTION",
    "DECADE_TAU_LO",
    "DecadeMislabelPoint",
    "RunArtifactEvidence",
    "SweepArtifactEvidence",
    "decade_mislabel_evidence",
    "run_evidence",
    "sweep_evidence",
]
