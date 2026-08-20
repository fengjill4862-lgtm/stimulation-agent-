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


def run_evidence(channel: str, peak_latency_ms: float, post_pulse_fraction: float) -> RunArtifactEvidence:
    """Score one run/channel on the two indicators available within a run."""
    reasons: list[str] = []

    fast = bool(np.isfinite(peak_latency_ms) and peak_latency_ms <= COUPLING_LATENCY_MS)
    if fast:
        reasons.append(f"peak at {peak_latency_ms:.2f} ms, too fast to be synaptic")

    stops = bool(np.isfinite(post_pulse_fraction) and post_pulse_fraction <= COUPLING_TAIL_FRACTION)
    if stops:
        reasons.append(f"only {post_pulse_fraction * 100:.0f}% of energy outlasts the pulse")

    return RunArtifactEvidence(
        channel=channel,
        fast_latency=fast,
        stops_with_pulse=stops,
        suspicion=float(fast + stops) / 2.0,
        reasons=tuple(reasons),
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


__all__ = [
    "COUPLING_LATENCY_MS",
    "COUPLING_TAIL_FRACTION",
    "RunArtifactEvidence",
    "SweepArtifactEvidence",
    "run_evidence",
    "sweep_evidence",
]
