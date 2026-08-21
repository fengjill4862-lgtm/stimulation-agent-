#!/usr/bin/env python3
"""Stimulus timing for RHD recordings, recovered from the amplifier trace.

RHD recordings carry no stim channel, so the stim-driven viewers (Function 3's
event grid, Function 5's pre/post split) have nothing to trigger on. When the
stimulus came from the external Keithley, its pulse train is still visible in
the amplifier trace; this module recovers it with Function 7's comb fit
(``evoked_sweep.pulses.recover_pulses``) and synthesises a stand-in ``stim_uA``
array -- unit-amplitude rectangles at the recovered pulses -- so every
downstream stim code path works unchanged. The amplitude of the proxy is
meaningless by construction; only the timing is real, and the display label
says so.

Reload chain: this module imports from ``evoked_sweep`` and must stay listed
after it in ``wideband_main_ui._RELOAD_CHAIN``.

Self-test: ``python3 -m rhd_timing`` builds a synthetic RHD run with known
onsets and checks they come back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evoked_sweep.config import EvokedConfig
from evoked_sweep.pulses import PulseTrain, recover_pulses

RECOVERED_LABEL = "recovered"

RECOVERY_FAILED = (
    ".rhd recordings carry no stim channel, and the pulse train could not be "
    "recovered from the amplifier trace. Check the pulse count and width, or "
    "use Function 7, which reports why a comb fit failed."
)


@dataclass(frozen=True)
class _AdapterData:
    raw_uV: np.ndarray


@dataclass(frozen=True)
class _AdapterRun:
    """The duck-typed subset of evoked_sweep.load.LoadedRun that recover_pulses reads."""

    data: _AdapterData
    healthy_channels: tuple[str, ...]
    sample_rate_hz: float

    @property
    def duration_s(self) -> float:
        return self.data.raw_uV.shape[1] / self.sample_rate_hz


@dataclass(frozen=True)
class RecoveredStim:
    """A synthetic stim channel standing in for the unrecorded Keithley train."""

    stim_uA: np.ndarray  # unit rectangles at the recovered pulses; timing only
    train: PulseTrain
    label: str
    note: str  # one status-line sentence: period, width, confidence, issues


def recover_stim_proxy(
    folder: Path,
    channel_data: Sequence[tuple[str, np.ndarray]],
    sample_rate_hz: float,
    expected_pulses: int = 50,
    pulse_width_ms: float = 5.0,
) -> RecoveredStim | None:
    """Recover the pulse train from already-read channels; None if no fit.

    ``channel_data`` is the (name, raw_uV) pairs the caller already loaded, so
    nothing is read twice. The comb fit combines all given channels, so passing
    the displayed response channels is enough.
    """
    if not channel_data:
        return None
    stack = np.vstack([np.asarray(raw, dtype=np.float64) for _name, raw in channel_data])
    config = EvokedConfig(
        session_folder=Path(folder),
        expected_pulses=int(expected_pulses),
        pulse_width_ms=float(pulse_width_ms),
    )
    run = _AdapterRun(
        data=_AdapterData(stack),
        healthy_channels=tuple(name for name, _raw in channel_data),
        sample_rate_hz=float(sample_rate_hz),
    )
    train = recover_pulses(run, config)
    if train.n_pulses == 0:
        return None

    width_ms = train.width_ms if np.isfinite(train.width_ms) else float(pulse_width_ms)
    width_samples = max(1, int(round(width_ms / 1000.0 * sample_rate_hz)))
    n_samples = stack.shape[1]
    stim_uA = np.zeros(n_samples, dtype=np.float32)
    for onset_s in train.onsets_s:
        start = int(round(onset_s * sample_rate_hz))
        stop = min(n_samples, start + width_samples)
        if stop > max(0, start):
            stim_uA[max(0, start) : stop] = 1.0

    note = (
        f"timing recovered from the trace (no stim channel in .rhd): "
        f"{train.n_pulses} pulses, period {train.period_s * 1000:.1f} ms, "
        f"width {width_ms:.1f} ms, comb z {train.comb_z:.1f}"
    )
    if train.issues:
        note += "; " + "; ".join(train.issues)
    return RecoveredStim(stim_uA=stim_uA, train=train, label=RECOVERED_LABEL, note=note)


def _selftest() -> int:
    """Synthetic round-trip: known onsets in, recovered proxy out."""
    import tempfile

    from evoked_sweep.selftest import FS, make_run

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rhd_timing_selftest_") as name:
        folder = Path(name) / "cfg stim 1 stim ground 2 recording ground 3" / "+0_5mA_260819_120000"
        truth = make_run(folder, amplitude_uV=800.0, shape="coupling", seed=7)

        from rhs_stim import folder_recording_format, read_selected_channels

        if folder_recording_format(folder) != "rhd":
            failures.append("synthetic folder not detected as rhd")
        read = read_selected_channels(folder, ["B-017", "B-018"])
        if read.sample_rate_hz != FS:
            failures.append(f"sample rate {read.sample_rate_hz} != {FS}")
        if any(np.any(stim != 0) for _c, _r, stim in read.stim_channel_data):
            failures.append("rhd stim_uA proxy in ChannelRead is not all zeros")

        recovered = recover_stim_proxy(
            folder, read.raw_channel_data, read.sample_rate_hz, 50, 5.0
        )
        if recovered is None:
            failures.append("recovery returned None on an obvious train")
        else:
            active = np.r_[0, (recovered.stim_uA > 0).astype(np.int8)]
            onsets = np.flatnonzero(np.diff(active) == 1)
            if abs(onsets.size - truth.size) > 0:
                failures.append(f"{onsets.size} proxy pulses != {truth.size} true")
            error_ms = np.abs(onsets / FS - truth) * 1000.0 if onsets.size == truth.size else None
            if error_ms is not None and float(error_ms.max()) > 5.0:
                failures.append(f"worst onset error {error_ms.max():.2f} ms > 5")

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1
    print("OK: rhd_timing self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())


__all__ = ["RECOVERED_LABEL", "RECOVERY_FAILED", "RecoveredStim", "recover_stim_proxy"]
