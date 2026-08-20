#!/usr/bin/env python3
"""Every parameter Function 7 uses, and the only place widget/CLI text is parsed.

The Layering Rule: UI modules read widget values and hand the raw strings here.
Nothing else in the codebase may parse user text into analysis parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Bands are (name, low_hz, high_hz). The upper band stops at 300 Hz because the
# amplifier's upper bandwidth is 7.6 kHz but the pulse harmonic comb dominates
# well below that; anything higher would measure the stimulus, not the tissue.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("beta", 12.0, 30.0),
    ("gamma", 30.0, 100.0),
    ("high", 100.0, 300.0),
)


@dataclass(frozen=True)
class EvokedConfig:
    """Parameters for one Function 7 run.

    Defaults encode the 2026-08-19 protocol as the user described it: 50 pulses,
    5 ms wide, starting about 10 s into each recording. The pulse interval was
    never logged, so it is recovered per run rather than configured.
    """

    session_folder: Path
    single_run: bool = False

    # Channel selection. None means "every amplifier channel whose header
    # impedance is below healthy_impedance_max_ohms" -- never a hard-coded list,
    # because floating contacts read ~10 Mohm and swing +/-5 mV.
    channels: tuple[str, ...] | None = None
    healthy_impedance_max_ohms: float = 1.0e6

    # Protocol (known, used as priors and as correctness checks).
    expected_pulses: int = 50
    pulse_width_ms: float = 5.0
    search_start_s: float = 5.0

    # Pulse recovery. The period floor is not cosmetic: each Keithley pulse costs
    # several serial writes with 40 ms sleeps between them, so a train cannot
    # actually repeat faster than about 100 ms. Anything quicker is a bad fit.
    envelope_ms: float = 0.5
    train_threshold_ratio: float = 3.0
    onset_z_threshold: float = 20.0
    min_period_s: float = 0.10
    max_period_s: float = 1.50
    min_comb_z: float = 5.0

    # Response measurement.
    response_window_ms: float = 50.0
    gap_baseline_start_ms: float = 150.0
    gap_baseline_end_ms: float = 250.0
    highpass_hz: float = 250.0

    # Post-train sustained change.
    post_train_gap_s: float = 2.0
    post_train_window_s: float = 10.0

    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS

    # Output.
    output_dir_name: str = "evoked_sweep"
    max_runs: int | None = None

    @property
    def output_dir(self) -> Path:
        return self.session_folder / self.output_dir_name

    @property
    def blank_ms(self) -> float:
        """Window after each onset excluded from gap-based measures."""
        return self.response_window_ms


def _clean(text: str) -> str:
    return (text or "").strip().strip('"').strip("'")


def config_from_text_fields(
    session_folder: str,
    channels: str = "all",
    expected_pulses: str = "50",
    pulse_width_ms: str = "5",
    single_run: str = "",
    max_runs: str = "",
) -> EvokedConfig:
    """Build an EvokedConfig from raw widget or CLI strings.

    Folder text is stripped of surrounding quotes and whitespace so a path
    pasted from Finder, spaces included, works unchanged.
    """
    folder_text = _clean(session_folder)
    if not folder_text:
        raise ValueError("Session folder is empty. Paste the folder path.")
    folder = Path(folder_text).expanduser()
    if not folder.exists():
        raise ValueError(f"Session folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Session folder is not a folder: {folder}")

    channel_text = _clean(channels).lower()
    if channel_text in ("", "all", "auto"):
        wanted: tuple[str, ...] | None = None
    else:
        wanted = tuple(
            part.strip() for part in channel_text.replace(",", " ").upper().split() if part.strip()
        )
        if not wanted:
            wanted = None

    def _positive_int(text: str, label: str, default: int) -> int:
        cleaned = _clean(text)
        if cleaned == "":
            return default
        try:
            value = int(float(cleaned))
        except ValueError as exc:
            raise ValueError(f"{label} must be a number, got {cleaned!r}") from exc
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        return value

    def _positive_float(text: str, label: str, default: float) -> float:
        cleaned = _clean(text)
        if cleaned == "":
            return default
        try:
            value = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number, got {cleaned!r}") from exc
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        return value

    max_runs_text = _clean(max_runs)
    max_runs_value = _positive_int(max_runs_text, "Max runs", 0) if max_runs_text else None
    if max_runs_value == 0:
        max_runs_value = None

    return EvokedConfig(
        session_folder=folder,
        single_run=_clean(single_run).lower() in ("1", "true", "yes", "on", "single"),
        channels=wanted,
        expected_pulses=_positive_int(expected_pulses, "Expected pulses", 50),
        pulse_width_ms=_positive_float(pulse_width_ms, "Pulse width (ms)", 5.0),
        max_runs=max_runs_value,
    )


__all__ = ["DEFAULT_BANDS", "EvokedConfig", "config_from_text_fields"]
