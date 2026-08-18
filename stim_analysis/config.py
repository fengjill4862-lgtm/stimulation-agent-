"""Analysis configuration: every parameter and every fixed plot limit.

All text parsing for the CLI and the notebook widgets happens in
``config_from_text_fields`` so neither front end interprets strings itself
(Layering Rule in STIM_AGENT_HANDOFF.md).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field

from plot_rhs_power_analysis import parse_power_bands, parse_time_window_ms

ADC_FULL_SCALE_UV = 0.195 * 32768.0  # 16-bit amplifier at 0.195 uV/LSB


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float

    @property
    def label(self) -> str:
        return f"{self.name} {self.low_hz:g}-{self.high_hz:g} Hz"


DEFAULT_BANDS: tuple[Band, ...] = (
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 12.0),
    Band("beta", 13.0, 30.0),
    Band("gamma", 30.0, 80.0),
)

DEFAULT_BANDS_TEXT = "delta 1-4; theta 4-8; alpha 8-12; beta 13-30; gamma 30-80"


@dataclass(frozen=True)
class AnalysisConfig:
    """Spec v2 defaults. Units are stated in every field name."""

    # --- identity / geometry -------------------------------------------------
    stim_channel: str | None = None  # None: auto-detect from stim_data
    contact_pitch_um: float = 500.0
    contact_order: tuple[str, ...] | None = None  # None: by contact index in "A-0NN"

    # --- epochs (ms relative to pulse onset) ---------------------------------
    epoch_ms: tuple[float, float] = (-600.0, 900.0)
    filter_pad_ms: float = 500.0
    baseline_ms: tuple[float, float] = (-500.0, -50.0)
    late_ms: tuple[float, float] = (400.0, 900.0)
    post_length_ms: float = 300.0
    post_start_max_ms: float = 300.0

    # --- filter (spec section 3) --------------------------------------------
    highpass_hz: float = 1.0
    lowpass_hz: float = 150.0
    filter_order: int = 4
    zero_phase: bool = True

    # --- recovery (spec section 4) ------------------------------------------
    threshold_k: float = 3.0
    threshold_floor_uV: float = 100.0
    quiet_ms: float = 20.0
    recovery_quantile: float = 0.9
    blank_margin_ms: float = 5.0
    blank_pre_ms: float = 1.0
    blank_mode: str = "per_epoch"  # or "per_condition"
    hardware_floor_ms: float | None = None  # None: pulse + amp settle from data/settings
    early_verdict_ms: float = 50.0

    # --- rail detection (spec section 2.4) ----------------------------------
    rail_min_run_ms: float = 0.5
    rail_tolerance_uV: float = 0.4
    adc_full_scale_uV: float = ADC_FULL_SCALE_UV

    # --- events / validation ------------------------------------------------
    pulse_merge_gap_ms: float = 10.0
    train_split_factor: float = 5.0

    # --- bands (spec section 7.1) -------------------------------------------
    bands: tuple[Band, ...] = DEFAULT_BANDS

    # --- statistics (spec section 8) ----------------------------------------
    bootstrap_n: int = 1000
    seed: int = 0
    drift_n: int = 10
    min_trials: int = 20
    shuffle_enabled: bool = True
    shuffle_n_events: int = 50
    use_statsmodels: bool = True

    # --- figures (spec section 9): fixed limits, never data-derived ---------
    lim_recovery_ms: tuple[float, float] = (0.5, 1000.0)  # log
    lim_amplitude_uA: tuple[float, float] = (5.0, 500.0)  # log
    lim_impedance_kohm: tuple[float, float] = (10.0, 10000.0)  # log
    lim_distance_um: tuple[float, float] = (0.0, 4000.0)
    lim_trace_uV: tuple[float, float] = (-7000.0, 7000.0)  # wider than the ADC rail
    lim_trace_zoom_uV: tuple[float, float] = (-500.0, 500.0)
    trace_window_ms: tuple[float, float] = (-50.0, 300.0)
    trace_amplitudes_uA: tuple[float, ...] = (10.0, 20.0, 30.0)
    lim_db: tuple[float, float] = (-10.0, 10.0)
    lim_power_uV2: tuple[float, float] = (1e-1, 1e5)  # log
    lim_rms_uV: tuple[float, float] = (1.0, 1e4)  # log
    dpi: int = 220
    output_subdir: str = "stim_analysis"

    # --- derived helpers ----------------------------------------------------
    @property
    def epoch_length_ms(self) -> float:
        return self.epoch_ms[1] - self.epoch_ms[0]

    @property
    def filter_label(self) -> str:
        phase = "zero-phase sosfiltfilt" if self.zero_phase else "causal sosfilt"
        return (
            f"Butterworth bandpass {self.highpass_hz:g}-{self.lowpass_hz:g} Hz, "
            f"order {self.filter_order}, SOS, {phase}, pad {self.filter_pad_ms:g} ms"
        )

    def validate(self) -> None:
        if self.epoch_ms[1] <= self.epoch_ms[0]:
            raise ValueError("Epoch end must be after epoch start.")
        if self.epoch_ms[0] > self.baseline_ms[0] or self.baseline_ms[1] > 0:
            raise ValueError("Baseline window must lie inside the epoch and end before 0 ms.")
        if self.baseline_ms[1] <= self.baseline_ms[0]:
            raise ValueError("Baseline window end must be after its start.")
        if self.late_ms[1] <= self.late_ms[0] or self.late_ms[1] > self.epoch_ms[1]:
            raise ValueError("Late window must be increasing and end within the epoch.")
        if self.highpass_hz <= 0 or self.lowpass_hz <= self.highpass_hz:
            raise ValueError("Filter needs 0 < high-pass < low-pass.")
        if self.highpass_hz < 1.0:
            raise ValueError(
                "High-pass below 1 Hz is not allowed by the spec (0.1 Hz has a ~1.6 s "
                "time constant that rings after a saturating artifact)."
            )
        if self.filter_order < 1:
            raise ValueError("Filter order must be at least 1.")
        if self.threshold_k <= 0 or self.threshold_floor_uV < 0:
            raise ValueError("Recovery threshold k must be > 0 and floor >= 0 uV.")
        if self.quiet_ms <= 0:
            raise ValueError("Quiet duration must be > 0 ms.")
        if not 0.0 < self.recovery_quantile <= 1.0:
            raise ValueError("Recovery quantile must be in (0, 1].")
        if self.post_length_ms <= 0:
            raise ValueError("Post window length must be > 0 ms.")
        if self.blank_mode not in ("per_epoch", "per_condition"):
            raise ValueError("Blank mode must be per_epoch or per_condition.")
        if not self.bands:
            raise ValueError("At least one band is required.")
        if self.bootstrap_n < 0 or self.drift_n < 1 or self.min_trials < 1:
            raise ValueError("bootstrap_n >= 0, drift_n >= 1, min_trials >= 1 required.")
        if self.dpi < 50:
            raise ValueError("dpi must be at least 50.")


def parse_ms_pair(text: str, name: str) -> tuple[float, float]:
    """'-600 to 900', '-600,900', '-600 900' -> (-600.0, 900.0)."""
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError(f"Enter {name} as two ms values, e.g. -500 to -50.")
    # Reuse the Function 5 parser (handles unicode dashes and 'to').
    window = parse_time_window_ms(cleaned.replace(",", " to "), default_name=name)
    return float(window.start_ms), float(window.end_ms)


def parse_bands_text(text: str) -> tuple[Band, ...]:
    """'delta 1-4; theta 4-8' -> Bands (Function 5 grammar)."""
    parsed = parse_power_bands(text)
    return tuple(Band(item.name, float(item.low_hz), float(item.high_hz)) for item in parsed)


def parse_amplitude_list(text: str) -> tuple[float, ...]:
    """'10 20 30' or '10, 20, 30' -> (10.0, 20.0, 30.0)."""
    values = [float(item) for item in re.split(r"[\s,;]+", str(text).strip()) if item]
    if not values:
        raise ValueError("Enter at least one trace amplitude, e.g. 10 20 30.")
    return tuple(values)


def parse_optional_channel(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if cleaned.lower() in ("", "auto", "none"):
        return None
    if not re.fullmatch(r"[A-Da-d]-\d{3}", cleaned):
        raise ValueError("Stim channel must look like A-030, or 'auto'.")
    return cleaned.upper()


def config_from_text_fields(
    *,
    epoch: str = "-600 to 900",
    baseline: str = "-500 to -50",
    late: str = "400 to 900",
    bands: str = DEFAULT_BANDS_TEXT,
    highpass_hz: float = 1.0,
    lowpass_hz: float = 150.0,
    filter_order: int = 4,
    zero_phase: bool = True,
    threshold_k: float = 3.0,
    threshold_floor_uV: float = 100.0,
    quiet_ms: float = 20.0,
    recovery_quantile: float = 0.9,
    post_length_ms: float = 300.0,
    bootstrap_n: int = 1000,
    seed: int = 0,
    shuffle: bool = True,
    stim_channel: str | None = None,
    trace_amplitudes: str = "10 20 30",
    filter_pad_ms: float = 500.0,
    blank_margin_ms: float = 5.0,
    dpi: int = 220,
    **overrides: object,
) -> AnalysisConfig:
    """Build a validated AnalysisConfig from widget/CLI text and numbers."""
    cfg = AnalysisConfig(
        epoch_ms=parse_ms_pair(epoch, "epoch"),
        baseline_ms=parse_ms_pair(baseline, "baseline"),
        late_ms=parse_ms_pair(late, "late window"),
        bands=parse_bands_text(bands),
        highpass_hz=float(highpass_hz),
        lowpass_hz=float(lowpass_hz),
        filter_order=int(filter_order),
        zero_phase=bool(zero_phase),
        threshold_k=float(threshold_k),
        threshold_floor_uV=float(threshold_floor_uV),
        quiet_ms=float(quiet_ms),
        recovery_quantile=float(recovery_quantile),
        post_length_ms=float(post_length_ms),
        bootstrap_n=int(bootstrap_n),
        seed=int(seed),
        shuffle_enabled=bool(shuffle),
        stim_channel=parse_optional_channel(stim_channel),
        trace_amplitudes_uA=parse_amplitude_list(trace_amplitudes),
        filter_pad_ms=float(filter_pad_ms),
        blank_margin_ms=float(blank_margin_ms),
        dpi=int(dpi),
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]
    cfg.validate()
    return cfg


def config_to_dict(cfg: AnalysisConfig) -> dict[str, object]:
    """JSON-safe dictionary of the configuration."""
    out: dict[str, object] = {}
    for item in dataclasses.fields(cfg):
        value = getattr(cfg, item.name)
        if item.name == "bands":
            value = [dataclasses.asdict(band) for band in value]
        elif isinstance(value, tuple):
            value = list(value)
        out[item.name] = value
    out["filter_label"] = cfg.filter_label
    return out


__all__ = [
    "ADC_FULL_SCALE_UV",
    "AnalysisConfig",
    "Band",
    "DEFAULT_BANDS",
    "DEFAULT_BANDS_TEXT",
    "config_from_text_fields",
    "config_to_dict",
    "parse_amplitude_list",
    "parse_bands_text",
    "parse_ms_pair",
    "parse_optional_channel",
]
