"""Sweep configuration: fixed threshold, arm definitions, every fixed plot limit."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

from stim_analysis.config import AnalysisConfig

DEFAULT_ROOT = Path("/Users/jf/SynologyDrive/Research/Stimulation/20260818 re stim in vitro filter settings")
OUTPUT_SUBDIR = "bandwidth_sweep"

# The RHS amplifier saturates at ADC code +-32764 = +-6388.98 uV in the
# converted data (0.195 uV/LSB about 32768); that is the "+-6389 uV rail".
RAIL_LEVEL_UV = 6388.9

KNOBS = ("lower_hz", "dsp_hz", "upper_hz")


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the sweep: which knob varies and what the others must equal."""

    name: str
    title: str
    knob: str  # the varied knob (one of KNOBS)
    folder_prefix: str  # expected folder label; a cross-check only
    knob_label: str  # axis label for the varied knob
    expected: tuple[float, ...]  # nominal values the experiment aimed for (Hz; 0 = DSP off)
    constant: dict[str, float] = field(default_factory=dict)  # other knobs -> nominal value

    def label_for(self, value: float) -> str:
        if self.knob == "dsp_hz" and value <= 0:
            return "off"
        return f"{format_hz(value)} Hz"


def format_hz(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A", "analog lower cutoff (DSP off, upper 7500 Hz)", "lower_hz", "analogsweep",
            "analog lower bandwidth (Hz)", (0.1, 1.0, 10.0, 30.0, 100.0, 300.0), {"dsp_hz": 0.0, "upper_hz": 7500.0}),
    ArmSpec("B", "DSP cutoff (analog lower 0.1 Hz, upper 7500 Hz)", "dsp_hz", "DSPsweep",
            "DSP cutoff (Hz; 0 = off)", (0.0, 0.29, 1.17, 4.66, 18.6, 149.0), {"lower_hz": 0.1, "upper_hz": 7500.0}),
    ArmSpec("C", "analog upper cutoff (analog lower 1 Hz, DSP off)", "upper_hz", "upperanalogbandwidth",
            "analog upper bandwidth (Hz)", (7500.0, 3000.0, 1000.0, 500.0, 300.0), {"lower_hz": 1.0, "dsp_hz": 0.0}),
)

ARM_BY_NAME = {arm.name: arm for arm in ARMS}


@dataclass(frozen=True)
class SweepConfig:
    """Every parameter of the sweep analysis. Units in the field names."""

    root: Path = DEFAULT_ROOT
    output_subdir: str = OUTPUT_SUBDIR

    # --- recovery: FIXED threshold, everything else = spec v2 defaults -----
    threshold_uV: float = 100.0

    # --- centring: local pre-pulse window (the spec's -500..-50 ms baseline still gives baseline_sd) --
    centre_window_ms: tuple[float, float] = (-50.0, -5.0)  # epoch centred on this mean: closest to the pulse, least previous-tail contamination
    local_drift_max_uV: float = 25.0  # |linear drift of raw across the centre window| above this = baseline still moving before the pulse

    # --- rail / peak / fit ---------------------------------------------------
    rail_level_uV: float = RAIL_LEVEL_UV
    peak_window_ms: float = 5.0
    fit_start_offset_ms: float = 2.0
    fit_end_ms: float = 800.0
    fit_tau_bounds_ms: tuple[float, float] = (2.0, 5000.0)
    fit_informative_tau_ms: float = 4.0  # tau_nominal below this: tail gone before the fit window opens
    r2_min: float = 0.9

    # --- clean noise floor before the train ---------------------------------
    prestim_skip_start_s: float = 1.0
    prestim_gap_before_first_s: float = 0.5
    prestim_min_s: float = 0.5

    # --- slope fits -----------------------------------------------------------
    censor_ms: float = 850.0  # a run median >= this is treated as censored (epoch ends at 900 ms)
    floor_margin_ms: float = 1.0  # a run median <= hardware floor + this sits on the floor
    drifting_fraction_max: float = 0.5  # more than this fraction of trials with |local drift| > local_drift_max_uV -> run excluded from slopes
    slope_ci_width_max: float = 1.0
    slope_one_band: float = 0.25  # CI entirely inside 1 +- band also counts as ~1
    slope_flat_band: float = 0.25  # CI entirely inside 0 +- band also counts as flat
    exclude_stim_contact_from_slopes: bool = True

    # --- arm assignment -------------------------------------------------------
    knob_match_log_tol: float = math.log(1.25)  # actual vs nominal within x1.25 counts as "the same setting"
    duplicate_log_tol: float = math.log(1.02)  # two runs within 2 % on the varied knob = replicates
    keep_duplicate: str = "both"  # both | first | last

    # --- Arm C rule / recommendation ----------------------------------------
    zero_rail_fraction: float = 0.95
    noise_factor: float = 2.0  # acceptable prestim SD <= factor x the reference setting's prestim SD
    reference_setting: tuple[float, float, float] = (0.1, 1.17, 7500.0)  # (lower, DSP, upper) Hz: the setting in use in vivo (DSP k=12); nan-safe fallback = sweep median
    min_upper_hz: float = 300.0

    # --- statistics -----------------------------------------------------------
    bootstrap_n: int = 1000
    seed: int = 0

    # --- figures: fixed limits, never data-derived --------------------------
    lim_tau_ms: tuple[float, float] = (0.1, 10000.0)  # log
    lim_recovery_ms: tuple[float, float] = (0.5, 1000.0)  # log
    lim_rail_ms: tuple[float, float] = (0.0, 50.0)  # linear
    lim_peak_uV: tuple[float, float] = (100.0, 10000.0)  # log
    lim_sd_uV: tuple[float, float] = (0.1, 1000.0)  # log
    lim_fc_hz: tuple[float, float] = (0.03, 1000.0)  # log, effective high-pass corner (arms A, B)
    lim_upper_hz: tuple[float, float] = (100.0, 10000.0)  # log
    lim_trace_uV: tuple[float, float] = (-7000.0, 7000.0)
    lim_trace_zoom_uV: tuple[float, float] = (-500.0, 500.0)
    trace_window_ms: tuple[float, float] = (-5.0, 300.0)
    dpi: int = 200

    def analysis_config(self) -> AnalysisConfig:
        """Spec-v2 config with the recovery threshold pinned to ``threshold_uV``.

        ``compute_recovery`` uses ``max(threshold_k * sd, threshold_floor_uV)``;
        with k = 0 that is the floor for every trial, whatever the baseline SD.
        (``AnalysisConfig.validate`` would reject k = 0; it is only called by
        the notebook/CLI front ends, never by ``compute_recovery``.)
        """
        return dataclasses.replace(AnalysisConfig(), threshold_k=0.0, threshold_floor_uV=float(self.threshold_uV))

    def with_(self, **changes: object) -> "SweepConfig":
        return dataclasses.replace(self, **changes)


def sweep_config_to_dict(cfg: SweepConfig) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in dataclasses.fields(cfg):
        value = getattr(cfg, item.name)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        out[item.name] = value
    return out


def tau_ms_from_hz(fc_hz: float) -> float:
    return 1000.0 / (2.0 * math.pi * fc_hz) if fc_hz and fc_hz > 0 and math.isfinite(fc_hz) else float("nan")


__all__ = [
    "ARMS", "ARM_BY_NAME", "ArmSpec", "DEFAULT_ROOT", "KNOBS", "OUTPUT_SUBDIR", "RAIL_LEVEL_UV",
    "SweepConfig", "format_hz", "sweep_config_to_dict", "tau_ms_from_hz",
]
