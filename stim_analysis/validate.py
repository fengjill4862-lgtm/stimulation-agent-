"""Validation before analysis (spec section 2).

Per run: detected vs commanded pulse count, compliance from data, empirical
rail detection per channel, sample rate, timestamp gaps, metadata mismatches.
Emits the validation table before any analysis runs, and the exclusion list
that gates the amplitude- and charge-dependence analyses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.load_rhs import RunRecord, hardware_floor_ms

BLOCK_ORDER = ("baseline", "block1", "block2", "block3", "other", "error")


# -----------------------------------------------------------------------------
# Rail detection (spec section 2.4)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RailEstimate:
    channel: str
    pos_level_uV: float
    neg_level_uV: float
    pos_longest_run_ms: float
    neg_longest_run_ms: float
    n_episodes: int
    pct_railed: float
    is_railed: bool
    at_adc_full_scale: bool
    is_flat: bool

    @property
    def levels(self) -> tuple[float, float]:
        return (self.neg_level_uV, self.pos_level_uV)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of each True run."""
    if mask.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2].astype(np.int64), edges[1::2].astype(np.int64)


def _extreme_runs(
    values: np.ndarray, tolerance_uV: float, min_run: int
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Level, run starts/ends (>= min_run) at the maximum of ``values``."""
    if values.size == 0:
        return float("nan"), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    vmax = float(np.max(values))
    at_max = values >= (vmax - tolerance_uV)
    starts, ends = _runs(at_max)
    lengths = ends - starts
    longest = int(lengths.max()) if lengths.size else 0
    keep = lengths >= min_run
    return vmax, starts[keep], ends[keep], longest


def estimate_rail(x: np.ndarray, sample_rate_hz: float, cfg: AnalysisConfig, channel: str = "") -> RailEstimate:
    """Empirical saturation estimate: constant runs at the extreme values."""
    x = np.asarray(x, dtype=np.float32)
    min_run = max(2, int(round(cfg.rail_min_run_ms * 1.0e-3 * sample_rate_hz)))
    pos_level, ps, pe, pos_longest = _extreme_runs(x, cfg.rail_tolerance_uV, min_run)
    neg_level_abs, ns, ne, neg_longest = _extreme_runs(-x, cfg.rail_tolerance_uV, min_run)
    neg_level = -neg_level_abs
    pos_railed = ps.size > 0
    neg_railed = ns.size > 0
    railed_samples = int(np.sum(pe - ps) + np.sum(ne - ns))
    pct = 100.0 * railed_samples / x.size if x.size else 0.0
    is_flat = bool(np.isfinite(pos_level) and np.isfinite(neg_level) and pos_level == neg_level)
    ms = 1.0e3 / sample_rate_hz
    full_scale = 0.98 * cfg.adc_full_scale_uV
    return RailEstimate(
        channel=channel,
        pos_level_uV=pos_level if pos_railed else float("nan"),
        neg_level_uV=neg_level if neg_railed else float("nan"),
        pos_longest_run_ms=pos_longest * ms,
        neg_longest_run_ms=neg_longest * ms,
        n_episodes=int(ps.size + ns.size),
        pct_railed=pct,
        is_railed=bool(pos_railed or neg_railed) and not is_flat,
        at_adc_full_scale=bool(
            (pos_railed and pos_level >= full_scale) or (neg_railed and -neg_level >= full_scale)
        ),
        is_flat=is_flat,
    )


def railed_mask(x: np.ndarray, rail: RailEstimate, cfg: AnalysisConfig, sample_rate_hz: float) -> np.ndarray:
    """Boolean mask of samples sitting in a rail run (>= rail_min_run_ms long)."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.zeros(x.size, dtype=bool)
    if not rail.is_railed:
        return mask
    min_run = max(2, int(round(cfg.rail_min_run_ms * 1.0e-3 * sample_rate_hz)))
    for level, sign in ((rail.pos_level_uV, 1.0), (rail.neg_level_uV, -1.0)):
        if not np.isfinite(level):
            continue
        at = sign * x >= sign * level - cfg.rail_tolerance_uV
        starts, ends = _runs(at)
        for start, end in zip(starts, ends):
            if end - start >= min_run:
                mask[start:end] = True
    return mask


# -----------------------------------------------------------------------------
# Per-run validation
# -----------------------------------------------------------------------------


@dataclass
class RunValidation:
    run_id: str
    run_folder: str
    label: str
    status: str  # ok | no_stim | error
    block: str = "other"
    in_block1: bool = False
    in_block2: bool = False
    in_block3: bool = False
    included: bool = True
    exclusion_reason: str = ""
    sample_rate_hz: float = float("nan")
    settings_sample_rate_hz: float = float("nan")
    n_files: int = 0
    duration_s: float = float("nan")
    timestamp_gaps: int = 0
    stim_channel_data: str = ""
    stim_channel_settings: str = ""
    stim_channel_folder: str = ""
    amplitude_uA_data: float = float("nan")
    amplitude_uA_settings: float = float("nan")
    amplitude_uA_folder: float = float("nan")
    phase_us_data: float = float("nan")
    phase_us_settings: float = float("nan")
    phase_us_folder: float = float("nan")
    charge_nC_per_phase: float = float("nan")
    polarity_data: str = ""
    polarity_settings: str = ""
    train_period_ms_settings: float = float("nan")
    ipi_median_s: float = float("nan")
    pulses_per_trigger_settings: float = float("nan")
    n_trains_detected: int = 0
    n_commanded_total: float = float("nan")
    n_detected: int = 0
    count_match: bool = False
    compliance_bit_seen: bool = False
    compliance_flag: bool = False
    amp_settle_bit_seen: bool = False
    charge_recovery_bit_seen: bool = False
    hardware_floor_ms: float = float("nan")
    impedance_source: str = "missing"
    rail_pct_by_channel: dict[str, float] = field(default_factory=dict)
    rail_pct_max: float = 0.0
    rails: list[RailEstimate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def validate_run(record: RunRecord, cfg: AnalysisConfig) -> RunValidation:
    """Build the validation row for one loaded run (data must still be attached)."""
    v = RunValidation(
        run_id=record.run_id,
        run_folder=str(record.folder),
        label=record.label,
        status="ok",
        sample_rate_hz=float(record.sample_rate_hz),
        n_files=record.n_files,
        duration_s=record.duration_s,
        timestamp_gaps=record.timestamp_gaps,
        stim_channel_data=record.stim_channel or "",
        stim_channel_folder=record.meta.stim_channel or "",
        amplitude_uA_folder=record.meta.amplitude_uA if record.meta.amplitude_uA is not None else float("nan"),
        phase_us_folder=record.meta.phase_us if record.meta.phase_us is not None else float("nan"),
        impedance_source=record.impedance_source,
    )
    v.warnings.extend(record.stim_channel_warnings)
    if record.load_error:
        v.status = "error"
        v.error = record.load_error
        v.included = False
        v.exclusion_reason = "error"
        return v

    settings = record.settings
    if settings is None:
        v.warnings.append("settings.xml missing: commanded pulse count unknown")
    else:
        v.stim_channel_settings = settings.stim_channel or ""
        v.settings_sample_rate_hz = settings.sample_rate_hz if settings.sample_rate_hz else float("nan")
        v.amplitude_uA_settings = settings.first_amp_uA if settings.first_amp_uA is not None else float("nan")
        v.phase_us_settings = settings.first_phase_us if settings.first_phase_us is not None else float("nan")
        v.polarity_settings = settings.polarity or ""
        v.train_period_ms_settings = (
            settings.train_period_us / 1000.0 if settings.train_period_us else float("nan")
        )
        ppt = settings.pulses_per_trigger
        v.pulses_per_trigger_settings = float(ppt) if ppt is not None else float("nan")
        if settings.sample_rate_hz and abs(settings.sample_rate_hz - record.sample_rate_hz) > 0.5:
            v.warnings.append(
                f"settings.xml sample rate {settings.sample_rate_hz:g} != header {record.sample_rate_hz:g}"
            )

    events = record.events
    if events is None or events.n_events == 0:
        v.status = "no_stim"
        v.n_detected = 0
        v.hardware_floor_ms = float("nan")
    else:
        v.n_detected = events.n_events
        v.n_trains_detected = events.n_trains
        v.amplitude_uA_data = float(np.median(events.amplitude_uA))
        v.phase_us_data = float(np.median(events.first_phase_us))
        sign = int(np.sign(np.median(events.first_phase_sign)))
        v.polarity_data = "cathodic first" if sign < 0 else ("anodic first" if sign > 0 else "")
        if events.n_events > 1:
            v.ipi_median_s = float(np.median(np.diff(events.onset_samples)) / record.sample_rate_hz)
        v.compliance_bit_seen = events.compliance_bit_any
        v.amp_settle_bit_seen = events.amp_settle_bit_any
        v.charge_recovery_bit_seen = events.charge_recovery_bit_any
        v.hardware_floor_ms = hardware_floor_ms(events, settings, record.sample_rate_hz)
        if np.isfinite(v.amplitude_uA_data) and np.isfinite(v.phase_us_data):
            v.charge_nC_per_phase = v.amplitude_uA_data * v.phase_us_data * 1e-3

        # Commanded count: pulses per trigger (settings) x triggers (trains detected)
        if settings is not None and settings.pulses_per_trigger:
            v.n_commanded_total = float(settings.pulses_per_trigger * max(1, events.n_trains))
        elif record.meta.pulses is not None:
            v.n_commanded_total = float(record.meta.pulses)
            v.warnings.append("commanded pulse count taken from folder name (no settings.xml)")
        if np.isfinite(v.n_commanded_total):
            v.count_match = int(v.n_commanded_total) == v.n_detected
            if v.n_detected < v.n_commanded_total:
                v.warnings.append(
                    f"detected {v.n_detected} < commanded {int(v.n_commanded_total)} pulses (compliance signature)"
                )
            elif v.n_detected > v.n_commanded_total:
                v.warnings.append(
                    f"detected {v.n_detected} > commanded {int(v.n_commanded_total)} pulses"
                )
        v.compliance_flag = bool(
            v.compliance_bit_seen
            or (np.isfinite(v.n_commanded_total) and v.n_detected < v.n_commanded_total)
        )
        # Metadata cross-checks (data wins)
        step = (settings.stim_step_uA if settings and settings.stim_step_uA else 2.0)
        if np.isfinite(v.amplitude_uA_settings) and abs(v.amplitude_uA_settings - v.amplitude_uA_data) > step:
            v.warnings.append(
                f"amplitude: settings {v.amplitude_uA_settings:g} uA vs data {v.amplitude_uA_data:g} uA"
            )
        if np.isfinite(v.amplitude_uA_folder) and abs(v.amplitude_uA_folder - v.amplitude_uA_data) > step:
            v.warnings.append(
                f"amplitude: folder {v.amplitude_uA_folder:g} uA vs data {v.amplitude_uA_data:g} uA"
            )
        one_sample_us = 1.0e6 / record.sample_rate_hz
        if np.isfinite(v.phase_us_settings) and abs(v.phase_us_settings - v.phase_us_data) > 1.5 * one_sample_us:
            v.warnings.append(
                f"phase: settings {v.phase_us_settings:g} us vs data {v.phase_us_data:g} us"
            )
        if np.isfinite(v.phase_us_folder) and abs(v.phase_us_folder - v.phase_us_data) > 1.5 * one_sample_us:
            v.warnings.append(f"phase: folder {v.phase_us_folder:g} us vs data {v.phase_us_data:g} us")
        if v.polarity_settings:
            expected = "cathodic first" if v.polarity_settings.lower().startswith("negative") else "anodic first"
            if v.polarity_data and expected != v.polarity_data:
                v.warnings.append(f"polarity: settings {expected} vs data {v.polarity_data}")

    if record.timestamp_gaps:
        v.warnings.append(f"{record.timestamp_gaps} timestamp discontinuities")

    # Rails, per analysis channel
    if record.data is not None:
        for channel in record.channels:
            x = record.data.raw_uV[record.data.channel_index(channel)]
            rail = estimate_rail(x, record.sample_rate_hz, cfg, channel)
            v.rails.append(rail)
            v.rail_pct_by_channel[channel] = rail.pct_railed
            if rail.is_flat:
                v.warnings.append(f"{channel} is flat (constant value)")
        v.rail_pct_max = max(v.rail_pct_by_channel.values(), default=0.0)
    if record.impedance_source == "missing":
        v.warnings.append("no impedance in RHS header (not measured before recording?)")
    return v


# -----------------------------------------------------------------------------
# Blocks and exclusions
# -----------------------------------------------------------------------------


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Optional CSV override: run_id, block[, include]."""
    table = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, dict[str, str]] = {}
    for _, row in table.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            out[run_id] = {key: str(row[key]).strip() for key in table.columns}
    return out


def _is_paired(v: RunValidation) -> bool:
    if np.isfinite(v.pulses_per_trigger_settings) and v.pulses_per_trigger_settings <= 2 and v.n_detected >= 2:
        if np.isfinite(v.train_period_ms_settings) and v.train_period_ms_settings <= 200:
            return True
    return bool(np.isfinite(v.ipi_median_s) and v.ipi_median_s < 0.2 and v.n_detected >= 2)


def assign_blocks(validations: list[RunValidation], manifest: dict[str, dict[str, str]] | None = None) -> None:
    """Fill block / in_blockN / included / exclusion_reason in place."""
    single = [
        v for v in validations
        if v.status == "ok" and not _is_paired(v) and np.isfinite(v.phase_us_data)
    ]
    phase_mode = None
    if single:
        counter = Counter(round(v.phase_us_data / 10.0) * 10.0 for v in single)
        phase_mode = counter.most_common(1)[0][0]
    ladder = [v for v in single if phase_mode is not None and abs(v.phase_us_data - phase_mode) <= 10.0]
    width_sweep = [v for v in single if v not in ladder]
    sweep_amp = None
    if width_sweep:
        sweep_amp = Counter(round(v.amplitude_uA_data) for v in width_sweep).most_common(1)[0][0]

    for v in validations:
        v.in_block1 = v.in_block2 = v.in_block3 = False
        if v.status == "error":
            v.block = "error"
        elif v.status == "no_stim" or v.n_detected == 0:
            v.block = "baseline"
        elif _is_paired(v):
            v.block = "block3"
            v.in_block3 = True
        elif v in ladder:
            v.block = "block1"
            v.in_block1 = True
            if sweep_amp is not None and abs(round(v.amplitude_uA_data) - sweep_amp) <= 2:
                v.in_block2 = True
        elif v in width_sweep:
            v.block = "block2"
            v.in_block2 = True
        else:
            v.block = "other"

        if manifest and v.run_id in manifest:
            entry = manifest[v.run_id]
            block = entry.get("block", "").strip().lower()
            if block:
                v.block = block
                v.in_block1 = block == "block1" or "block1" in block
                v.in_block2 = block == "block2" or "block2" in block
                v.in_block3 = block == "block3"

        # Exclusions
        reasons: list[str] = []
        if v.status == "error":
            reasons.append("error")
        if v.block == "block3":
            reasons.append("block3")
        if v.compliance_flag:
            reasons.append("compliance")
        elif np.isfinite(v.n_commanded_total) and not v.count_match and v.status == "ok":
            reasons.append("count_mismatch")
        if manifest and v.run_id in manifest:
            include = manifest[v.run_id].get("include", "").strip().lower()
            if include in ("0", "false", "no", "exclude"):
                reasons.append("manifest")
            elif include in ("1", "true", "yes", "include"):
                reasons = []
        v.included = not reasons and v.status in ("ok", "no_stim")
        v.exclusion_reason = "|".join(reasons)


def validation_frame(validations: list[RunValidation], channels: list[str]) -> pd.DataFrame:
    """table01: one row per run, sorted by block, then amplitude, then phase."""
    rows: list[dict[str, object]] = []
    for v in validations:
        row: dict[str, object] = {
            "run_id": v.run_id,
            "run_folder": Path(v.run_folder).name,
            "label": v.label,
            "block": v.block,
            "in_block1": v.in_block1,
            "in_block2": v.in_block2,
            "in_block3": v.in_block3,
            "included": v.included,
            "exclusion_reason": v.exclusion_reason,
            "status": v.status,
            "sample_rate_hz": v.sample_rate_hz,
            "settings_sample_rate_hz": v.settings_sample_rate_hz,
            "n_files": v.n_files,
            "duration_s": round(v.duration_s, 3) if np.isfinite(v.duration_s) else v.duration_s,
            "timestamp_gaps": v.timestamp_gaps,
            "stim_channel_data": v.stim_channel_data,
            "stim_channel_settings": v.stim_channel_settings,
            "stim_channel_folder": v.stim_channel_folder,
            "amplitude_uA_data": v.amplitude_uA_data,
            "amplitude_uA_settings": v.amplitude_uA_settings,
            "amplitude_uA_folder": v.amplitude_uA_folder,
            "phase_us_data": v.phase_us_data,
            "phase_us_settings": v.phase_us_settings,
            "phase_us_folder": v.phase_us_folder,
            "polarity_data": v.polarity_data,
            "charge_nC_per_phase": v.charge_nC_per_phase,
            "train_period_ms_settings": v.train_period_ms_settings,
            "ipi_median_s": v.ipi_median_s,
            "pulses_per_trigger_settings": v.pulses_per_trigger_settings,
            "n_trains_detected": v.n_trains_detected,
            "n_commanded_total": v.n_commanded_total,
            "n_detected": v.n_detected,
            "count_match": v.count_match,
            "compliance_bit_seen": v.compliance_bit_seen,
            "compliance_flag": v.compliance_flag,
            "amp_settle_bit_seen": v.amp_settle_bit_seen,
            "charge_recovery_bit_seen": v.charge_recovery_bit_seen,
            "hardware_floor_ms": v.hardware_floor_ms,
            "impedance_source": v.impedance_source,
        }
        for channel in channels:
            row[f"rail_pct_{channel}"] = v.rail_pct_by_channel.get(channel, float("nan"))
        row["rail_pct_max"] = v.rail_pct_max
        row["warnings"] = " | ".join(v.warnings)
        row["error"] = v.error
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {name: index for index, name in enumerate(BLOCK_ORDER)}
    frame["_block_order"] = frame["block"].map(lambda b: order.get(b, len(order)))
    frame = frame.sort_values(
        ["_block_order", "amplitude_uA_data", "phase_us_data", "run_id"], na_position="last"
    ).drop(columns="_block_order")
    return frame.reset_index(drop=True)


def rail_long_frame(validations: list[RunValidation]) -> pd.DataFrame:
    rows = []
    for v in validations:
        for rail in v.rails:
            rows.append(
                {
                    "run_id": v.run_id,
                    "channel": rail.channel,
                    "pos_level_uV": rail.pos_level_uV,
                    "neg_level_uV": rail.neg_level_uV,
                    "pos_longest_run_ms": rail.pos_longest_run_ms,
                    "neg_longest_run_ms": rail.neg_longest_run_ms,
                    "n_episodes": rail.n_episodes,
                    "pct_railed": rail.pct_railed,
                    "is_railed": rail.is_railed,
                    "at_adc_full_scale": rail.at_adc_full_scale,
                    "is_flat": rail.is_flat,
                }
            )
    return pd.DataFrame(rows)


def exclusion_frame(validations: list[RunValidation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": v.run_id,
                "run_folder": Path(v.run_folder).name,
                "block": v.block,
                "included": v.included,
                "excluded": not v.included,
                "reason": v.exclusion_reason,
            }
            for v in validations
        ]
    )


def format_validation_ascii(frame: pd.DataFrame) -> str:
    """Compact fixed-width view for the terminal / status line."""
    if frame.empty:
        return "(no runs)"
    columns = [
        "run_id", "block", "included", "exclusion_reason", "amplitude_uA_data", "phase_us_data",
        "n_commanded_total", "n_detected", "compliance_flag", "rail_pct_max", "sample_rate_hz",
    ]
    columns = [c for c in columns if c in frame.columns]
    view = frame[columns].copy()
    for column in ("amplitude_uA_data", "phase_us_data", "n_commanded_total"):
        if column in view:
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:g}")
    if "rail_pct_max" in view:
        view["rail_pct_max"] = view["rail_pct_max"].map(lambda x: f"{x:.2f}")
    return view.to_string(index=False)


__all__ = [
    "BLOCK_ORDER",
    "RailEstimate",
    "RunValidation",
    "assign_blocks",
    "estimate_rail",
    "exclusion_frame",
    "format_validation_ascii",
    "rail_long_frame",
    "railed_mask",
    "read_manifest",
    "validate_run",
    "validation_frame",
]
