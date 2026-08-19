"""Discover the sweep runs from their RHS headers and assign them to arms.

Arm membership is decided from the *header* (actual bandwidths, DSP enable),
not from the folder name: a run belongs to every arm whose constant knobs it
matches, so the two shared recordings (analog 0.1 Hz / DSP off = Arm A point
and Arm B "off"; analog 1 Hz / DSP off / 7500 Hz = Arm A point and Arm C
7500 Hz) fall out naturally.  The folder prefix is only a cross-check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from filter_diag.common import dsp_k_from_cutoff, dsp_tau_s
from plot_rhs_raw_wideband_with_stim_legend import RhsHeader
from rhs_reader import read_rhs_header
from stim_analysis.load_rhs import StimSettings, discover_run_folders, parse_run_folder_name, parse_stim_settings
from bw_sweep.config import ARMS, KNOBS, ArmSpec, SweepConfig, tau_ms_from_hz


@dataclass
class SweepRun:
    """One recording folder of the sweep with its instrument settings."""

    folder: Path
    run_id: str
    label: str
    header: RhsHeader
    settings: StimSettings | None
    rhs_files: list[Path]
    sample_rate_hz: float
    lower_hz: float  # actual analog lower bandwidth
    upper_hz: float  # actual analog upper bandwidth
    dsp_enabled: bool
    dsp_hz: float  # actual DSP cutoff when enabled, else 0.0
    dsp_k: int | None
    lower_settle_hz: float
    desired_lower_hz: float
    desired_upper_hz: float
    desired_dsp_hz: float
    channels: list[str]
    stim_channel: str | None
    impedance_kohm: dict[str, float]
    amplitude_uA: float | None
    phase_us: float | None
    n_pulses: int | None
    train_period_s: float | None
    amp_settle: str
    charge_recovery: str
    arms: list[str] = field(default_factory=list)
    arm_labels: dict[str, str] = field(default_factory=dict)  # arm -> "0.1 Hz" / "off"
    replicate: dict[str, str] = field(default_factory=dict)  # arm -> "a"/"b" when a setting is duplicated
    prefix_mismatch: bool = False
    warnings: list[str] = field(default_factory=list)

    # --- derived time constants -----------------------------------------------
    @property
    def knob_values(self) -> dict[str, float]:
        return {"lower_hz": self.lower_hz, "dsp_hz": self.dsp_hz, "upper_hz": self.upper_hz}

    @property
    def fc_hz(self) -> float:
        """Effective high-pass corner: whichever high-pass pole is higher."""
        return max(self.lower_hz, self.dsp_hz if self.dsp_enabled else 0.0)

    @property
    def tau_nominal_ms(self) -> float:
        return tau_ms_from_hz(self.fc_hz)

    @property
    def tau_analog_ms(self) -> float:
        return tau_ms_from_hz(self.lower_hz)

    @property
    def tau_dsp_ms(self) -> float:
        return tau_ms_from_hz(self.dsp_hz) if self.dsp_enabled else float("nan")

    @property
    def tau_dsp_from_k_ms(self) -> float:
        return dsp_tau_s(self.sample_rate_hz, self.dsp_k) * 1e3 if (self.dsp_enabled and self.dsp_k) else float("nan")

    @property
    def tau_second_ms(self) -> float:
        """Time constant of the lower of the two high-pass poles (nan when only one)."""
        if not self.dsp_enabled:
            return float("nan")
        return tau_ms_from_hz(min(self.lower_hz, self.dsp_hz))

    @property
    def dominant_pole(self) -> str:
        if self.dsp_enabled and self.dsp_hz > self.lower_hz:
            return "dsp"
        return "analog"

    def arm_label(self, arm: str) -> str:
        return self.arm_labels.get(arm, "")

    def setting_text(self) -> str:
        dsp = f"DSP {self.dsp_hz:.3g} Hz" if self.dsp_enabled else "DSP off"
        return f"lower {self.lower_hz:.3g} Hz, {dsp}, upper {self.upper_hz:.4g} Hz"


def _log_close(a: float, b: float, log_tol: float) -> bool:
    if a <= 0 or b <= 0:
        return a <= 0 and b <= 0
    return abs(math.log(a / b)) <= log_tol


def read_sweep_run(folder: Path) -> SweepRun:
    """Header + settings of one run folder (no data)."""
    folder = Path(folder)
    files = sorted(folder.glob("*.rhs"))
    if not files:
        raise FileNotFoundError(f"{folder}: no .rhs files")
    header = read_rhs_header(files[0])
    settings = parse_stim_settings(folder)
    meta = parse_run_folder_name(folder.name)
    dsp_enabled = bool(header.dsp_enabled)
    dsp_hz = float(header.actual_dsp_cutoff_hz) if dsp_enabled else 0.0
    dsp_k = dsp_k_from_cutoff(header.sample_rate_hz, dsp_hz) if (dsp_enabled and dsp_hz > 0) else None
    channels = list(header.amplifier_channels)
    impedance = {c: float(header.impedance_magnitude_ohms.get(c, float("nan"))) / 1e3 for c in channels}
    stim_channel = settings.stim_channel if settings is not None else None
    amp_settle = "off"
    charge = "off"
    if settings is not None:
        if settings.enable_amp_settle:
            amp_settle = f"on {settings.pre_amp_settle_us or 0:g}/{settings.post_amp_settle_us or 0:g} us"
        if settings.enable_charge_recovery:
            charge = f"on {settings.charge_recov_on_us or 0:g}-{settings.charge_recov_off_us or 0:g} us"
    return SweepRun(
        folder=folder,
        run_id=meta.run_id,
        label=meta.label,
        header=header,
        settings=settings,
        rhs_files=files,
        sample_rate_hz=float(header.sample_rate_hz),
        lower_hz=float(header.actual_lower_bandwidth_hz),
        upper_hz=float(header.actual_upper_bandwidth_hz),
        dsp_enabled=dsp_enabled,
        dsp_hz=dsp_hz,
        dsp_k=dsp_k,
        lower_settle_hz=float(header.actual_lower_settle_bandwidth_hz),
        desired_lower_hz=float(header.desired_lower_bandwidth_hz),
        desired_upper_hz=float(header.desired_upper_bandwidth_hz),
        desired_dsp_hz=float(header.desired_dsp_cutoff_hz),
        channels=channels,
        stim_channel=stim_channel,
        impedance_kohm=impedance,
        amplitude_uA=(settings.first_amp_uA if settings is not None else None),
        phase_us=(settings.first_phase_us if settings is not None else None),
        n_pulses=(settings.num_pulses if settings is not None else None),
        train_period_s=((settings.train_period_us or 0) * 1e-6 if settings is not None and settings.train_period_us else None),
        amp_settle=amp_settle,
        charge_recovery=charge,
    )


def _matches_constants(run: SweepRun, arm: ArmSpec, cfg: SweepConfig) -> bool:
    values = run.knob_values
    return all(_log_close(values[knob], nominal, cfg.knob_match_log_tol) for knob, nominal in arm.constant.items())


def assign_arms(runs: list[SweepRun], cfg: SweepConfig, arms: tuple[ArmSpec, ...] = ARMS) -> None:
    """Fill ``run.arms`` / ``arm_labels`` / ``replicate`` in place (header-driven)."""
    for run in runs:
        run.arms = []
        run.arm_labels = {}
        run.replicate = {}
        for arm in arms:
            if _matches_constants(run, arm, cfg):
                run.arms.append(arm.name)
                run.arm_labels[arm.name] = arm.label_for(run.knob_values[arm.knob])
        expected_prefix = {arm.folder_prefix.lower(): arm.name for arm in arms}
        by_prefix = expected_prefix.get(run.label.lower())
        if by_prefix is not None and by_prefix not in run.arms:
            run.prefix_mismatch = True
            run.warnings.append(f"folder prefix says arm {by_prefix} but the header settings match {run.arms or 'no arm'}")
        if not run.arms:
            run.warnings.append(f"matches no arm: {run.setting_text()}")
    # replicates: same varied-knob value within one arm
    for arm in arms:
        members = [r for r in runs if arm.name in r.arms]
        members.sort(key=lambda r: (r.knob_values[arm.knob], r.run_id))
        groups: list[list[SweepRun]] = []
        for run in members:
            value = run.knob_values[arm.knob]
            if groups and _log_close(groups[-1][0].knob_values[arm.knob], value, cfg.duplicate_log_tol):
                groups[-1].append(run)
            else:
                groups.append([run])
        for group in groups:
            if len(group) > 1:
                for index, run in enumerate(group):
                    run.replicate[arm.name] = chr(ord("a") + index)
                    run.arm_labels[arm.name] = f"{run.arm_labels[arm.name]} ({run.replicate[arm.name]})"


def apply_duplicate_policy(runs: list[SweepRun], cfg: SweepConfig) -> list[str]:
    """``keep_duplicate`` = both | first | last: drop the other replicate from that arm."""
    notes: list[str] = []
    if cfg.keep_duplicate == "both":
        return notes
    for arm in ARMS:
        reps = [r for r in runs if r.replicate.get(arm.name)]
        if not reps:
            continue
        reps.sort(key=lambda r: r.run_id)
        keep = reps[0] if cfg.keep_duplicate == "first" else reps[-1]
        for run in reps:
            if run is keep:
                continue
            run.arms = [a for a in run.arms if a != arm.name]
            notes.append(f"arm {arm.name}: replicate {run.folder.name} dropped (keep_duplicate={cfg.keep_duplicate})")
    return notes


@dataclass
class KnobCheck:
    arm: str
    ok: bool
    messages: list[str]
    n_runs: int
    values: list[float]


def one_knob_check(runs: list[SweepRun], arm: ArmSpec, cfg: SweepConfig) -> KnobCheck:
    """Within an arm only ``arm.knob`` may differ; everything else must be constant."""
    members = [r for r in runs if arm.name in r.arms]
    messages: list[str] = []
    ok = True
    if not members:
        return KnobCheck(arm.name, False, ["no runs"], 0, [])
    ref = members[0]
    for run in members[1:]:
        for knob in KNOBS:
            if knob == arm.knob:
                continue
            if not _log_close(run.knob_values[knob], ref.knob_values[knob], cfg.knob_match_log_tol):
                ok = False
                messages.append(f"{run.folder.name}: {knob} = {run.knob_values[knob]:g} differs from {ref.folder.name} ({ref.knob_values[knob]:g})")
        for name, a, b in (
            ("sample_rate_hz", run.sample_rate_hz, ref.sample_rate_hz),
            ("lower_settle_hz", run.lower_settle_hz, ref.lower_settle_hz),
            ("amplitude_uA", run.amplitude_uA, ref.amplitude_uA),
            ("phase_us", run.phase_us, ref.phase_us),
            ("stim_channel", run.stim_channel, ref.stim_channel),
            ("channels", tuple(run.channels), tuple(ref.channels)),
            ("amp_settle", run.amp_settle, ref.amp_settle),
            ("charge_recovery", run.charge_recovery, ref.charge_recovery),
        ):
            if a != b:
                ok = False
                messages.append(f"{run.folder.name}: {name} = {a!r} differs from {ref.folder.name} ({b!r})")
        if arm.knob == "dsp_hz" and (run.dsp_enabled != ref.dsp_enabled) and min(run.dsp_hz, ref.dsp_hz) > 0:
            ok = False
            messages.append(f"{run.folder.name}: dsp_enabled differs without an off point")
    values = [r.knob_values[arm.knob] for r in members]
    if len({round(math.log(v), 3) if v > 0 else -99 for v in values}) < 2:
        ok = False
        messages.append(f"the varied knob {arm.knob} does not vary ({values})")
    return KnobCheck(arm.name, ok, messages, len(members), values)


@dataclass
class SweepSet:
    root: Path
    runs: list[SweepRun]
    checks: dict[str, KnobCheck]
    notes: list[str]

    def arm_runs(self, arm: str) -> list[SweepRun]:
        spec = next(a for a in ARMS if a.name == arm)
        members = [r for r in self.runs if arm in r.arms]
        return sorted(members, key=lambda r: (r.knob_values[spec.knob], r.run_id))

    def by_run_id(self, run_id: str) -> SweepRun:
        return next(r for r in self.runs if r.run_id == run_id)

    @property
    def all_ok(self) -> bool:
        return all(check.ok for check in self.checks.values())


def discover_sweep(root: Path, cfg: SweepConfig) -> SweepSet:
    root = Path(root)
    runs = [read_sweep_run(folder) for folder in discover_run_folders(root)]
    runs.sort(key=lambda r: r.run_id)
    assign_arms(runs, cfg)
    notes = apply_duplicate_policy(runs, cfg)
    checks = {arm.name: one_knob_check(runs, arm, cfg) for arm in ARMS}
    for run in runs:
        for message in run.warnings:
            notes.append(f"{run.folder.name}: {message}")
        if len(run.rhs_files) > 1:
            notes.append(f"{run.folder.name}: {len(run.rhs_files)} .rhs files concatenated")
    return SweepSet(root, runs, checks, notes)


def settings_table(sweep: SweepSet) -> pd.DataFrame:
    """table0: one row per run with everything parsed from the header."""
    rows: list[dict[str, object]] = []
    for run in sweep.runs:
        rows.append(
            {
                "run_id": run.run_id,
                "folder": run.folder.name,
                "arms": "+".join(run.arms),
                "arm_A_label": run.arm_label("A"),
                "arm_B_label": run.arm_label("B"),
                "arm_C_label": run.arm_label("C"),
                "shared_run": len(run.arms) > 1,
                "replicate": ",".join(f"{a}:{r}" for a, r in sorted(run.replicate.items())),
                "n_rhs_files": len(run.rhs_files),
                "sample_rate_hz": run.sample_rate_hz,
                "actual_lower_bandwidth_hz": run.lower_hz,
                "actual_upper_bandwidth_hz": run.upper_hz,
                "dsp_enabled": run.dsp_enabled,
                "actual_dsp_cutoff_hz": run.header.actual_dsp_cutoff_hz,
                "effective_dsp_cutoff_hz": run.dsp_hz,
                "dsp_k": run.dsp_k if run.dsp_k is not None else "",
                "actual_lower_settle_bandwidth_hz": run.lower_settle_hz,
                "desired_lower_bandwidth_hz": run.desired_lower_hz,
                "desired_upper_bandwidth_hz": run.desired_upper_hz,
                "desired_dsp_cutoff_hz": run.desired_dsp_hz,
                "fc_effective_hz": run.fc_hz,
                "dominant_pole": run.dominant_pole,
                "tau_nominal_ms": run.tau_nominal_ms,
                "tau_analog_ms": run.tau_analog_ms,
                "tau_dsp_ms": run.tau_dsp_ms,
                "tau_dsp_from_k_ms": run.tau_dsp_from_k_ms,
                "tau_second_pole_ms": run.tau_second_ms,
                "channels": " ".join(run.channels),
                "stim_channel": run.stim_channel or "",
                "amplitude_uA": run.amplitude_uA,
                "phase_us": run.phase_us,
                "n_pulses_commanded": run.n_pulses,
                "train_period_s": run.train_period_s,
                "amp_settle": run.amp_settle,
                "charge_recovery": run.charge_recovery,
                "impedance_kohm": "; ".join(f"{c} {z:.1f}" for c, z in run.impedance_kohm.items()),
                "prefix_mismatch": run.prefix_mismatch,
                "warnings": " | ".join(run.warnings),
            }
        )
    return pd.DataFrame(rows)


def format_settings_ascii(sweep: SweepSet) -> str:
    lines = [f"{'run':16s} {'folder':36s} {'arms':6s} {'lower':>8s} {'DSP':>8s} {'upper':>8s} {'tau_nom':>9s} labels"]
    for run in sweep.runs:
        dsp = f"{run.dsp_hz:.3f}" if run.dsp_enabled else "off"
        labels = ", ".join(f"{a}={run.arm_label(a)}" for a in run.arms)
        lines.append(f"{run.run_id:16s} {run.folder.name[:36]:36s} {'+'.join(run.arms):6s} {run.lower_hz:8.3f} {dsp:>8s} {run.upper_hz:8.1f} {run.tau_nominal_ms:9.2f} {labels}")
    for name, check in sweep.checks.items():
        lines.append(f"one-knob check arm {name}: {'PASS' if check.ok else 'FAIL'} ({check.n_runs} runs)" + ("" if check.ok else " -- " + "; ".join(check.messages)))
    for note in sweep.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


__all__ = [
    "KnobCheck", "SweepRun", "SweepSet", "apply_duplicate_policy", "assign_arms", "discover_sweep",
    "format_settings_ascii", "one_knob_check", "read_sweep_run", "settings_table",
]
