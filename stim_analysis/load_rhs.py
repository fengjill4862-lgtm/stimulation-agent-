"""Loading one RHS run: folder-name and settings.xml metadata, data, events.

Filename metadata is parsed for convenience only and is validated against
``settings.xml`` and against the recorded ``stim_data`` (spec section 2).
Stim events come from the decoded stim marker, never from artifact
thresholding.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plot_rhs_raw_wideband_with_stim_legend import RhsHeader, decode_stim_flags
from rhs_reader import RhsRunData, read_rhs_run
from stim_analysis.config import AnalysisConfig

RUN_ID_PATTERN = re.compile(r"(\d{6})_(\d{6})")


# -----------------------------------------------------------------------------
# Folder-name metadata (convenience only; validated against data)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FolderMeta:
    run_id: str
    label: str
    date: str
    time: str
    stim_channel: str | None
    polarity: str | None
    phase_us: float | None
    amplitude_uA: float | None
    freq_hz: float | None
    pulses: int | None
    rp_ms: float | None
    comp_token: bool


def parse_run_folder_name(name: str) -> FolderMeta:
    """Parse the Function 0 style suffix. Missing pieces are None."""
    match = RUN_ID_PATTERN.search(name)
    date = match.group(1) if match else ""
    time = match.group(2) if match else ""
    run_id = f"{date}_{time}" if match else name
    label = name[: match.start()].strip(" _-") if match else name
    channel = re.search(r"\b([A-D]-\d{3})\b", name)
    polarity = re.search(r"\b(cathodic|anodic)\s+first\b", name, re.I)
    amplitude = re.search(r"(\d+(?:\.\d+)?)\s*uA\b", name, re.I)
    freq = re.search(r"(\d+(?:\.\d+)?)\s*Hz\b", name, re.I)
    pulses = re.search(r"(\d+)\s+pulses?\b", name, re.I)
    phase = re.search(r"(\d+(?:\.\d+)?)\s*us\b", name, re.I)
    rp = re.search(r"(\d+(?:\.\d+)?)\s*ms\s+RP\b", name, re.I)
    return FolderMeta(
        run_id=run_id,
        label=label or name,
        date=date,
        time=time,
        stim_channel=channel.group(1) if channel else None,
        polarity=(polarity.group(1).lower() + " first") if polarity else None,
        phase_us=float(phase.group(1)) if phase else None,
        amplitude_uA=float(amplitude.group(1)) if amplitude else None,
        freq_hz=float(freq.group(1)) if freq else None,
        pulses=int(pulses.group(1)) if pulses else None,
        rp_ms=float(rp.group(1)) if rp else None,
        comp_token=bool(re.search(r"\bcomp\b", name, re.I)),
    )


# -----------------------------------------------------------------------------
# settings.xml (Intan RHX)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StimSettings:
    path: Path
    stim_channels: tuple[str, ...]
    stim_channel: str | None
    polarity: str | None
    shape: str | None
    source: str | None
    pulse_or_train: str | None
    first_amp_uA: float | None
    second_amp_uA: float | None
    first_phase_us: float | None
    second_phase_us: float | None
    interphase_us: float | None
    num_pulses: int | None
    train_period_us: float | None
    refractory_us: float | None
    post_trigger_delay_us: float | None
    enable_amp_settle: bool | None
    pre_amp_settle_us: float | None
    post_amp_settle_us: float | None
    maintain_amp_settle: bool | None
    enable_charge_recovery: bool | None
    charge_recov_on_us: float | None
    charge_recov_off_us: float | None
    sample_rate_hz: float | None
    stim_step_uA: float | None
    dsp_enabled: bool | None
    dsp_cutoff_hz: float | None
    lower_bw_hz: float | None
    upper_bw_hz: float | None
    lower_settle_bw_hz: float | None
    save_dc_amp: bool | None
    save_stim_data: bool | None
    enabled_channels: tuple[str, ...]

    @property
    def pulses_per_trigger(self) -> int | None:
        if self.pulse_or_train == "SinglePulse":
            return 1
        return self.num_pulses

    @property
    def charge_nC_per_phase(self) -> float | None:
        if self.first_amp_uA is None or self.first_phase_us is None:
            return None
        return self.first_amp_uA * self.first_phase_us * 1e-3  # uA * us = pC ; /1000 -> nC


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value: str | None) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _as_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in ("true", "1", "yes")


def parse_stim_settings(folder: Path) -> StimSettings | None:
    """Parse ``settings.xml`` written by Intan RHX next to the .rhs files."""
    path = Path(folder) / "settings.xml"
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    # SampleRateHertz / StimStepSizeMicroAmps sit on the root <IntanRHX> element;
    # bandwidth, DSP and save options sit on <GeneralConfig>. Merge both.
    general = root.find(".//GeneralConfig")
    gattr: dict[str, str] = dict(root.attrib)
    if general is not None:
        gattr.update(general.attrib)

    enabled: list[str] = []
    for channel in root.findall(".//SignalGroup/Channel"):
        if _as_bool(channel.attrib.get("Enabled")):
            name = channel.attrib.get("NativeChannelName")
            if name:
                enabled.append(name)

    stim_nodes = [
        node for node in root.findall(".//StimChannel")
        if _as_bool(node.attrib.get("StimEnabled"))
    ]
    stim_channels = tuple(node.attrib.get("NativeChannelName", "") for node in stim_nodes)
    attr = stim_nodes[0].attrib if stim_nodes else {}

    return StimSettings(
        path=path,
        stim_channels=stim_channels,
        stim_channel=stim_channels[0] if stim_channels else None,
        polarity=attr.get("Polarity"),
        shape=attr.get("Shape"),
        source=attr.get("Source"),
        pulse_or_train=attr.get("PulseOrTrain"),
        first_amp_uA=_as_float(attr.get("FirstPhaseAmplitudeMicroAmps")),
        second_amp_uA=_as_float(attr.get("SecondPhaseAmplitudeMicroAmps")),
        first_phase_us=_as_float(attr.get("FirstPhaseDurationMicroseconds")),
        second_phase_us=_as_float(attr.get("SecondPhaseDurationMicroseconds")),
        interphase_us=_as_float(attr.get("InterphaseDelayMicroseconds")),
        num_pulses=_as_int(attr.get("NumberOfStimPulses")),
        train_period_us=_as_float(attr.get("PulseTrainPeriodMicroseconds")),
        refractory_us=_as_float(attr.get("RefractoryPeriodMicroseconds")),
        post_trigger_delay_us=_as_float(attr.get("PostTriggerDelayMicroseconds")),
        enable_amp_settle=_as_bool(attr.get("EnableAmpSettle")),
        pre_amp_settle_us=_as_float(attr.get("PreStimAmpSettleMicroseconds")),
        post_amp_settle_us=_as_float(attr.get("PostStimAmpSettleMicroseconds")),
        maintain_amp_settle=_as_bool(attr.get("MaintainAmpSettle")),
        enable_charge_recovery=_as_bool(attr.get("EnableChargeRecovery")),
        charge_recov_on_us=_as_float(attr.get("PostStimChargeRecovOnMicroseconds")),
        charge_recov_off_us=_as_float(attr.get("PostStimChargeRecovOffMicroseconds")),
        sample_rate_hz=_as_float(gattr.get("SampleRateHertz")),
        stim_step_uA=_as_float(gattr.get("StimStepSizeMicroAmps")),
        dsp_enabled=_as_bool(gattr.get("DSPEnabled")),
        dsp_cutoff_hz=_as_float(gattr.get("DesiredDSPCutoffFreqHertz")),
        lower_bw_hz=_as_float(gattr.get("DesiredLowerBandwidthHertz")),
        upper_bw_hz=_as_float(gattr.get("DesiredUpperBandwidthHertz")),
        lower_settle_bw_hz=_as_float(gattr.get("DesiredLowerSettleBandwidthHertz")),
        save_dc_amp=_as_bool(gattr.get("SaveDCAmplifierWaveforms")),
        save_stim_data=_as_bool(gattr.get("SaveStimData")),
        enabled_channels=tuple(enabled),
    )


# -----------------------------------------------------------------------------
# Stim events from the decoded marker channel
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StimEvents:
    """Per-pulse facts decoded from ``stim_data`` on the stim channel."""

    onset_samples: np.ndarray  # (E,) int64, first nonzero sample of each pulse
    end_samples: np.ndarray  # (E,) int64, exclusive end of the pulse's nonzero run
    amplitude_uA: np.ndarray  # (E,) max |current| in the pulse
    first_phase_us: np.ndarray  # (E,) duration of the first constant-sign run
    first_phase_sign: np.ndarray  # (E,) -1 cathodic first, +1 anodic first
    compliance_flag: np.ndarray  # (E,) bool, compliance bit seen during/just after pulse
    amp_settle_end_samples: np.ndarray  # (E,) int64, last amp-settle sample after onset, -1 if none
    train_index: np.ndarray  # (E,) int64
    n_trains: int
    sample_rate_hz: float
    compliance_bit_any: bool
    amp_settle_bit_any: bool
    charge_recovery_bit_any: bool

    @property
    def n_events(self) -> int:
        return int(self.onset_samples.size)

    @property
    def onset_s(self) -> np.ndarray:
        return self.onset_samples / self.sample_rate_hz

    @property
    def event_numbers(self) -> np.ndarray:
        return np.arange(1, self.n_events + 1, dtype=np.int64)

    def pulse_width_ms(self) -> np.ndarray:
        return (self.end_samples - self.onset_samples) * 1.0e3 / self.sample_rate_hz


def merged_pulse_segments(
    stim_uA: np.ndarray, sample_rate_hz: float, merge_gap_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero-current runs, merging runs closer than ``merge_gap_ms``.

    Same grouping rule as ``plot_rhs_stim_triggered_events.find_stim_trigger_onsets``
    but also returns the exclusive end of each merged run.
    """
    nonzero = np.flatnonzero(stim_uA != 0)
    if nonzero.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    split_after = np.flatnonzero(np.diff(nonzero) > 1)
    starts = np.r_[nonzero[0], nonzero[split_after + 1]].astype(np.int64)
    ends = np.r_[nonzero[split_after] + 1, nonzero[-1] + 1].astype(np.int64)
    gap = max(0, int(round(merge_gap_ms * 1.0e-3 * sample_rate_hz)))
    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]
    for start, end in zip(starts[1:], ends[1:]):
        if int(start) - merged_ends[-1] <= gap:
            merged_ends[-1] = int(end)
        else:
            merged_starts.append(int(start))
            merged_ends.append(int(end))
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def detect_stim_events(
    stim_uA: np.ndarray,
    stim_words: np.ndarray | None,
    sample_rate_hz: float,
    cfg: AnalysisConfig,
    train_period_s: float | None = None,
) -> StimEvents:
    """Detect pulses on the stim marker channel and decode per-pulse facts."""
    starts, ends = merged_pulse_segments(stim_uA, sample_rate_hz, cfg.pulse_merge_gap_ms)
    n = starts.size
    amplitude = np.zeros(n, dtype=float)
    first_phase_us = np.zeros(n, dtype=float)
    first_sign = np.zeros(n, dtype=np.int64)
    compliance = np.zeros(n, dtype=bool)
    settle_end = np.full(n, -1, dtype=np.int64)

    if stim_words is not None:
        amp_settle, charge_recovery, compliance_bits = decode_stim_flags(stim_words)
    else:
        amp_settle = charge_recovery = compliance_bits = np.zeros(stim_uA.size, dtype=bool)

    two_ms = int(round(2.0e-3 * sample_rate_hz))
    fifty_ms = int(round(50.0e-3 * sample_rate_hz))
    for index in range(n):
        s, e = int(starts[index]), int(ends[index])
        segment = stim_uA[s:e]
        amplitude[index] = float(np.max(np.abs(segment))) if segment.size else 0.0
        sign = float(np.sign(segment[0])) if segment.size else 0.0
        first_sign[index] = int(sign)
        run = 0
        for value in segment:
            if np.sign(value) == sign and value != 0:
                run += 1
            else:
                break
        first_phase_us[index] = run * 1.0e6 / sample_rate_hz
        compliance[index] = bool(np.any(compliance_bits[s : min(stim_uA.size, e + two_ms)]))
        window = amp_settle[s : min(stim_uA.size, s + fifty_ms)]
        hits = np.flatnonzero(window)
        if hits.size:
            settle_end[index] = s + int(hits[-1]) + 1

    # Trains: split when the inter-onset gap is far longer than the commanded period.
    train_index = np.zeros(n, dtype=np.int64)
    if n > 1:
        gaps = np.diff(starts) / sample_rate_hz
        if train_period_s is None or train_period_s <= 0:
            reference = float(np.median(gaps))
        else:
            reference = float(train_period_s)
        split = gaps > cfg.train_split_factor * reference
        train_index[1:] = np.cumsum(split)
    n_trains = int(train_index[-1] + 1) if n else 0

    return StimEvents(
        onset_samples=starts,
        end_samples=ends,
        amplitude_uA=amplitude,
        first_phase_us=first_phase_us,
        first_phase_sign=first_sign,
        compliance_flag=compliance,
        amp_settle_end_samples=settle_end,
        train_index=train_index,
        n_trains=n_trains,
        sample_rate_hz=float(sample_rate_hz),
        compliance_bit_any=bool(np.any(compliance_bits)),
        amp_settle_bit_any=bool(np.any(amp_settle)),
        charge_recovery_bit_any=bool(np.any(charge_recovery)),
    )


# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------


@dataclass
class RunRecord:
    folder: Path
    run_id: str
    label: str
    meta: FolderMeta
    settings: StimSettings | None
    header: RhsHeader
    channels: list[str]  # analysis channels
    all_channels: list[str]  # every enabled amplifier channel in the file
    sample_rate_hz: float
    n_samples: int
    n_files: int
    timestamp_gaps: int
    stim_channel: str | None
    stim_channel_warnings: list[str]
    events: StimEvents | None
    impedance_ohms: dict[str, float]
    impedance_phase_deg: dict[str, float]
    impedance_source: str
    data: RhsRunData | None = None
    stim_uA: np.ndarray | None = None
    stim_words: np.ndarray | None = None
    load_error: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz if self.sample_rate_hz else 0.0

    @property
    def n_events(self) -> int:
        return 0 if self.events is None else self.events.n_events

    def release_data(self) -> None:
        """Drop the large arrays once a run has been processed."""
        self.data = None


def discover_run_folders(parent: Path) -> list[Path]:
    """The parent itself if it holds .rhs files, else its immediate .rhs children."""
    parent = Path(parent).expanduser().resolve()
    if not parent.exists():
        raise FileNotFoundError(f"Folder does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Not a folder: {parent}")
    if list(parent.glob("*.rhs")):
        return [parent]
    return [
        child
        for child in sorted(parent.iterdir())
        if child.is_dir() and list(child.glob("*.rhs"))
    ]


def contact_index(channel: str) -> int:
    match = re.search(r"(\d+)$", channel)
    return int(match.group(1)) if match else -1


def contact_distance_um(channel: str, stim_channel: str | None, cfg: AnalysisConfig) -> float:
    """End-to-end distance from the stim contact along the array."""
    if stim_channel is None:
        return float("nan")
    if cfg.contact_order:
        order = list(cfg.contact_order)
        if channel in order and stim_channel in order:
            return abs(order.index(channel) - order.index(stim_channel)) * cfg.contact_pitch_um
        return float("nan")
    a, b = contact_index(channel), contact_index(stim_channel)
    if a < 0 or b < 0:
        return float("nan")
    return abs(a - b) * cfg.contact_pitch_um


def auto_stim_channel(data: RhsRunData) -> tuple[str | None, dict[str, int]]:
    """Channel with the most nonzero decoded stim current (None if all zero)."""
    counts: dict[str, int] = {}
    for channel in data.channels:
        counts[channel] = int(np.count_nonzero(data.stim_uA(channel)))
    if not counts or max(counts.values()) == 0:
        return None, counts
    return max(counts, key=lambda name: counts[name]), counts


def resolve_stim_channel(
    data: RhsRunData, settings: StimSettings | None, meta: FolderMeta, cfg: AnalysisConfig
) -> tuple[str | None, list[str]]:
    """Data wins; settings.xml and the folder name only produce warnings."""
    warnings: list[str] = []
    detected, counts = auto_stim_channel(data)
    if cfg.stim_channel is not None:
        choice: str | None = cfg.stim_channel
        if choice not in data.channels:
            warnings.append(f"configured stim channel {choice} is not recorded; using data")
            choice = detected
        elif detected is not None and detected != choice:
            warnings.append(
                f"configured stim channel {choice} but stim_data is on {detected} "
                f"({counts.get(detected, 0)} nonzero samples)"
            )
    else:
        choice = detected
    if settings is not None and settings.stim_channel and choice and settings.stim_channel != choice:
        warnings.append(f"settings.xml stim channel {settings.stim_channel} != data {choice}")
    if settings is not None and len(settings.stim_channels) > 1:
        warnings.append(f"settings.xml enables stim on {len(settings.stim_channels)} channels")
    if meta.stim_channel and choice and meta.stim_channel != choice:
        warnings.append(f"folder name says {meta.stim_channel} but stim_data is on {choice}")
    return choice, warnings


def hardware_floor_ms(
    events: StimEvents | None, settings: StimSettings | None, sample_rate_hz: float
) -> float:
    """Earliest sample after onset that can be data: pulse + amp settle.

    Uses the amp-settle flag bits when present, else settings.xml, else 1 ms.
    """
    pulse_ms = 0.0
    settle_ms = 0.0
    if events is not None and events.n_events:
        pulse_ms = float(np.median(events.pulse_width_ms()))
        flagged = events.amp_settle_end_samples >= 0
        if np.any(flagged):
            settle_ms = float(
                np.median(
                    (events.amp_settle_end_samples[flagged] - events.onset_samples[flagged])
                    * 1.0e3 / sample_rate_hz
                )
            )
    if settle_ms <= 0.0 and settings is not None and settings.enable_amp_settle:
        settle_ms = pulse_ms + float(settings.post_amp_settle_us or 0.0) / 1000.0
    floor = max(pulse_ms, settle_ms)
    return max(1.0, floor)


def read_intan_impedance_csv(path: Path) -> dict[str, tuple[float, float]]:
    """Intan RHX impedance export -> {channel: (ohms, phase_deg)}."""
    result: dict[str, tuple[float, float]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Channel Name") or row.get("Channel Number") or "").strip()
            if not name:
                continue
            magnitude = None
            phase = None
            for key, value in row.items():
                if key is None:
                    continue
                if "Magnitude" in key:
                    magnitude = _as_float(value)
                elif "Phase" in key:
                    phase = _as_float(value)
            if magnitude is not None:
                result[name] = (magnitude, phase if phase is not None else float("nan"))
    return result


def load_run(
    folder: Path,
    cfg: AnalysisConfig,
    channels: list[str] | None = None,
    impedance_override: dict[str, tuple[float, float]] | None = None,
) -> RunRecord:
    """Read one run folder: data, metadata, stim channel, events, impedance."""
    folder = Path(folder)
    meta = parse_run_folder_name(folder.name)
    settings = parse_stim_settings(folder)
    data = read_rhs_run(folder, None)  # every enabled amplifier channel
    header = data.header
    all_channels = list(data.channels)
    analysis_channels = list(channels) if channels else all_channels
    missing = [channel for channel in analysis_channels if channel not in all_channels]
    if missing:
        raise ValueError(
            f"{folder.name}: channels not recorded: {', '.join(missing)}. "
            f"Recorded: {', '.join(all_channels)}"
        )

    stim_channel, warnings = resolve_stim_channel(data, settings, meta, cfg)
    events: StimEvents | None = None
    stim_uA: np.ndarray | None = None
    stim_words: np.ndarray | None = None
    if stim_channel is not None:
        stim_uA = data.stim_uA(stim_channel)
        stim_words = data.stim_words[data.channel_index(stim_channel)]
        period_s = None
        if settings is not None and settings.train_period_us:
            period_s = settings.train_period_us * 1e-6
        events = detect_stim_events(stim_uA, stim_words, data.sample_rate_hz, cfg, period_s)

    impedance: dict[str, float] = {}
    phase: dict[str, float] = {}
    source = "missing"
    if impedance_override:
        for channel in analysis_channels:
            if channel in impedance_override:
                impedance[channel] = float(impedance_override[channel][0])
                phase[channel] = float(impedance_override[channel][1])
        if impedance:
            source = "csv"
    if not impedance:
        for channel in analysis_channels:
            magnitude = header.impedance_magnitude_ohms.get(channel, 0.0)
            if magnitude and magnitude > 0:
                impedance[channel] = float(magnitude)
                phase[channel] = float(header.impedance_phase_deg.get(channel, float("nan")))
        if impedance:
            source = "header"
    for channel in analysis_channels:
        impedance.setdefault(channel, float("nan"))
        phase.setdefault(channel, float("nan"))

    return RunRecord(
        folder=folder,
        run_id=meta.run_id,
        label=meta.label,
        meta=meta,
        settings=settings,
        header=header,
        channels=analysis_channels,
        all_channels=all_channels,
        sample_rate_hz=data.sample_rate_hz,
        n_samples=data.n_samples,
        n_files=len(data.rhs_files),
        timestamp_gaps=data.timestamp_gaps,
        stim_channel=stim_channel,
        stim_channel_warnings=warnings,
        events=events,
        impedance_ohms=impedance,
        impedance_phase_deg=phase,
        impedance_source=source,
        data=data,
        stim_uA=stim_uA,
        stim_words=stim_words,
    )


__all__ = [
    "FolderMeta",
    "RunRecord",
    "StimEvents",
    "StimSettings",
    "auto_stim_channel",
    "contact_distance_um",
    "contact_index",
    "detect_stim_events",
    "discover_run_folders",
    "hardware_floor_ms",
    "load_run",
    "merged_pulse_segments",
    "parse_run_folder_name",
    "parse_stim_settings",
    "read_intan_impedance_csv",
    "resolve_stim_channel",
]
