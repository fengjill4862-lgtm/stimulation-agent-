#!/usr/bin/env python3
"""Folder names to conditions, for the 20260819 session.

Amplitude lives only in the folder name, so this parser is the x-axis of the
whole analysis. The user's rule:

    underscore is the decimal point   -0_02mA -> -0.02 mA,  +0_2mA -> +0.2 mA
    a bare leading zero means tenths  -02mA   -> -0.2 mA,   -08mA  -> -0.8 mA

Names are parsed literally as titled. Some runs may be mislabelled by a
magnitude; nothing here corrects that, because a silent correction would be
indistinguishable from data. Suspect runs are flagged downstream from their
measured response, so the mislabelling can be judged on evidence later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_TIMESTAMP = re.compile(r"_\d{6}_\d{6}")
# The amplitude is the leading numeric token. "mA" is required somewhere in the
# name but not immediately after the digits: "-05 1 mA" puts a replicate index
# in between, and "-0_01 unsure mA" puts a caveat there.
_AMPLITUDE = re.compile(r"^([+-]?)(\d+(?:_\d+)?)")
_HAS_MILLIAMPS = re.compile(r"mA", re.IGNORECASE)
_INTEGER = re.compile(r"(?:^|\s)(\d+)(?=\s|$|mA)", re.IGNORECASE)
# The 20260821-style protocol name written by the stim-control script:
# 0.001mA_-0.001mA_pulsewidth0.3s_interval4.5s_pulsenumber25. Dots are real
# decimal points here, unlike the legacy underscore convention.
_PROTOCOL = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)mA_([+-]?\d+(?:\.\d+)?)mA"
    r"_pulsewidth(\d+(?:\.\d+)?)s_interval(\d+(?:\.\d+)?)s_pulsenumber(\d+)$"
)


@dataclass(frozen=True)
class Wiring:
    """Which wire stimulates, which is the stim return, which is the reference.

    Wires 1 and 2 sit close together in the olfactory bulb near the ACA; wire 3
    is far. That geometry is why some configurations couple into the recording
    and others do not.
    """

    raw_name: str
    stim_wire: int | None = None
    stim_ground_wire: int | None = None
    recording_ground_wire: int | None = None
    common_ground: bool = False
    user_marked_artifact: bool = False
    after_isoflurane: bool = False

    @property
    def label(self) -> str:
        parts = []
        if self.stim_wire is not None:
            parts.append(f"stim {self.stim_wire}")
        if self.stim_ground_wire is not None:
            parts.append(f"gnd {self.stim_ground_wire}")
        if self.recording_ground_wire is not None:
            parts.append(f"ref {self.recording_ground_wire}")
        if self.common_ground:
            parts.append("common")
        if self.after_isoflurane:
            parts.append("post-iso")
        return " / ".join(parts) if parts else self.raw_name


@dataclass(frozen=True)
class ProtocolInfo:
    """Stimulation parameters encoded in a 20260821-style protocol name."""

    amplitude_mA: float  # leading phase, signed: positive = anodic-leading
    second_phase_mA: float
    pulse_width_s: float
    interval_s: float
    pulse_number: int


@dataclass(frozen=True)
class RunCondition:
    """One recording folder's condition, plus every caveat attached to it."""

    run_folder: Path
    raw_name: str
    wiring: Wiring
    amplitude_mA: float | None = None
    sign_assumed: bool = False
    amplitude_from_parent: bool = False
    replicate: int | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    # Per-run protocol from a 20260821-style name; None/"" for legacy sessions.
    expected_pulses_run: int | None = None
    pulse_width_s_run: float | None = None
    interval_s_run: float | None = None
    protocol_source: str = ""  # "", "folder_name", "onset_txt", "rhd_name"
    has_onset_file: bool = False

    @property
    def amplitude_label(self) -> str:
        if self.amplitude_mA is None:
            return "unknown"
        return f"{self.amplitude_mA:+.3g} mA"

    @property
    def abs_amplitude_mA(self) -> float | None:
        return None if self.amplitude_mA is None else abs(self.amplitude_mA)

    @property
    def polarity(self) -> str:
        if self.amplitude_mA is None:
            return "unknown"
        return "cathodic" if self.amplitude_mA < 0 else "anodic"


def parse_protocol_name(text: str) -> ProtocolInfo | None:
    """Parse a 20260821-style protocol name, with or without its extension."""
    cleaned = text.strip()
    for suffix in (".onset.txt", ".rhd"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    match = _PROTOCOL.match(cleaned)
    if match is None:
        return None
    return ProtocolInfo(
        amplitude_mA=float(match.group(1)),
        second_phase_mA=float(match.group(2)),
        pulse_width_s=float(match.group(3)),
        interval_s=float(match.group(4)),
        pulse_number=int(match.group(5)),
    )


def protocol_for_run(run_folder: Path) -> tuple[ProtocolInfo | None, str]:
    """Find protocol info for a run: folder name, then file names inside.

    Name-only -- never opens a file; the onset epoch value inside the
    ``.onset.txt`` is read later by ``scope_sync``.
    """
    info = parse_protocol_name(run_folder.name)
    if info is not None:
        return info, "folder_name"
    try:
        children = sorted(child.name for child in run_folder.iterdir() if child.is_file())
    except OSError:
        return None, ""
    for name in children:
        if name.endswith(".onset.txt"):
            info = parse_protocol_name(name)
            if info is not None:
                return info, "onset_txt"
    for name in children:
        if name.endswith(".rhd"):
            info = parse_protocol_name(name)
            if info is not None:
                return info, "rhd_name"
    return None, ""


def parse_amplitude(text: str) -> tuple[float | None, bool]:
    """Parse a leading amplitude token. Returns (milliamps, sign_was_assumed).

    Returns (None, False) when the name carries no amplitude at all, which
    happens for nested run folders that inherit it from their parent, and for
    the numbered baseline recordings. Requiring "mA" somewhere in the name is
    what separates those from a real amplitude token.
    """
    text = text.strip()
    # A protocol-style name is handled by parse_protocol_name; the legacy rules
    # would misread its "0.001" prefix as 0 mA.
    if parse_protocol_name(text) is not None:
        return None, False
    if not _HAS_MILLIAMPS.search(text):
        return None, False

    match = _AMPLITUDE.match(text)
    if match is None:
        return None, False

    sign_text, digits = match.group(1), match.group(2)

    if "_" in digits:
        value = float(digits.replace("_", "."))
    elif digits.startswith("0") and len(digits) > 1:
        # "02" -> 0.2, "05" -> 0.5, "08" -> 0.8: the "0_" underscore was dropped.
        value = int(digits) / (10 ** (len(digits) - 1))
    else:
        value = float(digits)

    sign_assumed = sign_text == ""
    if sign_text == "-":
        value = -value
    return value, sign_assumed


def parse_wiring(folder_name: str) -> Wiring:
    """Parse a configuration folder name into a Wiring record."""
    lowered = folder_name.lower()

    def _wire_after(*phrases: str) -> int | None:
        for phrase in phrases:
            match = re.search(rf"{phrase}\s*(\d+)", lowered)
            if match:
                return int(match.group(1))
        return None

    stim_ground = _wire_after(r"stim ground", r"stim gnd")
    recording_ground = _wire_after(r"recording ground", r"rec ground", r"recording gnd")
    stim_wire = _wire_after(r"stim")

    # "stim 1 ground 2 recording ground 3": the bare "ground 2" is the stim return.
    if stim_ground is None:
        match = re.search(r"stim\s*\d+\s*ground\s*(\d+)", lowered)
        if match:
            stim_ground = int(match.group(1))

    return Wiring(
        raw_name=folder_name,
        stim_wire=stim_wire,
        stim_ground_wire=stim_ground,
        recording_ground_wire=recording_ground,
        common_ground="common ground" in lowered,
        user_marked_artifact="artifact" in lowered,
        after_isoflurane="iso" in lowered,
    )


def parse_run(
    run_folder: Path,
    wiring: Wiring,
    parent_name: str | None = None,
    protocol: tuple[ProtocolInfo | None, str] | None = None,
) -> RunCondition:
    """Parse one run folder into a RunCondition.

    ``parent_name`` is the immediate parent folder's name, used when the run
    folder itself lost the amplitude (``-0_05mA/-0_260819_173803``).
    ``protocol`` is a precomputed ``protocol_for_run`` result; None recomputes.
    """
    raw_name = run_folder.name
    stripped = _TIMESTAMP.sub(" ", raw_name).strip()

    info, protocol_source = protocol if protocol is not None else protocol_for_run(run_folder)
    try:
        has_onset = any(
            child.name.endswith(".onset.txt") for child in run_folder.iterdir() if child.is_file()
        )
    except OSError:
        has_onset = False
    protocol_evidence = info is not None or has_onset

    if info is not None:
        amplitude, sign_assumed = info.amplitude_mA, False
        from_parent = False
    else:
        amplitude, sign_assumed = parse_amplitude(stripped)
        from_parent = False
        if amplitude is None and parent_name:
            amplitude, sign_assumed = parse_amplitude(_TIMESTAMP.sub(" ", parent_name).strip())
            from_parent = amplitude is not None

    flags: list[str] = []
    lowered = stripped.lower()
    if "unsure" in lowered:
        flags.append("unsure")
    if "start at" in lowered:
        flags.append("late_start")
    if "interval" in lowered:
        flags.append("interval_noted")
    if "com port" in lowered:
        flags.append("com_port_noted")
    if sign_assumed and amplitude is not None:
        flags.append("sign_assumed")
    if from_parent:
        flags.append("amplitude_from_parent")
    if amplitude is None:
        flags.append("amplitude_unknown")
    if info is not None:
        flags.append("protocol_named")
    if wiring.user_marked_artifact:
        flags.append("user_marked_artifact")
    if wiring.common_ground:
        flags.append("common_ground")
    if wiring.after_isoflurane:
        flags.append("after_isoflurane")
    # Baseline classification: an explicit "baseline" name always wins; a run
    # carrying protocol evidence (a protocol name or an onset file) is never a
    # baseline; the legacy session-root rule covers the rest.
    if "baseline" in lowered:
        flags.append("baseline_recording")
    elif wiring.raw_name == "(session root)" and not protocol_evidence:
        flags.append("baseline_recording")

    # A standalone integer left over after the amplitude token is a replicate
    # index: "-05 1 mA" and "-0_01mA 2" both mean "the Nth run at this level".
    # Protocol-named runs encode no replicate; their folder prefix ("1_") is not one.
    replicate = None
    if info is None:
        remainder = _AMPLITUDE.sub("", stripped, count=1) if amplitude is not None else stripped
        remainder = re.sub(
            r"\b(unsure|com port|interval|start at)\b.*", "", remainder, flags=re.IGNORECASE
        )
        integer_match = _INTEGER.search(remainder)
        if integer_match:
            replicate = int(integer_match.group(1))

    return RunCondition(
        run_folder=run_folder,
        raw_name=raw_name,
        wiring=wiring,
        amplitude_mA=amplitude,
        sign_assumed=sign_assumed,
        amplitude_from_parent=from_parent,
        replicate=replicate,
        flags=tuple(flags),
        notes=stripped,
        expected_pulses_run=info.pulse_number if info else None,
        pulse_width_s_run=info.pulse_width_s if info else None,
        interval_s_run=info.interval_s if info else None,
        protocol_source=protocol_source,
        has_onset_file=has_onset,
    )


def discover_runs(
    session_folder: Path, wiring_label: str | None = None
) -> list[RunCondition]:
    """Find every run folder below a session folder, with its condition.

    A run folder is any folder directly containing ``*.rhd`` files. Walking for
    that rather than assuming a depth handles both the flat layout and the
    nested ``-0_05mA/<run>`` case in this session.

    The session folder may itself be one wiring-condition folder (its name
    carries "stim"): every run below it then belongs to that wiring, including
    the runs sitting directly inside it. Flat sessions whose runs carry
    protocol evidence (a protocol name or an onset file) are stim runs at the
    root; their wiring is ``wiring_label`` when given, else the session folder
    name -- never "(session root)".
    """
    session_folder = Path(session_folder).expanduser()
    root_wiring = (
        parse_wiring(session_folder.name) if "stim" in session_folder.name.lower() else None
    )
    runs: list[RunCondition] = []

    for path in sorted(session_folder.rglob("*")):
        if not path.is_dir():
            continue
        if not any(child.suffix.lower() == ".rhd" for child in path.iterdir() if child.is_file()):
            continue

        relative = path.relative_to(session_folder)
        parts = relative.parts
        protocol = protocol_for_run(path)
        try:
            has_onset = any(
                child.name.endswith(".onset.txt") for child in path.iterdir() if child.is_file()
            )
        except OSError:
            has_onset = False
        protocol_evidence = protocol[0] is not None or has_onset

        if len(parts) > 1:
            wiring = root_wiring if root_wiring is not None else parse_wiring(parts[0])
        elif protocol_evidence:
            # Flat-session stim run: the user's label wins, then a root that is
            # itself a wiring folder, then the session folder's name.
            if wiring_label:
                wiring = Wiring(raw_name=wiring_label)
            elif root_wiring is not None:
                wiring = root_wiring
            else:
                wiring = Wiring(raw_name=session_folder.name)
        else:
            wiring = root_wiring if root_wiring is not None else Wiring(raw_name="(session root)")
        parent_name = parts[-2] if len(parts) > 1 else None
        runs.append(parse_run(path, wiring, parent_name, protocol=protocol))

    return runs


def contact_index(channel: str) -> int:
    """Trailing integer of a channel name: B-017 -> 17. -1 when absent."""
    match = re.search(r"(\d+)$", channel)
    return int(match.group(1)) if match else -1


def contact_positions_um(
    channels,
    pitch_um: float,
    order: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """{channel: position in um}, zeroed at the lowest-index channel present.

    Relative positions only: the pitch between contacts is known (500 um on
    this array) but the offset from the stim site to the first contact was
    never measured, so no absolute distance exists to report.
    """
    indices: dict[str, float] = {}
    for channel in channels:
        if order is not None:
            indices[channel] = float(order.index(channel)) if channel in order else float("nan")
        else:
            index = contact_index(channel)
            indices[channel] = float(index) if index >= 0 else float("nan")

    valid = [value for value in indices.values() if value == value]  # NaN-safe
    origin = min(valid) if valid else 0.0
    return {
        channel: (value - origin) * pitch_um if value == value else float("nan")
        for channel, value in indices.items()
    }


__all__ = [
    "ProtocolInfo",
    "RunCondition",
    "Wiring",
    "contact_index",
    "contact_positions_um",
    "discover_runs",
    "parse_amplitude",
    "parse_protocol_name",
    "parse_run",
    "parse_wiring",
    "protocol_for_run",
]
