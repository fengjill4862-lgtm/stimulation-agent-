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


def parse_amplitude(text: str) -> tuple[float | None, bool]:
    """Parse a leading amplitude token. Returns (milliamps, sign_was_assumed).

    Returns (None, False) when the name carries no amplitude at all, which
    happens for nested run folders that inherit it from their parent, and for
    the numbered baseline recordings. Requiring "mA" somewhere in the name is
    what separates those from a real amplitude token.
    """
    text = text.strip()
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


def parse_run(run_folder: Path, wiring: Wiring, parent_name: str | None = None) -> RunCondition:
    """Parse one run folder into a RunCondition.

    ``parent_name`` is the immediate parent folder's name, used when the run
    folder itself lost the amplitude (``-0_05mA/-0_260819_173803``).
    """
    raw_name = run_folder.name
    stripped = _TIMESTAMP.sub(" ", raw_name).strip()

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
    if wiring.user_marked_artifact:
        flags.append("user_marked_artifact")
    if wiring.common_ground:
        flags.append("common_ground")
    if wiring.after_isoflurane:
        flags.append("after_isoflurane")
    if wiring.raw_name == "(session root)":
        flags.append("baseline_recording")

    # A standalone integer left over after the amplitude token is a replicate
    # index: "-05 1 mA" and "-0_01mA 2" both mean "the Nth run at this level".
    replicate = None
    remainder = _AMPLITUDE.sub("", stripped, count=1) if amplitude is not None else stripped
    remainder = re.sub(r"\b(unsure|com port|interval|start at)\b.*", "", remainder, flags=re.IGNORECASE)
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
    )


def discover_runs(session_folder: Path) -> list[RunCondition]:
    """Find every run folder below a session folder, with its condition.

    A run folder is any folder directly containing ``*.rhd`` files. Walking for
    that rather than assuming a depth handles both the flat layout and the
    nested ``-0_05mA/<run>`` case in this session.
    """
    session_folder = Path(session_folder).expanduser()
    runs: list[RunCondition] = []

    for path in sorted(session_folder.rglob("*")):
        if not path.is_dir():
            continue
        if not any(child.suffix.lower() == ".rhd" for child in path.iterdir() if child.is_file()):
            continue

        relative = path.relative_to(session_folder)
        parts = relative.parts
        # The configuration folder is the top-level one; runs directly under the
        # session folder (the numbered baselines) have no configuration.
        config_name = parts[0] if len(parts) > 1 else ""
        wiring = parse_wiring(config_name) if config_name else Wiring(raw_name="(session root)")
        parent_name = parts[-2] if len(parts) > 1 else None
        runs.append(parse_run(path, wiring, parent_name))

    return runs


__all__ = [
    "RunCondition",
    "Wiring",
    "discover_runs",
    "parse_amplitude",
    "parse_run",
    "parse_wiring",
]
