#!/usr/bin/env python3
"""Run loading and channel-health selection.

Only three of the eight recorded contacts are real electrodes in this session;
the rest read about 10 Mohm and float at several mV. Selecting them by header
impedance rather than by a hard-coded list keeps the analysis honest if the
montage changes between sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rhd_reader import RhdRunData, read_rhd_header, read_rhd_run

from .config import EvokedConfig
from .naming import RunCondition


@dataclass(frozen=True)
class LoadedRun:
    """One run's data plus the channel-health decision made about it."""

    condition: RunCondition
    data: RhdRunData
    healthy_channels: tuple[str, ...]
    floating_channels: tuple[str, ...]
    impedance_ohms: dict[str, float]

    @property
    def sample_rate_hz(self) -> float:
        return self.data.sample_rate_hz

    @property
    def duration_s(self) -> float:
        return self.data.duration_s


def healthy_channels(header, config: EvokedConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split amplifier channels into connected and floating, by impedance.

    Returns (healthy, floating). If the user named channels explicitly, that
    choice wins and nothing is filtered out.
    """
    if config.channels:
        wanted = tuple(c for c in config.channels if c in header.amplifier_channels)
        missing = tuple(c for c in config.channels if c not in header.amplifier_channels)
        if missing:
            raise ValueError(
                f"{header.path.name}: requested channels not in file: {', '.join(missing)}. "
                f"Available: {', '.join(header.amplifier_channels)}"
            )
        return wanted, ()

    healthy: list[str] = []
    floating: list[str] = []
    for channel in header.amplifier_channels:
        magnitude = header.impedance_magnitude_ohms.get(channel, float("inf"))
        if magnitude <= config.healthy_impedance_max_ohms:
            healthy.append(channel)
        else:
            floating.append(channel)
    return tuple(healthy), tuple(floating)


def load_run(condition: RunCondition, config: EvokedConfig) -> LoadedRun:
    """Load one run folder, keeping only the connected contacts."""
    rhd_files = sorted(condition.run_folder.glob("*.rhd"))
    if not rhd_files:
        raise FileNotFoundError(f"No .rhd files in {condition.run_folder}")

    header = read_rhd_header(rhd_files[0])
    healthy, floating = healthy_channels(header, config)
    if not healthy:
        raise ValueError(
            f"{condition.run_folder.name}: no channel is below "
            f"{config.healthy_impedance_max_ohms:.3g} ohm; every contact looks floating."
        )

    data = read_rhd_run(condition.run_folder, list(healthy))
    return LoadedRun(
        condition=condition,
        data=data,
        healthy_channels=healthy,
        floating_channels=floating,
        impedance_ohms={c: header.impedance_magnitude_ohms.get(c, float("nan")) for c in healthy},
    )


def run_size_bytes(condition: RunCondition) -> int:
    """Total .rhd bytes in a run folder, for progress reporting."""
    return sum(p.stat().st_size for p in condition.run_folder.glob("*.rhd"))


__all__ = ["LoadedRun", "healthy_channels", "load_run", "run_size_bytes"]
