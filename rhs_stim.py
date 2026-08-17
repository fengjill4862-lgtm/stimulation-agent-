#!/usr/bin/env python3
"""Reading selected RHS channels and locating the stimulation channel.

Both operations were previously copy-pasted across the Function 1-5 UI modules
and the batch runner, which is how the batch runner ended up without the
all-recorded-channel fallback that the notebook has.

`find_stim_channel_in_data` in `plot_rhs_raw_wideband_with_stim_legend` searches
only the channels it is handed and takes no folder, so it cannot fall back on
its own. `resolve_stim_channel` here adds that folder-aware fallback behind one
flag, so each function can state its own policy instead of reimplementing it:

- Functions 1 and 2 pass ``fallback=False``  (search selected channels only)
- Functions 3 and 5 pass ``fallback=True``   (then scan every recorded channel)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from plot_rhs_raw_wideband_with_stim_legend import (
    find_stim_channel_in_data,
    read_rhs_amplifier_channel_names,
    read_rhs_folder,
)

DIFFERENT_SAMPLE_RATES = "Selected channels have different sample rates."
STIM_SAMPLE_RATE_MISMATCH = (
    "Stim channel has a different sample rate from selected display channels."
)
NO_STIM_IN_SELECTION = "No nonzero stim_data found in selected channels."
NO_STIM_IN_FOLDER = "No nonzero stim_data found in any recorded channel in this folder."


class ChannelRead(NamedTuple):
    """Amplifier data for the selected channels of one RHS session folder."""

    raw_channel_data: list[tuple[str, object]]
    """(channel_name, raw_uV) pairs -- what the plotting helpers take."""

    stim_channel_data: list[tuple[str, object, object]]
    """(channel_name, raw_uV, stim_uA) triples -- what stim detection takes."""

    sample_rate_hz: float | None
    loaded: object
    """Per-file records from the last channel read, carrying timestamp_gaps."""


def read_selected_channels(folder: Path, channels: list[str]) -> ChannelRead:
    """Read every selected channel, requiring a consistent amplifier sample rate.

    Raises FileNotFoundError or ValueError, which callers surface as UI errors.
    """
    raw_channel_data: list[tuple[str, object]] = []
    stim_channel_data: list[tuple[str, object, object]] = []
    sample_rate_hz: float | None = None
    loaded = None
    for channel in channels:
        raw_uV, stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(folder, channel)
        if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
            raise ValueError(DIFFERENT_SAMPLE_RATES)
        sample_rate_hz = channel_sample_rate_hz
        loaded = channel_loaded
        raw_channel_data.append((channel, raw_uV))
        stim_channel_data.append((channel, raw_uV, stim_uA))
    return ChannelRead(raw_channel_data, stim_channel_data, sample_rate_hz, loaded)


def resolve_stim_channel(
    folder: Path,
    channels: list[str],
    stim_channel_data: list[tuple[str, object, object]],
    sample_rate_hz: float | None,
    *,
    fallback: bool = True,
):
    """Find the stimulation channel, optionally scanning unselected channels.

    Returns the `(channel_name, stim_uA, pulse_segments)` triple from
    `find_stim_channel_in_data`, or None when no channel carries stim data.

    With ``fallback=True`` the selected channels are searched first and every
    other recorded channel afterwards, so neural-response channels can be
    displayed without including the stimulation channel in the selection.

    Note: the fallback reads each candidate channel with `read_rhs_folder`, so a
    folder whose selected channels carry no stim costs one full pass per
    candidate. Only `stim_uA` is actually consulted, so a stim-only reader would
    be markedly cheaper; left as a follow-up to keep this change behavior-preserving.
    """
    stim_channel_info = find_stim_channel_in_data(
        stim_channel_data, slice(0, stim_channel_data[0][1].size)
    )
    if stim_channel_info is not None or not fallback:
        return stim_channel_info

    for candidate_channel in read_rhs_amplifier_channel_names(folder):
        if candidate_channel in channels:
            continue
        candidate_raw_uV, candidate_stim_uA, candidate_sample_rate_hz, _loaded = read_rhs_folder(
            folder, candidate_channel
        )
        if candidate_sample_rate_hz != sample_rate_hz:
            raise ValueError(STIM_SAMPLE_RATE_MISMATCH)
        candidate_info = find_stim_channel_in_data(
            [(candidate_channel, candidate_raw_uV, candidate_stim_uA)],
            slice(0, candidate_stim_uA.size),
        )
        if candidate_info is not None:
            return candidate_info
    return None


__all__ = [
    "DIFFERENT_SAMPLE_RATES",
    "NO_STIM_IN_FOLDER",
    "NO_STIM_IN_SELECTION",
    "STIM_SAMPLE_RATE_MISMATCH",
    "ChannelRead",
    "read_selected_channels",
    "resolve_stim_channel",
]
