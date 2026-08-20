#!/usr/bin/env python3
"""Batch-run the main wideband UI plot functions over RHS session folders."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rhs_filtered_wideband import (
    default_filtered_output_path,
    default_response_output_path,
    format_bandpass_filename_label,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
)
from plot_rhs_power_analysis import (
    analyze_event_locked_power,
    analyze_pre_post_power,
    parse_post_windows_ms,
    parse_power_bands,
    parse_time_window_ms,
    power_rows_to_csv,
)
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    default_output_path,
    plot_raw_channels_with_stim_pulse,
    resolve_channel_selection,
    time_window_label,
)
from plot_rhs_stim_triggered_events import (
    event_window_label,
    parse_post_time_ms,
    build_stim_triggered_events,
    default_stim_events_grid_output_path,
    epoch_filter_channel_data,
    plot_stim_triggered_events_grid,
)
from rhs_files import atomic_write_figure, atomic_write_text
from rhs_reader import RhsFolderData, read_rhs_folder_selected_channels
from rhs_stim import resolve_stim_channel


def find_rhs_session_folders(parent_folder: Path) -> list[Path]:
    """Return the selected RHS folder, or immediate RHS child folders."""
    parent_folder = parent_folder.expanduser().resolve()
    if not parent_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {parent_folder}")
    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {parent_folder}")
    if list(parent_folder.glob("*.rhs")):
        return [parent_folder]
    return [
        child
        for child in sorted(parent_folder.iterdir())
        if child.is_dir() and list(child.glob("*.rhs"))
    ]


def run_raw_plot(
    folder: Path,
    data: RhsFolderData,
    time_window: tuple[float, float] | None,
    max_points: int,
    pulse_number: int,
    dpi: int,
    skip_existing: bool,
) -> Path:
    channel_data = list(zip(data.channels, data.raw_uV, data.stim_uA))
    label = channel_selection_label(data.channels)
    if time_window is not None:
        label = f"{label}_{time_window_label(time_window)}"
    output_path = default_output_path(folder, label)
    if skip_existing and output_path.exists():
        return output_path
    # Matches Function 1: identify stim even when it is not a displayed channel.
    stim_channel_info = resolve_stim_channel(
        folder, data.channels, channel_data, data.sample_rate_hz
    )
    fig = plot_raw_channels_with_stim_pulse(
        channel_data=channel_data,
        sample_rate_hz=data.sample_rate_hz,
        folder=folder,
        output_path=output_path,
        max_points=max_points,
        pulse_number=pulse_number,
        time_window=time_window,
        stim_channel_name=None if stim_channel_info is None else stim_channel_info[0],
        stim_uA=None if stim_channel_info is None else stim_channel_info[1],
    )
    atomic_write_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def run_response_plot(
    folder: Path,
    data: RhsFolderData,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    time_window: tuple[float, float] | None,
    max_points: int,
    dpi: int,
    skip_existing: bool,
) -> Path:
    raw_channel_data = list(zip(data.channels, data.raw_uV))
    label = channel_selection_label(data.channels)
    output_path = default_response_output_path(folder, label, band_hz, amplitude_uV, time_window)
    if skip_existing and output_path.exists():
        return output_path
    fig, _summaries = plot_filtered_channels(
        channel_data=raw_channel_data,
        sample_rate_hz=data.sample_rate_hz,
        folder=folder,
        band_hz=band_hz,
        amplitude_uV=amplitude_uV,
        max_points=max_points,
        time_window=time_window,
        stim_channel_name=None,
    )
    atomic_write_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def run_filtered_plot(
    folder: Path,
    data: RhsFolderData,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    time_window: tuple[float, float] | None,
    max_points: int,
    dpi: int,
    skip_existing: bool,
) -> Path:
    raw_channel_data = list(zip(data.channels, data.raw_uV))
    # Matches Function 2: selected channels first, then the rest of the folder.
    stim_channel_info = resolve_stim_channel(
        folder,
        data.channels,
        list(zip(data.channels, data.raw_uV, data.stim_uA)),
        data.sample_rate_hz,
    )
    stim_channel_name = None if stim_channel_info is None else stim_channel_info[0]
    label = channel_selection_label(data.channels)
    if time_window is not None:
        label = f"{label}_{time_window_label(time_window)}"
    output_path = default_filtered_output_path(folder, label, band_hz, amplitude_uV)
    if skip_existing and output_path.exists():
        return output_path
    fig, _summaries = plot_filtered_channels(
        channel_data=raw_channel_data,
        sample_rate_hz=data.sample_rate_hz,
        folder=folder,
        band_hz=band_hz,
        amplitude_uV=amplitude_uV,
        max_points=max_points,
        time_window=time_window,
        stim_channel_name=stim_channel_name,
    )
    atomic_write_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def run_event_plot(
    folder: Path,
    data: RhsFolderData,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    max_points: int,
    train_gap_ms: float,
    pre_time_ms: float,
    post_window_ms: tuple[float, float | None],
    show_stim_current: bool,
    dpi: int,
    max_events_per_png: int,
    skip_existing: bool,
) -> tuple[list[Path], int, str | None]:
    raw_channel_data = list(zip(data.channels, data.raw_uV))
    stim_channel_data = list(zip(data.channels, data.raw_uV, data.stim_uA))
    # Matches Function 3: selected channels first, then every other recorded
    # channel, so response channels can be plotted without selecting stim.
    try:
        stim_channel_info = resolve_stim_channel(
            folder, data.channels, stim_channel_data, data.sample_rate_hz
        )
    except (FileNotFoundError, ValueError) as exc:
        return [], 0, str(exc)
    if stim_channel_info is None:
        return [], 0, "no nonzero stim_data in any recorded channel"

    stim_channel_name, stim_uA_for_events, _pulse_segments = stim_channel_info
    events = build_stim_triggered_events(
        stim_uA_for_events,
        data.sample_rate_hz,
        None,
        merge_gap_ms=train_gap_ms,
    )
    if not events:
        return [], 0, f"no stim_data onsets on {stim_channel_name}"

    label = channel_selection_label(data.channels)
    window_label = event_window_label(pre_time_ms, post_window_ms)
    base_output_path = default_stim_events_grid_output_path(
        folder=folder,
        channel_label=label,
        band_hz=band_hz,
        amplitude_uV=amplitude_uV,
        time_label=window_label,
        event_count=len(events),
    )
    if not show_stim_current:
        base_output_path = base_output_path.with_name(
            f"{base_output_path.stem}_noStim{base_output_path.suffix}"
        )

    if max_events_per_png <= 0 or len(events) <= max_events_per_png:
        chunks = [(0, events, base_output_path)]
    else:
        chunks = []
        for start in range(0, len(events), max_events_per_png):
            stop = min(start + max_events_per_png, len(events))
            part_index = len(chunks) + 1
            chunk_path = base_output_path.with_name(
                f"{base_output_path.stem}_part{part_index:03d}_"
                f"events{start + 1:03d}-{stop:03d}{base_output_path.suffix}"
            )
            chunks.append((start, events[start:stop], chunk_path))

    if skip_existing and all(output_path.exists() for _start, _chunk_events, output_path in chunks):
        return [output_path for _start, _chunk_events, output_path in chunks], len(events), stim_channel_name

    filtered_channel_data = epoch_filter_channel_data(
        raw_channel_data,
        data.sample_rate_hz,
        band_hz,
        events,
        stim_uA_for_events,
        pre_time_s=pre_time_ms / 1000.0,
        post_time_window_s=(
            post_window_ms[0] / 1000.0,
            None if post_window_ms[1] is None else post_window_ms[1] / 1000.0,
        ),
    )
    output_paths: list[Path] = []
    for _start, chunk_events, output_path in chunks:
        if skip_existing and output_path.exists():
            output_paths.append(output_path)
            continue
        fig, _summaries = plot_stim_triggered_events_grid(
            filtered_channel_data=filtered_channel_data,
            sample_rate_hz=data.sample_rate_hz,
            folder=folder,
            events=chunk_events,
            band_hz=band_hz,
            amplitude_uV=amplitude_uV,
            max_points=max_points,
            events_per_row=3,
            stim_channel_name=stim_channel_name,
            stim_uA=stim_uA_for_events if show_stim_current else None,
            pre_time_s=pre_time_ms / 1000.0,
            post_time_window_s=(
                post_window_ms[0] / 1000.0,
                None if post_window_ms[1] is None else post_window_ms[1] / 1000.0,
            ),
        )
        atomic_write_figure(fig, output_path, dpi=dpi)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths, len(events), stim_channel_name


def run_power_analysis(
    folder: Path,
    data: RhsFolderData,
    bands,
    baseline_window,
    post_windows,
    mode: str,
    prepost_window_s: float,
    prepost_step_s: float,
    prepost_guard_s: float,
    blank_window_ms: tuple[float, float],
    color_scale_db: float,
    train_gap_ms: float,
    dpi: int,
    skip_existing: bool,
) -> tuple[Path | None, Path | None, int, str | None]:
    raw_channel_data = list(zip(data.channels, data.raw_uV))
    stim_channel_data = list(zip(data.channels, data.raw_uV, data.stim_uA))
    # Matches Function 5: power can be computed for non-stim channels alone
    # while still using the correct stimulation timing.
    try:
        stim_channel_info = resolve_stim_channel(
            folder, data.channels, stim_channel_data, data.sample_rate_hz
        )
    except (FileNotFoundError, ValueError) as exc:
        return None, None, 0, str(exc)
    if stim_channel_info is None:
        return None, None, 0, "no nonzero stim_data in any recorded channel"

    stim_channel_name, stim_uA_for_events, _pulse_segments = stim_channel_info
    if mode == "prepost":
        result = analyze_pre_post_power(
            channel_data=raw_channel_data,
            stim_uA=stim_uA_for_events,
            sample_rate_hz=data.sample_rate_hz,
            folder=folder,
            bands=bands,
            stim_channel_name=stim_channel_name,
            window_s=prepost_window_s,
            step_s=prepost_step_s,
            guard_s=prepost_guard_s,
            color_scale_db=color_scale_db,
        )
    else:
        result = analyze_event_locked_power(
            channel_data=raw_channel_data,
            stim_uA=stim_uA_for_events,
            sample_rate_hz=data.sample_rate_hz,
            folder=folder,
            bands=bands,
            baseline_window=baseline_window,
            post_windows=post_windows,
            train_gap_ms=train_gap_ms,
            stim_channel_name=stim_channel_name,
            blank_window_ms=blank_window_ms,
            color_scale_db=color_scale_db,
        )
    if skip_existing and result.png_path.exists() and result.csv_path.exists():
        plt.close(result.figure)
        return result.png_path, result.csv_path, result.event_count, result.stim_channel_name

    atomic_write_figure(result.figure, result.png_path, dpi=dpi)
    plt.close(result.figure)
    atomic_write_text(result.csv_path, power_rows_to_csv(result.rows))
    return result.png_path, result.csv_path, result.event_count, result.stim_channel_name


def write_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(args: argparse.Namespace) -> int:
    parent = args.parent_folder.expanduser().resolve()
    folders = find_rhs_session_folders(parent)
    if args.limit:
        folders = folders[: args.limit]
    if not folders:
        raise FileNotFoundError(f"No immediate RHS session folders found in {parent}")

    requested = set(args.functions)
    band_hz = parse_frequency_range(args.bandpass)
    amplitude_uV = None if args.amplitude.strip().lower() in {"", "all", "none"} else parse_amplitude_range(args.amplitude)
    response_band_hz = parse_frequency_range(args.response_bandpass)
    response_amplitude_uV = (
        None
        if args.response_amplitude.strip().lower() in {"", "all", "none"}
        else parse_amplitude_range(args.response_amplitude)
    )
    post_window_ms = parse_post_time_ms(args.post_ms)
    power_bands = parse_power_bands(args.power_bands)
    power_baseline_window = parse_time_window_ms(
        args.power_baseline_ms,
        default_name="baseline",
    )
    power_post_windows = parse_post_windows_ms(args.power_post_ms)
    power_blank_window = parse_time_window_ms(args.power_blank_ms, default_name="blank")
    rows: list[dict[str, object]] = []
    summary_path = parent / "batch_main_ui_outputs.csv"

    for index, folder in enumerate(folders, start=1):
        started = time.time()
        print(f"[{index}/{len(folders)}] {folder.name}", flush=True)
        row: dict[str, object] = {
            "folder": str(folder),
            "folder_name": folder.name,
            "status": "ok",
        }
        try:
            channels = resolve_channel_selection(args.channels, folder)
            data = read_rhs_folder_selected_channels(folder, channels)
            row.update(
                {
                    "channels": ", ".join(data.channels),
                    "channel_count": len(data.channels),
                    "rhs_file_count": data.rhs_file_count,
                    "sample_rate_hz": data.sample_rate_hz,
                    "sample_count": data.raw_uV[0].size,
                    "duration_s": data.raw_uV[0].size / data.sample_rate_hz,
                    "timestamp_gaps": data.timestamp_gaps,
                }
            )

            if "raw" in requested:
                raw_path = run_raw_plot(
                    folder,
                    data,
                    time_window=None,
                    max_points=args.max_points,
                    pulse_number=args.pulse,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                )
                row["raw_png"] = str(raw_path)
                print(f"  raw: {raw_path.name}", flush=True)

            if "filtered" in requested:
                filtered_path = run_filtered_plot(
                    folder,
                    data,
                    band_hz=band_hz,
                    amplitude_uV=amplitude_uV,
                    time_window=None,
                    max_points=args.max_points,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                )
                row["filtered_png"] = str(filtered_path)
                print(f"  filtered: {filtered_path.name}", flush=True)

            if "response" in requested:
                response_path = run_response_plot(
                    folder,
                    data,
                    band_hz=response_band_hz,
                    amplitude_uV=response_amplitude_uV,
                    time_window=None,
                    max_points=args.max_points,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                )
                row["response_png"] = str(response_path)
                print(f"  response: {response_path.name}", flush=True)

            if "events" in requested:
                event_paths, event_count, stim_channel = run_event_plot(
                    folder,
                    data,
                    band_hz=band_hz,
                    amplitude_uV=amplitude_uV,
                    max_points=args.event_max_points,
                    train_gap_ms=args.train_gap_ms,
                    pre_time_ms=args.pre_ms,
                    post_window_ms=post_window_ms,
                    show_stim_current=not args.hide_stim_current,
                    dpi=args.dpi,
                    max_events_per_png=args.max_events_per_png,
                    skip_existing=args.skip_existing,
                )
                row["stim_channel"] = stim_channel or ""
                row["event_count"] = event_count
                if not event_paths:
                    row["event_status"] = stim_channel or "no events"
                    print(f"  events: skipped ({row['event_status']})", flush=True)
                else:
                    row["event_png_count"] = len(event_paths)
                    row["events_png"] = "; ".join(str(path) for path in event_paths)
                    if len(event_paths) == 1:
                        print(f"  events: {event_paths[0].name}", flush=True)
                    else:
                        print(
                            f"  events: {len(event_paths)} split PNGs "
                            f"({event_count} events total)",
                            flush=True,
                        )

            if "power" in requested:
                power_png, power_csv, power_event_count, power_stim_channel = run_power_analysis(
                    folder,
                    data,
                    bands=power_bands,
                    baseline_window=power_baseline_window,
                    post_windows=power_post_windows,
                    mode=args.power_mode,
                    prepost_window_s=args.power_window_s,
                    prepost_step_s=args.power_step_s,
                    prepost_guard_s=args.power_state_guard_s,
                    blank_window_ms=(
                        power_blank_window.start_ms,
                        power_blank_window.end_ms,
                    ),
                    color_scale_db=args.power_color_scale_db,
                    train_gap_ms=args.train_gap_ms,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                )
                row["power_stim_channel"] = power_stim_channel or ""
                row["power_event_count"] = power_event_count
                if power_png is None or power_csv is None:
                    row["power_status"] = power_stim_channel or "no events"
                    print(f"  power: skipped ({row['power_status']})", flush=True)
                else:
                    row["power_png"] = str(power_png)
                    row["power_csv"] = str(power_csv)
                    print(f"  power: {power_png.name}", flush=True)

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        finally:
            row["elapsed_s"] = round(time.time() - started, 3)
            rows.append(row)
            write_summary(summary_path, rows)

    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run main UI wideband plot functions over RHS session folders."
    )
    parser.add_argument("parent_folder", type=Path)
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=("raw", "filtered", "events", "response", "power"),
        default=("raw", "filtered", "events", "response", "power"),
        help="Main UI plot functions to save for each folder.",
    )
    parser.add_argument("--channels", default="all")
    parser.add_argument("--bandpass", default="all")
    parser.add_argument("--amplitude", default="-100 - 100")
    parser.add_argument("--response-bandpass", default="all")
    parser.add_argument("--response-amplitude", default="-500 - 500")
    parser.add_argument("--max-points", type=int, default=600_000)
    parser.add_argument("--event-max-points", type=int, default=200_000)
    parser.add_argument("--pulse", type=int, default=1)
    parser.add_argument("--train-gap-ms", type=float, default=10.0)
    parser.add_argument("--pre-ms", type=float, default=100.0)
    parser.add_argument("--post-ms", default="500")
    parser.add_argument(
        "--power-bands",
        default="delta 0.5-4; theta 4-8; alpha 8-12; beta 13-30; gamma 30-80",
    )
    parser.add_argument(
        "--power-mode",
        choices=("prepost", "event"),
        default="prepost",
        help="Function 5 mode: prepost neuromodulation or event-triggered.",
    )
    parser.add_argument("--power-window-s", type=float, default=10.0)
    parser.add_argument("--power-step-s", type=float, default=5.0)
    parser.add_argument("--power-state-guard-s", type=float, default=1.0)
    parser.add_argument("--power-baseline-ms", default="-500 to -50")
    parser.add_argument("--power-post-ms", default="1100 to 1600")
    parser.add_argument("--power-blank-ms", default="-10 to 50")
    parser.add_argument("--power-color-scale-db", type=float, default=3.0)
    parser.add_argument("--hide-stim-current", action="store_true")
    parser.add_argument(
        "--max-events-per-png",
        type=int,
        default=60,
        help="Split Function 3 into multiple PNGs when a folder has more events.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing target PNGs instead of regenerating them.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
