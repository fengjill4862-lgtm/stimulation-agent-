#!/usr/bin/env python3
"""Function 1: plot raw wideband data for one, many, or all recorded channels.

No bandpass filter is applied. Long traces use a min/max display envelope, which
reduces plotted points without filtering or modifying the underlying signal.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

from wideband_ui_common import (
    EXAMPLE_RHS_SESSION_DIR,
    NO_PREVIEW_TARGET,
    atomic_write_bytes,
    channels_text,
    default_data_dir,
    display,
    error_html,
    figure_to_png_bytes,
    folder_path_from_text,
    folder_textarea,
    generate_button,
    max_points_int,
    rhs_folder_error,
    save_button,
    show_preview_image,
    status_html,
    target_label_html,
    time_text,
    widgets,
)

from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    default_output_path,
    parse_time_window,
    plot_raw_channels_with_stim_pulse,
    resolve_channel_selection,
    sample_slice_for_time_window,
    time_window_label,
)
from rhs_stim import read_selected_channels, resolve_stim_channel

DEFAULT_CHANNELS = "A-014"
DEFAULT_TIME_WINDOW = "all"
DEFAULT_MAX_POINTS = 600_000
DEFAULT_PULSE_NUMBER = 1


def show_function1_raw_wideband(namespace: MutableMapping[str, object] | None = None) -> None:
    """Display Function 1 widgets inside VS Code/Jupyter."""
    data_dir = default_data_dir(
        namespace,
        keys=(),
        fallback=EXAMPLE_RHS_SESSION_DIR,
        publish_as="DEFAULT_DATA_DIR",
    )

    raw_folder_text = folder_textarea(str(data_dir))
    raw_channel_text = channels_text(DEFAULT_CHANNELS)
    raw_time_text = time_text(DEFAULT_TIME_WINDOW)
    raw_max_points_int = max_points_int(DEFAULT_MAX_POINTS)
    raw_pulse_number_int = widgets.IntText(
        value=DEFAULT_PULSE_NUMBER,
        description="Pulse",
        layout=widgets.Layout(width="150px"),
        style={"description_width": "55px"},
    )
    raw_generate_button = generate_button()
    raw_save_button = save_button()
    raw_status = status_html("Choose an RHS folder, then click <b>Generate Preview</b>.")
    raw_target_label = target_label_html()
    raw_preview_output = widgets.Output()
    raw_current_preview: dict[str, object] = {"png": None, "output_path": None}

    display(
        widgets.VBox(
            [
                raw_folder_text,
                widgets.HBox([raw_channel_text, raw_time_text, raw_max_points_int, raw_pulse_number_int]),
                widgets.HBox([raw_generate_button, raw_save_button, raw_target_label]),
                raw_status,
                raw_preview_output,
            ]
        )
    )

    def generate_raw_preview(_button=None) -> None:
        """Read selected RHS folder and show the raw wideband preview inline."""
        raw_save_button.disabled = True
        raw_current_preview["png"] = None
        raw_current_preview["output_path"] = None
        raw_target_label.value = NO_PREVIEW_TARGET
        raw_preview_output.clear_output()

        data_folder = folder_path_from_text(raw_folder_text.value)
        max_points = int(raw_max_points_int.value)
        pulse_number = int(raw_pulse_number_int.value)
        try:
            channels = resolve_channel_selection(raw_channel_text.value, data_folder)
            time_window = parse_time_window(raw_time_text.value)
        except ValueError as exc:
            raw_status.value = error_html(exc)
            return
        output_label = channel_selection_label(channels)
        if time_window is not None:
            output_label = f"{output_label}_{time_window_label(time_window)}"
        output_path = default_output_path(data_folder, output_label)

        folder_error = rhs_folder_error(data_folder)
        if folder_error is not None:
            raw_status.value = folder_error
            return

        raw_status.value = f"Reading RHS files from <b>{data_folder}</b>..."
        try:
            read = read_selected_channels(data_folder, channels)
        except (FileNotFoundError, ValueError) as exc:
            raw_status.value = error_html(exc)
            return
        channel_data = read.stim_channel_data
        sample_rate_hz = read.sample_rate_hz
        loaded = read.loaded

        # Search the selected channels first, then every other recorded channel,
        # so the stim channel is identified even when it is not being displayed.
        try:
            stim_channel_info = resolve_stim_channel(
                data_folder, channels, channel_data, sample_rate_hz
            )
        except (FileNotFoundError, ValueError) as exc:
            raw_status.value = error_html(exc)
            return
        stim_channel_name = None if stim_channel_info is None else stim_channel_info[0]
        stim_uA_for_caption = None if stim_channel_info is None else stim_channel_info[1]

        try:
            fig = plot_raw_channels_with_stim_pulse(
                channel_data=channel_data,
                sample_rate_hz=sample_rate_hz,
                folder=data_folder,
                output_path=output_path,
                max_points=max_points,
                pulse_number=pulse_number,
                time_window=time_window,
                stim_channel_name=stim_channel_name,
                stim_uA=stim_uA_for_caption,
            )
        except ValueError as exc:
            raw_status.value = error_html(exc)
            return

        preview_png = figure_to_png_bytes(fig)

        raw_current_preview["png"] = preview_png
        raw_current_preview["output_path"] = output_path
        raw_save_button.disabled = False
        raw_target_label.value = f"<b>Target:</b> {output_path}"

        _first_channel_name, first_raw_uV, _first_stim_uA = channel_data[0]
        display_slice, display_bounds = sample_slice_for_time_window(
            first_raw_uV.size, sample_rate_hz, time_window
        )
        displayed_samples = display_slice.stop - display_slice.start
        time_status = "all time" if time_window is None else f"{display_bounds[0]:g}-{display_bounds[1]:g} s"
        if stim_channel_info is None:
            stim_status = "no nonzero stim_data found in any recorded channel"
        else:
            pulse_segments = stim_channel_info[2]
            shown = " *" if stim_channel_name in channels else " (not displayed)"
            stim_status = f"{len(pulse_segments)} stim pulses on {stim_channel_name}{shown}"
        total_timestamp_gaps = sum(item.timestamp_gaps for item in loaded)
        raw_status.value = (
            f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s), "
            f"displaying {displayed_samples} of {first_raw_uV.size} samples/channel "
            f"({time_status}), "
            f"{stim_status}, "
            f"{total_timestamp_gaps} timestamp gaps."
        )

        show_preview_image(raw_preview_output, preview_png)

    def save_raw_png(_button=None) -> None:
        """Save the current raw wideband preview into the selected RHS folder only."""
        preview_png = raw_current_preview.get("png")
        output_path = raw_current_preview.get("output_path")
        if preview_png is None or output_path is None:
            raw_status.value = error_html("Generate a preview before saving.")
            return
        atomic_write_bytes(Path(str(output_path)), preview_png)
        raw_target_label.value = f"<b>Saved:</b> {output_path}"
        raw_status.value = f"Saved PNG inside selected data folder: <b>{Path(str(output_path)).parent}</b>"

    raw_generate_button.on_click(generate_raw_preview)
    raw_save_button.on_click(save_raw_png)


__all__ = ["show_function1_raw_wideband"]
