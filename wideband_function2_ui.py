#!/usr/bin/env python3
"""Function 2: plot bandpass-filtered samples inside a signed amplitude window.

Numeric bands use a zero-phase Butterworth bandpass; `all` keeps all recorded
frequencies. Only samples inside the signed amplitude window are displayed, and
the y-axis is pinned to those limits.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

from wideband_ui_common import (
    EXAMPLE_RHS_SESSION_DIR,
    NO_PREVIEW_TARGET,
    amplitude_text,
    atomic_write_bytes,
    bandpass_text,
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
    parse_time_window,
    resolve_channel_selection,
    time_window_label,
)
from rhs_stim import read_selected_channels, resolve_stim_channel
from plot_rhs_filtered_wideband import (
    default_filtered_output_path,
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
)


def show_function2_bandpass_filtered(namespace: MutableMapping[str, object] | None = None) -> None:
    """Display Function 2 widgets inside VS Code/Jupyter."""
    data_dir = default_data_dir(
        namespace,
        keys=("DEFAULT_DATA_DIR",),
        fallback=EXAMPLE_RHS_SESSION_DIR,
        publish_as="FILTER_DEFAULT_DATA_DIR",
    )

    filter_folder_text = folder_textarea(str(data_dir))
    filter_channel_text = channels_text("A-014")
    filter_bandpass_text = bandpass_text("all")
    filter_amplitude_text = amplitude_text("-100 - 100 uV")
    filter_time_text = time_text("all", placeholder="all or 10-20")
    filter_max_points_int = max_points_int(600_000)
    filter_generate_button = generate_button()
    filter_save_button = save_button()
    filter_status = status_html("Choose filter settings, then click <b>Generate Preview</b>.")
    filter_target_label = target_label_html()
    filter_preview_output = widgets.Output()
    filter_current_preview: dict[str, object] = {"png": None, "output_path": None}

    display(
        widgets.VBox(
            [
                filter_folder_text,
                widgets.HBox([filter_channel_text, filter_bandpass_text, filter_amplitude_text]),
                widgets.HBox([filter_time_text, filter_max_points_int]),
                widgets.HBox([filter_generate_button, filter_save_button, filter_target_label]),
                filter_status,
                filter_preview_output,
            ]
        )
    )

    def generate_filtered_preview(_button=None) -> None:
        """Read selected RHS folder, apply filters, and show inline preview."""
        filter_save_button.disabled = True
        filter_current_preview["png"] = None
        filter_current_preview["output_path"] = None
        filter_target_label.value = NO_PREVIEW_TARGET
        filter_preview_output.clear_output()

        data_folder = folder_path_from_text(filter_folder_text.value)
        max_points = int(filter_max_points_int.value)

        try:
            channels = resolve_channel_selection(filter_channel_text.value, data_folder)
            band_hz = parse_frequency_range(filter_bandpass_text.value)
            amplitude_uV = parse_amplitude_range(filter_amplitude_text.value)
            time_window = parse_time_window(filter_time_text.value)
        except ValueError as exc:
            filter_status.value = error_html(exc)
            return

        output_label = channel_selection_label(channels)
        if time_window is not None:
            output_label = f"{output_label}_{time_window_label(time_window)}"
        output_path = default_filtered_output_path(data_folder, output_label, band_hz, amplitude_uV)

        folder_error = rhs_folder_error(data_folder)
        if folder_error is not None:
            filter_status.value = folder_error
            return

        filter_status.value = f"Reading RHS files and {format_bandpass_status(band_hz)}..."
        try:
            read = read_selected_channels(data_folder, channels)
        except (FileNotFoundError, ValueError) as exc:
            filter_status.value = error_html(exc)
            return
        channel_data = read.raw_channel_data
        sample_rate_hz = read.sample_rate_hz
        loaded = read.loaded

        # Searches selected channels first, then every other recorded channel.
        stim_channel_info = resolve_stim_channel(
            data_folder, channels, read.stim_channel_data, sample_rate_hz
        )
        stim_channel_name = stim_channel_info[0] if stim_channel_info is not None else None

        try:
            fig, summaries = plot_filtered_channels(
                channel_data=channel_data,
                sample_rate_hz=sample_rate_hz,
                folder=data_folder,
                band_hz=band_hz,
                amplitude_uV=amplitude_uV,
                max_points=max_points,
                time_window=time_window,
                stim_channel_name=stim_channel_name,
            )
        except ValueError as exc:
            filter_status.value = error_html(exc)
            return

        preview_png = figure_to_png_bytes(fig)

        filter_current_preview["png"] = preview_png
        filter_current_preview["output_path"] = output_path
        filter_save_button.disabled = False
        filter_target_label.value = f"<b>Target:</b> {output_path}"

        rms_text = ", ".join(
            f"{item['channel_name']}: {item['filtered_rms_uV']:.2f} uV" for item in summaries
        )
        samples_selected = sum(int(item["samples_in_amplitude_range"]) for item in summaries)
        samples_total = sum(int(item["samples_total"]) for item in summaries)
        percent_selected = samples_selected / samples_total * 100.0 if samples_total else 0.0
        time_status = "all time" if time_window is None else f"{time_window[0]:g}-{time_window[1]:g} s"
        stim_status = (
            "no nonzero stim_data found in selected channels"
            if stim_channel_name is None
            else f"stim channel: {stim_channel_name} *"
        )
        filter_status.value = (
            f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s). "
            f"Time window: {time_status}. "
            f"{stim_status}. "
            f"Signal RMS: {rms_text}. "
            f"Amplitude-selected samples: {samples_selected} "
            f"({percent_selected:.3f}%)."
        )

        show_preview_image(filter_preview_output, preview_png)

    def save_filtered_png(_button=None) -> None:
        """Save the current filtered preview into the selected RHS folder only."""
        preview_png = filter_current_preview.get("png")
        output_path = filter_current_preview.get("output_path")
        if preview_png is None or output_path is None:
            filter_status.value = error_html("Generate a preview before saving.")
            return
        atomic_write_bytes(Path(str(output_path)), preview_png)
        filter_target_label.value = f"<b>Saved:</b> {output_path}"
        filter_status.value = (
            f"Saved PNG inside selected data folder: <b>{Path(str(output_path)).parent}</b>"
        )

    filter_generate_button.on_click(generate_filtered_preview)
    filter_save_button.on_click(save_filtered_png)


__all__ = ["show_function2_bandpass_filtered"]
