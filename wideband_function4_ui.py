#!/usr/bin/env python3
"""Function 4: plot recorded response only, over an absolute recording-time range.

This is the non-stimulation display path. It does not search for `stim_data`,
mark a stimulation channel, or draw a stim-current row. It reuses the Function 2
filtering and plotting code.
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
    read_rhs_folder,
    resolve_channel_selection,
)
from plot_rhs_filtered_wideband import (
    default_response_output_path,
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
)


def show_function4_recorded_response_only(
    namespace: MutableMapping[str, object] | None = None,
) -> None:
    """Display Function 4 widgets inside VS Code/Jupyter."""
    data_dir = default_data_dir(
        namespace,
        keys=("EVENT_DEFAULT_DATA_DIR", "FILTER_DEFAULT_DATA_DIR"),
        fallback=EXAMPLE_RHS_SESSION_DIR,
        publish_as="RESPONSE_DEFAULT_DATA_DIR",
    )

    response_folder_text = folder_textarea(str(data_dir))
    response_channel_text = channels_text("A-014")
    response_bandpass_text = bandpass_text("all", placeholder="all or 0.1-150 Hz")
    response_amplitude_text = amplitude_text("-500 - 500 uV")
    response_time_text = time_text("all", placeholder="all or 0-60")
    response_max_points_int = max_points_int(600_000)
    response_generate_button = generate_button("Generate Response Preview", width="210px")
    response_save_button = save_button(width="120px")
    response_status = status_html(
        "Choose response settings, then click <b>Generate Response Preview</b>."
    )
    response_target_label = target_label_html()
    response_preview_output = widgets.Output()
    response_current_preview: dict[str, object] = {"png": None, "output_path": None}

    display(
        widgets.VBox(
            [
                response_folder_text,
                widgets.HBox([response_channel_text, response_bandpass_text, response_amplitude_text]),
                widgets.HBox([response_time_text, response_max_points_int]),
                widgets.HBox([response_generate_button, response_save_button, response_target_label]),
                response_status,
                response_preview_output,
            ]
        )
    )

    def generate_response_preview(_button=None) -> None:
        """Read selected response channels and show a filtered full-session preview."""
        response_save_button.disabled = True
        response_current_preview["png"] = None
        response_current_preview["output_path"] = None
        response_target_label.value = NO_PREVIEW_TARGET
        response_preview_output.clear_output()

        data_folder = folder_path_from_text(response_folder_text.value)
        max_points = int(response_max_points_int.value)

        try:
            channels = resolve_channel_selection(response_channel_text.value, data_folder)
            band_hz = parse_frequency_range(response_bandpass_text.value)
            amplitude_uV = parse_amplitude_range(response_amplitude_text.value)
            time_window = parse_time_window(response_time_text.value)
        except ValueError as exc:
            response_status.value = error_html(exc)
            return

        folder_error = rhs_folder_error(data_folder)
        if folder_error is not None:
            response_status.value = folder_error
            return

        response_status.value = f"Reading RHS response data from <b>{data_folder}</b>..."
        raw_channel_data = []
        sample_rate_hz = None
        loaded = None
        try:
            for channel in channels:
                raw_uV, _stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(
                    data_folder, channel
                )
                if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
                    raise ValueError("Selected channels have different sample rates.")
                sample_rate_hz = channel_sample_rate_hz
                loaded = channel_loaded
                raw_channel_data.append((channel, raw_uV))
        except (FileNotFoundError, ValueError) as exc:
            response_status.value = error_html(exc)
            return

        channel_label = channel_selection_label(channels)
        output_path = default_response_output_path(
            data_folder, channel_label, band_hz, amplitude_uV, time_window
        )

        try:
            fig, _summaries = plot_filtered_channels(
                channel_data=raw_channel_data,
                sample_rate_hz=sample_rate_hz,
                folder=data_folder,
                band_hz=band_hz,
                amplitude_uV=amplitude_uV,
                max_points=max_points,
                time_window=time_window,
                stim_channel_name=None,
            )
        except ValueError as exc:
            response_status.value = error_html(exc)
            return

        preview_png = figure_to_png_bytes(fig)

        response_current_preview["png"] = preview_png
        response_current_preview["output_path"] = output_path
        response_save_button.disabled = False
        response_target_label.value = f"<b>Target:</b> {output_path}"

        show_preview_image(
            response_preview_output,
            preview_png,
            caption=f"<b>Response-only preview</b>; target {output_path.name}",
        )

        total_samples = raw_channel_data[0][1].size if raw_channel_data else 0
        duration_s = total_samples / sample_rate_hz if sample_rate_hz else 0.0
        response_status.value = (
            f"Generated response-only preview for {len(channels)} channel(s) "
            f"from {len(loaded)} RHS file(s), duration {duration_s:g} s; "
            f"{format_bandpass_status(band_hz)}."
        )

    def save_response_png(_button=None) -> None:
        """Save the current response-only preview into the selected data folder."""
        preview_png = response_current_preview.get("png")
        output_path = response_current_preview.get("output_path")
        if preview_png is None or output_path is None:
            response_status.value = error_html("Generate a response preview before saving.")
            return
        atomic_write_bytes(Path(str(output_path)), preview_png)
        response_target_label.value = f"<b>Saved:</b> {output_path}"
        response_status.value = (
            f"Saved response-only PNG inside selected data folder: "
            f"<b>{Path(str(output_path)).parent}</b>"
        )

    response_generate_button.on_click(generate_response_preview)
    response_save_button.on_click(save_response_png)


__all__ = ["show_function4_recorded_response_only"]
