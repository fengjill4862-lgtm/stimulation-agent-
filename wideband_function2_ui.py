#!/usr/bin/env python3
"""Function 2: continuous-trace viewer -- raw wideband, or bandpass + amplitude
filtered, with an optional ignore-stim (response-only) display.

This absorbed Functions 1 and 4 (2026-08-20): Raw mode is the old Function 1
(no filter, biphasic pulse caption, timestamp-gap report); Filtered mode with
"Ignore stim" checked is the old Function 4 (never searches for stim_data,
response filenames). Numeric bands use a zero-phase Butterworth bandpass; `all`
keeps all recorded frequencies. Only samples inside the signed amplitude window
are displayed, and the y-axis is pinned to those limits.

All loading goes through ``rhs_stim.read_selected_channels`` -- the single
place a future .rhd reader will plug in.
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
    default_output_path,
    parse_time_window,
    plot_raw_channels_with_stim_pulse,
    resolve_channel_selection,
    sample_slice_for_time_window,
    time_window_label,
)
from rhs_stim import read_selected_channels, resolve_stim_channel
from plot_rhs_filtered_wideband import (
    default_filtered_output_path,
    default_response_output_path,
    bandpass_warning,
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
)

STIM_AMPLITUDE_DEFAULT = "-100 - 100 uV"
RESPONSE_AMPLITUDE_DEFAULT = "-500 - 500 uV"


def show_function2_continuous_traces(namespace: MutableMapping[str, object] | None = None) -> None:
    """Display Function 2 widgets inside VS Code/Jupyter."""
    data_dir = default_data_dir(
        namespace,
        keys=("DEFAULT_DATA_DIR", "FILTER_DEFAULT_DATA_DIR", "RESPONSE_DEFAULT_DATA_DIR"),
        fallback=EXAMPLE_RHS_SESSION_DIR,
        publish_as="DEFAULT_DATA_DIR",
    )

    folder_text = folder_textarea(str(data_dir))
    mode_dropdown = widgets.Dropdown(
        options=[("Raw wideband", "raw"), ("Filtered", "filtered")],
        value="raw",
        description="Mode",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "50px"},
    )
    ignore_stim_checkbox = widgets.Checkbox(
        value=False,
        description="Ignore stim (response only)",
        indent=False,
        layout=widgets.Layout(width="220px"),
    )
    channel_text = channels_text("A-014")
    band_text = bandpass_text("all", placeholder="all or 0.1-150 Hz")
    amp_text = amplitude_text(STIM_AMPLITUDE_DEFAULT)
    window_text = time_text("all", placeholder="all or 10-20")
    points_int = max_points_int(600_000)
    pulse_number_int = widgets.IntText(
        value=1,
        description="Pulse",
        layout=widgets.Layout(width="150px"),
        style={"description_width": "55px"},
    )
    preview_button = generate_button()
    save_btn = save_button()
    status = status_html("Choose a folder and mode, then click <b>Generate Preview</b>.")
    target_label = target_label_html()
    preview_output = widgets.Output()
    current_preview: dict[str, object] = {"png": None, "output_path": None}

    def apply_mode(_change=None) -> None:
        raw = mode_dropdown.value == "raw"
        band_text.disabled = raw
        amp_text.disabled = raw
        ignore_stim_checkbox.disabled = raw
        pulse_number_int.disabled = not raw

    def apply_ignore_stim(change) -> None:
        # Swap the amplitude seed only while the field still holds the other
        # state's default, so a user-entered window is never clobbered.
        if change["new"] and amp_text.value.strip() == STIM_AMPLITUDE_DEFAULT:
            amp_text.value = RESPONSE_AMPLITUDE_DEFAULT
        elif not change["new"] and amp_text.value.strip() == RESPONSE_AMPLITUDE_DEFAULT:
            amp_text.value = STIM_AMPLITUDE_DEFAULT

    mode_dropdown.observe(apply_mode, names="value")
    ignore_stim_checkbox.observe(apply_ignore_stim, names="value")
    apply_mode()

    display(
        widgets.VBox(
            [
                folder_text,
                widgets.HBox([mode_dropdown, ignore_stim_checkbox]),
                widgets.HBox([channel_text, band_text, amp_text]),
                widgets.HBox([window_text, points_int, pulse_number_int]),
                widgets.HBox([preview_button, save_btn, target_label]),
                status,
                preview_output,
            ]
        )
    )

    def generate_preview(_button=None) -> None:
        """Read the selected folder and show the chosen continuous-trace view."""
        save_btn.disabled = True
        current_preview["png"] = None
        current_preview["output_path"] = None
        target_label.value = NO_PREVIEW_TARGET
        preview_output.clear_output()

        data_folder = folder_path_from_text(folder_text.value)
        max_points = int(points_int.value)
        raw_mode = mode_dropdown.value == "raw"
        ignore_stim = bool(ignore_stim_checkbox.value) and not raw_mode

        try:
            channels = resolve_channel_selection(channel_text.value, data_folder)
            time_window = parse_time_window(window_text.value)
            if raw_mode:
                band_hz = None
                amplitude_uV = None
            else:
                band_hz = parse_frequency_range(band_text.value)
                amplitude_uV = parse_amplitude_range(amp_text.value)
        except ValueError as exc:
            status.value = error_html(exc)
            return

        folder_error = rhs_folder_error(data_folder)
        if folder_error is not None:
            status.value = folder_error
            return

        output_label = channel_selection_label(channels)
        if time_window is not None:
            output_label = f"{output_label}_{time_window_label(time_window)}"

        status.value = (
            f"Reading RHS files from <b>{data_folder}</b>..."
            if raw_mode
            else f"Reading RHS files and {format_bandpass_status(band_hz)}..."
        )
        try:
            read = read_selected_channels(data_folder, channels)
        except (FileNotFoundError, ValueError) as exc:
            status.value = error_html(exc)
            return
        sample_rate_hz = read.sample_rate_hz
        loaded = read.loaded

        stim_channel_info = None
        if not ignore_stim:
            # Searches selected channels first, then every other recorded
            # channel, so the stim channel is identified even when hidden.
            try:
                stim_channel_info = resolve_stim_channel(
                    data_folder, channels, read.stim_channel_data, sample_rate_hz
                )
            except (FileNotFoundError, ValueError) as exc:
                status.value = error_html(exc)
                return
        stim_channel_name = None if stim_channel_info is None else stim_channel_info[0]

        try:
            if raw_mode:
                output_path = default_output_path(data_folder, output_label)
                fig = plot_raw_channels_with_stim_pulse(
                    channel_data=read.stim_channel_data,
                    sample_rate_hz=sample_rate_hz,
                    folder=data_folder,
                    output_path=output_path,
                    max_points=max_points,
                    pulse_number=int(pulse_number_int.value),
                    time_window=time_window,
                    stim_channel_name=stim_channel_name,
                    stim_uA=None if stim_channel_info is None else stim_channel_info[1],
                )
                summaries = None
            else:
                if ignore_stim:
                    output_path = default_response_output_path(
                        data_folder,
                        channel_selection_label(channels),
                        band_hz,
                        amplitude_uV,
                        time_window,
                    )
                else:
                    output_path = default_filtered_output_path(
                        data_folder, output_label, band_hz, amplitude_uV
                    )
                fig, summaries = plot_filtered_channels(
                    channel_data=read.raw_channel_data,
                    sample_rate_hz=sample_rate_hz,
                    folder=data_folder,
                    band_hz=band_hz,
                    amplitude_uV=amplitude_uV,
                    max_points=max_points,
                    time_window=time_window,
                    stim_channel_name=stim_channel_name,
                )
        except ValueError as exc:
            status.value = error_html(exc)
            return

        preview_png = figure_to_png_bytes(fig)
        current_preview["png"] = preview_png
        current_preview["output_path"] = output_path
        save_btn.disabled = False
        target_label.value = f"<b>Target:</b> {output_path}"

        if raw_mode:
            first_raw_uV = read.stim_channel_data[0][1]
            display_slice, display_bounds = sample_slice_for_time_window(
                first_raw_uV.size, sample_rate_hz, time_window
            )
            displayed_samples = display_slice.stop - display_slice.start
            time_status = (
                "all time" if time_window is None else f"{display_bounds[0]:g}-{display_bounds[1]:g} s"
            )
            if stim_channel_info is None:
                stim_status = "no nonzero stim_data found in any recorded channel"
            else:
                shown = " *" if stim_channel_name in channels else " (not displayed)"
                stim_status = (
                    f"{len(stim_channel_info[2])} stim pulses on {stim_channel_name}{shown}"
                )
            total_timestamp_gaps = sum(item.timestamp_gaps for item in loaded)
            status.value = (
                f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s), "
                f"displaying {displayed_samples} of {first_raw_uV.size} samples/channel "
                f"({time_status}), {stim_status}, {total_timestamp_gaps} timestamp gaps."
            )
            show_preview_image(preview_output, preview_png)
        elif ignore_stim:
            total_samples = read.raw_channel_data[0][1].size if read.raw_channel_data else 0
            duration_s = total_samples / sample_rate_hz if sample_rate_hz else 0.0
            status.value = (
                f"Generated response-only preview for {len(channels)} channel(s) "
                f"from {len(loaded)} RHS file(s), duration {duration_s:g} s; "
                f"{format_bandpass_status(band_hz)}."
            )
            show_preview_image(
                preview_output,
                preview_png,
                caption=f"<b>Response-only preview</b>; target {output_path.name}",
            )
        else:
            rms_text = ", ".join(
                f"{item['channel_name']}: {item['filtered_rms_uV']:.2f} uV" for item in summaries
            )
            samples_selected = sum(int(item["samples_in_amplitude_range"]) for item in summaries)
            samples_total = sum(int(item["samples_total"]) for item in summaries)
            percent_selected = samples_selected / samples_total * 100.0 if samples_total else 0.0
            time_status = (
                "all time" if time_window is None else f"{time_window[0]:g}-{time_window[1]:g} s"
            )
            if stim_channel_name is None:
                stim_status = "no nonzero stim_data found in any recorded channel"
            else:
                shown = " *" if stim_channel_name in channels else " (not displayed)"
                stim_status = f"stim channel: {stim_channel_name}{shown}"
            warning = bandpass_warning(band_hz)
            warning_html = (
                f" <b style='color:#b26a00'>Warning:</b> {warning}" if warning else ""
            )
            status.value = (
                f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s). "
                f"Time window: {time_status}. "
                f"{stim_status}. "
                f"Signal RMS: {rms_text}. "
                f"Amplitude-selected samples: {samples_selected} "
                f"({percent_selected:.3f}%).{warning_html}"
            )
            show_preview_image(preview_output, preview_png)

    def save_png(_button=None) -> None:
        """Save the current preview into the selected data folder only."""
        preview_png = current_preview.get("png")
        output_path = current_preview.get("output_path")
        if preview_png is None or output_path is None:
            status.value = error_html("Generate a preview before saving.")
            return
        atomic_write_bytes(Path(str(output_path)), preview_png)
        target_label.value = f"<b>Saved:</b> {output_path}"
        status.value = f"Saved PNG inside selected data folder: <b>{Path(str(output_path)).parent}</b>"

    preview_button.on_click(generate_preview)
    save_btn.on_click(save_png)


__all__ = ["show_function2_continuous_traces"]
