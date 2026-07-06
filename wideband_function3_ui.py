"""Function 3 widget UI for stim-triggered response plots."""

from __future__ import annotations

import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from collections.abc import MutableMapping

_cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    find_stim_channel_in_data,
    read_rhs_amplifier_channel_names,
    read_rhs_folder,
    resolve_channel_selection,
)
from plot_rhs_filtered_wideband import (
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
)
from plot_rhs_stim_triggered_events import (
    build_stim_triggered_events,
    default_stim_events_grid_output_path,
    filter_channel_data,
    plot_stim_triggered_events_grid,
)


def show_function3_stim_triggered_events(
    namespace: MutableMapping[str, object] | None = None,
) -> None:
    """Display Function 3 widgets inside VS Code/Jupyter."""
    default_data_dir = _default_event_data_dir(namespace)

    event_folder_text = widgets.Textarea(
        value=str(default_data_dir),
        description="RHS folder",
        placeholder="Paste RHS data folder path here",
        continuous_update=False,
        layout=widgets.Layout(width="100%", height="46px"),
        style={"description_width": "90px"},
    )
    event_channel_text = widgets.Text(
        value="A-014",
        description="Channels",
        placeholder="all, A-014, or A-014-16",
        layout=widgets.Layout(width="240px"),
        style={"description_width": "75px"},
    )
    event_bandpass_text = widgets.Text(
        value="all",
        description="Bandpass",
        placeholder="all or 0.1-150 Hz",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "80px"},
    )
    event_amplitude_text = widgets.Text(
        value="-100 - 100 uV",
        description="Amplitude",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "85px"},
    )
    event_pre_time_float = widgets.FloatText(
        value=100.0,
        description="Pre time (ms)",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "105px"},
    )
    event_post_time_text = widgets.Text(
        value="500",
        description="Post time (ms)",
        placeholder="500 or 20-300",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "110px"},
    )
    event_train_gap_float = widgets.FloatText(
        value=10.0,
        description="Train gap (ms)",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "105px"},
    )
    event_max_points_int = widgets.IntText(
        value=200_000,
        description="Max points",
        layout=widgets.Layout(width="210px"),
        style={"description_width": "90px"},
    )
    event_show_stim_checkbox = widgets.Checkbox(
        value=True,
        description="Show stim current",
        indent=False,
        layout=widgets.Layout(width="170px"),
    )
    event_generate_button = widgets.Button(
        description="Generate Event Previews",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width="200px"),
    )
    event_save_button = widgets.Button(
        description="Save PNG",
        icon="save",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width="120px"),
    )
    event_status = widgets.HTML(
        value="Choose event settings, then click <b>Generate Event Previews</b>."
    )
    event_target_label = widgets.HTML(value="<b>Target:</b> no previews generated yet")
    event_preview_output = widgets.Output()
    event_current_preview: dict[str, object] = {
        "png": None,
        "output_path": None,
        "events": [],
    }

    display(
        widgets.VBox(
            [
                event_folder_text,
                widgets.HBox(
                    [event_channel_text, event_bandpass_text, event_amplitude_text]
                ),
                widgets.HBox(
                    [event_pre_time_float, event_post_time_text, event_train_gap_float]
                ),
                widgets.HBox(
                    [
                        event_max_points_int,
                        event_show_stim_checkbox,
                        event_generate_button,
                        event_save_button,
                        event_target_label,
                    ]
                ),
                event_status,
                event_preview_output,
            ]
        )
    )

    def generate_event_previews(_button=None) -> None:
        """Read RHS data, detect stim onsets, and show one combined inline plot."""
        event_save_button.disabled = True
        event_current_preview["png"] = None
        event_current_preview["output_path"] = None
        event_current_preview["events"] = []
        event_target_label.value = "<b>Target:</b> no previews generated yet"
        event_preview_output.clear_output()

        data_folder = _folder_path_from_text(event_folder_text.value)
        max_points = int(event_max_points_int.value)
        train_gap_ms = float(event_train_gap_float.value)
        show_stim_current = bool(event_show_stim_checkbox.value)
        if train_gap_ms < 0:
            event_status.value = (
                "<b style='color:#b00020'>Train gap must be 0 ms or greater.</b>"
            )
            return

        try:
            channels = resolve_channel_selection(event_channel_text.value, data_folder)
            band_hz = parse_frequency_range(event_bandpass_text.value)
            amplitude_uV = parse_amplitude_range(event_amplitude_text.value)
            pre_time_ms = _parse_pre_time_ms(event_pre_time_float.value)
            post_window_ms = _parse_post_time_ms(event_post_time_text.value)
        except ValueError as exc:
            event_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        pre_time_s = pre_time_ms / 1.0e3
        post_time_window_s = (
            post_window_ms[0] / 1.0e3,
            None if post_window_ms[1] is None else post_window_ms[1] / 1.0e3,
        )
        window_label = _event_window_label(pre_time_ms, post_window_ms)

        if not data_folder.exists():
            event_status.value = (
                f"<b style='color:#b00020'>Folder not found:</b> {data_folder}"
            )
            return
        if not list(data_folder.glob("*.rhs")):
            event_status.value = (
                f"<b style='color:#b00020'>No .rhs files found in:</b> {data_folder}"
            )
            return

        event_status.value = f"Reading RHS files from <b>{data_folder}</b>..."
        raw_channel_data = []
        stim_channel_data = []
        sample_rate_hz = None
        loaded = None
        try:
            for channel in channels:
                raw_uV, stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(
                    data_folder, channel
                )
                if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
                    raise ValueError("Selected channels have different sample rates.")
                sample_rate_hz = channel_sample_rate_hz
                loaded = channel_loaded
                raw_channel_data.append((channel, raw_uV))
                stim_channel_data.append((channel, raw_uV, stim_uA))
        except (FileNotFoundError, ValueError) as exc:
            event_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        full_session_slice = slice(0, stim_channel_data[0][1].size)
        stim_channel_info = find_stim_channel_in_data(
            stim_channel_data, full_session_slice
        )
        if stim_channel_info is None:
            try:
                available_channels = read_rhs_amplifier_channel_names(data_folder)
                for candidate_channel in available_channels:
                    if candidate_channel in channels:
                        continue
                    (
                        candidate_raw_uV,
                        candidate_stim_uA,
                        candidate_sample_rate_hz,
                        _candidate_loaded,
                    ) = read_rhs_folder(data_folder, candidate_channel)
                    if candidate_sample_rate_hz != sample_rate_hz:
                        raise ValueError(
                            "Stim channel has a different sample rate from selected display channels."
                        )
                    candidate_info = find_stim_channel_in_data(
                        [(candidate_channel, candidate_raw_uV, candidate_stim_uA)],
                        slice(0, candidate_stim_uA.size),
                    )
                    if candidate_info is not None:
                        stim_channel_info = candidate_info
                        break
            except (FileNotFoundError, ValueError) as exc:
                event_status.value = f"<b style='color:#b00020'>{exc}</b>"
                return

        if stim_channel_info is None:
            event_status.value = (
                "<b style='color:#b00020'>No nonzero stim_data found in any recorded channel in this folder.</b>"
            )
            return
        stim_detection_channel, stim_uA_for_events, _pulse_segments = stim_channel_info

        try:
            events = build_stim_triggered_events(
                stim_uA_for_events,
                sample_rate_hz,
                None,
                merge_gap_ms=train_gap_ms,
            )
        except ValueError as exc:
            event_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        if not events:
            event_status.value = (
                f"<b style='color:#b00020'>No stim_data onsets found on {stim_detection_channel}.</b>"
            )
            return

        event_status.value = (
            f"Detected {len(events)} stim event(s) on {stim_detection_channel} "
            f"with train gap {train_gap_ms:g} ms; {format_bandpass_status(band_hz)}..."
        )
        try:
            filtered_channel_data = filter_channel_data(
                raw_channel_data, sample_rate_hz, band_hz
            )
        except ValueError as exc:
            event_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        channel_label = channel_selection_label(channels)
        output_path = default_stim_events_grid_output_path(
            folder=data_folder,
            channel_label=channel_label,
            band_hz=band_hz,
            amplitude_uV=amplitude_uV,
            time_label=window_label,
            event_count=len(events),
        )
        if not show_stim_current:
            output_path = output_path.with_name(
                f"{output_path.stem}_noStim{output_path.suffix}"
            )

        try:
            fig, summaries = plot_stim_triggered_events_grid(
                filtered_channel_data=filtered_channel_data,
                sample_rate_hz=sample_rate_hz,
                folder=data_folder,
                events=events,
                band_hz=band_hz,
                amplitude_uV=amplitude_uV,
                max_points=max_points,
                events_per_row=3,
                stim_channel_name=stim_detection_channel,
                stim_uA=stim_uA_for_events if show_stim_current else None,
                pre_time_s=pre_time_s,
                post_time_window_s=post_time_window_s,
            )
        except ValueError as exc:
            event_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        preview_buffer = BytesIO()
        fig.savefig(preview_buffer, format="png", dpi=220)
        plt.close(fig)
        preview_png = preview_buffer.getvalue()

        event_current_preview["png"] = preview_png
        event_current_preview["output_path"] = output_path
        event_current_preview["events"] = events
        event_save_button.disabled = False
        event_target_label.value = f"<b>Target:</b> {output_path}"

        with event_preview_output:
            display(
                widgets.HTML(
                    value=(
                        f"<b>{len(events)} event(s) in one PNG</b>, 3 events per row; "
                        f"window {window_label}; "
                        f"stim current {'shown' if show_stim_current else 'hidden'}; "
                        f"target {output_path.name}"
                    )
                )
            )
            display(
                widgets.Image(
                    value=preview_png,
                    format="png",
                    layout=widgets.Layout(width="100%"),
                )
            )

        _summary_count = len(summaries)
        first_onset = events[0].onset_s
        last_onset = events[-1].onset_s
        blank_note = (
            f"; blanked 0-{post_window_ms[0]:g} ms post-stim"
            if post_window_ms[0] > 0
            else ""
        )
        event_status.value = (
            f"Generated one combined PNG preview with {len(events)} event(s) "
            f"from {len(loaded)} RHS file(s), using {stim_detection_channel} stim_data "
            f"for grouped onsets ({first_onset:g} to {last_onset:g} s; "
            f"train gap {train_gap_ms:g} ms; window {window_label}{blank_note}; "
            f"stim current {'shown' if show_stim_current else 'hidden'})."
        )

    def save_event_pngs(_button=None) -> None:
        """Save the combined stim-triggered event preview into the selected data folder."""
        preview_png = event_current_preview.get("png")
        output_path = event_current_preview.get("output_path")
        events = event_current_preview.get("events", [])
        if preview_png is None or output_path is None:
            event_status.value = (
                "<b style='color:#b00020'>Generate an event preview before saving.</b>"
            )
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        temp_path.write_bytes(preview_png)
        os.replace(temp_path, output_path)

        event_target_label.value = f"<b>Saved:</b> {output_path}"
        event_status.value = (
            f"Saved one combined PNG with {len(events)} event(s) inside selected data folder: "
            f"<b>{output_path.parent}</b>"
        )

    event_generate_button.on_click(generate_event_previews)
    event_save_button.on_click(save_event_pngs)


def _default_event_data_dir(namespace: MutableMapping[str, object] | None) -> Path:
    if namespace and "EVENT_DEFAULT_DATA_DIR" in namespace:
        return Path(namespace["EVENT_DEFAULT_DATA_DIR"])
    if namespace and "DEFAULT_DATA_DIR" in namespace:
        return Path(namespace["DEFAULT_DATA_DIR"])
    return Path(
        "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
        "20260617 stimulation Pt/"
        "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
    )


def _folder_path_from_text(value: str) -> Path:
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


def _parse_pre_time_ms(value: float) -> float:
    pre_ms = float(value)
    if pre_ms < 0:
        raise ValueError("Pre time (ms) must be 0 or greater.")
    return pre_ms


def _parse_post_time_ms(value: str) -> tuple[float, float | None]:
    cleaned = value.strip().lower().replace("ms", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned in {"", "all", "end"}:
        return 0.0, None

    parts = cleaned.split("-", maxsplit=1)
    try:
        if len(parts) == 1:
            stop_ms = float(parts[0])
            if stop_ms <= 0:
                raise ValueError
            return 0.0, stop_ms

        start_ms = float(parts[0])
        stop_ms = float(parts[1])
        if start_ms < 0 or stop_ms <= start_ms:
            raise ValueError
        return start_ms, stop_ms
    except ValueError as exc:
        raise ValueError(
            "Post time (ms) must be a positive value like 500, or a range like 20-300."
        ) from exc


def _event_window_label(pre_time_ms: float, post_window_ms: tuple[float, float | None]) -> str:
    pre_label = _ms_label(pre_time_ms)
    post_start_ms, post_stop_ms = post_window_ms
    if post_stop_ms is None:
        post_label = f"{_ms_number(post_start_ms)}toEnd"
    elif post_start_ms > 0:
        post_label = f"{_ms_number(post_start_ms)}to{_ms_label(post_stop_ms)}"
    else:
        post_label = _ms_label(post_stop_ms)
    return f"pre{pre_label}_post{post_label}"


def _ms_label(value: float) -> str:
    return f"{_ms_number(value)}ms"


def _ms_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")
