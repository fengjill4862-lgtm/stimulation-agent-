"""Function 5 widget UI for RHS power analysis."""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from io import BytesIO
from pathlib import Path

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

from plot_rhs_power_analysis import (
    analyze_pre_post_power,
    parse_power_bands,
    power_rows_to_csv,
)
from plot_rhs_raw_wideband_with_stim_legend import resolve_channel_selection
from rhs_stim import folder_recording_format, read_selected_channels, resolve_stim_channel
from rhs_files import atomic_write_all
from rhd_timing import RECOVERY_FAILED, recover_stim_proxy


def show_function5_power_analysis(
    namespace: MutableMapping[str, object] | None = None,
) -> None:
    """Display Function 5 widgets inside VS Code/Jupyter."""
    default_data_dir = _default_power_data_dir(namespace)

    power_folder_text = widgets.Textarea(
        value=str(default_data_dir),
        description="RHS folder",
        placeholder="Paste RHS data folder path here",
        continuous_update=False,
        layout=widgets.Layout(width="100%", height="46px"),
        style={"description_width": "90px"},
    )
    power_channel_text = widgets.Text(
        value="all",
        description="Channels",
        placeholder="all, A-014, or A-014-16",
        layout=widgets.Layout(width="240px"),
        style={"description_width": "75px"},
    )
    power_bands_text = widgets.Textarea(
        value="delta 0.5-4\ntheta 4-8\nalpha 8-12\nbeta 13-30\ngamma 30-80",
        description="Bands",
        placeholder="delta 0.5-4; theta 4-8; beta 13-30",
        continuous_update=False,
        layout=widgets.Layout(width="520px", height="92px"),
        style={"description_width": "75px"},
    )
    power_color_scale_float = widgets.FloatText(
        value=3.0,
        description="Scale (dB)",
        layout=widgets.Layout(width="185px"),
        style={"description_width": "85px"},
    )

    prepost_window_float = widgets.FloatText(
        value=10.0,
        description="Window (s)",
        layout=widgets.Layout(width="190px"),
        style={"description_width": "85px"},
    )
    prepost_step_float = widgets.FloatText(
        value=5.0,
        description="Step (s)",
        layout=widgets.Layout(width="170px"),
        style={"description_width": "70px"},
    )
    prepost_guard_float = widgets.FloatText(
        value=1.0,
        description="State guard (s)",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "110px"},
    )
    # Used only for .rhd folders, whose stim timing must be recovered from the
    # amplifier trace (no stim channel exists in the format).
    rhd_pulses_int = widgets.IntText(
        value=50,
        description="RHD pulses",
        layout=widgets.Layout(width="180px"),
        style={"description_width": "95px"},
    )
    rhd_width_float = widgets.FloatText(
        value=5.0,
        description="RHD width (ms)",
        layout=widgets.Layout(width="210px"),
        style={"description_width": "125px"},
    )

    power_generate_button = widgets.Button(
        description="Generate Power Preview",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width="210px"),
    )
    power_save_button = widgets.Button(
        description="Save PNG + CSV",
        icon="save",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width="145px"),
    )
    power_status = widgets.HTML(
        value="Choose power settings, then click <b>Generate Power Preview</b>."
    )
    power_target_label = widgets.HTML(value="<b>Target:</b> no preview generated yet")
    power_preview_output = widgets.Output()
    power_current_preview: dict[str, object] = {
        "png": None,
        "csv": None,
        "png_path": None,
        "csv_path": None,
    }

    display(
        widgets.VBox(
            [
                power_folder_text,
                widgets.HBox([power_channel_text, power_color_scale_float]),
                power_bands_text,
                widgets.HTML(
                    value=(
                        "<b>Pre/Post neuromodulation:</b> compares clean recording before "
                        "first stim with clean recording after last stim. For event-locked "
                        "power use Function 6 (session analysis), or "
                        "<code>batch_run_wideband_main_ui.py --power-mode event</code>."
                    )
                ),
                widgets.HBox([prepost_window_float, prepost_step_float, prepost_guard_float]),
                widgets.HBox([rhd_pulses_int, rhd_width_float]),
                widgets.HBox([power_generate_button, power_save_button, power_target_label]),
                power_status,
                power_preview_output,
            ]
        )
    )

    def generate_power_preview(_button=None) -> None:
        """Read RHS data, compute power analysis, and show a preview."""
        power_save_button.disabled = True
        power_current_preview.update(
            {"png": None, "csv": None, "png_path": None, "csv_path": None}
        )
        power_target_label.value = "<b>Target:</b> no preview generated yet"
        power_preview_output.clear_output()

        data_folder = _folder_path_from_text(power_folder_text.value)
        if not data_folder.exists():
            power_status.value = (
                f"<b style='color:#b00020'>Folder not found:</b> {data_folder}"
            )
            return
        recording_format = folder_recording_format(data_folder)
        if recording_format is None:
            power_status.value = (
                f"<b style='color:#b00020'>No .rhs or .rhd files found in:</b> {data_folder}"
            )
            return

        try:
            channels = resolve_channel_selection(power_channel_text.value, data_folder)
            bands = parse_power_bands(power_bands_text.value)
            color_scale_db = float(power_color_scale_float.value)
            if color_scale_db <= 0:
                raise ValueError("Scale (dB) must be greater than 0.")
        except ValueError as exc:
            power_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        power_status.value = f"Reading {recording_format.upper()} files from <b>{data_folder}</b>..."
        try:
            raw_channel_data, stim_channel_info, sample_rate_hz, loaded = _read_power_inputs(
                data_folder, channels
            )
        except (FileNotFoundError, ValueError) as exc:
            power_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        recovery_note = ""
        if stim_channel_info is None and recording_format == "rhd":
            power_status.value = "Recovering the pulse train from the amplifier trace..."
            recovered = recover_stim_proxy(
                data_folder,
                raw_channel_data,
                sample_rate_hz,
                expected_pulses=int(rhd_pulses_int.value),
                pulse_width_ms=float(rhd_width_float.value),
            )
            if recovered is None:
                power_status.value = f"<b style='color:#b00020'>{RECOVERY_FAILED}</b>"
                return
            stim_channel_info = (recovered.label, recovered.stim_uA, ())
            recovery_note = f"; {recovered.note}"
        if stim_channel_info is None:
            power_status.value = (
                "<b style='color:#b00020'>No nonzero stim_data found in any recorded channel in this folder.</b>"
            )
            return
        stim_channel_name, stim_uA_for_events, _pulse_segments = stim_channel_info

        try:
            result = analyze_pre_post_power(
                channel_data=raw_channel_data,
                stim_uA=stim_uA_for_events,
                sample_rate_hz=sample_rate_hz,
                folder=data_folder,
                bands=bands,
                stim_channel_name=stim_channel_name,
                window_s=float(prepost_window_float.value),
                step_s=float(prepost_step_float.value),
                guard_s=float(prepost_guard_float.value),
                color_scale_db=color_scale_db,
            )
            mode_label = "pre/post neuromodulation"
        except ValueError as exc:
            power_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        preview_buffer = BytesIO()
        result.figure.savefig(preview_buffer, format="png", dpi=220)
        plt.close(result.figure)
        preview_png = preview_buffer.getvalue()
        csv_text = power_rows_to_csv(result.rows)

        power_current_preview.update(
            {
                "png": preview_png,
                "csv": csv_text,
                "png_path": result.png_path,
                "csv_path": result.csv_path,
            }
        )
        power_save_button.disabled = False
        power_target_label.value = (
            f"<b>Target:</b> {result.png_path.name} and {result.csv_path.name}"
        )

        with power_preview_output:
            display(
                widgets.HTML(
                    value=(
                        f"<b>{mode_label} power preview</b>; "
                        f"stim channel {result.stim_channel_name}; "
                        f"{len(result.rows)} table rows; target {result.png_path.name}"
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

        power_status.value = (
            f"Generated {mode_label} power preview from {len(loaded)} "
            f"{recording_format.upper()} file(s), "
            f"{len(channels)} channel(s), {len(bands)} band(s){recovery_note}."
        )

    def save_power_outputs(_button=None) -> None:
        """Save the current power-analysis PNG and CSV into the selected data folder."""
        preview_png = power_current_preview.get("png")
        csv_text = power_current_preview.get("csv")
        png_path = power_current_preview.get("png_path")
        csv_path = power_current_preview.get("csv_path")
        if preview_png is None or csv_text is None or png_path is None or csv_path is None:
            power_status.value = (
                "<b style='color:#b00020'>Generate a power preview before saving.</b>"
            )
            return

        assert isinstance(png_path, Path)
        assert isinstance(csv_path, Path)
        # Staged together so a failure cannot leave a new PNG beside a stale CSV.
        atomic_write_all([(png_path, preview_png), (csv_path, str(csv_text))])
        power_target_label.value = f"<b>Saved:</b> {png_path.name} and {csv_path.name}"
        power_status.value = (
            f"Saved power PNG and CSV inside selected data folder: <b>{png_path.parent}</b>"
        )

    power_generate_button.on_click(generate_power_preview)
    power_save_button.on_click(save_power_outputs)


def _read_power_inputs(folder: Path, channels: list[str]):
    """Read the selected channels and locate stim, scanning unselected channels.

    Power can therefore be computed for non-stimulation channels alone while
    still using the correct stimulation timing.
    """
    read = read_selected_channels(folder, channels)
    stim_channel_info = resolve_stim_channel(
        folder, channels, read.stim_channel_data, read.sample_rate_hz
    )
    return read.raw_channel_data, stim_channel_info, read.sample_rate_hz, read.loaded


def _folder_path_from_text(value: str) -> Path:
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


def _default_power_data_dir(namespace: MutableMapping[str, object] | None) -> Path:
    for key in (
        "RESPONSE_DEFAULT_DATA_DIR",
        "EVENT_DEFAULT_DATA_DIR",
        "FILTER_DEFAULT_DATA_DIR",
        "DEFAULT_DATA_DIR",
    ):
        if namespace and key in namespace:
            return Path(namespace[key])
    parent = Path(
        "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
        "20260715 ic implant re"
    )
    if parent.exists():
        if list(parent.glob("*.rhs")):
            return parent
        for child in sorted(parent.iterdir()):
            if child.is_dir() and list(child.glob("*.rhs")):
                return child
    return parent
