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
    analyze_event_locked_power,
    analyze_pre_post_power,
    parse_post_windows_ms,
    parse_power_bands,
    parse_time_window_ms,
    power_rows_to_csv,
)
from plot_rhs_raw_wideband_with_stim_legend import (
    find_stim_channel_in_data,
    read_rhs_amplifier_channel_names,
    read_rhs_folder,
    resolve_channel_selection,
)


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
    power_mode_dropdown = widgets.Dropdown(
        options=[
            ("Pre/Post neuromodulation", "prepost"),
            ("Stim-triggered events", "event"),
        ],
        value="prepost",
        description="Mode",
        layout=widgets.Layout(width="290px"),
        style={"description_width": "60px"},
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

    event_baseline_text = widgets.Text(
        value="-500 to -50",
        description="Baseline (ms)",
        placeholder="-500 to -50",
        layout=widgets.Layout(width="240px"),
        style={"description_width": "105px"},
    )
    event_post_text = widgets.Textarea(
        value="1100 to 1600",
        description="Post (ms)",
        placeholder="1100 to 1600; 1600 to 2100",
        continuous_update=False,
        layout=widgets.Layout(width="330px", height="70px"),
        style={"description_width": "85px"},
    )
    event_train_gap_float = widgets.FloatText(
        value=60.0,
        description="Train gap (ms)",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "105px"},
    )
    event_blank_text = widgets.Text(
        value="-10 to 50",
        description="Blank (ms)",
        placeholder="-10 to 50",
        layout=widgets.Layout(width="210px"),
        style={"description_width": "85px"},
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
                widgets.HBox([power_mode_dropdown, power_channel_text, power_color_scale_float]),
                power_bands_text,
                widgets.HTML(
                    value=(
                        "<b>Pre/Post mode:</b> compares clean recording before first stim "
                        "with clean recording after last stim."
                    )
                ),
                widgets.HBox([prepost_window_float, prepost_step_float, prepost_guard_float]),
                widgets.HTML(
                    value=(
                        "<b>Stim-triggered mode:</b> epochs each train, blanks each pulse, "
                        "then filters and computes paired baseline-vs-post power."
                    )
                ),
                widgets.HBox([event_baseline_text, event_post_text]),
                widgets.HBox([event_train_gap_float, event_blank_text]),
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
        if not list(data_folder.glob("*.rhs")):
            power_status.value = (
                f"<b style='color:#b00020'>No .rhs files found in:</b> {data_folder}"
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

        power_status.value = f"Reading RHS files from <b>{data_folder}</b>..."
        try:
            raw_channel_data, stim_channel_info, sample_rate_hz, loaded = _read_power_inputs(
                data_folder, channels
            )
        except (FileNotFoundError, ValueError) as exc:
            power_status.value = f"<b style='color:#b00020'>{exc}</b>"
            return

        if stim_channel_info is None:
            power_status.value = (
                "<b style='color:#b00020'>No nonzero stim_data found in any recorded channel in this folder.</b>"
            )
            return
        stim_channel_name, stim_uA_for_events, _pulse_segments = stim_channel_info

        try:
            if power_mode_dropdown.value == "prepost":
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
            else:
                baseline_window = parse_time_window_ms(
                    event_baseline_text.value,
                    default_name="baseline",
                )
                post_windows = parse_post_windows_ms(event_post_text.value)
                blank_window = parse_time_window_ms(
                    event_blank_text.value,
                    default_name="blank",
                )
                train_gap_ms = float(event_train_gap_float.value)
                if train_gap_ms < 0:
                    raise ValueError("Train gap must be 0 ms or greater.")
                result = analyze_event_locked_power(
                    channel_data=raw_channel_data,
                    stim_uA=stim_uA_for_events,
                    sample_rate_hz=sample_rate_hz,
                    folder=data_folder,
                    bands=bands,
                    baseline_window=baseline_window,
                    post_windows=post_windows,
                    train_gap_ms=train_gap_ms,
                    stim_channel_name=stim_channel_name,
                    blank_window_ms=(blank_window.start_ms, blank_window.end_ms),
                    color_scale_db=color_scale_db,
                )
                mode_label = "stim-triggered"
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
            f"Generated {mode_label} power preview from {len(loaded)} RHS file(s), "
            f"{len(channels)} channel(s), {len(bands)} band(s)."
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
        png_path.parent.mkdir(parents=True, exist_ok=True)
        temp_png = png_path.with_name(f".{png_path.stem}.tmp{png_path.suffix}")
        temp_csv = csv_path.with_name(f".{csv_path.stem}.tmp{csv_path.suffix}")
        temp_png.write_bytes(preview_png)
        temp_csv.write_text(str(csv_text))
        os.replace(temp_png, png_path)
        os.replace(temp_csv, csv_path)
        power_target_label.value = f"<b>Saved:</b> {png_path.name} and {csv_path.name}"
        power_status.value = (
            f"Saved power PNG and CSV inside selected data folder: <b>{png_path.parent}</b>"
        )

    power_generate_button.on_click(generate_power_preview)
    power_save_button.on_click(save_power_outputs)


def _read_power_inputs(folder: Path, channels: list[str]):
    raw_channel_data = []
    stim_channel_data = []
    sample_rate_hz = None
    loaded = None
    for channel in channels:
        raw_uV, stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(
            folder, channel
        )
        if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
            raise ValueError("Selected channels have different sample rates.")
        sample_rate_hz = channel_sample_rate_hz
        loaded = channel_loaded
        raw_channel_data.append((channel, raw_uV))
        stim_channel_data.append((channel, raw_uV, stim_uA))

    stim_channel_info = find_stim_channel_in_data(
        stim_channel_data,
        slice(0, stim_channel_data[0][1].size),
    )
    if stim_channel_info is None:
        available_channels = read_rhs_amplifier_channel_names(folder)
        for candidate_channel in available_channels:
            if candidate_channel in channels:
                continue
            candidate_raw_uV, candidate_stim_uA, candidate_sample_rate_hz, _loaded = read_rhs_folder(
                folder, candidate_channel
            )
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

    return raw_channel_data, stim_channel_info, sample_rate_hz, loaded


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
