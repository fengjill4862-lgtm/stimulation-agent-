from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) Configure notebook output so plots are embedded, not pop-up windows.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2) Import the raw wideband helper functions.
# -----------------------------------------------------------------------------
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    default_output_path,
    find_pulse_segments,
    find_stim_channel_in_data,
    parse_channel_selection,
    resolve_channel_selection,
    parse_time_window,
    plot_raw_channels_with_stim_pulse,
    plot_raw_with_stim_pulse,
    read_rhs_folder,
    sample_slice_for_time_window,
    time_window_label,
)


# -----------------------------------------------------------------------------
# 3) Choose the default data folder and raw plot settings.
# -----------------------------------------------------------------------------
# This path only seeds the UI text box. You can paste a different folder path
# into the RHS folder field before clicking Generate Preview.
DEFAULT_DATA_DIR = Path(
    "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
    "20260617 stimulation Pt/"
    "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
)
DEFAULT_CHANNELS = "A-014"
DEFAULT_TIME_WINDOW = "all"
DEFAULT_MAX_POINTS = 600_000
DEFAULT_PULSE_NUMBER = 1


def folder_path_from_text(value: str) -> Path:
    """Clean a pasted folder path from the widget and return an absolute Path."""
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


# -----------------------------------------------------------------------------
# 4) Build the raw wideband controls.
# -----------------------------------------------------------------------------
raw_folder_text = widgets.Textarea(
    value=str(DEFAULT_DATA_DIR),
    description="RHS folder",
    placeholder="Paste RHS data folder path here",
    continuous_update=False,
    layout=widgets.Layout(width="100%", height="46px"),
    style={"description_width": "90px"},
)
raw_channel_text = widgets.Text(
    value=DEFAULT_CHANNELS,
    description="Channels",
    placeholder="all, A-014, or A-014-16",
    layout=widgets.Layout(width="240px"),
    style={"description_width": "75px"},
)
raw_time_text = widgets.Text(
    value=DEFAULT_TIME_WINDOW,
    description="Time (s)",
    placeholder="all or 0-10",
    layout=widgets.Layout(width="190px"),
    style={"description_width": "70px"},
)
raw_max_points_int = widgets.IntText(
    value=DEFAULT_MAX_POINTS,
    description="Max points",
    layout=widgets.Layout(width="210px"),
    style={"description_width": "90px"},
)
raw_pulse_number_int = widgets.IntText(
    value=DEFAULT_PULSE_NUMBER,
    description="Pulse",
    layout=widgets.Layout(width="150px"),
    style={"description_width": "55px"},
)
raw_generate_button = widgets.Button(
    description="Generate Preview",
    icon="play",
    button_style="primary",
    layout=widgets.Layout(width="160px"),
)
raw_save_button = widgets.Button(
    description="Save PNG",
    icon="save",
    button_style="success",
    disabled=True,
    layout=widgets.Layout(width="110px"),
)
raw_status = widgets.HTML(value="Choose an RHS folder, then click <b>Generate Preview</b>.")
raw_target_label = widgets.HTML(value="<b>Target:</b> no preview generated yet")
raw_preview_output = widgets.Output()
raw_current_preview = {"png": None, "output_path": None}

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


# -----------------------------------------------------------------------------
# 5) Generate and save the raw wideband preview.
# -----------------------------------------------------------------------------
def generate_raw_preview(_button=None) -> None:
    """Read selected RHS folder and show the raw wideband preview inline."""
    raw_save_button.disabled = True
    raw_current_preview["png"] = None
    raw_current_preview["output_path"] = None
    raw_target_label.value = "<b>Target:</b> no preview generated yet"
    raw_preview_output.clear_output()

    data_folder = folder_path_from_text(raw_folder_text.value)
    max_points = int(raw_max_points_int.value)
    pulse_number = int(raw_pulse_number_int.value)
    try:
        channels = resolve_channel_selection(raw_channel_text.value, data_folder)
        time_window = parse_time_window(raw_time_text.value)
    except ValueError as exc:
        raw_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return
    output_label = channel_selection_label(channels)
    if time_window is not None:
        output_label = f"{output_label}_{time_window_label(time_window)}"
    output_path = default_output_path(data_folder, output_label)

    if not data_folder.exists():
        raw_status.value = f"<b style='color:#b00020'>Folder not found:</b> {data_folder}"
        return
    if not list(data_folder.glob("*.rhs")):
        raw_status.value = f"<b style='color:#b00020'>No .rhs files found in:</b> {data_folder}"
        return

    raw_status.value = f"Reading RHS files from <b>{data_folder}</b>..."
    channel_data = []
    sample_rate_hz = None
    loaded = None
    try:
        for channel in channels:
            raw_uV, stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(data_folder, channel)
            if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
                raise ValueError("Selected channels have different sample rates.")
            sample_rate_hz = channel_sample_rate_hz
            loaded = channel_loaded
            channel_data.append((channel, raw_uV, stim_uA))
    except (FileNotFoundError, ValueError) as exc:
        raw_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    try:
        fig = plot_raw_channels_with_stim_pulse(
            channel_data=channel_data,
            sample_rate_hz=sample_rate_hz,
            folder=data_folder,
            output_path=output_path,
            max_points=max_points,
            pulse_number=pulse_number,
            time_window=time_window,
        )
    except ValueError as exc:
        raw_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    preview_buffer = BytesIO()
    fig.savefig(preview_buffer, format="png", dpi=220)
    plt.close(fig)
    preview_png = preview_buffer.getvalue()

    raw_current_preview["png"] = preview_png
    raw_current_preview["output_path"] = output_path
    raw_save_button.disabled = False
    raw_target_label.value = f"<b>Target:</b> {output_path}"

    first_channel_name, first_raw_uV, first_stim_uA = channel_data[0]
    display_slice, display_bounds = sample_slice_for_time_window(first_raw_uV.size, sample_rate_hz, time_window)
    displayed_samples = display_slice.stop - display_slice.start
    time_status = "all time" if time_window is None else f"{display_bounds[0]:g}-{display_bounds[1]:g} s"
    stim_channel_info = find_stim_channel_in_data(channel_data, slice(0, first_raw_uV.size))
    if stim_channel_info is None:
        stim_status = "no nonzero stim_data found in selected channels"
    else:
        stim_channel_name, _display_stim_uA, pulse_segments = stim_channel_info
        stim_status = f"{len(pulse_segments)} stim pulses on {stim_channel_name} *"
    total_timestamp_gaps = sum(item.timestamp_gaps for item in loaded)
    raw_status.value = (
        f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s), "
        f"displaying {displayed_samples} of {first_raw_uV.size} samples/channel "
        f"({time_status}), "
        f"{stim_status}, "
        f"{total_timestamp_gaps} timestamp gaps."
    )

    with raw_preview_output:
        display(widgets.Image(value=preview_png, format="png", layout=widgets.Layout(width="100%")))


def save_raw_png(_button=None) -> None:
    """Save the current raw wideband preview into the selected RHS folder only."""
    preview_png = raw_current_preview.get("png")
    output_path = raw_current_preview.get("output_path")
    if preview_png is None or output_path is None:
        raw_status.value = "<b style='color:#b00020'>Generate a preview before saving.</b>"
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temp_path.write_bytes(preview_png)
    os.replace(temp_path, output_path)
    raw_target_label.value = f"<b>Saved:</b> {output_path}"
    raw_status.value = f"Saved PNG inside selected data folder: <b>{output_path.parent}</b>"


raw_generate_button.on_click(generate_raw_preview)
raw_save_button.on_click(save_raw_png)
