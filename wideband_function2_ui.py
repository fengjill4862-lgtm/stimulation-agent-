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
# 2) Import RHS reader and filtered-plot helper functions.
# -----------------------------------------------------------------------------
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    find_stim_channel_in_data,
    parse_channel_selection,
    resolve_channel_selection,
    parse_time_window,
    read_rhs_folder,
    time_window_label,
)
from plot_rhs_filtered_wideband import (
    default_filtered_output_path,
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
    plot_filtered_wideband,
)


# -----------------------------------------------------------------------------
# 3) Choose the default data folder and filtered plot settings.
# -----------------------------------------------------------------------------
# If Block 1 has already been run, this reuses its DEFAULT_DATA_DIR. Otherwise
# it falls back to the same Endovascular/Jill example path.
try:
    FILTER_DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
except NameError:
    FILTER_DEFAULT_DATA_DIR = Path(
        "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
        "20260617 stimulation Pt/"
        "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
    )


def folder_path_from_text(value: str) -> Path:
    """Clean a pasted folder path from the widget and return an absolute Path."""
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


# -----------------------------------------------------------------------------
# 4) Build the filtered plot controls.
# -----------------------------------------------------------------------------
filter_folder_text = widgets.Textarea(
    value=str(FILTER_DEFAULT_DATA_DIR),
    description="RHS folder",
    placeholder="Paste RHS data folder path here",
    continuous_update=False,
    layout=widgets.Layout(width="100%", height="46px"),
    style={"description_width": "90px"},
)
filter_channel_text = widgets.Text(
    value="A-014",
    description="Channels",
    placeholder="all, A-014, or A-014-16",
    layout=widgets.Layout(width="240px"),
    style={"description_width": "75px"},
)
bandpass_text = widgets.Text(
    value="all",
    description="Bandpass",
    placeholder="all or 200-400 Hz",
    layout=widgets.Layout(width="230px"),
    style={"description_width": "80px"},
)
amplitude_text = widgets.Text(
    value="-100 - 100 uV",
    description="Amplitude",
    layout=widgets.Layout(width="230px"),
    style={"description_width": "85px"},
)
filter_time_text = widgets.Text(
    value="all",
    description="Time (s)",
    placeholder="all or 10-20",
    layout=widgets.Layout(width="190px"),
    style={"description_width": "70px"},
)
filter_max_points_int = widgets.IntText(
    value=600_000,
    description="Max points",
    layout=widgets.Layout(width="210px"),
    style={"description_width": "90px"},
)
filter_generate_button = widgets.Button(
    description="Generate Preview",
    icon="play",
    button_style="primary",
    layout=widgets.Layout(width="160px"),
)
filter_save_button = widgets.Button(
    description="Save PNG",
    icon="save",
    button_style="success",
    disabled=True,
    layout=widgets.Layout(width="110px"),
)
filter_status = widgets.HTML(value="Choose filter settings, then click <b>Generate Preview</b>.")
filter_target_label = widgets.HTML(value="<b>Target:</b> no preview generated yet")
filter_preview_output = widgets.Output()
filter_current_preview = {"png": None, "output_path": None}

display(
    widgets.VBox(
        [
            filter_folder_text,
            widgets.HBox([filter_channel_text, bandpass_text, amplitude_text]),
            widgets.HBox([filter_time_text, filter_max_points_int]),
            widgets.HBox([filter_generate_button, filter_save_button, filter_target_label]),
            filter_status,
            filter_preview_output,
        ]
    )
)


# -----------------------------------------------------------------------------
# 5) Generate and save the bandpass filtered preview.
# -----------------------------------------------------------------------------
def generate_filtered_preview(_button=None) -> None:
    """Read selected RHS folder, apply filters, and show inline preview."""
    filter_save_button.disabled = True
    filter_current_preview["png"] = None
    filter_current_preview["output_path"] = None
    filter_target_label.value = "<b>Target:</b> no preview generated yet"
    filter_preview_output.clear_output()

    data_folder = folder_path_from_text(filter_folder_text.value)
    max_points = int(filter_max_points_int.value)

    try:
        channels = resolve_channel_selection(filter_channel_text.value, data_folder)
        band_hz = parse_frequency_range(bandpass_text.value)
        amplitude_uV = parse_amplitude_range(amplitude_text.value)
        time_window = parse_time_window(filter_time_text.value)
    except ValueError as exc:
        filter_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    output_label = channel_selection_label(channels)
    if time_window is not None:
        output_label = f"{output_label}_{time_window_label(time_window)}"
    output_path = default_filtered_output_path(data_folder, output_label, band_hz, amplitude_uV)

    if not data_folder.exists():
        filter_status.value = f"<b style='color:#b00020'>Folder not found:</b> {data_folder}"
        return
    if not list(data_folder.glob("*.rhs")):
        filter_status.value = f"<b style='color:#b00020'>No .rhs files found in:</b> {data_folder}"
        return

    filter_status.value = f"Reading RHS files and {format_bandpass_status(band_hz)}..."
    channel_data = []
    stim_channel_data = []
    sample_rate_hz = None
    loaded = None
    try:
        for channel in channels:
            raw_uV, stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(data_folder, channel)
            if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
                raise ValueError("Selected channels have different sample rates.")
            sample_rate_hz = channel_sample_rate_hz
            loaded = channel_loaded
            channel_data.append((channel, raw_uV))
            stim_channel_data.append((channel, raw_uV, stim_uA))
    except (FileNotFoundError, ValueError) as exc:
        filter_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    stim_channel_info = find_stim_channel_in_data(stim_channel_data, slice(0, stim_channel_data[0][1].size))
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
        filter_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    preview_buffer = BytesIO()
    fig.savefig(preview_buffer, format="png", dpi=220)
    plt.close(fig)
    preview_png = preview_buffer.getvalue()

    filter_current_preview["png"] = preview_png
    filter_current_preview["output_path"] = output_path
    filter_save_button.disabled = False
    filter_target_label.value = f"<b>Target:</b> {output_path}"

    rms_text = ", ".join(f"{item['channel_name']}: {item['filtered_rms_uV']:.2f} uV" for item in summaries)
    samples_selected = sum(int(item['samples_in_amplitude_range']) for item in summaries)
    samples_total = sum(int(item['samples_total']) for item in summaries)
    percent_selected = samples_selected / samples_total * 100.0 if samples_total else 0.0
    time_status = "all time" if time_window is None else f"{time_window[0]:g}-{time_window[1]:g} s"
    stim_status = "no nonzero stim_data found in selected channels" if stim_channel_name is None else f"stim channel: {stim_channel_name} *"
    filter_status.value = (
        f"Loaded {len(loaded)} RHS file(s) for {len(channels)} channel(s). "
        f"Time window: {time_status}. "
        f"{stim_status}. "
        f"Signal RMS: {rms_text}. "
        f"Amplitude-selected samples: {samples_selected} "
        f"({percent_selected:.3f}%)."
    )

    with filter_preview_output:
        display(widgets.Image(value=preview_png, format="png", layout=widgets.Layout(width="100%")))


def save_filtered_png(_button=None) -> None:
    """Save the current filtered preview into the selected RHS folder only."""
    preview_png = filter_current_preview.get("png")
    output_path = filter_current_preview.get("output_path")
    if preview_png is None or output_path is None:
        filter_status.value = "<b style='color:#b00020'>Generate a preview before saving.</b>"
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temp_path.write_bytes(preview_png)
    os.replace(temp_path, output_path)
    filter_target_label.value = f"<b>Saved:</b> {output_path}"
    filter_status.value = f"Saved PNG inside selected data folder: <b>{output_path.parent}</b>"


filter_generate_button.on_click(generate_filtered_preview)
filter_save_button.on_click(save_filtered_png)
