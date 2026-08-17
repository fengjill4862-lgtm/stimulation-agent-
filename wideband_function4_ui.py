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
# 2) Import RHS reader and response-only plotting helpers.
# -----------------------------------------------------------------------------
from plot_rhs_raw_wideband_with_stim_legend import (
    channel_selection_label,
    parse_channel_selection,
    resolve_channel_selection,
    parse_time_window,
    read_rhs_folder,
    time_window_label,
)
from plot_rhs_filtered_wideband import (
    format_bandpass_filename_label,
    format_bandpass_status,
    parse_amplitude_range,
    parse_frequency_range,
    plot_filtered_channels,
)


# -----------------------------------------------------------------------------
# 3) Choose the default data folder and response-only plot settings.
# -----------------------------------------------------------------------------
try:
    RESPONSE_DEFAULT_DATA_DIR = EVENT_DEFAULT_DATA_DIR
except NameError:
    try:
        RESPONSE_DEFAULT_DATA_DIR = FILTER_DEFAULT_DATA_DIR
    except NameError:
        RESPONSE_DEFAULT_DATA_DIR = Path(
            "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
            "20260617 stimulation Pt/"
            "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
        )


def response_folder_path_from_text(value: str) -> Path:
    """Clean a pasted folder path from the widget and return an absolute Path."""
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


def response_output_path(
    folder: Path,
    channel_label: str,
    band_hz: tuple[float, float] | None,
    amplitude_uV: tuple[float, float] | None,
    time_window: tuple[float, float] | None,
) -> Path:
    """Name the response-only PNG without using stimulation-current metadata."""
    safe_channel = _response_safe_label(channel_label)
    band_label = format_bandpass_filename_label(band_hz).replace(".", "p")
    if amplitude_uV is None:
        amp_label = "allAmp"
    else:
        amp_label = (
            f"{_response_signed_number(amplitude_uV[0])}-to-"
            f"{_response_signed_number(amplitude_uV[1])}uV"
        )
    return folder / f"recorded_response_{safe_channel}_{band_label}_{amp_label}_{time_window_label(time_window)}.png"


def _response_safe_label(text: str) -> str:
    safe = text.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
    return safe.replace(",", "_").replace("-", "").replace("__", "_") or "channels"


def _response_number(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _response_signed_number(value: float) -> str:
    if value < 0:
        return f"neg{_response_number(abs(value))}"
    return _response_number(value)


# -----------------------------------------------------------------------------
# 4) Build the response-only controls.
# -----------------------------------------------------------------------------
response_folder_text = widgets.Textarea(
    value=str(RESPONSE_DEFAULT_DATA_DIR),
    description="RHS folder",
    placeholder="Paste RHS data folder path here",
    continuous_update=False,
    layout=widgets.Layout(width="100%", height="46px"),
    style={"description_width": "90px"},
)
response_channel_text = widgets.Text(
    value="A-014",
    description="Channels",
    placeholder="all, A-014, or A-014-16",
    layout=widgets.Layout(width="240px"),
    style={"description_width": "75px"},
)
response_bandpass_text = widgets.Text(
    value="all",
    description="Bandpass",
    placeholder="all or 0.1-150 Hz",
    layout=widgets.Layout(width="230px"),
    style={"description_width": "80px"},
)
response_amplitude_text = widgets.Text(
    value="-500 - 500 uV",
    description="Amplitude",
    layout=widgets.Layout(width="230px"),
    style={"description_width": "85px"},
)
response_time_text = widgets.Text(
    value="all",
    description="Time (s)",
    placeholder="all or 0-60",
    layout=widgets.Layout(width="190px"),
    style={"description_width": "70px"},
)
response_max_points_int = widgets.IntText(
    value=600_000,
    description="Max points",
    layout=widgets.Layout(width="210px"),
    style={"description_width": "90px"},
)
response_generate_button = widgets.Button(
    description="Generate Response Preview",
    icon="play",
    button_style="primary",
    layout=widgets.Layout(width="210px"),
)
response_save_button = widgets.Button(
    description="Save PNG",
    icon="save",
    button_style="success",
    disabled=True,
    layout=widgets.Layout(width="120px"),
)
response_status = widgets.HTML(value="Choose response settings, then click <b>Generate Response Preview</b>.")
response_target_label = widgets.HTML(value="<b>Target:</b> no preview generated yet")
response_preview_output = widgets.Output()
response_current_preview = {"png": None, "output_path": None}

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


# -----------------------------------------------------------------------------
# 5) Generate and save a response-only full-recording preview.
# -----------------------------------------------------------------------------
def generate_response_preview(_button=None) -> None:
    """Read selected response channels and show a filtered full-session preview."""
    response_save_button.disabled = True
    response_current_preview["png"] = None
    response_current_preview["output_path"] = None
    response_target_label.value = "<b>Target:</b> no preview generated yet"
    response_preview_output.clear_output()

    data_folder = response_folder_path_from_text(response_folder_text.value)
    max_points = int(response_max_points_int.value)

    try:
        channels = resolve_channel_selection(response_channel_text.value, data_folder)
        band_hz = parse_frequency_range(response_bandpass_text.value)
        amplitude_uV = parse_amplitude_range(response_amplitude_text.value)
        time_window = parse_time_window(response_time_text.value)
    except ValueError as exc:
        response_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    if not data_folder.exists():
        response_status.value = f"<b style='color:#b00020'>Folder not found:</b> {data_folder}"
        return
    if not list(data_folder.glob("*.rhs")):
        response_status.value = f"<b style='color:#b00020'>No .rhs files found in:</b> {data_folder}"
        return

    response_status.value = f"Reading RHS response data from <b>{data_folder}</b>..."
    raw_channel_data = []
    sample_rate_hz = None
    loaded = None
    try:
        for channel in channels:
            raw_uV, _stim_uA, channel_sample_rate_hz, channel_loaded = read_rhs_folder(data_folder, channel)
            if sample_rate_hz is not None and channel_sample_rate_hz != sample_rate_hz:
                raise ValueError("Selected channels have different sample rates.")
            sample_rate_hz = channel_sample_rate_hz
            loaded = channel_loaded
            raw_channel_data.append((channel, raw_uV))
    except (FileNotFoundError, ValueError) as exc:
        response_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    channel_label = channel_selection_label(channels)
    output_path = response_output_path(data_folder, channel_label, band_hz, amplitude_uV, time_window)

    try:
        fig, summaries = plot_filtered_channels(
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
        response_status.value = f"<b style='color:#b00020'>{exc}</b>"
        return

    preview_buffer = BytesIO()
    fig.savefig(preview_buffer, format="png", dpi=220)
    plt.close(fig)
    preview_png = preview_buffer.getvalue()

    response_current_preview["png"] = preview_png
    response_current_preview["output_path"] = output_path
    response_save_button.disabled = False
    response_target_label.value = f"<b>Target:</b> {output_path}"

    with response_preview_output:
        display(widgets.HTML(value=f"<b>Response-only preview</b>; target {output_path.name}"))
        display(widgets.Image(value=preview_png, format="png", layout=widgets.Layout(width="100%")))

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
        response_status.value = "<b style='color:#b00020'>Generate a response preview before saving.</b>"
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temp_path.write_bytes(preview_png)
    os.replace(temp_path, output_path)

    response_target_label.value = f"<b>Saved:</b> {output_path}"
    response_status.value = f"Saved response-only PNG inside selected data folder: <b>{output_path.parent}</b>"


response_generate_button.on_click(generate_response_preview)
response_save_button.on_click(save_response_png)
