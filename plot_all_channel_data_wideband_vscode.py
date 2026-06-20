# %% [markdown]
# ## 4. Plot All Channel Data Wideband
#
# Run the code cell below in VS Code's Python Interactive/Jupyter interface.
# The wideband plot is generated and displayed inline. It is not saved until
# you click the embedded **Save PNG** button.

# %%
from __future__ import annotations
import os
import tempfile
from io import BytesIO
from pathlib import Path

# Keep this cell embedded in VS Code/Jupyter; do not open a Matplotlib pop-up.
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
from IPython.display import Markdown, display

from plot_rhs_raw_wideband_with_stim_legend import (
    default_output_path,
    find_pulse_segments,
    plot_raw_with_stim_pulse,
    read_rhs_folder,
)


DATA_FOLDER = Path(
    "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
    "20260617 stimulation Pt/"
    "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
)
CHANNEL = "A-014"
MAX_POINTS = 600_000
PULSE_NUMBER = 1

OUTPUT_PATH = default_output_path(DATA_FOLDER, CHANNEL)

display(Markdown("### plot all channel data wideband"))

status = widgets.HTML(value="Reading RHS files and generating inline preview...")
display(status)

raw_uV, stim_uA, sample_rate_hz, loaded = read_rhs_folder(DATA_FOLDER, CHANNEL)
fig = plot_raw_with_stim_pulse(
    raw_uV=raw_uV,
    stim_uA=stim_uA,
    sample_rate_hz=sample_rate_hz,
    channel_name=CHANNEL,
    folder=DATA_FOLDER,
    output_path=OUTPUT_PATH,
    max_points=MAX_POINTS,
    pulse_number=PULSE_NUMBER,
    add_save_button=False,
)

preview_buffer = BytesIO()
fig.savefig(preview_buffer, format="png", dpi=220)
plt.close(fig)
preview_png = preview_buffer.getvalue()

save_button = widgets.Button(
    description="Save PNG",
    icon="save",
    button_style="success",
    layout=widgets.Layout(width="110px"),
)
target_label = widgets.HTML(value=f"<b>Target:</b> {OUTPUT_PATH}")
preview_image = widgets.Image(
    value=preview_png,
    format="png",
    layout=widgets.Layout(width="100%"),
)


def save_png(_button) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.stem}.tmp{OUTPUT_PATH.suffix}")
    temp_path.write_bytes(preview_png)
    os.replace(temp_path, OUTPUT_PATH)
    target_label.value = f"<b>Saved:</b> {OUTPUT_PATH}"


save_button.on_click(save_png)

total_timestamp_gaps = sum(item.timestamp_gaps for item in loaded)
status.value = (
    f"Loaded {len(loaded)} RHS file(s), {raw_uV.size} samples "
    f"({raw_uV.size / sample_rate_hz:.3f} s), "
    f"{len(find_pulse_segments(stim_uA))} stim pulses, "
    f"{total_timestamp_gaps} timestamp gaps."
)

display(widgets.VBox([widgets.HBox([save_button, target_label]), preview_image]))

# %%
