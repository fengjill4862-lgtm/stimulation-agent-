#!/usr/bin/env python3
"""Shared widget and save boilerplate for the Function 0-5 notebook UI modules.

Every helper here replaced 4-7 copy-pasted duplicates that were previously
spread across the `_BLOCK_*_SOURCE` strings in `wideband_main_ui.py` and the two
already-extracted UI modules.

Import this module *before* matplotlib.pyplot anywhere in the UI layer. It
configures a writable matplotlib cache and forces the non-interactive Agg
backend at import time, then re-exports `plt`, `widgets` and `display` so that
importing them in the right order is not something a caller can get wrong.
"""

from __future__ import annotations

import html
import os
import tempfile
from collections.abc import MutableMapping
from io import BytesIO
from pathlib import Path


def configure_matplotlib_cache() -> None:
    """Point matplotlib at a writable cache dir and force the Agg backend.

    The default matplotlib config directory may not be writable, and the UI
    renders PNG bytes into `ipywidgets.Image` rather than opening windows. This
    must run before `matplotlib.pyplot` is first imported.
    """
    cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
    mpl_cache = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ["MPLBACKEND"] = "Agg"


configure_matplotlib_cache()

import ipywidgets as widgets  # noqa: E402  (must follow configure_matplotlib_cache)
import matplotlib.pyplot as plt  # noqa: E402
from IPython.display import display  # noqa: E402

# The example path that seeds most folder boxes. It is only a text-box seed;
# any folder can be pasted in.
EXAMPLE_RHS_SESSION_DIR = Path(
    "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
    "20260617 stimulation Pt/"
    "1_260617_180931 anodic-lead 100us 50uA 5 pulses 1000Hz"
)

ERROR_COLOR = "#b00020"
PREVIEW_DPI = 220
NO_PREVIEW_TARGET = "<b>Target:</b> no preview generated yet"


# -----------------------------------------------------------------------------
# Status and error text
# -----------------------------------------------------------------------------
def error_html(message: object) -> str:
    """Red bold error markup. The message is HTML-escaped."""
    return f"<b style='color:{ERROR_COLOR}'>{html.escape(str(message))}</b>"


def error_html_with_detail(label: str, detail: object) -> str:
    """Red bold label followed by plain detail, e.g. 'Folder not found: /path'."""
    return f"<b style='color:{ERROR_COLOR}'>{html.escape(label)}</b> {html.escape(str(detail))}"


# -----------------------------------------------------------------------------
# Folder input
# -----------------------------------------------------------------------------
def folder_path_from_text(value: str) -> Path:
    """Clean a pasted folder path from a widget and return an absolute Path.

    Handles surrounding quotes and stray newlines, which are common when a path
    is dragged or copied from Finder.
    """
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


def rhs_folder_error(folder: Path) -> str | None:
    """Return error markup if `folder` holds no .rhs or .rhd files, else None."""
    if not folder.exists():
        return error_html_with_detail("Folder not found:", folder)
    if not list(folder.glob("*.rhs")) and not list(folder.glob("*.rhd")):
        return error_html_with_detail("No .rhs or .rhd files found in:", folder)
    return None


def default_data_dir(
    namespace: MutableMapping[str, object] | None,
    keys: tuple[str, ...],
    fallback: Path,
    publish_as: str | None = None,
) -> Path:
    """Resolve a seed folder from the notebook namespace, then republish it.

    The exec-based blocks leaked their module-level names into the notebook
    globals, and later blocks read those names to seed their own folder box.
    Real modules do not leak names, so that chain is reproduced explicitly here:
    `keys` are read in priority order, and `publish_as` writes the result back so
    functions run later still find it.
    """
    resolved: Path | None = None
    for key in keys:
        if namespace is not None and key in namespace:
            resolved = Path(str(namespace[key]))
            break
    if resolved is None:
        resolved = fallback
    if publish_as is not None and namespace is not None:
        namespace[publish_as] = resolved
    return resolved


# -----------------------------------------------------------------------------
# Preview rendering
# -----------------------------------------------------------------------------
def figure_to_png_bytes(fig, dpi: int = PREVIEW_DPI) -> bytes:
    """Render a figure to PNG bytes and close it. Nothing is written to disk."""
    preview_buffer = BytesIO()
    fig.savefig(preview_buffer, format="png", dpi=dpi)
    plt.close(fig)
    return preview_buffer.getvalue()


def show_preview_image(output: "widgets.Output", png: bytes, caption: str | None = None) -> None:
    """Display PNG bytes inline inside an Output widget, with optional caption."""
    with output:
        if caption is not None:
            display(widgets.HTML(value=caption))
        display(widgets.Image(value=png, format="png", layout=widgets.Layout(width="100%")))


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
# Re-exported so UI modules have one import site; the implementation is in
# rhs_files, which the headless batch runner also uses.
from rhs_files import atomic_write_all, atomic_write_bytes, atomic_write_text  # noqa: E402,F401


# -----------------------------------------------------------------------------
# Widget factories: the layout constants live here once instead of per block
# -----------------------------------------------------------------------------
def folder_textarea(value: str, description: str = "RHS folder", description_width: str = "90px"):
    return widgets.Textarea(
        value=value,
        description=description,
        placeholder="Paste RHS data folder path here",
        continuous_update=False,
        layout=widgets.Layout(width="100%", height="46px"),
        style={"description_width": description_width},
    )


def channels_text(value: str = "A-014", placeholder: str = "all, A-014, or A-014-16"):
    return widgets.Text(
        value=value,
        description="Channels",
        placeholder=placeholder,
        layout=widgets.Layout(width="240px"),
        style={"description_width": "75px"},
    )


def bandpass_text(value: str = "all", placeholder: str = "all or 200-400 Hz"):
    return widgets.Text(
        value=value,
        description="Bandpass",
        placeholder=placeholder,
        layout=widgets.Layout(width="230px"),
        style={"description_width": "80px"},
    )


def amplitude_text(value: str = "-100 - 100 uV"):
    return widgets.Text(
        value=value,
        description="Amplitude",
        layout=widgets.Layout(width="230px"),
        style={"description_width": "85px"},
    )


def time_text(value: str = "all", placeholder: str = "all or 0-10"):
    return widgets.Text(
        value=value,
        description="Time (s)",
        placeholder=placeholder,
        layout=widgets.Layout(width="190px"),
        style={"description_width": "70px"},
    )


def max_points_int(value: int = 600_000):
    return widgets.IntText(
        value=value,
        description="Max points",
        layout=widgets.Layout(width="210px"),
        style={"description_width": "90px"},
    )


def generate_button(description: str = "Generate Preview", width: str = "160px"):
    return widgets.Button(
        description=description,
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width=width),
    )


def save_button(description: str = "Save PNG", width: str = "110px"):
    return widgets.Button(
        description=description,
        icon="save",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width=width),
    )


def status_html(value: str = ""):
    return widgets.HTML(value=value)


def target_label_html():
    return widgets.HTML(value=NO_PREVIEW_TARGET)


__all__ = [
    "EXAMPLE_RHS_SESSION_DIR",
    "ERROR_COLOR",
    "NO_PREVIEW_TARGET",
    "PREVIEW_DPI",
    "amplitude_text",
    "atomic_write_all",
    "atomic_write_bytes",
    "atomic_write_text",
    "bandpass_text",
    "channels_text",
    "configure_matplotlib_cache",
    "default_data_dir",
    "display",
    "error_html",
    "error_html_with_detail",
    "figure_to_png_bytes",
    "folder_path_from_text",
    "folder_textarea",
    "generate_button",
    "max_points_int",
    "plt",
    "rhs_folder_error",
    "save_button",
    "show_preview_image",
    "status_html",
    "target_label_html",
    "time_text",
    "widgets",
]
