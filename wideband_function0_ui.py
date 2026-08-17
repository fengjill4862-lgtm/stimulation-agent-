from __future__ import annotations

from pathlib import Path
import html

import ipywidgets as widgets
from IPython.display import HTML, display

# -----------------------------------------------------------------------------
# 1) Import the RHS folder rename helper functions.
# -----------------------------------------------------------------------------
from rename_rhs_folders_by_stim_waveform import apply_rename_plan, plan_renames


# -----------------------------------------------------------------------------
# 2) Choose the default parent folder that contains RHS session folders.
# -----------------------------------------------------------------------------
# This path only seeds the UI text box. Paste any parent folder from your computer
# before clicking Preview Rename Plan.
DEFAULT_RENAME_PARENT_DIR = Path(
    "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/"
    "20260624 pt stimulation"
)


def rename_folder_path_from_text(value: str) -> Path:
    """Clean a pasted folder path from the widget and return an absolute Path."""
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return Path(cleaned).expanduser().resolve()


# -----------------------------------------------------------------------------
# 3) Build the rename controls.
# -----------------------------------------------------------------------------
rename_parent_text = widgets.Textarea(
    value=str(DEFAULT_RENAME_PARENT_DIR),
    description="Parent folder",
    placeholder="Paste parent folder containing RHS session folders here",
    continuous_update=False,
    layout=widgets.Layout(width="100%", height="46px"),
    style={"description_width": "105px"},
)
preview_rename_button = widgets.Button(
    description="Preview Rename Plan",
    icon="search",
    button_style="info",
    layout=widgets.Layout(width="190px"),
)
apply_rename_button = widgets.Button(
    description="Apply Renames",
    icon="check",
    button_style="warning",
    disabled=True,
    layout=widgets.Layout(width="160px"),
)
rename_status = widgets.HTML()
rename_output = widgets.Output()
rename_state = {"plans": []}


# -----------------------------------------------------------------------------
# 4) Render the preview/apply results as an embedded table.
# -----------------------------------------------------------------------------
def rename_plan_table(plans) -> HTML:
    """Return an HTML table summarizing the current rename plan."""
    rows = []
    for plan in plans:
        source_name = html.escape(plan.source.name)
        target_name = "" if plan.target is None else html.escape(plan.target.name)
        stim_channel = ""
        suffix = ""
        if plan.summary is not None:
            stim_channel = html.escape(plan.summary.channel_name)
            suffix = html.escape(plan.summary.suffix)
        message = html.escape(plan.error or "")
        color = {
            "rename": "#0b6bcb",
            "renamed": "#1b7f2a",
            "already named": "#555555",
            "skip": "#8a6d00",
            "error": "#b00020",
        }.get(plan.status, "#333333")
        rows.append(
            "<tr>"
            f"<td style='padding:4px 8px;color:{color};font-weight:600'>{html.escape(plan.status)}</td>"
            f"<td style='padding:4px 8px'>{source_name}</td>"
            f"<td style='padding:4px 8px'>{target_name}</td>"
            f"<td style='padding:4px 8px'>{stim_channel}</td>"
            f"<td style='padding:4px 8px'>{suffix}</td>"
            f"<td style='padding:4px 8px;color:#b00020'>{message}</td>"
            "</tr>"
        )
    table = (
        "<table style='border-collapse:collapse;font-size:13px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 8px'>Status</th>"
        "<th style='text-align:left;padding:4px 8px'>Current folder</th>"
        "<th style='text-align:left;padding:4px 8px'>New folder</th>"
        "<th style='text-align:left;padding:4px 8px'>Stim channel</th>"
        "<th style='text-align:left;padding:4px 8px'>Waveform suffix</th>"
        "<th style='text-align:left;padding:4px 8px'>Message</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return HTML(table)


def preview_rename_plan(_button=None) -> None:
    """Scan the selected folder and show proposed names without renaming."""
    apply_rename_button.disabled = True
    with rename_output:
        rename_output.clear_output(wait=True)
        try:
            parent_folder = rename_folder_path_from_text(rename_parent_text.value)
            plans = plan_renames(parent_folder)
        except Exception as exc:
            rename_state["plans"] = []
            rename_status.value = f"<b style='color:#b00020'>Preview failed:</b> {html.escape(str(exc))}"
            return

        rename_state["plans"] = plans
        rename_count = sum(1 for plan in plans if plan.will_rename)
        error_count = sum(1 for plan in plans if plan.status == "error")
        apply_rename_button.disabled = rename_count == 0
        rename_status.value = (
            f"Previewed <b>{len(plans)}</b> RHS folder(s). "
            f"Folders to rename: <b>{rename_count}</b>. Errors: <b>{error_count}</b>."
        )
        display(rename_plan_table(plans))


def apply_renames(_button=None) -> None:
    """Apply the last previewed plan. Preview again after changing the path."""
    plans = rename_state.get("plans", [])
    if not plans:
        preview_rename_plan()
        plans = rename_state.get("plans", [])
    if not plans:
        return

    with rename_output:
        rename_output.clear_output(wait=True)
        applied = apply_rename_plan(plans)
        rename_state["plans"] = applied
        renamed_count = sum(1 for plan in applied if plan.status == "renamed")
        error_count = sum(1 for plan in applied if plan.status == "error")
        apply_rename_button.disabled = True
        rename_status.value = (
            f"Applied rename plan. Renamed: <b>{renamed_count}</b>. "
            f"Errors: <b>{error_count}</b>."
        )
        display(rename_plan_table(applied))


preview_rename_button.on_click(preview_rename_plan)
apply_rename_button.on_click(apply_renames)

# Display the full Function 0 interface inline in VS Code/Jupyter.
display(
    widgets.VBox(
        [
            rename_parent_text,
            widgets.HBox([preview_rename_button, apply_rename_button, rename_status]),
            rename_output,
        ]
    )
)
