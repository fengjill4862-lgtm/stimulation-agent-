#!/usr/bin/env python3
"""Atomic file writing shared by the UI layer and the headless batch runner.

Standard library only, deliberately: `wideband_ui_common` imports ipywidgets and
IPython, which the batch runner must not require.

Every generated output goes through here so that an interrupted or failed run
never leaves a half-written PNG or CSV in a data folder. Each write goes to a
hidden temp file beside the target and is then moved into place with
`os.replace`, which is atomic within a filesystem. Temp files are removed even
when the write fails.
"""

from __future__ import annotations

import os
from pathlib import Path


def _temp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _write_payload(temp_path: Path, payload: bytes | str) -> None:
    if isinstance(payload, bytes):
        temp_path.write_bytes(payload)
    else:
        temp_path.write_text(payload)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to `path` atomically."""
    atomic_write_all([(path, data)])


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` atomically."""
    atomic_write_all([(path, text)])


def atomic_write_all(items: list[tuple[Path, bytes | str]]) -> None:
    """Write several files, staging all of them before moving any into place.

    Used for outputs that belong together, such as Function 5's PNG and CSV
    pair: if generating or writing either one fails, neither target is touched,
    so a folder cannot end up with a new PNG beside a stale CSV.

    The final moves are separate `os.replace` calls, so this is not a true
    multi-file transaction -- a crash between them is still possible. It removes
    the far larger window during which the payloads were being generated and
    written.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in items:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = _temp_path_for(path)
            _write_payload(temp_path, payload)
            staged.append((temp_path, path))
        for temp_path, path in staged:
            os.replace(temp_path, path)
        staged.clear()
    finally:
        for temp_path, _path in staged:
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_write_figure(fig, path: Path, dpi: int = 220) -> None:
    """Render a matplotlib figure to PNG and write it atomically.

    The figure is rendered to memory first so that a rendering failure cannot
    leave a truncated file at `path`.
    """
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    atomic_write_bytes(path, buffer.getvalue())


__all__ = [
    "atomic_write_all",
    "atomic_write_bytes",
    "atomic_write_figure",
    "atomic_write_text",
]
