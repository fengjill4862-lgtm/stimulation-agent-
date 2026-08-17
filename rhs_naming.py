#!/usr/bin/env python3
"""Filename label helpers shared by the plotting and analysis modules.

These were previously 11 private copies spread across five modules, collapsing
to 7 genuinely distinct behaviors. They are gathered here so the differences are
visible in one place instead of being rediscovered by reading four files.

**The variants are not interchangeable.** Different functions currently produce
different filenames for the same value, and swapping one for another renames
existing output files. That reconciliation is deliberately deferred; see
STIM_AGENT_HANDOFF.md "Known Technical Limitations".

Channel-label sanitizers::

    channel_label            "A-014,,A-016" -> "A014__A016"   (no run collapsing)
    channel_label_collapsed  "A-014,,A-016" -> "A014_A016"    (collapses "__")
    safe_token               "A-014"        -> "A-014"        (keeps "-" and ".")

Both `channel_label` variants strip "-" entirely, so Function 5 -- the only user
of `safe_token` -- writes ``A-014`` where every other function writes ``A014``.

Number formatters::

    number_token(-2.5)                 -> "neg2p5"   "-"->neg, "."->p
    signed_number_token(-2.5)          -> "neg2p5"   same result, sign handled first
    signed_number_plain_decimal(-2.5)  -> "neg2.5"   keeps the decimal point
    ms_number(-2.5)                    -> "-2p5"     "."->p only, sign untouched

`signed_number_plain_decimal` is why Function 2 writes ``neg2.5`` where Function
3 writes ``neg2p5``.
"""

from __future__ import annotations

import re

__all__ = [
    "channel_label",
    "band_token",
    "channel_label_collapsed",
    "ms_label",
    "ms_number",
    "number_token",
    "safe_token",
    "signed_number_plain_decimal",
    "signed_number_token",
]


# -----------------------------------------------------------------------------
# Channel-label sanitizers
# -----------------------------------------------------------------------------
def channel_label(channel_text: str) -> str:
    """Sanitize a channel label, leaving repeated separators intact.

    Used by Function 1 (raw wideband).
    """
    safe = channel_text.strip().replace("-", "").replace(" ", "_")
    safe = safe.replace(",", "_").replace(":", "_").replace("/", "_")
    return safe or "channels"


def channel_label_collapsed(text: str) -> str:
    """Sanitize a channel label, collapsing doubled underscores.

    Used by Function 3 (stim-triggered events) and Function 4 (response only).
    Differs from `channel_label` only when the input contains adjacent
    separators.
    """
    safe = text.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
    return safe.replace(",", "_").replace("-", "").replace("__", "_") or "channels"


def safe_token(text: str) -> str:
    """Sanitize an arbitrary token, preserving "-" and "." characters.

    Used by Function 5 (power analysis), which is why its filenames retain the
    channel dash (``A-014``) while other functions strip it (``A014``).
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.strip("_") or "value"


# -----------------------------------------------------------------------------
# Number formatters
# -----------------------------------------------------------------------------
def band_token(text: str) -> str:
    """Sanitize a band label, keeping the range dash but mapping decimals to p.

    ``"delta_0.5-4Hz"`` -> ``"delta_0p5-4Hz"``. The dash separates the two
    frequencies and must survive, unlike the dash in a channel name.
    """
    return safe_token(text).replace(".", "p")


def number_token(value: float) -> str:
    """Format a number for a filename: "-" becomes neg, "." becomes p."""
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def signed_number_token(value: float) -> str:
    """Format a signed number, prefixing "neg" before formatting the magnitude.

    Produces the same output as `number_token` for every finite value; kept
    distinct because the two express different intent at their call sites.
    """
    if value < 0:
        return f"neg{number_token(abs(value))}"
    return number_token(value)


def signed_number_plain_decimal(value: float) -> str:
    """Format a signed number but keep the decimal point.

    Used by Function 2's amplitude labels, hence ``neg2.5`` there versus
    ``neg2p5`` elsewhere.
    """
    if value < 0:
        return f"neg{abs(value):g}"
    return f"{value:g}"


def ms_number(value: float) -> str:
    """Format a millisecond value: "." becomes p, the sign is left alone."""
    return f"{value:g}".replace(".", "p")


def ms_label(value: float) -> str:
    """Format a millisecond value with its unit, e.g. ``100ms``."""
    return f"{ms_number(value)}ms"
