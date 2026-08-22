#!/usr/bin/env python3
"""Headless CLI for the Function 7 evoked-response sweep.

Same pipeline as the notebook tab, no widgets. Adds no logic of its own:
argument strings go to the same ``config_from_text_fields`` the UI uses, so a
command line and a tab produce identical files.

    /usr/local/bin/python3 run_evoked_sweep.py "<session folder>"
    /usr/local/bin/python3 run_evoked_sweep.py "<run folder>" --single
    /usr/local/bin/python3 run_evoked_sweep.py "<session folder>" --save
"""

from __future__ import annotations

import argparse
import sys

from evoked_sweep.config import config_from_text_fields
from evoked_sweep.pipeline import run_session, run_single, write_outputs
from evoked_sweep.summary import verdict_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="session folder, or one run folder with --single")
    parser.add_argument("--single", action="store_true", help="analyse one run folder only")
    parser.add_argument("--channels", default="all", help="'all' or a list like 'B-017 B-018'")
    parser.add_argument("--pulses", default="50", help="expected pulses per train")
    parser.add_argument("--width-ms", default="5", help="expected pulse width in ms")
    parser.add_argument("--max-runs", default="", help="limit the number of runs analysed")
    parser.add_argument("--response-window-ms", default="", help="post-onset window, default 50")
    parser.add_argument("--max-period-s", default="", help="comb-search ceiling, default 1.5")
    parser.add_argument("--pitch-um", default="", help="contact pitch, default 500")
    parser.add_argument("--contact-order", default="", help="optional physical channel order")
    parser.add_argument(
        "--wiring-include", default="", help="comma-separated substrings; empty = every wiring"
    )
    parser.add_argument(
        "--wiring-exclude", default="", help="comma-separated substrings, e.g. artifact"
    )
    parser.add_argument(
        "--scope-dir", default="", help="oscilloscope capture folder; empty = auto <session>/oscilloscope"
    )
    parser.add_argument(
        "--wiring-label", default="", help="wiring name for flat sessions; empty = session folder name"
    )
    parser.add_argument("--save", action="store_true", help="write outputs (default: print only)")
    args = parser.parse_args(argv)

    try:
        config = config_from_text_fields(
            session_folder=args.folder,
            channels=args.channels,
            expected_pulses=args.pulses,
            pulse_width_ms=args.width_ms,
            single_run="1" if args.single else "",
            max_runs=args.max_runs,
            response_window_ms=args.response_window_ms,
            max_period_s=args.max_period_s,
            contact_pitch_um=args.pitch_um,
            contact_order=args.contact_order,
            wiring_include=args.wiring_include,
            wiring_exclude=args.wiring_exclude,
            scope_dir=args.scope_dir,
            wiring_label=args.wiring_label,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runner = run_single if args.single else run_session
    result = runner(config, progress=lambda message: print(f"  {message}", file=sys.stderr))

    print(verdict_text(result))

    if args.save:
        paths = write_outputs(result)
        print(f"Wrote {len(paths)} file(s) into {config.output_dir}")
    else:
        print("Nothing was written. Pass --save to write the output bundle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
