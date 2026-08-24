#!/usr/local/bin/python3
"""Claude Code hooks for the Stimulation agent repo.

One file, four subcommands, wired up in .claude/settings.json:

    write-guard    PreToolUse  Write|Edit|NotebookEdit -> ask before writing into a data folder
    bash-guard     PreToolUse  Bash                    -> deny the wrong Python; ask before
                                                          real-data runs and data-folder writes
    post-checks    PostToolUse Write|Edit|NotebookEdit -> py_compile + reload-chain + notebook JSON
    session-check  SessionStart                        -> flag broken worktree links / stale backup job

Each subcommand reads the hook event as JSON on stdin and prints a JSON decision
on stdout (nothing at all when there is nothing to say). Exit status is always 0;
the decision travels in the JSON.

These encode the working agreements in CLAUDE.md. They are guard rails, not locks:
every blocking decision is "ask", so JF stays the one who says yes.
"""

import json
import pathlib
import re
import shlex
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PYTHON = "/usr/local/bin/python3"

# Recordings and generated outputs live on Synology, never in the repo.
DATA_ROOTS = (
    pathlib.Path("/Users/jf/SynologyDrive/Research/Stimulation"),
    pathlib.Path("/Users/jf/SynologyDrive/Research/Chronic"),
    pathlib.Path("/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular"),
)

# Entry points that read real recordings and write analysis outputs.
# value = flag that makes the run harmless, or None when the tool has no such flag.
REAL_DATA_ENTRY_POINTS = {
    r"run_stim_analysis\.py": "--dry-run",
    r"-m\s+bw_sweep\.run": "--dry-run",
    r"-m\s+filter_diag\.run_all": "--dry-run",
    r"run_evoked_sweep\.py": None,
    r"batch_run_wideband_main_ui\.py": None,
    r"rename_rhs_folders_by_stim_waveform\.py": None,
}

# Shell constructs that modify what they touch (reads such as cat/ls/grep are fine).
MUTATING = re.compile(
    r"(^|[\s;&|(])(rm|mv|cp|mkdir|rmdir|touch|tee|truncate|chmod|chown|ln|rsync|unzip|"
    r"sed\s+-i)\b"          # a mutating command word
    r"|[\s/]python[0-9.]*\s+-c"  # inline Python that could write
    # a redirect whose target is the data folder itself (not 2>/dev/null)
    r"|>>?\s*[\"']?/Users/jf/(SynologyDrive|Library/CloudStorage)"
)

SEGMENT_SPLIT = re.compile(r"(?:\|\||&&|[;\n|&])")
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# A bare name resolves through PATH to Anaconda 3.9.7, which CLAUDE.md calls wrong
# for every project here. An explicit path (.venv/bin/python for Chronic, or PYTHON)
# is a deliberate choice and is left alone.
BARE_PYTHON = re.compile(r"^python[0-9.]*$")
ANACONDA = "/Users/jf/opt/anaconda3"


def event() -> dict:
    """The hook payload on stdin ({} if it is missing or malformed)."""
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}


def emit(payload: dict) -> None:
    print(json.dumps(payload))
    sys.exit(0)


def silent() -> None:
    """Nothing to say: the tool call proceeds under the normal permission flow."""
    sys.exit(0)


def pre_decision(decision: str, reason: str) -> None:
    emit(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        }
    )


def post_block(reason: str) -> None:
    emit({"decision": "block", "reason": reason})


def target_path(tool_input: dict) -> pathlib.Path | None:
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw:
        return None
    return pathlib.Path(raw).expanduser()


def in_data_root(path: pathlib.Path) -> pathlib.Path | None:
    resolved = path if path.is_absolute() else (REPO / path)
    for root in DATA_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return root
    return None


# --------------------------------------------------------------------------- #
# PreToolUse: Write | Edit | NotebookEdit
# --------------------------------------------------------------------------- #
def write_guard() -> None:
    data = event()
    path = target_path(data.get("tool_input") or {})
    if path is None:
        silent()
    root = in_data_root(path)
    if root is None:
        silent()
    pre_decision(
        "ask",
        f"{path} is inside the data folder {root}.\n"
        "CLAUDE.md: do not write into data folders without asking. Analysis outputs "
        "are written by the packages themselves (render_outputs -> write_outputs), "
        "not edited by hand. Confirm only if JF asked for this file to change.",
    )


# --------------------------------------------------------------------------- #
# PreToolUse: Bash
# --------------------------------------------------------------------------- #
def first_tokens(command: str) -> list[str]:
    """The command-position token of every segment of a shell command line."""
    tokens = []
    for segment in SEGMENT_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:  # unbalanced quotes: not worth guessing
            continue
        while words and ENV_ASSIGN.match(words[0]):
            words.pop(0)
        if words:
            tokens.append(words[0])
    return tokens


def bash_guard() -> None:
    data = event()
    command = (data.get("tool_input") or {}).get("command") or ""
    if not command.strip():
        silent()

    # 1. The wrong Python. Bare python3 on PATH is Anaconda 3.9.7 (ipywidgets 7,
    #    numpy 1) and silently renders the notebook UI blank.
    for token in first_tokens(command):
        if BARE_PYTHON.match(token) or token.startswith(ANACONDA):
            pre_decision(
                "deny",
                f"`{token}` resolves to Anaconda 3.9.7 (numpy 1, ipywidgets 7), which "
                "CLAUDE.md calls wrong for every project in ~/Claude — under it the "
                "widget UI renders blank with no error.\n"
                f"Name the interpreter: {PYTHON} in this repo (3.12, numpy 2, "
                "ipywidgets 8 — the py312-rhs kernel), .venv/bin/python in Chronic.",
            )

    # 2. An analysis entry point pointed at real recordings.
    for pattern, safe_flag in REAL_DATA_ENTRY_POINTS.items():
        if not re.search(pattern, command):
            continue
        if safe_flag and safe_flag in command:
            break
        tool = pattern.replace(r"\.", ".").replace(r"-m\s+", "-m ")
        detail = (
            f"Add {safe_flag} to compute without writing, or confirm to run for real."
            if safe_flag
            else "This tool has no dry-run flag: it reads recordings and writes outputs."
        )
        pre_decision(
            "ask",
            f"This runs {tool} on real data.\n"
            f"CLAUDE.md: running a CLI, the batch runner or a notebook function on real "
            f"recordings is JF's call. {detail}",
        )

    # 3. A shell command that modifies something inside a data folder.
    for root in DATA_ROOTS:
        if str(root) in command and MUTATING.search(command):
            pre_decision(
                "ask",
                f"This command modifies something under {root}.\n"
                "CLAUDE.md: do not write into data folders without asking. Reading "
                "(ls, cat, grep, find) is fine; this is not a read.",
            )
    silent()


# --------------------------------------------------------------------------- #
# PostToolUse: Write | Edit | NotebookEdit
# --------------------------------------------------------------------------- #
SKIP_DIRS = ("archive", "wave_clus-master", "__pycache__", ".claude")


def run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=REPO, capture_output=True, text=True, timeout=120
    )


def post_checks() -> None:
    data = event()
    path = target_path(data.get("tool_input") or {})
    if path is None:
        silent()
    path = path if path.is_absolute() else (REPO / path)
    try:
        relative = path.relative_to(REPO)
    except ValueError:
        silent()  # not our file
    if relative.parts and relative.parts[0] in SKIP_DIRS:
        silent()
    if not path.exists():
        silent()

    if path.suffix == ".ipynb":
        result = run([PYTHON, "-m", "json.tool", str(path)])
        if result.returncode != 0:
            post_block(
                f"{relative} is no longer valid JSON — the notebook will not open:\n"
                f"{result.stderr.strip()}"
            )
        silent()

    if path.suffix != ".py":
        silent()

    result = run([PYTHON, "-m", "py_compile", str(path)])
    if result.returncode != 0:
        post_block(f"py_compile failed on {relative}:\n{result.stderr.strip()}")

    # Module reload order is load-bearing: a dependent listed before its dependency
    # keeps a stale symbol on reload, or raises ImportError for a new name.
    chain = run([PYTHON, "wideband_main_ui.py"])
    if chain.returncode != 0:
        post_block(
            f"_RELOAD_CHAIN in wideband_main_ui.py is no longer leaves-first after "
            f"editing {relative}:\n{(chain.stdout + chain.stderr).strip()}\n"
            "Insert the module after every project module it imports."
        )
    silent()


# --------------------------------------------------------------------------- #
# SessionStart
# --------------------------------------------------------------------------- #
PLIST = pathlib.Path(
    "/Users/jf/Library/LaunchAgents/com.jillfeng123.stimulation-agent.git-backup.plist"
)


def session_check() -> None:
    notes = []

    status = run(["git", "status", "--porcelain"])
    if status.returncode != 0:
        notes.append(
            f"git is unhappy in {REPO}:\n{status.stderr.strip()}\n"
            "If this mentions a stale path, the worktree links need repairing:\n"
            '  git worktree repair .claude/worktrees/*\n'
            "Until then the 15-min auto-backup cannot commit or push."
        )

    if PLIST.exists():
        text = PLIST.read_text()
        if str(REPO) not in text:
            notes.append(
                f"The auto-backup launchd job does not point at {REPO}:\n  {PLIST}\n"
                "Its paths are stale (the repo was probably renamed or moved), so nothing "
                "is being committed or pushed every 15 minutes."
            )
    else:
        notes.append(f"The auto-backup launchd job is missing: {PLIST}")

    if not notes:
        silent()
    emit(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "Repo health check:\n\n" + "\n\n".join(notes),
            }
        }
    )


COMMANDS = {
    "write-guard": write_guard,
    "bash-guard": bash_guard,
    "post-checks": post_checks,
    "session-check": session_check,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
