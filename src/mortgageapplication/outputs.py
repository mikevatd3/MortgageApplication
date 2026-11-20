"""Where experiment outputs go.

Each output's filename carries the date, time and git commit of the run, e.g.
`approvals__completion__20261120_1432_587de11.csv`, and a sidecar `<file>.json` records how it was made.
`-dirty` after the commit means the code in `src/` or `scripts/` had uncommitted changes when it ran, so the
commit alone doesn't pin down the code. Outside a git checkout the commit is `nogit`.
"""
import hashlib
import json
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from mortgageapplication import load_prompt_template

HERE = Path(__file__).parent


def _git(*args):
    """Output of a git command run inside the repository, or None if git or the repository is missing."""
    try:
        done = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


@lru_cache
def code_version():
    """(short commit, uncommitted files under src/ and scripts/); commit is "nogit" outside a repository."""
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return "nogit", []
    status = _git("status", "--porcelain", "--", ":/src", ":/scripts") or ""
    return commit, [line[3:] for line in status.splitlines()]


@lru_cache
def run_tag():
    """Date, time and commit, fixed for the whole process so every output of one run shares it."""
    commit, uncommitted = code_version()
    return f"{datetime.now():%Y%m%d_%H%M}_{commit}{'-dirty' if uncommitted else ''}"


def output_path(directory, stem, suffix) -> Path:
    """`directory/stem__<date>_<time>_<commit>.suffix`"""
    return Path(directory) / f"{stem}__{run_tag()}{suffix}"


def latest_output(directory, stem, suffix) -> Path:
    """The most recent file `output_path` made for this stem (the date in the name sorts chronologically)."""
    found = sorted(Path(directory).glob(f"{stem}__*{suffix}"))
    if not found:
        raise FileNotFoundError(f"no {stem}__*{suffix} in {directory}; run the script that makes it")
    return found[-1]


def write_run_info(path, script, prompts=(), **extra):
    """Write `<path>.json` next to an output: the full commit, uncommitted files, prompt hashes, and any extras."""
    _, uncommitted = code_version()
    info = {
        "script": script,
        "created": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted": uncommitted,
        # a hash of each prompt template's text, since prompts are edited often and may not be committed yet
        "prompts": {name: hashlib.sha256(load_prompt_template(name).encode()).hexdigest()[:12] for name in prompts},
        **extra,
    }
    Path(f"{path}.json").write_text(json.dumps(info, indent=2, default=str))
