"""
Shared helper for locating the active conda installation without hardcoding
its path. Used by any script that needs to invoke a binary from a specific
conda env (e.g. `labelme` from `annotate`, `python` from `qwen_vlm`) via
subprocess, since those envs live under the conda base install and that
base install's location is machine-specific.
"""

import os
import shutil
import subprocess
from pathlib import Path


def resolve_conda_base():
    """
    Locate the conda base install directory.

    Tries, in order: CONDA_EXE (set by conda in any activated shell),
    CONDA_PREFIX (points at the currently active env, one level below
    the base install for non-base envs), then `conda info --base` as a
    last resort.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        # CONDA_EXE is .../<base>/bin/conda or .../<base>/condabin/conda
        return str(Path(conda_exe).resolve().parents[1])

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        base_marker = Path(conda_prefix) / "condabin"
        if base_marker.exists():
            return conda_prefix
        # An activated non-base env's prefix is .../<base>/envs/<name>
        candidate = Path(conda_prefix).parent.parent
        if (candidate / "condabin").exists():
            return str(candidate)

    conda_bin = shutil.which("conda")
    if conda_bin:
        result = subprocess.run(
            [conda_bin, "info", "--base"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()

    raise RuntimeError(
        "Could not locate a conda installation (checked CONDA_EXE, "
        "CONDA_PREFIX, and `conda info --base`)."
    )
