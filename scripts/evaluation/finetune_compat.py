#!/usr/bin/env python3
"""Run the official finetune script across Transformers API versions."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import transformers.utils


def install_transformers_compat() -> None:
    if not hasattr(transformers.utils, "send_example_telemetry"):
        transformers.utils.send_example_telemetry = lambda *args, **kwargs: None


if len(sys.argv) < 2:
    raise SystemExit("Usage: finetune_compat.py OFFICIAL_SCRIPT [arguments ...]")

script = Path(sys.argv[1]).resolve()
if not script.is_file():
    raise SystemExit(f"Official finetune script not found: {script}")

install_transformers_compat()
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
