#!/usr/bin/env python3
"""Run lm-eval with compatibility aliases for newer Transformers releases."""

from __future__ import annotations

import runpy

import transformers


def install_transformers_compat() -> None:
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return

    fallback = getattr(transformers, "AutoModelForImageTextToText", None)
    if fallback is None:
        fallback = getattr(transformers, "AutoModelForSeq2SeqLM", None)
    if fallback is None:
        raise RuntimeError(
            "lm-eval expects transformers.AutoModelForVision2Seq, but this "
            "Transformers installation has no compatible auto-model class."
        )
    transformers.AutoModelForVision2Seq = fallback


install_transformers_compat()
runpy.run_module("lm_eval.__main__", run_name="__main__")
