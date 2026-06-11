#!/usr/bin/env python3
"""Run lm-eval with compatibility aliases for newer Transformers releases."""

from __future__ import annotations

import runpy

import transformers


def install_transformers_compat() -> None:
    try:
        transformers.AutoModelForVision2Seq
    except AttributeError:
        pass
    else:
        return

    fallback = getattr(transformers, "AutoModelForImageTextToText", None)
    if fallback is None:
        fallback = getattr(transformers, "AutoModelForSeq2SeqLM", None)
    if fallback is None:
        raise RuntimeError(
            "lm-eval expects transformers.AutoModelForVision2Seq, but this "
            "Transformers installation has no compatible auto-model class."
        )
    # Transformers is a _LazyModule in recent releases. Direct setattr can be
    # shadowed by its custom attribute loader, so install the alias in the
    # module dictionary explicitly.
    transformers.__dict__["AutoModelForVision2Seq"] = fallback
    if getattr(transformers, "AutoModelForVision2Seq", None) is None:
        raise RuntimeError("Failed to install the AutoModelForVision2Seq compatibility alias.")


install_transformers_compat()
runpy.run_module("lm_eval.__main__", run_name="__main__")
