#!/usr/bin/env python3
"""Run lm-eval with compatibility aliases for newer Transformers releases."""

from __future__ import annotations

import runpy
import sys
import types

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


def disable_unused_visual_backends() -> None:
    """Skip lm-eval visual backends that are irrelevant to text evaluation.

    Some lm-eval releases import every backend eagerly. Their visual modules
    can require Transformers symbols that moved between releases, preventing
    even the plain text `hf` backend from starting. Pre-registering empty
    modules keeps those optional backends out of this text-only evaluation.
    """
    for module_name in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))


install_transformers_compat()
disable_unused_visual_backends()
runpy.run_module("lm_eval.__main__", run_name="__main__")
