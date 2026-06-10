#!/usr/bin/env python3
"""Submit official BabyLM multilingual evaluation jobs on LRZ.

The launcher scans local experiment outputs for HuggingFace final_model
checkpoints and submits one Slurm worker per model. It can also run in dry-run
mode to inspect the manifest before submitting anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_ROOTS = ("outputs",)
DEFAULT_LANGS = ("eng", "nld", "zho")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sanitize(value: str) -> str:
    value = value.replace(os.sep, "-").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def model_record_from_checkpoint(ckpt: Path) -> dict:
    root = repo_root()
    if ckpt.is_absolute():
        rel_ckpt = ckpt.relative_to(root)
    else:
        rel_ckpt = ckpt

    parts = rel_ckpt.parts
    if len(parts) >= 5 and parts[0] == "outputs" and parts[-2:] == ("checkpoints", "final_model"):
        experiment_root = parts[1]
        model_name = parts[-3]
    elif len(parts) >= 4 and parts[-2:] == ("checkpoints", "final_model"):
        experiment_root = parts[-4]
        model_name = parts[-3]
    else:
        parent = rel_ckpt.parent
        experiment_root = parent.parent.name if parent.parent.name else "manual"
        model_name = parent.name

    model_key = f"{experiment_root}/{model_name}"
    eval_model_id = sanitize(f"{experiment_root}--{model_name}")
    return {
        "eval_model_id": eval_model_id,
        "experiment_root": experiment_root,
        "model_name": model_name,
        "model_path": str(rel_ckpt),
        "model_key": model_key,
    }


def normalize_model_arg(model_arg: Path) -> Path:
    root = repo_root()
    model_path = model_arg if model_arg.is_absolute() else root / model_arg
    if model_path.name == "final_model" and model_path.parent.name == "checkpoints":
        return model_path
    candidate = model_path / "checkpoints" / "final_model"
    if candidate.exists():
        return candidate
    return model_path


def explicit_models(model_args: list[str], include_regex: str | None, exclude_regex: str | None) -> list[dict]:
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    root = repo_root()
    models: list[dict] = []

    for item in model_args:
        ckpt_abs = normalize_model_arg(Path(item))
        if not ckpt_abs.exists():
            print(f"[WARN] Missing model path: {ckpt_abs}")
            continue
        if not ckpt_abs.is_dir():
            print(f"[WARN] Model path is not a directory: {ckpt_abs}")
            continue
        try:
            ckpt_for_record = ckpt_abs.relative_to(root)
        except ValueError:
            print(f"[WARN] Model path is outside repository and cannot be used by the worker: {ckpt_abs}")
            continue

        model = model_record_from_checkpoint(ckpt_for_record)
        if include_re and not include_re.search(model["model_key"]):
            continue
        if exclude_re and exclude_re.search(model["model_key"]):
            continue
        models.append(model)

    return dedupe_models(models)


def dedupe_models(models: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique_models: list[dict] = []
    for model in models:
        if model["model_path"] in seen:
            continue
        seen.add(model["model_path"])
        unique_models.append(model)
    return unique_models


def discover_models(roots: list[Path], include_regex: str | None, exclude_regex: str | None) -> list[dict]:
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    root = repo_root()
    models: list[dict] = []

    for root_arg in roots:
        search_root = root_arg if root_arg.is_absolute() else root / root_arg
        if not search_root.exists():
            print(f"[WARN] Missing root: {search_root}")
            continue

        candidates = set(search_root.glob("*/checkpoints/final_model"))
        candidates.update(search_root.glob("*/*/checkpoints/final_model"))

        for ckpt in sorted(candidates):
            if not ckpt.is_dir():
                continue
            rel_ckpt = ckpt.relative_to(root)
            experiment_root = rel_ckpt.parts[1] if rel_ckpt.parts[0] == "outputs" else rel_ckpt.parts[0]
            model_name = rel_ckpt.parts[-3]
            model_key = f"{experiment_root}/{model_name}"

            if include_re and not include_re.search(model_key):
                continue
            if exclude_re and exclude_re.search(model_key):
                continue

            models.append(model_record_from_checkpoint(rel_ckpt))

    return dedupe_models(models)


def write_manifest(run_dir: Path, args: argparse.Namespace, models: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "roots": args.roots,
        "langs": args.langs,
        "include": args.include,
        "exclude": args.exclude,
        "worker": args.worker,
        "partition": args.partition,
        "time": args.time,
        "mem": args.mem,
        "cpus_per_task": args.cpus_per_task,
        "gpus": args.gpus,
        "batch_size": args.batch_size,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_lr": args.finetune_lr,
        "finetune_max_epochs": args.finetune_max_epochs,
        "finetune_patience": args.finetune_patience,
        "finetune_seed": args.finetune_seed,
    }
    (run_dir / "launcher_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest_path = run_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for model in models:
            f.write(json.dumps(model, sort_keys=True) + "\n")
    return manifest_path


def submit_job(args: argparse.Namespace, run_id: str, model: dict) -> str:
    root = repo_root()
    worker = root / args.worker
    (root / "logs" / "slurm").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(root),
            "RUN_ID": run_id,
            "EVAL_MODEL_ID": model["eval_model_id"],
            "MODEL_PATH": model["model_path"],
            "MODEL_NAME": model["model_name"],
            "EXPERIMENT_ROOT": model["experiment_root"],
            "LANGS": " ".join(args.langs),
            "BATCH_SIZE": args.batch_size,
            "FT_BSZ": str(args.finetune_batch_size),
            "FT_LR": args.finetune_lr,
            "FT_MAX_EPOCHS": str(args.finetune_max_epochs),
            "FT_PATIENCE": str(args.finetune_patience),
            "FT_SEED": str(args.finetune_seed),
            "FT_MAX_SEQ_LENGTH": str(args.finetune_max_seq_length),
            "CONDA_ENV": args.conda_env,
        }
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=eval_{model['eval_model_id'][:40]}",
        f"--partition={args.partition}",
        f"--gres=gpu:{args.gpus}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        str(worker),
    ]
    completed = subprocess.run(cmd, cwd=root, env=env, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch LRZ BabyLM multilingual evaluation jobs.")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS),
                        help="Roots to scan for */*/checkpoints/final_model.")
    parser.add_argument("--models", nargs="+",
                        help="Explicit models to evaluate. Accepts either model experiment dirs "
                             "or .../checkpoints/final_model dirs. Overrides --roots scanning.")
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS),
                        choices=["eng", "nld", "zho"], help="Zero-shot languages to evaluate.")
    parser.add_argument("--include", help="Regex over '<experiment_root>/<model_name>' to include.")
    parser.add_argument("--exclude", help="Regex over '<experiment_root>/<model_name>' to exclude.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--worker", default="scripts/slurm/eval_official_multilingual_worker.sh")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Omit for dry-run.")
    parser.add_argument("--limit", type=int, help="Only use the first N discovered models.")

    parser.add_argument("--partition", default="lrz-v100x2")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--cpus-per-task", default="8")
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--conda-env", default="babylm")

    parser.add_argument("--batch-size", default="auto:10", help="lm-eval batch size.")
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--finetune-lr", default="5e-5")
    parser.add_argument("--finetune-max-epochs", type=int, default=10)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--finetune-seed", type=int, default=12)
    parser.add_argument("--finetune-max-seq-length", type=int, default=128)

    args = parser.parse_args()

    root = repo_root()
    if args.models:
        models = explicit_models(args.models, args.include, args.exclude)
    else:
        models = discover_models([Path(p) for p in args.roots], args.include, args.exclude)
    if args.limit is not None:
        models = models[: args.limit]

    out_root = root / "outputs" / "evaluations" / "official_multilingual"
    run_dir = out_root / "runs" / args.run_id
    manifest_path = write_manifest(run_dir, args, models)

    print(f"[INFO] Run id: {args.run_id}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Discovered {len(models)} model(s)")
    for model in models:
        print(f"  - {model['eval_model_id']}: {model['model_path']}")

    if not models:
        return 1

    if not args.submit:
        print("[DRY-RUN] Add --submit to call sbatch.")
        return 0

    submitted_path = run_dir / "submitted_jobs.tsv"
    with submitted_path.open("w") as f:
        f.write("job_id\teval_model_id\tmodel_path\n")
        for model in models:
            job_id = submit_job(args, args.run_id, model)
            f.write(f"{job_id}\t{model['eval_model_id']}\t{model['model_path']}\n")
            print(f"[SUBMITTED] {job_id}: {model['eval_model_id']}")

    print(f"[INFO] Submitted jobs: {submitted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
