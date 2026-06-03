#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


TOKEN_TYPE_KEY = "bert.embeddings.token_type_embeddings.weight"


def load_safetensors(path):
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to patch model.safetensors checkpoints."
        ) from exc

    return load_file(path)


def save_safetensors(state, path):
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to patch model.safetensors checkpoints."
        ) from exc

    save_file(state, path, metadata={"format": "pt"})


def load_weights(checkpoint_dir):
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        return load_safetensors(safetensors_path), safetensors_path, "safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), bin_path, "bin"

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
    )


def save_weights(state, weights_path, weights_format):
    if weights_format == "safetensors":
        save_safetensors(state, weights_path)
    elif weights_format == "bin":
        torch.save(state, weights_path)
    else:
        raise ValueError(f"Unknown weights format: {weights_format}")


def patch_config(checkpoint_dir, dry_run=False):
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    if config.get("model_type") != "bert":
        print(f"[SKIP] Not a BERT checkpoint: {checkpoint_dir}")
        return False

    old_value = config.get("type_vocab_size")
    config["type_vocab_size"] = 2
    config.setdefault("cls_token_id", 1)
    config.setdefault("sep_token_id", 2)
    config.setdefault("mask_token_id", 4)

    if not dry_run:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    print(f"[CONFIG] type_vocab_size: {old_value} -> 2")
    return True


def patch_tokenizer_config(checkpoint_dir, dry_run=False):
    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    with tokenizer_config_path.open("r") as f:
        tokenizer_config = json.load(f)

    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config.setdefault("unk_token", "<unk>")
    tokenizer_config.setdefault("bos_token", "<s>")
    tokenizer_config.setdefault("eos_token", "</s>")
    tokenizer_config.setdefault("pad_token", "<pad>")
    tokenizer_config.setdefault("mask_token", "<mask>")
    tokenizer_config.setdefault("cls_token", "<s>")
    tokenizer_config.setdefault("sep_token", "</s>")

    if not dry_run:
        with tokenizer_config_path.open("w") as f:
            json.dump(tokenizer_config, f, indent=2)
            f.write("\n")

    print("[TOKENIZER] ensured PreTrainedTokenizerFast cls/sep aliases")


def patch_weights(checkpoint_dir, dry_run=False):
    state, weights_path, weights_format = load_weights(checkpoint_dir)

    if TOKEN_TYPE_KEY not in state:
        raise KeyError(f"Missing {TOKEN_TYPE_KEY} in {weights_path}")

    old_weight = state[TOKEN_TYPE_KEY]
    if old_weight.shape[0] == 2:
        print(f"[WEIGHTS] already patched: {tuple(old_weight.shape)}")
        return
    if old_weight.shape[0] != 1:
        raise ValueError(
            f"Expected token type embedding first dim 1 or 2, "
            f"got {tuple(old_weight.shape)}"
        )

    new_weight = torch.cat([old_weight, old_weight.clone()], dim=0)
    state[TOKEN_TYPE_KEY] = new_weight

    if not dry_run:
        save_weights(state, weights_path, weights_format)

    print(f"[WEIGHTS] {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}")


def patch_checkpoint(checkpoint_dir, dry_run=False):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Not a checkpoint directory: {checkpoint_dir}")

    print(f"\nPatching {checkpoint_dir}")
    should_patch = patch_config(checkpoint_dir, dry_run=dry_run)
    if not should_patch:
        return

    patch_weights(checkpoint_dir, dry_run=dry_run)
    patch_tokenizer_config(checkpoint_dir, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Patch old BERT checkpoints from type_vocab_size=1 to 2 by "
            "duplicating token_type_embeddings row 0."
        )
    )
    parser.add_argument("checkpoint_dirs", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for checkpoint_dir in args.checkpoint_dirs:
        patch_checkpoint(checkpoint_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
