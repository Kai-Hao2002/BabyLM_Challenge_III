import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk
from src.training.dataset import BabyLMPackedMaskedDataset, ID_TO_LANG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test wrapped packing with per-token language tracking."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed_10M/Stage_Baseline/train",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="tokenizers/tokenizer_10M_baseline.json",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Use 0 to pack the full dataset.",
    )
    parser.add_argument("--show-tokens", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()

    hf_dataset = load_from_disk(str(REPO_ROOT / args.dataset_path))
    if args.max_rows > 0:
        hf_dataset = hf_dataset.select(range(min(args.max_rows, len(hf_dataset))))

    packed_dataset = BabyLMPackedMaskedDataset(
        hf_dataset=hf_dataset,
        tokenizer_path=str(REPO_ROOT / args.tokenizer_path),
        max_length=args.max_length,
        mlm_probability=0.15,
        packing_strategy="wrapped",
    )

    assert len(packed_dataset) > 0, "Packed dataset is empty."

    sample = packed_dataset[args.sample_index]
    required_keys = {"input_ids", "attention_mask", "labels", "language_ids"}
    assert required_keys <= set(sample), f"Missing keys: {required_keys - set(sample)}"

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    labels = sample["labels"]
    language_ids = sample["language_ids"]

    assert input_ids.shape[0] == args.max_length
    assert attention_mask.shape[0] == args.max_length
    assert labels.shape[0] == args.max_length
    assert language_ids.shape[0] == args.max_length

    valid_token_count = int(attention_mask.sum().item())
    language_token_count = int(((language_ids >= 0) & attention_mask.bool()).sum().item())
    masked_target_count = int((labels != -100).sum().item())

    assert valid_token_count == language_token_count, (
        "Per-token language ids do not match valid attention positions."
    )
    assert masked_target_count > 0, "No MLM prediction targets in this sample."

    unique_lang_ids = sorted(
        int(x) for x in language_ids[language_ids >= 0].unique().tolist()
    )
    lang_counts = {
        ID_TO_LANG[lang_id]: int(((language_ids == lang_id) & attention_mask.bool()).sum().item())
        for lang_id in unique_lang_ids
    }

    print("\n===== PACKED SAMPLE CHECK =====")
    print(f"dataset_path: {args.dataset_path}")
    print(f"packed_samples: {len(packed_dataset)}")
    print(f"sample_index: {args.sample_index}")
    print(f"sequence_length: {input_ids.shape[0]}")
    print(f"valid_tokens: {valid_token_count}")
    print(f"mlm_targets: {masked_target_count}")
    print(f"language_counts: {lang_counts}")

    print("\n===== TOKEN DEBUG =====")
    print("idx  token_id  label    lang  token")

    show_n = min(args.show_tokens, args.max_length)
    for idx in range(show_n):
        token_id = int(input_ids[idx].item())
        label_id = int(labels[idx].item())
        lang_id = int(language_ids[idx].item())
        token = packed_dataset.tokenizer.decode([token_id])
        lang = ID_TO_LANG.get(lang_id, "PAD")
        label = "-" if label_id == -100 else str(label_id)

        print(
            f"{idx:<4} {token_id:<8} {label:<8} {lang:<4} {repr(token)}"
        )

    if len(unique_lang_ids) > 1:
        print("\nOK: this packed sample contains multiple languages.")
    else:
        print("\nOK: packing and language tracking are valid for this sample.")
        print("Note: this sample contains one language; try --sample-index N to inspect mixing.")


if __name__ == "__main__":
    main()
