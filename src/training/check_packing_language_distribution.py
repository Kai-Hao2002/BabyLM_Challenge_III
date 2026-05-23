import argparse
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

from src.training.dataset import (
    BabyLMPackedCausalDataset,
    BabyLMPackedMaskedDataset,
    ID_TO_LANG,
    LANG_TO_ID,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare language token distribution before and after packing."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed_10M/vocab_16k/Baseline_Naive/train",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="tokenizers/tokenizer_10M_16k.json",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--objective",
        choices=["mlm", "causal_lm"],
        default="causal_lm",
    )
    parser.add_argument("--insert-eos", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use 0 to check the full dataset.",
    )
    return parser.parse_args()


def format_counts(counts):
    total = sum(counts.values())
    pieces = []
    for lang in ["eng", "nld", "zho"]:
        value = counts.get(lang, 0)
        pct = 100 * value / total if total else 0.0
        pieces.append(f"{lang}: {value:,} ({pct:.2f}%)")
    return " | ".join(pieces) + f" | total: {total:,}"


def main():
    args = parse_args()

    hf_dataset = load_from_disk(str(REPO_ROOT / args.dataset_path))
    if args.max_rows > 0:
        hf_dataset = hf_dataset.select(range(min(args.max_rows, len(hf_dataset))))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(REPO_ROOT / args.tokenizer_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    raw_counts = Counter()
    raw_rows = Counter()
    raw_eos_in_text = 0

    print("Counting raw tokenized dataset...")
    for idx, item in enumerate(hf_dataset):
        lang = item["language"]
        if lang not in LANG_TO_ID:
            raise ValueError(f"Unknown language: {lang}")

        input_ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
        raw_counts[lang] += len(input_ids)
        raw_rows[lang] += 1
        raw_eos_in_text += sum(1 for token_id in input_ids if token_id == tokenizer.eos_token_id)

        if idx % 1000 == 0:
            print(f"Raw counted {idx} rows")

    dataset_cls = (
        BabyLMPackedMaskedDataset
        if args.objective == "mlm"
        else BabyLMPackedCausalDataset
    )

    packed_dataset = dataset_cls(
        hf_dataset=hf_dataset,
        tokenizer_path=str(REPO_ROOT / args.tokenizer_path),
        max_length=args.max_length,
        packing_strategy="wrapped",
        insert_eos=args.insert_eos,
    )

    packed_counts = Counter()
    ignored_tokens = 0
    eos_total = 0
    eos_ignored = 0
    valid_tokens = 0

    print("Counting packed dataset...")
    for idx in range(len(packed_dataset)):
        sample = packed_dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"].bool()
        language_ids = sample["language_ids"]

        valid_tokens += int(attention_mask.sum().item())
        ignored_tokens += int(((language_ids == -100) & attention_mask).sum().item())

        eos_mask = (input_ids == tokenizer.eos_token_id) & attention_mask
        eos_total += int(eos_mask.sum().item())
        eos_ignored += int(((language_ids == -100) & eos_mask).sum().item())

        for lang_id, lang in ID_TO_LANG.items():
            packed_counts[lang] += int(((language_ids == lang_id) & attention_mask).sum().item())

        if idx % 10000 == 0:
            print(f"Packed counted {idx} samples")

    print("\n========== RAW DATASET ==========")
    print(f"rows: {len(hf_dataset):,}")
    print("rows by language:", dict(raw_rows))
    print(format_counts(raw_counts))
    print(f"raw eos tokens already in text: {raw_eos_in_text:,}")

    print("\n========== PACKED DATASET ==========")
    print(f"packed samples: {len(packed_dataset):,}")
    print(f"valid tokens including ignored EOS/pad: {valid_tokens:,}")
    print(format_counts(packed_counts))
    print(f"ignored valid tokens language_id=-100: {ignored_tokens:,}")
    print(f"eos tokens in packed input: {eos_total:,}")
    print(f"eos ignored from exposure: {eos_ignored:,}")

    print("\n========== DIFF PACKED - RAW ==========")
    for lang in ["eng", "nld", "zho"]:
        print(f"{lang}: {packed_counts[lang] - raw_counts[lang]:,}")


if __name__ == "__main__":
    main()
