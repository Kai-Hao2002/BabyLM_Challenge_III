from collections import Counter
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.dataset import BabyLMPackedMaskedDataset, LANG_TO_ID, ID_TO_LANG

DATASET_PATH = "data/processed_10M/vocab_16k/Baseline_Naive/train"
TOKENIZER_PATH = "tokenizers/tokenizer_10M_16k.json"
MAX_LENGTH = 512


def build_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_file=str(REPO_ROOT / TOKENIZER_PATH),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def simple_paired_pack(all_input_ids, all_language_ids, max_length):
    packed = []

    for start in range(0, len(all_input_ids), max_length):
        input_chunk = all_input_ids[start:start + max_length]
        lang_chunk = all_language_ids[start:start + max_length]

        if len(input_chunk) == 0:
            continue

        assert len(input_chunk) == len(lang_chunk)

        packed.append({
            "input_ids": input_chunk,
            "language_ids": lang_chunk,
        })

    return packed


def count_language_ids(packed_examples):
    counts = Counter()

    for item in packed_examples:
        for lang_id in item["language_ids"]:
            counts[ID_TO_LANG[int(lang_id)]] += 1

    return counts


def print_comparison(title, before, after):
    print(f"\n=== {title} ===")
    for lang in ["eng", "nld", "zho"]:
        diff = after[lang] - before[lang]
        print(f"{lang}: before={before[lang]} after={after[lang]} diff={diff:+}")

    print(
        "total:",
        f"before={sum(before.values())}",
        f"after={sum(after.values())}",
        f"diff={sum(after.values()) - sum(before.values()):+}",
    )


hf_dataset = load_from_disk(str(REPO_ROOT / DATASET_PATH))
tokenizer = build_tokenizer()

before = Counter()
all_input_ids = []
all_language_ids = []

print("Tokenizing dataset...")

for idx, item in enumerate(hf_dataset):
    lang = item["language"]
    lang_id = LANG_TO_ID[lang]

    ids = tokenizer(
        item["text"],
        add_special_tokens=False,
    )["input_ids"]

    before[lang] += len(ids)
    all_input_ids.extend(ids)
    all_language_ids.extend([lang_id] * len(ids))

    if idx % 1000 == 0:
        print(f"Tokenized {idx} rows")

assert len(all_input_ids) == len(all_language_ids)

simple_packed = simple_paired_pack(
    all_input_ids=all_input_ids,
    all_language_ids=all_language_ids,
    max_length=MAX_LENGTH,
)

simple_after = count_language_ids(simple_packed)
print_comparison("SIMPLE PAIRED PACKING", before, simple_after)

for lang in ["eng", "nld", "zho"]:
    assert simple_after[lang] == before[lang], (
        f"Simple paired packing is not conserved for {lang}: "
        f"{before[lang]} -> {simple_after[lang]}"
    )

print("\nOK: simple paired packing preserves per-language token counts.")
print(f"Simple packed samples: {len(simple_packed)}")

print("\nBuilding TRL packed dataset for comparison...")

trl_packed = BabyLMPackedMaskedDataset(
    hf_dataset=hf_dataset,
    tokenizer_path=str(REPO_ROOT / TOKENIZER_PATH),
    max_length=MAX_LENGTH,
    mlm_probability=0.15,
    packing_strategy="wrapped",
)

trl_after = count_language_ids(trl_packed.packed_dataset)
print_comparison("TRL WRAPPED PACKING", before, trl_after)
