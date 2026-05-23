from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

ds = load_from_disk("data/processed_10M/vocab_16k/Baseline_Naive/train")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/tokenizer_10M_16k.json",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    mask_token="<mask>",
)

total_tokens = 0

eng_tokens = 0
nld_tokens = 0
zho_tokens = 0

print("Counting tokens...")

for idx, item in enumerate(ds):
    ids = tokenizer(item["text"])["input_ids"]
    token_count = len(ids)

    total_tokens += token_count

    lang = item["language"]

    if lang == "eng":
        eng_tokens += token_count
    elif lang == "nld":
        nld_tokens += token_count
    elif lang == "zho":
        zho_tokens += token_count

    if idx % 1000 == 0:
        print(f"Processed {idx} rows")

print("\n========== RESULT ==========")

print(f"Rows: {len(ds)}")
print(f"Total tokens: {total_tokens:,}")
print(f"Average tokens per row: {total_tokens / len(ds):.2f}")

print("\nPer-language token counts:")
print(f"ENG: {eng_tokens:,}")
print(f"NLD: {nld_tokens:,}")
print(f"ZHO: {zho_tokens:,}")

# ========== RESULT ==========
# Rows: 12401
# Total tokens: 9,874,102
# Average tokens per row: 796.23

# Per-language token counts:
# ENG: 3,334,147
# NLD: 3,170,024
# ZHO: 3,369,931