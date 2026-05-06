from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/tokenizer_10M.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

texts = [
    "I like apples.",
    "Ik hou van appels.",
    "我喜歡蘋果。",
]

for text in texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(text)
    print(ids)
    print(decoded)
