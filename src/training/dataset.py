# dataset.py → 負責把 text 變成 MLM 訓練資料
# 1. load_from_disk 讀 A 產生的 processed_10M stage dataset
# 2. 載入 tokenizer_10M.json
# 3. tokenize text
# 4. 隨機 mask 15% token(待修正, 目前是簡化版 MLM,可變參數max_length,mlm_probability,batch_size,shuffle)
#    -> mask_tokens() 會直接修改 input_ids
# 5. labels 只保留被 mask 的位置，其餘設為 -100
# 6. 保留 language 欄位給 trainer 計算 token exposure
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

LANG_TO_ID = {
    "eng": 0,
    "nld": 1,
    "zho": 2,
}

ID_TO_LANG = {
    0: "eng",
    1: "nld",
    2: "zho",
}


def build_paired_wrapped_packed_samples(
    #  packing 不再依賴 trl.pack_dataset，也不要用任何可能分欄位處理的 map() packing
    hf_dataset,
    tokenizer,
    max_length,
    lang_to_id,
    insert_eos=False,
    eos_language_id=-100,
    drop_last=True,
):
    all_input_ids = []
    all_language_ids = []

    for idx, item in enumerate(hf_dataset):
        text = item["text"]
        lang = item["language"]

        if lang not in lang_to_id:
            raise ValueError(f"Unknown language: {lang}")

        input_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        if len(input_ids) > 0:
            language_ids = [lang_to_id[lang]] * len(input_ids)

            if insert_eos and tokenizer.eos_token_id is not None:
                input_ids = input_ids + [tokenizer.eos_token_id]
                language_ids = language_ids + [eos_language_id]

            all_input_ids.extend(input_ids)
            all_language_ids.extend(language_ids)

        if idx % 1000 == 0:
            print(f"Tokenized {idx} rows")

    if len(all_input_ids) != len(all_language_ids):
        raise ValueError("input_ids and language_ids stream length mismatch.")

    packed_samples = []
    for start in range(0, len(all_input_ids), max_length):
        end = start + max_length
        input_chunk = all_input_ids[start:end]
        language_chunk = all_language_ids[start:end]

        if len(input_chunk) < max_length and drop_last:
            break

        packed_samples.append({
            "input_ids": input_chunk,
            "language_ids": language_chunk,
        })

    return packed_samples

def mask_tokens_bert_style(input_ids, tokenizer, mlm_probability):
    labels = input_ids.clone()

    special_tokens_mask = (
        (input_ids == tokenizer.pad_token_id)
        | (input_ids == tokenizer.bos_token_id)
        | (input_ids == tokenizer.eos_token_id)
        | (input_ids == tokenizer.mask_token_id)
    )

    valid_token_mask = ~special_tokens_mask

    probability_matrix = torch.full(
        labels.shape,
        mlm_probability,
        dtype=torch.float,
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # zero-label protection
    if masked_indices.sum() == 0 and valid_token_mask.sum() > 0:
        valid_indices = torch.nonzero(valid_token_mask, as_tuple=False).view(-1)
        random_pos = valid_indices[torch.randint(0, len(valid_indices), (1,))]
        masked_indices[random_pos] = True

    labels[~masked_indices] = -100

    # 80% -> <mask>
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )

    random_words = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=labels.shape,
        dtype=torch.long,
    )
    input_ids[indices_random] = random_words[indices_random]

    # remaining 10% unchanged
    return input_ids, labels

class BabyLMMaskedDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer_path,
        max_length=128,
        mlm_probability=0.15,
    ):
        self.dataset = hf_dataset

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)
    
   
    def mask_tokens(self, input_ids):
        return mask_tokens_bert_style(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]

        text = item["text"]
        language = item["language"]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        input_ids, labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "language": language,
        }

class BabyLMChunkedMaskedDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer_path,
        max_length=128,
        mlm_probability=0.15,
    ):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.chunks = []

        print("Building chunked dataset...")

        for idx, item in enumerate(hf_dataset):
            text = item["text"]
            language = item["language"]

            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
            )

            input_ids = encoded["input_ids"]

            for start in range(0, len(input_ids), max_length):
                chunk_ids = input_ids[start:start + max_length]

                if len(chunk_ids) == 0:
                    continue

                self.chunks.append({
                    "input_ids": chunk_ids,
                    "language": language,
                })

            if idx % 1000 == 0:
                print(f"Processed {idx} rows, chunks so far: {len(self.chunks)}")

        print(f"Total chunks: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def mask_tokens(self, input_ids):
        return mask_tokens_bert_style(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

    def __getitem__(self, idx):
        item = self.chunks[idx]

        input_ids = item["input_ids"]
        language = item["language"]

        attention_mask = [1] * len(input_ids)

        pad_length = self.max_length - len(input_ids)

        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        input_ids, labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "language": language,
        }
    
def collate_fn(batch):
    output = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }

    if "language" in batch[0]:
        output["language"] = [x["language"] for x in batch]

    if "language_ids" in batch[0]:
        output["language_ids"] = torch.stack([x["language_ids"] for x in batch])

    return output


def get_dataloaders(
    dataset_path,
    tokenizer_path,
    batch_size=8,
    max_length=128,
    mlm_probability=0.15,
    #val_ratio=0.1,
    seed=42,
):
    split_dataset = load_from_disk(dataset_path)

    train_hf_dataset = split_dataset["train"]
    val_hf_dataset = split_dataset["validation"]

    print(f"Train rows: {len(train_hf_dataset)}")
    print(f"Validation rows: {len(val_hf_dataset)}")

    train_dataset = BabyLMMaskedDataset(
        hf_dataset=train_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    val_dataset = BabyLMMaskedDataset(
        hf_dataset=val_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader

def get_baseline_dataloaders(
    train_path,
    val_path,
    tokenizer_path,
    batch_size=8,
    max_length=128,
    mlm_probability=0.15,
):
    train_hf_dataset = load_from_disk(train_path)
    val_hf_dataset = load_from_disk(val_path)

    print("\n[Baseline Dataset]")
    print(f"Train rows: {len(train_hf_dataset)}")
    print(f"Validation rows: {len(val_hf_dataset)}")

    train_dataset = BabyLMMaskedDataset(
        hf_dataset=train_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    val_dataset = BabyLMMaskedDataset(
        hf_dataset=val_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader

def get_baseline_chunked_dataloaders(
    train_path,
    val_path,
    tokenizer_path,
    batch_size=8,
    max_length=128,
    mlm_probability=0.15,
):
    train_hf_dataset = load_from_disk(train_path)
    val_hf_dataset = load_from_disk(val_path)

    print("\n[Baseline Chunked Dataset]")
    print(f"Original train rows: {len(train_hf_dataset)}")
    print(f"Original validation rows: {len(val_hf_dataset)}")

    train_dataset = BabyLMChunkedMaskedDataset(
        hf_dataset=train_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    val_dataset = BabyLMChunkedMaskedDataset(
        hf_dataset=val_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Validation chunks: {len(val_dataset)}")

    return train_loader, val_loader

class BabyLMPackedBaseDataset(TorchDataset):
    # 負責 tokenizer 載入, optional insert_eos
    def __init__(
        self,
        hf_dataset,
        tokenizer_path,
        max_length=256,
        mlm_probability=0.15,
        packing_strategy="wrapped",
        insert_eos=False,
    ):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.insert_eos = insert_eos

        if packing_strategy != "wrapped":
            raise ValueError(
                "Only packing_strategy='wrapped' is supported by paired packing."
            )

        print("Tokenizing dataset for packed training...")
        print("Packing dataset...")
        print(f"Packing strategy: {packing_strategy}")
        print(f"Sequence length: {max_length}")

        self.packed_dataset = build_paired_wrapped_packed_samples(
            hf_dataset=hf_dataset,
            tokenizer=self.tokenizer,
            max_length=max_length,
            lang_to_id=LANG_TO_ID,
            insert_eos=insert_eos,
            eos_language_id=-100,
            drop_last=True,
        )
        print("Packed dataset created.")
        print(f"Packed samples: {len(self.packed_dataset)}")

        first = self.packed_dataset[0]
        print("Packed sample keys:", first.keys())
        print("input_ids length:", len(first["input_ids"]))

        if "language_ids" not in first:
            raise ValueError(
                "language_ids missing from paired packed samples. "
                "We need language_ids for per-token language tracking."
            )
        print("language_ids length:", len(first["language_ids"]))

        if len(first["input_ids"]) != len(first["language_ids"]):
            raise ValueError(
                "input_ids and language_ids length mismatch after packing."
            )

    def __len__(self):
        return len(self.packed_dataset)


class BabyLMPackedMaskedDataset(BabyLMPackedBaseDataset):
    # 負責 BERT MLM 的 mask_tokens() 和 MLM __getitem__()
    def mask_tokens(self, input_ids):
        return mask_tokens_bert_style(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

    def __getitem__(self, idx):
        item = self.packed_dataset[idx]

        input_ids = item["input_ids"]
        language_ids = item["language_ids"]

        attention_mask = [1] * len(input_ids)

        # Safety padding, normally wrapped packing should already produce max_length
        pad_length = self.max_length - len(input_ids)

        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            language_ids = language_ids + [-100] * pad_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        language_ids = torch.tensor(language_ids, dtype=torch.long)

        input_ids, labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "language_ids": language_ids,
        }


class BabyLMPackedCausalDataset(BabyLMPackedBaseDataset):
    # 負責 GPT-2 causal LM 的 __getitem__(), padding label 設 -100, 預設 insert_eos=True
    def __init__(
        self,
        hf_dataset,
        tokenizer_path,
        max_length=256,
        mlm_probability=0.15,
        packing_strategy="wrapped",
        insert_eos=True,
    ):
        super().__init__(
            hf_dataset=hf_dataset,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            mlm_probability=mlm_probability,
            packing_strategy=packing_strategy,
            insert_eos=insert_eos,
        )

    def __getitem__(self, idx):
        item = self.packed_dataset[idx]

        input_ids = item["input_ids"]
        language_ids = item["language_ids"]

        attention_mask = [1] * len(input_ids)

        pad_length = self.max_length - len(input_ids)

        if pad_length > 0:
            # GPT-2 的 labels 等於 input_ids, 非 padding token 都要被預測, padding token 不計算 loss
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            language_ids = language_ids + [-100] * pad_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        language_ids = torch.tensor(language_ids, dtype=torch.long)
        # shift 是 GPT2LMHeadModel 內部做的, -100 的位置會被 loss 忽略
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "language_ids": language_ids,
        }
    
def get_baseline_packed_dataloaders(
    train_path,
    val_path,
    tokenizer_path,
    batch_size=8,
    max_length=256,
    mlm_probability=0.15,
    packing_strategy="wrapped",
    objective="mlm",
    insert_eos=False,
):
    train_hf_dataset = load_from_disk(train_path)
    val_hf_dataset = load_from_disk(val_path)

    if objective == "mlm": # -> BabyLMPackedMaskedDataset
        dataset_cls = BabyLMPackedMaskedDataset 
    elif objective == "causal_lm": # -> BabyLMPackedCausalDataset
        dataset_cls = BabyLMPackedCausalDataset
    else:
        raise ValueError(f"Unknown packed objective: {objective}")

    print("\n[Baseline Packed Dataset]")
    print(f"Objective: {objective}")
    print(f"Insert EOS: {insert_eos}")
    print(f"Original train rows: {len(train_hf_dataset)}")
    print(f"Original validation rows: {len(val_hf_dataset)}")

    train_dataset = dataset_cls(
        hf_dataset=train_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
        packing_strategy=packing_strategy,
        insert_eos=insert_eos,
    )

    val_dataset = dataset_cls(
        hf_dataset=val_hf_dataset,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mlm_probability=mlm_probability,
        packing_strategy=packing_strategy,
        insert_eos=insert_eos,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Train packed samples: {len(train_dataset)}")
    print(f"Validation packed samples: {len(val_dataset)}")

    return train_loader, val_loader
    
def get_curriculum_dataloaders(
    curriculum_stages,
    tokenizer_path,
    batch_size=8,
    max_length=128,
    mlm_probability=0.15,
    use_packing=False,
    packing_strategy="wrapped",
    objective="mlm",
    insert_eos=False,
    validation_path=None,
    val_ratio=0.1,
    seed=42,
):
    stage_loaders = []

    for stage in curriculum_stages:
        stage_name = stage["name"]
        train_hf_dataset = load_from_disk(stage["train_path"])
        stage_val_path = validation_path or stage["val_path"]
        val_hf_dataset = load_from_disk(stage_val_path)

        print(f"\n[{stage_name}]")
        print(f"Objective: {objective}")
        print(f"Use packing: {use_packing}")
        print(f"Insert EOS: {insert_eos}")
        print(f"Validation path: {stage_val_path}")
        print(f"Train rows: {len(train_hf_dataset)}")
        print(f"Validation rows: {len(val_hf_dataset)}")

        if use_packing:
            if objective == "mlm":
                dataset_cls = BabyLMPackedMaskedDataset
            elif objective == "causal_lm":
                dataset_cls = BabyLMPackedCausalDataset
            else:
                raise ValueError(f"Unknown curriculum objective: {objective}")

            train_dataset = dataset_cls(
                hf_dataset=train_hf_dataset,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                mlm_probability=mlm_probability,
                packing_strategy=packing_strategy,
                insert_eos=insert_eos,
            )

            val_dataset = dataset_cls(
                hf_dataset=val_hf_dataset,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                mlm_probability=mlm_probability,
                packing_strategy=packing_strategy,
                insert_eos=insert_eos,
            )
        else:
            train_dataset = BabyLMMaskedDataset(
                hf_dataset=train_hf_dataset,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                mlm_probability=mlm_probability,
            )

            val_dataset = BabyLMMaskedDataset(
                hf_dataset=val_hf_dataset,
                tokenizer_path=tokenizer_path,
                max_length=max_length,
                mlm_probability=mlm_probability,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        stage_loaders.append({
            "name": stage_name,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "target_adjusted_tokens": stage.get("target_adjusted_tokens"),
            "checkpoint_name": stage.get("checkpoint_name", stage_name),
        })

    return stage_loaders
