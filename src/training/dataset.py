# dataset.py → 負責把 text 變成 MLM 訓練資料
# 1. load_from_disk 讀 A 產生的 processed_10M stage dataset
# 2. 載入 tokenizer_10M.json
# 3. tokenize text
# 4. 隨機 mask 15% token(待修正, 目前是簡化版 MLM,可變參數max_length,mlm_probability,batch_size,shuffle)
#    -> mask_tokens() 會直接修改 input_ids
# 5. labels 只保留被 mask 的位置，其餘設為 -100
# 6. 保留 language 欄位給 trainer 計算 token exposure
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast


class BabyLMMaskedDataset(Dataset):
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
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = (
            (input_ids == self.tokenizer.pad_token_id)
            | (input_ids == self.tokenizer.bos_token_id)
            | (input_ids == self.tokenizer.eos_token_id)
            | (input_ids == self.tokenizer.mask_token_id)
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, labels

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


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "language": [x["language"] for x in batch],
    }


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

def get_curriculum_dataloaders(
    curriculum_stages,
    tokenizer_path,
    batch_size=8,
    max_length=128,
    mlm_probability=0.15,
    #val_ratio=0.1,
    seed=42,
):
    stage_loaders = []

    for stage in curriculum_stages:
        stage_name = stage["name"]
        dataset_path = stage["path"]

        split_dataset = load_from_disk(dataset_path)

        train_hf_dataset = split_dataset["train"]
        val_hf_dataset = split_dataset["validation"]

        print(f"\n[{stage_name}]")
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

        stage_loaders.append({
            "name": stage_name,
            "train_loader": train_loader,
            "val_loader": val_loader,
        })

    return stage_loaders