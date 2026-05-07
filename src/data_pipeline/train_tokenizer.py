import os
import logging
from datasets import load_from_disk, concatenate_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def batch_iterator(dataset, batch_size=10000):
    """
    Yield texts in batches using a generator to prevent Out-Of-Memory (OOM) errors.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

def train_custom_tokenizer():
    # 1. Load the mixed 10M V1 dataset
    ds_stage1 = load_from_disk("data/processed_10M/Stage_1_Foundation")
    ds_stage2 = load_from_disk("data/processed_10M/Stage_2_Alignment")
    ds_stage3 = load_from_disk("data/processed_10M/Stage_3_HardBoosting")

    
    dataset = concatenate_datasets([ds_stage1, ds_stage2, ds_stage3])
    logging.info(f"Loaded training corpus: {len(dataset):,} docs")

    # 2. Initialize Byte-Level BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Setup training parameters (Vocab size 32k)
    # 加入特殊 Token，這對後續 LLaMA-style 訓練很重要
    # Add special tokens, crucial for LLaMA-style training later
    zh_punctuation = [
        "，", "。", "、", "！", "？", "：", "；", 
        "「", "」", "（", "）", "《", "》", "【", "】"
    ]
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + zh_punctuation
    trainer = trainers.BpeTrainer(
        vocab_size=18000, #16000~24000
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 4. Start Training
    logging.info("Training BPE Tokenizer (may take a few mins)...")
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)

    # 5. Add Post-Processor for BOS/EOS tags
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 6. Save Model
    os.makedirs("tokenizers", exist_ok=True)
    save_path = "tokenizers/tokenizer_10M.json"
    tokenizer.save(save_path)
    logging.info(f"✅ Tokenizer trained and saved to: {save_path}")

if __name__ == "__main__":
    train_custom_tokenizer()