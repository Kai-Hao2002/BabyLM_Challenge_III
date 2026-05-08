import os
import logging
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
# 引入你寫好的計數函數
from clean_and_mix import get_exact_row_count_for_budget, calculate_adjusted_tokens

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_baseline_dataset(total_budget=10_000_000, val_ratio=0.1):
    logging.info("========== Generating Baseline Dataset ==========")
    langs = ['eng', 'zho', 'nld']
    
    # 假設你已經有一顆最普通的 baseline_tokenizer.json (1:1:1 訓練出來的)
    try:
        tokenizer = Tokenizer.from_file("tokenizers/tokenizer_10M_baseline.json")
    except Exception:
        tokenizer = None
        logging.warning("Tokenizer not found!")

    train_chunks = []
    val_chunks = []
    
    # Baseline 策略：100M 預算直接平分 (1:1:1)
    ratio_per_lang = 1.0 / 3.0

    for lang in langs:
        logging.info(f"Processing baseline for {lang.upper()}...")
        # 1. 直接載入 raw dataset，不做 deep_quality_filter (保留雜訊)
        raw_ds = load_from_disk(f"data/raw/{lang}_dataset")['train']
        shuffled_ds = raw_ds.shuffle(seed=42)
        
        # 2. 計算預算 (套用官方 Byte Premium)
        target_budget = total_budget * ratio_per_lang
        actual_allowed_tokens = calculate_adjusted_tokens(target_budget, lang)
        
        # 3. 精確截斷
        needed_train_rows = get_exact_row_count_for_budget(shuffled_ds, actual_allowed_tokens, lang, tokenizer)
        needed_val_rows = int(needed_train_rows * val_ratio)
        
        # 4. 切分與標記
        train_sampled = shuffled_ds.select(range(needed_train_rows))
        val_sampled = shuffled_ds.select(range(needed_train_rows, needed_train_rows + needed_val_rows))
        
        if "language" in train_sampled.column_names:
            train_sampled = train_sampled.remove_columns("language")
            val_sampled = val_sampled.remove_columns("language")
            
        train_sampled = train_sampled.add_column("language", [lang] * len(train_sampled))
        val_sampled = val_sampled.add_column("language", [lang] * len(val_sampled))
        
        train_chunks.append(train_sampled)
        val_chunks.append(val_sampled)

    # 5. 混合所有語言，直接打包成單一的 Stage_Baseline
    mixed_train = concatenate_datasets(train_chunks).shuffle(seed=42)
    mixed_val = concatenate_datasets(val_chunks).shuffle(seed=42)
    
    final_dataset = DatasetDict({
        'train': mixed_train,
        'validation': mixed_val
    })
    
    scale_folder = "processed_100M" if total_budget > 10_000_000 else "processed_10M"
    save_path = os.path.join("data", scale_folder, "Stage_Baseline")
    
    final_dataset.save_to_disk(save_path)
    logging.info(f"✅ Baseline Mixed! Train: {len(mixed_train):,} rows, Val: {len(mixed_val):,} rows. Saved to: {save_path}")

if __name__ == "__main__":
    # 產出 10M 版本的 Baseline
    create_baseline_dataset(total_budget=10_000_000)