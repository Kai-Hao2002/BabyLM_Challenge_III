import os
import logging
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from clean_and_mix import get_exact_row_count_for_budget, calculate_adjusted_tokens

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_baseline_dataset(total_budget=10_000_000, val_ratio=0.1):
    logging.info("========== Generating Baseline Dataset ==========")
    langs = ['eng', 'zho', 'nld']
    
    try:
        tokenizer = Tokenizer.from_file("tokenizers/tokenizer_10M_baseline.json")
    except Exception:
        tokenizer = None
        logging.warning("Tokenizer not found!")

    train_chunks = []
    val_chunks = []
    
    # Baseline strategy: Equal token budget for each language (1:1:1 ratio)
    ratio_per_lang = 1.0 / 3.0

    for lang in langs:
        logging.info(f"Processing baseline for {lang.upper()}...")
        # 1. load and shuffle the raw dataset for the language
        raw_ds = load_from_disk(f"data/raw/{lang}_dataset")['train']
        shuffled_ds = raw_ds.shuffle(seed=42)
        
        # 2. calculate budget and adjust for tokenization differences
        target_budget = total_budget * ratio_per_lang
        actual_allowed_tokens = calculate_adjusted_tokens(target_budget, lang)
        
        # 3. precisely calculate how many rows we need to meet the adjusted token budget, using the tokenizer if available
        needed_train_rows = get_exact_row_count_for_budget(shuffled_ds, actual_allowed_tokens, lang, tokenizer)
        needed_val_rows = int(needed_train_rows * val_ratio)
        
        # 4. cut the dataset to the needed number of rows for train and validation
        train_sampled = shuffled_ds.select(range(needed_train_rows))
        val_sampled = shuffled_ds.select(range(needed_train_rows, needed_train_rows + needed_val_rows))
        
        if "language" in train_sampled.column_names:
            train_sampled = train_sampled.remove_columns("language")
            val_sampled = val_sampled.remove_columns("language")
            
        train_sampled = train_sampled.add_column("language", [lang] * len(train_sampled))
        val_sampled = val_sampled.add_column("language", [lang] * len(val_sampled))
        
        train_chunks.append(train_sampled)
        val_chunks.append(val_sampled)

    # 5. mix the three languages together and shuffle again to create the final baseline dataset
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
    # generate baseline dataset with a total token budget of 10M for quick prototyping
    create_baseline_dataset(total_budget=10_000_000)