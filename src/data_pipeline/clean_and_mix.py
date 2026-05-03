import os,re
import logging
from datasets import load_from_disk, concatenate_datasets
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""

"""
try:
    GLOBAL_TOKENIZER = Tokenizer.from_file("tokenizers/tokenizer_100M.json")
except Exception:
    GLOBAL_TOKENIZER = None
    logging.warning("Custom Tokenizer not detected, falling back to approximation.")


#GLOBAL_TOKENIZER = None

def deep_quality_filter(example, lang):
    """
    Deep Data Quality Filter
    """
    text = example['text']
    text_len = len(text)
    
    # 1. Basic length filtering
    if lang == 'zho' and text_len < 10:
        return False
    if lang in ['eng', 'nld'] and text_len < 30:
        return False
        
    # 2. Alpha Ratio Check
    # Filter out paragraphs full of numbers, punctuation, or gibberish (like log files)
    # Calculate the number of Chinese, English, and Dutch characters (excluding spaces, numbers, and symbols)
    alpha_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text)) 
    if text_len > 0 and (alpha_chars / text_len) < 0.5:
        return False # 如果文字佔不到整段的一半，丟棄
        
    # 3. Language Purity Check - Critical Optimization!
    # If it's a Chinese dataset, ensure a high ratio of Chinese characters to prevent English contamination
    if lang == 'zho':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if alpha_chars > 0 and (chinese_chars / alpha_chars) < 0.7:
            return False # 文字中少於 70% 是中文字，丟棄
            
    # If it's a Dutch dataset, check for specific character features (simplified version)
    # if lang == 'nld':

    
    return True

def calculate_adjusted_tokens(target_tokens, lang):
    """
    Calculate actual allowed sampling tokens adjusted by Byte Premium
    """
    premiums = {'eng': 1.0, 'nld': 1.0516, 'zho': 0.9894}
    # 公式: 實際抽取量 = 目標預算 / 溢價係數
    # Formula: Actual sampled = Target budget / Premium factor
    return int(target_tokens / premiums[lang])

def get_exact_row_count_for_budget(dataset, target_budget, lang, tokenizer=GLOBAL_TOKENIZER):
    """
    Core conversion (Precise Version): Calculate exact token consumption by running actual BPE encoding.
    """
    accumulated_tokens = 0
    
    for idx, item in enumerate(dataset):
        text = item['text']
        
        if tokenizer is not None:
            # Precise calculation: Use BPE Tokenizer to convert text to ID array and count exact length
            encoded_ids = tokenizer.encode(text).ids
            exact_tokens = len(encoded_ids)
            accumulated_tokens += exact_tokens
        else:
            # 回退機制 (Fallback)：如果 Tokenizer 還沒訓練好，沿用先前的粗估邏輯
            if lang == 'zho':
                accumulated_tokens += len(text)
            else:
                accumulated_tokens += len(text.split())
        
        if accumulated_tokens >= target_budget:
            # 記錄誤差，你會發現精確版與粗估版會有差距！
            logging.debug(f"[{lang}] Actual sampled tokens: {accumulated_tokens:,} (Target: {int(target_budget):,})")
            return idx + 1 
            
    logging.warning(f"[{lang}] Warning: Reached end of dataset before hitting the target budget. Total tokens accumulated: {accumulated_tokens:,}")
    return len(dataset)

#def get_row_count_for_budget(dataset, target_budget, lang):
    """
    Core conversion: Iterate through the dataset, accumulate estimated tokens, 
    and return the row index when the budget is reached.
    """
    accumulated_tokens = 0
    
    for idx, item in enumerate(dataset):
        text = item['text']
        # Approximation logic: Characters for ZH, space-separated words for EN/NL
        if lang == 'zho':
            estimated_tokens = len(text)
        else:
            estimated_tokens = len(text.split())
            
        accumulated_tokens += estimated_tokens
        
        # When we reach or exceed the target budget, return the current index + 1 (since index starts at 0)
        if accumulated_tokens >= target_budget:
            return idx + 1 
            
    # If we exhaust the dataset before reaching the budget, return the total length and log a warning
    logging.warning(f"[{lang}] Warning: Reached end of dataset before hitting the target budget. Total tokens accumulated: {accumulated_tokens:,}")
    return len(dataset)

def prepare_stage_data(datasets, stage_name, ratios, total_budget, output_base_dir="data"):
    """
    Create, sample, mix, shuffle, and save the dataset for a specific stage.
    """
    logging.info(f"========== Preparing {stage_name} Data ==========")
    stage_chunks = []
    
    for lang, ratio in ratios.items():
        # 1. Calculate the target token budget for this language and adjust it by Byte Premium
        target_budget = total_budget * ratio
        actual_allowed_tokens = calculate_adjusted_tokens(target_budget, lang)
        
        # 2. [Practical Conversion] Convert Token budget to Row count
        dataset_lang = datasets[lang]['train'] 
        
        # 為了避免每次都抽到開頭的文章，我們可以在抽取前先打亂該語言的資料集
        shuffled_lang_ds = dataset_lang.shuffle(seed=42) 
        
        needed_rows = get_exact_row_count_for_budget(shuffled_lang_ds, actual_allowed_tokens, lang)
        logging.info(f"[{lang}] Actual allowed tokens: {actual_allowed_tokens:,} -> Needed rows: {needed_rows:,}")
        
        # 3. Perform selection and add a language label column for later analysis
        sampled_ds = shuffled_lang_ds.select(range(needed_rows))
        
        if "language" in sampled_ds.column_names:
            sampled_ds = sampled_ds.remove_columns("language")
            
        # Add a language column to keep track of the source language (useful for later analysis)
        sampled_ds = sampled_ds.add_column("language", [lang] * needed_rows)
        stage_chunks.append(sampled_ds)
        
    # 4. Mix and completely shuffle the combined dataset
    mixed_dataset = concatenate_datasets(stage_chunks)
    final_mixed_dataset = mixed_dataset.shuffle(seed=42)
    
    # 5. Save to disk
    scale_folder = "processed_10M" if total_budget <= 10_000_000 else "processed_100M"
    save_path = os.path.join(output_base_dir, scale_folder, stage_name)
    
    final_mixed_dataset.save_to_disk(save_path)
    logging.info(f"✅ {stage_name} Mixed! {len(final_mixed_dataset):,} rows, saved to: {save_path}\n")
    
    return final_mixed_dataset

def main():
    langs = ['eng', 'zho', 'nld']
    datasets = {}
    
    # 1. Load and Clean Data
    for lang in langs:
        path = f"data/raw/{lang}_dataset"
        raw_ds = load_from_disk(path)
        # Apply filtering rules
        cleaned_ds = raw_ds.filter(lambda x: deep_quality_filter(x, lang))
        datasets[lang] = cleaned_ds
        logging.info(f"{lang.upper()} Clean Finished: {len(raw_ds['train'])} -> {len(cleaned_ds['train'])}")

    # 2. Define total budget (10M for prototyping, 100M for official)
    TOTAL_BUDGET = 100_000_000  

    # 3. Define staged curriculum ratios
    curriculum = {
        "Stage_1_Foundation": {'eng': 0.50, 'zho': 0.25, 'nld': 0.25},
        "Stage_2_Alignment": {'eng': 0.33, 'zho': 0.33, 'nld': 0.34},
        "Stage_3_HardBoosting": {'eng': 0.20, 'zho': 0.40, 'nld': 0.40}
    }

    # 4. Generate Mixed Data
    for stage, ratios in curriculum.items():
        prepare_stage_data(datasets, stage, ratios, TOTAL_BUDGET)

if __name__ == "__main__":
    main()