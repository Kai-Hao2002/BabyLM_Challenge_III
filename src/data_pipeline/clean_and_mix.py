import os
import logging
from datasets import load_from_disk, concatenate_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_by_length(example, lang):
    """
    Filter meaningless ultra-short texts based on EDA results.
    Filter out text under 10 characters; for English and Dutch, 
    filter out text under 30 characters (roughly 5-6 words).
    """
    text_len = len(example['text'])
    if lang == 'zho' and text_len < 10:
        return False
    if lang in ['eng', 'nld'] and text_len < 30:
        return False
    return True

def calculate_adjusted_tokens(target_tokens, lang):
    """
    Calculate actual allowed sampling tokens adjusted by Byte Premium
    """
    premiums = {'eng': 1.0, 'nld': 1.0516, 'zho': 0.9894}
    # 公式: 實際抽取量 = 目標預算 / 溢價係數
    # Formula: Actual sampled = Target budget / Premium factor
    return int(target_tokens / premiums[lang])

def get_row_count_for_budget(dataset, target_budget, lang):
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
        
        needed_rows = get_row_count_for_budget(shuffled_lang_ds, actual_allowed_tokens, lang)
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
        cleaned_ds = raw_ds.filter(lambda x: filter_by_length(x, lang))
        datasets[lang] = cleaned_ds
        logging.info(f"{lang.upper()} Clean Finished: {len(raw_ds['train'])} -> {len(cleaned_ds['train'])}")

    # 2. Define total budget (10M for prototyping, 100M for official)
    TOTAL_BUDGET = 10_000_000  

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