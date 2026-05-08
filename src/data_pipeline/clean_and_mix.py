import os,re
import logging
from datasets import load_from_disk, concatenate_datasets
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""

"""
try:
    #GLOBAL_TOKENIZER = Tokenizer.from_file("tokenizers/tokenizer_10M.json")
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
    # Filter out paragraphs full of numbers, punctuation, or gibberish
    alpha_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text)) 
    if text_len > 0 and (alpha_chars / text_len) < 0.5:
        return False # 如果文字佔不到整段的一半，丟棄
        
    # 3. Language Purity Check - Critical Optimization!
    if lang == 'zho':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if alpha_chars > 0 and (chinese_chars / alpha_chars) < 0.7:
            return False # 文字中少於 70% 是中文字，丟棄
            
    # 荷蘭文純度檢查 (防止英文污染與非自然語言)
    if lang == 'nld':
        # 將文本轉小寫並切分成單字集合
        words = set(text.lower().split())
        
        # 荷蘭文特有的高頻停用詞 (定冠詞、不定冠詞、常見介系詞與動詞)
        dutch_stopwords = {"de", "het", "een", "en", "van", "is", "dat", "in", "te", "op", "voor", "met", "zijn", "niet", "om"}
        # 英文特有的高頻停用詞 (作為對抗組)
        english_stopwords = {"the", "and", "of", "to", "a", "that", "was", "he", "it", "with", "as", "his", "on", "be"}
        
        # 計算文本中包含了多少個獨特的荷文/英文停用詞
        nld_count = len(words.intersection(dutch_stopwords))
        eng_count = len(words.intersection(english_stopwords))
        
        # 規則 A：自然語言檢查。如果句子夠長（超過 10 個單字），但完全沒有任何常見荷蘭文停用詞，
        # 這高機率是列表、程式碼、雜訊或純外文，直接丟棄。
        if len(words) >= 10 and nld_count == 0:
            return False
            
        # 規則 B：語系對抗檢查。如果文本中的英文高頻詞多於荷蘭文高頻詞，
        # 代表這段文本受到了嚴重的英文污染，直接丟棄。
        if eng_count > nld_count:
            return False
            
        # 規則 C：荷蘭文特殊字母結構檢查 (可選加強版)。荷蘭文常有連續雙母音 (aa, ee, oo, uu) 
        # 或特定子音群 (sch)。如果需要更嚴格，可以檢查這些特徵，但目前 A 與 B 已經能過濾掉 90% 的雜訊。
    
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


from datasets import DatasetDict

def prepare_stage_data(datasets, stage_name, ratios, total_budget, val_ratio, output_base_dir="data"):
    """
    Create, sample, mix, shuffle, split train/val, and save the dataset for a specific stage.
    """
    logging.info(f"========== Preparing {stage_name} Data ==========")
    train_chunks = []
    val_chunks = []
    
    for lang, ratio in ratios.items():
        # 1. 根據 Byte Premium 計算該語言的 Train Token 預算
        target_budget = total_budget * ratio
        actual_allowed_tokens = calculate_adjusted_tokens(target_budget, lang)
        
        dataset_lang = datasets[lang]['train'] 
        shuffled_lang_ds = dataset_lang.shuffle(seed=42) 
        
        # 2. compute the exact number of rows needed to meet the adjusted token budget using the precise BPE-based calculation
        needed_train_rows = get_exact_row_count_for_budget(shuffled_lang_ds, actual_allowed_tokens, lang)
        
        # 3. decide val size based on the train size and val_ratio
        needed_val_rows = int(needed_train_rows * val_ratio)
        
        logging.info(f"[{lang}] Train tokens: {actual_allowed_tokens:,} -> Train rows: {needed_train_rows:,} | Val rows: {needed_val_rows:,}")
        
        # security check: if the needed rows exceed the dataset length, adjust the val size accordingly
        if (needed_train_rows + needed_val_rows) > len(shuffled_lang_ds):
            logging.warning(f"[{lang}] Not enough data for val! Reducing val size.")
            needed_val_rows = len(shuffled_lang_ds) - needed_train_rows

        # 4. spilt the dataset into train and val based on the calculated row counts
        train_sampled = shuffled_lang_ds.select(range(needed_train_rows))
        val_sampled = shuffled_lang_ds.select(range(needed_train_rows, needed_train_rows + needed_val_rows))
        
        # process the language column
        if "language" in train_sampled.column_names:
            train_sampled = train_sampled.remove_columns("language")
            val_sampled = val_sampled.remove_columns("language")
            
        train_sampled = train_sampled.add_column("language", [lang] * len(train_sampled))
        val_sampled = val_sampled.add_column("language", [lang] * len(val_sampled))
        
        train_chunks.append(train_sampled)
        val_chunks.append(val_sampled)
        
    # 5. Mix, shuffle, and save the final dataset for this stage
    mixed_train = concatenate_datasets(train_chunks).shuffle(seed=42)
    mixed_val = concatenate_datasets(val_chunks).shuffle(seed=42)
    
    # 6. Save the mixed dataset to disk for later training use
    final_dataset = DatasetDict({
        'train': mixed_train,
        'validation': mixed_val
    })
    
    scale_folder = "processed_10M" if total_budget <= 10_000_000 else "processed_100M"
    save_path = os.path.join(output_base_dir, scale_folder, stage_name)
    
    final_dataset.save_to_disk(save_path)
    logging.info(f"✅ {stage_name} Mixed! Train: {len(mixed_train):,} rows, Val: {len(mixed_val):,} rows. Saved to: {save_path}\n")
    
    return final_dataset

def main():
    langs = ['eng', 'zho', 'nld']
    datasets = {}
    
    # 1. Load and Clean Data
    for lang in langs:
        path = f"data/raw/{lang}_dataset"
        raw_ds = load_from_disk(path)

        # 紀錄清洗前大小
        original_size = len(raw_ds['train'])

        # Apply filtering rules
        cleaned_ds = raw_ds.filter(lambda x: deep_quality_filter(x, lang))

        # 紀錄清洗後大小與計算差異
        cleaned_size = len(cleaned_ds['train'])
        removed_size = original_size - cleaned_size
        removed_ratio = (removed_size / original_size) * 100 if original_size > 0 else 0
        

        datasets[lang] = cleaned_ds
        # 輸出至終端機
        logging.info(f"{lang.upper()} Clean Finished: {original_size:,} -> {cleaned_size:,} (Removed {removed_ratio:.2f}%)")

    # 2. Define total budget (10M for prototyping, 100M for official)
    #TOTAL_BUDGET = 100_000_000
    TOTAL_BUDGET = 100_000_000

    # 3. Define staged curriculum ratios
    curriculum = {
        "Stage_1_Foundation": {
            'budget_ratio': 0.30, 
            'lang_ratios': {'eng': 0.50, 'zho': 0.25, 'nld': 0.25}
        },
        "Stage_2_Alignment": {
            'budget_ratio': 0.30, 
            'lang_ratios': {'eng': 0.33, 'zho': 0.33, 'nld': 0.34}
        },
        "Stage_3_HardBoosting": {
            'budget_ratio': 0.40, 
            'lang_ratios': {'eng': 0.20, 'zho': 0.40, 'nld': 0.40}
        }
    }

    # 4. Generate Mixed Data
    for stage, config in curriculum.items():
        stage_budget = TOTAL_BUDGET * config['budget_ratio']
        logging.info(f"\n {stage} budget is : {int(stage_budget):,} Tokens")

        prepare_stage_data(datasets, stage, config['lang_ratios'], stage_budget, val_ratio=0.1, output_base_dir="data")

if __name__ == "__main__":
    main()