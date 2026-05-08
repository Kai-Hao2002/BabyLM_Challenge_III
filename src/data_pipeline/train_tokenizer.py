import os
import logging
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

from clean_and_mix import deep_quality_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_rough_10m_corpus():
    """
    冷啟動解決方案：從 Raw Data 中使用「粗估邏輯」抽取約 10M 預算的資料作為 Tokenizer 訓練語料。
    """
    langs = ['eng', 'zho', 'nld']
    # 將 10M 預算平均分配給三種語言 (約每種語言 3.33M)
    #target_budget_per_lang = 10_000_000 / 3 

    """"
    # 總計約 15M 抽樣預算，祭出終極壓抑策略
    budgets = {
        'eng': 12_000_000,  # 80% 
        'nld': 2_550_000,   # 17% 
        'zho': 450_000      # 3%  
    }
    """

    budgets = {
        'eng': 7_500_000, # 75%
        'nld': 2_000_000, # 20%
        'zho': 500_000.   # 5%
    }
    
    training_texts = []
    
    for lang in langs:
        logging.info(f"Sampling raw data for {lang}...")
        raw_ds = load_from_disk(f"data/raw/{lang}_dataset")
        
        # 打亂資料確保詞彙多樣性
        shuffled_ds = raw_ds['train'].shuffle(seed=42)
        
        accumulated_tokens = 0
        for item in shuffled_ds:
            text = item['text']
            
            # 1. 套用清洗規則，過濾掉雜訊
            if not deep_quality_filter(item, lang):
                continue
                
            # 2. 粗估 Token 數量 (冷啟動階段的代理指標)
            if lang == 'zho':
                estimated_tokens = len(text) # 中文以「字元數」粗估
            else:
                estimated_tokens = len(text.split()) # 英荷以「空白分隔單字數」粗估
                
            training_texts.append(text)
            accumulated_tokens += estimated_tokens
            
            # 3. 達到該語言的分配預算即停止抽取
            if accumulated_tokens >= budgets[lang]:
                logging.info(f"[{lang}] Gathered ~{accumulated_tokens:,} heuristic tokens.")
                break
                
    return training_texts

def batch_iterator(texts, batch_size=10000):
    """防止 OOM 的 Generator"""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]

def train_custom_tokenizer():
    # 1. 取得符合 10M 限制的粗估訓練語料
    texts = get_rough_10m_corpus()
    logging.info(f"Loaded training corpus: {len(texts):,} docs")

    # 2. Initialize Byte-Level BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Setup training parameters 
    # 【優化點】：下修 vocab_size 至 24,000，避免 10M 預算下的 Embedding 稀疏問題
    zh_punctuation = [
        "，", "。", "、", "！", "？", "：", "；", 
        "「", "」", "（", "）", "《", "》", "【", "】"
    ]
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + zh_punctuation
    
    trainer = trainers.BpeTrainer(
        vocab_size=16000, 
        #vocab_size=32000, 
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 4. Start Training
    logging.info("Training BPE Tokenizer (may take a few mins)...")
    tokenizer.train_from_iterator(batch_iterator(texts), trainer=trainer)

    # 5. Add Post-Processor for BOS/EOS tags
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 6. Save Model
    os.makedirs("tokenizers", exist_ok=True)
    save_path = "tokenizers/tokenizer_10M.json"
    tokenizer.save(save_path)
    logging.info(f"✅ Tokenizer trained and saved to: {save_path}")

if __name__ == "__main__":
    train_custom_tokenizer()