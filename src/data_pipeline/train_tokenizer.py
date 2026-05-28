import os
import logging
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

from clean_and_mix import normalize_text, tag_aligned_corpus, deep_quality_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_rough_10m_corpus():
    """
    Step 1: Get a Rough 10M Token Corpus
    This function samples from the raw datasets of the three languages (English, Chinese, Dutch) to create a training corpus that roughly meets the 10M token budget. 
    The sampling strategy is designed to balance the languages while also applying a deep quality filter to ensure that the training data is as clean and useful as possible for tokenizer training.
    The function returns a list of text samples that can be used to train the custom tokenizer.
    """
    langs = ['eng', 'zho', 'nld']
    """
    # 100M budget tokenizer(Only 15M tokens 3% for Chinese, 80% for English, 17% for Dutch)
    budgets = {
        'eng': 12_000_000,  # 80% 
        'nld': 2_550_000,   # 17% 
        'zho': 450_000      # 3%  
    }

    """
    # 10M budget tokenizer(Total 10M tokens 5% for Chinese, 75% for English, 20% for Dutch)
    budgets = {
        'eng': 7_500_000, # 75%
        'nld': 2_000_000, # 20%
        'zho': 500_000    # 5%
    }  
    training_texts = []
    
    for lang in langs:
        logging.info(f"Sampling clean data from split train pool for {lang}...")
        
        
        train_pool_path = f"data/train_pool/{lang}"
        if not os.path.exists(train_pool_path):
            raise FileNotFoundError(
                f"❌ Cannot find train pool: {train_pool_path}\n"
                f"Please run clean_and_mix.py first to generate the train pool!"
            )
            
        # load the pre-cleaned and pre-mixed train pool for the language
        train_pool_ds = load_from_disk(train_pool_path)
        
        accumulated_tokens = 0
        
        for item in train_pool_ds:
            text = item['text']
            
            if lang == 'zho':
                estimated_tokens = len(text) 
            else:
                estimated_tokens = len(text.split()) 
                
            training_texts.append(text)
            accumulated_tokens += estimated_tokens
            
            if accumulated_tokens >= budgets[lang]:
                logging.info(f"[{lang}] Gathered ~{accumulated_tokens:,} heuristic tokens from safe train pool.")
                break
                
    return training_texts

def batch_iterator(texts, batch_size=10000):
    """防止 OOM 的 Generator"""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]

def train_custom_tokenizer():
    # 1. get a rough 10M token corpus for training the tokenizer
    texts = get_rough_10m_corpus()
    logging.info(f"Loaded training corpus: {len(texts):,} docs")

    # 2. Initialize Byte-Level BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Setup training parameters 
    zh_punctuation = [
        "，", "。", "、", "！", "？", "：", "；", 
        "「", "」", "（", "）", "《", "》", "【", "】"
    ]
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + zh_punctuation
    
    trainer = trainers.BpeTrainer(
        vocab_size=16000, #for 100M
        #vocab_size=30000, # for 10M
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    # 4. Start Training
    logging.info("Training BPE Tokenizer (may take a few mins)...")
    tokenizer.train_from_iterator(batch_iterator(texts), trainer=trainer)

    # 5. Add Post-Processor for BOS/EOS tags
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # 6. Save Model
    os.makedirs("tokenizers", exist_ok=True)
    save_path = "tokenizers/tokenizer_10M_16k.json"
    tokenizer.save(save_path)
    logging.info(f"✅ Tokenizer trained and saved to: {save_path}")

if __name__ == "__main__":
    train_custom_tokenizer()