# architecture.py → 負責建立 BERT-style encoder-only model (Masked Language Modeling, MLM)
from transformers import BertConfig, BertForMaskedLM


def build_model(vocab_size, config_args):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=config_args.get("hidden_size", 256),
        num_hidden_layers=config_args.get("num_hidden_layers", 4),
        num_attention_heads=config_args.get("num_attention_heads", 4),
        intermediate_size=config_args.get("intermediate_size", 1024),
        max_position_embeddings=config_args.get("max_position_embeddings", 512),
        pad_token_id=config_args.get("pad_token_id", 3),
        type_vocab_size=1,
    )

    return BertForMaskedLM(config)
