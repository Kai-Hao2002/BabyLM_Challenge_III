# architecture.py → 負責建立 BERT-style encoder-only model (Masked Language Modeling, MLM)
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel


def build_model(vocab_size, config_args):
    model_type = config_args.get("model_type", "bert").lower()

    if model_type == "bert":
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

    if model_type == "gpt2":
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=config_args.get("n_embd", 256),
            n_layer=config_args.get("n_layer", 4),
            n_head=config_args.get("n_head", 4),
            n_inner=config_args.get("n_inner", 1024),
            n_positions=config_args.get("n_positions", 512),
            n_ctx=config_args.get("n_ctx", config_args.get("n_positions", 512)),
            pad_token_id=config_args.get("pad_token_id", 3),
            bos_token_id=config_args.get("bos_token_id", 1),
            eos_token_id=config_args.get("eos_token_id", 2),
        )

        return GPT2LMHeadModel(config)

    raise ValueError(f"Unknown model_type: {model_type}")
