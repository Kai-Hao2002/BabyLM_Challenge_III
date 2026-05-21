import argparse
import yaml
import logging
import os
import random
import numpy as np
import torch

# =====================================================================
# Custom module imports (based on src/ directory structure)
# Note: These modules will need to be implemented by you and your team
# =====================================================================
#from src.data_pipeline.clean_and_mix import prepare_stage_data
#from src.data_pipeline.train_tokenizer import train_custom_tokenizer

from src.model.architecture import build_model
from src.training.dataset import (
    get_baseline_dataloaders,
    get_baseline_chunked_dataloaders,
    get_baseline_packed_dataloaders,
    get_curriculum_dataloaders,
)
from src.training.trainer import train_model, train_model_curriculum
from transformers import PreTrainedTokenizerFast

def setup_logging(log_file):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 1.Parse command-line arguments
    parser = argparse.ArgumentParser(description="BabyLM 2026 Multilingual Training Entry Point")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML config file (e.g., configs/experiment_10M.yaml)"
    )
    args = parser.parse_args()

    # 2.Load YAML configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Basic Setup (Logging & Random Seed)
    os.makedirs(config['output_dir'], exist_ok=True)
    setup_logging(os.path.join(config['output_dir'], 'training.log'))
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loaded configuration from {args.config}")
    set_seed(config.get('seed', 42))

    # =====================================================================
    # Execution Pipeline
    # The following code is commented out. Uncomment it once src/ modules are ready.
    # The following code is commented out. Uncomment it once src/ modules are ready.
    # =====================================================================

    try:
        # --- Step 1: Data Preparation & Tokenizer (Member A) ---
        # logger.info("Step 1: Preparing data and tokenizer...")
        # data_path = prepare_data(config['data_args'])
        # tokenizer = load_or_train_tokenizer(config['tokenizer_args'], data_path)
        logger.info("Step 1: Loading tokenizer...")

        tokenizer_path = config["tokenizer_path"]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"pad: {tokenizer.pad_token} {tokenizer.pad_token_id}")
        logger.info(f"bos: {tokenizer.bos_token} {tokenizer.bos_token_id}")
        logger.info(f"eos: {tokenizer.eos_token} {tokenizer.eos_token_id}")
        logger.info(f"mask: {tokenizer.mask_token} {tokenizer.mask_token_id}")
        # --- Step 2: Model Initialization (Member B) ---
        # logger.info("Step 2: Initializing model...")
        # model = build_model(config['model_args'], vocab_size=tokenizer.vocab_size)
        logger.info("Step 2: Building model...")

        model = build_model(
            vocab_size=tokenizer.vocab_size,
            config_args=config["model_args"],
        )
        
        # --- Step 3: Dataset & DataLoader (Member B) ---
        # --- Step 4: Training Loop (Member B) ---
        mode = config.get("mode", "baseline")

        if mode == "baseline":
            logger.info("Step 3: Setting up baseline dataloaders...")

            use_packing = config["data_args"].get("use_packing", False)
            use_chunking = config["data_args"].get("use_chunking", False)

            if use_packing:
                logger.info("Using packed baseline dataloaders...")

                train_loader, val_loader = get_baseline_packed_dataloaders(
                    train_path=config["data_args"]["train_path"],
                    val_path=config["data_args"]["val_path"],
                    tokenizer_path=tokenizer_path,
                    batch_size=config["training_args"]["batch_size"],
                    max_length=config["training_args"]["max_length"],
                    mlm_probability=config["training_args"].get("mlm_probability", 0.15),
                    packing_strategy=config["data_args"].get("packing_strategy", "wrapped"),
                    objective=config["data_args"].get("objective", "mlm"),
                    insert_eos=config["data_args"].get("insert_eos", False),
                )
            elif use_chunking:
                logger.info("Using chunked baseline dataloaders...")

                train_loader, val_loader = get_baseline_chunked_dataloaders(
                    train_path=config["data_args"]["train_path"],
                    val_path=config["data_args"]["val_path"],
                    tokenizer_path=tokenizer_path,
                    batch_size=config["training_args"]["batch_size"],
                    max_length=config["training_args"]["max_length"],
                    mlm_probability=config["training_args"].get("mlm_probability", 0.15),
                )
            else:
                logger.info("Using non-chunked baseline dataloaders...")

                train_loader, val_loader = get_baseline_dataloaders(
                    train_path=config["data_args"]["train_path"],
                    val_path=config["data_args"]["val_path"],
                    tokenizer_path=tokenizer_path,
                    batch_size=config["training_args"]["batch_size"],
                    max_length=config["training_args"]["max_length"],
                    mlm_probability=config["training_args"].get("mlm_probability", 0.15),
                )

            logger.info(f"Train batches: {len(train_loader)}")
            logger.info(f"Val batches: {len(val_loader)}")

            logger.info("Step 4: Starting baseline training...")

            config["training_args"]["output_dir"] = config["output_dir"]

            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                config=config["training_args"],
            )

        elif mode == "curriculum":
            logger.info("Step 3: Setting up curriculum dataloaders...")

            stage_loaders = get_curriculum_dataloaders(
                curriculum_stages=config["data_args"]["curriculum_stages"],
                tokenizer_path=tokenizer_path,
                batch_size=config["training_args"]["batch_size"],
                max_length=config["training_args"]["max_length"],
                mlm_probability=config["training_args"].get("mlm_probability", 0.15),
                val_ratio=config["data_args"].get("val_ratio", 0.1),
                seed=config.get("seed", 42),
            )

            for stage in stage_loaders:
                logger.info(
                    f"{stage['name']} | "
                    f"Train batches: {len(stage['train_loader'])} | "
                    f"Val batches: {len(stage['val_loader'])}"
                )

            logger.info("Step 4: Starting curriculum training...")
        # Make trainer save logs/checkpoints to the experiment-specific folder
            config["training_args"]["output_dir"] = config["output_dir"]
            
            train_model_curriculum(
                model=model,
                stage_loaders=stage_loaders,
                tokenizer=tokenizer,
                config=config["training_args"],
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
