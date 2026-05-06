    # trainer.py → 負責訓練、validation、token exposure logging、checkpoint
    # 1. forward
    # 2. loss
    # 3. backward
    # 4. optimizer step
    # 5. 用 attention_mask 計算 batch token 數
    # 6. 根據 language 分別累加 EN / NL / ZH raw tokens
    # 7. 套用 Byte Premium
    # 8. 記錄 CSV
    # 9. 如果達到 token milestone，存 checkpoint
    # 10. 如果超過 exposure limit，停止
import os
import csv
import math
import logging
import torch
from torch.optim import AdamW
from tqdm import tqdm


BYTE_PREMIUM = {
    "eng": 1.0,
    "nld": 1.0516,
    "zho": 0.9894,
}


class TokenExposureTracker:
    def __init__(self):
        self.raw_seen_tokens_eng = 0
        self.raw_seen_tokens_nld = 0
        self.raw_seen_tokens_zho = 0

    def update(self, attention_mask, languages):
        """
        attention_mask: Tensor [batch_size, seq_len]
        languages: list[str], e.g. ["eng", "nld", "zho"]
        """
        tokens_per_sample = attention_mask.sum(dim=1).detach().cpu().tolist()

        for tokens, lang in zip(tokens_per_sample, languages):
            tokens = int(tokens)

            if lang == "eng":
                self.raw_seen_tokens_eng += tokens
            elif lang == "nld":
                self.raw_seen_tokens_nld += tokens
            elif lang == "zho":
                self.raw_seen_tokens_zho += tokens
            else:
                raise ValueError(f"Unknown language label: {lang}")

    @property
    def raw_seen_tokens_total(self):
        return (
            self.raw_seen_tokens_eng
            + self.raw_seen_tokens_nld
            + self.raw_seen_tokens_zho
        )

    @property
    def adjusted_seen_tokens_eng(self):
        return self.raw_seen_tokens_eng * BYTE_PREMIUM["eng"]

    @property
    def adjusted_seen_tokens_nld(self):
        return self.raw_seen_tokens_nld * BYTE_PREMIUM["nld"]

    @property
    def adjusted_seen_tokens_zho(self):
        return self.raw_seen_tokens_zho * BYTE_PREMIUM["zho"]

    @property
    def adjusted_seen_tokens_total(self):
        return (
            self.adjusted_seen_tokens_eng
            + self.adjusted_seen_tokens_nld
            + self.adjusted_seen_tokens_zho
        )


def get_next_checkpoint_target(current_adjusted_tokens):
    """
    Official-style checkpoint schedule:
    0–10M: every 1M
    10M–100M: every 10M
    100M–1B: every 100M
    """
    if current_adjusted_tokens < 10_000_000:
        interval = 1_000_000
    elif current_adjusted_tokens < 100_000_000:
        interval = 10_000_000
    else:
        interval = 100_000_000

    return math.ceil((current_adjusted_tokens + 1) / interval) * interval


def format_checkpoint_name(token_count):
    if token_count < 1_000_000:
        return f"step_{int(token_count)}"
    if token_count < 1_000_000_000:
        return f"step_{int(token_count / 1_000_000)}M"
    return f"step_{int(token_count / 1_000_000_000)}B"


def save_checkpoint(model, tokenizer, output_dir, checkpoint_name):
    checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)

    return checkpoint_path


def evaluate(model, val_loader, device, max_val_steps=None):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            languages = batch.pop("language")

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            steps += 1

            if max_val_steps is not None and steps >= max_val_steps:
                break

    model.train()

    if steps == 0:
        return None

    return total_loss / steps


def train_model(
    model,
    train_loader,
    val_loader,
    tokenizer,
    config,
):
    logger = logging.getLogger(__name__)

    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.csv")

    learning_rate = float(config.get("learning_rate", 5e-4))
    max_epochs = int(config.get("max_epochs", 1))
    max_steps = config.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None
    max_val_steps = int(config.get("max_val_steps", 20))
    max_adjusted_token_exposure = float(
        config.get("max_adjusted_token_exposure", 1_000_000_000)
    )

    if max_epochs > 10:
        raise ValueError("BabyLM rule: max_epochs must be <= 10")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    tracker = TokenExposureTracker()

    global_step = 0
    next_checkpoint_target = get_next_checkpoint_target(
        tracker.adjusted_seen_tokens_total
    )

    fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "validation_loss",
        "raw_seen_tokens_total",
        "raw_seen_tokens_eng",
        "raw_seen_tokens_nld",
        "raw_seen_tokens_zho",
        "adjusted_seen_tokens_eng",
        "adjusted_seen_tokens_nld",
        "adjusted_seen_tokens_zho",
        "adjusted_seen_tokens_total",
        "checkpoint_path",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, max_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{max_epochs}")

            progress = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch in progress:
                languages = batch.pop("language")

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                optimizer.zero_grad()

                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                global_step += 1

                # Count raw seen tokens by language
                tracker.update(
                    attention_mask=batch["attention_mask"],
                    languages=languages,
                )

                adjusted_total = tracker.adjusted_seen_tokens_total

                checkpoint_path = ""

                # Save checkpoint by adjusted token milestones
                if adjusted_total >= next_checkpoint_target:
                    checkpoint_name = format_checkpoint_name(next_checkpoint_target)
                    checkpoint_path = save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        checkpoint_name=checkpoint_name,
                    )

                    logger.info(
                        f"Saved checkpoint at adjusted token target "
                        f"{next_checkpoint_target:,}: {checkpoint_path}"
                    )

                    next_checkpoint_target = get_next_checkpoint_target(adjusted_total)

                # Validation
                validation_loss = ""
                if val_loader is not None and global_step % config.get("eval_every_steps", 20) == 0:
                    validation_loss_value = evaluate(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        max_val_steps=max_val_steps,
                    )
                    validation_loss = (
                        f"{validation_loss_value:.6f}"
                        if validation_loss_value is not None
                        else ""
                    )

                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": f"{loss.item():.6f}",
                    "validation_loss": validation_loss,
                    "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                    "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                    "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                    "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                    "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                    "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                    "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                    "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                    "checkpoint_path": checkpoint_path,
                }

                writer.writerow(row)
                f.flush()

                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "adj_tokens": int(tracker.adjusted_seen_tokens_total),
                })

                # Stop if exposure limit is reached
                if tracker.adjusted_seen_tokens_total >= max_adjusted_token_exposure:
                    logger.info(
                        f"Stopping: reached adjusted token exposure limit "
                        f"{max_adjusted_token_exposure:,}"
                    )
                    final_path = save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        checkpoint_name="final_model",
                    )
                    logger.info(f"Final checkpoint saved to {final_path}")
                    return

                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Stopping: reached max_steps={max_steps}")
                    final_path = save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        checkpoint_name="final_model",
                    )
                    logger.info(f"Final checkpoint saved to {final_path}")
                    return

    final_path = save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        checkpoint_name="final_model",
    )

    logger.info("Training finished.")
    logger.info(f"Final checkpoint saved to {final_path}")
    logger.info(f"Training log saved to {log_path}")

def train_model_curriculum(
    model,
    stage_loaders,
    tokenizer,
    config,
):
    logger = logging.getLogger(__name__)

    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.csv")

    learning_rate = float(config.get("learning_rate", 5e-4))
    max_epochs = int(config.get("max_epochs", 1))

    max_steps = config.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    max_val_steps = int(config.get("max_val_steps", 20))
    eval_every_steps = int(config.get("eval_every_steps", 10))

    max_adjusted_token_exposure = float(
        config.get("max_adjusted_token_exposure", 1_000_000_000)
    )

    if max_epochs > 10:
        raise ValueError("BabyLM rule: max_epochs must be <= 10")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    tracker = TokenExposureTracker()
    global_step = 0
    next_checkpoint_target = get_next_checkpoint_target(
        tracker.adjusted_seen_tokens_total
    )

    fieldnames = [
        "epoch",
        "stage",
        "global_step",
        "train_loss",
        "validation_loss",
        "raw_seen_tokens_total",
        "raw_seen_tokens_eng",
        "raw_seen_tokens_nld",
        "raw_seen_tokens_zho",
        "adjusted_seen_tokens_eng",
        "adjusted_seen_tokens_nld",
        "adjusted_seen_tokens_zho",
        "adjusted_seen_tokens_total",
        "checkpoint_path",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, max_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{max_epochs}")

            for stage in stage_loaders:
                stage_name = stage["name"]
                train_loader = stage["train_loader"]
                val_loader = stage["val_loader"]

                logger.info(f"Starting stage: {stage_name}")

                progress = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch} | {stage_name}",
                )

                for batch in progress:
                    languages = batch.pop("language")

                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                    }

                    optimizer.zero_grad()

                    outputs = model(**batch)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()

                    global_step += 1

                    tracker.update(
                        attention_mask=batch["attention_mask"],
                        languages=languages,
                    )

                    adjusted_total = tracker.adjusted_seen_tokens_total
                    checkpoint_path = ""

                    if adjusted_total >= next_checkpoint_target:
                        checkpoint_name = format_checkpoint_name(
                            next_checkpoint_target
                        )
                        checkpoint_path = save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=output_dir,
                            checkpoint_name=checkpoint_name,
                        )

                        logger.info(
                            f"Saved checkpoint at adjusted token target "
                            f"{next_checkpoint_target:,}: {checkpoint_path}"
                        )

                        next_checkpoint_target = get_next_checkpoint_target(
                            adjusted_total
                        )

                    validation_loss = ""
                    if val_loader is not None and global_step % eval_every_steps == 0:
                        validation_loss_value = evaluate(
                            model=model,
                            val_loader=val_loader,
                            device=device,
                            max_val_steps=max_val_steps,
                        )
                        validation_loss = (
                            f"{validation_loss_value:.6f}"
                            if validation_loss_value is not None
                            else ""
                        )

                    row = {
                        "epoch": epoch,
                        "stage": stage_name,
                        "global_step": global_step,
                        "train_loss": f"{loss.item():.6f}",
                        "validation_loss": validation_loss,
                        "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                        "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                        "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                        "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                        "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                        "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                        "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                        "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                        "checkpoint_path": checkpoint_path,
                    }

                    writer.writerow(row)
                    f.flush()

                    progress.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "adj_tokens": int(tracker.adjusted_seen_tokens_total),
                    })

                    if tracker.adjusted_seen_tokens_total >= max_adjusted_token_exposure:
                        logger.info(
                            f"Stopping: reached adjusted token exposure limit "
                            f"{max_adjusted_token_exposure:,}"
                        )
                        final_path = save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=output_dir,
                            checkpoint_name="final_model",
                        )
                        logger.info(f"Final checkpoint saved to {final_path}")
                        return

                    if max_steps is not None and global_step >= max_steps:
                        logger.info(f"Stopping: reached max_steps={max_steps}")
                        final_path = save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=output_dir,
                            checkpoint_name="final_model",
                        )
                        logger.info(f"Final checkpoint saved to {final_path}")
                        return

    final_path = save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        checkpoint_name="final_model",
    )

    logger.info("Curriculum training finished.")
    logger.info(f"Final checkpoint saved to {final_path}")
    logger.info(f"Training log saved to {log_path}")