# ============================================================
# training/trainer.py
#
# Training pipeline using HuggingFace Trainer API.
# Handles: optimizer, scheduler, evaluation, checkpointing.
#
# Key design choices:
#   - eval_strategy = "steps" → evaluate every N steps
#   - load_best_model_at_end  → auto saves best checkpoint
#   - fp16 auto-detected      → uses GPU mixed precision if available
#   - mlm=False               → GPT-2 is causal LM, not masked LM
# ============================================================

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import yaml
import os
import logging
import torch

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(model, train_dataset, val_dataset, tokenizer, config_path="configs/config.yaml"):
    """
    Full training pipeline.

    Args:
        model        : PEFT LoRA model from model_setup.py
        train_dataset: FinanceDataset (train split)
        val_dataset  : FinanceDataset (val split)
        tokenizer    : GPT-2 tokenizer
        config_path  : Path to config.yaml

    Returns:
        train_result, eval_results
    """
    config    = load_config(config_path)
    train_cfg = config["training"]

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    # Auto-detect fp16 — only use if GPU available
    use_fp16 = train_cfg["fp16"] and torch.cuda.is_available()
    if train_cfg["fp16"] and not torch.cuda.is_available():
        logger.warning("fp16=true in config but no GPU. Disabling fp16.")

    training_args = TrainingArguments(
        output_dir                  = train_cfg["output_dir"],
        num_train_epochs            = train_cfg["num_train_epochs"],
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size  = train_cfg["per_device_eval_batch_size"],
        learning_rate               = train_cfg["learning_rate"],
        warmup_steps                = train_cfg["warmup_steps"],
        weight_decay                = train_cfg["weight_decay"],
        logging_steps               = train_cfg["logging_steps"],
        save_steps                  = train_cfg["save_steps"],
        eval_strategy               = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        fp16                        = use_fp16,
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        seed                        = train_cfg["seed"],
        report_to                   = "none",
    )

    # mlm=False: GPT-2 is autoregressive (causal), not masked
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm       = False,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = tokenizer,
        data_collator    = data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    final_path = os.path.join(train_cfg["output_dir"], "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Model saved to: {final_path}")

    eval_results = trainer.evaluate()
    return train_result, eval_results
