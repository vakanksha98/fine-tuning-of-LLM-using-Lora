# ============================================================
# main.py  <-  ENTRY POINT
#
# Runs the full pipeline:
#   1. Load & prepare datasets (stratified split + weighted sampler)
#   2. Evaluate BASE GPT-2 (before fine-tuning)
#   3. Build LoRA model
#   4. Train
#   5. Evaluate fine-tuned model (after fine-tuning)
#   6. Print before vs after report
#
# Run: python main.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import math
import yaml
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from peft import PeftModel, PeftConfig

from data.data_loader import prepare_datasets
from model.model_setup import build_lora_model, print_trainable_parameters
from training.trainer import run_training
from evaluate import evaluate_sentiment_token_loss, print_report


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  GPT-2 LoRA Fine-Tuning — Finance Sentiment")
    print("=" * 60)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # -- STEP 1: Data --------------------------------------
    print("\n[1/4] Preparing datasets...")
    train, val, test, tokenizer, test_texts, test_labels, label_map, train_sampler = prepare_datasets()
    print(f"Train:{len(train)} | Val:{len(val)} | Test:{len(test)}")

    # -- STEP 2: Base GPT-2 Evaluation (BEFORE) ------------
    print("\n[2/4] Evaluating BASE GPT-2 (before fine-tuning)...")
    base_model                    = GPT2LMHeadModel.from_pretrained(config["model"]["base_model"])
    base_loss, base_ppl, base_acc = evaluate_sentiment_token_loss(
        base_model, test_texts, test_labels, tokenizer, label_map, device, "Base GPT-2"
    )
    print(f"Base → Loss:{base_loss} | PPL:{base_ppl} | Acc:{base_acc}%")
    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # -- STEP 3: Build + Train LoRA ------------------------
    print("\n[3/4] Building LoRA model & training...")
    model = build_lora_model()
    print_trainable_parameters(model)
    train_result, eval_results = run_training(model, train, val, tokenizer)

    loss       = eval_results["eval_loss"]
    perplexity = math.exp(loss)
    print(f"\nTraining Loss : {train_result.training_loss:.4f}")
    print(f"Val Loss      : {loss:.4f}")
    print(f"Perplexity    : {perplexity:.2f}")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    # -- STEP 4: Fine-Tuned Evaluation (AFTER) ------------─
    print("\n[4/4] Evaluating FINE-TUNED LoRA (after training)...")
    model_path  = os.path.join(config["training"]["output_dir"], "final_model")
    peft_config = PeftConfig.from_pretrained(model_path)
    ft_base     = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
    ft_model    = PeftModel.from_pretrained(ft_base, model_path)
    ft_model    = ft_model.merge_and_unload()
    ft_loss, ft_ppl, ft_acc = evaluate_sentiment_token_loss(
        ft_model, test_texts, test_labels, tokenizer, label_map, device, "Fine-Tuned LoRA"
    )
    print(f"LoRA → Loss:{ft_loss} | PPL:{ft_ppl} | Acc:{ft_acc}%")

    # -- FINAL REPORT --------------------------------------
    print_report(base_loss, base_ppl, base_acc, ft_loss, ft_ppl, ft_acc)

    print("\nDone! Run inference:")
    print('  python inference/generate.py --text "$AAPL beats earnings"')


if __name__ == "__main__":
    main()
