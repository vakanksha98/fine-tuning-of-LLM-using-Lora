# ============================================================
# evaluate.py
#
# Compares GPT-2 performance BEFORE and AFTER LoRA fine-tuning.
#
# Evaluation Method: Sentiment Token Loss
#   - Only the sentiment token (positive/negative/neutral)
#     loss is measured — not the full sequence
#   - This is fair for both base and fine-tuned models
#   - Accuracy = did model predict correct sentiment token?
#
# Results achieved:
#   Metric          Base GPT-2    Fine-Tuned LoRA
#   Loss        :   14.97     →   4.8959   (67% drop)
#   Perplexity  :   3,172,524 →   133.74
#   Accuracy    :   20.52%    →   66.75%   (+46% gain)
#
# Run: python evaluate.py
# ============================================================

import torch
import math
import os
from transformers import GPT2LMHeadModel
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_sentiment_token_loss(
    model, test_texts, test_labels,
    tokenizer, label_map, device, model_name="Model"
):
    """
    Sentiment token level evaluation.

    WHY SENTIMENT TOKEN ONLY?
    Full sequence loss is unfair — base GPT-2 was trained on
    internet text so common tokens have low loss naturally.
    We only care about: did the model predict the RIGHT sentiment?

    HOW:
    1. Give prompt: "Sentence: <text> Sentiment:"
    2. Get logits at last position
    3. Compare probabilities of "positive"/"negative"/"neutral" tokens
    4. Predicted = highest probability among 3 sentiment tokens
    5. Loss = -log_prob(correct sentiment token)

    Args:
        model      : GPT-2 or fine-tuned LoRA model
        test_texts : List of full prompt strings
        test_labels: List of int labels (0/1/2)
        tokenizer  : GPT-2 tokenizer
        label_map  : {0: "negative", 1: "positive", 2: "neutral"}
        device     : "cuda" or "cpu"
        model_name : Label for logging

    Returns:
        avg_loss, perplexity, accuracy (all floats)
    """
    model.eval()
    model.to(device)

    total_loss       = 0
    correct          = 0
    total            = 0

    # Get token IDs for each sentiment word
    # Note: " positive" (with space) tokenizes correctly in GPT-2
    sentiment_tokens = {
        v: tokenizer.encode(" " + v)[0]
        for v in label_map.values()
    }
    logger.info(f"Sentiment token IDs: {sentiment_tokens}")

    for text, true_label in tqdm(
        zip(test_texts, test_labels),
        total   = len(test_texts),
        desc    = model_name
    ):
        # Remove last word (sentiment) — model must predict it
        prompt    = text.rsplit(" ", 1)[0] + " "
        true_word = label_map[int(true_label)]
        true_tok  = sentiment_tokens[true_word]

        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 120,
        ).to(device)

        with torch.no_grad():
            outputs  = model(**inputs)
            logits   = outputs.logits[0, -1, :]   # Last token position

            # Loss on correct sentiment token
            log_probs  = torch.nn.functional.log_softmax(logits, dim=-1)
            total_loss += -log_probs[true_tok].item()

            # Accuracy — which sentiment has highest prob?
            sent_ids      = list(sentiment_tokens.values())
            sent_logits   = logits[sent_ids]
            pred_idx      = sent_logits.argmax().item()
            pred_word     = list(sentiment_tokens.keys())[pred_idx]

            correct += int(pred_word == true_word)
            total   += 1

    avg_loss   = total_loss / total
    perplexity = math.exp(min(avg_loss, 300))
    accuracy   = (correct / total) * 100

    return round(avg_loss, 4), round(perplexity, 2), round(accuracy, 2)


def print_report(base_loss, base_ppl, base_acc, ft_loss, ft_ppl, ft_acc, output_dir="./outputs"):
    """Print and save before vs after comparison report."""
    report = f"""
╔========================================================══╗
║         BEFORE vs AFTER — FINAL EVALUATION REPORT       ║
╠============================══╦============═╦============╣
║  Metric                      ║  Base GPT-2 ║  LoRA      ║
╠============================══╬============═╬============╣
║  Sentiment Loss         ↓    ║  {base_loss:<11} ║  {ft_loss:<10} ║
║  Sentiment Perplexity   ↓    ║  {base_ppl:<11} ║  {ft_ppl:<10} ║
║  Sentiment Accuracy %   ↑    ║  {base_acc:<11} ║  {ft_acc:<10} ║
╠============================══╩============═╩============╣
║  Accuracy Gain : {base_acc}% → {ft_acc}% (+{round(ft_acc-base_acc,2)}%)          ║
║  Loss Drop     : {base_loss} → {ft_loss} ({round(base_loss-ft_loss,2)} better)        ║
╚========================================================══╝
"""
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved: {report_path}")


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(level=logging.INFO)
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    from data.data_loader import prepare_datasets
    _, _, test_dataset, tokenizer, test_texts, test_labels, label_map, _ = prepare_datasets()

    # -- BASE GPT-2 ----------------------------------------
    logger.info("Evaluating BASE GPT-2...")
    base_model                    = GPT2LMHeadModel.from_pretrained(config["model"]["base_model"])
    base_loss, base_ppl, base_acc = evaluate_sentiment_token_loss(
        base_model, test_texts, test_labels, tokenizer, label_map, device, "Base GPT-2"
    )
    print(f"Base → Loss:{base_loss} | Perplexity:{base_ppl} | Accuracy:{base_acc}%")
    del base_model
    torch.cuda.empty_cache()

    # -- FINE-TUNED ----------------------------------------
    logger.info("Evaluating Fine-Tuned LoRA model...")
    model_path  = os.path.join(config["training"]["output_dir"], "final_model")
    peft_config = PeftConfig.from_pretrained(model_path)
    ft_base     = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
    ft_model    = PeftModel.from_pretrained(ft_base, model_path)
    ft_model    = ft_model.merge_and_unload()
    ft_loss, ft_ppl, ft_acc = evaluate_sentiment_token_loss(
        ft_model, test_texts, test_labels, tokenizer, label_map, device, "Fine-Tuned LoRA"
    )
    print(f"LoRA → Loss:{ft_loss} | Perplexity:{ft_ppl} | Accuracy:{ft_acc}%")

    # -- REPORT --------------------------------------------
    print_report(base_loss, base_ppl, base_acc, ft_loss, ft_ppl, ft_acc)


if __name__ == "__main__":
    main()
