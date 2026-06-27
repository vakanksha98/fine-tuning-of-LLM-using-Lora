# ============================================================
# inference/generate.py
#
# Load fine-tuned LoRA model and predict sentiment.
#
# Run:
#   python inference/generate.py --text "$AAPL beats earnings expectations"
# ============================================================

import torch
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel, PeftConfig


def load_model(model_path="./outputs/final_model"):
    """Load fine-tuned LoRA model and merge adapters."""
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model  = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
    model       = PeftModel.from_pretrained(base_model, model_path)
    model       = model.merge_and_unload()
    model.eval()

    tokenizer             = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token   = tokenizer.eos_token
    return model, tokenizer


def predict_sentiment(text, model, tokenizer, device="cpu"):
    """
    Predict sentiment of a financial news sentence.

    Args:
        text     : Financial news text
        model    : Fine-tuned model
        tokenizer: GPT-2 tokenizer
        device   : "cuda" or "cpu"

    Returns:
        predicted sentiment string + confidence scores
    """
    model.to(device)

    prompt = f"Sentence: {text.strip()} Sentiment:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=120).to(device)

    sentiment_tokens = {
        "negative" : tokenizer.encode(" negative")[0],
        "positive" : tokenizer.encode(" positive")[0],
        "neutral"  : tokenizer.encode(" neutral")[0],
    }

    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits[0, -1, :]

        # Get probabilities for sentiment tokens only
        sent_ids    = list(sentiment_tokens.values())
        sent_logits = logits[sent_ids]
        probs       = torch.nn.functional.softmax(sent_logits, dim=-1)

        pred_idx   = probs.argmax().item()
        pred_label = list(sentiment_tokens.keys())[pred_idx]
        confidence = {k: round(probs[i].item() * 100, 2) for i, k in enumerate(sentiment_tokens.keys())}

    return pred_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       type=str, default="The stock market rallied strongly today")
    parser.add_argument("--model_path", type=str, default="./outputs/final_model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)

    label, confidence = predict_sentiment(args.text, model, tokenizer, device)

    print("\n" + "="*55)
    print(f"Text      : {args.text}")
    print(f"Sentiment : {label.upper()}")
    print(f"Confidence: {confidence}")
    print("="*55)
