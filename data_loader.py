# ============================================================
# data/data_loader.py
#
# Loads twitter-financial-news-sentiment dataset from HuggingFace.
# Applies:
#   1. Stratified Train/Val/Test Split (80/10/10)
#   2. Generative Prompt Format conversion
#   3. Weighted Random Sampler for training (handles class imbalance)
#
# Dataset Distribution:
#   Neutral  (2): 64.7%  <- majority
#   Positive (1): 20.2%
#   Negative (0): 15.1%  <- minority
# ============================================================

from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FinanceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels_list    = labels   # int labels for weighted sampler

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids"     : torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels"        : torch.tensor(self.input_ids[idx], dtype=torch.long),
        }


def get_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def convert_to_prompt_format(texts, labels, label_map):
    """
    Generative prompt format:

    Input  : sentence + label
    Output : "Sentence: <text> Sentiment: <positive/negative/neutral>"

    WHY THIS FORMAT?
    GPT-2 is autoregressive — it predicts next token.
    During inference:
      Input  → "Sentence: $BYND JPMorgan... Sentiment:"
      Output → "positive" / "negative" / "neutral"

    This teaches GPT-2 BOTH finance language AND sentiment task.
    """
    prompts = []
    for text, label in zip(texts, labels):
        sentiment = label_map[int(label)]
        prompt    = f"Sentence: {text.strip()} Sentiment: {sentiment}"
        prompts.append(prompt)
    return prompts


def get_weighted_sampler(dataset):
    """
    Weighted Random Sampler — fixes class imbalance during training.

    HOW IT WORKS:
      Class weight = 1 / class_frequency
      Negative (15%) → weight = 6.67  <- sampled more often
      Positive (20%) → weight = 5.00
      Neutral  (65%) → weight = 1.54  <- sampled less often

    Result: Model sees all 3 classes roughly equally during training.
    Applied ONLY on training set — val/test keep original distribution.
    """
    labels        = dataset.labels_list
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(
        [class_weights[label] for label in labels],
        dtype=torch.float
    )
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


def prepare_datasets(config_path="configs/config.yaml"):
    """
    Full data pipeline:
    1. Load dataset from HuggingFace
    2. Stratified split → train(80%) / val(10%) / test(10%)
    3. Convert to prompt format
    4. Tokenize
    5. Create WeightedRandomSampler for training

    Returns:
        train_dataset, val_dataset, test_dataset,
        tokenizer, test_texts, test_labels,
        label_map, train_sampler
    """
    config    = load_config(config_path)
    model_cfg = config["model"]

    label_map = {
        0: "negative",
        1: "positive",
        2: "neutral",
    }

    logger.info("Loading twitter-financial-news-sentiment...")
    raw_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    raw_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")

    # Combine and convert to lists
    all_texts  = list(raw_train["text"])  + list(raw_val["text"])
    all_labels = list(raw_train["label"]) + list(raw_val["label"])
    logger.info(f"Total samples: {len(all_texts)}")

    # -- STRATIFIED SPLIT ----------------------------------
    # Ensures each split has same class distribution as full dataset
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels,
        test_size    = 0.20,
        random_state = 42,
        stratify     = all_labels,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size    = 0.50,
        random_state = 42,
        stratify     = temp_labels,
    )

    # -- VERIFY DISTRIBUTION ------------------------------─
    for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        c = Counter(lbls)
        t = len(lbls)
        logger.info(
            f"{name} ({t}) → "
            f"Neg:{c[0]/t*100:.1f}% "
            f"Pos:{c[1]/t*100:.1f}% "
            f"Neu:{c[2]/t*100:.1f}%"
        )

    # -- PROMPT FORMAT ------------------------------------─
    train_prompts = convert_to_prompt_format(train_texts, train_labels, label_map)
    val_prompts   = convert_to_prompt_format(val_texts,   val_labels,   label_map)
    test_prompts  = convert_to_prompt_format(test_texts,  test_labels,  label_map)

    # -- TOKENIZE ------------------------------------------
    tokenizer = get_tokenizer(model_cfg["base_model"])

    def tokenize(text_list):
        return tokenizer(
            text_list,
            truncation     = True,
            padding        = "max_length",
            max_length     = model_cfg["max_length"],
            return_tensors = None,
        )

    train_dataset = FinanceDataset(tokenize(train_prompts), train_labels)
    val_dataset   = FinanceDataset(tokenize(val_prompts),   val_labels)
    test_dataset  = FinanceDataset(tokenize(test_prompts),  test_labels)

    # -- WEIGHTED SAMPLER — train only --------------------─
    train_sampler = get_weighted_sampler(train_dataset)

    return (
        train_dataset, val_dataset, test_dataset,
        tokenizer, test_prompts, test_labels,
        label_map, train_sampler
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train, val, test, tokenizer, test_texts, test_labels, label_map, sampler = prepare_datasets()
    print(f"Train:{len(train)} | Val:{len(val)} | Test:{len(test)}")
    print(f"Sample: {test_texts[0]}")
