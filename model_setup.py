# ============================================================
# model/model_setup.py
#
# Loads GPT-2 and applies LoRA adapters using PEFT library.
#
# WHY LORA?
#   Full fine-tuning = update all 124M parameters → expensive
#   LoRA = freeze base model + add small matrices A and B
#   W_new = W_original + (B x A)
#   Only A and B trained → 0.23% parameters → same quality!
#
# Results achieved:
#   Trainable params : 294,912
#   Total params     : 124,734,720
#   Trainable %      : 0.2364%
# ============================================================

from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_trainable_parameters(model):
    """Print trainable vs total parameters — key LoRA efficiency metric."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,}")
    print(f"Total params     : {total:,}")
    print(f"Trainable %      : {100 * trainable / total:.4f}%")
    return trainable, total


def build_lora_model(config_path="configs/config.yaml"):
    """
    Build GPT-2 + LoRA model.

    Steps:
    1. Load base GPT-2 (124M params, all frozen)
    2. Define LoRA config
    3. Apply LoRA via PEFT — injects trainable A & B matrices
       into attention layers (c_attn)
    4. Return model ready for training

    target_modules = ["c_attn"]:
    GPT-2 attention projection layer — most important for
    language understanding. LoRA applied here gives best results.
    """
    config    = load_config(config_path)
    model_cfg = config["model"]
    lora_cfg  = config["lora"]

    logger.info(f"Loading base model: {model_cfg['base_model']}")
    base_model = GPT2LMHeadModel.from_pretrained(model_cfg["base_model"])

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["lora_alpha"],
        lora_dropout   = lora_cfg["lora_dropout"],
        target_modules = lora_cfg["target_modules"],
        bias           = lora_cfg["bias"],
    )

    model = get_peft_model(base_model, lora_config)
    logger.info("LoRA model built successfully.")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = build_lora_model()
    print_trainable_parameters(model)
