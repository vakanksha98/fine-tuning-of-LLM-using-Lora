# Fine-Tuning GPT-2 using LoRA

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![LoRA](https://img.shields.io/badge/LoRA-PEFT-FF6B6B)
![License](https://img.shields.io/badge/License-MIT-green)

Parameter-efficient fine-tuning of GPT-2 using LoRA (Low-Rank Adaptation) for custom NLP tasks.

## Overview

This project implements parameter-efficient fine-tuning of GPT-2 using LoRA (Low-Rank Adaptation). The goal is to adapt a pre-trained language model to a custom dataset while significantly reducing training cost and memory usage.

## Key Concept: LoRA

LoRA (Low-Rank Adaptation) is a technique that:
- **Freezes** pre-trained model weights
- **Introduces** small trainable low-rank matrices
- **Updates** only these matrices during training

### Benefits
- Drastically reduces trainable parameters
- Faster training
- Lower GPU memory usage
- Maintains comparable performance

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Framework | PyTorch |
| LLM | HuggingFace Transformers |
| Fine-tuning | PEFT (LoRA) |
| Acceleration | Accelerate, bitsandbytes |

## Project Structure

```
fine-tuning-of-LLM-using-Lora/
├── AKKUFINETUNING1.ipynb           # Main fine-tuning notebook
├── Copy_of_final_evaluation_finetuning.ipynb
├── akdm_final_model.ipynb           # Final model evaluation
├── README.md                         # This file
└── requirements.txt                  # Dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/vakanksha98/fine-tuning-of-LLM-using-Lora.git
cd fine-tuning-of-LLM-using-Lora

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

```
transformers>=4.30.0
datasets>=2.14.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
torch>=2.0.0
jupyter>=1.0.0
```

## Approach

```
Load Pre-trained GPT-2
       ↓
Apply LoRA Configuration (PEFT)
       ↓
Prepare Dataset (Instruction format)
       ↓
Tokenize Text
       ↓
Fine-tune using Trainer API
       ↓
Evaluate Performance
       ↓
Save Fine-tuned Model
```

## Key Insights (Interview Important)

- LoRA avoids updating full model weights → reduces overfitting
- GPT-2 can be adapted to domain-specific tasks efficiently
- Parameter-efficient fine-tuning is crucial for large-scale LLM deployment
- LoRA enables fine-tuning on limited GPU resources

## Portfolio Presentation

This project demonstrates:
- LLM fine-tuning methodology
- LoRA/PEFT implementation
- Parameter-efficient training
- HuggingFace ecosystem
- GPU memory optimization

## Future Work

- [ ] Implement QLoRA for further optimization
- [ ] Fine-tune larger models (LLaMA, Mistral)
- [ ] Hyperparameter tuning (rank, alpha, dropout)
- [ ] Deploy as chatbot/API

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur

## License

MIT License - See [LICENSE](LICENSE) for details.