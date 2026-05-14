---
title: LoRA GPT-2 Fine-tuning
emoji: 🤖
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10
app_file: app.py
pinned: false
---

# Fine-Tuning GPT-2 using LoRA

An interactive demo showcasing parameter-efficient fine-tuning of GPT-2 using Low-Rank Adaptation (LoRA).

## What is LoRA?

LoRA (Low-Rank Adaptation) freezes pre-trained model weights and introduces small trainable low-rank matrices, drastically reducing trainable parameters while maintaining performance.

## Features

- **Parameter Efficient** - Only 0.1% of parameters trainable
- **60% Cost Reduction** - Compared to full fine-tuning
- **Easy to Deploy** - Swap adapters without changing base model

## How It Works

```
Pre-trained GPT-2 → Freeze Weights → Add LoRA Adapters → Train → Deploy
```

## Tech Stack

- Python · PyTorch · HuggingFace Transformers · PEFT · Gradio

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur