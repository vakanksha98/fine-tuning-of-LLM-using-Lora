# GPT-2 Finance Sentiment Analysis
### Fine-Tuning with LoRA (Low-Rank Adaptation)

Fine-tuning GPT-2 on financial news data to classify market sentiment into positive, negative, and neutral categories using parameter-efficient LoRA adaptation via HuggingFace PEFT.

---

## Results

| Metric | Base GPT-2 | Fine-Tuned (LoRA) | Improvement |
|:---|:---:|:---:|:---:|
| Sentiment Loss | 14.97 | 5.85 | 61% drop |
| Perplexity | 3,172,524 | 348.5 | 99.9% drop |
| Accuracy | 20.52% | **65.75%** | +45% |
| Trainable Parameters | 124M (100%) | 294K (0.23%) | 99.77% fewer |

65.75% sentiment accuracy achieved by training only 0.23% of model parameters in under 15 minutes on a T4 GPU.

---

## Project Structure

```
gpt2-lora-finance/
|
|-- configs/
|   └-- config.yaml              # All hyperparameters and settings
|
|-- data/
|   |-- data_loader.py           # Data pipeline and preprocessing
|   └-- samples/
|       |-- sample_data.csv      # Example data samples
|       └-- DATASET_INFO.md      # Dataset documentation
|
|-- model/
|   └-- model_setup.py           # GPT-2 + LoRA architecture
|
|-- training/
|   └-- trainer.py               # Training loop and checkpointing
|
|-- inference/
|   └-- generate.py              # Sentiment prediction on new text
|
|-- notebooks/
|   └-- gpt2_lora_finance_finetune.ipynb  # End-to-end Colab notebook
|
|-- evaluate.py                  # Before vs After evaluation
|-- main.py                      # Entry point
|-- requirements.txt
└-- README.md
```

---

## Dataset

**Source:** [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

| Property | Detail |
|:---|:---|
| Total Samples | 11,931 financial news sentences |
| Domain | Financial Twitter — earnings, stocks, analyst ratings |
| Labels | Negative, Positive, Neutral |
| Class Distribution | Neutral: 64.7% / Positive: 20.2% / Negative: 15.1% |

**Sample Data:**

| Sentence | Sentiment |
|:---|:---:|
| $AAPL - Morgan Stanley raises price target to $220 | Positive |
| $BYND - JPMorgan reels in expectations on Beyond Meat | Negative |
| Iran to Seek Advanced Arms as UN Embargo Expires | Neutral |

**Handling Class Imbalance:**

The dataset is imbalanced with the neutral class dominating at 64.7%. This was handled using:

- Stratified Split: each of train/val/test maintains the same class distribution as the full dataset
- Weighted Random Sampler: minority classes are sampled more frequently during training so the model does not get biased toward neutral
- Test set kept at original distribution for fair real-world evaluation

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that makes it possible to fine-tune large language models with very few trainable parameters.

**The Problem with Full Fine-Tuning:**

When you fine-tune GPT-2 normally, all 124 million parameters are updated. This requires a lot of GPU memory and takes a long time.

**How LoRA Solves This:**

Instead of updating the original weight matrix W directly, LoRA freezes W and adds two small trainable matrices A and B alongside it:

```
W_new = W_original + (B x A)

Where:
W_original  shape: (768 x 768)  -- frozen, not updated
A           shape: (768 x 8)    -- trainable
B           shape: (8 x 768)    -- trainable
r = 8 is the rank (we chose this in config)
```

Only A and B are trained. Since r=8 is much smaller than 768, the number of trainable parameters drops dramatically.

**Why This Works:**

Research has shown that the weight changes needed for fine-tuning tend to have low intrinsic rank. This means a low-rank approximation captures most of the useful information without needing to update all parameters.

**LoRA vs Full Fine-Tuning:**

| | Full Fine-Tuning | LoRA |
|:---|:---:|:---:|
| Trainable Parameters | 124M | 294K |
| GPU Memory | High | Low |
| Training Time | Slow | Fast |
| Accuracy | High | Almost same |
| Base Model Modified | Yes | No |

LoRA was applied to the c_attn layer in GPT-2, which is the attention projection layer and the most important layer for language understanding.

---

## How the Training Works

**Prompt Format:**

Each financial news sentence is converted into the following format before training:

```
Input  : "Sentence: $AAPL beats earnings expectations Sentiment:"
Output : "positive"
```

GPT-2 is an autoregressive model — it predicts the next token. By framing the task this way, the model learns to read a finance sentence and predict the correct sentiment word as the next token.

**Training Configuration:**

```
Base Model    : GPT-2 (124M parameters)
LoRA Rank     : 8
LoRA Alpha    : 32
Dropout       : 0.1
Epochs        : 3
Learning Rate : 2e-4
Batch Size    : 8
GPU           : T4 (Google Colab)
Training Time : ~13 minutes
```

**Training Progress:**

Validation loss consistently decreased over training with no overfitting observed:

```
Step  200  :  3.203
Step  500  :  3.194
Step 1000  :  3.161
Step 1500  :  3.147
Step 2000  :  3.123
Step 2500  :  3.117
Step 3000  :  3.110
Step 3579  :  3.104  (best checkpoint saved)
```

---

## Quickstart

**1. Clone and Install**

```bash
git clone https://github.com/vakanksha98/gpt2-lora-finance.git
cd gpt2-lora-finance
pip install -r requirements.txt
```

**2. Train and Evaluate**

```bash
python main.py
```

**3. Run Inference**

```bash
python inference/generate.py --text "$AAPL beats earnings expectations"
```

Output:

```
Text      : $AAPL beats earnings expectations
Sentiment : POSITIVE
Confidence: {'negative': 8.2%, 'positive': 79.4%, 'neutral': 12.4%}
```

**4. Run on Google Colab**

Open `notebooks/gpt2_lora_finance_finetune.ipynb`, set runtime to T4 GPU, and run all cells.

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Base Model | GPT-2 (OpenAI) |
| Fine-Tuning | LoRA via HuggingFace PEFT |
| Training | HuggingFace Trainer API |
| Dataset | HuggingFace Datasets |
| Framework | PyTorch |
| Platform | Google Colab (T4 GPU) |

---

## Author

**Akanksha Verma**
M.Tech, Computer Science and Data Processing
Indian Institute of Technology, Kharagpur

GitHub: [vakanksha98](https://github.com/vakanksha98)
