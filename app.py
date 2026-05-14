import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def draw_lora_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Draw boxes
    boxes = [
        (1, 5, 'Pre-trained\nGPT-2', '#7b61ff'),
        (1, 2.5, 'Frozen\nWeights', '#ff4d6d'),
        (5, 5, 'LoRA\nAdapters', '#00ffb2'),
        (9, 5, 'Fine-tuned\nModel', '#ffd700'),
    ]
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.8, y-0.8), 1.6, 1.6, fill=True, facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrows
    ax.annotate('', xy=(4.2, 5), xytext=(1.8, 5), arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(8.2, 5), xytext=(5.8, 5), arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(2.5, 3.3), xytext=(2.5, 4.2), arrowprops=dict(arrowstyle='->', color='#ff4d6d80', lw=1.5, ls='--'))

    # Labels
    ax.text(3, 5.3, 'freeze', fontsize=8, color='#ff4d6d', ha='center')
    ax.text(7, 5.3, 'merge', fontsize=8, color='#ffd700', ha='center')

    ax.text(1, 1.5, '124M params', ha='center', fontsize=8, color='#ff4d6d')
    ax.text(5, 3.5, '0.1%\ntrainable', ha='center', fontsize=9, color='#00ffb2', fontweight='bold')
    ax.text(5, 2.5, '~124K params', ha='center', fontsize=8, color='#00ffb2')

    ax.set_title('LoRA Architecture: Parameter-Efficient Fine-tuning', color='white', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

def draw_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')

    methods = ['Full Fine-tune', 'LoRA', 'QLoRA']
    params = [124, 0.124, 0.062]  # Millions of trainable params
    memory = [24, 2.4, 1.2]  # GB

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, params, width, label='Trainable Params (M)', color='#7b61ff', alpha=0.8)
    bars2 = ax.bar(x + width/2, memory, width, label='GPU Memory (GB)', color='#00ffb2', alpha=0.8)

    for bar, v in zip(bars1, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{v}M', ha='center', fontsize=10, color='white', fontweight='bold')

    for bar, v in zip(bars2, memory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{v}GB', ha='center', fontsize=10, color='white', fontweight='bold')

    ax.set_xlabel('Fine-tuning Method', color='white', fontsize=11)
    ax.set_ylabel('Resource Usage', color='white', fontsize=11)
    ax.set_title('LoRA vs Full Fine-tuning Efficiency', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, color='white')
    ax.legend(facecolor='#0f1320', edgecolor='#ffffff20', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#ffffff20')
    plt.tight_layout()
    return fig

def draw_training_curve():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')

    steps = np.arange(0, 101, 10)
    full_loss = 2.5 * np.exp(-steps/40) + 0.3 + np.random.randn(11)*0.05
    lora_loss = 2.3 * np.exp(-steps/35) + 0.35 + np.random.randn(11)*0.05

    ax.plot(steps, full_loss, 'o-', label='Full Fine-tune', color='#ff4d6d', linewidth=2, markersize=6)
    ax.plot(steps, lora_loss, 's-', label='LoRA', color='#00ffb2', linewidth=2, markersize=6)

    ax.set_xlabel('Training Steps', color='white', fontsize=11)
    ax.set_ylabel('Loss', color='white', fontsize=11)
    ax.set_title('Training Loss Comparison', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.legend(facecolor='#0f1320', edgecolor='#ffffff20', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, 3)
    for spine in ax.spines.values(): spine.set_color('#ffffff20')
    plt.tight_layout()
    return fig

def generate_prompt_analysis(prompt):
    if not prompt.strip():
        return "❌ Please enter a prompt!"

    # Simulated analysis based on prompt content
    words = prompt.lower().split()
    topics = []
    if any(w in words for w in ['medical', 'health', 'doctor', 'symptom']): topics.append("Medical QA")
    if any(w in words for w in ['code', 'programming', 'python', 'function']): topics.append("Code Generation")
    if any(w in words for w in ['write', 'story', 'essay', 'article']): topics.append("Creative Writing")
    if any(w in words for w in ['explain', 'what', 'how', 'why']): topics.append("Explanation")

    if not topics: topics = ["General NLP"]

    analysis = f"""
## 📝 Prompt Analysis

### Your Prompt
> "{prompt}"

### Detected Topics
"""
    for t in topics:
        analysis += f"- **{t}**\n"

    analysis += f"""
### LoRA Fine-tuning Benefits

| Aspect | Without LoRA | With LoRA |
|--------|-------------|-----------|
| Training Time | ~3 hours | ~20 minutes |
| GPU Memory | ~24GB | ~2.4GB |
| Storage | Full model | ~1MB adapter |

### Recommendation
This prompt would benefit from domain-specific LoRA fine-tuning if it relates to your use case (medical, code, creative, etc.)

**Next Steps:**
1. Prepare domain-specific dataset
2. Apply LoRA adapters with rank=8, alpha=16
3. Fine-tune for 100-500 steps
4. Merge and deploy
"""

    return analysis

css = """
body, .gradio-container { background: #080b12 !important; }
footer { display: none !important; }
textarea, input[type='text'] { background: #161c2e !important; border: 1px solid #ffffff1a !important; border-radius: 12px !important; color: #dde3f0 !important; }
.gr-button-primary { background: linear-gradient(135deg, #7b61ff, #ff4d6d) !important; border: none !important; border-radius: 12px !important; color: white !important; font-weight: 600 !important; }
"""

with gr.Blocks(css=css, title="LoRA GPT-2 Fine-tuning") as demo:
    gr.HTML("""
    <div style="background:linear-gradient(90deg,#080b12,#0d1220);border-bottom:1px solid #ffffff0f;padding:16px 28px;display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#7b61ff,#ff4d6d);display:flex;align-items:center;justify-content:center;font-size:16px">🤖</div>
        <div>
          <div style="font-size:15px;font-weight:600;color:#dde3f0">LoRA GPT-2 Fine-tuning</div>
          <div style="font-size:10px;color:#5a6280;font-family:monospace">Parameter-Efficient LLM Adaptation</div>
        </div>
      </div>
      <div style="display:flex;gap:10px">
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #00ffb260;color:#00ffb2">0.1% params</span>
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #7b61ff60;color:#7b61ff">60% cheaper</span>
      </div>
    </div>
    """)

    with gr.Tab("LoRA Architecture"):
        gr.Plot(draw_lora_architecture, label="Architecture")

    with gr.Tab("Efficiency Comparison"):
        gr.Plot(draw_comparison, label="Comparison")

    with gr.Tab("Training Loss"):
        gr.Plot(draw_training_curve, label="Loss Curve")

    with gr.Tab("Try Your Prompt"):
        gr.Markdown("### Enter a prompt to see how LoRA fine-tuning helps")
        prompt_input = gr.Textbox(lines=3, placeholder="e.g. Explain quantum computing in simple terms", label="Your Prompt")
        analyze_btn = gr.Button("Analyze Prompt", variant="primary")
        output = gr.Markdown()
        analyze_btn.click(fn=generate_prompt_analysis, inputs=prompt_input, outputs=output)

demo.launch()