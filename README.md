# SorosAdvisor: Financial Advisory Model Inspired by George Soros

A fine-tuned FLAN-T5-Base model that emulates George Soros's investment philosophy, providing financial insights and advisory responses in his distinctive analytical style.

**The model and the source code are hosted on HuggingFace**

[![View on HuggingFace](https://img.shields.io/badge/View%20on%20%F0%9F%A4%97%20Hugging%20Face-ffc107?style=flat&color=ffc107&logoColor=white)](https://huggingface.co/coffeewithyogurt/SorosAdvisor)

## Model Details

### Model Description

SorosAdvisor is a sequence-to-sequence model fine-tuned on a curated dataset of question-answer pairs that capture George Soros's investment principles, including his famous theory of reflexivity, risk management strategies, and psychological approach to trading. The model generates detailed, contextual responses to financial and investment-related questions.

- **Model type:** Encoder-Decoder (Seq2Seq)
- **Language(s):** English
- **License:** MIT
- **Finetuned from:** [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

### Model Sources

- **Repository:** [GitHub - SorosAdvisor](https://github.com/kumarsandeep567/SorosAdvisor)
- **Base Model:** [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

## Uses

### Direct Use

The model can be used directly for:
- Generating investment philosophy insights in the style of George Soros *(For educational purposes only)*
- Educational purposes to understand Soros's trading psychology and principles
- Exploring concepts like reflexivity, risk management, and market psychology *(For educational purposes only)*
- Research on financial NLP and domain-specific fine-tuning

### Downstream Use

- Integration into financial education platforms *(Refer Out-of-Scope use below)*
- Chatbot backends for investment philosophy discussions *(Refer Out-of-Scope use below)*
- Research tools for studying investment strategies

### Out-of-Scope Use

⚠️ **This model should NOT be used for:**
- Actual financial advice or investment decisions
- Trading recommendations or portfolio management
- Any use case where financial loss could occur
- Replacing professional financial advisors
- Making real-world investment choices

**This is an educational and research model only.**

## Bias, Risks, and Limitations

### Known Limitations

1. **Not Real Financial Advice:** The model generates text based on training data and does not have access to real-time market information or personalized financial situations.

2. **Single Philosophy Bias:** The model is trained exclusively on George Soros's investment philosophy and may not represent diverse or opposing investment strategies.

3. **Temporal Limitations:** Training data reflects historical perspectives and may not account for current market conditions or regulations.

4. **Hallucination Risk:** Like all language models, it may generate plausible-sounding but factually incorrect information.

5. **Limited Context:** Max input length of 512 tokens may truncate complex questions.

### Risks

- Users may incorrectly interpret outputs as actionable financial advice
- The model reflects one investor's philosophy which may not suit all investment goals
- Generated content should be verified against authoritative sources

### Recommendations

- Always consult qualified financial professionals for investment decisions
- Use this model for educational and research purposes only
- Cross-reference any information with reliable financial sources
- Do not use outputs for actual trading or investment activities

## How to Get Started with the Model

### Installation

1. Clone the repository
```bash
git clone https://huggingface.co/coffeewithyogurt/SorosAdvisor
```

2. Create and activate the virtual environment (Install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/) if not already installed)
```bash
uv venv finv3_env --python 3.12

# For Linux/macOS
source finv3_env/bin/activate

# For Windows
finv3_env\Scripts\activate
```

3. Install the dependencies
```bash
uv sync --active

# Alternatively, you can also install via pip
uv pip install -r requirements.txt
```

4. Set the API keys
```bash
cp .env.example .env
```

5. Run the training script
```bash
python train.py
```

6. Run the inference script
```bash
python inference.py
```
---
### Basic Usage (Inference)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "path/to/sorosT5Base_Finetuned/model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Prepare input
prompt = "You are a financial advisor embodying George Soros's investment philosophy. Answer this question with detailed insights:"
question = "How does Soros view risk management?"
input_text = f"{prompt} {question}"

# Tokenize
inputs = tokenizer(
    input_text,
    max_length=512,
    truncation=True,
    return_tensors="pt"
).to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        min_length=30
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Generation Modes

The model supports two generation modes:

**1. Sampling Mode (Creative/Diverse)**
```python
generation_kwargs = {
    "max_length": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "min_length": 30
}
```

**2. Beam Search Mode (Deterministic/Consistent)**
```python
generation_kwargs = {
    "max_length": 256,
    "do_sample": False,
    "num_beams": 4,
    "length_penalty": 1.0,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "min_length": 30,
    "early_stopping": True
}
```

## Training Details

### Training Data

The model was fine-tuned on a custom dataset (`QuestionsParaphrased.csv`) containing ~600 question-answer pairs covering George Soros's investment philosophy across multiple categories:

| Category | Description |
|----------|-------------|
| **Psychology** | Trading psychology, emotional discipline, bias recognition |
| **Risk Management** | Position sizing, loss management, portfolio protection |
| **Adaptability** | Market feedback loops, market analysis, trend identification |
| **Timing** | Market timing, market entry and exit points |
| **Strategy Development** | Investment process, conviction |

**Data Format:**
- **Input:** Financial/investment questions with system prompt
- **Target:** Detailed responses in Soros's analytical style

### Training Procedure

#### Preprocessing

- Questions and answers cleaned for empty/null values
- System prompt prepended to all inputs:
  > "You are a financial advisor embodying George Soros's investment philosophy. Answer this question with detailed insights:"
- Dynamic padding using `DataCollatorForSeq2Seq`

#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Base Model** | google/flan-t5-base |
| **Training Regime** | FP32 (full precision) |
| **Batch Size** | 8 |
| **Gradient Accumulation Steps** | 2 |
| **Effective Batch Size** | 16 |
| **Learning Rate** | 1e-5 |
| **Weight Decay** | 0.01 |
| **Warmup Ratio** | 0.1 |
| **Epochs** | 20 (with early stopping) |
| **Early Stopping Patience** | 3 epochs |
| **Label Smoothing** | 0.1 |
| **Max Input Length** | 512 tokens |
| **Max Target Length** | 256 tokens |
| **Optimizer** | AdamW |

## Evaluation

### Metrics

The model is evaluated using **ROUGE scores**, which measure the overlap between generated and reference texts:

| Metric | Description |
|--------|-------------|
| **ROUGE-1** | Unigram overlap |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |
| **ROUGE-Lsum** | Summary-level ROUGE-L |

### Results

*Results will vary based on training run. Check Weights & Biases logs for specific run metrics.*

| Metric | Score |
|--------|-------|
| **ROUGE-1** | TBD |
| **ROUGE-2** | TBD |
| **ROUGE-L** | TBD |
| **ROUGE-Lsum** | TBD |

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** T5 Encoder-Decoder Transformer
- **Parameters:** ~248M (FLAN-T5-Base)
- **Objective:** Conditional text generation (Seq2Seq)
- **Vocabulary Size:** 32,128 tokens (SentencePiece)

### Compute Infrastructure

#### Hardware

- **GPU:** NVIDIA GPU with CUDA support (recommended)

#### Software

- **Framework:** PyTorch 2.7+
- **Library:** Hugging Face Transformers 4.40+
- **Python:** 3.12+
- **Package management:** uv

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** NVIDIA GPU
- **Hours used:** Varies by hardware (~1-2 hours typical)
- **Cloud Provider:** Local / On-premise
- **Compute Region:** Varies
- **Carbon Emitted:** Estimated based on hardware and duration

## Citation

**BibTeX:**
```bibtex
@misc{sorosadvisor2025,
  author = {CoffeeWithYogurt},
  title = {SorosAdvisor: A Fine-tuned FLAN-T5 Model for George Soros Investment Philosophy},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/coffeewithyogurt/soros-advisor}}
}
```

## Glossary

| Term | Definition |
|------|------------|
| **Reflexivity** | Soros's theory that market participants' biased views can influence market fundamentals, creating feedback loops |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation - metrics for evaluating text generation |
| **Seq2Seq** | Sequence-to-Sequence - model architecture that transforms input sequences to output sequences |
| **FLAN-T5** | Fine-tuned Language Net T5 - Google's instruction-tuned version of T5 |

## More Information

### Project Structure

```
SorosAdvisor/
├── FullFull-FineTuned/
│   └── (remaining files)
└── LoRA-FineTuned/
    └── (remaining files)
```

### Example Questions

The model can answer questions like:
- "How does Soros apply self-awareness to his trading decisions?"
- "What is the theory of reflexivity and how does it apply to markets?"
- "How does Soros manage risk in volatile markets?"
- "What role does psychology play in Soros's investment approach?"


## Contact

- Please contact by opening an issue on the GitHub repository
---

**Disclaimer:** This model is for educational and research purposes only. It does not provide real financial advice. Use at your own risk. Always consult qualified financial professionals for investment decisions.
