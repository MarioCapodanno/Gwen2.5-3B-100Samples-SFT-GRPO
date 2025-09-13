## Model Details

- **Developed by:** o5-mini team, Politecnico di Milano
- **Model type:** Causal Language Model
- **Language(s):** English (primary), Python code
- **Finetuned from model:** Qwen/Qwen2.5-3B
- **Model size:** 3B parameters + LoRA adapters (rank 32)

### Model Sources
- **Base Model:** [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- **Training Dataset:** nvidia/OpenCodeReasoning (split_0)
- **Training Framework:** Unsloth + TRL (Transformers Reinforcement Learning)

## Uses

### Direct Use
This model is designed for:
- Competitive programming problem solving
- Code generation with step-by-step reasoning
- Algorithm implementation and explanation

## Training Details

### Training Data
- **Primary Dataset:** nvidia/OpenCodeReasoning (split_0)
- **Training Samples:** 
  - SFT: 80 samples
  - GRPO: 100 samples
- **Data Filtering:** Samples were filtered based on reasoning token length.

##  Results (LiveCodeBench Evaluation)

| Model | Pass@1 | Pass@5 | Easy Pass@1 | Medium Pass@1 | Hard Pass@1 |
|-------|:------:|:------:|:-----------:|:-------------:|:-----------:|
| **Fine-tuned Model** | **18.85%** | **28.00%** | **42.39%** | **9.05%** | **0.00%** |
|  Base Qwen2.5-3B | 15.85% | 21.75% | 31.27% | 11.31% | 0.00% |
| **Improvement** | **+3.00%** | **+7.25%** | **+11.12%** | **-2.26%** | **±0.00%** |

### Installation

You can install the whole package in editable mode, or only the requirements you need per task.

Editable install (recommended for development):
```bash
python -m pip install -U pip
pip install -e .
```

Per-task installs:
```bash
# Core shared deps
pip install -r requirements/base.txt

# EDA tooling
pip install -r requirements/nlp_eda.txt

# Training (SFT/TRL/Unsloth, etc.)
pip install -r requirements/train.txt

# Evaluation (vLLM, transformers)
pip install -r requirements/eval.txt

# Merge + GGUF export
pip install -r requirements/convert_export.txt

# Ollama client helpers
pip install -r requirements/ollama.txt
```

Aggregate installs via Makefile:
```bash
make install          # pip install -e .
make install-all      # installs all requirement sets
make install-train    # training requirements only
```

### CLI Usage

After installing, use the `gwen25` command.

- EDA
```bash
gwen25 eda --dataset your_org/your_dataset
```

- Training (baseline SFT)
```bash
gwen25 train --model Qwen/Qwen2.5-3B --dataset your_org/your_dataset
```

- Evaluation (simple generation)
```bash
gwen25 eval --model path-or-hub-id
```

- Merge LoRA + Convert to GGUF
```bash
# Requires llama.cpp conversion script accessible as `convert_hf_to_gguf.py`.
gwen25 export --base_model Qwen/Qwen2.5-3B --lora_path /path/to/lora --outdir /path/to/out
```

- Ollama Packaging
```bash
# Requires an existing GGUF file and Ollama installed system-wide.
gwen25 ollama --model_name qwen2_5_3b_ocr_100s --gguf_path /path/to/model.gguf
```

### Makefile Shortcuts

```bash
# install
make install
make install-all
# EDA
make eda DATASET=your_org/your_dataset
# Training
make train MODEL=Qwen/Qwen2.5-3B DATASET=your_org/your_dataset
# Evaluation
make eval MODEL=path-or-hub-id
# Export to GGUF
make export BASE=Qwen/Qwen2.5-3B LORA=/path/to/lora OUTDIR=/path/to/out
# Ollama
make ollama MODEL_NAME=qwen2_5_3b_ocr_100s GGUF=/path/to/model.gguf
```


