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

### Training Procedure

#### Stage 1: Supervised Fine-Tuning (SFT)
- **Training objective:** Next token prediction on formatted reasoning + code pairs
- **Batch size:** 1 (with gradient accumulation steps: 2)
- **Learning rate:** 2e-4
- **Epochs:** 2
- **Optimizer:** AdamW 8-bit
- **Weight decay:** 0.01
- **Warmup steps:** 5

#### Stage 2: Generalized Reward-guided Policy Optimization (GRPO)
- **Training objective:** Policy optimization using multiple reward functions
- **Reward functions:**
  - Format matching (exact and approximate)
  - Solution correctness evaluation (using Gemini-2.0-flash as reward model)
- **Learning rate:** 5e-5
- **Max steps:** 100
- **Temperature:** 0.6
- **Generations per step:** 4

### Technical Specifications
- **Maximum sequence length:** 32768 tokens
- **LoRA configuration:**
  - Rank: 32
  - Alpha: 64 (2 × rank)
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision:** 16-bit training
- **Hardware:** GPU A100 40GB (but can fit easily on a T4 GPU)

## Evaluation

### Testing Data, Factors & Metrics

#### LiveCodeBench Evaluation
The model was evaluated on LiveCodeBench problem set v1, focusing on code generation tasks.

**Performance Comparison:**

| Model | Pass@1 | Pass@5 | Easy Pass@1 | Medium Pass@1 | Hard Pass@1 |
|-------|--------|--------|-------------|---------------|-------------|
| **Fine-tuned Model** | 0.1885 (18.85%) | 0.28 (28.00%) | 0.4239 (42.39%) | 0.0905 (9.05%) | 0.0 (0%) |
| **Base Qwen2.5-3B** | 0.1585 (15.85%) | 0.2175 (21.75%) | 0.3127 (31.27%) | 0.1131 (11.31%) | 0.0 (0%) |
| **Improvement** | **+3.0%** | **+7.25%** | **+11.12%** | **-2.26%** | **±0%** |

### Model Architecture & Reasoning Format

The model generates responses in a structured format:
```
<think>
[Step-by-step reasoning and problem analysis]
</think>
```python
[Python code solution]
```

This format encourages the model to:
1. Think through the problem systematically
2. Provide clear reasoning steps
3. Generate clean, executable code solutions

## Technical Limitations and Biases


### Biases
- **Dataset Bias:** Inherits biases from the nvidia/OpenCodeReasoning dataset
- **Problem Type Bias:** Optimized for competitive programming style problems
- **Language Bias:** Strongly biased toward Python implementations

## Additional Information

### Not Recommended For
- Production code generation without review
- Complex software architecture decisions
- Security-critical code implementation
- Problems requiring extensive domain knowledge beyond basic algorithms

### Model Access
- **Inference:** Compatible with vLLM for fast inference
- **Format:** LoRA adapters can be merged with base model or used separately
- **Hardware Requirements:** Supports both CPU and GPU inference

