# Auto-generated from 2_FineTuning.ipynb. Do not edit by hand.
# Original code cells flattened into a single main() for reproducibility.

def main():
    # !pip install unsloth vllm google.generativeai
    # %%capture
    import os
    if "COLAB_" not in "".join(os.environ.keys()):
    #     !pip install unsloth vllm google.generativeai
    else:
        # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    #     !pip install --no-deps unsloth vllm

    #@title Colab Extra Install { display-mode: "form" }
    # %%capture
    import os
    if "COLAB_" not in "".join(os.environ.keys()):
    #     !pip install unsloth vllm
    else:
    #     !pip install --no-deps unsloth vllm
        # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
        # Skip restarting message in Colab
        import sys, re, requests; modules = list(sys.modules.keys())
        for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
    #     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
    #     !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

        # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
        f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
        with open("vllm_requirements.txt", "wb") as file:
            file.write(re.sub(rb"(transformers|numpy|xformers)[^\n]{1,}\n", b"", f))
    #     !pip install -r vllm_requirements.txt
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 8192 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    #The FastLanguageModel.from_pretrained function loads the base model (Qwen2.5-3B) along with its tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-3B",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable fast inference by using vLLM
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    #LoRA adds a set of trainable parameters while freezing the rest of the model
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", # Attention Mechanism Projections
            "gate_proj", "up_proj", "down_proj", # Feed-Forward Network Layers
        ],
        lora_alpha = lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    reasoning_start = "<think>"
    reasoning_end   = "</think>"
    solution_start  = "```python"
    solution_end    = "```"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your code solution between {solution_start} and {solution_end}"""
    system_prompt
    # user prompt is the problem from the dataset
    # system prompt is what we pass to the model to guide its behavior
    # <reasoning start> is a marker indicating the beginning of the reasoning process

    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    # split_0 is the portion of the dataset containing reasoning steps and public information
    split='split_0'

    # Load the dataset in streaming mode(otherwise 700K+ samples will be downloaded D:)
    dataset = load_dataset( "nvidia/OpenCodeReasoning", split, split = 'split_0', streaming = True)

    # We will collect the first 80 samples with a reasoning length < 2000 tokens
    data_list = []
    counter_accepted = 0

    for example in dataset:
        print(example)
        # Extract the output text
        output_text = example.get("output")
        if output_text is None:
            continue

        # Tokenize the reasoning
        tokens = tokenizer.encode(output_text, add_special_tokens=False)

        # Accept if the reasoning is short enough (< 2000 tokens)
        if len(tokens) < 2000:

            data_list.append({
                "output":   output_text,
                "input":    example.get("input"),
                "solution": example.get("solution")
            })
            counter_accepted += 1

        # Stop after collecting 80 valid samples
        if counter_accepted >= 80:
            break

    # Convert the collected data into a pandas DataFrame
    dataset = pd.DataFrame(data_list)

    dataset.head()


    # Check the data shape
    dataset.shape
    def format_dataset(x):
        solution = x["solution"]
        problem = x["input"]

        # Remove markers <think> and </think> from the reasoning
        thoughts = x["output"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Format the reasoning
        thoughts = thoughts.strip()

        # Add our custom formatting
        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + solution + solution_end
        return [
            {"role": "system",    "content": system_prompt}, # Instruction provided to the model to try solving problems
            {"role": "user",      "content": problem},       # The problem prompt taken from the dataset
            {"role": "assistant", "content": final_prompt},  # Reasoning and solution output from the dataset
        ]
    # Apply the formatting function to each row and store the result in a new 'Messages' column
    # This column contains the formatted system-user-assistant chat used for model training or inference
    dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
    # Check the column names of the dataset
    dataset.columns
    # Display the formatted conversation (system, user, assistant) for the first sample
    dataset["Messages"][0]
    # Compute the number of tokens (or characters) for each formatted message using the tokenizer's chat template,
    # and store the result in a new column 'N'
    dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    # Keep only samples with N ≤ 2000 to limit input size
    dataset = dataset.loc[dataset["N"] <= 2000].copy()
    # Check the dataset shape
    dataset.shape
    from datasets import Dataset

    # Apply the tokenizer's chat template to generate full formatted text (without tokenizing)
    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)

    # Convert the DataFrame into a Hugging Face Dataset object
    dataset = Dataset.from_pandas(dataset)
    # Show a sample formatted input prompt
    dataset["text"][13]
    from trl import SFTTrainer, SFTConfig

    # Initialize the SFTTrainer to fine-tune the Qwen2.5-3B model with LoRA adapters.
    # The model passed here was previously loaded via FastLanguageModel.from_pretrained and wrapped with LoRA using get_peft_model.
    # This allows us to fine-tune only a small number of trainable parameters (efficient PEFT),
    # while keeping the rest of the model frozen — reducing memory usage and accelerating training.
    # We use SFTTrainer because it makes it easy to fine-tune Hugging Face models — especially when using LoRA.


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text", # setting here the prompt we created
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 2,
            warmup_steps = 5,
            num_train_epochs = 2, # Set this for 1 full training run.
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
        ),
    )
    # Start the supervised fine-tuning process using the configured trainer
    trainer.train()
    # Preview the first fully formatted training example (used in 'text' field for SFT)
    dataset["text"][0]
    # Check the number of characters (or tokens, depending on tokenizer settings) in each prompt
    # Useful for filtering or validating prompt length before training
    dataset['N']
    # Save the fine-tuned model to a variable for later use
    ft_model = trainer.model
    from transformers import TextStreamer

    # Construct a prompt for inference using only the system instruction and the problem statement (without the solution)
    # This simulates the real use case where the model must reason and generate an answer
    text = tokenizer.apply_chat_template(
        dataset[7]["Messages"][:2],  # First two message blocks: system prompt + user problem (no assistant solution)
        tokenize = False,
        add_generation_prompt = True,   # Required to append the reasoning start marker (e.g., <think>)
                                        # This aligns with the system prompt template defined earlier (reasoning_start tag)
    )


    # Generate a prediction using the fine-tuned model
    _ = ft_model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        temperature = 0, # For greedy decoding (most likely tokens)
        max_new_tokens = 4096, # Allows for long reasoning chains
        streamer = TextStreamer(tokenizer, skip_prompt = False), # TextStreamer provides real-time output printing during generation
    )
    # Free up unused GPU and CPU memory before starting a new data processing session
    import torch
    torch.cuda.empty_cache()  # Clears cached GPU memory in PyTorch

    import gc
    gc.collect()              # Triggers Python garbage collection for unused CPU memory

    from datasets import load_dataset
    from datasets import Dataset
    import pandas as pd

    # Load the OpenCodeReasoning dataset in streaming mode (efficient for large datasets)
    dataset = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)

    data_list = []
    counter_accepted = 0

    # Iterate over streaming dataset to filter examples by reasoning length and difficulty
    for example in dataset:
        output_text = example.get("output")
        if output_text is None:
            continue

        # Tokenize the reasoning output
        tokens = tokenizer.encode(output_text, add_special_tokens=False)

        # Accept examples with reasonable length and high difficulty
        if len(tokens) < 3000:
            data_list.append({
                "output":   output_text,
                "input":    example.get("input"),
                "solution": example.get("solution")
            })
            counter_accepted += 1

        # Stop after collecting 250 valid examples
        if counter_accepted >= 250:
            break

    # Convert the filtered list to a Hugging Face Dataset object
    dataset = Dataset.from_list(data_list)

    # Adapt the dataset structure for GRPO-style fine-tuning or reward modeling
    # Format each sample as a conversation with a prompt (system + user) and a separate answer field
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["input"]},
        ],
        "answer": x["solution"],
    })

    # Convert the Hugging Face Dataset object to a pandas DataFrame for easier inspection or manipulation
    dataset_pd = pd.DataFrame(dataset)

    # Display the solution field of the first example (used as the ground-truth answer)
    dataset_pd['solution'][0]
    import re

    def extract_code(generated_text: str) -> str | None:
        """Extracts Python code block, supporting both ```python``` and <python>...</python> formats."""
        # Match ```python ... ```
        code_block_match = re.search(r"```python\n(.*?)\n```", generated_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Match <python> ... </python>
        xml_block_match = re.search(r"<python>(.*?)</python>", generated_text, re.DOTALL)
        if xml_block_match:
            return xml_block_match.group(1).strip()

        # Fallback: try using full text if it looks like Python
        print("Warning: Could not find code block. Assuming entire output is code.")
        if re.search(r"^\s*(def |class |import |from )", generated_text, re.MULTILINE):
            return generated_text.strip()

        print("Warning: Fallback code doesn't look like Python definition/import. Skipping execution.")
        return None

    # Isolate the Python code
    extract_code(dataset[0]["output"])
    # Define a regex to optionally match closing code block with optional EOS (End Of Sequence) token
    solution_end_regex = r"```[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    # Compile regex to extract the solution code between reasoning_end and code block
    # Supports optional EOS token and whitespace at the end
    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    match_format
    # Test examples: does the pattern correctly extract code after </think> and ```python ...```?
    match_format.findall(
        "Let me think!</think>"\
        f"```python a=input()```",
    )
    # Test examples: does the pattern correctly extract code after </think> and ```python ...```?
    match_format.findall(
        "<think>Let me think!</think>"\
        f"```python  a=input()  ```\n\n",
    )
    # Reward function component: exact match score
    # Gives a fixed score if the format is perfectly correct (reasoning + solution blocks)
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if match_format.search(response) is not None: score += 1.5
            scores.append(score)
        return scores
    # Approximate structural validation
    # Gives partial score based on presence of key format markers in the model's output
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]

            # Reward presence of each structural marker; penalize if missing
            # Note: we skip reasoning_start ("<think>") since it's always prepended
            score += 0.25 if response.count(reasoning_end)   == 1 else -0.5
            score += 0.25 if response.count(solution_start)  == 1 else -0.5
            score += 0.25 if response.count(solution_end)    == 1 else -0.5
            scores.append(score)
        return scores
    match_format.search(dataset['output'][0])
    import os
    import google.generativeai as genai

    # Initialize Gemini model used as the reward function
    # This model will evaluate completions based on reasoning and code quality
    os.environ["GOOGLE_API_KEY"] = "YOURKEY" # Using colab secret is advised
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model_gem = genai.GenerativeModel("models/gemini-2.0-flash")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Compile a regex to extract numeric scores from Gemini output text
    _NUMBER_RE = re.compile(r"(-?\d+(\.\d+)?)")

    # prompts: list of message pairs (system + user), typically from dataset["prompts"]
    # completions: model-generated responses (e.g., from the fine-tuned policy model)
    # answer: reference (ground-truth) solutions from the dataset, usually in dataset["solution"]

    def evaluate_answer(prompts, completions, answer, **kwargs):
        """
        Computes reward scores for model-generated answers using Gemini as the evaluator.
        Each answer is evaluated on:
        - reasoning quality
        - code correctness
        - alignment with the dataset solution

        Returns a score in [-3, 3], where:
        -3 = completely wrong or hallucinated
         3 = perfect step-by-step reasoning and correct solution
        """

        # Build the instruction for Reward LLM
        SCORING_PROMPT = (
            "You are a code expert. You will be given a competitive programming problem, "
            "the candidate's internal reasoning between <think>…</think>, the Python solution "
            "between ```python ... ``` and the official reference solution from the dataset.\n\n"
            "Evaluate the following:\n"
            " - correctness of the reasoning steps\n"
            " - completeness and accuracy of the code\n"
            " - adherence to step-by-step methodology\n"
            " - whether the code uses the correct functions/APIs\n"
            " - whether the code is free of bugs and code smells\n"
            " - whether the code is sufficient to accomplish the task\n"
            " - whether the code uses quotes in string literals correctly\n"
            " - whether the code contains duplicate parameters in functions\n\n"
            "You must also compare the candidate’s solution with the official reference solution from the dataset.\n\n"
            "Assign an integer score from -3 to 3 where:\n"
            " -3 = completely incorrect reasoning and/or code\n"
            " 3 = perfect step-by-step reasoning and a flawless solution\n\n"
            "Respond only with the score and nothing else.\n\n"
            "Reminder: If the candidate’s internal reasoning exceeds 8000 tokens, "
            "they should have admitted inability to continue; penalize that accordingly."
        )

        scores = []
        for prompt, completion_list, ans in zip(prompts, completions, answer):
            problem = prompt[1]["content"]  # user input (problem description)
            candidate = completion_list[0]["content"]  # model's generated answer

            # Build the full input prompt for the reward model
            inp = (
                f"{SCORING_PROMPT}\n\n"
                f"Problem:\n{problem}\n\n"
                f"Dataset answer: \n{ans}\n\n"
                f"Candidate answer:\n{candidate}\n\n"
                "Score:"
            )
            # Generate reward score from Gemini
            resp = model_gem.generate_content(inp).text.strip()
            m = _NUMBER_RE.search(resp)

            # Extract and normalize the score
            if m:
                score = float(m.group(1))  # Extract numeric score
                score = max(-3.0, min(3.0, score))  # Clamp to valid range
            else:
                score = 0.0  # Default fallback if no score found
            scores.append(score)

        return scores
    # Tokenize the dataset using the chat template (includes system + user, and adds generation marker)
    # This will generate input token IDs for each prompt
    tokenized = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )

    # Decode the first tokenized example to inspect the final input string (for debugging/validation)
    print(tokenizer.decode(tokenized[0]["tokens"]))

    # Add a new column 'L' that stores the length (number of tokens) for each example
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    # Compute the 90th percentile of token lengths — this is used as a filtering threshold
    import numpy as np
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)

    # Keep only the samples whose token length is below or equal to the 90th percentile
    # This filters out the longest 10% of samples to avoid extreme lengths that may break context limits
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])

    # Clean up memory by deleting the temporary tokenized dataset
    del tokenized
    print(dataset.column_names)
    print(dataset[0])
    # Compute prompt and completion lengths based on the 90th percentile from previous filtering
    max_prompt_length = maximum_length + 1  # +1 for safety margin
    max_completion_length = max_seq_length - max_prompt_length


    # Set sampling parameters for vLLM during generation (used within GRPO)
    from vllm import SamplingParams


    vllm_sampling_params = SamplingParams(
        min_p = 0.1,                  # Minimum nucleus sampling probability
        top_p = 1.0,                  # Top-p (nucleus) sampling
        top_k = -1,                   # Disable top-k filtering
        seed = 3407,                  # For reproducibility
        stop = [tokenizer.eos_token],# Stop generation when EOS token is produced
        include_stop_str_in_output = True,
    )


    from trl import GRPOConfig, GRPOTrainer

    # Configure the GRPO trainer — similar to PPO, but adapted for reward-based generation
    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 0.6,                   # Sampling diversity
        learning_rate = 5e-5,                # Suitable for LoRA training
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",                # Memory-efficient optimizer
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,     # Can be increased for larger batch sizes
        num_generations = 4,                 # Number of completions sampled per prompt
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = 200,                     # Short run; increase for full training
        save_steps = 100,
        report_to = "none",                  # Can enable Weights & Biases later
        output_dir = "outputs",              # Directory to save checkpoints
    )
    # Initialize the GRPO trainer with multiple reward functions:
    # - match_format_exactly: reward if full output format matches
    # - match_format_approximately: softer reward for partial format structure
    # - evaluate_answer: semantic reward from Gemini (LLM-based scoring)
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            evaluate_answer,

        ],
        args = training_args,
        train_dataset = dataset, # Full dataset; can split if evaluation is enabled

    )

    trainer.train() # Start training with GRPO (reward-based fine-tuning)
    # FAI VEDDERE CHE QUESTO VA IN LOOP CONTINUO(MODELLO PRE FINETUNING CHE NON SA FARE BENE IL REASONING)
    # Simple inference example using vLLM fast_generate interface (no reward model involved)
    text = "Solve the famous twosum coding problem"

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_k = 50,
        max_tokens = 1024,
    )

    # Generate a response using the base model
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,  # No LoRA applied here
    )[0].outputs[0].text


    output
    from safetensors import safe_open

    # Load and inspect the saved LoRA adapter weights to ensure they are not empty
    tensors = {}
    with safe_open("grpo_trainer_lora_model/adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()

            # Sanity check: ensure the entire tensor is not zero-filled
            assert(n_zeros.item() != 1.0), f"Tensor {key} appears to be empty or uninitialized"

    # Construct a full prompt using system + user (structured via chat template)
    from safetensors import safe_open
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Solve the famous twosum coding problem"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,  # Required to trigger generation after user input
        tokenize = False,
    )

    # Define sampling parameters for a longer response
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_k = 50,
        max_tokens = 2048,
    )

    # Generate using the trained GRPO LoRA weights
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_trainer_lora_model"),
    )[0].outputs[0].text

    output


if __name__ == "__main__":
    main()
