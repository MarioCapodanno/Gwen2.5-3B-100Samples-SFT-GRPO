# Auto-generated from 5_OllamaHF.ipynb. Do not edit by hand.
# Original code cells flattened into a single main() for reproducibility.

def main():
    # !git lfs install
    # !git clone https://huggingface.co/MarioCap/Qwen2.5-3B-OCR-100S
    # !cd Qwen2.5-3B-OCR-100S
    # !pip install transformers sentencepiece

    # !git clone https://github.com/ggerganov/llama.cpp
    import os
    os.chdir("llama.cpp")
    # !python convert_hf_to_gguf.py ../Qwen2.5-3B-OCR-100S --outfile qwen2.5-3b-ocr-100s.gguf --outtype q8_0

    # !curl -fsSL https://ollama.com/install.sh | sh
    # !pip install colab-xterm
    # %load_ext colabxterm
    # %xterm
    modelfile_content = """
    FROM qwen2.5-3b-ocr-100s.gguf

    TEMPLATE \"\"\"<|im_start|>system
    {{ .System }}<|im_end|>
    <|im_start|>user
    {{ .Prompt }}<|im_end|>
    <|im_start|>assistant
    \"\"\"

    SYSTEM \"\"\"You are Coding Assistant, created by o5mini Team of Polimi. You are a python assistant.\"\"\"

    PARAMETER temperature 0.75
    PARAMETER top_p 0.8
    PARAMETER top_k 20
    PARAMETER repeat_penalty 1.05
    PARAMETER stop <|im_start|>
    PARAMETER stop <|im_end|>
    PARAMETER stop <|endoftext|>
    """

    with open('Modelfile', 'w') as f:
        f.write(modelfile_content)
    # !ollama create qwen2_5_3b_ocr_100s -f Modelfile
    # !ollama list
    # !ollama run qwen2_5_3b_ocr_100s "create a simple python code?"
    # !ollama cp qwen2_5_3b_ocr_100s Duss02/qwen2_5_3b_ocr_100s

    # !ollama push Duss02/qwen2_5_3b_ocr_100s


if __name__ == "__main__":
    main()
