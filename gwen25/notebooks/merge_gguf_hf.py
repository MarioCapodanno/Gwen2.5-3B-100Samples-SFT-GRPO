# Auto-generated from 4_merge_gguf_hf.ipynb. Do not edit by hand.
# Original code cells flattened into a single main() for reproducibility.

def main():
    # !git clone https://huggingface.co/Qwen/Qwen2.5-3B
    # !pip install transformers peft
    # !pip install -U bitsandbytes
    from google.colab import drive
    drive.mount('/content/drive')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    # Paths to your models
    base_model_path = "Qwen2.5-3B"        # Replace with the path to your base model
    lora_adapter_path = "/content/drive/MyDrive/checkpoint-50"    # Replace with the path to your LoRA adapter
    output_path = "Qwen2.5-3B-OCR-50S"          # Replace with the desired output path

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge the adapter with the base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(output_path)

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    from huggingface_hub import upload_folder

    repo_id = "MarioCap/Qwen2.5-3B-OCR-50S"

    upload_folder(
        repo_id=repo_id,
        folder_path="Qwen2.5-3B-OCR-50S",
        path_in_repo=".",          # radice del repo
        commit_message="first push",
        token="xxxxxxxxxx"
    )



if __name__ == "__main__":
    main()
