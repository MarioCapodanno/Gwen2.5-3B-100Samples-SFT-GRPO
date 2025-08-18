def run_merge_and_convert(base_model: str, lora_path: str, outdir: str) -> None:
    try:
        from gwen25.notebooks.merge_gguf_hf import main as nb_main
    except Exception as e:
        raise RuntimeError(
            "Notebook export for merge/gguf not found. Run `make nb-export` first."
        ) from e
    nb_main()


