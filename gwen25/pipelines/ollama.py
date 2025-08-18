def run_ollama_ops(model_name: str, gguf_path: str) -> None:
    try:
        from gwen25.notebooks.ollama_hf import main as nb_main
    except Exception as e:
        raise RuntimeError(
            "Notebook export for Ollama not found. Run `make nb-export` first."
        ) from e
    nb_main()


