def run_finetune(model_name: str, dataset_id: str) -> None:
    try:
        from gwen25.notebooks.finetuning import main as nb_main
    except Exception as e:
        raise RuntimeError(
            "Notebook export for FineTuning not found. Run `make nb-export` first."
        ) from e
    nb_main()


