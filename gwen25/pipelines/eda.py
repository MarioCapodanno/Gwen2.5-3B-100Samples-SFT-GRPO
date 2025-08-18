def run_nlp_eda(dataset_id: str) -> None:
    try:
        from gwen25.notebooks.nlp_eda import main as nb_main
    except Exception as e:
        raise RuntimeError(
            "Notebook export for EDA not found. Run `make nb-export` first."
        ) from e
    # Optionally pass dataset via env or global; notebooks often load inside.
    nb_main()


