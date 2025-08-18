def run_eval(model_name: str) -> None:
    try:
        from gwen25.notebooks.final_eval_base_50s_100s import main as nb_main
    except Exception as e:
        raise RuntimeError(
            "Notebook export for FinalEval not found. Run `make nb-export` first."
        ) from e
    nb_main()


