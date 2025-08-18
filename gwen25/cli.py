import argparse
from typing import Optional

from gwen25.pipelines.eda import run_nlp_eda
from gwen25.pipelines.train import run_finetune
from gwen25.pipelines.evaluate import run_eval
from gwen25.pipelines.export import run_merge_and_convert
from gwen25.pipelines.ollama import run_ollama_ops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gwen25", description="Gwen2.5 pipelines")
    sub = parser.add_subparsers(dest="command")

    p_eda = sub.add_parser("eda", help="Run NLP EDA pipeline")
    p_eda.add_argument("--dataset", required=True)

    p_train = sub.add_parser("train", help="Run fine-tuning pipeline")
    p_train.add_argument("--model", required=True)
    p_train.add_argument("--dataset", required=True)

    p_eval = sub.add_parser("eval", help="Run evaluation pipeline")
    p_eval.add_argument("--model", required=True)

    p_export = sub.add_parser("export", help="Merge LoRA and convert to GGUF")
    p_export.add_argument("--base_model", required=True)
    p_export.add_argument("--lora_path", required=True)
    p_export.add_argument("--outdir", required=True)

    p_ollama = sub.add_parser("ollama", help="Ollama model packaging and tests")
    p_ollama.add_argument("--model_name", required=True)
    p_ollama.add_argument("--gguf_path", required=True)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eda":
        run_nlp_eda(dataset_id=args.dataset)
    elif args.command == "train":
        run_finetune(model_name=args.model, dataset_id=args.dataset)
    elif args.command == "eval":
        run_eval(model_name=args.model)
    elif args.command == "export":
        run_merge_and_convert(base_model=args.base_model, lora_path=args.lora_path, outdir=args.outdir)
    elif args.command == "ollama":
        run_ollama_ops(model_name=args.model_name, gguf_path=args.gguf_path)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


