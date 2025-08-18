#!/usr/bin/env python3
"""
Lightweight notebook-to-python exporter.

It extracts code cells from each .ipynb and writes a Python module with a main() entry.
Notebook magics and shell commands (lines starting with ! or %) are commented out for safety.
"""

import json
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    ("1_NLP_EDA.ipynb", "gwen25/notebooks/nlp_eda.py"),
    ("2_FineTuning.ipynb", "gwen25/notebooks/finetuning.py"),
    ("3_FinalEval_Base_50S_100S.ipynb", "gwen25/notebooks/final_eval_base_50s_100s.py"),
    ("4_merge_gguf_hf.ipynb", "gwen25/notebooks/merge_gguf_hf.py"),
    ("5_OllamaHF.ipynb", "gwen25/notebooks/ollama_hf.py"),
]


def _clean_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("!") or stripped.startswith("%"):
        return "# " + line
    return line


def _indent(lines: Iterable[str], spaces: int = 4) -> Iterable[str]:
    pad = " " * spaces
    for line in lines:
        if line.strip() == "":
            yield "\n"
        else:
            yield pad + line


def export_notebook(ipynb_path: Path, py_out_path: Path) -> None:
    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    sources: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell_src = cell.get("source") or []
        if isinstance(cell_src, str):
            cell_lines = cell_src.splitlines(keepends=True)
        else:
            cell_lines = cell_src
        for raw in cell_lines:
            sources.append(_clean_line(raw))
        sources.append("\n\n")

    py_out_path.parent.mkdir(parents=True, exist_ok=True)
    with py_out_path.open("w", encoding="utf-8") as out:
        out.write(f"# Auto-generated from {ipynb_path.name}. Do not edit by hand.\n")
        out.write("# Original code cells flattened into a single main() for reproducibility.\n\n")
        out.write("def main():\n")
        for line in _indent(sources, spaces=4):
            out.write(line)
        out.write("\n\nif __name__ == \"__main__\":\n    main()\n")


def main():
    for nb_rel, out_rel in NOTEBOOKS:
        ip = REPO_ROOT / nb_rel
        op = REPO_ROOT / out_rel
        if not ip.exists():
            print(f"[skip] Notebook not found: {ip}")
            continue
        export_notebook(ip, op)
        print(f"[ok] Exported {ip.name} -> {op}")


if __name__ == "__main__":
    main()


