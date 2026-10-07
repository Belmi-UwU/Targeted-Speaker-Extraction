"""Convert wesep_retrain.py (percent format, the source of truth) into wesep_retrain.ipynb.

    python training/build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def convert(source):
    cells, kind, lines = [], None, []

    def flush():
        text = "\n".join(lines).strip("\n")
        if kind is None or not text:
            return
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                          "source": text.splitlines(keepends=True)})

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    notebook = convert((HERE / "wesep_retrain.py").read_text())
    (HERE / "wesep_retrain.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote wesep_retrain.ipynb with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
