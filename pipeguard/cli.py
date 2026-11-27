"""Command-line entry point and source loading."""

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import tokenize

from .analyzer import Analyzer


SUFFIXES = {".py", ".ipynb"}
SKIP_DIRECTORIES = {"__pycache__", "build", "dist", "node_modules", "venv"}
PYTHON_CELL_MAGICS = {"capture", "debug", "prun", "time", "timeit"}
LINE_MAGIC = re.compile(r"^( *)(?:%[^%]|!|\?)")


def sources(path):
    if path.suffix == ".py":
        with tokenize.open(path) as handle:
            yield 0, handle.read()
        return

    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("expected a notebook with a cells array")
    for number, cell in enumerate(notebook["cells"], 1):
        if not isinstance(cell, dict):
            raise ValueError(f"cell {number} must be an object")
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError(f"cell {number} has invalid source")
        lines = source.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            if line.startswith("%%"):
                magic = line[2:].split()
                if not magic or magic[0] not in PYTHON_CELL_MAGICS:
                    lines = []
                else:
                    lines[index] = "\n"
            break
        for index, line in enumerate(lines):
            match = LINE_MAGIC.match(line)
            if match:
                lines[index] = f"{match[1]}pass\n"
        yield number, "".join(lines)


def discover(paths):
    seen = set()

    def fail(error):
        raise error

    for path in paths:
        if path.is_dir():
            candidates = directory_files(path, fail)
        elif not path.is_file():
            raise ValueError(f"{path} does not exist or is not a regular file")
        elif path.suffix not in SUFFIXES:
            raise ValueError(f"{path} must end in .py or .ipynb")
        else:
            candidates = (path,)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def directory_files(path, onerror):
    for root, dirs, files in os.walk(path, onerror=onerror):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRECTORIES
                         and not name.startswith("."))
        for name in sorted(files):
            if Path(name).suffix in SUFFIXES and not name.startswith("."):
                yield Path(root) / name


def scan(path, index_splits=()):
    analyzer = Analyzer(index_splits)
    errors = []
    try:
        for cell, source in sources(path):
            analyzer.cell = cell
            prefix = f"{path}:cell {cell}" if cell else str(path)
            try:
                tree = ast.parse(source, filename=prefix)
            except (SyntaxError, ValueError) as error:
                if isinstance(error, SyntaxError):
                    prefix += f":{error.lineno or 1}:{error.offset or 1}"
                    errors.append(f"{prefix}: {error.msg}")
                else:
                    errors.append(f"{prefix}: {error}")
                continue
            analyzer.visit(tree)
    except (OSError, ValueError, SyntaxError) as error:
        errors.append(f"{path}: {error}")
    return sorted(analyzer.diagnostics), errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find ML preprocessing data leakage.")
    parser.add_argument("paths", type=Path, nargs="+", help="Python files, notebooks, or directories")
    parser.add_argument(
        "--index-split", action="append", default=[], metavar="MODULE.FUNCTION",
        help="recognize an imported helper that takes X and returns train and test indices",
    )
    options = parser.parse_args(argv)
    if any("." not in name or not all(part.isidentifier() for part in name.split("."))
           for name in options.index_split):
        parser.error("--index-split requires a fully qualified module and function")
    status = 0
    try:
        for path in discover(options.paths):
            diagnostics, errors = scan(path, options.index_split)
            for finding in diagnostics:
                prefix = f"{path}:cell {finding.cell}" if finding.cell else str(path)
                print(f"{prefix}:{finding.line}:{finding.column}: {finding.code} {finding.message}")
            for error in errors:
                print(error, file=sys.stderr)
            status = max(status, 2 if errors else int(bool(diagnostics)))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return status
