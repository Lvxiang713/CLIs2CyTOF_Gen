from __future__ import annotations

import ast
import types
from pathlib import Path

import nbformat


_ALLOWED_TOPLEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Try,
)


def load_legacy_notebook_namespace(notebook_path: str | Path) -> types.SimpleNamespace:
    """Load only imports and function/class definitions from a legacy notebook.

    This keeps the original analysis logic available without executing the notebook's
    imperative cells.
    """
    notebook_path = Path(notebook_path)
    nb = nbformat.read(notebook_path, as_version=4)
    chunks: list[str] = []
    ns: dict[str, object] = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, _ALLOWED_TOPLEVEL):
                continue
            if isinstance(node, ast.Import):
                skip = any(alias.name == "FM_mulGPU_train_multiToken_noClsToken_maskToken" for alias in node.names)
                if skip:
                    continue
            if isinstance(node, ast.Try):
                # only keep optional import try blocks
                all_try_nodes = list(node.body) + list(node.handlers[0].body if node.handlers else []) + list(node.orelse) + list(node.finalbody)
                if not all(isinstance(x, (ast.Import, ast.ImportFrom, ast.Assign, ast.Pass)) for x in all_try_nodes):
                    continue
            mod = ast.Module(body=[node], type_ignores=[])
            chunk = ast.unparse(mod)
            try:
                exec(chunk, ns, ns)
            except ModuleNotFoundError:
                continue
            except Exception:
                continue
    return types.SimpleNamespace(**ns)
