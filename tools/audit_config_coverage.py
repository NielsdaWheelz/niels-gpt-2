#!/usr/bin/env python3
"""
Audit for centralized configuration coverage.

Fails if:
- denylisted hyperparameter literals appear outside settings
- special token literals leak outside tokenizer/settings
- optimizers are constructed with hardcoded hyperparameters
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ["niels_gpt", "train", "tools"]
EXCLUDE_DIR_PARTS = {"tests", "venv", "__pycache__", "server", "ui"}
ALLOWED_HYPERPARAM_FILES = {ROOT / "niels_gpt" / "settings.py"}
ALLOW_SPECIAL_TOKEN_FILES = {ROOT / "niels_gpt" / "tokenizer.py", ROOT / "niels_gpt" / "settings.py"}

DENYLIST_NAMES = {
    "base_lr",
    "min_lr",
    "warmup_steps",
    "weight_decay",
    "betas",
    "eps",
    "grad_clip",
    "total_steps",
    "eval_every",
    "eval_steps",
    "ckpt_every",
    "micro_B",
    "accum_steps",
    "dropout",
    "rope_theta",
    "vocab_size",
    "T",
    "C",
    "L",
    "H",
    "d_ff",
    "temperature",
    "top_k",
    "top_p",
    "max_new_tokens",
}

SPECIAL_TOKENS = {"<|sys|>", "<|usr|>", "<|asst|>", "<|eot|>"}
OPT_KW = {"lr", "betas", "eps", "weight_decay"}


def _is_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_numeric_literal(elt) for elt in node.elts)
    return False


def _should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_PARTS for part in path.parts)


class AuditVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.issues: list[tuple[int, str]] = []

    def _record(self, node: ast.AST, message: str) -> None:
        self.issues.append((getattr(node, "lineno", 0), message))

    def _is_allowed_hparam_file(self) -> bool:
        return self.path in ALLOWED_HYPERPARAM_FILES

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_allowed_hparam_file():
            return
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in DENYLIST_NAMES:
                if _is_numeric_literal(node.value):
                    self._record(node, f"suspicious literal for {target.id}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._is_allowed_hparam_file():
            return
        target = node.target
        if isinstance(target, ast.Name) and target.id in DENYLIST_NAMES and node.value and _is_numeric_literal(node.value):
            self._record(node, f"suspicious literal for {target.id}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._is_allowed_hparam_file():
            arg_defaults = list(node.args.defaults)
            args = node.args.args[-len(arg_defaults) :] if arg_defaults else []
            for arg, default in zip(args, arg_defaults, strict=False):
                if arg.arg in DENYLIST_NAMES and _is_numeric_literal(default):
                    self._record(default, f"default for arg {arg.arg}")

            kw_defaults = node.args.kw_defaults
            for kw_arg, default in zip(node.args.kwonlyargs, kw_defaults, strict=False):
                if kw_arg.arg in DENYLIST_NAMES and default and _is_numeric_literal(default):
                    self._record(default, f"default for kwarg {kw_arg.arg}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in SPECIAL_TOKENS:
            if self.path not in ALLOW_SPECIAL_TOKEN_FILES:
                self._record(node, f"special token literal '{node.value}' outside tokenizer/settings")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name.lower().startswith("adam"):
            for kw in node.keywords or []:
                if kw.arg in OPT_KW and kw.value and _is_numeric_literal(kw.value):
                    if not self._is_allowed_hparam_file():
                        self._record(kw, f"hardcoded optimizer {kw.arg}")
        self.generic_visit(node)


def _iter_files() -> Iterable[Path]:
    for root in SEARCH_DIRS:
        base = ROOT / root
        for path in base.rglob("*.py"):
            if _should_skip(path):
                continue
            if path == Path(__file__).resolve():
                continue
            yield path


def main() -> None:
    issues: list[tuple[Path, int, str]] = []
    count = 0
    for path in _iter_files():
        count += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = AuditVisitor(path)
        visitor.visit(tree)
        for line, msg in visitor.issues:
            issues.append((path, line, msg))

    if issues:
        for path, line, msg in issues:
            rel = path.relative_to(ROOT)
            print(f"{rel}:{line}: {msg}")
        sys.exit(1)

    print(f"OK {count} files scanned")


if __name__ == "__main__":
    main()

