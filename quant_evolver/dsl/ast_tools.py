from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExprStats:
    valid_syntax: bool
    depth: int = 0
    nodes: int = 0
    operators: dict[str, int] = field(default_factory=dict)
    canonical: str = ""
    ast_hash: str = ""
    error: str = ""


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + (max((_depth(c) for c in children), default=0))


def analyse_expr(expr: str) -> ExprStats:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        return ExprStats(valid_syntax=False, error=str(exc))
    ops: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            ops[node.func.id] = ops.get(node.func.id, 0) + 1
        elif isinstance(node, ast.BinOp):
            name = type(node.op).__name__
            ops[name] = ops.get(name, 0) + 1
    canonical = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return ExprStats(
        valid_syntax=True,
        depth=_depth(tree),
        nodes=sum(1 for _ in ast.walk(tree)),
        operators=ops,
        canonical=canonical,
        ast_hash=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )
