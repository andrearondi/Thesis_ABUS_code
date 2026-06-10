#!/usr/bin/env python
"""Pre-flight guard: flag Python-3.10-only *runtime* constructs in modules that
execute in the server's nnDetection env (Python 3.8).

Why this exists
---------------
The project targets Python 3.10 (`pyproject.toml`: requires-python >=3.10,
ruff target-version py310). That is correct for the `thesis` env. But a subset
of `abus` modules is imported and run on the server inside the `nndet` conda env,
which is **Python 3.8.20** (nnDetection 0.1 pins it). Code that is valid and even
*recommended* by the linter on 3.10 can crash at runtime on 3.8.

This bit us at STORY_01_02 Step 13 (2026-06-09): `zip(..., strict=False)` and
`isinstance(x, list | tuple)` — both valid on 3.10, both runtime TypeErrors on 3.8.
The laptop pre-flight (ruff/black/mypy/pytest on 3.10) cannot catch these, because
they are not syntax errors and not type errors — they fail only when executed on 3.8.

What it catches
---------------
AST-level detection (no import needed, so torch-dependent modules can be scanned):
  1. ``zip(..., strict=...)``            — `strict=` kwarg on zip is 3.10+.
  2. ``isinstance(x, A | B)`` /
     ``issubclass(x, A | B)``            — `X | Y` union as a class arg is 3.10+.
  3. ``match``/``case`` statements        — structural pattern matching is 3.10+.

This is deliberately a *small, high-precision* guard for the constructs that have
actually reached the server, not a general 3.8 compatibility checker. For a
comprehensive check, run `vermin -t=3.8- <files>` (not installed by default).

Scope
-----
Only the modules below run in the nndet (3.8) env on the server — the ones the
server scripts import via the sys.path trick (see STORY_01_02 runbook Step 5 / 12).
Application code in the `thesis` (3.10) env is intentionally NOT scanned.

Exit code: 0 if clean, 1 if any 3.10-only runtime construct is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Modules imported and executed in the server's nnDetection (Python 3.8) env.
# Keep this list in sync with what scripts/generate_candidates.py and
# scripts/train_all_folds.py import from the `abus` package.
NNDET_ENV_MODULES = [
    "src/abus/detect/nndet_inference.py",
    "src/abus/detect/ensemble.py",
    "src/abus/detect/candidates.py",
    "src/abus/detect/train.py",
    "src/abus/data/split.py",
]


class Py310RuntimeVisitor(ast.NodeVisitor):
    """Collect (lineno, message) for 3.10-only runtime constructs."""

    def __init__(self) -> None:
        self.findings: list[tuple[int, str]] = []

    def _is_union(self, node: ast.expr) -> bool:
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None

        # 1. zip(..., strict=...)
        if name == "zip":
            for kw in node.keywords:
                if kw.arg == "strict":
                    self.findings.append(
                        (node.lineno, "zip(..., strict=...) — `strict=` kwarg is Python 3.10+")
                    )

        # 2. isinstance/issubclass(x, A | B)
        if name in {"isinstance", "issubclass"} and len(node.args) >= 2:
            if self._is_union(node.args[1]):
                self.findings.append(
                    (
                        node.lineno,
                        f"{name}(x, A | B) — `X | Y` union as a class arg is Python 3.10+ "
                        "(use a (A, B) tuple instead)",
                    )
                )

        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:  # ast.Match exists only on 3.10+
        self.findings.append(
            (
                getattr(node, "lineno", 0),
                "match/case statement — structural pattern matching is 3.10+",
            )
        )
        self.generic_visit(node)


def scan(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = Py310RuntimeVisitor()
    visitor.visit(tree)
    return sorted(visitor.findings)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    targets = sys.argv[1:] or NNDET_ENV_MODULES
    total = 0
    for rel in targets:
        path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            print(f"  SKIP (not found): {rel}")
            continue
        findings = scan(path)
        if findings:
            total += len(findings)
            for lineno, msg in findings:
                print(f"  {rel}:{lineno}: {msg}")

    if total:
        print(
            f"\nFAIL — {total} Python-3.10-only runtime construct(s) found in nndet-env "
            "(Python 3.8) module(s). These will crash on the server. Fix before the runbook."
        )
        return 1
    print("OK — no Python-3.10-only runtime constructs in nndet-env modules (Python 3.8 safe).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
