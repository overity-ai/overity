"""
Parser for bench abstraction definition
=======================================

**September 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from overity.errors import EmptyMethodDescription
from overity.exchange.bench_abstraction import description_md

####################################################
# Private interface
####################################################


def _extract_slug(path: Path):
    path = Path(path)
    slug = path.name.removesuffix(".py")

    return slug


def _read_docstring(path: Path):
    source = Path(path).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            return ast.get_docstring(node)

    return None


####################################################
# Public interface
####################################################


def from_file(path: Path):
    """Import bench abstraction metadata from file"""

    docstr = _read_docstring(path)
    slug = _extract_slug(path)

    if docstr is not None:
        return description_md.from_md_desc(x=docstr, slug=slug, file_path=path)
    else:
        raise EmptyMethodDescription(file_path=path)  # TODO # Use another exception?


def import_definitions(path: Path):
    """Import bench abstraction settings and definition from python file"""

    path = Path(path)

    module_name = "bench_abstraction." + path.name.removesuffix(".py")

    # Load module from file path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load bench abstraction from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get constructors
    if not hasattr(module, "BenchSettings") or not hasattr(module, "BenchDefinition"):
        raise AttributeError(
            "Bench abstraction must define a BenchSettings dataclass and BenchDefinition class"
        )

    BenchSettings = module.BenchSettings
    BenchDefinition = module.BenchDefinition

    # TODO # Maybe add some conformity checks?

    return BenchSettings, BenchDefinition
