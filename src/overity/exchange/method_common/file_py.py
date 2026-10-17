"""
Parse method information from python file
=========================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import ast
from pathlib import Path

from overity.errors import EmptyMethodDescription
from overity.exchange.method_common import description_md
from overity.model.general_info.method import MethodKind

# --------------------------- Private interface


def _extract_slug(path: Path):
    path = Path(path)
    slug = path.stem

    return slug


def _read_docstring(path: Path):
    source = Path(path).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            return ast.get_docstring(node)

    return None


# --------------------------- Public interface


def from_file(path: Path, kind: MethodKind):
    docstr = _read_docstring(path)
    slug = _extract_slug(path)

    if docstr is not None:
        return description_md.from_md_desc(
            x=docstr, slug=slug, kind=kind, file_path=path
        )

    else:
        raise EmptyMethodDescription(file_path=path)
