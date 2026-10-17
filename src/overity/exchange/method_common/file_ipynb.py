"""
Parse method information from jupyter notebook
==============================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

from overity.errors import EmptyMethodDescription
from overity.exchange.method_common import description_md
from overity.model.general_info.method import MethodKind

# --------------------------- Private interface


def _extract_slug(path: Path):
    path = Path(path)
    slug = path.stem

    return slug


def _extract_first_md_cell(path: Path):
    path = Path(path)

    with open(path, "r") as fhandle:
        nb = nbformat.read(fhandle, as_version=4)

    return next(filter(lambda x: x["cell_type"] == "markdown", nb["cells"]))


# --------------------------- Public interface


def from_file(path: Path, kind: MethodKind):
    path = Path(path)
    slug = _extract_slug(path)

    try:
        md_desc = _extract_first_md_cell(path=path)
        md_text = "".join(md_desc["source"])

        if not md_text:
            raise EmptyMethodDescription(file_path=path)

        infos = description_md.from_md_desc(
            x=md_text, slug=slug, kind=kind, file_path=path
        )

        return infos

    except StopIteration:
        raise EmptyMethodDescription(file_path=path)
