"""
Overity.ai path manipulation utilities
======================================

**February 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr): Initial design

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from pathlib import Path


def iter_path(pp: Path):
    """Iterate path from cwd to filesystem root, generator style!

    Args:
        pp: Starting path

    Returns:
        A generator that iterates through paths, starting from current folder
    """

    cur_path = pp

    while True:
        yield cur_path

        if cur_path.parent != cur_path:
            cur_path = cur_path.parent
        else:
            break


def is_subpath(a: Path, b: Path) -> bool:
    """Check if a is a child of b

    Args:
        a: Path to check (the potential subpath)
        b: base Path (the potential parent)
    Returns:
        a boolean indicating if a is a sub-path of b

    NOTE: This is quite hacky, I think there is a better way to do this
    """

    try:
        Path(a).relative_to(Path(b))
        return True
    except ValueError:  # Raised if is not a sub-path
        return False
