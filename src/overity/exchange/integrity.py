"""
Integrity utilities
===================

**April 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import hashlib

from pathlib import Path


def file_sha256(pp: Path):
    """Get the sha256 hash for a given file

    Args:
        pp: Path to input file

    Returns:
        The computed sha256 digest
    """

    path = Path(pp)

    with open(path, "rb") as fhandle:
        digest = hashlib.file_digest(fhandle, "sha256")

    return digest
