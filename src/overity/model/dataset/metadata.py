"""
Dataset metadata model
======================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetAuthor:
    """Describes an author entry for a dataset"""

    """Name of the author"""
    name: str

    """Email address of the author"""
    email: str

    """Optional string describing what its contribution was to the program"""
    contribution: str | None = None


@dataclass
class DatasetMaintainer:
    """Describes a maintainer entry for a dataset"""

    """Name of the maintainer"""
    name: str

    """Email of the maintainer"""
    email: str


@dataclass
class DatasetMetadata:
    """Describes the properties of a dataset"""

    """Name of the dataset"""
    name: str

    """List of authors"""
    authors: list[DatasetAuthor]

    """List of maintainers"""
    maintainers: list[DatasetMaintainer]

    """Optional dataset description"""
    description: str | None = None
