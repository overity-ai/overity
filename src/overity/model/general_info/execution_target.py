"""
Execution target information
============================

**March 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from dataclasses import dataclass


@dataclass
class ExecutionTarget:
    """An execution target identifies on which target a model can be deployed"""

    """Slug acts as a text identifier for the execution target"""
    slug: str

    """Display name for Execution target"""
    display_name: str

    """Tags can help idnetify this target among others.

    Example tags can be: SoC, MCU, etc.
    """
    tags: frozenset[str]

    """Optional description text."""
    description: str | None = None

    def __hash__(self):
        return hash(self.slug)
