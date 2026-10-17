"""
Model class for bench capability identification
===============================================

**June 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Capability:
    """Identifies a requirement on a test bench

    For instance, it can represent the bench ability to measure energy consumption per inference.
    A capability is associated with a requirement into what the bench should provide, i.e. a
    given metric or a specific method.
    """

    """Identifier of the capability"""
    slug: str

    """Display name"""
    display_name: str

    """Optional description"""
    description: str | None = None

    def __hash__(self):
        return hash(self.slug)
