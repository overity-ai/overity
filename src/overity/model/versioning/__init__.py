"""
Versioning information classes
==============================

**February 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr): Initial design

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from enum import Enum


class VersioningStatus(Enum):
    """Indicates versioning status for a given asset.

    This is primarly used for git-tracked assets (ingredients)
    """

    """No versioning information has been found for the asset"""
    NotVersioned = "not_versioned"

    """The last versioned version does not correspond to the asset the user use"""
    Dirty = "dirty"

    """The asset is correctly versioned"""
    Clean = "clean"
