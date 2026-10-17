"""
Model class for program information
===================================

**February 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class ProgramInitiator:
    """A program initiator is is responsible for the initialization of the program."""

    """Name of person"""
    name: str

    """Email of person"""
    email: str

    """Optional role for this person in its organization"""
    role: str | None = None


@dataclass
class ProgramInfo:
    """Information for a program"""

    """The slug is the recognizable unique identifier for the program"""
    slug: str

    """Display name represents a more readable name for user display"""
    display_name: str

    """Program creation date"""
    date_created: date

    """Who initiated this program"""
    initiator: ProgramInitiator

    """Is the program active?"""
    active: bool

    """An optioanl description can act as a tagline to tell what the program is for"""
    description: str | None = None
