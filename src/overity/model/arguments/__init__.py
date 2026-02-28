"""
Model to store arguments when using methods
===========================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Argument:
    name: str
    value: str


@dataclass
class Option:
    name: str
    value: str | None
    default: str


@dataclass
class Flag:
    name: str
    value: bool


@dataclass(frozen=True)
class ArgumentSchema:
    name: str
    help: str


@dataclass(frozen=True)
class OptionSchema:
    name: str
    default: str
    help: str


@dataclass(frozen=True)
class FlagSchema:
    name: str
    help: str


@dataclass
class List:
    name: str
    value: list[str]


@dataclass(frozen=True)
class ListSchema:
    name: str
    help: str
