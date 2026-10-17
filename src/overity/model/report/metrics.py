"""
Model definition for report metrics
===================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Metric(ABC):
    @classmethod
    @abstractmethod
    def kind(cls) -> str:
        """Get value kind encoding string"""

    @abstractmethod
    def data(self) -> dict[str, Any]:
        """Output data dict."""

    @classmethod
    @abstractmethod
    def from_data(cls, data: dict[str, Any]) -> Metric:
        pass


@dataclass
class SimpleValue(Metric):
    value: int | float

    @classmethod
    def kind(cls) -> str:
        return "simple"

    def data(self) -> dict[str, Any]:
        return {"kind": self.kind(), "value": self.value}

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> SimpleValue:
        return cls(value=data["value"])


@dataclass
class LinScaleValue(Metric):
    low: float
    high: float
    value: float

    @classmethod
    def kind(cls) -> str:
        return "lin_scale"

    def data(self) -> dict[str, Any]:
        return {
            "kind": self.kind(),
            "low": self.low,
            "high": self.high,
            "value": self.value,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> LinScaleValue:
        return cls(
            low=float(data["low"]),
            high=float(data["high"]),
            value=float(data["value"]),
        )


@dataclass
class LinRangeValue(Metric):
    low: int
    high: int
    value: int

    @classmethod
    def kind(cls) -> str:
        return "lin_range"

    def data(self) -> dict[str, Any]:
        return {
            "kind": self.kind(),
            "low": self.low,
            "high": self.high,
            "value": self.value,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> LinRangeValue:
        return cls(
            low=int(data["low"]),
            high=int(data["high"]),
            value=int(data["value"]),
        )


@dataclass
class PercentageValue(Metric):
    value: float  # 0..1

    @classmethod
    def kind(cls) -> str:
        return "percentage"

    def data(self) -> dict[str, Any]:
        return {
            "kind": self.kind(),
            "value": self.value,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> PercentageValue:
        return cls(value=float(data["value"]))


###########################

_KIND_ARRAY = {
    SimpleValue.kind(): SimpleValue,
    LinScaleValue.kind(): LinScaleValue,
    LinRangeValue.kind(): LinRangeValue,
    PercentageValue.kind(): PercentageValue,
}


def from_data(data: dict[str, Any]) -> Metric:
    cls_def = _KIND_ARRAY[data["kind"]]
    return cls_def.from_data(data)
