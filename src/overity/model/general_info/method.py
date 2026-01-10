"""
Model classes to describe a method
==================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class MethodKind(Enum):
    """Describe what this method is for"""

    """Traning and optimization is for creating new models"""
    TrainingOptimization = "training_optimization"

    """Measurement and qualification is to assess requirements and performances on a model"""
    MeasurementQualification = "measurement_qualification"

    """Deployment method is to deploy the method on execution target"""
    Deployment = "deployment"

    """Analysis method is to do comparative analysis"""
    Analysis = "analysis"


@dataclass
class MethodAuthor:
    """Describes an author for a method"""

    """Name of the author"""
    name: str

    """Email of the author"""
    email: str

    """Optional contribution details"""
    contribution: str | None = None


@dataclass
class MethodInfo:
    """Contains method metadata information"""

    """Slug identifier for this method"""
    slug: str

    """What this method is for"""
    kind: MethodKind

    """Display name"""
    display_name: str

    """Authors for this method"""
    authors: list[MethodAuthor]

    """Metadata fields"""
    metadata: dict[str, str]

    """Optional description text"""
    description: str | None

    """Optional original file path"""
    path: Path | None
