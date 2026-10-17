"""
Dataset package model
=====================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from overity.model.dataset.metadata import DatasetMetadata


@dataclass
class DatasetPackageInfo:
    """Describes what is needed to build a dataset package"""

    """Dataset metadata information"""
    metadata: DatasetMetadata

    """Path to dataset path on local disk"""
    dataset_data_path: Path
