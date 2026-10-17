"""
Inference agent package model
=============================

**June 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from overity.model.inference_agent.metadata import InferenceAgentMetadata


@dataclass
class InferenceAgentPackageInfo:
    """Describes what is needed to build an inference agent package"""

    """Inference agent metadata information"""
    metadata: InferenceAgentMetadata

    """Path to inference agent data on local disk"""
    agent_data_path: Path
