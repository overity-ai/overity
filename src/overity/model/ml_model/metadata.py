"""
ML Model metadata model
=======================

**December 2024**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from overity.model.ml_model.attachment import AttachmentMetadata


@dataclass
class MLModelAuthor:
    """Describes an author entry for ML model metadata"""

    """Name of the author"""
    name: str

    """Email address of the author"""
    email: str

    """Optional string describing what its contribution was to the model"""
    contribution: Optional[str] = None


@dataclass
class MLModelMaintainer:
    """Describes a maintainer entry for ML model metadata"""

    """Name of the maintainer"""
    name: str

    """Email of the maintainer"""
    email: str


@dataclass
class MLModelMetadata:
    """Describes the properties of a ML model"""

    """Name of the ML model"""
    name: str

    """Version of the ML model"""
    version: str

    """List of authors"""
    authors: List[MLModelAuthor]

    """List of maitainers"""
    maintainers: List[MLModelMaintainer]

    """Execution target identifier for ML model

    Note that for implementation agnostic models, the special "agnostic" name can be used.
    """
    target: str

    """Format of the model

    Example formats may include "onnx", "keras", "tf", "tflite", etc.
    """
    exchange_format: str

    """Relative to input model file in the package archive file"""
    model_file: str

    """List of attachments available in the model archive"""
    attachments: List[AttachmentMetadata] = field(default_factory=list)

    """Optional identification string indicating which model this model derives from"""
    derives: Optional[str] = None
