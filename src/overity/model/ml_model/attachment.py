"""
Model attachment model
======================

**April 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttachmentMetadata:
    """Defines the metadata for an attachment"""

    """Name of the attachment in model archive"""
    filename: str

    """sha256 hash"""
    sha256_hash: str

    """MIME Type indication"""
    mimetype: str | None = None

    """Optional description"""
    description: str | None = None


@dataclass
class ExtractedAttachment:
    """Represents an extracted attachment. Useful for API"""

    """Metadata of the extracted attachment"""
    meta: AttachmentMetadata

    """Path of local extracted file"""
    path: Path
