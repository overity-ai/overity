"""
Attachment metadata encoder/decoder
===================================

**April 2026**

- Florian Dupeyron (florian.dupeyron@mugcat.fr)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import mimetypes

from pathlib import Path

from overity.model.ml_model.attachment import AttachmentMetadata
from overity.exchange import integrity

from overity.errors import AttachmentIntegrityError


####################################################
# Decoder
####################################################


def meta_decode(data: dict[str, any]) -> AttachmentMetadata:
    # TODO Data check and sanitization, especially for hash

    return AttachmentMetadata(
        filename=data["filename"],
        sha256_hash=data["sha256_hash"],
        mimetype=data.get("mimetype"),
        description=data.get("description"),
    )


####################################################
# Encoder
####################################################


def meta_encode(attachment: AttachmentMetadata) -> dict[str, any]:
    obj = {
        "filename": str(attachment.filename),
        "sha256_hash": str(attachment.sha256_hash),
    }

    if attachment.mimetype:
        obj.update({"mimetype": attachment.mimetype})

    if attachment.description:
        obj.update({"description": attachment.description})

    return obj


####################################################
# Metadata from file
####################################################


def meta_from_file(pp: Path, description: str | None = None) -> AttachmentMetadata:
    """Generate attachment from an existing file"""

    # Guess file type
    mime_type, _ = mimetypes.guess_file_type(str(pp))

    # Get file's sha256
    digest = integrity.file_sha256(pp).hexdigest()

    # Create the output AttachmentMetadata
    return AttachmentMetadata(
        filename=pp.name,
        sha256_hash=digest,
        mimetype=mime_type,
        description=description,
    )


def integrity_check(pp: Path, meta: AttachmentMetadata):
    """Check for an attachment's integrity"""

    digest = integrity.file_sha256(pp)
    hdigest = digest.hexdigest()

    if hdigest != meta.sha256_hash:
        raise AttachmentIntegrityError(pp, hdigest, meta.sha256_hash)
