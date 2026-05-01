"""
Model metadata file encoder/decoder
===================================

**December 2024**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import json
from pathlib import Path

from overity.model.ml_model.metadata import (
    MLModelAuthor,
    MLModelMaintainer,
    MLModelMetadata,
)

from overity.exchange.model_package_v1 import attachment


####################################################
# Decoder
####################################################


def _metadata_decode_model_author(data: dict[str, any]):
    name = data["name"]
    email = data["email"]
    contribution = data.get("contribution")

    return MLModelAuthor(name=name, email=email, contribution=contribution)


def _metadata_decode_model_maintainer(data: dict[str, any]):
    name = data["name"]
    email = data["email"]

    return MLModelMaintainer(name=name, email=email)


def _metadata_decode(data: dict[str, any]):
    name = data["name"]
    version = data["version"]
    authors = [_metadata_decode_model_author(x) for x in data["authors"]]
    maintainers = [_metadata_decode_model_maintainer(x) for x in data["maintainers"]]
    target = data["target"]
    exchange_format = data["format"]
    model_file = data["model_file"]
    attachments = [attachment.meta_decode(att) for att in data.get("attachments", [])]
    derives = data.get("derives")

    return MLModelMetadata(
        name=name,
        version=version,
        authors=authors,
        maintainers=maintainers,
        target=target,
        exchange_format=exchange_format,
        model_file=model_file,
        attachments=attachments,
        derives=derives,
    )


def from_dict(data: dict[str, str]):
    # Schema validation
    # TODO

    return _metadata_decode(data)


def from_file(json_path: Path):
    json_path = Path(json_path)

    # Parse JSON data
    with open(json_path) as fhandle:
        data = json.load(fhandle)

    return from_dict(data)


####################################################
# Encoder
####################################################


def _metadata_encode_model_author(author: MLModelAuthor):
    encode_obj = {
        "name": author.name,
        "email": author.email,
    }

    if author.contribution is not None:
        encode_obj["contribution"] = author.contribution

    return encode_obj


def _metadata_encode_model_maintainer(maintainer: MLModelMaintainer):
    return {
        "name": maintainer.name,
        "email": maintainer.email,
    }


def _metadata_encode(metadata: MLModelMetadata):
    encode_obj = {
        "name": metadata.name,
        "version": metadata.version,
        "authors": [_metadata_encode_model_author(x) for x in metadata.authors],
        "maintainers": [
            _metadata_encode_model_maintainer(x) for x in metadata.maintainers
        ],
        "target": metadata.target,
        "format": metadata.exchange_format,
        "model_file": metadata.model_file,
    }

    if metadata.derives is not None:
        encode_obj["derives"] = metadata.derives

    if metadata.attachments:
        encode_obj["attachments"] = [
            attachment.meta_encode(att) for att in metadata.attachments
        ]

    return encode_obj


def to_file(metadata: MLModelMetadata, json_path: Path):
    json_path = Path(json_path)  # Ensure path is in correct type

    with open(json_path, "w") as fhandle:
        json.dump(_metadata_encode(metadata), fhandle)
