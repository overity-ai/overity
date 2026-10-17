"""
Dataset metadata encoder/decoder
================================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import json
from pathlib import Path

from overity.model.dataset.metadata import (
    DatasetAuthor,
    DatasetMaintainer,
    DatasetMetadata,
)

####################################################
# Decoder
####################################################


def _metadata_decode_author(data: dict[str, any]):
    name = data["name"]
    email = data["email"]
    contribution = data.get("contribution")

    return DatasetAuthor(
        name=name,
        email=email,
        contribution=contribution,
    )


def _metadata_decode_maintainer(data: dict[str, any]):
    name = data["name"]
    email = data["email"]

    return DatasetMaintainer(name=name, email=email)


def _metadata_decode(data: dict[str, any]):
    name = data["name"]
    authors = [_metadata_decode_author(x) for x in data["authors"]]
    maintainers = [_metadata_decode_maintainer(x) for x in data["maintainers"]]
    description = data.get("description")

    return DatasetMetadata(
        name=name, authors=authors, maintainers=maintainers, description=description
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


def _metadata_encode_author(author: DatasetAuthor):
    encode_obj = {
        "name": author.name,
        "email": author.email,
    }

    if author.contribution is not None:
        encode_obj["contribution"] = author.contribution

    return encode_obj


def _metadata_encode_maintainer(maintainer: DatasetMaintainer):
    return {"name": maintainer.name, "email": maintainer.email}


def _metadata_encode(metadata: DatasetMetadata):
    encode_obj = {
        "name": metadata.name,
        "authors": [_metadata_encode_author(x) for x in metadata.authors],
        "maintainers": [_metadata_encode_maintainer(x) for x in metadata.maintainers],
    }

    if metadata.description:
        encode_obj["description"] = metadata.description

    return encode_obj


def to_file(metadata: DatasetMetadata, json_path: Path):
    json_path = Path(json_path)

    with open(json_path, "w") as fhandle:
        json.dump(_metadata_encode(metadata), fhandle)
