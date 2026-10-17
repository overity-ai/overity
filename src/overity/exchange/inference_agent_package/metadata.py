"""
Inference agent metadata file encoder/decoder
=============================================

**June 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overity.model.inference_agent.metadata import (
    InferenceAgentAuthor,
    InferenceAgentMaintainer,
    InferenceAgentMetadata,
)

####################################################
# Decoder
####################################################


def _metadata_decode_author(data: dict[str, str]) -> InferenceAgentAuthor:
    name = data["name"]
    email = data["email"]
    contribution = data.get("contribution")

    return InferenceAgentAuthor(name=name, email=email, contribution=contribution)


def _metadata_decode_maintainer(data: dict[str, str]):
    name = data["name"]
    email = data["email"]

    return InferenceAgentMaintainer(name=name, email=email)


def _metadata_decode(data: dict[str, Any]) -> InferenceAgentMetadata:
    name = data["name"]
    version = data["version"]
    authors = [_metadata_decode_author(x) for x in data["authors"]]
    maintainers = [_metadata_decode_maintainer(x) for x in data["maintainers"]]
    capabilities = frozenset(data["capabilities"])
    compatible_targets = frozenset(data["compatible_targets"])
    compatible_tags = frozenset(data["compatible_tags"])

    return InferenceAgentMetadata(
        name=name,
        version=version,
        authors=authors,
        maintainers=maintainers,
        capabilities=capabilities,
        compatible_targets=compatible_targets,
        compatible_tags=compatible_tags,
    )


def from_dict(data: dict[str, Any]) -> InferenceAgentMetadata:
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


def _metadata_encode_author(author: InferenceAgentAuthor):
    encode_obj = {"name": author.name, "email": author.email}

    if author.contribution is not None:
        encode_obj["contribution"] = author.contribution

    return encode_obj


def _metadata_encode_maintainer(maintainer: InferenceAgentMaintainer):
    return {
        "name": maintainer.name,
        "email": maintainer.email,
    }


def _metadata_encode(metadata: InferenceAgentMetadata):
    encode_obj = {
        "name": metadata.name,
        "version": metadata.version,
        "authors": [_metadata_encode_author(x) for x in metadata.authors],
        "maintainers": [_metadata_encode_maintainer(x) for x in metadata.maintainers],
        "capabilities": list(metadata.capabilities),
        "compatible_targets": list(metadata.compatible_targets),
        "compatible_tags": list(metadata.compatible_tags),
    }

    return encode_obj


def to_file(metadata: InferenceAgentMetadata, json_path: Path):
    json_path = Path(json_path)

    with open(json_path, "w") as fhandle:
        json.dump(_metadata_encode(metadata), fhandle)
