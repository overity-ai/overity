"""
Capability description storage in TOML format
=============================================

**June 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from pathlib import Path

import jsonschema
import toml

from overity.model.general_info.capability import Capability

####################################################
# Validation schema
####################################################

SCHEMA = {
    "type": "object",
    "properties": {
        "capability": {
            "type": "object",
            "properties": {
                # TODO # Slug is deduced from toml file name in local storage mode
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": [
                "name",
            ],
        }
    },
    "required": ["capability"],
}


####################################################
# Decoder
####################################################


def _capability_decode(slug: str, data: dict):
    capability_info = data["capability"]

    return Capability(
        slug=slug,
        display_name=capability_info["name"],
        description=capability_info.get("description", None),
    )


def from_file(toml_path: Path):
    """Decode the given file containing capability information"""

    toml_path = Path(toml_path)

    # Parse TOML data
    with open(toml_path) as fhandle:
        data = toml.load(fhandle)

    # Validate data
    jsonschema.validate(data, SCHEMA)

    # Get slug from file name
    slug = toml_path.stem

    # Parse information
    return _capability_decode(slug, data)


####################################################
# Encoder
####################################################

# TODO #
