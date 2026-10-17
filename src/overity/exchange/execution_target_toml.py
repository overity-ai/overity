"""
Execution target storage in TOML format
=======================================

**February 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from pathlib import Path

import jsonschema
import toml

from overity.model.general_info.execution_target import ExecutionTarget

####################################################
# Validation schema
####################################################

SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "object",
            "properties": {
                # TODO # Slug is deduced for toml file name in local storage mode
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "name",
            ],
        }
    },
    "required": ["target"],
}


####################################################
# Decoder
####################################################


def _execution_target_decode(slug: str, data: dict[str, any]):
    target_info = data["target"]

    return ExecutionTarget(
        slug=slug,
        display_name=target_info["name"],
        description=target_info.get("description", None),
        tags=target_info.get("tags", None),
    )


def from_file(toml_path: Path):
    """Decode the given file containing execution target information"""

    toml_path = Path(toml_path)

    # Parse TOML data
    with open(toml_path) as fhandle:
        data = toml.load(fhandle)

    # Validate data
    jsonschema.validate(data, SCHEMA)

    # Get slug from file name
    slug = toml_path.stem

    # Parse information
    return _execution_target_decode(slug, data)


####################################################
# Encoder
####################################################

# TODO #
