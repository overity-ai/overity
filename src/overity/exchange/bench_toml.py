"""
Bench toml definition parser
============================

**September 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from pathlib import Path

import jsonschema
import toml

from overity.model.general_info.bench import BenchInstanciationMetadata

####################################################
# Validation schema
####################################################

SCHEMA = {
    "type": "object",
    "properties": {
        "bench": {
            "type": "object",
            "properties": {
                # slug is deduced from file name,
                "name": {"type": "string"},  # Bench display name
                "description": {"type": "string"},  # Bench optional description text
                "abstraction": {"type": "string"},  # Used bench abstraction slug
                "settings": {
                    "type": "object",
                    "additionalProperties": {
                        "type": ["string", "boolean", "number", "integer"]
                    },
                },
            },
            "required": [
                "name",
                "abstraction",
            ],
        }
    },
    "required": ["bench"],
}


####################################################
# Decoder
####################################################


def _bench_instanciation_decode(slug: str, data: dict):
    bench_info = data["bench"]

    return BenchInstanciationMetadata(
        slug=slug,
        display_name=bench_info["name"],
        abstraction_slug=bench_info["abstraction"],
        settings=bench_info.get("settings"),
        description=bench_info.get("description"),
    )


def from_file(toml_path: Path):
    """Decode the given file containing bench instanciation information"""

    toml_path = Path(toml_path)

    # Parse TOML data
    with open(toml_path) as fhandle:
        data = toml.load(fhandle)

    # Validate data
    jsonschema.validate(data, SCHEMA)

    # Get slug from file name
    slug = toml_path.stem

    # Parse information
    return _bench_instanciation_decode(slug, data)
