"""
Program information storage in TOML format
==========================================

**February 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from datetime import date
from pathlib import Path

import jsonschema
import toml

from overity.model.general_info.program import ProgramInfo, ProgramInitiator

####################################################
# Validation schema
####################################################

SCHEMA = {
    "type": "object",
    "properties": {
        "program": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "display_name": {"type": "string"},
                "description": {"type": "string"},
                "date_created": {"type": "string", "format": "date-time"},
                "active": {"type": "boolean"},
            },
            "required": ["slug", "display_name", "date_created", "active"],
        },
        "initiator": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "role": {"type": "string"},
            },
            "required": ["name", "email"],
        },
    },
    "required": ["program", "initiator"],
}


####################################################
# Decoder
####################################################


def _program_decode_initiator(data: dict[str, any]):
    return ProgramInitiator(
        name=data["name"], email=data["email"], role=data.get("role")
    )


def _program_decode(data: dict[str, any]):
    program_info = data["program"]
    initiator_info = data["initiator"]

    initiator = _program_decode_initiator(initiator_info)
    return ProgramInfo(
        slug=program_info["slug"],
        display_name=program_info["display_name"],
        description=program_info.get("description"),
        date_created=date.fromisoformat(program_info["date_created"]),
        initiator=initiator,
        active=program_info["active"],
    )


def from_file(toml_path: Path):
    """Decode the given file containing program information"""

    # Parse TOML data
    with open(toml_path) as fhandle:
        data = toml.load(fhandle)

    # Validate data
    jsonschema.validate(data, SCHEMA)

    # Parse information
    return _program_decode(data)


####################################################
# Encoder
####################################################


def _program_encode_initiator(data: ProgramInitiator):
    result = {"name": data.name, "email": data.email}

    if data.role:
        result["role"] = data.role

    return result


def _program_encode(data: ProgramInfo):
    result = {
        "slug": data.slug,
        "display_name": data.display_name,
        "date_created": data.date_created.isoformat(),
        "active": data.active,
    }

    if data.description:
        result["description"] = data.description

    return result


def to_file(program_info: ProgramInfo, toml_path: Path):
    toml_path = Path(toml_path)

    # Create output dictionary
    result_dict = {
        "program": _program_encode(program_info),
        "initiator": _program_encode_initiator(program_info.initiator),
    }

    # Validate against schema
    jsonschema.validate(result_dict, SCHEMA)

    # Output data to toml file
    with open(toml_path, "w") as fhandle:
        toml.dump(result_dict, fhandle)
