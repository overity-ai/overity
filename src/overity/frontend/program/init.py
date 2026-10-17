"""
Initialize program
==================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path

from overity.backend import program as program_backend

log = logging.getLogger("frontend.program.init")


def setup_parser(parser: ArgumentParser):
    subparser = parser.add_parser("init", help="Initialize program")

    subparser.add_argument("display_name", help="Descriptive name of programme")
    subparser.add_argument("initiator_name", help="Name of initiator")
    subparser.add_argument("initiator_email", help="Email of initiator")
    subparser.add_argument("initiator_role", help="Role of initiator")
    subparser.add_argument(
        "--slug", help="Optional program slug, will init in subfolder if given"
    )
    subparser.add_argument("--description", help="Optional program description")


def run(args: Namespace):
    cwd = Path.cwd()
    dest_path = (cwd / args.slug).resolve() if args.slug else cwd

    # Gather program information
    slug = dest_path.stem
    display_name = args.display_name
    initiator_name = args.initiator_name
    initiator_email = args.initiator_email
    initiator_role = args.initiator_role
    date_created = date.today()

    print(f"Initializing programme: {slug} - {display_name}")
    print(f"- Initiator: {initiator_name} ({initiator_email}): {initiator_role}")
    print(f"- Created: {date_created!s}")

    program_backend.initialize(
        dest_path,
        slug=slug,
        display_name=display_name,
        initiator_name=initiator_name,
        initiator_email=initiator_email,
        initiator_role=initiator_role,
        date_created=date_created,
    )
