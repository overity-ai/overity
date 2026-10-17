"""
Show program informations
=========================

**March 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from textwrap import wrap

from overity.backend import program as program_backend
from overity.errors import ProgramNotFound

log = logging.getLogger("frontend.program.infos")


def setup_parser(parser: ArgumentParser):
    return parser.add_parser(
        "infos", help="Get informations on program in current folder"
    )


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = program_backend.find_current(start_path=cwd)
        prginfo = program_backend.infos(pdir)

        title_str = f"Program: {prginfo.display_name}"
        print()
        print(title_str)
        print("=" * len(title_str))
        print()
        print(f"- Slug:         {prginfo.slug}")
        print(f"- Created:      {prginfo.date_created}")
        print(
            f"- Initiated by: {prginfo.initiator.name} ({prginfo.initiator.email})"
            + (f": {prginfo.initiator.role}" if prginfo.initiator.role else "")
        )
        print()

        if prginfo.description is not None:
            print("\n".join(wrap(prginfo.description, width=70)))

    except ProgramNotFound:
        log.exception("Program not found")
        log.debug(traceback.format_exc())
