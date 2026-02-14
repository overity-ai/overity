"""
List available models for a given programme
===========================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
import traceback

from argparse import ArgumentParser, Namespace
from pathlib import Path

from overity.backend import program as b_program
from overity.backend import model as b_model

from overity.errors import ProgramNotFound

from overity.frontend.utils import table as f_table


log = logging.getLogger("frontend.method.list_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser("list", aliases=["ls"], help="List available models")

    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = b_program.find_current(start_path=cwd)

        models, errors = b_model.list_models(pdir)

        # Displaying results
        print("")
        print(f"Found the following models in {pdir}:")
        print("")

        headers = ("Model slug", "Model name")
        rows = (
            (
                mod_slug,
                mod_info.name,
            )
            for mod_slug, mod_info in models
        )

        print(f_table.table_format(headers, rows))

        if errors:
            print("")
            print("While processing, the following errors has been found:")
            print("")
            for slug, err in errors:
                print(f"- in {slug!s}: {err!s}")

    except ProgramNotFound as exc:
        log.exception(str(exc))
        log.debug(traceback.format_exc())
