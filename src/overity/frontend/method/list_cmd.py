"""
List available methods of a certain kind
========================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
import sys
import traceback

from argparse import ArgumentParser, Namespace
from pathlib import Path

from overity.backend import method as b_method
from overity.backend import program as b_program

from overity.errors import ProgramNotFound

from overity.frontend import types
from overity.frontend.utils import table as f_table

from overity.model.general_info.method import MethodKind


log = logging.getLogger("frontend.method.list_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "list", aliases=["ls"], help="List available methods of a certain kind"
    )
    subcommand.add_argument(
        "kind", type=types.parse_method_kind, help="What method kind to list"
    )

    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = b_program.find_current(start_path=cwd)

        # List available methods
        methods, errors = [], None
        if args.kind == MethodKind.TrainingOptimization:
            methods, errors = b_method.list_topt_methods(pdir)
        elif args.kind == MethodKind.MeasurementQualification:
            methods, errors = b_method.list_measurement_qualification_methods(pdir)
        else:
            log.error(f"Unimplemented kind list: {args.kind}")
            sys.exit(1)

        # Display results
        print("")
        print(f"Found the following methods in {pdir}:")
        print("")

        headers = ("Slug", "Display name", "Path")
        rows = (
            (mtd.slug, mtd.display_name, mtd.path.relative_to(pdir)) for mtd in methods
        )

        print(f_table.table_format(headers, rows))

        if errors:
            print("")
            print("While processing, the following errors has been found:")
            print("")
            for fpath, err in errors:
                print(f"- in {fpath.relative_to(pdir)!s}: {err!s}")

    except ProgramNotFound as exc:
        log.exception(str(exc))
        log.debug(traceback.format_exc())
