"""
List available bench instanciations in a given programme
========================================================

**September 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path

from overity.backend import bench as b_bench
from overity.backend import program as b_program
from overity.errors import ProgramNotFound
from overity.frontend.utils import table as f_table

log = logging.getLogger("frontend.bench.list_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "list", aliases=["ls"], help="List available bench instanciations"
    )
    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = b_program.find_current(start_path=cwd)
        benches, errors = b_bench.list_benches(pdir)

        # Displaying results
        print()
        print(f"Found the following benches in {pdir}:")
        print()

        headers = ("Bench slug", "Bench name", "Description")
        rows = (
            (b_slug, b_info.display_name, b_info.description or "")
            for b_slug, b_info in benches
        )

        print(f_table.table_format(headers, rows))

        if errors:
            print()
            print("While processing, the following errors has been found:")
            print()
            for slug, err in errors:
                print(f"- in {slug}: {err!s}")

    except ProgramNotFound as exc:
        log.excpetion(exc)
        log.debug(traceback.format_exc())
