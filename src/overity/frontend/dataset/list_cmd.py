"""
List available datasets in a given programme
============================================

**August 2025**

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
from overity.backend import dataset as b_dataset

from overity.errors import ProgramNotFound
from overity.frontend.utils import table as f_table

log = logging.getLogger("frontend.dataset.list_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "list", aliases=["ls"], help="List available datasets"
    )

    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = b_program.find_current(start_path=cwd)

        datasets, errors = b_dataset.list_datasets(pdir)

        # Displaying results
        print("")
        print(f"Found the following datasets in {pdir}:")
        print("")

        headers = ("Dataset slug", "Dataset name", "SHA256 checksum")
        rows = (
            (
                dataset_slug,
                dataset_info.name,
                checksum[:16] + "..." if len(checksum) > 16 else checksum,
            )
            for dataset_slug, dataset_info, checksum in datasets
        )

        print(f_table.table_format(headers, rows))

        if errors:
            print("")
            print("While processing, the following errors has been found:")
            print("")

            for slug, err in errors:
                print(f"- in {slug}: {err!s}")

    except ProgramNotFound as exc:
        log.exception(exc)
        log.debug(traceback.format_exc())
