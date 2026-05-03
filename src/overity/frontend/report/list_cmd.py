"""
List reports of a given kind for a program
==========================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from argparse import ArgumentParser, Namespace
from pathlib import Path

from overity.backend import report as b_report
from overity.backend import program as b_program
from overity.frontend import types


log = logging.getLogger("frontend.report.list")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "list", aliases=["ls"], help="List reports of a certain kind"
    )

    subcommand.add_argument(
        "kind", type=types.parse_report_kind, help="What report kind to list"
    )

    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()
    pdir = b_program.find_current(start_path=cwd)
    reports = b_report.list(pdir, kind=args.kind)

    for rp in reports:
        print(rp)
