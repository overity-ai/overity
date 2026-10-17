"""
Overity.ai frontend commands to manipulate reports
==================================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from argparse import ArgumentParser, Namespace

from overity.frontend.report import list_cmd, prune, view

CLI_SUBCOMMANDS = {view, list_cmd, prune}


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "report", aliases=["rpt"], help="Report manipulation"
    )
    subparsers = subcommand.add_subparsers(dest="report_subcommand")

    for cmd in CLI_SUBCOMMANDS:
        subp = cmd.setup_parser(subparsers)
        subp.set_defaults(report_subcommand_clbk=cmd)

    return subcommand


def run(args: Namespace):
    if hasattr(args, "report_subcommand_clbk"):
        args.report_subcommand_clbk.run(args)
