"""
Command subgroup for methods related manipulations
==================================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from argparse import ArgumentParser, Namespace

from overity.frontend.method import list_cmd, run_cmd

CLI_SUBCOMMANDS = {list_cmd, run_cmd}


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "method", aliases=["mtd"], help="Method manipulation"
    )
    subparsers = subcommand.add_subparsers(dest="method_subcommand")

    for cmd in CLI_SUBCOMMANDS:
        subp = cmd.setup_parser(subparsers)
        subp.set_defaults(method_subcommand_clbk=cmd)

    return subcommand


def run(args: Namespace):
    if hasattr(args, "method_subcommand_clbk"):
        args.method_subcommand_clbk.run(args)
