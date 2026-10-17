"""
Command subgroup for bench related manipulations
================================================

**September 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from argparse import ArgumentParser, Namespace

from overity.frontend.bench import list_abstractions_cmd, list_cmd

CLI_SUBCOMMANDS = {list_cmd, list_abstractions_cmd}


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "bench",
        aliases=["bc"],
        help="Test bench manipulation",
    )
    subparsers = subcommand.add_subparsers(dest="bench_subcommand")

    for cmd in CLI_SUBCOMMANDS:
        subp = cmd.setup_parser(subparsers)
        subp.set_defaults(bench_subcommand_clbk=cmd)

    return subcommand


def run(args: Namespace):
    if hasattr(args, "bench_subcommand_clbk"):
        args.bench_subcommand_clbk.run(args)
