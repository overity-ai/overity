"""
Command subgroup for program related manipulation
=================================================

**March 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from overity.frontend.program import infos, init

CLI_SUBCOMMANDS = {"infos": infos, "init": init}


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "program", aliases=["prg"], help="Program manipulation"
    )
    subparsers = subcommand.add_subparsers(dest="program_subcommand")

    for cmd in CLI_SUBCOMMANDS.values():
        cmd.setup_parser(subparsers)

    return subcommand


def run(args: Namespace):
    k_cmd = args.program_subcommand

    if k_cmd in CLI_SUBCOMMANDS:
        CLI_SUBCOMMANDS[k_cmd].run(args)
