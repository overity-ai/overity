"""
List available agents in a given programme
==========================================

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
from overity.backend import inference_agent as b_agent

from overity.errors import ProgramNotFound

from overity.frontend.utils import table as f_table


log = logging.getLogger("frontend.inference_agent.list_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "list", aliases=["ls"], help="List available inference agents"
    )
    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()

    try:
        pdir = b_program.find_current(start_path=cwd)

        agents, errors = b_agent.list_agents(pdir)

        # Displaying results
        print("")
        print(f"Found the following agents in {pdir}:")
        print("")

        headers = ("Agent slug", "Agent name")
        rows = (
            (
                agt_slug,
                agt_info.name,
            )
            for agt_slug, agt_info in agents
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
