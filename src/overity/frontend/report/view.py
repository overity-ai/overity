"""
Command to view a report
========================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from overity.backend import program as b_program
from overity.backend import report as b_report
from overity.backend.report_view import simple_server, topt_html
from overity.frontend import types

log = logging.getLogger("frontend.report.view")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "view", aliases=["v"], help="View a report in an HTML page"
    )
    subcommand.add_argument(
        "kind",
        type=types.parse_report_kind,
        help="What report kind to view a report from",
    )
    subcommand.add_argument("report_id", type=str, help="Report identifier")

    return subcommand


def run(args: Namespace):
    cwd = Path.cwd()
    pdir = b_program.find_current(start_path=cwd)

    report_path, report_data = b_report.load(pdir, args.kind, args.report_id)

    # Render template
    report_html = topt_html.render(report_data, report_path)

    # Serve...
    try:
        print("Serving at http://localhost:3000... Press [Ctrl+C] to exit")
        simple_server.serve(report_html, port=3000)
    except KeyboardInterrupt:
        print("Bye!")
