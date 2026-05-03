"""
Overity.ai command to prune unsucessful reports
===============================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from argparse import ArgumentParser, Namespace
from pathlib import Path
from functools import partial

from overity.backend import report as b_report
from overity.backend import program as b_program
from overity.frontend import types

from overity.model.report import (
    MethodReportKind,
    MethodExecutionStatus,
    MethodExecutionStage,
)

log = logging.getLogger("frontend.report.prune")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser(
        "prune", help="Remove reports that are not succesful"
    )
    subcommand.add_argument(
        "kind", type=types.parse_report_kind, help="What report kind to prune"
    )

    subcommand.add_argument(
        "--keep-preview",
        dest="keep_preview",
        action="store_true",
        help="Do not remove reports generated in the preview phase",
    )

    subcommand.add_argument(
        "--keep-intruders",
        dest="keep_intruders",
        action="store_true",
        help="Do not remove reports that can't be parsed or are not report json files",
    )

    subcommand.add_argument(
        "--remove-failed-exec",
        dest="remove_failed_exec",
        action="store_true",
        help="Remove reports that have an execution failure status",
    )

    subcommand.add_argument(
        "--remove-failed-constraints",
        dest="remove_failed_constraints",
        action="store_true",
        help="Remove reports that have an constraints failure status",
    )

    return subcommand


def _check_report(
    uuid: str,
    pdir: Path,
    kind: MethodReportKind,
    keep_preview: bool,
    keep_intruders: bool,
    remove_failed_exec: bool,
    remove_failed_constraints: bool,
):
    """Check if the remort needs to be kept or not. Return True if removed"""

    # Try to parse report information
    try:
        report_path, report_info = b_report.load_info(pdir, kind, uuid)

        if (report_info.stage == MethodExecutionStage.Preview) and (not keep_preview):
            print("Remove preview")
            return True

        elif (
            report_info.status == MethodExecutionStatus.ExecutionFailureException
        ) and remove_failed_exec:
            print("remove failed exec")
            return True

        elif (
            report_info.status == MethodExecutionStatus.ExecutionFailureConstraints
        ) and remove_failed_constraints:
            print("Remove failed constraints")
            return True

        else:
            print("Keep report")
            return False

    except Exception:
        # TODO Log the error?
        return not keep_intruders


def run(args: Namespace):
    cwd = Path.cwd()
    pdir = b_program.find_current(start_path=cwd)

    reports = b_report.list(pdir, kind=args.kind)

    reports_to_remove = filter(
        partial(
            _check_report,
            pdir=pdir,
            kind=args.kind,
            keep_preview=args.keep_preview,
            keep_intruders=args.keep_intruders,
            remove_failed_exec=args.remove_failed_exec,
            remove_failed_constraints=args.remove_failed_constraints,
        ),
        reports,
    )

    for report in reports_to_remove:
        print(report)
        b_report.remove(pdir, args.kind, report)
