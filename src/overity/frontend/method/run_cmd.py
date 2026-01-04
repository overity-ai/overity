"""
Run method CLI command
======================

**January 2026**

- Florian Dupeyron (florian.dupeyron@elsys-design.com): Initial implementation

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from argparse import ArgumentParser, Namespace

from overity.backend import program as b_program
from overity.backend import method as b_method

from overity.frontend import types
from overity.model.general_info.method import MethodKind


log = logging.getLogger("frontend.method.run_cmd")


def setup_parser(parser: ArgumentParser):
    subcommand = parser.add_parser("run", help="Run a method with the given arguments")

    # Add the --operation flag
    subcommand.add_argument(
        "--operation",
        action="store_true",
        help="Run the method in operation stage (default: preview)",
    )

    # Add the --bench argument for DMQ methods
    subcommand.add_argument(
        "--bench",
        type=str,
        help="Bench slug for measurement/qualification methods (sets OVERITY_BENCH environment variable)",
    )

    # Add method kind argument
    subcommand.add_argument(
        "method_kind",
        type=types.parse_method_kind,
        help="Method kind: training-optimization (to), measurement-qualification (mq), deployment (dp), or analysis (an)",
    )

    # Add method slug argument
    subcommand.add_argument(
        "method_slug", type=str, help="Slug (identifier) of the method to run"
    )

    # Add method arguments (remaining arguments)
    subcommand.add_argument(
        "method_arguments", nargs="*", help="Additional arguments to pass to the method"
    )

    return subcommand


def run(args: Namespace):
    """Run the method execution procedure."""

    cwd = Path.cwd()

    # Step 1: Identify the current program folder
    log.debug(f"Finding current program from {cwd}")
    program_path = b_program.find_current(start_path=cwd)
    log.info(f"Found program at: {program_path}")

    # Step 2: Find the path to the method script
    log.debug(f"Finding method path for {args.method_kind.value}/{args.method_slug}")
    method_path = b_method.find_method_path(
        program_path, args.method_kind, args.method_slug
    )
    log.info(f"Found method at: {method_path}")

    # Step 3: Set the OVERITY_STAGE environment variable
    stage = "operation" if args.operation else "preview"
    os.environ["OVERITY_STAGE"] = stage
    log.info(f"Set OVERITY_STAGE to: {stage}")

    # Step 4: If necessary, set the OVERITY_BENCH variable for DMQ methods
    if args.method_kind == MethodKind.MeasurementQualification:
        if args.bench:
            os.environ["OVERITY_BENCH"] = args.bench
            log.info(f"Set OVERITY_BENCH to: {args.bench}")
        else:
            # Check if OVERITY_BENCH is already set
            if "OVERITY_BENCH" not in os.environ:
                log.error(
                    "Measurement/qualification methods require --bench argument or OVERITY_BENCH environment variable"
                )
                sys.exit(1)

    # Step 5: Run the method script directly
    log.info(
        f"Running method {args.method_slug} with arguments: {args.method_arguments}"
    )

    # Change working directory to method directory (as per specification)
    method_dir = method_path.parent
    original_cwd = Path.cwd()

    try:
        os.chdir(method_dir)

        # Execute the method script using subprocess (as per specification)
        cmd = [sys.executable, str(method_path)] + args.method_arguments

        log.info(f"Executing command: {' '.join(cmd)}")

        # Run the method and capture exit code
        result = subprocess.run(cmd, capture_output=False, text=True)

        # Exit with the return code of the run command (Step 5)
        sys.exit(result.returncode)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
