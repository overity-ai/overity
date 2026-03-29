"""
Manage arguments for methods
============================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from overity.model.arguments import (
    ArgumentSchema,
    OptionSchema,
    FlagSchema,
    ListSchema,
)
from overity.backend.flow.ctx import FlowCtx, RunMode

from overity.errors import DuplicateArgumentNameError

from argparse import ArgumentParser as CmdArgs
from typing import Dict, Any


class ArgumentParser:
    def __init__(self, ctx: FlowCtx):
        self.ctx = ctx

        self.schema: Dict[
            str, ArgumentSchema | OptionSchema | FlagSchema | ListSchema
        ] = {}

        self.parsed_vars: Dict[str, Any] = {}  # Parsed arguments values are stored here

    def add_argument(self, name: str, help: str):
        if name in self.schema:
            raise DuplicateArgumentNameError(name)
        self.schema[name] = ArgumentSchema(name=name, help=help)

    def add_option(self, name: str, help: str, default: str):
        if name in self.schema:
            raise DuplicateArgumentNameError(name)
        self.schema[name] = OptionSchema(name=name, help=help, default=default)

    def add_flag(self, name: str, help: str):
        if name in self.schema:
            raise DuplicateArgumentNameError(name)
        self.schema[name] = FlagSchema(name=name, help=help)

    def add_list(self, name: str, help: str):
        if name in self.schema:
            raise DuplicateArgumentNameError(name)
        self.schema[name] = ListSchema(name=name, help=help)

    def _parse_args_standalone(self):
        """Parse arguments in standalone running mode using ArgumentParser from argparse"""

        parser = CmdArgs()

        # Build argument aprser
        for item in self.schema.values():
            if isinstance(item, ArgumentSchema):
                parser.add_argument(item.name, help=item.help)
            elif isinstance(item, OptionSchema):
                parser.add_argument(
                    f"--{item.name}", default=item.default, help=item.help
                )
            elif isinstance(item, FlagSchema):
                parser.add_argument(
                    f"--{item.name}", action="store_true", help=item.help
                )
            elif isinstance(item, ListSchema):
                parser.add_argument(f"--{item.name}", nargs="+", help=item.help)

        # Parse arguments
        args = parser.parse_args()

        # Fill context information
        context = {v.name: getattr(args, v.name) for v in self.schema.values()}

        self.parsed_vars = context

    def _parse_args_interactive(self):
        """Parse arguments in interactive mode"""

        self.parsed_vars = {}

        for item in self.schema.values():
            if isinstance(item, ArgumentSchema):
                self.parsed_vars[item.name] = input(
                    f"Please provide value for argument: {item.name}"
                )
            elif isinstance(item, OptionSchema):
                self.parsed_vars[item.name] = item.default
            elif isinstance(item, FlagSchema):
                # By default all flags disabled
                # TODO: Process differently?
                self.parsed_vars[item.name] = False
            elif isinstance(item, ListSchema):
                # For list arguments, prompt user for multiple values
                values_input = input(
                    f"Please provide values for list argument: {item.name} (space-separated): "
                )
                # Split by whitespace and filter out empty strings
                self.parsed_vars[item.name] = [v for v in values_input.split() if v]

    def parse_args(self):
        if self.ctx.run_mode == RunMode.Standalone:
            self._parse_args_standalone()
        elif self.ctx.run_mode == RunMode.Interactive:
            self._parse_args_interactive()

    def context(self):
        """Return the list of parsed variables as a dict"""

        return self.parsed_vars
