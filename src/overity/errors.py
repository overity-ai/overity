"""
Overity.ai toolkit errors
=========================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from pathlib import Path

from overity.model.report import MethodReportKind
from overity.model.general_info.method import MethodKind


class EmptyMethodDescription(Exception):
    def __init__(self, file_path: Path):
        super().__init__(f"No method description found in file: {file_path!s}")
        self.file_path = file_path


class ProgramNotFound(Exception):
    def __init__(self, start_path: Path, recursive: bool = False):
        super().__init__(
            f"No program storage found starting from {start_path}. Recursive search was {'on' if recursive else 'off'}"
        )


class MethodNotFound(Exception):
    def __init__(self, kind: MethodKind, slug: str):
        super().__init__(
            f"Can't find method {slug!r} of kind {kind.value} in current program"
        )


class ModelNotFound(Exception):
    def __init__(self, slug: str):
        super().__init__(f"Model '{slug}' not found")


class AgentNotFound(Exception):
    def __init__(self, slug: str):
        super().__init__(f"Agent '{slug}' not found")


class DatasetNotFound(Exception):
    def __init__(self, slug: str):
        super().__init__(f"Dataset '{slug}' not found")


class DuplicateSlugError(Exception):
    def __init__(self, path: Path, slug: str):
        super().__init__(f"Duplicate slug found in {path!s}: {slug!s}")

        self.path = path
        self.slug = slug


class UnidentifiedMethodError(Exception):
    def __init__(self, path: Path):
        super().__init__(f"Method in {path} can't be identified")

        self.path = path


class UnknownMethodError(Exception):
    def __init__(self):
        super().__init__("Can't find called method file path")


class NotInDMQError(Exception):
    def __init__(self):
        super().__init__(
            "Current running method is not a measurement/qualification method"
        )


class UninitAPIError(Exception):
    def __init__(self):
        super().__init__("Please initialize API with overity.api.init() before using")


class ArgumentNotFoundError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Argument '{name}' not providen")

        self.name = name


class DuplicateArgumentNameError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Argument '{name}' is already defined")


class MalformedModelPackage(Exception):
    def __init__(self, archive_path: Path, what: str):
        super().__init__(f"Malformed archive package in {archive_path}: {what}")

        self.archive_path = archive_path
        self.what = what


class MalformedAgentPackage(Exception):
    def __init__(self, archive_path: Path, what: str):
        super().__init__(f"Malformed agent package in {archive_path}: {what}")

        self.archive_path = archive_path
        self.what = what


class MalformedDatasetPackage(Exception):
    def __init__(self, archive_path: Path, what: str):
        super().__init__(f"Malformed dataset package in {archive_path}: {what}")

        self.archive_path = archive_path
        self.what = what


class ReportNotFound(Exception):
    def __init__(
        self, program_slug: str, report_type: MethodReportKind, identifier: str
    ):
        super().__init__(
            f"Report '{identifier}' of type '{report_type.value}' from program '{program_slug}' not found"
        )

        self.program_slug = program_slug
        self.report_type = report_type
        self.identifier = identifier


class InvalidBenchSettingsError(Exception):
    def __init__(
        self,
        bench_slug: str,
        bench_settings: dict[str, str | int | float | bool],
        exc: Exception,
    ):
        super().__init__(
            f"Failed to parse bench settings for bench {bench_slug} ({type(exc)}): {exc!s}"
        )

        self.bench_slug = bench_slug
        self.bench_settings = bench_settings
        self.exc = exc


class BenchInstanciationError(Exception):
    def __init__(
        self,
        bench_slug: str,
        bench_settings: dict[str, str | int | float | bool],
        exc: Exception,
    ):
        super().__init__(
            f"Failed to instanciate bench {bench_slug} ({type(exc)}): {exc!s}"
        )

        self.bench_slug = bench_slug
        self.bench_settings = bench_settings
        self.exc = exc


class NoBenchDefinedError(Exception):
    def __init__(self):
        super().__init__(
            "No bench defined to execute the measure/qualification method. Please set the OVERITY_BENCH variable to an existing bench in programme"
        )


class InvalidEpochValue(Exception):
    def __init__(self, epoch: int):
        super().__init__(
            f"Invalid epoch value: {epoch}. Epoch must be a non-negative integer."
        )
        self.epoch = epoch
