"""
Overity.ai Method flow management
=================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import atexit
import logging
import tempfile
import sys
import traceback

from pathlib import Path
from functools import partial

from datetime import datetime as dt
from dataclasses import dataclass

from overity.backend import program
from overity.backend import bench as b_bench

from overity.storage.local import LocalStorage

from overity.model.general_info.method import MethodKind
from overity.model.report import (
    MethodExecutionStatus,
    MethodExecutionStage,
    MethodReport,
)

from overity.model.ml_model.metadata import (
    MLModelAuthor,
    MLModelMaintainer,
    MLModelMetadata,
)
from overity.model.ml_model.package import MLModelPackage

from overity.model.dataset.metadata import (
    DatasetAuthor,
    DatasetMaintainer,
    DatasetMetadata,
)

from overity.model.dataset.package import DatasetPackageInfo

from overity.model.traceability import (
    ArtifactKind,
    ArtifactKey,
    ArtifactLinkKind,
    ArtifactLink,
)

from overity.model.report.metrics import (
    Metric,
    SimpleValue,
    LinScaleValue,
    LinRangeValue,
    PercentageValue,
)

from overity.exchange import report_json
from overity.exchange.method_common import file_py, file_ipynb
from overity.errors import (
    UnidentifiedMethodError,
    UninitAPIError,
    NotInDMQError,
    InvalidEpochValue,
)

from overity.backend.flow.ctx import FlowCtx, RunMode
from overity.backend.flow import environment as b_env
from overity.backend.flow.arguments import ArgumentParser

from contextlib import contextmanager


log = logging.getLogger("backend.flow")


class LogArrayHandler(logging.Handler):
    def __init__(self, report: MethodReport):
        super().__init__()

        self.report = report

    def emit(self, record):
        self.report.log_add(
            tstamp=dt.now(),
            severity=record.levelname,
            source=f"{record.filename}:{record.lineno}",
            message=record.getMessage(),
        )


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.strip()

        # Remove any trailing newlines to avoid empty log entries
        if message not in ["", "^"]:
            self.logger.log(self.level, message.strip())

    def flush(self):
        # Flush method is required but can be a no-op
        pass


@dataclass
class ModelPackageInfo:
    model_metadata: MLModelMetadata
    model_file_path: Path  # Path to store model file
    inference_example_path: Path  # Optional folder path to store inference example


def _api_guard(fkt):
    def call(ctx, *args, **kwargs):
        if not ctx.init_ok:
            raise UninitAPIError()
        else:
            return fkt(ctx, *args, **kwargs)

    return call


def _dmq_guard(fkt):
    def call(ctx, *args, **kwargs):
        if not ctx.method_kind == MethodKind.MeasurementQualification:
            raise NotInDMQError()
        else:
            return fkt(ctx, *args, **kwargs)

    return call


def init(ctx: FlowCtx, method_path: Path, run_mode: RunMode):
    log.info(f"Initialize API for method {method_path}")
    date_started = dt.now()

    # Set run mode
    ctx.run_mode = run_mode
    log.info(f"Running in mode: {run_mode}")

    # Detect execution stage
    ctx.stage = b_env.execution_stage()
    log.info(f"Running in stage: {ctx.stage}")

    # Get current programme
    ctx.pdir = program.find_current(start_path=method_path.parent)
    log.info(f"Programme directory: {ctx.pdir}")
    ctx.pinfo = program.infos(ctx.pdir)
    log.info(f"Programme slug: {ctx.pinfo.slug}")
    log.info(f"Programme name: {ctx.pinfo.display_name}")

    # Init local storage
    ctx.storage = LocalStorage(ctx.pdir)
    ctx.storage.initialize()  # Ensure all folders exist!

    # Identify method slug and kind
    ctx.method_path = method_path
    ctx.method_slug = ctx.storage.identify_method_slug(method_path)
    ctx.method_kind = ctx.storage.identify_method_kind(method_path)

    # Read method information
    ctx.method_info = method_info_get(ctx)  # Read method information from file

    log.info(f"Method identification: {ctx.method_slug} ({ctx.method_kind.value})")

    # Initialize report and environment information
    ctx.report = MethodReport.default(
        uuid=ctx.storage.method_report_uuid_get(ctx.method_kind),
        program=ctx.pinfo.slug,
        method_info=ctx.method_info,
        date_started=date_started,
    )

    ctx.report.environment = {
        "installed_packages": b_env.installed_packages(),
        **b_env.platform_info(),
    }

    # Add lib folder to python path
    log.info(f"Add lib folder to python path: {ctx.storage.lib()}")
    sys.path.append(str(ctx.storage.lib()))

    # Initialize run traceability information
    # TODO: For other types
    # TODO: This is ugly.
    if ctx.method_kind == MethodKind.TrainingOptimization:
        ctx.report.method_key = ArtifactKey(
            kind=ArtifactKind.TrainingOptimizationMethod, id=ctx.method_slug
        )
        ctx.report.report_key = ArtifactKey(
            kind=ArtifactKind.OptimizationReport, id=ctx.report.uuid
        )
        ctx.report.run_key = ArtifactKey(
            kind=ArtifactKind.OptimizationRun, id=ctx.report.uuid
        )

        # Add link between run and report
        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=ctx.report.report_key,
                b=ctx.report.run_key,
                kind=ArtifactLinkKind.ReportFor,
            )
        )

        # Add link between run and method
        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=ctx.report.run_key,
                b=ctx.report.method_key,
                kind=ArtifactLinkKind.MethodUse,
            )
        )

    elif ctx.method_kind == MethodKind.MeasurementQualification:
        ctx.report.method_key = ArtifactKey(
            kind=ArtifactKind.MeasurementQualificationMethod, id=ctx.method_slug
        )
        ctx.report.report_key = ArtifactKey(
            kind=ArtifactKind.ExecutionReport, id=ctx.report.uuid
        )
        ctx.report.run_key = ArtifactKey(
            kind=ArtifactKind.ExecutionRun, id=ctx.report.uuid
        )

        # Add link between run and report
        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=ctx.report.report_key,
                b=ctx.report.run_key,
                kind=ArtifactLinkKind.ReportFor,
            )
        )

        # Add link between run and method
        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=ctx.report.run_key,
                b=ctx.report.method_key,
                kind=ArtifactLinkKind.MethodUse,
            )
        )

    else:
        raise NotImplementedError

    # Initialize logger
    root_log = logging.getLogger("")
    root_log.addHandler(LogArrayHandler(report=ctx.report))

    stdout_log = logging.getLogger("stdout")
    stderr_log = logging.getLogger("stderr")

    # Redirect stdout and stderr...
    sys.stdout = LoggerWriter(stdout_log, logging.INFO)
    sys.stderr = LoggerWriter(stderr_log, logging.ERROR)

    # Set exception handler to store exceptions in context
    sys.excepthook = partial(exception_handler, ctx)

    # Add exit handler to save report file
    atexit.register(exit_handler, ctx)

    # For DMQ Method, extract bench information
    if ctx.method_kind == MethodKind.MeasurementQualification:
        bench_slug = (
            b_env.bench()
        )  # Get bench slug from OVERITY_BENCH env var, raise exception if not defined
        log.info(
            f"Running a measurement/qualification method. Load bench information for {bench_slug}"
        )

        ctx.bench_infos = b_bench.load_bench_infos(ctx.pdir, bench_slug)
        ctx.bench_abstraction = b_bench.load_bench_abstraction_infos(
            ctx.pdir, ctx.bench_infos.abstraction_slug
        )
        ctx.bench_instance = b_bench.instanciate(
            program_path=ctx.pdir,
            bench_slug=bench_slug,
            storage=ctx.storage,
        )

        # -> Traceability information
        k_bench = ArtifactKey(kind=ArtifactKind.BenchInstanciation, id=bench_slug)
        k_abstr = ArtifactKey(
            kind=ArtifactKind.BenchAbstraction, id=ctx.bench_infos.abstraction_slug
        )

        ctx.report.traceability_graph.add(
            ArtifactLink(a=k_bench, b=k_abstr, kind=ArtifactLinkKind.InstanciateBench)
        )
        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=ctx.report.run_key, b=k_bench, kind=ArtifactLinkKind.BenchUse
            )
        )

        # -> Log some infos
        log.info("Bench information:")
        log.info(
            f" - Name:               {ctx.bench_infos.display_name} ({bench_slug})"
        )
        log.info(
            f" - Instanciates:       {ctx.bench_abstraction.display_name} ({ctx.bench_abstraction.slug})"
        )
        log.info(f" - Compatible tags:    {ctx.bench_instance.compatible_tags}")
        log.info(f" - Compatible targets: {ctx.bench_instance.compatible_targets}")
        log.info(f" - Capabilities:       {ctx.bench_instance.capabilities}")

        # -> Start bench
        log.info("Start the bench")
        ctx.bench_instance.bench_start()

        # -> Sanity check
        log.info("Sanity check for bench")
        ctx.bench_instance.sanity_check()

        # -> Set initial state
        log.info("Set initial state for bench")
        ctx.bench_instance.state_initial()

    # Init is done!
    ctx.init_ok = True


def exception_handler(ctx: FlowCtx, exc_type, exc_value, exc_traceback):
    # exc_type can be retrieved using type(exc_value), and exc_traceback
    # can be retrieved using exc_value.__traceback__
    ctx.exceptions.append(exc_value)
    log.error("".join(traceback.format_exception(exc_value)))
    # log.error(f"Got Error of type {exc_type}: {exc_value}")


def exit_handler(ctx: FlowCtx):
    log.info("Exiting method execution")

    if ctx.method_kind == MethodKind.MeasurementQualification:
        # Bench cleanup if in DMQ method
        log.info("Bench cleanup")
        try:
            if ctx.bench_instance:
                ctx.bench_instance.bench_cleanup()
                ctx.bench_instance.tmpdir_cleanup()

        except Exception as exc:
            log.error("Error when trying to stop the bench")
            ctx.exceptions.append(exc)
            log.error("".join(traceback.format_exception(exc)))

        # Merge bench traceability graph into main report graph
        ctx.report.traceability_graph += ctx.bench_instance.traceability_graph

    # Set end date
    ctx.report.date_ended = dt.now()

    # Set return status
    # TODO: Manage constraints failure status. For now only checking
    # if execution OK without exception.

    if len(ctx.exceptions):
        log.error("Failed method execution with errors.")
        ctx.report.status = MethodExecutionStatus.ExecutionFailureException
    else:
        ctx.report.status = MethodExecutionStatus.ExecutionSuccess

    # Save report file
    output_path = ctx.storage.method_run_report_path(ctx.report.uuid, ctx.method_kind)
    log.info(f"Save output report to {output_path}")

    report_json.to_file(ctx.report, output_path)


def method_info_get(ctx):
    if ctx.method_path.suffix == ".py":
        return file_py.from_file(ctx.method_path, kind=ctx.method_kind)
    elif ctx.method_path.suffix == ".ipynb":
        return file_ipynb.from_file(ctx.method_path, kind=ctx.method_kind)
    else:
        raise UnidentifiedMethodError(ctx.method_kind)


@_api_guard
@contextmanager
def describe_arguments(ctx):
    parser = ArgumentParser(ctx)

    yield parser

    # Parse arguments
    log.info("Parse arguments")
    parser.parse_args()

    ctx.args = parser.context()

    # Save context information
    ctx.report.context = ctx.args


@_api_guard
def argument(ctx, name: str):
    return ctx.args[name]


@_api_guard
def model_use(ctx, slug: str):
    log.info(f"Search for model: {slug}")

    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name).resolve()

    pkginfo = ctx.storage.model_load(slug, tmpdir_path)

    # Add traceability
    # TODO add hash information
    # -> Create artifact key for model
    model_key = ArtifactKey(
        kind=ArtifactKind.Model,
        id=slug,
    )

    # -> Model use for optimization run
    ctx.report.traceability_graph.add(
        ArtifactLink(
            a=ctx.report.run_key,
            b=model_key,
            kind=ArtifactLinkKind.ModelUse,
        )
    )

    ctx.tmpdirs.append(tmpdir)

    return tmpdir_path / pkginfo.model_file, pkginfo


@_api_guard
@contextmanager
def model_package(
    ctx: FlowCtx, slug: str, exchange_format: str, target: str = "agnostic"
):
    # TODO # Add name argument as the display name is not the same as the slug
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize package metadata
        meta = MLModelMetadata(
            name=slug,
            version="TODO",  # TODO: How to treat this?
            # TODO # How to treat the list of authors?
            authors=[
                MLModelAuthor(name=a.name, email=a.email, contribution=a.contribution)
                for a in ctx.method_info.authors
            ],
            # TODO: How to determine list of maintainers? Maybe store as program information?
            maintainers=[
                MLModelMaintainer(name=a.name, email=a.email)
                for a in ctx.method_info.authors
            ],
            target=target,
            exchange_format=exchange_format,
            model_file=f"model.{exchange_format}",
        )

        # Initialize context information
        # TODO # Use MLModelPackage instead of this class?
        pkginfo = ModelPackageInfo(
            model_metadata=meta,
            model_file_path=Path(tmpdir).resolve() / meta.model_file,
            inference_example_path=Path(tmpdir).resolve() / "inference-example",
        )

        yield pkginfo

        # -> Now the user should have stored files...
        # TODO: Add check that model file is effectively stored here, or else raise some exception

        # Create traceability information
        model_key = ArtifactKey(
            kind=ArtifactKind.Model,
            id=slug,
        )

        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=model_key,
                b=ctx.report.run_key,
                kind=ArtifactLinkKind.ModelGeneratedBy,
            )
        )

        # Now that package is created, we can create the archive
        sha256 = ctx.storage.model_store(
            slug,
            MLModelPackage(
                metadata=meta,
                model_file_path=pkginfo.model_file_path,
                example_implementation_path=(
                    pkginfo.inference_example_path
                    if pkginfo.inference_example_path.exists()
                    else None
                ),
            ),
        )

        # Add artifact metadata
        ctx.report.traceability_graph.metadata_store(
            model_key, "sha256", sha256.hexdigest()
        )


@_api_guard
def dataset_use(ctx, slug: str):
    log.info(f"Search for dataset: {slug}")

    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name).resolve()

    pkginfo = ctx.storage.dataset_load(slug, tmpdir_path)

    # Add traceability
    # TODO add hash information
    # -> Create artifact key for dataset
    dataset_key = ArtifactKey(kind=ArtifactKind.Dataset, id=slug)

    # -> Dataset use for run
    ctx.report.traceability_graph.add(
        ArtifactLink(
            a=ctx.report.run_key,
            b=dataset_key,
            kind=ArtifactLinkKind.DatasetUse,
        )
    )

    ctx.tmpdirs.append(tmpdir)

    return tmpdir_path / "data", pkginfo


@_api_guard
@contextmanager
def dataset_package(ctx, slug: str, name: str, description: str | None = None):

    # TODO # Maybe not useful to create tmpdir for that thing here, as we may fetch data
    # from internet or directly use an existing folder on the PC. To check

    # TODO # Maybe better to just ask for the slug here, then implement some kind of builder pattern
    # in the yielded object to set the metadata information?

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize package metadata
        meta = DatasetMetadata(
            name=name,
            # TODO # How to treat this?
            authors=[
                DatasetAuthor(name=a.name, email=a.email, contribution=a.contribution)
                for a in ctx.method_info.authors
            ],
            # TODO # By default maintainers are the authors of the method
            maintainers=[
                DatasetMaintainer(name=a.name, email=a.email)
                for a in ctx.method_info.authors
            ],
            description=description,
        )

        # Initialize context information
        pkginfo = DatasetPackageInfo(
            metadata=meta,
            dataset_data_path=Path(tmpdir).resolve(),
        )

        yield pkginfo

        # -> Now the user should have stored its file in the temp folder

        # Create traceability information
        dataset_key = ArtifactKey(
            kind=ArtifactKind.Dataset,
            id=slug,
        )

        ctx.report.traceability_graph.add(
            ArtifactLink(
                a=dataset_key,
                b=ctx.report.run_key,
                kind=ArtifactLinkKind.DatasetGeneratedBy,
            )
        )

        # Now that the package is created, we can create the archive
        sha256 = ctx.storage.dataset_store(slug, pkginfo)

        # Add artifact metadata
        ctx.report.traceability_graph.metadata_store(
            dataset_key, "sha256", sha256.hexdigest()
        )


# TODO Add checks for duplicates and value constraints (when constructing?)
class MetricSaver:

    SECTION_LENGTH = 64
    VAR_NAME_LENGTH = 16

    def __init__(self, name: str, output_dict: dict[str, Metric]):
        self.name = name
        self.output_dict = output_dict

    def _sect(self, x: str):
        pad_str = " " + x + " "

        pad_left = self.SECTION_LENGTH // 2 - len(pad_str) // 2
        pad_right = self.SECTION_LENGTH - pad_left - len(pad_str)

        return ("-" * pad_left) + pad_str + ("-" * pad_right)

    def _sect_end(self):
        return "-" * self.SECTION_LENGTH

    def _var(self, name: str, value: str):
        # Generate a format string using f-string and call the format function
        return (f"{{:{self.VAR_NAME_LENGTH}s}} = {{}}").format(name, value)

    def __enter__(self):
        log.info(self._sect(self.name))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        log.info(self._sect_end())

    ###

    def simple(self, name: str, value: float):
        log.info(self._var(name, f"{value:.3f}"))
        self.output_dict[name] = SimpleValue(value=value)

    def scale_lin(self, name: str, value: float, low: float, high: float):
        log.info(self._var(name, f"{value:.3f} ({low:.3f}/{high:.3f})"))
        self.output_dict[name] = LinScaleValue(low=low, high=high, value=value)

    def range_lin(self, name: str, value: int, low: int, high: int):
        log.info(self._var(name, f"{value:.3f} ({low:.3f}/{high:.3f})"))
        self.output_dict[name] = LinRangeValue(low=low, high=high, value=value)

    def percentage(self, name: str, value: float):
        log.info(self._var(name, f"{value:.2f} %"))
        self.output_dict[name] = PercentageValue(value=value)


@_api_guard
def metrics_save(ctx: FlowCtx):
    return MetricSaver("Output metrics", ctx.report.metrics)


@_api_guard
def epoch_metrics(ctx: FlowCtx, epoch: int):
    # Validate epoch value
    if epoch < 0:
        raise InvalidEpochValue(epoch)

    if epoch not in ctx.report.epoch_metrics:
        ctx.report.epoch_metrics[epoch] = {}

    return MetricSaver(f"Epoch {epoch} metrics", ctx.report.epoch_metrics[epoch])


@_api_guard
def in_preview_stage(ctx: FlowCtx):
    return ctx.stage == MethodExecutionStage.Preview


####################################################
# Bench API
####################################################

# TODO: Design: Just allow to get the instance, or create a specific
# API call for each bench method? How to manage specific API calls
# Given by additional capabilities?


@_api_guard
@_dmq_guard
def bench_instance(ctx):
    return ctx.bench_instance
