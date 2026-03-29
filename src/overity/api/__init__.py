"""
Overity.ai API for method writing
=================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import contextmanager

from overity.backend import flow
from overity.errors import UnknownMethodError

from overity.backend.flow.ctx import FlowCtx, RunMode
from overity.model.report import MethodReportKind, MethodReport

from matplotlib.figure import Figure as MplFigure
from plotly.graph_objects import Figure as PlotlyFigure
import pandas as pd


# Initialize global flow object
_CTX = FlowCtx.default()


def _get_method_path():
    # Strategy 1: if method is python file
    try:
        import __main__

        return Path(__main__.__file__).resolve(), RunMode.Standalone
    except AttributeError:  # No __file__ -> Not from python file!
        pass

    # Strategy 2: called from VScode
    try:
        import __main__

        return Path(__main__.__vsc_ipynb_file__).resolve(), RunMode.Interactive
    except AttributeError:  # No __vsc_ipynb_file__ -> Not from VSCode!
        pass

    # No strategy worked, can't identify method
    raise UnknownMethodError()


def init():
    # Initialize logging facilities
    # TODO: Improve logging format, and add environment variable for debug
    logging.basicConfig(level=logging.INFO)

    caller_fpath, run_mode = _get_method_path()

    # Call flow init.
    flow.init(_CTX, caller_fpath, run_mode)


@contextmanager
def describe_arguments():
    with flow.describe_arguments(_CTX) as vargs:
        yield vargs


def argument(name: str):
    return flow.argument(_CTX, name)


def model_use(slug: str):
    return flow.model_use(_CTX, slug)


@contextmanager
def model_package(slug: str, exchange_format: str, target: str = "agnostic"):
    with flow.model_package(_CTX, slug, exchange_format, target) as vpkg:
        yield vpkg


def dataset_use(slug: str):
    return flow.dataset_use(_CTX, slug)


@contextmanager
def dataset_package(slug: str, name: str, description: str | None = None):
    with flow.dataset_package(_CTX, slug, name, description) as vpkg:
        yield vpkg


def metrics_save():
    return flow.metrics_save(_CTX)


def epoch_metrics(epoch: int, display_output: bool = True):
    return flow.epoch_metrics(_CTX, epoch, display_output=display_output)


def in_preview_stage():
    return flow.in_preview_stage(_CTX)


def epoch_metric_df(key: str):
    return flow.epoch_metric_df(_CTX, key)


def graph_save_mpl(identifier: str, fig: MplFigure):
    return flow.graph_save_mpl(_CTX, identifier, fig)


def graph_save_plotly(identifier: str, fig: PlotlyFigure):
    return flow.graph_save_plotly(_CTX, identifier, fig)


####################################################
# Tables API
####################################################


def table_save_df(identifier: str, df: pd.DataFrame, caption: str = ""):
    """Save a pandas DataFrame as a table in the report.

    Args:
        identifier: Unique identifier for the table
        df: pandas DataFrame to save
        caption: Optional description/caption for the table
    """
    return flow.table_save_df(_CTX, identifier, df, caption)


def table_save_dict(identifier: str, data: list[dict], caption: str = ""):
    """Save a list of dictionaries as a table in the report.

    Args:
        identifier: Unique identifier for the table
        data: List of dictionaries where each dict represents a row
        caption: Optional description/caption for the table
    """
    return flow.table_save_dict(_CTX, identifier, data, caption)


####################################################
# Bench API
####################################################


def bench_instance():
    return flow.bench_instance(_CTX)


####################################################
# Report Retrieval API
####################################################


def report_get_experiment(uuid: str) -> MethodReport:
    """Get an experiment report by UUID.

    Args:
        uuid: The UUID of the experiment report to retrieve

    Returns:
        The experiment report with the specified UUID

    Raises:
        ReportNotFoundError: If no experiment report with the given UUID exists
        UninitAPIError: If the API has not been initialized
    """
    return flow.report_get(_CTX, MethodReportKind.Experiment, uuid)


def report_get_training_optimization(uuid: str) -> MethodReport:
    """Get a training/optimization report by UUID.

    Args:
        uuid: The UUID of the training/optimization report to retrieve

    Returns:
        The training/optimization report with the specified UUID

    Raises:
        ReportNotFoundError: If no training/optimization report with the given UUID exists
        UninitAPIError: If the API has not been initialized
    """
    return flow.report_get(_CTX, MethodReportKind.TrainingOptimization, uuid)


def report_get_execution(uuid: str) -> MethodReport:
    """Get an execution (measurement/qualification) report by UUID.

    Args:
        uuid: The UUID of the execution report to retrieve

    Returns:
        The execution report with the specified UUID

    Raises:
        ReportNotFoundError: If no execution report with the given UUID exists
        UninitAPIError: If the API has not been initialized
    """
    return flow.report_get(_CTX, MethodReportKind.Execution, uuid)


def report_get_analysis(uuid: str) -> MethodReport:
    """Get an analysis report by UUID.

    Args:
        uuid: The UUID of the analysis report to retrieve

    Returns:
        The analysis report with the specified UUID

    Raises:
        ReportNotFoundError: If no analysis report with the given UUID exists
        UninitAPIError: If the API has not been initialized
    """
    return flow.report_get(_CTX, MethodReportKind.Analysis, uuid)
