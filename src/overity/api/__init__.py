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


def epoch_metrics(epoch: int):
    return flow.epoch_metrics(_CTX, epoch)


def in_preview_stage():
    return flow.in_preview_stage(_CTX)


def epoch_metric_df(key: str):
    return flow.epoch_metric_df(_CTX, key)


####################################################
# Bench API
####################################################


def bench_instance():
    return flow.bench_instance(_CTX)
