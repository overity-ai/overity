"""
Utility types for frontend
==========================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from argparse import ArgumentError

from overity.model.general_info.method import MethodKind
from overity.model.report import MethodReportKind


def parse_method_kind(x: str):
    """Parse method kind from string argument."""

    # Training optimization
    if x in {"training-optimization", "to"}:
        return MethodKind.TrainingOptimization

    elif x in {"measurement-qualification", "mq"}:
        return MethodKind.MeasurementQualification

    elif x in {"analysis", "an"}:
        return MethodKind.Analysis

    else:
        raise ArgumentError(None, f"Invalid method kind: {x}")


def parse_report_kind(x: str):
    """Parse report kind from string argument."""

    if x in {"training-optimization", "to", "topt"}:
        return MethodReportKind.TrainingOptimization

    elif x in {"execution", "exec", "ex"}:
        return MethodReportKind.Execution

    elif x in {"analysis", "an"}:
        return MethodReportKind.Analysis

    else:
        raise ArgumentError(None, f"Invalid report kind: {x}")
