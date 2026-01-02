"""
Overity.ai model for reports
============================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from enum import Enum
from datetime import datetime as dt
from dataclasses import dataclass
from datetime import datetime

from overity.model.traceability import ArtifactGraph
from overity.model.general_info.method import MethodInfo
from overity.model.report.metrics import (
    Metric,
)


class MethodExecutionStatus(Enum):
    """Method execution status"""

    """Method execution failed with exception"""
    ExecutionFailureException = "execution_failure_exception"

    """Method execution succeeded but did not meet expected conditions"""
    ExecutionFailureConstraints = "execution_failure_constraints"

    """Method execution succeeded and goals are OK"""
    ExecutionSuccess = "execution_success"


class MethodExecutionStage(Enum):
    """Method execution stage

    The method execution stage indicates what is the intended purpose of this run.
    For instance, it may concern the prototyping/preview stage, or the running/operational
    phase.
    """

    """Preview: the user is designing/testing the running method"""
    Preview = "preview"

    """Operation: this is a 'final' run"""
    Operation = "operation"


class MethodReportKind(Enum):
    Experiment = "experiment"
    TrainingOptimization = "training_optimization"
    Execution = "execution"
    Analysis = "analysis"


@dataclass
class MethodReportLogItem:
    timestamp: datetime
    severity: str
    source: str
    message: str


@dataclass
class MethodReport:
    uuid: str
    program: str  # Name of programme
    date_started: datetime
    date_ended: datetime | None
    stage: MethodExecutionStage | None
    status: MethodExecutionStatus | None
    environment: dict[str, str]
    context: dict[str, str]
    traceability_graph: ArtifactGraph
    method_info: MethodInfo
    logs: list[MethodReportLogItem]
    outputs: any | None = None
    metrics: dict[str, Metric] | None = None

    @classmethod
    def default(
        cls,
        uuid: str,
        program: str,
        method_info: MethodInfo,
        date_started: dt | None = None,
    ) -> MethodReport:
        date_started = date_started or dt.now()

        return cls(
            uuid=uuid,
            program=program,
            method_info=method_info,
            date_started=date_started,
            date_ended=date_started,
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            outputs=None,
            metrics={},
        )

    def log_add(self, tstamp: dt, severity: str, source: str, message: str):
        self.logs.append(
            MethodReportLogItem(
                timestamp=tstamp,
                severity=severity,
                source=source,
                message=message,
            )
        )
