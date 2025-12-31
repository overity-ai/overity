"""
Storage backend base class
==========================

**February 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import re
import uuid
from abc import ABC, abstractmethod
from typing import Callable

from overity.model.general_info.method import MethodKind
from overity.model.ml_model.metadata import MLModelMetadata
from overity.model.ml_model.package import MLModelPackage
from overity.model.report import MethodReportKind

from overity.model.inference_agent.metadata import InferenceAgentMetadata
from overity.model.inference_agent.package import InferenceAgentPackageInfo

from overity.model.dataset.metadata import DatasetMetadata
from overity.model.dataset.package import DatasetPackageInfo

from pathlib import Path


class StorageBackend(ABC):

    # -------------------------- Catalyst

    @abstractmethod
    def program_info(self):
        """Get program information"""

    @abstractmethod
    def execution_targets(self):
        """Get list of execution targets registered in program"""

    @abstractmethod
    def capabilities(self):
        """Get list of specific capabilities defined in program"""

    @abstractmethod
    def benches(self):
        """Get list of bench definitions"""

    # -------------------------- Ingredients

    @abstractmethod
    def training_optimization_methods(self):
        """Get list of optimization methods registered in program"""

    @abstractmethod
    def measurement_qualification_methods(self):
        """Get list of measurement and qualification methods registered in program"""

    @abstractmethod
    def bench_abstractions(self):
        """Get list of bench abstractions registered in program"""

    @abstractmethod
    def analysis_methods(self):
        """Get list of analysis methods registered in program"""

    @abstractmethod
    def experiments(self):
        """Get list of experiments definitions registered in program"""

    @abstractmethod
    def lib(self):
        """Get path to directory containing additional python modules"""

    # -------------------------- Shelf

    @abstractmethod
    def experiment_runs(self, include_all: bool = False):
        """Get list of experiment runs reports in program"""

    @abstractmethod
    def optimization_reports(self, include_all: bool = False):
        """Get list of optimization reports in program"""

    @abstractmethod
    def execution_reports(self, include_all: bool = False):
        """Get list of execution reports in program"""

    @abstractmethod
    def analysis_reports(self, include_all: bool = False):
        """Get list of analysis reports in program"""

    def reports_list(self, kind: MethodReportKind, include_all: bool = False):
        if kind == MethodReportKind.Experiment:
            return self.experiment_runs(include_all=include_all)
        elif kind == MethodReportKind.TrainingOptimization:
            return self.optimization_reports(include_all=include_all)
        elif kind == MethodReportKind.Execution:
            return self.execution_reports(include_all=include_all)
        elif kind == MethodReportKind.Analysis:
            return self.analysis_reports(include_all=include_all)

    @abstractmethod
    def experiment_report_load(self, identifier: str):
        """Load an experiment run report"""

    @abstractmethod
    def optimization_report_load(self, identifier: str):
        """Load an optimization report"""

    @abstractmethod
    def execution_report_load(self, identifier: str):
        """Load an execution report"""

    @abstractmethod
    def analysis_report_load(self, identifier: str):
        """Load an analysis report"""

    def report_load(self, report_kind: MethodReportKind, identifier: str):
        """Load a report for a givne kind value"""

        if report_kind == MethodReportKind.Experiment:
            return self.experiment_report_load(identifier)
        elif report_kind == MethodReportKind.TrainingOptimization:
            return self.optimization_report_load(identifier)
        elif report_kind == MethodReportKind.Execution:
            return self.execution_report_load(identifier)
        elif report_kind == MethodReportKind.Analysis:
            return self.analysis_report_load(identifier)

    @abstractmethod
    def experiment_report_remove(self, identifier: str):
        """Remove an experiment run report"""

    @abstractmethod
    def optimization_report_remove(self, identifier: str):
        """Remove an optimization report"""

    @abstractmethod
    def execution_report_remove(self, identifier: str):
        """Remove an execution report"""

    @abstractmethod
    def analysis_report_remove(self, identifier: str):
        """Remove an analysis report"""

    def report_remove(self, report_kind: MethodReportKind, identifier: str):
        if report_kind == MethodReportKind.Experiment:
            self.experiment_report_remove(identifier)
        elif report_kind == MethodReportKind.TrainingOptimization:
            self.optimization_report_remove(identifier)
        elif report_kind == MethodReportKind.Execution:
            self.execution_report_remove(identifier)
        elif report_kind == MethodReportKind.Analysis:
            self.analysis_report_remove(identifier)

    # -------------------------- Precipitates

    @abstractmethod
    def models(self):
        """Get list of available models in program"""

    @abstractmethod
    def inference_agents(self):
        """Get list of available inference agents in program"""

    @abstractmethod
    def datasets(self):
        """Get list of available datasets in program"""

    @abstractmethod
    def model_info_get(self, slug: str) -> MLModelMetadata:
        """Get info for a given model slug"""

    @abstractmethod
    def model_load(self, slug: str, target_folder: Path) -> MLModelMetadata:
        """Load a model package"""

    @abstractmethod
    def model_store(self, slug: str, package: MLModelPackage):
        """Store a model. Shall return the hash of stored model."""

    @abstractmethod
    def inference_agent_info_get(self, slug: str) -> InferenceAgentMetadata:
        """Get info for a given inference agent slug"""

    @abstractmethod
    def inference_agent_load(
        self, slug: str, target_folder: Path
    ) -> InferenceAgentPackageInfo:
        """Load an inference agent package into the specified target folder"""

    @abstractmethod
    def inference_agent_store(self, slug: str, package: InferenceAgentPackageInfo):
        """Store an inference agent package. Shall return the hash of stored agent"""

    @abstractmethod
    def dataset_info_get(self, slug: str) -> DatasetMetadata:
        """Get info for a given dataset slug"""

    @abstractmethod
    def dataset_load(self, slug: str, target_folder: Path):
        """Load a dataset package into the specified target folder"""

    @abstractmethod
    def dataset_store(self, slug: str, package: DatasetPackageInfo):
        """Store a dataset package. Shall return the hash of stored dataset archive"""

    # -------------------------- Generate IDs

    def _check_is_uuid4(self, string):
        # is_uuid4 function using regular expression
        pattern = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
        return bool(re.match(pattern, string))

    def _default_uuid_get(self, exists_fkt: Callable[[str], bool]):
        """Default UUID generation for a report. Generate the ID and check if it is available"""
        id = None
        while True:
            id = str(uuid.uuid4())
            if not exists_fkt(id):
                break

        return id

    def experiment_run_uuid_get(self):
        """Get an available run uuid"""
        return self._default_uuid_get(self.experiment_run_uuid_exists)

    def optimization_report_uuid_get(self):
        """Get an available optimization report uuid"""
        return self._default_uuid_get(self.optimization_report_uuid_exists)

    def execution_report_uuid_get(self):
        """Get an available execution report uuid"""
        return self._default_uuid_get(self.execution_report_uuid_exists)

    def analysis_report_uuid_get(self):
        """Get an available analysis report uuid"""
        return self._default_uuid_get(self.analysis_report_uuid_exists)

    def method_report_uuid_get(self, kind: MethodKind):
        """Get an available report for a given method kind"""

        if kind == MethodKind.TrainingOptimization:
            return self.optimization_report_uuid_get()
        elif kind == MethodKind.MeasurementQualification:
            return self.execution_report_uuid_get()
        elif kind == MethodKind.Deployment:
            return self.execution_report_uuid_get()
        elif kind == MethodKind.Analysis:
            return self.analysis_report_uuid_get()

    # -------------------------- Check for IDs

    @abstractmethod
    def experiment_run_uuid_exists(self, run_uuid: str):
        """Check if the given uuid exists in experiment run reports"""

    @abstractmethod
    def optimization_report_uuid_exists(self, report_uuid: str):
        """Check if the given optimization report exists with given uuid"""

    @abstractmethod
    def execution_report_uuid_exists(self, report_uuid: str):
        """Check if the given execution report exists with given uuid"""

    @abstractmethod
    def analysis_report_uuid_exists(self, report_uuid: str):
        """Check if the given analysis report exists with given uuid"""
