"""
Artifact traceability data model
================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce


class ArtifactKind(Enum):
    """Indicates what is the kind of the artefact"""

    # Catalyst
    ExecutionTargetInfo = "execution_target_info"
    BenchInstanciation = "bench_instanciation"

    # Ingredients
    AnalysisMethod = "analysis_method"
    DeploymentMethod = "deployment_method"
    MeasurementQualificationMethod = "measurement_qualification_method"
    TrainingOptimizationMethod = "training_optimization_method"
    BenchAbstraction = "bench_abstraction"

    # Precipitates
    Dataset = "dataset"
    Model = "model"
    InferenceAgent = "inference_agent"

    # Shelf
    AnalysisReport = "analysis_report"
    ExecutionReport = "execution_report"
    ExperimentRun = "experiment_run"
    OptimizationReport = "optimization_report"

    # Runs
    OptimizationRun = "optimization_run"
    ExecutionRun = "execution_run"


class ArtifactLinkKind(Enum):
    """Categorize the link between two artifacts"""

    """Links a run to an used model"""
    ModelUse = "model_use"

    """Links a run to an used inference_agent"""
    InferenceAgentUse = "inference_agent_use"

    """Links a run to an used method"""
    MethodUse = "method_use"

    """Links a run to an used dataset"""
    DatasetUse = "dataset_use"

    """Links a dataset to the dataset it is extracted from"""
    DatasetSubsetFrom = "dataset_subset_from"

    """Links a run to a generated dataset"""
    DatasetGeneratedBy = "dataset_generated_by"

    """Links a model to the run that generated it"""
    ModelGeneratedBy = "model_generated_by"

    """Links a report to its corresponding run"""
    ReportFor = "report_for"

    """Links a bench instanciation to a bench abstraction"""
    InstanciateBench = "instanciate_bench"

    """Links a run to a bench"""
    BenchUse = "bench_use"


@dataclass(frozen=True, eq=True)
class ArtifactKey:
    """Identifies an artifact into the dependency graph"""

    """What is the kind of the artifact"""
    kind: ArtifactKind

    """What is the identifier (slug or uuid) of the targetted artifact?"""
    id: str


@dataclass(frozen=True, eq=True)
class ArtifactLink:
    """Identifies the link between two artifacts

    The link between two artifacts should be child to parent. For instance, if a training method
    uses a specific dataset, a = the training method, b = the used dataset.
    """

    """What is the kind of link between the two artifacts?"""
    kind: ArtifactLinkKind

    """Starting point"""
    a: ArtifactKey

    """Target artefact"""
    b: ArtifactKey


@dataclass
class ArtifactGraph:
    """Links between the nodes"""

    links: set[ArtifactLink] = field(default_factory=set)
    metadata: dict[ArtifactKey, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def default(cls):
        return cls()

    @property
    def nodes(self):
        """Return the set of nodes in the graph"""

        # Build a set containing the two nodes a and b, and
        # combine all the generated sets.
        return reduce(
            frozenset.union,
            map(lambda lk: frozenset({lk.a, lk.b}), self.links),
            frozenset(),
        )

    def add(self, lk: ArtifactLink):
        assert isinstance(lk, ArtifactLink), type(lk)

        self.links.add(lk)

    def metadata_store(self, art: ArtifactKey, key: str, data: str):
        if art not in self.metadata:
            self.metadata[art] = dict()

        self.metadata[art][key] = data

    def __add__(self, other):
        """Merge two traceability graphs together"""

        return ArtifactGraph(
            links=self.links | other.links, metadata=self.metadata | other.metadata
        )
