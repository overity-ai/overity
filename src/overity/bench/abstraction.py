"""
Bench abstraction base class
============================

**June 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.

TODO: Parameters for base methods

"""

from __future__ import annotations

import tempfile
import logging

from abc import ABC, abstractmethod

from overity.model.ml_model.metadata import MLModelMetadata
from overity.model.traceability import (
    ArtifactKind,
    ArtifactLinkKind,
    ArtifactKey,
    ArtifactLink,
    ArtifactGraph,
)
from overity.model.general_info.bench import (
    BenchAbstractionMetadata,
    BenchInstanciationMetadata,
)
from overity.storage.base import StorageBackend

from pathlib import Path


class BenchAbstraction(ABC):
    """
    Base class to define a bench abstraction
    """

    def __init__(
        self,
        settings: any,
        storage_backend: StorageBackend,
        abstraction_infos: BenchAbstractionMetadata,
        instance_infos: BenchInstanciationMetadata,
    ):

        self.traceability_graph = (
            ArtifactGraph()
        )  # Bench abstraction has its own traceability graph
        self.storage = (
            storage_backend  # Bench abstraction can call the available storage backend
        )
        self.log = logging.getLogger("Bench")  # TODO: name for bench

        self.abstraction_infos = abstraction_infos
        self.instance_infos = instance_infos

        self.key_instance = ArtifactKey(
            kind=ArtifactKind.BenchInstanciation, id=instance_infos.slug
        )
        self.key_abstraction = ArtifactKey(
            kind=ArtifactKind.BenchAbstraction, id=abstraction_infos.slug
        )

        # TODO: Type annotation for bench settings?
        self.__configure__(settings)

        self.tmpdirs = set()

    ####################################################
    # Private assets use interface
    ####################################################

    def _agent_use(self, slug: str):
        self.log.info(f"Search for agent: {slug}")

        tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir.name).resolve()
        pkginfo = self.storage.inference_agent_load(slug, tmpdir_path)

        # Traceability information
        # TODO Add hash information
        # FIXME Duplicate with flow backend code?
        # -> Artifact key for agent
        agent_key = ArtifactKey(
            kind=ArtifactKind.InferenceAgent,
            id=slug,
        )

        # -> Agent use for run
        self.traceability_graph.add(
            ArtifactLink(
                kind=ArtifactLinkKind.InferenceAgentUse,
                a=self.key_abstraction,
                b=agent_key,
            )
        )

        self.tmpdirs.add(tmpdir)

        return tmpdir_path / "data", pkginfo

    ####################################################
    # Public common interface
    ####################################################

    def tmpdir_cleanup(self):
        for tmpdir in self.tmpdirs:
            self.log.debug(f"Remove temporary directory: {tmpdir}")
            tmpdir.cleanup()

    ####################################################
    # User-implemented features
    ####################################################

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the set of available capabilities"""
        return frozenset({})

    @property
    def compatible_tags(self) -> frozenset[str]:
        """Return the list of compatible execution targets tags"""
        return frozenset({})

    @property
    def compatible_targets(self) -> frozenset[str]:
        """Return the list of compatible execution targets slugs"""
        return frozenset({})

    @abstractmethod
    def __configure__(self, settings):
        """This method is implemented in children classes to configure the bench, given input settings"""
        pass

    @abstractmethod
    def bench_start(self):
        """Called to start bench (open connections, etc.)"""
        pass

    @abstractmethod
    def bench_cleanup(self):
        """Called to stop bench (close connections, remove temp files, etc.)"""
        pass

    @abstractmethod
    def sanity_check(self):
        """Called to check that bench is working OK"""

    @abstractmethod
    def state_initial(self):
        """Called to set bench to initial status"""

    @abstractmethod
    def state_panic(self):
        """Called for emergency bench stop"""

    # @abstractmethod
    # def agent_infos(self):
    #    """Get infos of used inference agent"""

    @abstractmethod
    def agent_deploy(self, model_file: Path, model_data: MLModelMetadata):
        """Called to deploy inference agent"""

    @abstractmethod
    def agent_start(self):
        """Called to start deployed inference agent"""

    @abstractmethod
    def agent_hello(self):
        """Called to test communication channel between bench and agent"""

    @abstractmethod
    def agent_inference(self, vectors: dict[str, any]):
        """Called to run an inference on the inference agent"""

    def has_capability(self, capability_name: str) -> bool:
        return capability_name in self.capabilities
