"""
Local storage implementation
============================

**February 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.

TODO: fix inconsistencies with slug extraction
"""

import logging
import itertools
import traceback
from pathlib import Path

from overity.model.general_info.method import MethodKind, MethodInfo
from overity.model.general_info.bench import (
    BenchAbstractionMetadata,
    BenchInstanciationMetadata,
)
from overity.model.ml_model.metadata import MLModelMetadata
from overity.model.ml_model.package import MLModelPackage
from overity.model.ml_model.attachment import ExtractedAttachment
from overity.model.report import MethodReportKind, MethodExecutionStatus

from overity.model.inference_agent.metadata import InferenceAgentMetadata
from overity.model.inference_agent.package import InferenceAgentPackageInfo

from overity.model.dataset.metadata import DatasetMetadata
from overity.model.dataset.package import DatasetPackageInfo

from overity.model.versioning import VersioningStatus

from overity.utils import git as git_utils
from overity.utils import path as path_utils


from overity.exchange import (
    execution_target_toml,
    program_toml,
    report_json,
    capability_toml,
    bench_toml,
)

from overity.exchange.bench_abstraction import file_py as bench_abstraction_py

from overity.storage.base import StorageBackend, HashObject
from overity.exchange.method_common import file_ipynb, file_py

from overity.errors import (
    DuplicateSlugError,
    UnidentifiedMethodError,
    ModelNotFound,
    AgentNotFound,
    DatasetNotFound,
    ReportNotFound,
    MethodNotFound,
    NoVersionAvailable,
)

from overity.exchange.model_package_v1 import package as ml_package
from overity.exchange.inference_agent_package import package as agent_package
from overity.exchange.dataset_package import package as dataset_package
from overity.exchange import integrity


log = logging.getLogger("Local storage")


class LocalStorage(StorageBackend):
    def __init__(self, folder: Path):
        self.base_folder = folder.resolve()

        # Initialize sub folders
        self.catalyst_folder = self.base_folder / "catalyst"
        self.ingredients_folder = self.base_folder / "ingredients"
        self.shelf_folder = self.base_folder / "shelf"
        self.precipitates_folder = self.base_folder / "precipitates"

        self.execution_targets_folder = self.catalyst_folder / "execution_targets"
        self.capabilities_folder = self.catalyst_folder / "capabilities"
        self.benches_folder = self.catalyst_folder / "benches"

        self.training_optimization_folder = (
            self.ingredients_folder / "training_optimization"
        )
        self.measurement_qualification_folder = (
            self.ingredients_folder / "measurement_qualification"
        )
        self.bench_abstractions_folder = self.ingredients_folder / "bench_abstraction"
        self.analysis_folder = self.ingredients_folder / "analysis"
        self.experiments_folder = self.ingredients_folder / "experiments"
        self.lib_folder = self.ingredients_folder / "lib"

        self.experiment_runs_folder = self.shelf_folder / "experiment_runs"
        self.optimization_reports_folder = self.shelf_folder / "optimization_reports"
        self.execution_reports_folder = self.shelf_folder / "execution_reports"
        self.analysis_reports_folder = self.shelf_folder / "analysis_reports"

        self.models_folder = self.precipitates_folder / "models"
        self.datasets_folder = self.precipitates_folder / "datasets"
        self.agents_folder = self.precipitates_folder / "inference_agents"

        # Leaf folders are deepest folders that we use
        self.leaf_folders = [
            self.execution_targets_folder,
            self.capabilities_folder,
            self.benches_folder,
            self.training_optimization_folder,
            self.measurement_qualification_folder,
            self.bench_abstractions_folder,
            self.analysis_folder,
            self.experiments_folder,
            self.lib_folder,
            self.experiment_runs_folder,
            self.optimization_reports_folder,
            self.execution_reports_folder,
            self.analysis_reports_folder,
            self.experiment_runs_folder,
            self.optimization_reports_folder,
            self.execution_reports_folder,
            self.analysis_reports_folder,
            self.models_folder,
            self.datasets_folder,
            self.agents_folder,
        ]

        # Various path
        self.program_info_path = self.base_folder / "program.toml"

    def initialize(self):
        """Ensure folder exists and are writeable"""

        log.info(f"Initialize local storage in {self.base_folder!s}")

        for folder in self.leaf_folders:
            log.debug(f"Ensure {folder!s} exists")
            folder.mkdir(parents=True, exist_ok=True)

    # -------------------------- Get file paths

    def _execution_target_path(self, slug: str):
        return self.execution_targets_folder / f"{slug}.toml"

    def _capability_path(self, slug: str):
        return self.capabilities_folder / f"{slug}.toml"

    def _bench_path(self, slug: str):
        return self.benches_folder / f"{slug}.toml"

    def _bench_abstraction_path(self, slug: str):
        return self.bench_abstractions_folder / f"{slug}.py"

    def _experiment_run_report_path(self, run_uuid: str):
        return self.experiment_runs_folder / f"{run_uuid}.json"

    def _optimization_report_path(self, run_uuid: str):
        return self.optimization_reports_folder / f"{run_uuid}.json"

    def _execution_report_path(self, run_uuid: str):
        return self.execution_reports_folder / f"{run_uuid}.json"

    def _analysis_report_path(self, run_uuid: str):
        return self.analysis_reports_folder / f"{run_uuid}.json"

    def _model_path(self, slug: str):
        return self.models_folder / f"{slug}.tar.gz"

    def _dataset_path(self, slug: str):
        return self.datasets_folder / f"{slug}.tar.gz"

    def _agent_path(self, slug: str):
        return self.agents_folder / f"{slug}.tar.gz"

    def method_run_report_path(self, run_uuid: str, method_kind: MethodKind):
        if method_kind == MethodKind.TrainingOptimization:
            return self._optimization_report_path(run_uuid)
        elif method_kind == MethodKind.MeasurementQualification:
            return self._execution_report_path(run_uuid)
        elif method_kind == MethodKind.Analysis:
            return self._analysis_report_path(run_uuid)

    # -------------------------- Catalyst

    def program_info(self):
        """Get program information"""

        if not self.program_info_path.is_file():
            msg = f"{self.program_info_path} is not a valid file or is not readable"
            raise FileNotFoundError(msg)

        return program_toml.from_file(self.program_info_path)

    def execution_targets(self):
        """Get list of execution targets registered in program as a generator
        TODO: Return fonud_targets, found_errors as in other methods
        """

        log.debug(f"Get list of execution targets from {self.execution_targets_folder}")

        # List TOML files in execution target folder

        def process_file(path):
            log.debug(f"Check file {path}")

            try:
                return execution_target_toml.from_file(path)

            except Exception as exc:
                log.debug(f"Error checking {path}: {exc!s}")
                log.debug(traceback.format_exc())

        return map(process_file, self.execution_targets_folder.glob("**/*.toml"))

    def capabilities(self):
        """Get list of defined capabilities in current program as a generator
        TODO: Return found_capabilities, found_errors as in other methods
        """

        log.debug(f"Get list of capabilities from {self.capabilities_folder}")

        def process_file(path):
            log.debug(f"Check file {path}")

            try:
                return capability_toml.from_file(path)
            except Exception as exc:
                log.debug(f"Error checking {path}: {exc!s}")
                log.debug(traceback.format_exc())

                return (path, exc)

        return map(process_file, self.capabilities_folder.glob("**/*.toml"))

    def benches(self):
        """Get a list of defined bench instanciations in current program as a generator"""

        log.debug(f"Get list of benches from {self.benches_folder}")

        def process_file(path):
            log.debug(f"Check file {path}")

            slug = path.name.removesuffix(".toml")

            try:
                return (
                    slug,
                    bench_toml.from_file(path),
                )
            except Exception as exc:
                log.debug(f"Error checking {path}: {exc!s}")
                log.debug(traceback.format_exc())

                return (
                    slug,
                    exc,
                )

        processed = list(map(process_file, self.benches_folder.glob("**/*.toml")))

        # Isolate found benches and errors
        found_benches = list(
            filter(lambda x: isinstance(x[1], BenchInstanciationMetadata), processed)
        )

        found_errors = list(filter(lambda x: isinstance(x[1], Exception), processed))

        return (
            found_benches,
            found_errors,
        )

    def catalyst_version_status(self) -> VersioningStatus:
        """Get the current versioning status of the catalyst folder

        Returns:
            VersioningStatus

        1. If any change is detected within the "catalyst" folder, the version status is dirty.
        2. If no change is detected, then the catalyst folder is clean
        3. If there is no git repository, we cannot track the current version

        """

        # Find the nearest repo from the catalyst folder
        path_to_repo = git_utils.nearest_repo(self.base_folder)

        if path_to_repo is None:
            return VersioningStatus.NotVersioned

        # Check if the catalyst folder is actually under the found repo
        # If not, we cannot track the version
        try:
            catalyst_rel = self.catalyst_folder.relative_to(path_to_repo)
        except ValueError:
            # catalyst folder is not under the repo root
            return VersioningStatus.NotVersioned

        # Get git status for the repository
        git_changes = git_utils.git_status(str(path_to_repo))

        # Check if any changes are within the catalyst folder
        changes = [
            change
            for change in git_changes
            if path_utils.is_subpath(Path(change.from_path), catalyst_rel)
        ]

        if changes:
            return VersioningStatus.Dirty
        else:
            return VersioningStatus.Clean

    def catalyst_version_info(self) -> str:
        """Get the current version information for the catalyst folder

        Returns:
            a string that describes the used version for catalyst (e.g., git commit hash)

        NOTE: if there is no versioning information available, this function raises NoVersionAvailable error
        """

        try:
            cur_commit = git_utils.current_commit(self.base_folder)
            return cur_commit
        except RuntimeError:
            raise NoVersionAvailable(f"catalyst in {str(self.catalyst_folder)!r}")

    # -------------------------- Ingredients

    def training_optimization_methods(self):
        """Get list of optimization methods registered in program"""

        def process_file(x: Path):
            try:
                ext = x.suffix

                if ext == ".py":
                    return file_py.from_file(x, kind=MethodKind.TrainingOptimization)
                elif ext == ".ipynb":
                    return file_ipynb.from_file(x, kind=MethodKind.TrainingOptimization)

            except Exception as exc:
                return (x, exc)

        # Process files
        py_files = self.training_optimization_folder.glob("*.py")
        ipynb_files = self.training_optimization_folder.glob("*.ipynb")
        processed = list(map(process_file, itertools.chain(py_files, ipynb_files)))

        # Isolate found methods and errors
        found_methods = list(filter(lambda x: isinstance(x, MethodInfo), processed))
        found_errors = list(filter(lambda x: isinstance(x, tuple), processed))

        # Look for duplicates in found methods
        mtd_dict = {}
        for mtd in found_methods:
            mtd_dict[mtd.slug] = (mtd_dict.get(mtd.slug) or []) + [mtd]

        duplicates = {k: v for k, v in mtd_dict.items() if len(v) > 1}

        for slug, mtds in duplicates.items():
            for mtd in mtds:
                found_errors.append(
                    (
                        mtd.path,
                        DuplicateSlugError(mtd.path, slug),
                    )
                )

        return found_methods, found_errors

    def measurement_qualification_methods(self):
        """Get list of measurement and qualification methods registered in program"""

        def process_file(x: Path):
            try:
                ext = x.suffix

                if ext == ".py":
                    return file_py.from_file(
                        x, kind=MethodKind.MeasurementQualification
                    )
                elif ext == ".ipynb":
                    return file_ipynb.from_file(
                        x, kind=MethodKind.MeasurementQualification
                    )

            except Exception as exc:
                return (x, exc)

        # Process files
        py_files = self.measurement_qualification_folder.glob("*.py")
        ipynb_files = self.measurement_qualification_folder.glob("*.ipynb")
        processed = list(map(process_file, itertools.chain(py_files, ipynb_files)))

        # Isolate found methods and errors
        found_methods = list(filter(lambda x: isinstance(x, MethodInfo), processed))
        found_errors = list(filter(lambda x: isinstance(x, tuple), processed))

        # Look for duplicates in found methods
        mtd_dict = {}
        for mtd in found_methods:
            mtd_dict[mtd.slug] = (mtd_dict.get(mtd.slug) or []) + [mtd]

        duplicates = {k: v for k, v in mtd_dict.items() if len(v) > 1}

        for slug, mtds in duplicates.items():
            for mtd in mtds:
                found_errors.append(
                    (
                        mtd.path,
                        DuplicateSlugError(mtd.path, slug),
                    )
                )

        return found_methods, found_errors

    def bench_abstractions(self):
        """Get list of bench abstractions registered in program"""

        def process_file(x: Path):
            try:
                infos = bench_abstraction_py.from_file(x)
                return (
                    infos.slug,
                    infos,
                )
            except Exception as exc:
                slug = bench_abstraction_py._extract_slug(x)
                return (
                    slug,
                    exc,
                )

        processed = list(map(process_file, self.bench_abstractions_folder.glob("*.py")))

        found_abstractions = list(
            filter(lambda x: isinstance(x[1], BenchAbstractionMetadata), processed)
        )
        found_errors = list(filter(lambda x: isinstance(x[1], Exception), processed))

        return found_abstractions, found_errors

    def analysis_methods(self):
        """Get list of analysis methods registered in program"""

        def process_file(x: Path):
            try:
                ext = x.suffix

                if ext == ".py":
                    return file_py.from_file(x, kind=MethodKind.Analysis)
                elif ext == ".ipynb":
                    return file_ipynb.from_file(x, kind=MethodKind.Analysis)

            except Exception as exc:
                return (x, exc)

        # Process files
        py_files = self.analysis_folder.glob("*.py")
        ipynb_files = self.analysis_folder.glob("*.ipynb")
        processed = list(map(process_file, itertools.chain(py_files, ipynb_files)))

        # Isolate found methods and errors
        found_methods = list(filter(lambda x: isinstance(x, MethodInfo), processed))
        found_errors = list(filter(lambda x: isinstance(x, tuple), processed))

        # Look for duplicates in found methods
        mtd_dict = {}
        for mtd in found_methods:
            mtd_dict[mtd.slug] = (mtd_dict.get(mtd.slug) or []) + [mtd]

        duplicates = {k: v for k, v in mtd_dict.items() if len(v) > 1}
        for slug, mtds in duplicates.items():
            for mtd in mtds:
                found_errors.append(
                    (
                        mtd.path,
                        DuplicateSlugError(mtd.path, slug),
                    )
                )

        return found_methods, found_errors

    def experiments(self):
        """Get list of experiments definitions registered in program"""
        raise NotImplementedError

    def identify_method_kind(self, pp: Path):
        """Identify the method kind from a file, given its absolute path"""

        if pp.is_relative_to(self.training_optimization_folder):
            return MethodKind.TrainingOptimization
        elif pp.is_relative_to(self.measurement_qualification_folder):
            return MethodKind.MeasurementQualification
        elif pp.is_relative_to(self.analysis_folder):
            return MethodKind.Analysis
        else:
            raise UnidentifiedMethodError(pp)

    def identify_method_slug(self, pp: Path):
        return pp.stem

    def get_method_path(self, kind: MethodKind, slug: str):
        """Get the path to the specified method kind"""

        dirs = {
            MethodKind.TrainingOptimization: self.training_optimization_folder,
            MethodKind.MeasurementQualification: self.measurement_qualification_folder,
            MethodKind.Analysis: self.analysis_folder,
        }

        mtd_dir = dirs[kind]

        # Try to locate the method file, trying both .py and .ipynb extensions
        py_file = mtd_dir / f"{slug}.py"
        ipynb_file = mtd_dir / f"{slug}.ipynb"

        # Duplicate found
        if py_file.is_file() and ipynb_file.is_file():
            raise DuplicateSlugError(ipynb_file, slug)

        # No file found
        if (not py_file.is_file()) and (not ipynb_file.is_file()):
            raise MethodNotFound(kind, slug)

        # Get the found file
        if py_file.is_file():
            return py_file
        else:
            return ipynb_file

    def bench_load_infos(self, slug: str):
        """Import a bench instanciation's information"""

        bench_path = self._bench_path(slug)
        metadata = bench_toml.from_file(bench_path)

        return metadata

    def bench_abstraction_import_infos(self, slug: str):
        """Import a bench abstraction's metadata information given its slug"""

        abstraction_path = self._bench_abstraction_path(slug)
        metadata = bench_abstraction_py.from_file(abstraction_path)

        return metadata

    def bench_abstraction_import_definitions(self, slug: str):
        """Import a bench abstraction's definitions given its slug"""
        # TODO # Is this kind of method future-proof against other kinds of storages?

        abstraction_path = self._bench_abstraction_path(slug)
        BenchSettings, BenchDefinition = bench_abstraction_py.import_definitions(
            abstraction_path
        )

        return BenchSettings, BenchDefinition

    def lib(self):
        """Return the path containing additional python modules for methods implementation"""
        return self.lib_folder

    def ingredients_version_status(self) -> VersioningStatus:
        """Get the current versioning status of the ingredients folder

        Returns:
            VersioningStatus

        In local storage backend implementation, consistency is given by the git status of the "ingredients" folder.
        So the idea is to check the git status of the ingredients folder.

        1. If any change is detected within the "ingredients" folder, the version status is dirty.
        2. If no change is detected, then the ingredients folder is clean
        3. If there is no git repository, we cannot track the current version

        TODO: How to manage when the ingredients folder is in a different git repository than the laboratory folder?
        """

        # Find the nearest repo from the ingredients folder
        path_to_repo = git_utils.nearest_repo(self.base_folder)

        if path_to_repo is None:
            return VersioningStatus.NotVersioned

        # Check if the ingredients folder is actually under the found repo
        # If not, we cannot track the version
        try:
            ingredients_rel = self.ingredients_folder.relative_to(path_to_repo)
        except ValueError:
            # ingredients folder is not under the repo root
            return VersioningStatus.NotVersioned

        # Get git status for the repository
        git_changes = git_utils.git_status(str(path_to_repo))

        # Check if any changes are within the ingredients folder
        changes = [
            change
            for change in git_changes
            if path_utils.is_subpath(Path(change.from_path), ingredients_rel)
        ]

        if changes:
            return VersioningStatus.Dirty
        else:
            return VersioningStatus.Clean

    def ingredients_version_info(self) -> str:
        """Get the current version information for the ingredients folder

        Returns:
            a string that describes the used version for ingredients (e.g., git commit hash)

        NOTE: if there is no versioning information available, this function raises NoVersionAvailable error
        """

        try:
            cur_commit = git_utils.current_commit(self.base_folder)
            return cur_commit
        except RuntimeError:
            raise NoVersionAvailable(f"ingredients in {str(self.ingredients_folder)!r}")

    # -------------------------- Shelf

    def experiment_runs(self):
        """Get list of experiment runs reports in program"""
        raise NotImplementedError

    def optimization_reports(self):
        """Get list of identifiers for optimization reports in program"""

        return tuple(
            map(lambda x: x.stem, self.optimization_reports_folder.glob("*.json"))
        )

    def execution_reports(self):
        """Get list of execution reports in program"""

        return tuple(
            map(lambda x: x.stem, self.execution_reports_folder.glob("*.json")),
        )

    def analysis_reports(self, include_all: bool = False):
        """Get list of analysis reports in program

        Args:
            include_all: If True, return all reports regardless of status.
                        If False (default), only return reports with ExecutionSuccess status.

        Returns:
            Tuple of report identifiers, filtered by status if include_all=False
        """
        import logging

        log = logging.getLogger("LocalStorage.analysis_reports")

        report_files = list(self.analysis_reports_folder.glob("*.json"))
        valid_reports = []

        for report_file in report_files:
            try:
                # Load the report to check its status
                report = report_json.from_file(report_file)

                # If include_all is False, only include successful reports
                if not include_all:
                    if report.status != MethodExecutionStatus.ExecutionSuccess:
                        continue

                valid_reports.append(report_file.stem)

            except Exception as exc:
                # Log the error but continue processing other files
                log.info(f"Error loading report {report_file}: {exc!s}")
                continue

        return tuple(valid_reports)

    def experiment_report_load(self, identifier: str):
        raise NotImplementedError

    def optimization_report_load(self, identifier: str):
        report_path = self._optimization_report_path(identifier)

        if not report_path.is_file():
            raise ReportNotFound(
                self.base_folder, MethodReportKind.TrainingOptimization, identifier
            )

        return report_path, report_json.from_file(report_path)

    def execution_report_load(self, identifier: str):
        report_path = self._execution_report_path(identifier)

        if not report_path.is_file():
            raise ReportNotFound(
                self.base_folder, MethodReportKind.Execution, identifier
            )

        return report_path, report_json.from_file(report_path)

    def analysis_report_load(self, identifier: str):
        report_path = self._analysis_report_path(identifier)

        if not report_path.is_file():
            raise ReportNotFound(
                self.base_folder, MethodReportKind.Analysis, identifier
            )

        return report_path, report_json.from_file(report_path)

    def experiment_report_remove(self, identifier: str):
        raise NotImplementedError

    def optimization_report_remove(self, identifier: str):
        pp = self._optimization_report_path(identifier)
        pp.unlink(missing_ok=True)

    def execution_report_remove(self, identifier: str):
        pp = self._execution_report_path(identifier)
        pp.unlink(missing_ok=True)

    def analysis_report_remove(self, identifier: str):
        raise NotImplementedError

    # -------------------------- Precipitates

    def models(self):
        """Get list of available models in program

        Returns:
            A tuple of (found_models, found_errors) where found_models is a list of
            (slug, metadata) tuples and found_errors is a list of (slug, exception) tuples
        """

        def process_file(x: Path):
            slug = x.name.removesuffix(".tar.gz")

            try:
                metadata = ml_package.metadata_load(x)
                return (slug, metadata)
            except Exception as exc:
                return (slug, exc)

        processed = list(map(process_file, self.models_folder.glob("*.tar.gz")))

        # Isolate found models and errors
        found_models = list(
            filter(lambda x: isinstance(x[1], MLModelMetadata), processed)
        )
        found_errors = list(filter(lambda x: isinstance(x[1], Exception), processed))

        return found_models, found_errors

    def datasets(self):
        """Get list of available datasets in program

        Returns:
            A tuple of (found_datasets, found_errors) where found_datasets is a list of
            (slug, metadata) tuples and found_errors is a list of (slug, exception) tuples
        """

        def process_file(x: Path):
            slug = x.name.removesuffix(".tar.gz")

            try:
                metadata = dataset_package.metadata_load(x)
                return (slug, metadata)
            except Exception as exc:
                return (slug, exc)

        processed = list(map(process_file, self.datasets_folder.glob("*.tar.gz")))

        # Isolate found datasets and errors
        found_datasets = list(
            filter(lambda x: isinstance(x[1], DatasetMetadata), processed)
        )
        found_errors = list(filter(lambda x: isinstance(x[1], Exception), processed))

        return found_datasets, found_errors

    def inference_agents(self):
        """Get a list of available inference agents in program

        Returns:
            A tuple of (found_agents, found_errors) where found_agents is a list of
            (slug, metadata) tuples and found_errors is a list of (slug, exception) tuples
        """

        def process_file(x: Path):
            slug = x.name.removesuffix(".tar.gz")

            try:
                metadata = agent_package.metadata_load(x)
                return (slug, metadata)
            except Exception as exc:
                return (slug, exc)

        processed = list(map(process_file, self.agents_folder.glob("*.tar.gz")))

        # Isolate found agents and errors
        found_agents = list(
            filter(
                lambda x: isinstance(x[1], InferenceAgentMetadata),
                processed,
            )
        )
        found_errors = list(filter(lambda x: isinstance(x[1], Exception), processed))

        return (
            found_agents,
            found_errors,
        )

    def model_info_get(self, slug: str) -> MLModelMetadata:
        # Try to find model package file
        fpath = self._model_path(slug)

        if not fpath.is_file():
            raise ModelNotFound(slug)

        # Load json info from package
        pkginfo = ml_package.metadata_load(fpath)

        return pkginfo

    def model_load(
        self, slug: str, target_folder: Path
    ) -> tuple[MLModelMetadata, HashObject, dict[str, ExtractedAttachment]]:
        fpath = self._model_path(slug)

        if not fpath.is_file():
            raise ModelNotFound(slug)

        pkginfo, attachments = ml_package.model_load(fpath, target_folder)

        # Compute SHA256 of the archive file
        sha256 = integrity.file_sha256(fpath)

        return pkginfo, sha256, attachments

    def model_store(self, slug: str, pkg: MLModelPackage):
        fpath = self._model_path(slug)  # Get path for target archive
        sha256 = ml_package.package_archive_create(pkg, fpath)

        log.info(f"Stored model {slug} to {fpath}")

        return sha256

    def inference_agent_info_get(self, slug: str):
        # Try to fin agent package file
        fpath = self._agent_path(slug)

        if not fpath.is_file():
            raise AgentNotFound(slug)

        # Load json info from package
        pkginfo = agent_package.metadata_load(fpath)

        return pkginfo

    def inference_agent_load(
        self, slug: str, target_folder: Path
    ) -> tuple[InferenceAgentPackageInfo, HashObject]:
        fpath = self._agent_path(slug)

        if not fpath.is_file():
            raise AgentNotFound(slug)

        pkginfo = agent_package.agent_load(fpath, target_folder)

        # Compute SHA256 of the archive file
        sha256 = integrity.file_sha256(fpath)

        return pkginfo, sha256

    def inference_agent_store(self, slug: str, package: InferenceAgentPackageInfo):
        fpath = self._agent_path(slug)
        sha256 = agent_package.package_archive_create(package, fpath)

        log.info(f"Stored model {slug} to {fpath}")

        return sha256

    def dataset_info_get(self, slug: str):
        # Try to find dataset metadata file
        fpath = self._dataset_path(slug)

        if not fpath.is_file():
            return DatasetNotFound(slug)

        # Load json info from package
        pkginfo = dataset_package.metadata_load(fpath)

        return pkginfo

    def dataset_load(
        self, slug: str, target_folder: Path
    ) -> tuple[DatasetPackageInfo, HashObject]:
        fpath = self._dataset_path(slug)

        if not fpath.is_file():
            raise DatasetNotFound(slug)

        pkginfo = dataset_package.dataset_load(fpath, target_folder)

        # Compute SHA256 of the archive file
        sha256 = integrity.file_sha256(fpath)

        return pkginfo, sha256

    def dataset_store(self, slug: str, package: DatasetPackageInfo):
        fpath = self._dataset_path(slug)
        sha256 = dataset_package.package_archive_create(package, fpath)

        log.info(f"Store dataset {slug} to {fpath}")

        return sha256

    # -------------------------- Check for IDs

    def experiment_run_uuid_exists(self, run_uuid: str):
        """Check if the given uuid exists in experiment run reports"""

        return self._experiment_run_report_path(run_uuid).is_file()

    def optimization_report_uuid_exists(self, report_uuid: str):
        """Check if the given optimization report exists with given uuid"""

        return self._optimization_report_path(report_uuid).is_file()

    def execution_report_uuid_exists(self, report_uuid: str):
        """Check if the given execution report exists with given uuid"""

        return self._execution_report_path(report_uuid).is_file()

    def analysis_report_uuid_exists(self, report_uuid: str):
        """Check if the given analysis report exists with given uuid"""

        return self._analysis_report_path(report_uuid).is_file()
