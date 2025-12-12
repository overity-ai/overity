"""
Unit tests for overity.storage.local module
"""

import pytest
import tempfile
import json
import toml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import uuid

from overity.storage.local import LocalStorage
from overity.model.general_info.method import MethodKind
from overity.model.report import MethodReportKind, MethodExecutionStatus
from overity.errors import (
    DuplicateSlugError,
    UnidentifiedMethodError,
    ModelNotFound,
    AgentNotFound,
    DatasetNotFound,
    ReportNotFound,
)


class TestLocalStorage:
    """Test suite for LocalStorage class"""

    def test_initialization(self):
        """Test LocalStorage initialization creates correct folder structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Check base folder
            assert storage.base_folder == Path(tmpdir).resolve()

            # Check subfolders
            assert storage.catalyst_folder == Path(tmpdir).resolve() / "catalyst"
            assert storage.ingredients_folder == Path(tmpdir).resolve() / "ingredients"
            assert storage.shelf_folder == Path(tmpdir).resolve() / "shelf"
            assert (
                storage.precipitates_folder == Path(tmpdir).resolve() / "precipitates"
            )

            # Check leaf folders
            assert len(storage.leaf_folders) > 0
            for folder in storage.leaf_folders:
                # Check that parent exists or is base folder (after initialization)
                # Note: parent may not exist before initialize() is called
                pass

    def test_initialize_creates_folders(self):
        """Test initialize() method creates all required folders"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Check all leaf folders are created
            for folder in storage.leaf_folders:
                assert folder.exists()
                assert folder.is_dir()

    def test_path_generation_methods(self):
        """Test path generation methods return correct paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Test various path generation methods
            test_slug = "test-slug"
            test_uuid = str(uuid.uuid4())

            assert (
                storage._execution_target_path(test_slug)
                == storage.execution_targets_folder / f"{test_slug}.toml"
            )
            assert (
                storage._capability_path(test_slug)
                == storage.capabilities_folder / f"{test_slug}.toml"
            )
            assert (
                storage._bench_path(test_slug)
                == storage.benches_folder / f"{test_slug}.toml"
            )
            assert (
                storage._bench_abstraction_path(test_slug)
                == storage.bench_abstractions_folder / f"{test_slug}.py"
            )
            assert (
                storage._experiment_run_report_path(test_uuid)
                == storage.experiment_runs_folder / f"{test_uuid}.json"
            )
            assert (
                storage._optimization_report_path(test_uuid)
                == storage.optimization_reports_folder / f"{test_uuid}.json"
            )
            assert (
                storage._execution_report_path(test_uuid)
                == storage.execution_reports_folder / f"{test_uuid}.json"
            )
            assert (
                storage._analysis_report_path(test_uuid)
                == storage.analysis_reports_folder / f"{test_uuid}.json"
            )
            assert (
                storage._model_path(test_slug)
                == storage.models_folder / f"{test_slug}.tar.gz"
            )
            assert (
                storage._dataset_path(test_slug)
                == storage.datasets_folder / f"{test_slug}.tar.gz"
            )
            assert (
                storage._agent_path(test_slug)
                == storage.agents_folder / f"{test_slug}.tar.gz"
            )

    def test_method_run_report_path(self):
        """Test method_run_report_path returns correct paths based on method kind"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            test_uuid = str(uuid.uuid4())

            # Test each method kind
            assert storage.method_run_report_path(
                test_uuid, MethodKind.TrainingOptimization
            ) == storage._optimization_report_path(test_uuid)
            assert storage.method_run_report_path(
                test_uuid, MethodKind.MeasurementQualification
            ) == storage._execution_report_path(test_uuid)

            assert storage.method_run_report_path(
                test_uuid, MethodKind.Analysis
            ) == storage._analysis_report_path(test_uuid)

    def test_program_info_file_not_found(self):
        """Test program_info raises FileNotFoundError when program.toml doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            with pytest.raises(FileNotFoundError):
                storage.program_info()

    def test_program_info_success(self):
        """Test program_info successfully reads program.toml file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Create a valid program.toml file
            program_data = {
                "program": {
                    "name": "Test Program",
                    "description": "A test program",
                    "version": "1.0.0",
                }
            }

            with open(storage.program_info_path, "w") as f:
                toml.dump(program_data, f)

            # Mock the from_file method to return a mock object
            with patch(
                "overity.storage.local.program_toml.from_file"
            ) as mock_from_file:
                mock_result = Mock()
                mock_from_file.return_value = mock_result

                result = storage.program_info()
                assert result == mock_result
                mock_from_file.assert_called_once_with(storage.program_info_path)

    def test_execution_targets_empty_folder(self):
        """Test execution_targets returns empty generator when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            targets = list(storage.execution_targets())
            assert len(targets) == 0

    def test_execution_targets_with_files(self):
        """Test execution_targets processes TOML files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Create a test TOML file
            test_file = storage.execution_targets_folder / "test-target.toml"
            test_data = {"execution_target": {"name": "Test Target"}}

            with open(test_file, "w") as f:
                toml.dump(test_data, f)

            # Mock the from_file method
            with patch(
                "overity.storage.local.execution_target_toml.from_file"
            ) as mock_from_file:
                mock_result = Mock()
                mock_from_file.return_value = mock_result

                targets = list(storage.execution_targets())
                assert len(targets) == 1
                assert targets[0] == mock_result
                mock_from_file.assert_called_once_with(test_file)

    def test_capabilities_empty_folder(self):
        """Test capabilities returns empty generator when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            capabilities = list(storage.capabilities())
            assert len(capabilities) == 0

    def test_benches_empty_folder(self):
        """Test benches returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            found_benches, found_errors = storage.benches()
            assert len(found_benches) == 0
            assert len(found_errors) == 0

    def test_training_optimization_methods_empty_folder(self):
        """Test training_optimization_methods returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            methods, errors = storage.training_optimization_methods()
            assert len(methods) == 0
            assert len(errors) == 0

    def test_bench_abstractions_empty_folder(self):
        """Test bench_abstractions returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            abstractions, errors = storage.bench_abstractions()
            assert len(abstractions) == 0
            assert len(errors) == 0

    def test_identify_method_kind(self):
        """Test identify_method_kind correctly identifies method kinds"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Create test files in appropriate folders
            training_file = storage.training_optimization_folder / "test.py"
            training_file.touch()

            measurement_file = storage.measurement_qualification_folder / "test.py"
            measurement_file.touch()

            analysis_file = storage.analysis_folder / "test.py"
            analysis_file.touch()

            # Test identification for folders that exist
            assert (
                storage.identify_method_kind(training_file)
                == MethodKind.TrainingOptimization
            )
            assert (
                storage.identify_method_kind(measurement_file)
                == MethodKind.MeasurementQualification
            )
            assert storage.identify_method_kind(analysis_file) == MethodKind.Analysis

            # Test unidentified method
            unknown_file = Path(tmpdir) / "unknown.py"
            unknown_file.touch()

            with pytest.raises(UnidentifiedMethodError):
                storage.identify_method_kind(unknown_file)

    def test_identify_method_slug(self):
        """Test identify_method_slug extracts slug from filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            test_path = Path("/some/path/test-method.py")
            assert storage.identify_method_slug(test_path) == "test-method"

            test_path2 = Path("/another/path/another_test.ipynb")
            assert storage.identify_method_slug(test_path2) == "another_test"

    def test_optimization_reports_empty_folder(self):
        """Test optimization_reports returns empty tuple when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            reports = storage.optimization_reports()
            assert len(reports) == 0

    def test_execution_reports_empty_folder(self):
        """Test execution_reports returns empty tuple when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            reports = storage.execution_reports()
            assert len(reports) == 0

    def test_optimization_report_load_not_found(self):
        """Test optimization_report_load raises ReportNotFound when report doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            with pytest.raises(ReportNotFound):
                storage.optimization_report_load("non-existent-uuid")

    def test_execution_report_load_not_found(self):
        """Test execution_report_load raises ReportNotFound when report doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            with pytest.raises(ReportNotFound):
                storage.execution_report_load("non-existent-uuid")

    def test_optimization_report_remove(self):
        """Test optimization_report_remove removes report file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Create a report file
            report_uuid = "test-uuid"
            report_path = storage._optimization_report_path(report_uuid)
            report_path.touch()

            assert report_path.exists()
            storage.optimization_report_remove(report_uuid)
            assert not report_path.exists()

    def test_execution_report_remove(self):
        """Test execution_report_remove removes report file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Create a report file
            report_uuid = "test-uuid"
            report_path = storage._execution_report_path(report_uuid)
            report_path.touch()

            assert report_path.exists()
            storage.execution_report_remove(report_uuid)
            assert not report_path.exists()

    def test_models_empty_folder(self):
        """Test models returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            models, errors = storage.models()
            assert len(models) == 0
            assert len(errors) == 0

    def test_datasets_empty_folder(self):
        """Test datasets returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            datasets, errors = storage.datasets()
            assert len(datasets) == 0
            assert len(errors) == 0

    def test_inference_agents_empty_folder(self):
        """Test inference_agents returns empty lists when folder is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            agents, errors = storage.inference_agents()
            assert len(agents) == 0
            assert len(errors) == 0

    def test_model_info_get_not_found(self):
        """Test model_info_get raises ModelNotFound when model doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            with pytest.raises(ModelNotFound):
                storage.model_info_get("non-existent-model")

    def test_inference_agent_info_get_not_found(self):
        """Test inference_agent_info_get raises AgentNotFound when agent doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            with pytest.raises(AgentNotFound):
                storage.inference_agent_info_get("non-existent-agent")

    def test_dataset_info_get_not_found(self):
        """Test dataset_info_get returns DatasetNotFound when dataset doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Note: dataset_info_get returns DatasetNotFound instead of raising it
            result = storage.dataset_info_get("non-existent-dataset")
            assert isinstance(result, DatasetNotFound)

    def test_experiment_run_uuid_exists(self):
        """Test experiment_run_uuid_exists checks for report file existence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_uuid = str(uuid.uuid4())
            report_path = storage._experiment_run_report_path(test_uuid)

            # Initially should not exist
            assert not storage.experiment_run_uuid_exists(test_uuid)

            # Create file and check again
            report_path.touch()
            assert storage.experiment_run_uuid_exists(test_uuid)

    def test_optimization_report_uuid_exists(self):
        """Test optimization_report_uuid_exists checks for report file existence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_uuid = str(uuid.uuid4())
            report_path = storage._optimization_report_path(test_uuid)

            # Initially should not exist
            assert not storage.optimization_report_uuid_exists(test_uuid)

            # Create file and check again
            report_path.touch()
            assert storage.optimization_report_uuid_exists(test_uuid)

    def test_execution_report_uuid_exists(self):
        """Test execution_report_uuid_exists checks for report file existence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_uuid = str(uuid.uuid4())
            report_path = storage._execution_report_path(test_uuid)

            # Initially should not exist
            assert not storage.execution_report_uuid_exists(test_uuid)

            # Create file and check again
            report_path.touch()
            assert storage.execution_report_uuid_exists(test_uuid)

    def test_analysis_report_uuid_exists(self):
        """Test analysis_report_uuid_exists checks for report file existence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_uuid = str(uuid.uuid4())
            report_path = storage._analysis_report_path(test_uuid)

            # Initially should not exist
            assert not storage.analysis_report_uuid_exists(test_uuid)

            # Create file and check again
            report_path.touch()
            assert storage.analysis_report_uuid_exists(test_uuid)

    def test_check_is_uuid4(self):
        """Test _check_is_uuid4 validates UUID v4 strings"""
        storage = LocalStorage(Path("/tmp"))

        # Valid UUID v4
        valid_uuid = str(uuid.uuid4())
        assert storage._check_is_uuid4(valid_uuid)

        # Invalid UUIDs
        assert not storage._check_is_uuid4("not-a-uuid")
        assert not storage._check_is_uuid4(
            "12345678-1234-1234-1234-123456789abc"
        )  # Wrong version
        assert not storage._check_is_uuid4("")
        assert not storage._check_is_uuid4("123")

    def test_not_implemented_methods(self):
        """Test that NotImplementedError is raised for unimplemented methods"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Test methods that should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                storage.measurement_qualification_methods()

            with pytest.raises(NotImplementedError):
                storage.analysis_methods()

            with pytest.raises(NotImplementedError):
                storage.experiments()

            with pytest.raises(NotImplementedError):
                storage.experiment_runs()

            with pytest.raises(NotImplementedError):
                storage.analysis_reports()

            with pytest.raises(NotImplementedError):
                storage.experiment_report_load("test")

            with pytest.raises(NotImplementedError):
                storage.analysis_report_load("test")

            with pytest.raises(NotImplementedError):
                storage.experiment_report_remove("test")

            with pytest.raises(NotImplementedError):
                storage.analysis_report_remove("test")

    def test_reports_list_method(self):
        """Test reports_list method delegates to appropriate report type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            # Mock the individual report methods
            with patch.object(storage, "experiment_runs") as mock_exp_runs:
                mock_exp_runs.return_value = ["exp1", "exp2"]
                result = storage.reports_list(
                    MethodReportKind.Experiment, include_all=True
                )
                mock_exp_runs.assert_called_once_with(include_all=True)
                assert result == ["exp1", "exp2"]

            with patch.object(storage, "optimization_reports") as mock_opt_reports:
                mock_opt_reports.return_value = ["opt1", "opt2"]
                result = storage.reports_list(
                    MethodReportKind.TrainingOptimization, include_all=False
                )
                mock_opt_reports.assert_called_once_with(include_all=False)
                assert result == ["opt1", "opt2"]

            with patch.object(storage, "execution_reports") as mock_exec_reports:
                mock_exec_reports.return_value = ["exec1", "exec2"]
                result = storage.reports_list(
                    MethodReportKind.Execution, include_all=True
                )
                mock_exec_reports.assert_called_once_with(include_all=True)
                assert result == ["exec1", "exec2"]

            with patch.object(storage, "analysis_reports") as mock_analysis_reports:
                mock_analysis_reports.return_value = ["analysis1", "analysis2"]
                result = storage.reports_list(
                    MethodReportKind.Analysis, include_all=False
                )
                mock_analysis_reports.assert_called_once_with(include_all=False)
                assert result == ["analysis1", "analysis2"]

    def test_report_load_method(self):
        """Test report_load method delegates to appropriate report type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_id = "test-id"

            # Mock the individual report load methods
            with patch.object(storage, "experiment_report_load") as mock_exp_load:
                mock_exp_load.return_value = ("path1", "report1")
                result = storage.report_load(MethodReportKind.Experiment, test_id)
                mock_exp_load.assert_called_once_with(test_id)
                assert result == ("path1", "report1")

            with patch.object(storage, "optimization_report_load") as mock_opt_load:
                mock_opt_load.return_value = ("path2", "report2")
                result = storage.report_load(
                    MethodReportKind.TrainingOptimization, test_id
                )
                mock_opt_load.assert_called_once_with(test_id)
                assert result == ("path2", "report2")

            with patch.object(storage, "execution_report_load") as mock_exec_load:
                mock_exec_load.return_value = ("path3", "report3")
                result = storage.report_load(MethodReportKind.Execution, test_id)
                mock_exec_load.assert_called_once_with(test_id)
                assert result == ("path3", "report3")

            with patch.object(storage, "analysis_report_load") as mock_analysis_load:
                mock_analysis_load.return_value = ("path4", "report4")
                result = storage.report_load(MethodReportKind.Analysis, test_id)
                mock_analysis_load.assert_called_once_with(test_id)
                assert result == ("path4", "report4")

    def test_report_remove_method(self):
        """Test report_remove method delegates to appropriate report type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            storage.initialize()

            test_id = "test-id"

            # Mock the individual report remove methods
            with patch.object(storage, "experiment_report_remove") as mock_exp_remove:
                storage.report_remove(MethodReportKind.Experiment, test_id)
                mock_exp_remove.assert_called_once_with(test_id)

            with patch.object(storage, "optimization_report_remove") as mock_opt_remove:
                storage.report_remove(MethodReportKind.TrainingOptimization, test_id)
                mock_opt_remove.assert_called_once_with(test_id)

            with patch.object(storage, "execution_report_remove") as mock_exec_remove:
                storage.report_remove(MethodReportKind.Execution, test_id)
                mock_exec_remove.assert_called_once_with(test_id)

            with patch.object(
                storage, "analysis_report_remove"
            ) as mock_analysis_remove:
                storage.report_remove(MethodReportKind.Analysis, test_id)
                mock_analysis_remove.assert_called_once_with(test_id)

    def test_method_report_uuid_get(self):
        """Test method_report_uuid_get delegates to appropriate UUID getter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Mock the individual UUID getter methods
            with patch.object(storage, "optimization_report_uuid_get") as mock_opt_uuid:
                mock_opt_uuid.return_value = "opt-uuid"
                result = storage.method_report_uuid_get(MethodKind.TrainingOptimization)
                mock_opt_uuid.assert_called_once()
                assert result == "opt-uuid"

            with patch.object(storage, "execution_report_uuid_get") as mock_exec_uuid:
                mock_exec_uuid.return_value = "exec-uuid"
                result = storage.method_report_uuid_get(
                    MethodKind.MeasurementQualification
                )
                mock_exec_uuid.assert_called_once()
                assert result == "exec-uuid"

                result = storage.method_report_uuid_get(MethodKind.MeasurementQualification)
                assert result == "exec-uuid"
                assert mock_exec_uuid.call_count == 2

            with patch.object(
                storage, "analysis_report_uuid_get"
            ) as mock_analysis_uuid:
                mock_analysis_uuid.return_value = "analysis-uuid"
                result = storage.method_report_uuid_get(MethodKind.Analysis)
                mock_analysis_uuid.assert_called_once()
                assert result == "analysis-uuid"

    def test_default_uuid_get(self):
        """Test _default_uuid_get generates unique UUIDs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Mock exists_fkt to return False (UUID doesn't exist)
            mock_exists = Mock(return_value=False)
            uuid1 = storage._default_uuid_get(mock_exists)

            # Should be a valid UUID v4
            assert storage._check_is_uuid4(uuid1)

            # Test with exists_fkt that returns True then False
            call_count = 0

            def mock_exists_sequence(_):
                nonlocal call_count
                call_count += 1
                return call_count == 1  # First call returns True, second False

            uuid2 = storage._default_uuid_get(mock_exists_sequence)
            assert storage._check_is_uuid4(uuid2)
            assert call_count == 2  # Should have been called twice
