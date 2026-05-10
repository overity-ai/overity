"""
Unit tests for preparation_reports function in LocalStorage
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.storage.local import LocalStorage
from overity.model.report import MethodExecutionStatus, MethodExecutionStage
from overity.model.general_info.method import MethodKind, MethodInfo


class TestPreparationReports:
    def test_preparation_reports_success(self):
        """Test successful retrieval of preparation reports."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Create mock report files - these represent the actual Path objects returned by glob
            mock_report_files = [
                Path("/test/program/shelf/preparation_reports/report1.json"),
                Path("/test/program/shelf/preparation_reports/report2.json"),
                Path("/test/program/shelf/preparation_reports/report3.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.preparation_reports()

            # Verify results - all reports should be returned (no filtering)
            assert len(reports) == 3
            assert "report1" in reports
            assert "report2" in reports
            assert "report3" in reports

            # Verify glob was called
            mock_glob.assert_called_once_with("*.json")

    def test_preparation_reports_single_file(self):
        """Test preparation_reports with single file."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Create single mock report file
            mock_report_files = [
                Path("/test/program/shelf/preparation_reports/single_report.json")
            ]

            # Setup glob to return the mock file
            mock_glob.return_value = mock_report_files

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.preparation_reports()

            # Verify results
            assert len(reports) == 1
            assert "single_report" in reports

            # Verify glob was called
            mock_glob.assert_called_once_with("*.json")

    def test_preparation_reports_empty_folder(self):
        """Test preparation_reports with empty folder."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Setup glob to return empty list
            mock_glob.return_value = []

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.preparation_reports()

            # Verify results - empty tuple should be returned
            assert len(reports) == 0
            assert reports == ()

            # Verify glob was called
            mock_glob.assert_called_once_with("*.json")

    def test_preparation_reports_non_json_files_ignored(self):
        """Test that non-JSON files are ignored by the glob pattern."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Create mock files - only JSON files should be returned by glob("*.json")
            mock_files = [
                Path("/test/program/shelf/preparation_reports/report1.json"),
                Path("/test/program/shelf/preparation_reports/report2.json"),
            ]

            # Setup glob to return only JSON files (this is what glob("*.json") should return)
            mock_glob.return_value = mock_files

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            reports = storage.preparation_reports()

            # Verify results - both JSON files should be processed
            assert len(reports) == 2
            assert "report1" in reports
            assert "report2" in reports

            # Verify glob was called with correct pattern
            mock_glob.assert_called_once_with("*.json")

    def test_preparation_report_load_success(self):
        """Test successful loading of a preparation report."""
        with patch("pathlib.Path.is_file") as mock_is_file, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Setup mock file existence check
            mock_is_file.return_value = True

            # Create mock report object
            mock_report = MagicMock()
            mock_report.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file
            mock_report_json_from_file.return_value = mock_report

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            report_path, report = storage.preparation_report_load("test_report")

            # Verify results
            assert report_path == Path("/test/program/shelf/preparation_reports/test_report.json")
            assert report == mock_report

            # Verify is_file was called
            mock_is_file.assert_called_once()

            # Verify report_json.from_file was called with correct path
            mock_report_json_from_file.assert_called_once_with(
                Path("/test/program/shelf/preparation_reports/test_report.json")
            )

    def test_preparation_report_load_not_found(self):
        """Test loading a preparation report that doesn't exist."""
        with patch("pathlib.Path.is_file") as mock_is_file:
            # Setup mock file existence check (file doesn't exist)
            mock_is_file.return_value = False

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function and expect exception
            with pytest.raises(Exception) as exc_info:
                storage.preparation_report_load("nonexistent_report")

            # Verify the exception contains the expected information
            assert "nonexistent_report" in str(exc_info.value)
            assert "preparation" in str(exc_info.value)

            # Verify is_file was called
            mock_is_file.assert_called_once()

    def test_preparation_report_remove_success(self):
        """Test successful removal of a preparation report."""
        with patch("pathlib.Path.unlink") as mock_unlink:
            # Setup mock unlink to succeed
            mock_unlink.return_value = None

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            storage.preparation_report_remove("test_report")

            # Verify unlink was called with correct path and parameters
            mock_unlink.assert_called_once_with(missing_ok=True)

    def test_preparation_report_remove_missing_file(self):
        """Test removal of a preparation report that doesn't exist (should not raise exception)."""
        with patch("pathlib.Path.unlink") as mock_unlink:
            # Setup mock unlink to handle missing file
            mock_unlink.return_value = None

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function - should not raise exception due to missing_ok=True
            storage.preparation_report_remove("nonexistent_report")

            # Verify unlink was called with correct parameters
            mock_unlink.assert_called_once_with(missing_ok=True)

    def test_preparation_report_uuid_exists_true(self):
        """Test checking if a preparation report UUID exists (exists case)."""
        with patch("pathlib.Path.is_file") as mock_is_file:
            # Setup mock file existence check (file exists)
            mock_is_file.return_value = True

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            exists = storage.preparation_report_uuid_exists("test_uuid")

            # Verify result
            assert exists is True

            # Verify is_file was called
            mock_is_file.assert_called_once()

    def test_preparation_report_uuid_exists_false(self):
        """Test checking if a preparation report UUID exists (doesn't exist case)."""
        with patch("pathlib.Path.is_file") as mock_is_file:
            # Setup mock file existence check (file doesn't exist)
            mock_is_file.return_value = False

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            exists = storage.preparation_report_uuid_exists("nonexistent_uuid")

            # Verify result
            assert exists is False

            # Verify is_file was called
            mock_is_file.assert_called_once()

    def test_preparation_report_uuid_get_success(self):
        """Test successful generation of a preparation report UUID."""
        with patch("uuid.uuid4") as mock_uuid4, patch(
            "overity.storage.local.LocalStorage.preparation_report_uuid_exists"
        ) as mock_exists:
            # Setup mock UUID generation
            mock_uuid4.side_effect = ["test-uuid-1", "test-uuid-2", "test-uuid-3"]

            # Setup mock existence check (first two exist, third doesn't)
            mock_exists.side_effect = [True, True, False]

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            uuid_result = storage.preparation_report_uuid_get()

            # Verify result
            assert uuid_result == "test-uuid-3"

            # Verify uuid4 was called 3 times
            assert mock_uuid4.call_count == 3

            # Verify existence check was called 3 times
            assert mock_exists.call_count == 3

    def test_reports_list_preparation_kind(self):
        """Test that reports_list correctly routes to preparation_reports for Preparation kind."""
        with patch("overity.storage.local.LocalStorage.preparation_reports") as mock_prep_reports:
            # Setup mock preparation_reports
            mock_prep_reports.return_value = ("report1", "report2")

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call reports_list with Preparation kind
            from overity.model.report import MethodReportKind
            reports = storage.reports_list(MethodReportKind.Preparation)

            # Verify result
            assert reports == ("report1", "report2")

            # Verify preparation_reports was called
            mock_prep_reports.assert_called_once()

    def test_report_load_preparation_kind(self):
        """Test that report_load correctly routes to preparation_report_load for Preparation kind."""
        with patch("overity.storage.local.LocalStorage.preparation_report_load") as mock_prep_load:
            # Setup mock preparation_report_load
            mock_prep_load.return_value = (Path("test.json"), MagicMock())

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call report_load with Preparation kind
            from overity.model.report import MethodReportKind
            result = storage.report_load(MethodReportKind.Preparation, "test_report")

            # Verify result
            assert result[0] == Path("test.json")

            # Verify preparation_report_load was called
            mock_prep_load.assert_called_once_with("test_report")

    def test_report_remove_preparation_kind(self):
        """Test that report_remove correctly routes to preparation_report_remove for Preparation kind."""
        with patch("overity.storage.local.LocalStorage.preparation_report_remove") as mock_prep_remove:
            # Setup mock preparation_report_remove
            mock_prep_remove.return_value = None

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call report_remove with Preparation kind
            from overity.model.report import MethodReportKind
            storage.report_remove(MethodReportKind.Preparation, "test_report")

            # Verify preparation_report_remove was called
            mock_prep_remove.assert_called_once_with("test_report")

    def test_method_run_report_path_preparation_kind(self):
        """Test that method_run_report_path returns correct path for preparation methods."""
        # Create LocalStorage instance
        storage = LocalStorage(Path("/test/program"))

        # Call method_run_report_path with Preparation kind
        path = storage.method_run_report_path("test_uuid", MethodKind.Preparation)

        # Verify result
        assert path == Path("/test/program/shelf/preparation_reports/test_uuid.json")

    def test_preparation_methods_success(self):
        """Test successful retrieval of preparation methods."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.method_common.file_py.from_file"
        ) as mock_file_py, patch(
            "overity.exchange.method_common.file_ipynb.from_file"
        ) as mock_file_ipynb:
            # Create mock method files
            mock_py_files = [
                Path("/test/program/ingredients/preparation/method1.py"),
                Path("/test/program/ingredients/preparation/method2.py"),
            ]
            mock_ipynb_files = [
                Path("/test/program/ingredients/preparation/method3.ipynb"),
            ]

            # Setup glob to return different file types
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return mock_py_files
                elif pattern == "*.ipynb":
                    return mock_ipynb_files
                return []

            mock_glob.side_effect = glob_side_effect

            # Create mock method info objects
            mock_method1 = MagicMock(spec=MethodInfo)
            mock_method1.slug = "method1"
            mock_method2 = MagicMock(spec=MethodInfo)
            mock_method2.slug = "method2"
            mock_method3 = MagicMock(spec=MethodInfo)
            mock_method3.slug = "method3"

            # Setup file parsers to return mock methods (avoid file system access)
            mock_file_py.side_effect = [mock_method1, mock_method2]
            mock_file_ipynb.return_value = mock_method3

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            methods, errors = storage.preparation_methods()

            # Verify results
            assert len(methods) == 3
            assert len(errors) == 0
            assert "method1" in [m.slug for m in methods]
            assert "method2" in [m.slug for m in methods]
            assert "method3" in [m.slug for m in methods]

            # Verify glob was called for both file types
            assert mock_glob.call_count == 2
            mock_glob.assert_any_call("*.py")
            mock_glob.assert_any_call("*.ipynb")

            # Verify file parsers were called
            assert mock_file_py.call_count == 2
            assert mock_file_ipynb.call_count == 1

    def test_preparation_methods_with_errors(self):
        """Test preparation methods with some files causing errors."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.method_common.file_py.from_file"
        ) as mock_file_py, patch("logging.getLogger") as mock_get_logger:
            # Create mock method files
            mock_py_files = [
                Path("/test/program/ingredients/preparation/valid_method.py"),
                Path("/test/program/ingredients/preparation/invalid_method.py"),
            ]

            # Setup glob to return files
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return mock_py_files
                return []

            mock_glob.side_effect = glob_side_effect

            # Create mock method info objects for each file
            mock_valid_method = MagicMock(spec=MethodInfo)
            mock_valid_method.slug = "valid_method"
            mock_valid_method.path = Path("/test/program/ingredients/preparation/valid_method.py")

            # Setup file parser to return valid method for first file, raise exception for second
            def file_py_side_effect(path, kind):
                basename = Path(str(path)).name
                if basename == "valid_method.py":
                    return mock_valid_method
                elif basename == "invalid_method.py":
                    raise ValueError("Invalid method format")
                return None

            mock_file_py.side_effect = file_py_side_effect

            # Create mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            methods, errors = storage.preparation_methods()

            # Verify results
            assert len(methods) == 1
            assert len(errors) == 1
            assert "valid_method" in [m.slug for m in methods]
            assert any("invalid_method" in str(error[0]) for error in errors)

            # Verify file parser was called for both files
            assert mock_file_py.call_count == 2

    def test_identify_method_kind_preparation(self):
        """Test identify_method_kind correctly identifies preparation methods."""
        # Create LocalStorage instance
        storage = LocalStorage(Path("/test/program"))

        # Create mock path for preparation method
        prep_path = Path("/test/program/ingredients/preparation/test_method.py")

        # Call the function
        kind = storage.identify_method_kind(prep_path)

        # Verify result
        assert kind == MethodKind.Preparation