"""
Unit tests for analysis_reports function in LocalStorage
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.storage.local import LocalStorage
from overity.model.report import MethodExecutionStatus, MethodExecutionStage
from overity.model.general_info.method import MethodKind


class TestAnalysisReports:
    def test_analysis_reports_success_include_all_false(self):
        """Test successful retrieval of analysis reports with include_all=False (default)."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Create mock report files - these represent the actual Path objects returned by glob
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/report1.json"),
                Path("/test/program/shelf/analysis_reports/report2.json"),
                Path("/test/program/shelf/analysis_reports/report3.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info objects
            mock_report1 = MagicMock()
            mock_report1.status = MethodExecutionStatus.ExecutionSuccess

            mock_report2 = MagicMock()
            mock_report2.status = MethodExecutionStatus.ExecutionSuccess

            mock_report3 = MagicMock()
            mock_report3.status = MethodExecutionStatus.ExecutionFailureException

            # Setup report_json.from_file to return different reports
            # The function will call _analysis_report_path to reconstruct the full path
            def from_file_side_effect(path):
                # Check which report this is based on the basename
                basename = os.path.basename(str(path))
                if basename == "report1.json":
                    return mock_report1
                elif basename == "report2.json":
                    return mock_report2
                elif basename == "report3.json":
                    return mock_report3
                return None

            mock_report_json_from_file.side_effect = from_file_side_effect

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function with default include_all=False
            reports = storage.analysis_reports(include_all=False)

            # Verify results - only successful reports should be returned
            assert len(reports) == 2
            assert "report1" in reports
            assert "report2" in reports
            assert "report3" not in reports

            # Verify glob was called correctly
            mock_glob.assert_called_once_with("*.json")

            # Verify report_json.from_file was called for each report
            assert mock_report_json_from_file.call_count == 3

    def test_analysis_reports_success_include_all_true(self):
        """Test successful retrieval of analysis reports with include_all=True."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Create mock report files
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/report1.json"),
                Path("/test/program/shelf/analysis_reports/report2.json"),
                Path("/test/program/shelf/analysis_reports/report3.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info objects with different statuses
            mock_report1 = MagicMock()
            mock_report1.status = MethodExecutionStatus.ExecutionSuccess

            mock_report2 = MagicMock()
            mock_report2.status = MethodExecutionStatus.ExecutionFailureException

            mock_report3 = MagicMock()
            mock_report3.status = MethodExecutionStatus.ExecutionFailureConstraints

            # Setup report_json.from_file to return different reports
            def from_file_side_effect(path):
                basename = os.path.basename(str(path))
                if basename == "report1.json":
                    return mock_report1
                elif basename == "report2.json":
                    return mock_report2
                elif basename == "report3.json":
                    return mock_report3
                return None

            mock_report_json_from_file.side_effect = from_file_side_effect

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function with include_all=True
            reports = storage.analysis_reports(include_all=True)

            # Verify results - all reports should be returned
            assert len(reports) == 3
            assert "report1" in reports
            assert "report2" in reports
            assert "report3" in reports

            # Verify glob was called correctly
            mock_glob.assert_called_once_with("*.json")

    def test_analysis_reports_empty_folder(self):
        """Test analysis_reports with empty folder."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Setup glob to return empty list
            mock_glob.return_value = []

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.analysis_reports()

            # Verify results - empty tuple should be returned
            assert len(reports) == 0
            assert reports == ()

            # Verify glob was called
            mock_glob.assert_called_once_with("*.json")

    def test_analysis_reports_with_invalid_files(self):
        """Test analysis_reports with invalid/corrupted JSON files."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file, patch("overity.storage.local.log") as mock_log:
            # Create mock report files
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/valid_report.json"),
                Path("/test/program/shelf/analysis_reports/invalid_report.json"),
                Path("/test/program/shelf/analysis_reports/corrupted_report.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info object for valid report
            mock_valid_report = MagicMock()
            mock_valid_report.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file to raise exceptions for invalid files
            # The function will call _analysis_report_path to reconstruct the full path from the stem
            def from_file_side_effect(path):
                # The path will be the reconstructed full path: /test/program/shelf/analysis_reports/{stem}.json
                # Use basename for precise matching to avoid substring issues
                basename = os.path.basename(str(path))
                if basename == "valid_report.json":
                    return mock_valid_report
                elif basename == "invalid_report.json":
                    raise ValueError("Invalid JSON format")
                elif basename == "corrupted_report.json":
                    raise Exception("Corrupted file")
                return None

            mock_report_json_from_file.side_effect = from_file_side_effect

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.analysis_reports()

            # Verify results - only valid report should be returned
            assert len(reports) == 1
            assert "valid_report" in reports
            assert "invalid_report" not in reports
            assert "corrupted_report" not in reports

            # Verify logging for invalid files
            assert mock_log.info.call_count == 2  # Two invalid files
            mock_log.debug.assert_called()

    def test_analysis_reports_mixed_valid_invalid(self):
        """Test analysis_reports with mix of valid and invalid files."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file, patch("overity.storage.local.log") as mock_log:
            # Create mock report files
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/report1.json"),
                Path("/test/program/shelf/analysis_reports/report2.json"),
                Path("/test/program/shelf/analysis_reports/report3.json"),
                Path("/test/program/shelf/analysis_reports/report4.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info objects
            mock_report1 = MagicMock()
            mock_report1.status = MethodExecutionStatus.ExecutionSuccess

            mock_report3 = MagicMock()
            mock_report3.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file with mixed results
            def from_file_side_effect(path):
                basename = os.path.basename(str(path))
                if basename == "report1.json":
                    return mock_report1
                elif basename == "report2.json":
                    raise Exception("Invalid report format")
                elif basename == "report3.json":
                    return mock_report3
                elif basename == "report4.json":
                    raise ValueError("Missing required fields")
                return None

            mock_report_json_from_file.side_effect = from_file_side_effect

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.analysis_reports()

            # Verify results - only valid reports should be returned
            assert len(reports) == 2
            assert "report1" in reports
            assert "report3" in reports
            assert "report2" not in reports
            assert "report4" not in reports

            # Verify logging for invalid files
            assert mock_log.info.call_count == 2  # Two invalid files

    def test_analysis_reports_default_parameter(self):
        """Test analysis_reports with default parameter (include_all=False)."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Create mock report files
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/report1.json")
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info object
            mock_report = MagicMock()
            mock_report.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file
            mock_report_json_from_file.return_value = mock_report

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function without explicit parameter (should default to False)
            reports = storage.analysis_reports()

            # Verify results
            assert len(reports) == 1
            assert "report1" in reports

            # Verify that status filtering was applied (include_all=False by default)
            # This is verified by the fact that we get the report (status was ExecutionSuccess)

    def test_analysis_reports_failed_status_filtered(self):
        """Test that reports with failed status are filtered out when include_all=False."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Create mock report files
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/failed_report.json"),
                Path("/test/program/shelf/analysis_reports/running_report.json"),
                Path("/test/program/shelf/analysis_reports/success_report.json"),
            ]

            # Setup glob to return the mock files
            mock_glob.return_value = mock_report_files

            # Create mock report info objects with different statuses
            mock_failed_report = MagicMock()
            mock_failed_report.status = MethodExecutionStatus.ExecutionFailureException

            mock_running_report = MagicMock()
            mock_running_report.status = (
                MethodExecutionStatus.ExecutionFailureConstraints
            )

            mock_success_report = MagicMock()
            mock_success_report.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file to return different reports
            def from_file_side_effect(path):
                basename = os.path.basename(str(path))
                if basename == "failed_report.json":
                    return mock_failed_report
                elif basename == "running_report.json":
                    return mock_running_report
                elif basename == "success_report.json":
                    return mock_success_report
                return None

            mock_report_json_from_file.side_effect = from_file_side_effect

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function with default include_all=False
            reports = storage.analysis_reports()

            # Verify results - only successful report should be returned
            assert len(reports) == 1
            assert "success_report" in reports
            assert "failed_report" not in reports
            assert "running_report" not in reports

    def test_analysis_reports_non_json_files_ignored(self):
        """Test that non-JSON files are ignored by the glob pattern."""
        with patch("pathlib.Path.glob") as mock_glob:
            # Create mock files - only JSON files should be returned by glob("*.json")
            mock_files = [
                Path("/test/program/shelf/analysis_reports/report1.json"),
                Path("/test/program/shelf/analysis_reports/report2.json"),
            ]

            # Setup glob to return only JSON files (this is what glob("*.json") should return)
            mock_glob.return_value = mock_files

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Mock report_json.from_file
            with patch("overity.exchange.report_json.from_file") as mock_from_file:
                mock_report = MagicMock()
                mock_report.status = MethodExecutionStatus.ExecutionSuccess
                mock_from_file.return_value = mock_report

                reports = storage.analysis_reports()

                # Verify results - both JSON files should be processed
                assert len(reports) == 2
                assert "report1" in reports
                assert "report2" in reports

                # Verify glob was called with correct pattern
                mock_glob.assert_called_once_with("*.json")

    def test_analysis_reports_single_file(self):
        """Test analysis_reports with single file."""
        with patch("pathlib.Path.glob") as mock_glob, patch(
            "overity.exchange.report_json.from_file"
        ) as mock_report_json_from_file:
            # Create single mock report file
            mock_report_files = [
                Path("/test/program/shelf/analysis_reports/single_report.json")
            ]

            # Setup glob to return the mock file
            mock_glob.return_value = mock_report_files

            # Create mock report info object
            mock_report = MagicMock()
            mock_report.status = MethodExecutionStatus.ExecutionSuccess

            # Setup report_json.from_file
            mock_report_json_from_file.return_value = mock_report

            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))

            # Call the function
            reports = storage.analysis_reports()

            # Verify results
            assert len(reports) == 1
            assert "single_report" in reports

            # Verify glob was called
            mock_glob.assert_called_once_with("*.json")

            # Verify report_json.from_file was called
            mock_report_json_from_file.assert_called_once()
