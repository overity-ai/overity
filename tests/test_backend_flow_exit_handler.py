"""
Unit tests for backend flow exit_handler function and UUID printing
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from io import StringIO
from datetime import datetime

from overity.backend.flow import exit_handler
from overity.model.report import MethodExecutionStatus
from overity.model.general_info.method import MethodKind


class TestBackendFlowExitHandler:
    """Test cases for exit_handler function and UUID printing to stdout"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock context
        self.mock_ctx = Mock()
        self.mock_ctx.init_ok = True
        self.mock_ctx.method_kind = MethodKind.TrainingOptimization
        self.mock_ctx.exceptions = []
        
        # Mock report
        self.mock_ctx.report = Mock()
        self.mock_ctx.report.uuid = "test-report-uuid-12345"
        self.mock_ctx.report.status = None
        self.mock_ctx.report.date_ended = None
        self.mock_ctx.report.traceability_graph = []
        
        # Mock storage
        self.mock_ctx.storage = Mock()
        self.mock_ctx.storage.method_run_report_path.return_value = Path("/test/report/path.json")
        
        # Mock stdout using StringIO to capture output
        self.mock_stdout = StringIO()
        self.mock_ctx.stdout = self.mock_stdout
        
        # Mock bench-related attributes for DMQ methods
        self.mock_ctx.bench_instance = None

    @patch('overity.backend.flow.report_json.to_file')
    def test_exit_handler_prints_uuid_to_stdout(self, mock_report_to_file):
        """Test that exit_handler prints the report UUID to stdout."""
        # Setup
        mock_report_to_file.return_value = None
        
        # Call exit_handler
        exit_handler(self.mock_ctx)
        
        # Verify UUID was printed to stdout
        output = self.mock_stdout.getvalue()
        assert output == "test-report-uuid-12345\n"
        
        # Verify report status was set to success (no exceptions)
        assert self.mock_ctx.report.status == MethodExecutionStatus.ExecutionSuccess
        
        # Verify report end date was set (should be a datetime object)
        assert self.mock_ctx.report.date_ended is not None
        
        # Verify report was saved
        mock_report_to_file.assert_called_once_with(
            self.mock_ctx.report, 
            Path("/test/report/path.json")
        )

    @patch('overity.backend.flow.report_json.to_file')
    def test_exit_handler_with_exceptions_sets_failure_status(self, mock_report_to_file):
        """Test that exit_handler sets failure status when there are exceptions."""
        # Setup exceptions
        self.mock_ctx.exceptions = [Exception("Test error 1"), ValueError("Test error 2")]
        mock_report_to_file.return_value = None
        
        # Call exit_handler
        exit_handler(self.mock_ctx)
        
        # Verify UUID was still printed to stdout
        output = self.mock_stdout.getvalue()
        assert output == "test-report-uuid-12345\n"
        
        # Verify report status was set to failure
        assert self.mock_ctx.report.status == MethodExecutionStatus.ExecutionFailureException

    @patch('overity.backend.flow.report_json.to_file')
    def test_exit_handler_dmq_method_with_bench_cleanup(self, mock_report_to_file):
        """Test exit_handler for DMQ methods with bench cleanup."""
        # Setup for DMQ method
        self.mock_ctx.method_kind = MethodKind.MeasurementQualification
        mock_bench_instance = Mock()
        mock_bench_instance.traceability_graph = [("bench_key", "bench_value")]
        self.mock_ctx.bench_instance = mock_bench_instance
        
        mock_report_to_file.return_value = None
        
        # Call exit_handler
        exit_handler(self.mock_ctx)
        
        # Verify bench cleanup was called
        mock_bench_instance.bench_cleanup.assert_called_once()
        mock_bench_instance.tmpdir_cleanup.assert_called_once()
        
        # Verify traceability graph was merged
        assert ("bench_key", "bench_value") in self.mock_ctx.report.traceability_graph
        
        # Verify UUID was still printed
        output = self.mock_stdout.getvalue()
        assert output == "test-report-uuid-12345\n"

    @patch('overity.backend.flow.report_json.to_file')
    @patch('overity.backend.flow.log')
    def test_exit_handler_dmq_method_bench_cleanup_error(self, mock_log, mock_report_to_file):
        """Test exit_handler handles bench cleanup errors gracefully."""
        # Setup for DMQ method with bench cleanup error
        self.mock_ctx.method_kind = MethodKind.MeasurementQualification
        mock_bench_instance = Mock()
        mock_bench_instance.bench_cleanup.side_effect = Exception("Bench cleanup failed")
        mock_bench_instance.traceability_graph = []  # Empty list to avoid iteration issues
        self.mock_ctx.bench_instance = mock_bench_instance
        
        mock_report_to_file.return_value = None
        
        # Call exit_handler - should not raise exception
        exit_handler(self.mock_ctx)
        
        # Verify bench cleanup was attempted
        mock_bench_instance.bench_cleanup.assert_called_once()
        
        # Verify error was logged and exception was added to context
        mock_log.error.assert_called()
        assert len(self.mock_ctx.exceptions) == 1
        assert str(self.mock_ctx.exceptions[0]) == "Bench cleanup failed"
        
        # Verify UUID was still printed despite cleanup error
        output = self.mock_stdout.getvalue()
        assert output == "test-report-uuid-12345\n"

    @patch('overity.backend.flow.report_json.to_file')
    def test_exit_handler_different_uuid_values(self, mock_report_to_file):
        """Test exit_handler with different UUID values."""
        test_cases = [
            "simple-uuid",
            "123e4567-e89b-12d3-a456-426614174000",
            "uuid-with-special-chars-!@#$%",
            "very-long-uuid-that-could-be-generated-by-some-uuid-library-function"
        ]
        
        mock_report_to_file.return_value = None
        
        for test_uuid in test_cases:
            with StringIO() as mock_stdout:
                self.mock_ctx.stdout = mock_stdout
                self.mock_ctx.report.uuid = test_uuid
                
                # Call exit_handler
                exit_handler(self.mock_ctx)
                
                # Verify UUID was printed correctly
                output = mock_stdout.getvalue()
                assert output == f"{test_uuid}\n"

    @patch('overity.backend.flow.report_json.to_file')
    def test_exit_handler_empty_uuid(self, mock_report_to_file):
        """Test exit_handler with empty UUID."""
        self.mock_ctx.report.uuid = ""
        mock_report_to_file.return_value = None
        
        # Call exit_handler
        exit_handler(self.mock_ctx)
        
        # Verify empty UUID was printed (just newline)
        output = self.mock_stdout.getvalue()
        assert output == "\n"


# Import Path for use in tests
from pathlib import Path