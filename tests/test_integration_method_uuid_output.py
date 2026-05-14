"""
Integration tests for method execution with report UUID output
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from io import StringIO
import subprocess
import sys

from overity.frontend.method.run_cmd import run
from overity.model.general_info.method import MethodKind
from overity.backend.flow import exit_handler
from overity.backend.flow.ctx import FlowCtx
from overity.model.report import MethodReport, MethodExecutionStatus


class TestMethodExecutionWithUUID:
    """Integration tests for complete method execution flow with UUID output"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_args = Mock()
        self.test_args.operation = False
        self.test_args.bench = None
        self.test_args.method_kind = MethodKind.TrainingOptimization
        self.test_args.method_slug = "test_method"
        self.test_args.method_arguments = ["arg1", "arg2"]

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_complete_method_execution_with_uuid_capture(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test complete method execution where subprocess captures and returns UUID."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        
        # Create a StringIO to capture what would be the method's stdout
        method_stdout_capture = StringIO()
        
        # Mock subprocess to simulate a method that prints output and ends with UUID
        def mock_subprocess_run(cmd, **kwargs):
            # Simulate method execution output
            method_stdout_capture.write("Training model...\n")
            method_stdout_capture.write("Epoch 1/10 completed\n")
            method_stdout_capture.write("Epoch 2/10 completed\n")
            method_stdout_capture.write("Training completed successfully\n")
            method_stdout_capture.write("report-uuid-abc123def456\n")  # UUID at the end
            
            # Create mock result
            result = MagicMock()
            result.returncode = 0
            result.stdout = method_stdout_capture.getvalue()
            return result
        
        mock_subprocess.side_effect = mock_subprocess_run
        mock_environ.get.return_value = "preview"
        
        # Run the method
        with patch('sys.exit') as mock_exit:
            run(self.test_args)
            
            # Verify subprocess was called
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "python" in call_args[0]
            assert str(mock_find_method_path.return_value) in call_args
            
            # Verify exit code
            mock_exit.assert_called_once_with(0)
            
            # Verify the captured output contains UUID
            output = method_stdout_capture.getvalue()
            assert "report-uuid-abc123def456" in output
            assert output.endswith("report-uuid-abc123def456\n")

    def test_exit_handler_integration_with_real_stdout(self):
        """Test exit_handler with real stdout capture."""
        # Create a real StringIO to simulate stdout
        captured_stdout = StringIO()
        
        # Create a mock context similar to what would be used in real execution
        mock_ctx = Mock(spec=FlowCtx)
        mock_ctx.method_kind = MethodKind.TrainingOptimization
        mock_ctx.exceptions = []
        mock_ctx.stdout = captured_stdout
        
        # Create a mock report with UUID
        mock_report = Mock(spec=MethodReport)
        mock_report.uuid = "integration-test-uuid-789"
        mock_report.status = None
        mock_report.date_ended = None
        mock_report.traceability_graph = []
        mock_ctx.report = mock_report
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.method_run_report_path.return_value = Path("/test/path/report.json")
        mock_ctx.storage = mock_storage
        
        # Mock bench instance (None for TO method)
        mock_ctx.bench_instance = None
        
        with patch('overity.backend.flow.report_json.to_file'):
            # Call exit_handler
            exit_handler(mock_ctx)
            
            # Verify UUID was printed to stdout
            output = captured_stdout.getvalue()
            assert output == "integration-test-uuid-789\n"
            
            # Verify report status was set to success
            assert mock_report.status == MethodExecutionStatus.ExecutionSuccess

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_method_execution_failure_still_outputs_uuid(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test that method execution failure still results in UUID output."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        
        # Mock subprocess to return failure but still output UUID
        mock_result = MagicMock()
        mock_result.returncode = 1  # Failure exit code
        mock_result.stdout = "Method failed with error\nBut still outputs UUID: report-uuid-fail123\n"
        mock_subprocess.return_value = mock_result
        
        mock_environ.get.return_value = "preview"
        
        # Run the method (should exit with error code)
        with patch('sys.exit') as mock_exit:
            run(self.test_args)
            
            # Verify subprocess was called
            mock_subprocess.assert_called_once()
            
            # Verify exit code reflects failure
            mock_exit.assert_called_once_with(1)

    def test_exit_handler_preserves_uuid_format(self):
        """Test that exit_handler preserves the exact UUID format from the report."""
        test_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",  # Standard UUID format
            "simple-id-123",
            "complex.uuid.with.dots-and-dashes_123",
            "very-long-uuid-that-might-come-from-some-uuid-generation-library",
            "",  # Empty UUID edge case
        ]
        
        for test_uuid in test_uuids:
            captured_stdout = StringIO()
            
            # Create mock context
            mock_ctx = Mock()
            mock_ctx.method_kind = MethodKind.TrainingOptimization
            mock_ctx.exceptions = []
            mock_ctx.stdout = captured_stdout
            
            # Mock report with test UUID
            mock_report = Mock()
            mock_report.uuid = test_uuid
            mock_report.status = None
            mock_report.date_ended = None
            mock_report.traceability_graph = []
            mock_ctx.report = mock_report
            
            # Mock storage
            mock_storage = Mock()
            mock_storage.method_run_report_path.return_value = Path("/test/path.json")
            mock_ctx.storage = mock_storage
            mock_ctx.bench_instance = None
            
            with patch('overity.backend.flow.report_json.to_file'):
                # Call exit_handler
                exit_handler(mock_ctx)
                
                # Verify UUID was printed exactly as is
                output = captured_stdout.getvalue()
                expected_output = f"{test_uuid}\n"
                assert output == expected_output, f"UUID '{test_uuid}' was not preserved correctly"

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_measurement_qualification_method_with_bench_uuid_output(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test DMQ method execution with bench setup and UUID output."""
        # Setup for DMQ method
        self.test_args.method_kind = MethodKind.MeasurementQualification
        self.test_args.bench = "test_bench"
        
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/measurement_qualification/test_method.py")
        
        # Mock subprocess for DMQ method
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Bench measurement completed\nreport-uuid-dmq789\n"
        mock_subprocess.return_value = mock_result
        
        # Setup environment mocks - simulate no existing OVERITY_BENCH env var to trigger setting it
        def mock_contains(key):
            return False  # Neither OVERITY_STAGE nor OVERITY_BENCH are initially in environ
        
        def mock_get(key, default=None):
            return default  # Return default for all keys
        
        mock_environ.__contains__ = Mock(side_effect=mock_contains)
        mock_environ.get.side_effect = mock_get
        
        # Run the method
        with patch('sys.exit') as mock_exit:
            run(self.test_args)
            
            # Verify subprocess was called (meaning validation passed)
            mock_subprocess.assert_called_once()
            
            # Verify environment variables were set correctly
            # OVERITY_BENCH should be set during execution
            assert mock_environ.__setitem__.call_count >= 1
            
            # Verify exit code is 0 (success)
            # Note: sys.exit might be called multiple times, but the last call should be with 0
            last_exit_call = mock_exit.call_args
            assert last_exit_call[0][0] == 0
            
            # Verify the output contains UUID
            assert "report-uuid-dmq789" in mock_result.stdout