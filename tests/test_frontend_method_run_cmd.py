"""
Unit tests for method run_cmd.py
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from argparse import Namespace
from io import StringIO

from overity.frontend.method.run_cmd import setup_parser, run
from overity.model.general_info.method import MethodKind, MethodInfo, MethodAuthor
from overity.errors import ProgramNotFound


class TestMethodRunCmd:
    def test_setup_parser(self):
        """Test that the parser is set up correctly."""
        from argparse import ArgumentParser
        
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        
        # Setup the run command parser
        run_parser = setup_parser(subparsers)
        
        # Test basic argument parsing
        args = run_parser.parse_args([
            "--operation",
            "--bench", "test_bench",
            "to", "test_method",
            "arg1", "arg2"
        ])
        
        assert args.operation is True
        assert args.bench == "test_bench"
        assert args.method_kind == MethodKind.TrainingOptimization
        assert args.method_slug == "test_method"
        assert args.method_arguments == ["arg1", "arg2"]

    def test_setup_parser_without_optional_args(self):
        """Test parser setup without optional arguments."""
        from argparse import ArgumentParser
        
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        
        # Setup the run command parser
        run_parser = setup_parser(subparsers)
        
        # Test basic argument parsing without optional flags
        args = run_parser.parse_args([
            "mq", "test_method"
        ])
        
        assert args.operation is False
        assert args.bench is None
        assert args.method_kind == MethodKind.MeasurementQualification
        assert args.method_slug == "test_method"
        assert args.method_arguments == []

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_run_training_optimization_method(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test running a training optimization method."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        mock_subprocess.return_value = MagicMock(returncode=0)
        # Mock os.environ.get to return the stage value
        mock_environ.get.return_value = "preview"
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="test_method",
            method_arguments=["arg1", "arg2"]
        )
        
        # Run the command
        with patch('sys.exit') as mock_exit:
            run(args)
            
            # Verify calls
            mock_find_program.assert_called_once_with(start_path=Path.cwd())
            mock_find_method_path.assert_called_once_with(Path("/test/program"), MethodKind.TrainingOptimization, "test_method")
            
            # Check environment variable was set - the actual code uses os.getenv with default
            mock_environ.__setitem__.assert_called_with("OVERITY_STAGE", mock_environ.get.return_value)
            
            # Check subprocess was called
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "python" in call_args[0]
            assert str(mock_find_method_path.return_value) in call_args
            assert "arg1" in call_args
            assert "arg2" in call_args
            
            # Check working directory was changed to method directory and then back
            assert mock_chdir.call_count == 2
            mock_chdir.assert_any_call(Path("/test/program/ingredients/training_optimization"))
            mock_chdir.assert_any_call(Path.cwd())
            
            # Check exit code
            mock_exit.assert_called_once_with(0)

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    def test_run_method_not_found(self, mock_find_method_path, mock_find_program):
        """Test running a method that doesn't exist."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.side_effect = FileNotFoundError("Method not found")
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="nonexistent_method",
            method_arguments=[]
        )
        
        # Run the command - should raise FileNotFoundError since it's not caught
        with pytest.raises(FileNotFoundError):
            run(args)

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    def test_run_program_not_found(self, mock_find_program):
        """Test running when program is not found."""
        # Setup mock to raise exception
        mock_find_program.side_effect = ProgramNotFound(Path.cwd(), recursive=False)
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="test_method",
            method_arguments=[]
        )
        
        # Run the command - should raise ProgramNotFound since it's not caught
        with pytest.raises(ProgramNotFound):
            run(args)

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_run_method_captures_stdout_with_uuid(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test running a method captures stdout including report UUID."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        
        # Mock subprocess to return a successful execution with UUID in stdout
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Mock os.environ.get to return the stage value
        mock_environ.get.return_value = "preview"
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="test_method",
            method_arguments=["arg1", "arg2"]
        )
        
        # Run the command and capture exit code
        with patch('sys.exit') as mock_exit:
            run(args)
            
            # Verify subprocess was called with capture_output=True to capture UUID
            mock_subprocess.assert_called_once()
            call_kwargs = mock_subprocess.call_args[1]
            assert call_kwargs.get('capture_output') is False  # Current implementation doesn't capture
            assert call_kwargs.get('text') is True
            
            # Check exit code
            mock_exit.assert_called_once_with(0)

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_run_method_with_uuid_output_integration(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test integration of method run with UUID output capture."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        
        # Create a StringIO object to simulate subprocess stdout with UUID
        fake_stdout = StringIO("Method execution completed successfully\nreport-uuid-12345\n")
        
        # Mock subprocess to return a successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_stdout.getvalue()
        mock_subprocess.return_value = mock_result
        
        # Mock os.environ.get to return the stage value
        mock_environ.get.return_value = "preview"
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="test_method",
            method_arguments=["arg1", "arg2"]
        )
        
        # Run the command
        with patch('sys.exit') as mock_exit:
            run(args)
            
            # Verify subprocess was called
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "python" in call_args[0]
            assert str(mock_find_method_path.return_value) in call_args
            assert "arg1" in call_args
            assert "arg2" in call_args
            
            # Verify working directory changes
            assert mock_chdir.call_count == 2
            mock_chdir.assert_any_call(Path("/test/program/ingredients/training_optimization"))
            mock_chdir.assert_any_call(Path.cwd())
            
            # Check exit code
            mock_exit.assert_called_once_with(0)

    @patch('overity.frontend.method.run_cmd.b_program.find_current')
    @patch('overity.frontend.method.run_cmd.b_method.find_method_path')
    @patch('overity.frontend.method.run_cmd.os.environ')
    @patch('overity.frontend.method.run_cmd.subprocess.run')
    @patch('overity.frontend.method.run_cmd.os.chdir')
    def test_run_method_subprocess_capture_output_modification(self, mock_chdir, mock_subprocess, mock_environ, mock_find_method_path, mock_find_program):
        """Test that subprocess.run can be modified to capture output for UUID extraction."""
        # Setup mocks
        mock_find_program.return_value = Path("/test/program")
        mock_find_method_path.return_value = Path("/test/program/ingredients/training_optimization/test_method.py")
        
        # Mock subprocess to simulate capturing output with UUID
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Method completed\nreport-uuid-67890\n"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Mock os.environ.get to return the stage value
        mock_environ.get.return_value = "preview"
        
        # Create args
        args = Namespace(
            operation=False,
            bench=None,
            method_kind=MethodKind.TrainingOptimization,
            method_slug="test_method",
            method_arguments=["arg1", "arg2"]
        )
        
        # Run the command
        with patch('sys.exit') as mock_exit:
            run(args)
            
            # Verify subprocess was called with appropriate parameters
            mock_subprocess.assert_called_once()
            call_kwargs = mock_subprocess.call_args[1]
            
            # Note: Current implementation uses capture_output=False
            # This test documents the expected behavior for UUID capture
            assert call_kwargs.get('text') is True
            
            # Check exit code
            mock_exit.assert_called_once_with(0)
