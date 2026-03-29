"""
Unit tests for backend method analysis functions
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.backend import method as b_method
from overity.storage.local import LocalStorage
from overity.model.general_info.method import MethodInfo


class TestBackendMethodAnalysis:
    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_success(self, mock_storage_class):
        """Test successful listing of analysis methods."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Create mock methods
        mock_method1 = MagicMock(spec=MethodInfo)
        mock_method2 = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method1, mock_method2]
        mock_errors = []

        # Setup mock return value
        mock_storage.analysis_methods.return_value = (mock_methods, mock_errors)

        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 2
        assert len(errors) == 0

        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.analysis_methods.assert_called_once()

    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_with_errors(self, mock_storage_class):
        """Test listing analysis methods with errors."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Create mock methods and errors
        mock_method = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method]
        mock_error = (Path("/test/error.py"), Exception("Parse error"))
        mock_errors = [mock_error]

        # Setup mock return value
        mock_storage.analysis_methods.return_value = (mock_methods, mock_errors)

        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 1
        assert len(errors) == 1

        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.analysis_methods.assert_called_once()

    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_empty_folder(self, mock_storage_class):
        """Test listing analysis methods from empty folder."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Setup mock return value for empty folder
        mock_storage.analysis_methods.return_value = ([], [])

        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert len(methods) == 0
        assert len(errors) == 0

        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.analysis_methods.assert_called_once()

    @patch("overity.backend.method.log")
    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_logging(self, mock_storage_class, mock_log):
        """Test that the function logs appropriately."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Setup mock return value
        mock_storage.analysis_methods.return_value = ([], [])

        # Call the function
        program_path = Path("/test/program")
        b_method.list_analysis_methods(program_path)

        # Verify logging
        mock_log.info.assert_called_once_with(
            f"List analysis methods from program in {program_path.resolve()}"
        )

    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_string_path(self, mock_storage_class):
        """Test that the function accepts string paths."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Create mock methods
        mock_method = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method]
        mock_errors = []

        # Setup mock return value
        mock_storage.analysis_methods.return_value = (mock_methods, mock_errors)

        # Call the function with string path
        program_path = "/test/program"
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 1
        assert len(errors) == 0

        # Verify storage was created with resolved path
        expected_path = Path(program_path).resolve()
        mock_storage_class.assert_called_once_with(expected_path)
        mock_storage.analysis_methods.assert_called_once()

    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_large_number_of_methods(self, mock_storage_class):
        """Test listing a large number of analysis methods."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Create many mock methods
        mock_methods = []
        for i in range(100):
            mock_method = MagicMock(spec=MethodInfo)
            mock_method.slug = f"analysis_method_{i}"
            mock_methods.append(mock_method)

        mock_errors = []

        # Setup mock return value
        mock_storage.analysis_methods.return_value = (mock_methods, mock_errors)

        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 100
        assert len(errors) == 0

        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.analysis_methods.assert_called_once()

    @patch("overity.backend.method.LocalStorage")
    def test_list_analysis_methods_with_many_errors(self, mock_storage_class):
        """Test listing analysis methods with many errors."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage

        # Create mock methods and many errors
        mock_method = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method]

        mock_errors = []
        for i in range(10):
            mock_error = (Path(f"/test/error_{i}.py"), Exception(f"Parse error {i}"))
            mock_errors.append(mock_error)

        # Setup mock return value
        mock_storage.analysis_methods.return_value = (mock_methods, mock_errors)

        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_analysis_methods(program_path)

        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 1
        assert len(errors) == 10

        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.analysis_methods.assert_called_once()
