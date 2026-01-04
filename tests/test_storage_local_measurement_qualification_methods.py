"""
Unit tests for measurement_qualification_methods function
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from overity.storage.local import LocalStorage
from overity.model.general_info.method import MethodKind, MethodInfo, MethodAuthor
from overity.errors import DuplicateSlugError


class TestMeasurementQualificationMethods:
    def test_measurement_qualification_methods_success(self):
        """Test successful retrieval of measurement qualification methods."""
        # Create a temporary directory structure
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('overity.exchange.method_common.file_py.from_file') as mock_py_from_file, \
             patch('overity.exchange.method_common.file_ipynb.from_file') as mock_ipynb_from_file:
            
            # Create mock method info objects
            mock_method1 = MagicMock(spec=MethodInfo)
            mock_method1.slug = "method1"
            mock_method1.path = Path("/test/program/ingredients/measurement_qualification/method1.py")
            
            mock_method2 = MagicMock(spec=MethodInfo)
            mock_method2.slug = "method2"
            mock_method2.path = Path("/test/program/ingredients/measurement_qualification/method2.ipynb")
            
            # Setup file mocks
            mock_py_from_file.return_value = mock_method1
            mock_ipynb_from_file.return_value = mock_method2
            
            # Setup glob to return different files for different calls
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return [Path("/test/program/ingredients/measurement_qualification/method1.py")]
                elif pattern == "*.ipynb":
                    return [Path("/test/program/ingredients/measurement_qualification/method2.ipynb")]
                return []
            
            mock_glob.side_effect = glob_side_effect
            
            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))
            
            # Call the function
            methods, errors = storage.measurement_qualification_methods()
            
            # Verify results
            assert len(methods) == 2
            assert len(errors) == 0
            assert mock_method1 in methods
            assert mock_method2 in methods
            
            # Verify file processing calls
            mock_py_from_file.assert_called_once_with(
                Path("/test/program/ingredients/measurement_qualification/method1.py"),
                kind=MethodKind.MeasurementQualification
            )
            mock_ipynb_from_file.assert_called_once_with(
                Path("/test/program/ingredients/measurement_qualification/method2.ipynb"),
                kind=MethodKind.MeasurementQualification
            )

    def test_measurement_qualification_methods_with_errors(self):
        """Test handling of errors during method processing."""
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('overity.exchange.method_common.file_py.from_file') as mock_py_from_file:
            
            # Setup file mock to raise exception
            mock_py_from_file.side_effect = Exception("Parse error")
            
            # Setup glob to return test files
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return [Path("/test/program/ingredients/measurement_qualification/error_method.py")]
                return []
            
            mock_glob.side_effect = glob_side_effect
            
            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))
            
            # Call the function
            methods, errors = storage.measurement_qualification_methods()
            
            # Verify results
            assert len(methods) == 0
            assert len(errors) == 1
            assert isinstance(errors[0], tuple)
            assert errors[0][0] == Path("/test/program/ingredients/measurement_qualification/error_method.py")
            assert isinstance(errors[0][1], Exception)

    def test_measurement_qualification_methods_with_duplicates(self):
        """Test handling of duplicate method slugs."""
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('overity.exchange.method_common.file_py.from_file') as mock_py_from_file:
            
            # Create mock method info objects with same slug
            mock_method1 = MagicMock(spec=MethodInfo)
            mock_method1.slug = "duplicate_method"
            mock_method1.path = Path("/test/program/ingredients/measurement_qualification/duplicate_method1.py")
            
            mock_method2 = MagicMock(spec=MethodInfo)
            mock_method2.slug = "duplicate_method"
            mock_method2.path = Path("/test/program/ingredients/measurement_qualification/duplicate_method2.py")
            
            # Setup file mock to return different methods
            def from_file_side_effect(path, kind=None):
                if "duplicate_method1.py" in str(path):
                    return mock_method1
                elif "duplicate_method2.py" in str(path):
                    return mock_method2
                return None
            
            mock_py_from_file.side_effect = from_file_side_effect
            
            # Setup glob to return test files
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return [
                        Path("/test/program/ingredients/measurement_qualification/duplicate_method1.py"),
                        Path("/test/program/ingredients/measurement_qualification/duplicate_method2.py")
                    ]
                return []
            
            mock_glob.side_effect = glob_side_effect
            
            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))
            
            # Call the function
            methods, errors = storage.measurement_qualification_methods()
            
            # Verify results
            assert len(methods) == 2  # Both methods are returned
            assert len(errors) == 2  # Two duplicate errors (one for each method)
            
            # Verify error types
            for error_path, error_exc in errors:
                assert isinstance(error_exc, DuplicateSlugError)
                assert error_exc.slug == "duplicate_method"

    def test_measurement_qualification_methods_empty_folder(self):
        """Test handling of empty measurement qualification folder."""
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = []
            
            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))
            
            # Call the function
            methods, errors = storage.measurement_qualification_methods()
            
            # Verify results
            assert len(methods) == 0
            assert len(errors) == 0
            
            # Verify glob was called for both file types
            assert mock_glob.call_count == 2
            mock_glob.assert_any_call("*.py")
            mock_glob.assert_any_call("*.ipynb")

    def test_measurement_qualification_methods_no_notebooks(self):
        """Test when only Python files are present."""
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('overity.exchange.method_common.file_py.from_file') as mock_py_from_file:
            
            # Create mock method info object
            mock_method = MagicMock(spec=MethodInfo)
            mock_method.slug = "python_method"
            mock_method.path = Path("/test/program/ingredients/measurement_qualification/python_method.py")
            
            # Setup file mock
            mock_py_from_file.return_value = mock_method
            
            # Setup glob to return only Python files
            def glob_side_effect(pattern):
                if pattern == "*.py":
                    return [Path("/test/program/ingredients/measurement_qualification/python_method.py")]
                elif pattern == "*.ipynb":
                    return []  # No notebooks
                return []
            
            mock_glob.side_effect = glob_side_effect
            
            # Create LocalStorage instance
            storage = LocalStorage(Path("/test/program"))
            
            # Call the function
            methods, errors = storage.measurement_qualification_methods()
            
            # Verify results
            assert len(methods) == 1
            assert len(errors) == 0
            assert mock_method in methods
            
            # Verify only Python processing was called
            mock_py_from_file.assert_called_once_with(
                Path("/test/program/ingredients/measurement_qualification/python_method.py"),
                kind=MethodKind.MeasurementQualification
            )