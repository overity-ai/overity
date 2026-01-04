"""
Unit tests for backend method measurement qualification functions
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.backend import method as b_method
from overity.storage.local import LocalStorage
from overity.model.general_info.method import MethodInfo


class TestBackendMethodMeasurementQualification:
    @patch('overity.backend.method.LocalStorage')
    def test_list_measurement_qualification_methods_success(self, mock_storage_class):
        """Test successful listing of measurement qualification methods."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage
        
        # Create mock methods
        mock_method1 = MagicMock(spec=MethodInfo)
        mock_method2 = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method1, mock_method2]
        mock_errors = []
        
        # Setup mock return value
        mock_storage.measurement_qualification_methods.return_value = (mock_methods, mock_errors)
        
        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_measurement_qualification_methods(program_path)
        
        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 2
        assert len(errors) == 0
        
        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.measurement_qualification_methods.assert_called_once()

    @patch('overity.backend.method.LocalStorage')
    def test_list_measurement_qualification_methods_with_errors(self, mock_storage_class):
        """Test listing measurement qualification methods with errors."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage
        
        # Create mock methods and errors
        mock_method = MagicMock(spec=MethodInfo)
        mock_methods = [mock_method]
        mock_error = (Path("/test/error.py"), Exception("Parse error"))
        mock_errors = [mock_error]
        
        # Setup mock return value
        mock_storage.measurement_qualification_methods.return_value = (mock_methods, mock_errors)
        
        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_measurement_qualification_methods(program_path)
        
        # Verify results
        assert methods == mock_methods
        assert errors == mock_errors
        assert len(methods) == 1
        assert len(errors) == 1
        
        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.measurement_qualification_methods.assert_called_once()

    @patch('overity.backend.method.LocalStorage')
    def test_list_measurement_qualification_methods_empty_folder(self, mock_storage_class):
        """Test listing measurement qualification methods from empty folder."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage
        
        # Setup mock return value for empty folder
        mock_storage.measurement_qualification_methods.return_value = ([], [])
        
        # Call the function
        program_path = Path("/test/program")
        methods, errors = b_method.list_measurement_qualification_methods(program_path)
        
        # Verify results
        assert len(methods) == 0
        assert len(errors) == 0
        
        # Verify storage was created and called correctly
        mock_storage_class.assert_called_once_with(program_path.resolve())
        mock_storage.measurement_qualification_methods.assert_called_once()

    @patch('overity.backend.method.log')
    @patch('overity.backend.method.LocalStorage')
    def test_list_measurement_qualification_methods_logging(self, mock_storage_class, mock_log):
        """Test that the function logs appropriately."""
        # Create mock storage instance
        mock_storage = MagicMock(spec=LocalStorage)
        mock_storage_class.return_value = mock_storage
        
        # Setup mock return value
        mock_storage.measurement_qualification_methods.return_value = ([], [])
        
        # Call the function
        program_path = Path("/test/program")
        b_method.list_measurement_qualification_methods(program_path)
        
        # Verify logging
        mock_log.info.assert_called_once_with(f"List measurement/qualification methods from program in {program_path.resolve()}")