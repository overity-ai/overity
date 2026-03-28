"""
Unit tests for table API functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

import overity.api
from overity.model.report.table import Table
from overity.errors import UninitAPIError


class TestApiTables:
    """Test cases for table API functions: table_save_df, table_save_dict"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock report with tables dictionary
        self.mock_report = Mock()
        self.mock_report.tables = {}

        # Create a sample DataFrame for testing
        self.test_df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [95.5, 87.2, 92.1],
            }
        )

        # Create sample dictionary data for testing
        self.test_dict_data = [
            {"name": "Alice", "age": 25, "score": 95.5},
            {"name": "Bob", "age": 30, "score": 87.2},
            {"name": "Charlie", "age": 35, "score": 92.1},
        ]

    def test_table_save_df_basic(self):
        """Test basic functionality of table_save_df API function."""
        with patch("overity.api._CTX") as mock_ctx:
            # Set up mock context
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Call the API function
            result = overity.api.table_save_df(
                "test_table", self.test_df, "Test scores"
            )

            # Verify the table was saved
            assert "test_table" in mock_ctx.report.tables
            table = mock_ctx.report.tables["test_table"]
            assert table.identifier == "test_table"
            assert table.caption == "Test scores"
            assert table.columns == ("name", "age", "score")
            assert len(table.rows) == 3

            # Verify the result is None (no return value expected)
            assert result is None

    def test_table_save_df_no_caption(self):
        """Test table_save_df without caption."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Call without caption
            overity.api.table_save_df("test_table", self.test_df)

            # Verify caption is empty string
            table = mock_ctx.report.tables["test_table"]
            assert table.caption == ""

    def test_table_save_df_empty_dataframe(self):
        """Test table_save_df with empty DataFrame."""
        empty_df = pd.DataFrame()

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_df("empty_table", empty_df, "Empty table")

            table = mock_ctx.report.tables["empty_table"]
            assert table.identifier == "empty_table"
            assert table.caption == "Empty table"
            assert table.columns == ()
            assert table.rows == ()

    def test_table_save_df_with_nan_values(self):
        """Test table_save_df with NaN values."""
        df_with_nan = pd.DataFrame(
            {"name": ["Alice", "Bob", None], "score": [95.5, np.nan, 92.1]}
        )

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_df("nan_table", df_with_nan)

            table = mock_ctx.report.tables["nan_table"]
            assert table.columns == ("name", "score")
            assert len(table.rows) == 3
            # Check that NaN values are converted to None
            assert table.rows[1][1] is None  # Bob's score (NaN)
            assert table.rows[2][0] is None  # Third name (None)

    def test_table_save_df_duplicate_identifier(self):
        """Test table_save_df with duplicate identifier raises error."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Save first table
            overity.api.table_save_df("duplicate_id", self.test_df)

            # Try to save second table with same identifier
            with pytest.raises(
                ValueError, match="Table with identifier 'duplicate_id' already exists"
            ):
                overity.api.table_save_df("duplicate_id", self.test_df)

    def test_table_save_dict_basic(self):
        """Test basic functionality of table_save_dict API function."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Call the API function
            result = overity.api.table_save_dict(
                "test_dict_table", self.test_dict_data, "Test dict scores"
            )

            # Verify the table was saved
            assert "test_dict_table" in mock_ctx.report.tables
            table = mock_ctx.report.tables["test_dict_table"]
            assert table.identifier == "test_dict_table"
            assert table.caption == "Test dict scores"
            assert table.columns == ("name", "age", "score")  # Preserves original order
            assert len(table.rows) == 3

            # Verify the result is None (no return value expected)
            assert result is None

    def test_table_save_dict_no_caption(self):
        """Test table_save_dict without caption."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Call without caption
            overity.api.table_save_dict("test_dict_table", self.test_dict_data)

            # Verify caption is empty string
            table = mock_ctx.report.tables["test_dict_table"]
            assert table.caption == ""

    def test_table_save_dict_empty_list(self):
        """Test table_save_dict with empty list."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_dict("empty_dict_table", [], "Empty dict table")

            table = mock_ctx.report.tables["empty_dict_table"]
            assert table.identifier == "empty_dict_table"
            assert table.caption == "Empty dict table"
            assert table.columns == ()
            assert table.rows == ()

    def test_table_save_dict_missing_keys(self):
        """Test table_save_dict with dictionaries having different keys."""
        incomplete_data = [
            {"name": "Alice", "age": 25, "score": 95.5},
            {"name": "Bob", "score": 87.2},  # Missing age
            {"age": 35, "score": 92.1},  # Missing name
        ]

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_dict("incomplete_table", incomplete_data)

            table = mock_ctx.report.tables["incomplete_table"]
            assert table.columns == (
                "name",
                "age",
                "score",
            )  # All unique keys in order of first appearance
            assert len(table.rows) == 3

            # Check that missing values are None (based on column order: name, age, score)
            assert table.rows[1][1] is None  # Bob's age (index 1)
            assert table.rows[2][0] is None  # Third person's name (index 0)

    def test_table_save_dict_duplicate_identifier(self):
        """Test table_save_dict with duplicate identifier raises error."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Save first table
            overity.api.table_save_dict("duplicate_dict_id", self.test_dict_data)

            # Try to save second table with same identifier
            with pytest.raises(
                ValueError,
                match="Table with identifier 'duplicate_dict_id' already exists",
            ):
                overity.api.table_save_dict("duplicate_dict_id", self.test_dict_data)

    def test_table_save_dict_single_dict(self):
        """Test table_save_dict with single dictionary."""
        single_dict = [{"name": "Alice", "age": 25}]

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_dict("single_dict_table", single_dict)

            table = mock_ctx.report.tables["single_dict_table"]
            assert table.columns == ("name", "age")
            assert len(table.rows) == 1
            assert table.rows[0] == ("Alice", 25)  # Original order (name, age)

    def test_table_save_df_uninitialized_api(self):
        """Test table_save_df raises UninitAPIError when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.table_save_df("test_table", self.test_df)

    def test_table_save_dict_uninitialized_api(self):
        """Test table_save_dict raises UninitAPIError when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.table_save_dict("test_table", self.test_dict_data)

    def test_table_save_df_large_dataframe(self):
        """Test table_save_df with large DataFrame."""
        # Create a larger DataFrame
        large_df = pd.DataFrame(
            {
                "id": range(1000),
                "value": np.random.randn(1000),
                "category": ["A", "B", "C"] * 333 + ["A"],
            }
        )

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_df("large_table", large_df)

            table = mock_ctx.report.tables["large_table"]
            assert table.identifier == "large_table"
            assert table.columns == ("id", "value", "category")
            assert len(table.rows) == 1000

    def test_table_save_dict_unicode_data(self):
        """Test table_save_dict with unicode characters."""
        unicode_data = [
            {"name": "José", "city": "São Paulo", "language": "Português"},
            {"name": "François", "city": "Paris", "language": "Français"},
            {"name": "北京", "city": "北京", "language": "中文"},
        ]

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_dict("unicode_table", unicode_data)

            table = mock_ctx.report.tables["unicode_table"]
            assert table.columns == ("name", "city", "language")
            assert len(table.rows) == 3
            # Verify unicode data is preserved (columns: name, city, language)
            assert table.rows[0][0] == "José"  # name column (index 0)
            assert table.rows[1][1] == "Paris"  # city column (index 1)
            assert table.rows[2][2] == "中文"  # language column (index 2)

    def test_table_save_df_mixed_types(self):
        """Test table_save_df with mixed data types."""
        mixed_df = pd.DataFrame(
            {
                "string_col": ["hello", "world", "test"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
                "none_col": [None, "value", None],
            }
        )

        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            overity.api.table_save_df("mixed_types_table", mixed_df)

            table = mock_ctx.report.tables["mixed_types_table"]
            assert table.columns == (
                "string_col",
                "int_col",
                "float_col",
                "bool_col",
                "none_col",
            )
            assert len(table.rows) == 3
            # Verify data types are preserved (columns: string_col, int_col, float_col, bool_col, none_col)
            assert table.rows[0][3] is True  # bool_col (index 3)
            assert table.rows[1][2] == 2.2  # float_col (index 2)
            assert table.rows[2][1] == 3  # int_col (index 1)
            assert table.rows[0][4] is None  # none_col (index 4)
            assert table.rows[1][0] == "world"  # string_col (index 0)

    def test_multiple_tables_same_report(self):
        """Test saving multiple tables to the same report."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = self.mock_report

            # Save multiple tables
            overity.api.table_save_df("df_table", self.test_df, "DataFrame table")
            overity.api.table_save_dict("dict_table", self.test_dict_data, "Dict table")

            # Verify both tables exist
            assert "df_table" in mock_ctx.report.tables
            assert "dict_table" in mock_ctx.report.tables

            # Verify their properties
            df_table = mock_ctx.report.tables["df_table"]
            dict_table = mock_ctx.report.tables["dict_table"]

            assert df_table.caption == "DataFrame table"
            assert dict_table.caption == "Dict table"

            # Both should have same data structure
            assert df_table.columns == dict_table.columns
            assert len(df_table.rows) == len(dict_table.rows)
