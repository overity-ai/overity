"""
Unit tests for table backend flow functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from overity.backend.flow import table_save, table_save_df, table_save_dict
from overity.model.report.table import Table
from overity.errors import UninitAPIError


class TestBackendFlowTables:
    """Test cases for table backend flow functions"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock context
        self.mock_ctx = Mock()
        self.mock_ctx.init_ok = True
        self.mock_ctx.report = Mock()
        self.mock_ctx.report.tables = {}

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

    def test_table_save_basic(self):
        """Test basic functionality of table_save backend function."""
        # Create a table object
        table = Table(
            identifier="test_table",
            caption="Test table",
            columns=("name", "age", "score"),
            rows=(("Alice", 25, 95.5), ("Bob", 30, 87.2)),
        )

        # Call the backend function
        table_save(self.mock_ctx, table)

        # Verify the table was saved
        assert "test_table" in self.mock_ctx.report.tables
        saved_table = self.mock_ctx.report.tables["test_table"]
        assert saved_table.identifier == "test_table"
        assert saved_table.caption == "Test table"
        assert saved_table.columns == ("name", "age", "score")
        assert len(saved_table.rows) == 2

    def test_table_save_duplicate_identifier(self):
        """Test table_save with duplicate identifier raises error."""
        # Create two tables with same identifier
        table1 = Table(
            identifier="duplicate_id",
            caption="First table",
            columns=("col1",),
            rows=(("value1",),),
        )
        table2 = Table(
            identifier="duplicate_id",
            caption="Second table",
            columns=("col2",),
            rows=(("value2",),),
        )

        # Save first table
        table_save(self.mock_ctx, table1)

        # Try to save second table with same identifier
        with pytest.raises(
            ValueError, match="Table with identifier 'duplicate_id' already exists"
        ):
            table_save(self.mock_ctx, table2)

    def test_table_save_uninitialized_api(self):
        """Test table_save raises UninitAPIError when API not initialized."""
        self.mock_ctx.init_ok = False

        table = Table(
            identifier="test_table",
            caption="Test table",
            columns=("col1",),
            rows=(("value1",),),
        )

        with pytest.raises(UninitAPIError):
            table_save(self.mock_ctx, table)

    def test_table_save_df_basic(self):
        """Test basic functionality of table_save_df backend function."""
        table_save_df(
            self.mock_ctx, "test_df_table", self.test_df, "Test DataFrame table"
        )

        # Verify the table was saved
        assert "test_df_table" in self.mock_ctx.report.tables
        table = self.mock_ctx.report.tables["test_df_table"]
        assert table.identifier == "test_df_table"
        assert table.caption == "Test DataFrame table"
        assert table.columns == ("name", "age", "score")
        assert len(table.rows) == 3

    def test_table_save_df_no_caption(self):
        """Test table_save_df without caption."""
        table_save_df(self.mock_ctx, "test_df_table", self.test_df)

        table = self.mock_ctx.report.tables["test_df_table"]
        assert table.caption == ""

    def test_table_save_df_empty_dataframe(self):
        """Test table_save_df with empty DataFrame."""
        empty_df = pd.DataFrame()

        table_save_df(self.mock_ctx, "empty_df_table", empty_df, "Empty DataFrame")

        table = self.mock_ctx.report.tables["empty_df_table"]
        assert table.identifier == "empty_df_table"
        assert table.caption == "Empty DataFrame"
        assert table.columns == ()
        assert table.rows == ()

    def test_table_save_df_with_nan_values(self):
        """Test table_save_df with NaN values."""
        df_with_nan = pd.DataFrame(
            {"name": ["Alice", "Bob", None], "score": [95.5, np.nan, 92.1]}
        )

        table_save_df(self.mock_ctx, "nan_df_table", df_with_nan)

        table = self.mock_ctx.report.tables["nan_df_table"]
        assert table.columns == ("name", "score")
        assert len(table.rows) == 3
        # Check that NaN values are converted to None
        assert table.rows[1][1] is None  # Bob's score (NaN)
        assert table.rows[2][0] is None  # Third name (None)

    def test_table_save_df_duplicate_identifier(self):
        """Test table_save_df with duplicate identifier raises error."""
        # Save first table
        table_save_df(self.mock_ctx, "duplicate_id", self.test_df)

        # Try to save second table with same identifier
        with pytest.raises(
            ValueError, match="Table with identifier 'duplicate_id' already exists"
        ):
            table_save_df(self.mock_ctx, "duplicate_id", self.test_df)

    def test_table_save_df_uninitialized_api(self):
        """Test table_save_df raises UninitAPIError when API not initialized."""
        self.mock_ctx.init_ok = False

        with pytest.raises(UninitAPIError):
            table_save_df(self.mock_ctx, "test_table", self.test_df)

    def test_table_save_dict_basic(self):
        """Test basic functionality of table_save_dict backend function."""
        table_save_dict(
            self.mock_ctx, "test_dict_table", self.test_dict_data, "Test dict table"
        )

        # Verify the table was saved
        assert "test_dict_table" in self.mock_ctx.report.tables
        table = self.mock_ctx.report.tables["test_dict_table"]
        assert table.identifier == "test_dict_table"
        assert table.caption == "Test dict table"
        assert table.columns == ("name", "age", "score")  # Preserves original order
        assert len(table.rows) == 3

    def test_table_save_dict_no_caption(self):
        """Test table_save_dict without caption."""
        table_save_dict(self.mock_ctx, "test_dict_table", self.test_dict_data)

        table = self.mock_ctx.report.tables["test_dict_table"]
        assert table.caption == ""

    def test_table_save_dict_empty_list(self):
        """Test table_save_dict with empty list."""
        table_save_dict(self.mock_ctx, "empty_dict_table", [], "Empty dict table")

        table = self.mock_ctx.report.tables["empty_dict_table"]
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

        table_save_dict(self.mock_ctx, "incomplete_table", incomplete_data)

        table = self.mock_ctx.report.tables["incomplete_table"]
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
        # Save first table
        table_save_dict(self.mock_ctx, "duplicate_id", self.test_dict_data)

        # Try to save second table with same identifier
        with pytest.raises(
            ValueError, match="Table with identifier 'duplicate_id' already exists"
        ):
            table_save_dict(self.mock_ctx, "duplicate_id", self.test_dict_data)

    def test_table_save_dict_uninitialized_api(self):
        """Test table_save_dict raises UninitAPIError when API not initialized."""
        self.mock_ctx.init_ok = False

        with pytest.raises(UninitAPIError):
            table_save_dict(self.mock_ctx, "test_table", self.test_dict_data)

    def test_table_save_dict_single_dict(self):
        """Test table_save_dict with single dictionary."""
        single_dict = [{"name": "Alice", "age": 25}]

        table_save_dict(self.mock_ctx, "single_dict_table", single_dict)

        table = self.mock_ctx.report.tables["single_dict_table"]
        assert table.columns == ("name", "age")
        assert len(table.rows) == 1
        assert table.rows[0] == ("Alice", 25)  # Original order (name, age)

    def test_table_save_df_series_input(self):
        """Test table_save_df with pandas Series input."""
        test_series = pd.Series([1, 2, 3, 4, 5], name="values")

        table_save_df(self.mock_ctx, "series_table", test_series, "Test series")

        table = self.mock_ctx.report.tables["series_table"]
        assert table.identifier == "series_table"
        assert table.caption == "Test series"
        assert table.columns == ("values",)
        assert len(table.rows) == 5
        assert table.rows[0] == (1,)
        assert table.rows[4] == (5,)

    def test_table_save_df_anonymous_series(self):
        """Test table_save_df with anonymous pandas Series."""
        test_series = pd.Series([1, 2, 3])  # No name

        table_save_df(self.mock_ctx, "anonymous_series_table", test_series)

        table = self.mock_ctx.report.tables["anonymous_series_table"]
        assert table.columns == ("value",)  # Should default to "value"
        assert len(table.rows) == 3

    def test_multiple_tables_same_context(self):
        """Test saving multiple tables to the same context."""
        # Save multiple tables
        table_save_df(self.mock_ctx, "df_table", self.test_df, "DataFrame table")
        table_save_dict(self.mock_ctx, "dict_table", self.test_dict_data, "Dict table")

        # Verify both tables exist
        assert "df_table" in self.mock_ctx.report.tables
        assert "dict_table" in self.mock_ctx.report.tables

        # Verify their properties
        df_table = self.mock_ctx.report.tables["df_table"]
        dict_table = self.mock_ctx.report.tables["dict_table"]

        assert df_table.caption == "DataFrame table"
        assert dict_table.caption == "Dict table"

        # Both should have same data structure
        assert df_table.columns == dict_table.columns
        assert len(df_table.rows) == len(dict_table.rows)
