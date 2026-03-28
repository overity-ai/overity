"""
Unit tests for Table class from_df function
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from overity.model.report.table import Table


class TestTableFromDf:
    """Test the Table.from_df class method"""

    def test_from_df_basic_2d_dataframe(self):
        """Test from_df with basic 2D DataFrame."""
        # Create a simple 2D DataFrame
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [95.5, 87.2, 92.1],
            }
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="test_table", caption="Test scores")

        # Verify table properties
        assert table.identifier == "test_table"
        assert table.caption == "Test scores"
        assert table.columns == ("name", "age", "score")
        assert len(table.rows) == 3

        # Verify row data
        expected_rows = (("Alice", 25, 95.5), ("Bob", 30, 87.2), ("Charlie", 35, 92.1))
        assert table.rows == expected_rows

    def test_from_df_empty_dataframe(self):
        """Test from_df with empty DataFrame."""
        # Create empty DataFrame
        df = pd.DataFrame(columns=["col1", "col2", "col3"])

        # Create table from empty DataFrame
        table = Table.from_df(df, identifier="empty_table")

        # Verify table properties
        assert table.identifier == "empty_table"
        assert table.caption == ""
        assert table.columns == ("col1", "col2", "col3")
        assert len(table.rows) == 0
        assert table.rows == ()

    def test_from_df_with_nan_values(self):
        """Test from_df with NaN values in DataFrame."""
        # Create DataFrame with NaN values
        df = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10.5, np.nan, 20.3], "name": ["A", None, "C"]}
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="nan_table")

        # Verify NaN handling - should be converted to None
        expected_rows = ((1, 10.5, "A"), (2, None, None), (3, 20.3, "C"))
        assert table.rows == expected_rows

    def test_from_df_pandas_series(self):
        """Test from_df with pandas Series (1D data)."""
        # Create a Series with a name
        series = pd.Series([1, 2, 3, 4], name="numbers")

        # Create table from Series
        table = Table.from_df(
            series, identifier="series_table", caption="Number series"
        )

        # Verify table properties
        assert table.identifier == "series_table"
        assert table.caption == "Number series"
        assert table.columns == ("numbers",)
        assert len(table.rows) == 4

        # Verify row data
        expected_rows = ((1,), (2,), (3,), (4,))
        assert table.rows == expected_rows

    def test_from_df_anonymous_series(self):
        """Test from_df with anonymous pandas Series."""
        # Create a Series without a name
        series = pd.Series([10, 20, 30])

        # Create table from anonymous Series
        table = Table.from_df(series, identifier="anonymous_series")

        # Verify table properties - should default to "value" column name
        assert table.identifier == "anonymous_series"
        assert table.columns == ("value",)
        assert len(table.rows) == 3

        # Verify row data
        expected_rows = ((10,), (20,), (30,))
        assert table.rows == expected_rows

    def test_from_df_mixed_data_types(self):
        """Test from_df with mixed data types."""
        # Create DataFrame with mixed types
        df = pd.DataFrame(
            {
                "integer_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "string_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "datetime_col": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
            }
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="mixed_types")

        # Verify columns
        assert table.columns == (
            "integer_col",
            "float_col",
            "string_col",
            "bool_col",
            "datetime_col",
        )
        assert len(table.rows) == 3

        # Verify data types are preserved
        first_row = table.rows[0]
        assert isinstance(first_row[0], int)  # integer_col
        assert isinstance(first_row[1], float)  # float_col
        assert isinstance(first_row[2], str)  # string_col
        assert isinstance(first_row[3], bool)  # bool_col
        assert isinstance(first_row[4], pd.Timestamp)  # datetime_col

    def test_from_df_with_special_characters_in_column_names(self):
        """Test from_df with special characters in column names."""
        # Create DataFrame with special characters in column names
        df = pd.DataFrame(
            {
                "col with spaces": [1, 2],
                "col-with-dashes": [3, 4],
                "col_with_underscores": [5, 6],
                "123numeric_start": [7, 8],
                "UPPERCASE": [9, 10],
            }
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="special_cols")

        # Verify column names are preserved as strings
        expected_columns = (
            "col with spaces",
            "col-with-dashes",
            "col_with_underscores",
            "123numeric_start",
            "UPPERCASE",
        )
        assert table.columns == expected_columns

    def test_from_df_numeric_column_names(self):
        """Test from_df with numeric column names."""
        # Create DataFrame with numeric column names
        df = pd.DataFrame({1: [10, 20], 2.5: [30, 40], 0: [50, 60]})

        # Create table from DataFrame
        table = Table.from_df(df, identifier="numeric_cols")

        # Verify numeric column names are converted to strings
        # Note: str(1) gives '1', but str(1.0) gives '1.0', so we need to adjust our expectation
        expected_columns = ("1.0", "2.5", "0.0")
        assert table.columns == expected_columns

    def test_from_df_large_dataframe(self):
        """Test from_df with larger DataFrame."""
        # Create a larger DataFrame
        n_rows = 1000
        df = pd.DataFrame(
            {
                "id": range(n_rows),
                "value": np.random.randn(n_rows),
                "category": np.random.choice(["A", "B", "C"], n_rows),
            }
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="large_table")

        # Verify table properties
        assert table.identifier == "large_table"
        assert table.columns == ("id", "value", "category")
        assert len(table.rows) == n_rows

        # Verify first few rows
        assert table.rows[0] == (0, df.iloc[0]["value"], df.iloc[0]["category"])
        assert table.rows[1] == (1, df.iloc[1]["value"], df.iloc[1]["category"])

        # Verify last row
        assert table.rows[-1] == (
            n_rows - 1,
            df.iloc[-1]["value"],
            df.iloc[-1]["category"],
        )

    def test_from_df_with_index_column(self):
        """Test from_df with DataFrame that has a custom index."""
        # Create DataFrame with custom string index
        df = pd.DataFrame(
            {"value1": [10, 20, 30], "value2": [1.1, 2.2, 3.3]},
            index=["row_a", "row_b", "row_c"],
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="indexed_df")

        # Verify that index is ignored (itertuples with index=False)
        assert table.columns == ("value1", "value2")
        assert len(table.rows) == 3

        # Verify data (index should not be included)
        expected_rows = ((10, 1.1), (20, 2.2), (30, 3.3))
        assert table.rows == expected_rows

    def test_from_df_invalid_input_type(self):
        """Test from_df with invalid input type."""
        # Test with invalid input types
        invalid_inputs = ["not a dataframe", 123, [1, 2, 3], {"key": "value"}, None]

        for invalid_input in invalid_inputs:
            with pytest.raises(
                TypeError, match="df must be a pandas DataFrame or Series"
            ):
                Table.from_df(invalid_input, identifier="invalid")

    def test_from_df_empty_series(self):
        """Test from_df with empty pandas Series."""
        # Create empty Series
        empty_series = pd.Series([], dtype="float64", name="empty_col")

        # Create table from empty Series
        table = Table.from_df(empty_series, identifier="empty_series")

        # Verify table properties
        assert table.identifier == "empty_series"
        assert table.columns == ("empty_col",)
        assert len(table.rows) == 0
        assert table.rows == ()

    def test_from_df_series_with_nan_values(self):
        """Test from_df with Series containing NaN values."""
        # Create Series with NaN values
        series = pd.Series([1.0, np.nan, 3.0, np.nan], name="series_with_nan")

        # Create table from Series
        table = Table.from_df(series, identifier="series_nan")

        # Verify NaN handling
        expected_rows = ((1.0,), (None,), (3.0,), (None,))
        assert table.rows == expected_rows

    def test_from_df_default_caption(self):
        """Test from_df with default empty caption."""
        # Create simple DataFrame
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        # Create table without specifying caption (should default to empty string)
        table = Table.from_df(df, identifier="no_caption")

        # Verify caption defaults to empty string
        assert table.caption == ""

    def test_from_df_preserves_original_dataframe(self):
        """Test that from_df doesn't modify the original DataFrame."""
        # Create original DataFrame
        original_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        # Store original values for comparison
        original_shape = original_df.shape
        original_columns = list(original_df.columns)
        original_values = original_df.copy()

        # Create table from DataFrame
        table = Table.from_df(original_df, identifier="preserved")

        # Verify original DataFrame is unchanged
        assert original_df.shape == original_shape
        assert list(original_df.columns) == original_columns
        pd.testing.assert_frame_equal(original_df, original_values)

        # Verify table was created correctly
        assert table.columns == ("a", "b")
        assert len(table.rows) == 3

    def test_from_df_tuple_data_types(self):
        """Test from_df with tuple data types."""
        # Create DataFrame with tuple types (which should work fine)
        df = pd.DataFrame(
            {"tuple_col": [(1, 2), (3, 4), (5, 6)], "string_col": ["a", "b", "c"]}
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="tuple_types")

        # Verify tuple types are preserved
        expected_rows = (((1, 2), "a"), ((3, 4), "b"), ((5, 6), "c"))
        assert table.rows == expected_rows

    def test_from_df_limitation_with_complex_objects(self):
        """Test that from_df has limitations with complex objects like lists and dicts."""
        # This test documents a current limitation of the implementation
        # The pd.isna() function doesn't work well with complex objects

        # Create DataFrame with list objects (this would fail with current implementation)
        df_with_lists = pd.DataFrame(
            {
                "list_col": [[1, 2, 3], [4, 5], [6]],
            }
        )

        # This would raise a ValueError due to pd.isna() not handling lists well
        with pytest.raises(
            ValueError,
            match="The truth value of an array with more than one element is ambiguous",
        ):
            Table.from_df(df_with_lists, identifier="list_test")

    def test_from_df_unicode_strings(self):
        """Test from_df with unicode strings."""
        # Create DataFrame with unicode characters
        df = pd.DataFrame(
            {
                "emoji_col": ["😀", "😎", "🚀"],
                "chinese_col": ["你好", "世界", "测试"],
                "arabic_col": ["مرحبا", "العالم", "اختبار"],
            }
        )

        # Create table from DataFrame
        table = Table.from_df(df, identifier="unicode_test")

        # Verify unicode strings are preserved correctly
        expected_rows = (
            ("😀", "你好", "مرحبا"),
            ("😎", "世界", "العالم"),
            ("🚀", "测试", "اختبار"),
        )
        assert table.rows == expected_rows
