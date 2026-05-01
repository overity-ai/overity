"""
Unit tests for MethodReport tables functionality
"""

import pytest
from datetime import datetime as dt
from pathlib import Path

from overity.model.report import (
    MethodReport,
    MethodExecutionStage,
    MethodExecutionStatus,
)
from overity.model.report.table import Table

from overity.model.general_info.method import MethodInfo, MethodKind, MethodAuthor
from overity.model.traceability import ArtifactGraph

# Import plotly for graph testing
try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestMethodReportTables:
    """Test class for MethodReport tables functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.method_info = MethodInfo(
            slug="test-method",
            kind=MethodKind.TrainingOptimization,
            display_name="Test Method",
            authors=[MethodAuthor(name="Test Author", email="test@example.com")],
            metadata={},
            description="A test method",
            path=Path("/path/to/method"),
        )

    def test_method_report_with_tables(self):
        """Test that MethodReport can handle tables field correctly."""
        # Create sample tables
        table1 = Table(
            identifier="results_table",
            caption="Training Results",
            columns=("epoch", "accuracy", "loss"),
            rows=((1, 0.85, 0.45), (2, 0.90, 0.30), (3, 0.92, 0.25)),
        )

        table2 = Table(
            identifier="config_table",
            caption="Configuration",
            columns=("parameter", "value"),
            rows=(("learning_rate", 0.001), ("batch_size", 32), ("epochs", 100)),
        )

        # Create report with tables
        report_with_tables = MethodReport(
            uuid="test-tables-uuid",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            method_info=self.method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics={},
            outputs=None,
            tables={"results": table1, "config": table2},
        )

        # Verify tables are stored correctly
        assert report_with_tables.tables is not None
        assert len(report_with_tables.tables) == 2
        assert "results" in report_with_tables.tables
        assert "config" in report_with_tables.tables

        # Verify the tables are correct
        assert isinstance(report_with_tables.tables["results"], Table)
        assert isinstance(report_with_tables.tables["config"], Table)

        # Verify table data
        results_table = report_with_tables.tables["results"]
        assert results_table.identifier == "results_table"
        assert results_table.caption == "Training Results"
        assert results_table.columns == ("epoch", "accuracy", "loss")
        assert results_table.rows == ((1, 0.85, 0.45), (2, 0.90, 0.30), (3, 0.92, 0.25))

        config_table = report_with_tables.tables["config"]
        assert config_table.identifier == "config_table"
        assert config_table.caption == "Configuration"
        assert config_table.columns == ("parameter", "value")
        assert config_table.rows == (
            ("learning_rate", 0.001),
            ("batch_size", 32),
            ("epochs", 100),
        )

    def test_method_report_empty_tables(self):
        """Test that MethodReport can handle empty tables dictionary."""
        report_empty_tables = MethodReport(
            uuid="test-empty-tables-uuid",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            method_info=self.method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics={},
            outputs=None,
            tables={},
        )

        # Verify empty tables dictionary
        assert report_empty_tables.tables is not None
        assert len(report_empty_tables.tables) == 0

    def test_method_report_none_tables(self):
        """Test that MethodReport can handle None tables field."""
        report_none_tables = MethodReport(
            uuid="test-none-tables-uuid",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            method_info=self.method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics={},
            outputs=None,
            tables=None,
        )

        # Verify None tables field
        assert report_none_tables.tables is None

    def test_method_report_default_tables(self):
        """Test that MethodReport default() method includes empty tables."""
        # Create a minimal report using the default() method
        report_default = MethodReport.default(
            uuid="test-default-uuid",
            program="test-default-program",
            stage=MethodExecutionStage.Preview,
            method_info=self.method_info,
            date_started=dt(2023, 1, 1, 10, 0, 0),
        )

        # Verify that default() method initializes tables as empty dict
        assert report_default.tables is not None
        assert isinstance(report_default.tables, dict)
        assert len(report_default.tables) == 0
