"""
Unit tests for MethodReport epoch_metric_df functionality
"""

import pytest
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from overity.model.report import (
    MethodReport,
    MethodExecutionStage,
    MethodExecutionStatus,
)
from overity.model.report.metrics import (
    SimpleValue,
    LinScaleValue,
    LinRangeValue,
    PercentageValue,
)
from overity.model.general_info.method import MethodInfo, MethodKind, MethodAuthor
from overity.model.traceability import ArtifactGraph

# Import plotly for graph testing
try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

    # Create a dummy Figure class for type hints
    class Figure:
        pass


class TestMethodReportEpochMetricDf:
    """Test the epoch_metric_df method of MethodReport class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a basic method info
        self.method_info = MethodInfo(
            slug="test-method",
            kind=MethodKind.TrainingOptimization,
            display_name="Test Method",
            authors=[MethodAuthor(name="Test Author", email="test@example.com")],
            metadata={},
            description="A test method",
            path=Path("/path/to/method"),
        )

        # Create sample epoch metrics with different metric types
        self.epoch_metrics = {
            1: {
                "accuracy": SimpleValue(0.8),
                "loss": SimpleValue(0.2),
                "learning_rate": LinScaleValue(low=0.001, high=0.1, value=0.01),
                "progress": PercentageValue(0.1),
            },
            2: {
                "accuracy": SimpleValue(0.85),
                "loss": SimpleValue(0.15),
                "learning_rate": LinScaleValue(low=0.001, high=0.1, value=0.009),
                "progress": PercentageValue(0.2),
            },
            3: {
                "accuracy": SimpleValue(0.9),
                "loss": SimpleValue(0.1),
                "learning_rate": LinScaleValue(low=0.001, high=0.1, value=0.008),
                "progress": PercentageValue(0.3),
            },
        }

        # Create a complete MethodReport
        self.report = MethodReport(
            uuid="test-uuid",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={"python": "3.9"},
            context={"batch_size": "32"},
            method_info=self.method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics=self.epoch_metrics,
            outputs=None,
            graphs={} if PLOTLY_AVAILABLE else None,
        )

    def test_epoch_metric_df_simple_value(self):
        """Test epoch_metric_df with SimpleValue metrics."""
        # Test accuracy metric
        df = self.report.epoch_metric_df("accuracy")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "epoch"
        assert list(df.columns) == ["value"]

        # Verify data
        assert len(df) == 3
        assert df.loc[1, "value"] == 0.8
        assert df.loc[2, "value"] == 0.85
        assert df.loc[3, "value"] == 0.9

    def test_epoch_metric_df_lin_scale_value(self):
        """Test epoch_metric_df with LinScaleValue metrics."""
        # Test learning_rate metric
        df = self.report.epoch_metric_df("learning_rate")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "epoch"
        assert list(df.columns) == ["low", "high", "value"]

        # Verify data
        assert len(df) == 3
        assert df.loc[1, "low"] == 0.001
        assert df.loc[1, "high"] == 0.1
        assert df.loc[1, "value"] == 0.01

        assert df.loc[2, "low"] == 0.001
        assert df.loc[2, "high"] == 0.1
        assert df.loc[2, "value"] == 0.009

        assert df.loc[3, "low"] == 0.001
        assert df.loc[3, "high"] == 0.1
        assert df.loc[3, "value"] == 0.008

    def test_epoch_metric_df_percentage_value(self):
        """Test epoch_metric_df with PercentageValue metrics."""
        # Test progress metric
        df = self.report.epoch_metric_df("progress")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "epoch"
        assert list(df.columns) == ["value"]

        # Verify data
        assert len(df) == 3
        assert df.loc[1, "value"] == 0.1
        assert df.loc[2, "value"] == 0.2
        assert df.loc[3, "value"] == 0.3

    def test_epoch_metric_df_loss_metric(self):
        """Test epoch_metric_df with loss metrics."""
        # Test loss metric
        df = self.report.epoch_metric_df("loss")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "epoch"
        assert list(df.columns) == ["value"]

        # Verify data
        assert len(df) == 3
        assert df.loc[1, "value"] == 0.2
        assert df.loc[2, "value"] == 0.15
        assert df.loc[3, "value"] == 0.1

    def test_epoch_metric_df_empty_epoch_metrics(self):
        """Test epoch_metric_df with empty epoch_metrics."""
        # Create report with empty epoch_metrics
        report = MethodReport(
            uuid="test-uuid",
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
        )

        df = report.epoch_metric_df("accuracy")

        # Verify empty DataFrame
        assert df is None

    def test_epoch_metric_df_none_epoch_metrics(self):
        """Test epoch_metric_df with None epoch_metrics."""
        # Create report with None epoch_metrics
        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=None,
            outputs=None,
        )

        # This should raise an error since epoch_metrics is None
        with pytest.raises((AttributeError, TypeError)):
            report.epoch_metric_df("accuracy")

    def test_epoch_metric_df_missing_metric_key(self):
        """Test epoch_metric_df with missing metric key."""
        # Try to get a metric that doesn't exist
        with pytest.raises(KeyError):
            self.report.epoch_metric_df("nonexistent_metric")

    def test_epoch_metric_df_single_epoch(self):
        """Test epoch_metric_df with only one epoch."""
        # Create report with single epoch
        single_epoch_metrics = {
            1: {
                "accuracy": SimpleValue(0.95),
                "loss": SimpleValue(0.05),
            }
        }

        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=single_epoch_metrics,
            outputs=None,
        )

        df = report.epoch_metric_df("accuracy")

        # Verify single row DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.index.name == "epoch"
        assert df.loc[1, "value"] == 0.95

    def test_epoch_metric_df_non_sequential_epochs(self):
        """Test epoch_metric_df with non-sequential epoch numbers."""
        # Create report with non-sequential epochs
        non_sequential_metrics = {
            1: {"accuracy": SimpleValue(0.8)},
            5: {"accuracy": SimpleValue(0.85)},
            10: {"accuracy": SimpleValue(0.9)},
        }

        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=non_sequential_metrics,
            outputs=None,
        )

        df = report.epoch_metric_df("accuracy")

        # Verify DataFrame with non-sequential indices
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.index) == [1, 5, 10]
        assert df.loc[1, "value"] == 0.8
        assert df.loc[5, "value"] == 0.85
        assert df.loc[10, "value"] == 0.9

    def test_epoch_metric_df_mixed_metric_types_across_epochs(self):
        """Test epoch_metric_df when different epochs have different metric structures."""
        # This should not happen in practice, but let's test the behavior
        mixed_metrics = {
            1: {"accuracy": SimpleValue(0.8)},
            2: {"accuracy": LinScaleValue(low=0.0, high=1.0, value=0.85)},
        }

        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=mixed_metrics,
            outputs=None,
        )

        # This will create a DataFrame with mixed column structures
        # which may not be ideal but should not crash
        df = report.epoch_metric_df("accuracy")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # The exact structure depends on how pandas handles missing columns
        assert df.index.name == "epoch"

    def test_epoch_metric_df_large_epoch_numbers(self):
        """Test epoch_metric_df with large epoch numbers."""
        # Create report with large epoch numbers
        large_epoch_metrics = {
            1000: {"accuracy": SimpleValue(0.95)},
            2000: {"accuracy": SimpleValue(0.96)},
            5000: {"accuracy": SimpleValue(0.97)},
        }

        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=large_epoch_metrics,
            outputs=None,
        )

        df = report.epoch_metric_df("accuracy")

        # Verify DataFrame with large epoch numbers
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.index) == [1000, 2000, 5000]
        assert df.loc[1000, "value"] == 0.95
        assert df.loc[2000, "value"] == 0.96
        assert df.loc[5000, "value"] == 0.97

    def test_epoch_metric_df_lin_range_value(self):
        """Test epoch_metric_df with LinRangeValue metrics."""
        # Create metrics with LinRangeValue
        lin_range_metrics = {
            1: {"iterations": LinRangeValue(low=0, high=100, value=25)},
            2: {"iterations": LinRangeValue(low=0, high=100, value=50)},
            3: {"iterations": LinRangeValue(low=0, high=100, value=75)},
        }

        report = MethodReport(
            uuid="test-uuid",
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
            epoch_metrics=lin_range_metrics,
            outputs=None,
        )

        df = report.epoch_metric_df("iterations")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "epoch"
        assert list(df.columns) == ["low", "high", "value"]

        # Verify data
        assert len(df) == 3
        assert df.loc[1, "low"] == 0
        assert df.loc[1, "high"] == 100
        assert df.loc[1, "value"] == 25

        assert df.loc[2, "low"] == 0
        assert df.loc[2, "high"] == 100
        assert df.loc[2, "value"] == 50

        assert df.loc[3, "low"] == 0
        assert df.loc[3, "high"] == 100
        assert df.loc[3, "value"] == 75

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
    def test_method_report_with_graphs(self):
        """Test that MethodReport can handle graphs field correctly."""
        # Create sample plotly figures
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=["A", "B", "C"], y=[10, 20, 30]))

        # Create report with graphs
        report_with_graphs = MethodReport(
            uuid="test-graphs-uuid",
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
            graphs={"accuracy_plot": fig1, "loss_plot": fig2},
        )

        # Verify graphs are stored correctly
        assert report_with_graphs.graphs is not None
        assert len(report_with_graphs.graphs) == 2
        assert "accuracy_plot" in report_with_graphs.graphs
        assert "loss_plot" in report_with_graphs.graphs

        # Verify the figures are correct
        assert isinstance(report_with_graphs.graphs["accuracy_plot"], go.Figure)
        assert isinstance(report_with_graphs.graphs["loss_plot"], go.Figure)

        # Verify figure data
        acc_fig = report_with_graphs.graphs["accuracy_plot"]
        assert acc_fig.data[0].type == "scatter"
        assert list(acc_fig.data[0].x) == [1, 2, 3]
        assert list(acc_fig.data[0].y) == [4, 5, 6]

        loss_fig = report_with_graphs.graphs["loss_plot"]
        assert loss_fig.data[0].type == "bar"
        assert list(loss_fig.data[0].x) == ["A", "B", "C"]
        assert list(loss_fig.data[0].y) == [10, 20, 30]

    def test_method_report_empty_graphs(self):
        """Test that MethodReport can handle empty graphs dictionary."""
        report_empty_graphs = MethodReport(
            uuid="test-empty-graphs-uuid",
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
            graphs={},
        )

        # Verify empty graphs dictionary
        assert report_empty_graphs.graphs is not None
        assert len(report_empty_graphs.graphs) == 0

    def test_method_report_none_graphs(self):
        """Test that MethodReport can handle None graphs field."""
        report_none_graphs = MethodReport(
            uuid="test-none-graphs-uuid",
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
            graphs=None,
        )

        # Verify None graphs field
        assert report_none_graphs.graphs is None

    def test_method_report_default_graphs(self):
        """Test that MethodReport default() method includes empty graphs."""
        # Create a minimal report using the default() method
        report_default = MethodReport.default(
            uuid="test-default-uuid",
            program="test-default-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            method_info=self.method_info,
        )

        # Verify that default() method initializes graphs as empty dict
        assert report_default.graphs is not None
        assert isinstance(report_default.graphs, dict)
        assert len(report_default.graphs) == 0
