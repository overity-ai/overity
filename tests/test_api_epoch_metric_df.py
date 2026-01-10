"""
Unit tests for API epoch_metric_df functionality
"""

import pytest
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from overity.api import epoch_metric_df
from overity.errors import UninitAPIError
from overity.backend.flow.ctx import FlowCtx
from overity.model.report import MethodReport, MethodExecutionStage, MethodExecutionStatus
from overity.model.report.metrics import SimpleValue, LinScaleValue, PercentageValue
from overity.model.general_info.method import MethodInfo, MethodKind, MethodAuthor
from overity.model.traceability import ArtifactGraph


class TestApiEpochMetricDf:
    """Test the epoch_metric_df function in overity.api module."""

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

        # Create sample epoch metrics
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
        )

    @patch('overity.api._CTX')
    def test_epoch_metric_df_basic_functionality(self, mock_ctx):
        """Test basic functionality of API epoch_metric_df."""
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Test SimpleValue metric
        result = epoch_metric_df("accuracy")
        
        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "epoch"
        assert list(result.columns) == ["value"]
        
        # Verify data
        assert len(result) == 3
        assert result.loc[1, "value"] == 0.8
        assert result.loc[2, "value"] == 0.85
        assert result.loc[3, "value"] == 0.9

    @patch('overity.api._CTX')
    def test_epoch_metric_df_lin_scale_value(self, mock_ctx):
        """Test API epoch_metric_df with LinScaleValue metrics."""
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Test learning_rate metric
        result = epoch_metric_df("learning_rate")
        
        # Verify DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "epoch"
        assert list(result.columns) == ["low", "high", "value"]
        
        # Verify data
        assert len(result) == 3
        assert result.loc[1, "low"] == 0.001
        assert result.loc[1, "high"] == 0.1
        assert result.loc[1, "value"] == 0.01

    @patch('overity.api._CTX')
    def test_epoch_metric_df_percentage_value(self, mock_ctx):
        """Test API epoch_metric_df with PercentageValue metrics."""
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Test progress metric
        result = epoch_metric_df("progress")
        
        # Verify DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "epoch"
        assert list(result.columns) == ["value"]
        
        # Verify data
        assert len(result) == 3
        assert result.loc[1, "value"] == 0.1
        assert result.loc[2, "value"] == 0.2
        assert result.loc[3, "value"] == 0.3

    @patch('overity.api._CTX')
    def test_epoch_metric_df_missing_metric_key(self, mock_ctx):
        """Test API epoch_metric_df with missing metric key."""
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Try to get a metric that doesn't exist
        with pytest.raises(KeyError):
            epoch_metric_df("nonexistent_metric")

    @patch('overity.api._CTX')
    def test_epoch_metric_df_empty_epoch_metrics(self, mock_ctx):
        """Test API epoch_metric_df with empty epoch_metrics."""
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
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        result = epoch_metric_df("accuracy")
        
        # Verify None result
        assert result is None

    @patch('overity.api._CTX')
    def test_epoch_metric_df_none_epoch_metrics(self, mock_ctx):
        """Test API epoch_metric_df with None epoch_metrics."""
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
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        # This should raise an error since epoch_metrics is None
        with pytest.raises((AttributeError, TypeError)):
            epoch_metric_df("accuracy")

    @patch('overity.api._CTX')
    def test_epoch_metric_df_uninitialized_context(self, mock_ctx):
        """Test API epoch_metric_df with uninitialized context."""
        # Set up mock context with init_ok = False
        mock_ctx.init_ok = False
        
        # The @_api_guard decorator should prevent execution
        with pytest.raises(UninitAPIError):
            epoch_metric_df("accuracy")

    @patch('overity.api._CTX')
    def test_epoch_metric_df_single_epoch(self, mock_ctx):
        """Test API epoch_metric_df with single epoch."""
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
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        result = epoch_metric_df("accuracy")
        
        # Verify single row DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.index.name == "epoch"
        assert result.loc[1, "value"] == 0.95

    @patch('overity.api._CTX')
    def test_epoch_metric_df_non_sequential_epochs(self, mock_ctx):
        """Test API epoch_metric_df with non-sequential epoch numbers."""
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
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        result = epoch_metric_df("accuracy")
        
        # Verify DataFrame with non-sequential indices
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.index) == [1, 5, 10]
        assert result.loc[1, "value"] == 0.8
        assert result.loc[5, "value"] == 0.85
        assert result.loc[10, "value"] == 0.9

    @patch('overity.api._CTX')
    def test_epoch_metric_df_zero_epoch(self, mock_ctx):
        """Test API epoch_metric_df with epoch 0."""
        # Create report with epoch 0
        zero_epoch_metrics = {
            0: {"accuracy": SimpleValue(0.7)},
            1: {"accuracy": SimpleValue(0.8)},
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
            epoch_metrics=zero_epoch_metrics,
            outputs=None,
        )
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        result = epoch_metric_df("accuracy")
        
        # Verify DataFrame includes epoch 0
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 0 in result.index
        assert result.loc[0, "value"] == 0.7
        assert result.loc[1, "value"] == 0.8

    @patch('overity.api._CTX')
    def test_epoch_metric_df_large_epoch_numbers(self, mock_ctx):
        """Test API epoch_metric_df with large epoch numbers."""
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
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = report
        
        result = epoch_metric_df("accuracy")
        
        # Verify DataFrame with large epoch numbers
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.index) == [1000, 2000, 5000]
        assert result.loc[1000, "value"] == 0.95
        assert result.loc[2000, "value"] == 0.96
        assert result.loc[5000, "value"] == 0.97

    @patch('overity.api._CTX')
    def test_epoch_metric_df_delegates_to_backend_flow(self, mock_ctx):
        """Test that API epoch_metric_df delegates to backend flow epoch_metric_df."""
        # Mock the backend flow epoch_metric_df function
        mock_df = pd.DataFrame({"value": [0.8, 0.85, 0.9]}, index=[1, 2, 3])
        mock_df.index.name = "epoch"
        
        with patch('overity.backend.flow.epoch_metric_df', return_value=mock_df) as mock_backend:
            result = epoch_metric_df("accuracy")
            
            # Verify the backend function was called with correct arguments
            mock_backend.assert_called_once_with(mock_ctx, "accuracy")
            
            # Verify the result is the mocked DataFrame
            pd.testing.assert_frame_equal(result, mock_df)

    @patch('overity.api._CTX')
    def test_epoch_metric_df_context_manager_integration(self, mock_ctx):
        """Test API epoch_metric_df integration with context manager usage."""
        # This test simulates the typical usage pattern where metrics are added
        # via context manager and then retrieved via epoch_metric_df
        
        # Set up mock context
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Test that we can retrieve metrics that were added through the API
        result = epoch_metric_df("accuracy")
        
        # Verify the data matches what we expect
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result.loc[1, "value"] == 0.8
        assert result.loc[2, "value"] == 0.85
        assert result.loc[3, "value"] == 0.9

    def test_epoch_metric_df_without_mock_context(self):
        """Test API epoch_metric_df by mocking the global context directly."""
        # Create a mock for the global _CTX
        mock_ctx = Mock(spec=FlowCtx)
        mock_ctx.init_ok = True
        mock_ctx.report = self.report
        
        # Patch the global _CTX variable
        with patch('overity.api._CTX', mock_ctx):
            result = epoch_metric_df("accuracy")
            
            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert result.index.name == "epoch"
            assert len(result) == 3
            assert result.loc[1, "value"] == 0.8
