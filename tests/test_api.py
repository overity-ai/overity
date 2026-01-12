"""
Unit tests for overity.api module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

import overity.api
from overity.backend.flow import MetricSaver
from overity.errors import UninitAPIError, InvalidEpochValue


class TestEpochMetrics:
    """Test cases for the epoch_metrics function."""

    def test_epoch_metrics_returns_metric_saver(self):
        """Test that epoch_metrics returns a MetricSaver instance."""
        # Mock the flow.epoch_metrics function
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Create a mock MetricSaver instance
            mock_metric_saver = Mock(spec=MetricSaver)
            mock_flow_epoch_metrics.return_value = mock_metric_saver

            # Call the function
            result = overity.api.epoch_metrics(1)

            # Verify the result
            assert result == mock_metric_saver
            mock_flow_epoch_metrics.assert_called_once()

    def test_epoch_metrics_calls_flow_with_correct_args(self):
        """Test that epoch_metrics calls flow.epoch_metrics with correct arguments."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            mock_metric_saver = Mock(spec=MetricSaver)
            mock_flow_epoch_metrics.return_value = mock_metric_saver

            # Call with different epoch numbers
            overity.api.epoch_metrics(5)
            overity.api.epoch_metrics(10)
            overity.api.epoch_metrics(0)

            # Verify calls
            assert mock_flow_epoch_metrics.call_count == 3
            mock_flow_epoch_metrics.assert_any_call(
                overity.api._CTX, 5, display_output=True
            )
            mock_flow_epoch_metrics.assert_any_call(
                overity.api._CTX, 10, display_output=True
            )
            mock_flow_epoch_metrics.assert_any_call(
                overity.api._CTX, 0, display_output=True
            )

    def test_epoch_metrics_with_context_manager_usage(self):
        """Test epoch_metrics used as a context manager."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Create a real MetricSaver instance for testing context manager
            mock_output_dict = {}
            metric_saver = MetricSaver("Epoch 1 metrics", mock_output_dict)
            mock_flow_epoch_metrics.return_value = metric_saver

            # Use as context manager
            with overity.api.epoch_metrics(1) as metrics:
                # Verify we can call metric methods
                metrics.simple("accuracy", 0.95)
                metrics.percentage("validation_accuracy", 0.92)
                metrics.scale_lin("loss", 0.05, 0.0, 1.0)
                metrics.range_lin("epoch", 1, 1, 100)

            # Verify metrics were recorded
            assert "accuracy" in mock_output_dict
            assert "validation_accuracy" in mock_output_dict
            assert "loss" in mock_output_dict
            assert "epoch" in mock_output_dict

    def test_epoch_metrics_multiple_epochs(self):
        """Test using epoch_metrics with multiple different epochs."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Create different output dicts for different epochs
            epoch1_dict = {}
            epoch5_dict = {}
            epoch10_dict = {}

            epoch1_saver = MetricSaver("Epoch 1 metrics", epoch1_dict)
            epoch5_saver = MetricSaver("Epoch 5 metrics", epoch5_dict)
            epoch10_saver = MetricSaver("Epoch 10 metrics", epoch10_dict)

            mock_flow_epoch_metrics.side_effect = [
                epoch1_saver,
                epoch5_saver,
                epoch10_saver,
            ]

            # Use different epochs
            with overity.api.epoch_metrics(1) as metrics1:
                metrics1.simple("loss", 0.8)
                metrics1.percentage("accuracy", 0.65)

            with overity.api.epoch_metrics(5) as metrics5:
                metrics5.simple("loss", 0.3)
                metrics5.percentage("accuracy", 0.85)

            with overity.api.epoch_metrics(10) as metrics10:
                metrics10.simple("loss", 0.1)
                metrics10.percentage("accuracy", 0.95)

            # Verify each epoch has its own metrics
            assert epoch1_dict["loss"].value == 0.8
            assert epoch1_dict["accuracy"].value == 0.65

            assert epoch5_dict["loss"].value == 0.3
            assert epoch5_dict["accuracy"].value == 0.85

            assert epoch10_dict["loss"].value == 0.1
            assert epoch10_dict["accuracy"].value == 0.95

    def test_epoch_metrics_negative_epoch(self):
        """Test epoch_metrics with negative epoch number raises InvalidEpochValue."""
        # Mock the flow function to bypass API initialization check
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Configure the flow function to raise InvalidEpochValue for negative epochs
            def side_effect(ctx, epoch, display_output=True):
                if epoch < 0:
                    raise InvalidEpochValue(epoch)
                return Mock()  # Return a mock MetricSaver for valid epochs

            mock_flow_epoch_metrics.side_effect = side_effect

            # Should raise InvalidEpochValue for negative epoch numbers
            with pytest.raises(InvalidEpochValue) as exc_info:
                overity.api.epoch_metrics(-1)

            # Verify the exception contains the correct epoch value
            assert exc_info.value.epoch == -1

            # Verify that the flow function was called
            mock_flow_epoch_metrics.assert_called_once_with(
                overity.api._CTX, -1, display_output=True
            )

    def test_epoch_metrics_large_epoch_number(self):
        """Test epoch_metrics with large epoch number."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            mock_metric_saver = Mock(spec=MetricSaver)
            mock_flow_epoch_metrics.return_value = mock_metric_saver

            # Test with large epoch number
            large_epoch = 1000000
            result = overity.api.epoch_metrics(large_epoch)

            assert result == mock_metric_saver
            mock_flow_epoch_metrics.assert_called_once_with(
                overity.api._CTX, large_epoch, display_output=True
            )

    def test_epoch_metrics_zero_epoch(self):
        """Test epoch_metrics with epoch 0."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            mock_output_dict = {}
            metric_saver = MetricSaver("Epoch 0 metrics", mock_output_dict)
            mock_flow_epoch_metrics.return_value = metric_saver

            # Use epoch 0 (common in ML frameworks)
            with overity.api.epoch_metrics(0) as metrics:
                metrics.simple("initial_loss", 2.5)
                metrics.percentage("initial_accuracy", 0.10)

            # Verify metrics were recorded
            assert mock_output_dict["initial_loss"].value == 2.5
            assert mock_output_dict["initial_accuracy"].value == 0.10

    @patch("overity.api._CTX")
    def test_epoch_metrics_with_uninitialized_api(self, mock_ctx):
        """Test epoch_metrics behavior when API is not initialized."""
        # Mock context to simulate uninitialized API
        mock_ctx.init_ok = False

        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Configure the flow function to raise UninitAPIError
            mock_flow_epoch_metrics.side_effect = UninitAPIError()

            # Should raise UninitAPIError when API not initialized
            with pytest.raises(UninitAPIError):
                overity.api.epoch_metrics(1)

    def test_epoch_metrics_same_epoch_multiple_calls(self):
        """Test multiple calls to epoch_metrics with the same epoch."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            # Same output dict for same epoch
            shared_dict = {}
            metric_saver = MetricSaver("Epoch 1 metrics", shared_dict)
            mock_flow_epoch_metrics.return_value = metric_saver

            # Multiple calls with same epoch
            with overity.api.epoch_metrics(1) as metrics1:
                metrics1.simple("accuracy", 0.80)

            with overity.api.epoch_metrics(1) as metrics2:
                metrics2.simple("loss", 0.4)
                metrics2.percentage("validation_accuracy", 0.75)

            # Verify both sets of metrics are in the same dictionary
            assert len(shared_dict) == 3
            assert shared_dict["accuracy"].value == 0.80
            assert shared_dict["loss"].value == 0.4
            assert shared_dict["validation_accuracy"].value == 0.75

    def test_epoch_metrics_metric_types(self):
        """Test all available metric types in epoch_metrics."""
        with patch("overity.api.flow.epoch_metrics") as mock_flow_epoch_metrics:
            mock_output_dict = {}
            metric_saver = MetricSaver("Epoch 1 metrics", mock_output_dict)
            mock_flow_epoch_metrics.return_value = metric_saver

            # Test all metric types
            with overity.api.epoch_metrics(1) as metrics:
                metrics.simple("accuracy", 0.95)
                metrics.percentage("accuracy_pct", 0.95)
                metrics.scale_lin("loss_scaled", 0.3, 0.0, 1.0)
                metrics.range_lin("batch_size", 32, 8, 128)

            # Verify correct metric types were created
            from overity.model.report.metrics import (
                SimpleValue,
                PercentageValue,
                LinScaleValue,
                LinRangeValue,
            )

            assert isinstance(mock_output_dict["accuracy"], SimpleValue)
            assert mock_output_dict["accuracy"].value == 0.95

            assert isinstance(mock_output_dict["accuracy_pct"], PercentageValue)
            assert mock_output_dict["accuracy_pct"].value == 0.95

            assert isinstance(mock_output_dict["loss_scaled"], LinScaleValue)
            assert mock_output_dict["loss_scaled"].value == 0.3
            assert mock_output_dict["loss_scaled"].low == 0.0
            assert mock_output_dict["loss_scaled"].high == 1.0

            assert isinstance(mock_output_dict["batch_size"], LinRangeValue)
            assert mock_output_dict["batch_size"].value == 32
            assert mock_output_dict["batch_size"].low == 8
            assert mock_output_dict["batch_size"].high == 128

    def test_backend_epoch_metrics_negative_epoch_direct(self):
        """Test backend flow.epoch_metrics directly with negative epoch."""
        from overity.backend.flow import epoch_metrics as backend_epoch_metrics
        from overity.backend.flow.ctx import FlowCtx

        # Create a mock context with initialized report
        mock_ctx = Mock(spec=FlowCtx)
        mock_ctx.report = Mock()
        mock_ctx.report.epoch_metrics = {}
        mock_ctx.init_ok = True  # Bypass API guard

        # Should raise InvalidEpochValue for negative epoch
        with pytest.raises(InvalidEpochValue) as exc_info:
            backend_epoch_metrics(mock_ctx, -5)

        # Verify the exception contains the correct epoch value
        assert exc_info.value.epoch == -5

        # Verify that no metrics dictionary was created
        assert len(mock_ctx.report.epoch_metrics) == 0

    def test_backend_epoch_metrics_valid_epochs(self):
        """Test backend flow.epoch_metrics with valid non-negative epochs."""
        from overity.backend.flow import epoch_metrics as backend_epoch_metrics
        from overity.backend.flow.ctx import FlowCtx

        # Create a mock context with initialized report
        mock_ctx = Mock(spec=FlowCtx)
        mock_ctx.report = Mock()
        mock_ctx.report.epoch_metrics = {}
        mock_ctx.init_ok = True  # Bypass API guard

        # Test with valid epochs (including zero)
        result_0 = backend_epoch_metrics(mock_ctx, 0)
        result_1 = backend_epoch_metrics(mock_ctx, 1)
        result_100 = backend_epoch_metrics(mock_ctx, 100)

        # Verify that MetricSaver instances are returned
        assert isinstance(result_0, MetricSaver)
        assert isinstance(result_1, MetricSaver)
        assert isinstance(result_100, MetricSaver)

        # Verify that metrics dictionaries were created
        assert 0 in mock_ctx.report.epoch_metrics
        assert 1 in mock_ctx.report.epoch_metrics
        assert 100 in mock_ctx.report.epoch_metrics
