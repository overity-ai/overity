"""
Unit tests for report_json.py
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime as dt

from overity.exchange.report_json import from_file, to_file
from overity.model.report import (
    MethodReport,
    MethodExecutionStatus,
    MethodExecutionStage,
    MethodReportLogItem,
)
from overity.model.general_info.method import MethodKind, MethodAuthor, MethodInfo
from overity.model.traceability import (
    ArtifactKey,
    ArtifactGraph,
    ArtifactLink,
    ArtifactLinkKind,
    ArtifactKind,
)
from overity.model.report.metrics import Metric, SimpleValue

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


class TestReportJson:
    def test_round_trip_method_report(self):
        """Test that encoding and decoding a MethodReport works correctly."""
        # Create test data
        method_info = MethodInfo(
            slug="test-method",
            kind=MethodKind.TrainingOptimization,
            display_name="Test Method",
            authors=[
                MethodAuthor(
                    name="John Doe",
                    email="john@example.com",
                    contribution="Lead developer",
                ),
                MethodAuthor(
                    name="Jane Smith", email="jane@example.com", contribution=None
                ),
            ],
            metadata={"version": "1.0", "language": "python"},
            description="A test method",
            path=Path("/path/to/method"),
        )

        artifact_graph = ArtifactGraph(
            links=[
                ArtifactLink(
                    a=ArtifactKey(kind=ArtifactKind.Model, id="model1"),
                    b=ArtifactKey(kind=ArtifactKind.Dataset, id="dataset1"),
                    kind=ArtifactLinkKind.ModelUse,
                )
            ],
            metadata={ArtifactKey(kind=ArtifactKind.Model, id="model1"): {"size": 100}},
        )

        logs = [
            MethodReportLogItem(
                timestamp=dt(2023, 1, 1, 12, 0, 0),
                severity="INFO",
                source="test",
                message="Test log message",
            )
        ]

        metrics = {
            "accuracy": SimpleValue(0.95),
            "loss": SimpleValue(0.05),
        }

        epoch_metrics = {
            1: {"accuracy": SimpleValue(0.8), "loss": SimpleValue(0.2)},
            2: {"accuracy": SimpleValue(0.85), "loss": SimpleValue(0.15)},
            3: {"accuracy": SimpleValue(0.9), "loss": SimpleValue(0.1)},
        }

        # Create sample graphs if plotly is available
        graphs = {}
        if PLOTLY_AVAILABLE:
            # Create a simple line chart for accuracy
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=[1, 2, 3],
                    y=[0.8, 0.85, 0.9],
                    mode="lines+markers",
                    name="Accuracy",
                )
            )
            fig1.update_layout(
                title="Accuracy over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
            )

            # Create a bar chart for loss
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=[1, 2, 3], y=[0.2, 0.15, 0.1], name="Loss"))
            fig2.update_layout(
                title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss"
            )

            graphs = {"accuracy_plot": fig1, "loss_plot": fig2}

        original_report = MethodReport(
            uuid="test-uuid-123",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={"python": "3.9", "cuda": "11.0"},
            context={"batch_size": 32, "epochs": 100},
            method_info=method_info,
            traceability_graph=artifact_graph,
            logs=logs,
            metrics=metrics,
            epoch_metrics=epoch_metrics,
            outputs=None,
            graphs=graphs,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original_report, temp_path)

            print(temp_path.read_text())

            # Decode from file
            result = from_file(temp_path)

            # Assertions
            assert result.uuid == original_report.uuid
            assert result.program == original_report.program
            assert result.date_started == original_report.date_started
            assert result.date_ended == original_report.date_ended
            assert result.status == original_report.status
            assert result.environment == original_report.environment
            assert result.context == original_report.context

            # Method info
            assert result.method_info.slug == method_info.slug
            assert result.method_info.kind == method_info.kind
            assert result.method_info.display_name == method_info.display_name
            assert len(result.method_info.authors) == len(method_info.authors)
            assert result.method_info.authors[0].name == method_info.authors[0].name
            assert result.method_info.authors[0].email == method_info.authors[0].email
            assert (
                result.method_info.authors[0].contribution
                == method_info.authors[0].contribution
            )
            assert result.method_info.metadata == method_info.metadata
            assert result.method_info.description == method_info.description
            assert result.method_info.path == method_info.path

            # Traceability graph
            assert len(result.traceability_graph.links) == len(artifact_graph.links)
            # Convert sets to lists for comparison
            result_link = list(result.traceability_graph.links)[0]
            original_link = list(artifact_graph.links)[0]
            assert result_link.a.kind == original_link.a.kind
            assert result_link.a.id == original_link.a.id
            assert result_link.b.kind == original_link.b.kind
            assert result_link.b.id == original_link.b.id
            assert result_link.kind == original_link.kind

            # Metadata
            key = ArtifactKey(kind=ArtifactKind.Model, id="model1")
            assert key in result.traceability_graph.metadata
            assert result.traceability_graph.metadata[key] == {"size": 100}

            # Logs
            assert len(result.logs) == len(logs)
            assert result.logs[0].timestamp == logs[0].timestamp
            assert result.logs[0].severity == logs[0].severity
            assert result.logs[0].source == logs[0].source
            assert result.logs[0].message == logs[0].message

            # Metrics
            assert len(result.metrics) == len(metrics)
            assert result.metrics["accuracy"].data() == metrics["accuracy"].data()
            assert result.metrics["loss"].data() == metrics["loss"].data()

            # Epoch metrics
            assert len(result.epoch_metrics) == len(epoch_metrics)
            assert 1 in result.epoch_metrics
            assert 2 in result.epoch_metrics
            assert 3 in result.epoch_metrics
            assert (
                result.epoch_metrics[1]["accuracy"].data()
                == epoch_metrics[1]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[1]["loss"].data()
                == epoch_metrics[1]["loss"].data()
            )
            assert (
                result.epoch_metrics[2]["accuracy"].data()
                == epoch_metrics[2]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[2]["loss"].data()
                == epoch_metrics[2]["loss"].data()
            )
            assert (
                result.epoch_metrics[3]["accuracy"].data()
                == epoch_metrics[3]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[3]["loss"].data()
                == epoch_metrics[3]["loss"].data()
            )

            # Graphs
            if PLOTLY_AVAILABLE:
                assert result.graphs is not None
                assert len(result.graphs) == len(graphs)
                assert "accuracy_plot" in result.graphs
                assert "loss_plot" in result.graphs

                # Verify the figures are preserved
                assert isinstance(result.graphs["accuracy_plot"], Figure)
                assert isinstance(result.graphs["loss_plot"], Figure)

                # Verify figure data is preserved
                acc_fig = result.graphs["accuracy_plot"]
                assert acc_fig.data[0].type == "scatter"
                assert list(acc_fig.data[0].x) == [1, 2, 3]
                assert list(acc_fig.data[0].y) == [0.8, 0.85, 0.9]

                loss_fig = result.graphs["loss_plot"]
                assert loss_fig.data[0].type == "bar"
                assert list(loss_fig.data[0].x) == [1, 2, 3]
                assert list(loss_fig.data[0].y) == [0.2, 0.15, 0.1]
            else:
                # When plotly is not available, graphs should be empty dict
                assert result.graphs == {}

        finally:
            temp_path.unlink()

    def test_parse_minimal_valid_json(self):
        """Test parsing a minimal valid JSON report."""
        data = {
            "uuid": "minimal-uuid",
            "program": "minimal-program",
            "date_started": "2023-01-01T10:00:00",
            "date_ended": "2023-01-01T11:00:00",
            "stage": "preview",
            "status": "execution_success",
            "environment": {},
            "context": {},
            "method_info": {
                "slug": "minimal-method",
                "kind": "training_optimization",
                "display_name": "Minimal Method",
                "authors": [{"name": "Test Author", "email": "test@example.com"}],
                "metadata": {},
            },
            "traceability_graph": {"links": [], "metadata": []},
            "logs": [],
            "metrics": {},
            "epoch_metrics": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MethodReport)
                assert result.uuid == "minimal-uuid"
                assert result.program == "minimal-program"
                assert result.status == MethodExecutionStatus.ExecutionSuccess
                assert result.method_info.slug == "minimal-method"
                assert result.method_info.kind == MethodKind.TrainingOptimization
                assert len(result.traceability_graph.links) == 0
                assert len(result.logs) == 0
                assert len(result.metrics) == 0
                assert len(result.epoch_metrics) == 0
            finally:
                Path(f.name).unlink()

    def test_epoch_metrics_round_trip(self):
        """Test that encoding and decoding epoch metrics works correctly."""
        # Create test data with epoch metrics
        method_info = MethodInfo(
            slug="test-method",
            kind=MethodKind.TrainingOptimization,
            display_name="Test Method",
            authors=[
                MethodAuthor(
                    name="John Doe",
                    email="john@example.com",
                    contribution="Lead developer",
                )
            ],
            metadata={},
            description="A test method",
            path=Path("/path/to/method"),
        )

        epoch_metrics = {
            1: {"accuracy": SimpleValue(0.7), "loss": SimpleValue(0.3)},
            5: {"accuracy": SimpleValue(0.8), "loss": SimpleValue(0.2)},
            10: {"accuracy": SimpleValue(0.9), "loss": SimpleValue(0.1)},
        }

        original_report = MethodReport(
            uuid="test-epoch-metrics",
            program="test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            method_info=method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics=epoch_metrics,
            outputs=None,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original_report, temp_path)

            # Decode from file
            result = from_file(temp_path)

            # Assertions for epoch metrics
            assert len(result.epoch_metrics) == len(epoch_metrics)
            assert 1 in result.epoch_metrics
            assert 5 in result.epoch_metrics
            assert 10 in result.epoch_metrics

            # Check epoch 1 metrics
            assert (
                result.epoch_metrics[1]["accuracy"].data()
                == epoch_metrics[1]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[1]["loss"].data()
                == epoch_metrics[1]["loss"].data()
            )

            # Check epoch 5 metrics
            assert (
                result.epoch_metrics[5]["accuracy"].data()
                == epoch_metrics[5]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[5]["loss"].data()
                == epoch_metrics[5]["loss"].data()
            )

            # Check epoch 10 metrics
            assert (
                result.epoch_metrics[10]["accuracy"].data()
                == epoch_metrics[10]["accuracy"].data()
            )
            assert (
                result.epoch_metrics[10]["loss"].data()
                == epoch_metrics[10]["loss"].data()
            )

        finally:
            temp_path.unlink()

    def test_invalid_json_missing_required_field(self):
        """Test that parsing fails for JSON missing required fields."""
        data = {
            "uuid": "invalid-uuid",
            # Missing "program", "date_started", etc.
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                with pytest.raises(KeyError):
                    from_file(Path(f.name))
            finally:
                Path(f.name).unlink()

    def test_invalid_json_wrong_status(self):
        """Test that parsing fails for invalid status value."""
        data = {
            "uuid": "invalid-uuid",
            "program": "test-program",
            "date_started": "2023-01-01T10:00:00",
            "date_ended": "2023-01-01T11:00:00",
            "stage": "preview",
            "status": "invalid_status",
            "environment": {},
            "context": {},
            "method_info": {
                "slug": "test-method",
                "kind": "training_optimization",
                "display_name": "Test Method",
                "authors": [{"name": "Test Author", "email": "test@example.com"}],
                "metadata": {},
            },
            "traceability_graph": {"links": [], "metadata": []},
            "logs": [],
            "metrics": {},
            "epoch_metrics": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                with pytest.raises(ValueError):
                    from_file(Path(f.name))
            finally:
                Path(f.name).unlink()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
    def test_graphs_encoding_decoding(self):
        """Test that graphs are properly encoded and decoded."""
        # Create test data with various plotly figures
        method_info = MethodInfo(
            slug="graph-test-method",
            kind=MethodKind.TrainingOptimization,
            display_name="Graph Test Method",
            authors=[MethodAuthor(name="Test Author", email="test@example.com")],
            metadata={},
            description="A test method for graphs",
            path=Path("/path/to/method"),
        )

        # Create various types of plotly figures
        figures = {
            "simple_scatter": go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6])),
            "bar_chart": go.Figure(data=go.Bar(x=["A", "B", "C"], y=[10, 20, 30])),
            "line_chart": go.Figure(
                data=go.Scatter(x=[1, 2, 3, 4], y=[1, 4, 9, 16], mode="lines")
            ),
            "heatmap": go.Figure(data=go.Heatmap(z=[[1, 2], [3, 4]])),
            "multi_trace": go.Figure(
                [
                    go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Line 1"),
                    go.Scatter(x=[1, 2, 3], y=[3, 2, 1], name="Line 2"),
                ]
            ),
        }

        original_report = MethodReport(
            uuid="graph-test-uuid",
            program="graph-test-program",
            date_started=dt(2023, 1, 1, 10, 0, 0),
            date_ended=dt(2023, 1, 1, 11, 0, 0),
            stage=MethodExecutionStage.Preview,
            status=MethodExecutionStatus.ExecutionSuccess,
            environment={},
            context={},
            method_info=method_info,
            traceability_graph=ArtifactGraph.default(),
            logs=[],
            metrics={},
            epoch_metrics={},
            outputs=None,
            graphs=figures,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original_report, temp_path)

            # Decode from file
            result = from_file(temp_path)

            # Verify graphs are preserved
            assert result.graphs is not None
            assert len(result.graphs) == len(figures)

            # Verify each figure type
            for fig_name, original_fig in figures.items():
                assert fig_name in result.graphs
                result_fig = result.graphs[fig_name]

                # Verify it's a Figure object
                assert isinstance(result_fig, Figure)

                # Verify the figure has the same number of traces
                assert len(result_fig.data) == len(original_fig.data)

                # Verify basic properties are preserved
                for i, original_trace in enumerate(original_fig.data):
                    result_trace = result_fig.data[i]
                    assert result_trace.type == original_trace.type

                    # For simple traces, verify some data is preserved
                    if hasattr(original_trace, "x") and original_trace.x is not None:
                        assert list(result_trace.x) == list(original_trace.x)
                    if hasattr(original_trace, "y") and original_trace.y is not None:
                        assert list(result_trace.y) == list(original_trace.y)

        finally:
            temp_path.unlink()
