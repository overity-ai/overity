"""
Unit tests for report retrieval API functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import overity.api
from overity.model.report import (
    MethodReport,
    MethodReportKind,
    MethodExecutionStatus,
    MethodExecutionStage,
)
from overity.model.traceability import (
    ArtifactGraph,
    ArtifactKey,
    ArtifactLink,
    ArtifactKind,
    ArtifactLinkKind,
)
from overity.errors import ReportNotFound, UninitAPIError


class TestApiReportRetrieval:
    """Test cases for report retrieval API functions"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock report for testing
        self.mock_report = Mock(spec=MethodReport)
        self.mock_report.uuid = "test-uuid-123"
        self.mock_report.program = "test-program"
        self.mock_report.date_started = datetime.now()
        self.mock_report.date_ended = datetime.now()
        self.mock_report.stage = MethodExecutionStage.Operation
        self.mock_report.status = MethodExecutionStatus.ExecutionSuccess
        self.mock_report.environment = {"test": "env"}
        self.mock_report.context = {"test": "context"}
        self.mock_report.traceability_graph = ArtifactGraph.default()
        self.mock_report.method_info = Mock()
        self.mock_report.logs = []
        self.mock_report.outputs = None
        self.mock_report.metrics = {}
        self.mock_report.epoch_metrics = {}
        self.mock_report.graphs = {}
        self.mock_report.tables = {}

    def test_report_get_experiment_basic(self):
        """Test basic functionality of report_get_experiment API function."""
        with patch("overity.api._CTX") as mock_ctx:
            # Set up mock context
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.run_key = ArtifactKey(
                kind=ArtifactKind.OptimizationRun, id="current-run-123"
            )
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()
            mock_ctx.storage.report_load.return_value = self.mock_report

            # Call the API function
            result = overity.api.report_get_experiment("test-uuid-123")

            # Verify the result
            assert result == self.mock_report
            mock_ctx.storage.report_load.assert_called_once_with(
                MethodReportKind.Experiment, "test-uuid-123"
            )

            # Verify traceability was updated
            # Should have added a ReportUse link
            links = list(mock_ctx.report.traceability_graph.links)
            assert len(links) == 1
            link = links[0]
            assert link.a == mock_ctx.report.run_key
            assert link.b.kind == ArtifactKind.ExperimentRun
            assert link.b.id == "test-uuid-123"
            assert link.kind == ArtifactLinkKind.ReportUse

    def test_report_get_training_optimization_basic(self):
        """Test basic functionality of report_get_training_optimization API function."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.run_key = ArtifactKey(
                kind=ArtifactKind.OptimizationRun, id="current-run-123"
            )
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()
            mock_ctx.storage.report_load.return_value = self.mock_report

            result = overity.api.report_get_training_optimization("test-uuid-456")

            assert result == self.mock_report
            mock_ctx.storage.report_load.assert_called_once_with(
                MethodReportKind.TrainingOptimization, "test-uuid-456"
            )

            # Verify traceability was updated
            links = list(mock_ctx.report.traceability_graph.links)
            assert len(links) == 1
            link = links[0]
            assert link.b.kind == ArtifactKind.OptimizationReport
            assert link.b.id == "test-uuid-456"

    def test_report_get_execution_basic(self):
        """Test basic functionality of report_get_execution API function."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.run_key = ArtifactKey(
                kind=ArtifactKind.ExecutionRun, id="current-run-123"
            )
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()
            mock_ctx.storage.report_load.return_value = self.mock_report

            result = overity.api.report_get_execution("test-uuid-789")

            assert result == self.mock_report
            mock_ctx.storage.report_load.assert_called_once_with(
                MethodReportKind.Execution, "test-uuid-789"
            )

            # Verify traceability was updated
            links = list(mock_ctx.report.traceability_graph.links)
            assert len(links) == 1
            link = links[0]
            assert link.b.kind == ArtifactKind.ExecutionReport
            assert link.b.id == "test-uuid-789"

    def test_report_get_analysis_basic(self):
        """Test basic functionality of report_get_analysis API function."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.run_key = ArtifactKey(
                kind=ArtifactKind.AnalysisRun, id="current-run-123"
            )
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()
            mock_ctx.storage.report_load.return_value = self.mock_report

            result = overity.api.report_get_analysis("test-uuid-abc")

            assert result == self.mock_report
            mock_ctx.storage.report_load.assert_called_once_with(
                MethodReportKind.Analysis, "test-uuid-abc"
            )

            # Verify traceability was updated
            links = list(mock_ctx.report.traceability_graph.links)
            assert len(links) == 1
            link = links[0]
            assert link.b.kind == ArtifactKind.AnalysisReport
            assert link.b.id == "test-uuid-abc"

    def test_report_get_experiment_not_found(self):
        """Test report_get_experiment when report doesn't exist."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()

            # Make storage raise ReportNotFound
            mock_ctx.storage.report_load.side_effect = ReportNotFound(
                "test-program", MethodReportKind.Experiment, "nonexistent-uuid"
            )

            # Should raise ReportNotFound (from storage backend)
            with pytest.raises(ReportNotFound) as exc_info:
                overity.api.report_get_experiment("nonexistent-uuid")

            assert exc_info.value.report_type == MethodReportKind.Experiment
            assert exc_info.value.identifier == "nonexistent-uuid"

    def test_report_get_training_optimization_not_found(self):
        """Test report_get_training_optimization when report doesn't exist."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()

            mock_ctx.storage.report_load.side_effect = ReportNotFound(
                "test-program",
                MethodReportKind.TrainingOptimization,
                "nonexistent-uuid",
            )

            with pytest.raises(ReportNotFound) as exc_info:
                overity.api.report_get_training_optimization("nonexistent-uuid")

            assert exc_info.value.report_type == MethodReportKind.TrainingOptimization
            assert exc_info.value.identifier == "nonexistent-uuid"

    def test_report_get_execution_not_found(self):
        """Test report_get_execution when report doesn't exist."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()

            mock_ctx.storage.report_load.side_effect = ReportNotFound(
                "test-program", MethodReportKind.Execution, "nonexistent-uuid"
            )

            with pytest.raises(ReportNotFound) as exc_info:
                overity.api.report_get_execution("nonexistent-uuid")

            assert exc_info.value.report_type == MethodReportKind.Execution
            assert exc_info.value.identifier == "nonexistent-uuid"

    def test_report_get_analysis_not_found(self):
        """Test report_get_analysis when report doesn't exist."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()

            mock_ctx.storage.report_load.side_effect = ReportNotFound(
                "test-program", MethodReportKind.Analysis, "nonexistent-uuid"
            )

            with pytest.raises(ReportNotFound) as exc_info:
                overity.api.report_get_analysis("nonexistent-uuid")

            assert exc_info.value.report_type == MethodReportKind.Analysis
            assert exc_info.value.identifier == "nonexistent-uuid"

    def test_report_get_experiment_uninitialized_api(self):
        """Test report_get_experiment when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.report_get_experiment("test-uuid")

    def test_report_get_training_optimization_uninitialized_api(self):
        """Test report_get_training_optimization when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.report_get_training_optimization("test-uuid")

    def test_report_get_execution_uninitialized_api(self):
        """Test report_get_execution when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.report_get_execution("test-uuid")

    def test_report_get_analysis_uninitialized_api(self):
        """Test report_get_analysis when API not initialized."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = False

            with pytest.raises(UninitAPIError):
                overity.api.report_get_analysis("test-uuid")

    def test_multiple_report_accesses_same_run(self):
        """Test accessing multiple reports in the same method run."""
        with patch("overity.api._CTX") as mock_ctx:
            mock_ctx.init_ok = True
            mock_ctx.report = Mock()
            mock_ctx.report.run_key = ArtifactKey(
                kind=ArtifactKind.AnalysisRun, id="current-run-123"
            )
            mock_ctx.report.traceability_graph = ArtifactGraph.default()
            mock_ctx.storage = Mock()
            mock_ctx.storage.report_load.return_value = self.mock_report

            # Access multiple reports
            overity.api.report_get_experiment("exp-uuid-1")
            overity.api.report_get_training_optimization("train-uuid-2")
            overity.api.report_get_execution("exec-uuid-3")
            overity.api.report_get_analysis("analysis-uuid-4")

            # Verify all 4 reports were accessed
            assert mock_ctx.storage.report_load.call_count == 4

            # Verify traceability has 4 links
            links = list(mock_ctx.report.traceability_graph.links)
            assert len(links) == 4

            # Verify each link is correct (order doesn't matter since it's a set)
            expected_links = {
                (ArtifactKind.ExperimentRun, "exp-uuid-1"),
                (ArtifactKind.OptimizationReport, "train-uuid-2"),
                (ArtifactKind.ExecutionReport, "exec-uuid-3"),
                (ArtifactKind.AnalysisReport, "analysis-uuid-4"),
            }

            actual_links = {(link.b.kind, link.b.id) for link in links}
            assert actual_links == expected_links

            # Verify all links have correct source and kind
            for link in links:
                assert link.a == mock_ctx.report.run_key
            assert link.kind == ArtifactLinkKind.ReportUse

    def test_report_access_different_run_types(self):
        """Test report access from different types of method runs."""
        run_types = [
            (ArtifactKind.OptimizationRun, "opt-run-123"),
            (ArtifactKind.ExecutionRun, "exec-run-456"),
            (ArtifactKind.AnalysisRun, "analysis-run-789"),
        ]

        for run_kind, run_id in run_types:
            with patch("overity.api._CTX") as mock_ctx:
                mock_ctx.init_ok = True
                mock_ctx.report = Mock()
                mock_ctx.report.run_key = ArtifactKey(kind=run_kind, id=run_id)
                mock_ctx.report.traceability_graph = ArtifactGraph.default()
                mock_ctx.storage = Mock()
                mock_ctx.storage.report_load.return_value = self.mock_report

                # Access a report
                overity.api.report_get_experiment("test-exp-uuid")

                # Verify the link shows correct run type
                links = list(mock_ctx.report.traceability_graph.links)
                assert len(links) == 1
                link = links[0]
                assert link.a.kind == run_kind
                assert link.a.id == run_id
