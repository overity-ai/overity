"""
Unit tests for report retrieval backend flow functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from overity.backend.flow import report_get, _report_kind_to_artifact_kind
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


class TestBackendFlowReportRetrieval:
    """Test cases for report retrieval backend flow functions"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock context
        self.mock_ctx = Mock()
        self.mock_ctx.init_ok = True
        self.mock_ctx.report = Mock()
        self.mock_ctx.report.run_key = ArtifactKey(
            kind=ArtifactKind.OptimizationRun, id="current-run-123"
        )
        self.mock_ctx.report.traceability_graph = ArtifactGraph.default()
        self.mock_ctx.storage = Mock()

        # Create a mock report for testing
        self.mock_report = Mock(spec=MethodReport)
        self.mock_report.uuid = "test-uuid-123"
        self.mock_report.program = "test-program"

    def test_report_kind_to_artifact_kind_mapping(self):
        """Test the mapping from MethodReportKind to ArtifactKind."""
        # Test experiment mapping
        result = _report_kind_to_artifact_kind(MethodReportKind.Experiment)
        assert result == ArtifactKind.ExperimentRun

        # Test training optimization mapping
        result = _report_kind_to_artifact_kind(MethodReportKind.TrainingOptimization)
        assert result == ArtifactKind.OptimizationReport

        # Test execution mapping
        result = _report_kind_to_artifact_kind(MethodReportKind.Execution)
        assert result == ArtifactKind.ExecutionReport

        # Test analysis mapping
        result = _report_kind_to_artifact_kind(MethodReportKind.Analysis)
        assert result == ArtifactKind.AnalysisReport

    def test_report_get_experiment_basic(self):
        """Test basic functionality of report_get for experiment reports."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        result = report_get(self.mock_ctx, MethodReportKind.Experiment, "exp-uuid-123")

        assert result == self.mock_report
        self.mock_ctx.storage.report_load.assert_called_once_with(
            MethodReportKind.Experiment, "exp-uuid-123"
        )

        # Verify traceability was updated
        links = list(self.mock_ctx.report.traceability_graph.links)
        assert len(links) == 1
        link = links[0]
        assert link.a == self.mock_ctx.report.run_key
        assert link.b.kind == ArtifactKind.ExperimentRun
        assert link.b.id == "exp-uuid-123"
        assert link.kind == ArtifactLinkKind.ReportUse

    def test_report_get_training_optimization_basic(self):
        """Test basic functionality of report_get for training optimization reports."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        result = report_get(
            self.mock_ctx, MethodReportKind.TrainingOptimization, "train-uuid-456"
        )

        assert result == self.mock_report
        self.mock_ctx.storage.report_load.assert_called_once_with(
            MethodReportKind.TrainingOptimization, "train-uuid-456"
        )

        # Verify traceability was updated
        links = list(self.mock_ctx.report.traceability_graph.links)
        assert len(links) == 1
        link = links[0]
        assert link.b.kind == ArtifactKind.OptimizationReport
        assert link.b.id == "train-uuid-456"

    def test_report_get_execution_basic(self):
        """Test basic functionality of report_get for execution reports."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        result = report_get(self.mock_ctx, MethodReportKind.Execution, "exec-uuid-789")

        assert result == self.mock_report
        self.mock_ctx.storage.report_load.assert_called_once_with(
            MethodReportKind.Execution, "exec-uuid-789"
        )

        # Verify traceability was updated
        links = list(self.mock_ctx.report.traceability_graph.links)
        assert len(links) == 1
        link = links[0]
        assert link.b.kind == ArtifactKind.ExecutionReport
        assert link.b.id == "exec-uuid-789"

    def test_report_get_analysis_basic(self):
        """Test basic functionality of report_get for analysis reports."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        result = report_get(
            self.mock_ctx, MethodReportKind.Analysis, "analysis-uuid-abc"
        )

        assert result == self.mock_report
        self.mock_ctx.storage.report_load.assert_called_once_with(
            MethodReportKind.Analysis, "analysis-uuid-abc"
        )

        # Verify traceability was updated
        links = list(self.mock_ctx.report.traceability_graph.links)
        assert len(links) == 1
        link = links[0]
        assert link.b.kind == ArtifactKind.AnalysisReport
        assert link.b.id == "analysis-uuid-abc"

    def test_report_get_not_found(self):
        """Test report_get when report doesn't exist."""
        # Make storage raise ReportNotFound
        self.mock_ctx.storage.report_load.side_effect = ReportNotFound(
            "test-program", MethodReportKind.Experiment, "nonexistent-uuid"
        )

        # Should raise ReportNotFound (from storage backend)
        with pytest.raises(ReportNotFound) as exc_info:
            report_get(self.mock_ctx, MethodReportKind.Experiment, "nonexistent-uuid")

        assert exc_info.value.report_type == MethodReportKind.Experiment
        assert exc_info.value.identifier == "nonexistent-uuid"

    def test_report_get_uninitialized_api(self):
        """Test report_get when API not initialized."""
        self.mock_ctx.init_ok = False

        with pytest.raises(UninitAPIError):
            report_get(self.mock_ctx, MethodReportKind.Experiment, "test-uuid")

    def test_report_get_preserves_original_exception(self):
        """Test that report_get lets storage exceptions bubble up directly."""
        original_exception = ReportNotFound(
            "test-program", MethodReportKind.Experiment, "test-uuid"
        )
        self.mock_ctx.storage.report_load.side_effect = original_exception

        # Should raise the original ReportNotFound directly (no wrapping)
        with pytest.raises(ReportNotFound) as exc_info:
            report_get(self.mock_ctx, MethodReportKind.Experiment, "test-uuid")

        assert exc_info.value == original_exception

    def test_report_get_multiple_accesses_same_run(self):
        """Test accessing multiple reports in the same method run."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        # Access multiple reports
        report_get(self.mock_ctx, MethodReportKind.Experiment, "exp-uuid-1")
        report_get(self.mock_ctx, MethodReportKind.TrainingOptimization, "train-uuid-2")
        report_get(self.mock_ctx, MethodReportKind.Execution, "exec-uuid-3")
        report_get(self.mock_ctx, MethodReportKind.Analysis, "analysis-uuid-4")

        # Verify all 4 reports were loaded
        assert self.mock_ctx.storage.report_load.call_count == 4

        # Verify traceability has 4 links
        links = list(self.mock_ctx.report.traceability_graph.links)
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
            assert link.a == self.mock_ctx.report.run_key
        assert link.kind == ArtifactLinkKind.ReportUse

    def test_report_get_different_run_types(self):
        """Test report access from different types of method runs."""
        run_types = [
            (ArtifactKind.OptimizationRun, "opt-run-123"),
            (ArtifactKind.ExecutionRun, "exec-run-456"),
            (ArtifactKind.AnalysisRun, "analysis-run-789"),
        ]

        for run_kind, run_id in run_types:
            # Reset traceability graph for each test
            self.mock_ctx.report.traceability_graph = ArtifactGraph.default()
            self.mock_ctx.report.run_key = ArtifactKey(kind=run_kind, id=run_id)
            self.mock_ctx.storage.report_load.return_value = self.mock_report

            # Access a report
            report_get(self.mock_ctx, MethodReportKind.Experiment, "test-exp-uuid")

            # Verify the link shows correct run type
            links = list(self.mock_ctx.report.traceability_graph.links)
            assert len(links) == 1
            link = links[0]
            assert link.a.kind == run_kind
            assert link.a.id == run_id

    def test_report_get_logging(self):
        """Test that report_get logs appropriately."""
        self.mock_ctx.storage.report_load.return_value = self.mock_report

        with patch("overity.backend.flow.log") as mock_log:
            report_get(self.mock_ctx, MethodReportKind.Experiment, "test-uuid")

            # Verify logging
            mock_log.info.assert_called_once_with("-> Get experiment report: test-uuid")

    def test_report_get_no_traceability_on_error(self):
        """Test that traceability is not updated when report loading fails."""
        # Make storage raise ReportNotFound
        self.mock_ctx.storage.report_load.side_effect = ReportNotFound(
            "test-program", MethodReportKind.Experiment, "nonexistent-uuid"
        )

        # Initial link count should be 0
        initial_link_count = len(list(self.mock_ctx.report.traceability_graph.links))
        assert initial_link_count == 0

        # Try to get report (should fail)
        with pytest.raises(ReportNotFound):
            report_get(self.mock_ctx, MethodReportKind.Experiment, "nonexistent-uuid")

        # Verify no links were added due to the error
        final_link_count = len(list(self.mock_ctx.report.traceability_graph.links))
        assert final_link_count == 0
