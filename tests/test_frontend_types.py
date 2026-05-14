"""
Unit tests for overity.frontend.types module
"""

import pytest
from argparse import ArgumentError

from overity.frontend.types import parse_method_kind, parse_report_kind
from overity.model.general_info.method import MethodKind
from overity.model.report import MethodReportKind


class TestParseMethodKind:
    """Test cases for parse_method_kind function."""

    def test_parse_method_kind_preparation_full_name(self):
        """Test parsing preparation method kind with full name."""
        result = parse_method_kind("preparation")
        assert result == MethodKind.Preparation

    def test_parse_method_kind_preparation_short_form_pr(self):
        """Test parsing preparation method kind with 'pr' short form."""
        result = parse_method_kind("pr")
        assert result == MethodKind.Preparation

    def test_parse_method_kind_preparation_short_form_prep(self):
        """Test parsing preparation method kind with 'prep' short form."""
        result = parse_method_kind("prep")
        assert result == MethodKind.Preparation

    def test_parse_method_kind_training_optimization_full_name(self):
        """Test parsing training optimization method kind with full name."""
        result = parse_method_kind("training-optimization")
        assert result == MethodKind.TrainingOptimization

    def test_parse_method_kind_training_optimization_short_form(self):
        """Test parsing training optimization method kind with short form."""
        result = parse_method_kind("to")
        assert result == MethodKind.TrainingOptimization

    def test_parse_method_kind_measurement_qualification_full_name(self):
        """Test parsing measurement qualification method kind with full name."""
        result = parse_method_kind("measurement-qualification")
        assert result == MethodKind.MeasurementQualification

    def test_parse_method_kind_measurement_qualification_short_form(self):
        """Test parsing measurement qualification method kind with short form."""
        result = parse_method_kind("mq")
        assert result == MethodKind.MeasurementQualification

    def test_parse_method_kind_analysis_full_name(self):
        """Test parsing analysis method kind with full name."""
        result = parse_method_kind("analysis")
        assert result == MethodKind.Analysis

    def test_parse_method_kind_analysis_short_form(self):
        """Test parsing analysis method kind with short form."""
        result = parse_method_kind("an")
        assert result == MethodKind.Analysis

    def test_parse_method_kind_case_sensitivity(self):
        """Test that method kind parsing is case sensitive."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("PREPARATION")
        assert "Invalid method kind: PREPARATION" in str(exc_info.value)

    def test_parse_method_kind_invalid_input(self):
        """Test parsing invalid method kind raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("invalid")
        assert "Invalid method kind: invalid" in str(exc_info.value)

    def test_parse_method_kind_empty_string(self):
        """Test parsing empty string raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("")
        assert "Invalid method kind: " in str(exc_info.value)

    def test_parse_method_kind_none_input(self):
        """Test that None input raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind(None)
        assert "Invalid method kind: None" in str(exc_info.value)

    def test_parse_method_kind_whitespace_input(self):
        """Test parsing whitespace-only string raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("   ")
        assert "Invalid method kind:    " in str(exc_info.value)

    def test_parse_method_kind_partial_match(self):
        """Test that partial matches don't work (must be exact)."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("pre")
        assert "Invalid method kind: pre" in str(exc_info.value)

    def test_parse_method_kind_similar_but_invalid(self):
        """Test similar but invalid inputs."""
        invalid_inputs = ["preparations", "preparationn", "prr", "p", "pre", "train", "measure", "analyse"]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ArgumentError) as exc_info:
                parse_method_kind(invalid_input)
            assert f"Invalid method kind: {invalid_input}" in str(exc_info.value)


class TestParseReportKind:
    """Test cases for parse_report_kind function."""

    def test_parse_report_kind_preparation_full_name(self):
        """Test parsing preparation report kind with full name."""
        result = parse_report_kind("preparation")
        assert result == MethodReportKind.Preparation

    def test_parse_report_kind_preparation_short_form_pr(self):
        """Test parsing preparation report kind with 'pr' short form."""
        result = parse_report_kind("pr")
        assert result == MethodReportKind.Preparation

    def test_parse_report_kind_preparation_short_form_prep(self):
        """Test parsing preparation report kind with 'prep' short form."""
        result = parse_report_kind("prep")
        assert result == MethodReportKind.Preparation

    def test_parse_report_kind_training_optimization_full_name(self):
        """Test parsing training optimization report kind with full name."""
        result = parse_report_kind("training-optimization")
        assert result == MethodReportKind.TrainingOptimization

    def test_parse_report_kind_training_optimization_short_form(self):
        """Test parsing training optimization report kind with short form."""
        result = parse_report_kind("to")
        assert result == MethodReportKind.TrainingOptimization

    def test_parse_report_kind_training_optimization_alt_short_form(self):
        """Test parsing training optimization report kind with alternative short form."""
        result = parse_report_kind("topt")
        assert result == MethodReportKind.TrainingOptimization

    def test_parse_report_kind_execution_full_name(self):
        """Test parsing execution report kind with full name."""
        result = parse_report_kind("execution")
        assert result == MethodReportKind.Execution

    def test_parse_report_kind_execution_short_form(self):
        """Test parsing execution report kind with short form."""
        result = parse_report_kind("exec")
        assert result == MethodReportKind.Execution

    def test_parse_report_kind_execution_alt_short_form(self):
        """Test parsing execution report kind with alternative short form."""
        result = parse_report_kind("ex")
        assert result == MethodReportKind.Execution

    def test_parse_report_kind_analysis_full_name(self):
        """Test parsing analysis report kind with full name."""
        result = parse_report_kind("analysis")
        assert result == MethodReportKind.Analysis

    def test_parse_report_kind_analysis_short_form(self):
        """Test parsing analysis report kind with short form."""
        result = parse_report_kind("an")
        assert result == MethodReportKind.Analysis

    def test_parse_report_kind_case_sensitivity(self):
        """Test that report kind parsing is case sensitive."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("PREPARATION")
        assert "Invalid report kind: PREPARATION" in str(exc_info.value)

    def test_parse_report_kind_invalid_input(self):
        """Test parsing invalid report kind raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("invalid")
        assert "Invalid report kind: invalid" in str(exc_info.value)

    def test_parse_report_kind_empty_string(self):
        """Test parsing empty string raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("")
        assert "Invalid report kind: " in str(exc_info.value)

    def test_parse_report_kind_none_input(self):
        """Test that None input raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind(None)
        assert "Invalid report kind: None" in str(exc_info.value)

    def test_parse_report_kind_whitespace_input(self):
        """Test parsing whitespace-only string raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("   ")
        assert "Invalid report kind:    " in str(exc_info.value)

    def test_parse_report_kind_partial_match(self):
        """Test that partial matches don't work (must be exact)."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("pre")
        assert "Invalid report kind: pre" in str(exc_info.value)

    def test_parse_report_kind_similar_but_invalid(self):
        """Test similar but invalid inputs."""
        invalid_inputs = ["preparations", "preparationn", "prr", "p", "pre", "prepper", "execut", "analyse", "train"]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ArgumentError) as exc_info:
                parse_report_kind(invalid_input)
            assert f"Invalid report kind: {invalid_input}" in str(exc_info.value)

    def test_parse_report_kind_missing_experiment_and_optimization(self):
        """Test that experiment and optimization report kinds are not supported (as per implementation)."""
        # These should raise errors since they're not in the current implementation
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("experiment")
        assert "Invalid report kind: experiment" in str(exc_info.value)

        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("optimization")
        assert "Invalid report kind: optimization" in str(exc_info.value)


class TestParseFunctionsIntegration:
    """Integration tests for both parsing functions."""

    def test_preparation_kind_consistency(self):
        """Test that preparation method and report kinds are consistent."""
        method_kind = parse_method_kind("preparation")
        report_kind = parse_report_kind("preparation")
        
        assert method_kind == MethodKind.Preparation
        assert report_kind == MethodReportKind.Preparation

    def test_analysis_kind_consistency(self):
        """Test that analysis method and report kinds are consistent."""
        method_kind = parse_method_kind("analysis")
        report_kind = parse_report_kind("analysis")
        
        assert method_kind == MethodKind.Analysis
        assert report_kind == MethodReportKind.Analysis

    def test_all_valid_method_kinds(self):
        """Test all valid method kind inputs."""
        valid_inputs = [
            ("preparation", MethodKind.Preparation),
            ("pr", MethodKind.Preparation),
            ("prep", MethodKind.Preparation),
            ("training-optimization", MethodKind.TrainingOptimization),
            ("to", MethodKind.TrainingOptimization),
            ("measurement-qualification", MethodKind.MeasurementQualification),
            ("mq", MethodKind.MeasurementQualification),
            ("analysis", MethodKind.Analysis),
            ("an", MethodKind.Analysis),
        ]
        
        for input_str, expected_kind in valid_inputs:
            result = parse_method_kind(input_str)
            assert result == expected_kind, f"Failed for input: {input_str}"

    def test_all_valid_report_kinds(self):
        """Test all valid report kind inputs."""
        valid_inputs = [
            ("preparation", MethodReportKind.Preparation),
            ("pr", MethodReportKind.Preparation),
            ("prep", MethodReportKind.Preparation),
            ("training-optimization", MethodReportKind.TrainingOptimization),
            ("to", MethodReportKind.TrainingOptimization),
            ("topt", MethodReportKind.TrainingOptimization),
            ("execution", MethodReportKind.Execution),
            ("exec", MethodReportKind.Execution),
            ("ex", MethodReportKind.Execution),
            ("analysis", MethodReportKind.Analysis),
            ("an", MethodReportKind.Analysis),
        ]
        
        for input_str, expected_kind in valid_inputs:
            result = parse_report_kind(input_str)
            assert result == expected_kind, f"Failed for input: {input_str}"

    def test_error_messages_are_descriptive(self):
        """Test that error messages are descriptive and helpful."""
        with pytest.raises(ArgumentError) as exc_info:
            parse_method_kind("xyz")
        assert "Invalid method kind: xyz" in str(exc_info.value)
        
        with pytest.raises(ArgumentError) as exc_info:
            parse_report_kind("xyz")
        assert "Invalid report kind: xyz" in str(exc_info.value)
