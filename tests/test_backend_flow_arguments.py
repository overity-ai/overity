"""
Unit tests for ArgumentParser class in overity.backend.flow.arguments module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

from overity.backend.flow.arguments import ArgumentParser
from overity.backend.flow.ctx import FlowCtx, RunMode
from overity.model.arguments import ArgumentSchema, OptionSchema, FlagSchema, ListSchema
from overity.errors import DuplicateArgumentNameError


class TestArgumentParser:
    """Test the ArgumentParser class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock FlowCtx for testing
        self.mock_ctx = Mock(spec=FlowCtx)
        self.mock_ctx.run_mode = RunMode.Standalone

        # Create ArgumentParser instance
        self.parser = ArgumentParser(self.mock_ctx)

    def test_init(self):
        """Test ArgumentParser initialization."""
        assert self.parser.ctx == self.mock_ctx
        assert self.parser.schema == {}
        assert self.parser.parsed_vars == {}

    def test_add_argument(self):
        """Test adding a positional argument."""
        self.parser.add_argument("input_file", "Path to input file")

        assert "input_file" in self.parser.schema
        assert isinstance(self.parser.schema["input_file"], ArgumentSchema)
        assert self.parser.schema["input_file"].name == "input_file"
        assert self.parser.schema["input_file"].help == "Path to input file"

    def test_add_option(self):
        """Test adding an option with default value."""
        self.parser.add_option("output_dir", "Output directory", "./output")

        assert "output_dir" in self.parser.schema
        assert isinstance(self.parser.schema["output_dir"], OptionSchema)
        assert self.parser.schema["output_dir"].name == "output_dir"
        assert self.parser.schema["output_dir"].help == "Output directory"
        assert self.parser.schema["output_dir"].default == "./output"

    def test_add_flag(self):
        """Test adding a boolean flag."""
        self.parser.add_flag("verbose", "Enable verbose output")

        assert "verbose" in self.parser.schema
        assert isinstance(self.parser.schema["verbose"], FlagSchema)
        assert self.parser.schema["verbose"].name == "verbose"
        assert self.parser.schema["verbose"].help == "Enable verbose output"

    def test_add_argument_duplicate_name_raises_error(self):
        """Test that adding argument with duplicate name raises DuplicateArgumentNameError."""
        self.parser.add_argument("test_arg", "First argument")

        with pytest.raises(DuplicateArgumentNameError) as exc_info:
            self.parser.add_argument("test_arg", "Second argument")

        assert "test_arg" in str(exc_info.value)

    def test_add_option_duplicate_name_raises_error(self):
        """Test that adding option with duplicate name raises DuplicateArgumentNameError."""
        self.parser.add_option("test_opt", "First option", "default1")

        with pytest.raises(DuplicateArgumentNameError) as exc_info:
            self.parser.add_option("test_opt", "Second option", "default2")

        assert "test_opt" in str(exc_info.value)

    def test_add_flag_duplicate_name_raises_error(self):
        """Test that adding flag with duplicate name raises DuplicateArgumentNameError."""
        self.parser.add_flag("test_flag", "First flag")

        with pytest.raises(DuplicateArgumentNameError) as exc_info:
            self.parser.add_flag("test_flag", "Second flag")

        assert "test_flag" in str(exc_info.value)

    def test_mixed_duplicate_names_raises_error(self):
        """Test that adding different argument types with same name raises error."""
        self.parser.add_argument("test_name", "Argument")

        with pytest.raises(DuplicateArgumentNameError):
            self.parser.add_option("test_name", "Option", "default")

        with pytest.raises(DuplicateArgumentNameError):
            self.parser.add_flag("test_name", "Flag")

    def test_parse_args_standalone_with_mixed_arguments(self):
        """Test parsing arguments in standalone mode with mixed argument types."""
        # Setup schema
        self.parser.add_argument("input_file", "Input file path")
        self.parser.add_option("output_dir", "Output directory", "./output")
        self.parser.add_flag("verbose", "Enable verbose output")

        # Mock argparse
        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            # Mock parsed args
            mock_args = Namespace(
                input_file="data.txt", output_dir="./results", verbose=True
            )
            mock_parser_instance.parse_args.return_value = mock_args

            # Parse args
            self.parser._parse_args_standalone()

            # Verify argparse was configured correctly
            mock_cmdargs_class.assert_called_once()

            # Check that arguments were added correctly
            calls = mock_parser_instance.add_argument.call_args_list
            assert len(calls) == 3

            # Check positional argument
            assert calls[0][0][0] == "input_file"
            assert calls[0][1]["help"] == "Input file path"

            # Check option argument
            assert calls[1][0][0] == "--output_dir"
            assert calls[1][1]["default"] == "./output"
            assert calls[1][1]["help"] == "Output directory"

            # Check flag argument
            assert calls[2][0][0] == "--verbose"
            assert calls[2][1]["action"] == "store_true"
            assert calls[2][1]["help"] == "Enable verbose output"

            # Verify parsed values
            assert self.parser.parsed_vars["input_file"] == "data.txt"
            assert self.parser.parsed_vars["output_dir"] == "./results"
            assert self.parser.parsed_vars["verbose"] is True

    @patch("builtins.input")
    def test_parse_args_interactive_with_mixed_arguments(self, mock_input):
        """Test parsing arguments in interactive mode with mixed argument types."""
        # Setup schema
        self.parser.add_argument("input_file", "Input file path")
        self.parser.add_option("output_dir", "Output directory", "./default_output")
        self.parser.add_flag("verbose", "Enable verbose output")

        # Mock user input
        mock_input.return_value = "user_input.txt"

        # Parse args
        self.parser._parse_args_interactive()

        # Verify input was called for positional argument
        mock_input.assert_called_once_with(
            "Please provide value for argument: input_file"
        )

        # Verify parsed values
        assert self.parser.parsed_vars["input_file"] == "user_input.txt"
        assert self.parser.parsed_vars["output_dir"] == "./default_output"
        assert self.parser.parsed_vars["verbose"] is False  # Flags default to False

    @patch("builtins.input")
    def test_parse_args_interactive_multiple_arguments(self, mock_input):
        """Test parsing multiple positional arguments in interactive mode."""
        # Setup schema with multiple arguments
        self.parser.add_argument("arg1", "First argument")
        self.parser.add_argument("arg2", "Second argument")
        self.parser.add_option("opt1", "First option", "default1")

        # Mock different inputs for each call
        inputs = ["value1", "value2"]
        mock_input.side_effect = inputs

        # Parse args
        self.parser._parse_args_interactive()

        # Verify inputs were called correctly
        assert mock_input.call_count == 2
        mock_input.assert_any_call("Please provide value for argument: arg1")
        mock_input.assert_any_call("Please provide value for argument: arg2")

        # Verify parsed values
        assert self.parser.parsed_vars["arg1"] == "value1"
        assert self.parser.parsed_vars["arg2"] == "value2"
        assert self.parser.parsed_vars["opt1"] == "default1"

    def test_parse_args_standalone_mode(self):
        """Test parse_args method in standalone mode."""
        self.mock_ctx.run_mode = RunMode.Standalone

        with patch.object(self.parser, "_parse_args_standalone") as mock_standalone:
            self.parser.parse_args()
            mock_standalone.assert_called_once()

    def test_parse_args_interactive_mode(self):
        """Test parse_args method in interactive mode."""
        self.mock_ctx.run_mode = RunMode.Interactive

        with patch.object(self.parser, "_parse_args_interactive") as mock_interactive:
            self.parser.parse_args()
            mock_interactive.assert_called_once()

    def test_context_method(self):
        """Test the context method returns parsed variables."""
        # Setup some parsed variables
        expected_context = {
            "input_file": "test.txt",
            "output_dir": "./output",
            "verbose": True,
        }
        self.parser.parsed_vars = expected_context

        # Test context method
        result = self.parser.context()
        assert result == expected_context

    def test_context_method_empty(self):
        """Test the context method with empty parsed variables."""
        result = self.parser.context()
        assert result == {}

    def test_parse_args_standalone_empty_schema(self):
        """Test parsing with empty schema in standalone mode."""
        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            # Mock empty namespace
            mock_args = Namespace()
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Should still work with empty schema
            assert self.parser.parsed_vars == {}

    def test_parse_args_interactive_empty_schema(self):
        """Test parsing with empty schema in interactive mode."""
        self.parser._parse_args_interactive()
        assert self.parser.parsed_vars == {}

    def test_parse_args_interactive_only_flags(self):
        """Test parsing only flags in interactive mode."""
        self.parser.add_flag("flag1", "First flag")
        self.parser.add_flag("flag2", "Second flag")

        with patch("builtins.input") as mock_input:
            self.parser._parse_args_interactive()

            # No input should be requested for flags
            mock_input.assert_not_called()

            # Flags should default to False
            assert self.parser.parsed_vars["flag1"] is False
            assert self.parser.parsed_vars["flag2"] is False

    def test_parse_args_interactive_only_options(self):
        """Test parsing only options in interactive mode."""
        self.parser.add_option("opt1", "First option", "default1")
        self.parser.add_option("opt2", "Second option", "default2")

        with patch("builtins.input") as mock_input:
            self.parser._parse_args_interactive()

            # No input should be requested for options
            mock_input.assert_not_called()

            # Options should use their defaults
            assert self.parser.parsed_vars["opt1"] == "default1"
            assert self.parser.parsed_vars["opt2"] == "default2"

    def test_parse_args_standalone_argument_with_special_chars(self):
        """Test parsing arguments with special characters in names."""
        self.parser.add_argument("input-file", "Input file with dashes")
        self.parser.add_option("output.dir", "Output with dots", "./out")

        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            mock_args = Namespace(
                **{"input-file": "data.txt", "output.dir": "./results"}
            )
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify the arguments were added with original names
            calls = mock_parser_instance.add_argument.call_args_list
            assert calls[0][0][0] == "input-file"
            assert calls[1][0][0] == "--output.dir"

            # Verify parsed values use original names
            assert self.parser.parsed_vars["input-file"] == "data.txt"
            assert self.parser.parsed_vars["output.dir"] == "./results"

    def test_argument_parser_with_complex_schema(self):
        """Test ArgumentParser with a complex schema containing all argument types."""
        # Setup complex schema
        self.parser.add_argument("data_file", "Path to data file")
        self.parser.add_argument("model_file", "Path to model file")
        self.parser.add_option("epochs", "Number of training epochs", "100")
        self.parser.add_option("learning_rate", "Learning rate", "0.001")
        self.parser.add_flag("train", "Enable training mode")
        self.parser.add_flag("evaluate", "Enable evaluation mode")
        self.parser.add_flag("verbose", "Enable verbose output")

        # Verify schema was built correctly
        assert len(self.parser.schema) == 7
        assert isinstance(self.parser.schema["data_file"], ArgumentSchema)
        assert isinstance(self.parser.schema["epochs"], OptionSchema)
        assert isinstance(self.parser.schema["train"], FlagSchema)

        # Test standalone parsing
        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            mock_args = Namespace(
                data_file="train.csv",
                model_file="model.pkl",
                epochs="50",
                learning_rate="0.01",
                train=True,
                evaluate=False,
                verbose=True,
            )
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify all arguments were parsed
            assert self.parser.parsed_vars["data_file"] == "train.csv"
            assert self.parser.parsed_vars["model_file"] == "model.pkl"
            assert self.parser.parsed_vars["epochs"] == "50"
            assert self.parser.parsed_vars["learning_rate"] == "0.01"
            assert self.parser.parsed_vars["train"] is True
            assert self.parser.parsed_vars["evaluate"] is False
            assert self.parser.parsed_vars["verbose"] is True

    def test_interactive_mode_flag_case_sensitivity_bug(self):
        """Test for the case sensitivity bug in interactive mode (line 95)."""
        self.parser.add_flag("test_flag", "Test flag")

        with patch("builtins.input") as mock_input:
            # This should not raise an AttributeError due to case sensitivity
            try:
                self.parser._parse_args_interactive()
                # If we get here, the bug might be fixed or not triggered
                assert "test_flag" in self.parser.parsed_vars
                assert self.parser.parsed_vars["test_flag"] is False
            except AttributeError as e:
                # This indicates the bug is present (item.Name vs item.name)
                if "Name" in str(e):
                    pytest.fail(
                        "Case sensitivity bug detected: item.Name should be item.name"
                    )
                else:
                    raise

    # ===== List Argument Tests =====
    def test_add_list(self):
        """Test adding a list argument."""
        self.parser.add_list("tags", "List of tags")

        assert "tags" in self.parser.schema
        assert isinstance(self.parser.schema["tags"], ListSchema)
        assert self.parser.schema["tags"].name == "tags"
        assert self.parser.schema["tags"].help == "List of tags"

    def test_add_list_duplicate_name_raises_error(self):
        """Test that adding list with duplicate name raises DuplicateArgumentNameError."""
        self.parser.add_list("test_list", "First list")

        with pytest.raises(DuplicateArgumentNameError) as exc_info:
            self.parser.add_list("test_list", "Second list")

        assert "test_list" in str(exc_info.value)

    def test_parse_args_standalone_with_list_argument(self):
        """Test parsing list arguments in standalone mode."""
        self.parser.add_list("tags", "List of tags")
        self.parser.add_list("files", "List of files")

        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            # Mock parsed args with lists
            mock_args = Namespace(
                tags=["python", "testing", "cli"],
                files=["file1.txt", "file2.py", "file3.json"],
            )
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify argparse was configured correctly for lists
            calls = mock_parser_instance.add_argument.call_args_list
            assert len(calls) == 2

            # Check first list argument
            assert calls[0][0][0] == "--tags"
            assert calls[0][1]["nargs"] == "+"
            assert calls[0][1]["help"] == "List of tags"

            # Check second list argument
            assert calls[1][0][0] == "--files"
            assert calls[1][1]["nargs"] == "+"
            assert calls[1][1]["help"] == "List of files"

            # Verify parsed values are lists
            assert self.parser.parsed_vars["tags"] == ["python", "testing", "cli"]
            assert self.parser.parsed_vars["files"] == [
                "file1.txt",
                "file2.py",
                "file3.json",
            ]

    @patch("builtins.input")
    def test_parse_args_interactive_with_list_argument(self, mock_input):
        """Test parsing list arguments in interactive mode."""
        self.parser.add_list("tags", "List of tags")
        self.parser.add_list("files", "List of files")

        # Mock user input for lists
        inputs = ["python testing cli", "file1.txt file2.py file3.json"]
        mock_input.side_effect = inputs

        self.parser._parse_args_interactive()

        # Verify inputs were called correctly
        assert mock_input.call_count == 2
        mock_input.assert_any_call(
            "Please provide values for list argument: tags (space-separated): "
        )
        mock_input.assert_any_call(
            "Please provide values for list argument: files (space-separated): "
        )

        # Verify parsed values are lists
        assert self.parser.parsed_vars["tags"] == ["python", "testing", "cli"]
        assert self.parser.parsed_vars["files"] == [
            "file1.txt",
            "file2.py",
            "file3.json",
        ]

    @patch("builtins.input")
    def test_parse_args_interactive_with_empty_list_input(self, mock_input):
        """Test parsing list arguments with empty input in interactive mode."""
        self.parser.add_list("tags", "List of tags")

        # Mock empty user input
        mock_input.return_value = ""

        self.parser._parse_args_interactive()

        # Verify empty input results in empty list
        assert self.parser.parsed_vars["tags"] == []

    @patch("builtins.input")
    def test_parse_args_interactive_with_whitespace_list_input(self, mock_input):
        """Test parsing list arguments with whitespace-only input in interactive mode."""
        self.parser.add_list("tags", "List of tags")

        # Mock whitespace user input
        mock_input.return_value = "   \t\n  "

        self.parser._parse_args_interactive()

        # Verify whitespace-only input results in empty list
        assert self.parser.parsed_vars["tags"] == []

    def test_parse_args_standalone_mixed_arguments_with_list(self):
        """Test parsing mixed argument types including lists in standalone mode."""
        self.parser.add_argument("input_file", "Input file path")
        self.parser.add_option("output_dir", "Output directory", "./output")
        self.parser.add_flag("verbose", "Enable verbose output")
        self.parser.add_list("tags", "List of tags")
        self.parser.add_list("include_patterns", "Include patterns")

        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            # Mock parsed args with mixed types including lists
            mock_args = Namespace(
                input_file="data.txt",
                output_dir="./results",
                verbose=True,
                tags=["python", "testing"],
                include_patterns=["*.py", "*.txt"],
            )
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify all argument types were parsed correctly
            assert self.parser.parsed_vars["input_file"] == "data.txt"
            assert self.parser.parsed_vars["output_dir"] == "./results"
            assert self.parser.parsed_vars["verbose"] is True
            assert self.parser.parsed_vars["tags"] == ["python", "testing"]
            assert self.parser.parsed_vars["include_patterns"] == ["*.py", "*.txt"]

    def test_parse_args_interactive_mixed_arguments_with_list(self):
        """Test parsing mixed argument types including lists in interactive mode."""
        self.parser.add_argument("project_name", "Project name")
        self.parser.add_option("config_file", "Config file", "default.conf")
        self.parser.add_flag("debug", "Enable debug mode")
        self.parser.add_list("tags", "List of tags")

        with patch("builtins.input") as mock_input:
            # Mock inputs for different argument types
            inputs = ["MyProject", "python testing cli"]
            mock_input.side_effect = inputs

            self.parser._parse_args_interactive()

            # Verify inputs were called for arguments and lists only
            assert mock_input.call_count == 2
            mock_input.assert_any_call(
                "Please provide value for argument: project_name"
            )
            mock_input.assert_any_call(
                "Please provide values for list argument: tags (space-separated): "
            )

            # Verify all parsed values
            assert self.parser.parsed_vars["project_name"] == "MyProject"
            assert self.parser.parsed_vars["config_file"] == "default.conf"
            assert self.parser.parsed_vars["debug"] is False
            assert self.parser.parsed_vars["tags"] == ["python", "testing", "cli"]

    def test_list_argument_with_special_chars_in_values(self):
        """Test list arguments with special characters in values."""
        self.parser.add_list("patterns", "File patterns")

        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            # Mock parsed args with special characters
            mock_args = Namespace(patterns=["*.py", "test-*", "file.name"])
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify special characters are preserved
            assert self.parser.parsed_vars["patterns"] == [
                "*.py",
                "test-*",
                "file.name",
            ]

    def test_list_argument_duplicate_name_with_different_types(self):
        """Test that list arguments conflict with other argument types."""
        self.parser.add_argument("test_name", "Argument")

        with pytest.raises(DuplicateArgumentNameError):
            self.parser.add_list("test_name", "List")

        self.parser.add_option("test_option", "Option", "default")

        with pytest.raises(DuplicateArgumentNameError):
            self.parser.add_list("test_option", "List")

        self.parser.add_flag("test_flag", "Flag")

        with pytest.raises(DuplicateArgumentNameError):
            self.parser.add_list("test_flag", "List")

    def test_list_argument_in_complex_schema(self):
        """Test list arguments as part of a complex schema."""
        # Setup complex schema with all argument types including lists
        self.parser.add_argument("data_file", "Path to data file")
        self.parser.add_option("output_dir", "Output directory", "./output")
        self.parser.add_flag("verbose", "Enable verbose output")
        self.parser.add_list("tags", "List of tags")
        self.parser.add_list("exclude_patterns", "Exclude patterns")
        self.parser.add_option("config", "Config file", "config.json")
        self.parser.add_flag("debug", "Enable debug mode")

        # Test standalone parsing
        with patch("overity.backend.flow.arguments.CmdArgs") as mock_cmdargs_class:
            mock_parser_instance = Mock()
            mock_cmdargs_class.return_value = mock_parser_instance

            mock_args = Namespace(
                data_file="train.csv",
                output_dir="./results",
                verbose=True,
                tags=["python", "testing", "ml"],
                exclude_patterns=["*.log", "*.tmp"],
                config="custom.conf",
                debug=False,
            )
            mock_parser_instance.parse_args.return_value = mock_args

            self.parser._parse_args_standalone()

            # Verify all argument types including lists
            assert self.parser.parsed_vars["data_file"] == "train.csv"
            assert self.parser.parsed_vars["output_dir"] == "./results"
            assert self.parser.parsed_vars["verbose"] is True
            assert self.parser.parsed_vars["tags"] == ["python", "testing", "ml"]
            assert self.parser.parsed_vars["exclude_patterns"] == ["*.log", "*.tmp"]
            assert self.parser.parsed_vars["config"] == "custom.conf"
            assert self.parser.parsed_vars["debug"] is False


class TestArgumentParserIntegration:
    """Integration tests for ArgumentParser with different FlowCtx configurations."""

    def test_argument_parser_with_different_run_modes(self):
        """Test ArgumentParser behavior with different run modes."""
        # Test with Standalone mode
        standalone_ctx = Mock(spec=FlowCtx)
        standalone_ctx.run_mode = RunMode.Standalone

        standalone_parser = ArgumentParser(standalone_ctx)
        standalone_parser.add_argument("test_arg", "Test argument")

        with patch.object(
            standalone_parser, "_parse_args_standalone"
        ) as mock_standalone:
            standalone_parser.parse_args()
            mock_standalone.assert_called_once()

        # Test with Interactive mode
        interactive_ctx = Mock(spec=FlowCtx)
        interactive_ctx.run_mode = RunMode.Interactive

        interactive_parser = ArgumentParser(interactive_ctx)
        interactive_parser.add_argument("test_arg", "Test argument")

        with patch.object(
            interactive_parser, "_parse_args_interactive"
        ) as mock_interactive:
            interactive_parser.parse_args()
            mock_interactive.assert_called_once()

    def test_argument_parser_preserves_ctx_reference(self):
        """Test that ArgumentParser preserves reference to FlowCtx."""
        different_ctx = Mock(spec=FlowCtx)
        different_ctx.run_mode = RunMode.Standalone

        parser = ArgumentParser(different_ctx)
        assert parser.ctx == different_ctx

        # Modify ctx and verify parser sees changes
        different_ctx.run_mode = RunMode.Interactive
        assert parser.ctx.run_mode == RunMode.Interactive
