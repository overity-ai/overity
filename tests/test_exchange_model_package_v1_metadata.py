"""
Unit tests for model_package_v1/metadata.py
"""

import pytest
import tempfile
import json
from pathlib import Path
from overity.exchange.model_package_v1.metadata import from_file, to_file
from overity.model.ml_model.metadata import (
    MLModelMetadata,
    MLModelAuthor,
    MLModelMaintainer,
)
from overity.model.ml_model.attachment import AttachmentMetadata


class TestModelPackageV1Metadata:
    def test_valid_model_metadata(self):
        """Test parsing a valid model metadata JSON file."""
        data = {
            "name": "Test Model",
            "version": "1.0.0",
            "authors": [
                {"name": "John Doe", "email": "john@example.com"},
                {
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "contribution": "Model architecture",
                },
            ],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx",
            "derives": "base-model",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Test Model"
                assert result.version == "1.0.0"
                assert result.target == "test-target"
                assert result.exchange_format == "onnx"
                assert result.model_file == "model.onnx"
                assert result.derives == "base-model"

                assert len(result.authors) == 2
                assert isinstance(result.authors[0], MLModelAuthor)
                assert result.authors[0].name == "John Doe"
                assert result.authors[0].email == "john@example.com"
                assert result.authors[0].contribution is None

                assert isinstance(result.authors[1], MLModelAuthor)
                assert result.authors[1].name == "Jane Smith"
                assert result.authors[1].email == "jane@example.com"
                assert result.authors[1].contribution == "Model architecture"

                assert len(result.maintainers) == 1
                assert isinstance(result.maintainers[0], MLModelMaintainer)
                assert result.maintainers[0].name == "Maintainer One"
                assert result.maintainers[0].email == "maintainer1@example.com"
            finally:
                Path(f.name).unlink()

    def test_minimal_model_metadata(self):
        """Test parsing a minimal model metadata JSON file."""
        data = {
            "name": "Minimal Model",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "minimal-target",
            "format": "onnx",
            "model_file": "model.onnx",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Minimal Model"
                assert result.version == "1.0.0"
                assert result.target == "minimal-target"
                assert result.exchange_format == "onnx"
                assert result.model_file == "model.onnx"
                assert result.derives is None

                assert len(result.authors) == 1
                assert isinstance(result.authors[0], MLModelAuthor)
                assert result.authors[0].name == "John Doe"
                assert result.authors[0].email == "john@example.com"
                assert result.authors[0].contribution is None

                assert len(result.maintainers) == 1
                assert isinstance(result.maintainers[0], MLModelMaintainer)
                assert result.maintainers[0].name == "Maintainer One"
                assert result.maintainers[0].email == "maintainer1@example.com"
            finally:
                Path(f.name).unlink()

    def test_round_trip_model_metadata(self):
        """Test that encoding and decoding a MLModelMetadata works correctly."""
        original = MLModelMetadata(
            name="Round Trip Model",
            version="2.0.0",
            authors=[
                MLModelAuthor(
                    name="John Doe", email="john@example.com", contribution=None
                ),
                MLModelAuthor(
                    name="Jane Smith",
                    email="jane@example.com",
                    contribution="Model architecture",
                ),
            ],
            maintainers=[
                MLModelMaintainer(
                    name="Maintainer One", email="maintainer1@example.com"
                )
            ],
            target="round-trip-target",
            exchange_format="onnx",
            model_file="model.onnx",
            derives="base-model",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original, temp_path)

            # Decode from file
            result = from_file(temp_path)

            # Assertions
            assert result.name == original.name
            assert result.version == original.version
            assert result.target == original.target
            assert result.exchange_format == original.exchange_format
            assert result.model_file == original.model_file
            assert result.derives == original.derives

            assert len(result.authors) == len(original.authors)
            for i in range(len(result.authors)):
                assert result.authors[i].name == original.authors[i].name
                assert result.authors[i].email == original.authors[i].email
                assert (
                    result.authors[i].contribution == original.authors[i].contribution
                )

            assert len(result.maintainers) == len(original.maintainers)
            for i in range(len(result.maintainers)):
                assert result.maintainers[i].name == original.maintainers[i].name
                assert result.maintainers[i].email == original.maintainers[i].email
        finally:
            temp_path.unlink()

    def test_model_metadata_with_attachments(self):
        """Test parsing model metadata with attachments."""
        data = {
            "name": "Model with Attachments",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx",
            "attachments": [
                {
                    "filename": "README.md",
                    "sha256_hash": "abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                    "mimetype": "text/markdown",
                    "description": "Model documentation"
                },
                {
                    "filename": "license.txt",
                    "sha256_hash": "def456abc1237890def456abc1237890def456abc1237890def456abc1237890",
                    "mimetype": "text/plain"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Model with Attachments"
                assert result.version == "1.0.0"
                assert len(result.attachments) == 2

                # Check first attachment
                assert isinstance(result.attachments[0], AttachmentMetadata)
                assert result.attachments[0].filename == "README.md"
                assert result.attachments[0].sha256_hash == "abc123def4567890abc123def4567890abc123def4567890abc123def4567890"
                assert result.attachments[0].mimetype == "text/markdown"
                assert result.attachments[0].description == "Model documentation"

                # Check second attachment (without description)
                assert isinstance(result.attachments[1], AttachmentMetadata)
                assert result.attachments[1].filename == "license.txt"
                assert result.attachments[1].sha256_hash == "def456abc1237890def456abc1237890def456abc1237890def456abc1237890"
                assert result.attachments[1].mimetype == "text/plain"
                assert result.attachments[1].description is None

            finally:
                Path(f.name).unlink()

    def test_model_metadata_with_empty_attachments(self):
        """Test parsing model metadata with empty attachments list."""
        data = {
            "name": "Model with Empty Attachments",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx",
            "attachments": []
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Model with Empty Attachments"
                assert len(result.attachments) == 0

            finally:
                Path(f.name).unlink()

    def test_model_metadata_without_attachments_field(self):
        """Test parsing model metadata without attachments field (should default to empty)."""
        data = {
            "name": "Model without Attachments",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx"
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Model without Attachments"
                assert len(result.attachments) == 0

            finally:
                Path(f.name).unlink()

    def test_round_trip_model_metadata_with_attachments(self):
        """Test that encoding and decoding a MLModelMetadata with attachments works correctly."""
        original = MLModelMetadata(
            name="Round Trip Model with Attachments",
            version="2.0.0",
            authors=[
                MLModelAuthor(
                    name="John Doe", email="john@example.com", contribution=None
                )
            ],
            maintainers=[
                MLModelMaintainer(
                    name="Maintainer One", email="maintainer1@example.com"
                )
            ],
            target="round-trip-target",
            exchange_format="onnx",
            model_file="model.onnx",
            attachments=[
                AttachmentMetadata(
                    filename="README.md",
                    sha256_hash="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                    mimetype="text/markdown",
                    description="Model documentation"
                ),
                AttachmentMetadata(
                    filename="license.txt",
                    sha256_hash="def456abc1237890def456abc1237890def456abc1237890def456abc1237890",
                    mimetype="text/plain"
                )
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original, temp_path)

            # Decode from file
            result = from_file(temp_path)

            # Assertions
            assert result.name == original.name
            assert result.version == original.version
            assert result.target == original.target
            assert result.exchange_format == original.exchange_format
            assert result.model_file == original.model_file

            # Check attachments
            assert len(result.attachments) == len(original.attachments)
            for i in range(len(result.attachments)):
                assert isinstance(result.attachments[i], AttachmentMetadata)
                assert result.attachments[i].filename == original.attachments[i].filename
                assert result.attachments[i].sha256_hash == original.attachments[i].sha256_hash
                assert result.attachments[i].mimetype == original.attachments[i].mimetype
                assert result.attachments[i].description == original.attachments[i].description

            assert len(result.authors) == len(original.authors)
            for i in range(len(result.authors)):
                assert result.authors[i].name == original.authors[i].name
                assert result.authors[i].email == original.authors[i].email

            assert len(result.maintainers) == len(original.maintainers)
            for i in range(len(result.maintainers)):
                assert result.maintainers[i].name == original.maintainers[i].name
                assert result.maintainers[i].email == original.maintainers[i].email

        finally:
            temp_path.unlink()

    def test_model_metadata_encoding_with_attachments(self):
        """Test encoding model metadata with attachments to JSON."""
        original = MLModelMetadata(
            name="Encoding Test Model",
            version="1.0.0",
            authors=[
                MLModelAuthor(name="John Doe", email="john@example.com")
            ],
            maintainers=[
                MLModelMaintainer(name="Maintainer One", email="maintainer1@example.com")
            ],
            target="encoding-test",
            exchange_format="onnx",
            model_file="model.onnx",
            attachments=[
                AttachmentMetadata(
                    filename="config.yaml",
                    sha256_hash="xyz789uvw456abc123def4567890abc123def4567890abc123def4567890abc",
                    mimetype="application/x-yaml",
                    description="Model configuration"
                )
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original, temp_path)

            # Read the JSON content
            with open(temp_path) as f:
                data = json.load(f)

            # Verify the structure
            assert data["name"] == "Encoding Test Model"
            assert data["version"] == "1.0.0"
            assert data["target"] == "encoding-test"
            assert data["format"] == "onnx"
            assert data["model_file"] == "model.onnx"

            # Verify attachments
            assert "attachments" in data
            assert len(data["attachments"]) == 1
            attachment = data["attachments"][0]
            assert attachment["filename"] == "config.yaml"
            assert attachment["sha256_hash"] == "xyz789uvw456abc123def4567890abc123def4567890abc123def4567890abc"
            assert attachment["mimetype"] == "application/x-yaml"
            assert attachment["description"] == "Model configuration"

        finally:
            temp_path.unlink()

    def test_model_metadata_with_invalid_attachment_data(self):
        """Test parsing model metadata with invalid attachment data."""
        data = {
            "name": "Model with Invalid Attachments",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx",
            "attachments": [
                {
                    # Missing required fields
                    "filename": "invalid.txt"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                # Should raise KeyError due to missing required fields
                with pytest.raises(KeyError):
                    from_file(Path(f.name))

            finally:
                Path(f.name).unlink()

    def test_model_metadata_with_attachments_having_none_mimetype(self):
        """Test parsing model metadata with attachments that have mimetype=None."""
        data = {
            "name": "Model with None Mimetype Attachments",
            "version": "1.0.0",
            "authors": [{"name": "John Doe", "email": "john@example.com"}],
            "maintainers": [
                {"name": "Maintainer One", "email": "maintainer1@example.com"}
            ],
            "target": "test-target",
            "format": "onnx",
            "model_file": "model.onnx",
            "attachments": [
                {
                    "filename": "unknown_type.xyz",
                    "sha256_hash": "abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                    "description": "File with unknown mimetype"
                },
                {
                    "filename": "another_unknown.tmp",
                    "sha256_hash": "def456abc1237890def456abc1237890def456abc1237890def456abc1237890"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = from_file(Path(f.name))
                assert isinstance(result, MLModelMetadata)
                assert result.name == "Model with None Mimetype Attachments"
                assert len(result.attachments) == 2

                # Check first attachment (with description, no mimetype)
                assert isinstance(result.attachments[0], AttachmentMetadata)
                assert result.attachments[0].filename == "unknown_type.xyz"
                assert result.attachments[0].sha256_hash == "abc123def4567890abc123def4567890abc123def4567890abc123def4567890"
                assert result.attachments[0].mimetype is None
                assert result.attachments[0].description == "File with unknown mimetype"

                # Check second attachment (no mimetype, no description)
                assert isinstance(result.attachments[1], AttachmentMetadata)
                assert result.attachments[1].filename == "another_unknown.tmp"
                assert result.attachments[1].sha256_hash == "def456abc1237890def456abc1237890def456abc1237890def456abc1237890"
                assert result.attachments[1].mimetype is None
                assert result.attachments[1].description is None

            finally:
                Path(f.name).unlink()

    def test_round_trip_model_metadata_with_none_mimetype_attachments(self):
        """Test round-trip encoding/decoding of model metadata with attachments having mimetype=None."""
        original = MLModelMetadata(
            name="Round Trip Model with None Mimetype Attachments",
            version="1.0.0",
            authors=[
                MLModelAuthor(name="John Doe", email="john@example.com")
            ],
            maintainers=[
                MLModelMaintainer(name="Maintainer One", email="maintainer1@example.com")
            ],
            target="round-trip-target",
            exchange_format="onnx",
            model_file="model.onnx",
            attachments=[
                AttachmentMetadata(
                    filename="unknown_type.xyz",
                    sha256_hash="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                    mimetype=None,
                    description="File with unknown mimetype"
                ),
                AttachmentMetadata(
                    filename="minimal_file.tmp",
                    sha256_hash="def456abc1237890def456abc1237890def456abc1237890def456abc1237890",
                    mimetype=None
                )
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Encode to file
            to_file(original, temp_path)

            # Read the JSON content to verify encoding
            with open(temp_path) as f:
                json_data = json.load(f)

            # Verify that attachments don't have mimetype field in JSON when it's None
            assert "attachments" in json_data
            assert len(json_data["attachments"]) == 2
            assert "mimetype" not in json_data["attachments"][0]
            assert "mimetype" not in json_data["attachments"][1]
            assert json_data["attachments"][0]["description"] == "File with unknown mimetype"
            assert "description" not in json_data["attachments"][1]

            # Decode from file
            result = from_file(temp_path)

            # Verify round-trip integrity
            assert result.name == original.name
            assert len(result.attachments) == len(original.attachments)

            for i in range(len(result.attachments)):
                assert isinstance(result.attachments[i], AttachmentMetadata)
                assert result.attachments[i].filename == original.attachments[i].filename
                assert result.attachments[i].sha256_hash == original.attachments[i].sha256_hash
                assert result.attachments[i].mimetype is None  # Should remain None
                assert result.attachments[i].description == original.attachments[i].description

        finally:
            temp_path.unlink()
