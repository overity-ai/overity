"""
Unit tests for model_package_v1/package.py
"""

import pytest
import tempfile
import tarfile
import json
import os
from pathlib import Path
from overity.exchange.model_package_v1.package import (
    package_archive_create,
    metadata_load,
    model_load,
)
from overity.exchange import integrity
from overity.model.ml_model.metadata import (
    MLModelMetadata,
    MLModelAuthor,
    MLModelMaintainer,
)
from overity.model.ml_model.package import MLModelPackage
from overity.errors import MalformedModelPackage


class TestModelPackageV1Package:
    def test_file_sha256(self):
        """Test computing SHA256 hash of a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            result = integrity.file_sha256(path).hexdigest()
            # Known SHA256 hash of "test content"
            assert (
                result
                == "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
            )
        finally:
            path.unlink()

    def test_package_archive_create(self):
        """Test creating a model package archive."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        # Create temporary example implementation directory
        with tempfile.TemporaryDirectory() as example_dir:
            example_path = Path(example_dir)

            # Create a test file in the example directory
            example_file = example_path / "example.py"
            example_file.write_text("# Example implementation\nprint('Hello World')")

            try:
                # Create metadata
                metadata = MLModelMetadata(
                    name="Test Model",
                    version="1.0.0",
                    authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                    maintainers=[
                        MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                    ],
                    target="test-target",
                    exchange_format="onnx",
                    model_file="model.onnx",
                    derives="base-model",
                )

                # Create package info
                package_info = MLModelPackage(
                    metadata=metadata,
                    model_file_path=model_file_path,
                    example_implementation_path=example_path,
                )

                # Create output path
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    output_path = Path(f.name)

                try:
                    # Create package archive
                    sha256_hash = package_archive_create(
                        package_info, output_path
                    ).hexdigest()

                    # Verify the archive was created
                    assert output_path.exists()
                    assert isinstance(sha256_hash, str)

                    # Verify the archive contents
                    with tarfile.open(output_path, "r:gz") as archive:
                        members = archive.getnames()
                        assert "model-metadata.json" in members
                        assert "model.onnx" in members
                        assert "inference-example" in members

                        # Verify metadata content
                        metadata_file = archive.getmember("model-metadata.json")
                        f = archive.extractfile(metadata_file)
                        assert f is not None
                        with f:
                            metadata_content = json.load(f)
                            assert metadata_content["name"] == "Test Model"

                        # Verify model file content
                        model_file = archive.getmember("model.onnx")
                        f = archive.extractfile(model_file)
                        assert f is not None
                        with f:
                            model_content = f.read()
                            assert model_content == b"fake model content"

                        # Verify example implementation content
                        example_member = archive.getmember("inference-example")
                        assert example_member.isdir()
                finally:
                    output_path.unlink()
            finally:
                model_file_path.unlink()

    def test_package_archive_create_without_example(self):
        """Test creating a model package archive without example implementation."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        try:
            # Create metadata
            metadata = MLModelMetadata(
                name="Test Model",
                version="1.0.0",
                authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                target="test-target",
                exchange_format="onnx",
                model_file="model.onnx",
            )

            # Create package info without example implementation
            package_info = MLModelPackage(
                metadata=metadata,
                model_file_path=model_file_path,
                example_implementation_path=None,
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            try:
                # Create package archive
                sha256_hash = package_archive_create(
                    package_info, output_path
                ).hexdigest()

                # Verify the archive was created
                assert output_path.exists()
                assert isinstance(sha256_hash, str)

                # Verify the archive contents (no example implementation)
                with tarfile.open(output_path, "r:gz") as archive:
                    members = archive.getnames()
                    assert "model-metadata.json" in members
                    assert "model.onnx" in members
                    assert "inference-example" not in members

                    # Verify metadata content
                    metadata_file = archive.getmember("model-metadata.json")
                    f = archive.extractfile(metadata_file)
                    assert f is not None
                    with f:
                        metadata_content = json.load(f)
                        assert metadata_content["name"] == "Test Model"

                    # Verify model file content
                    model_file = archive.getmember("model.onnx")
                    f = archive.extractfile(model_file)
                    assert f is not None
                    with f:
                        model_content = f.read()
                        assert model_content == b"fake model content"
            finally:
                output_path.unlink()
        finally:
            model_file_path.unlink()

    def test_metadata_load(self):
        """Test loading metadata from a model package."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        # Create temporary example implementation directory
        with tempfile.TemporaryDirectory() as example_dir:
            example_path = Path(example_dir)

            # Create a test file in the example directory
            example_file = example_path / "example.py"
            example_file.write_text("# Example implementation")

            try:
                # Create metadata
                metadata = MLModelMetadata(
                    name="Test Model",
                    version="1.0.0",
                    authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                    maintainers=[
                        MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                    ],
                    target="test-target",
                    exchange_format="onnx",
                    model_file="model.onnx",
                    derives="base-model",
                )

                # Create package info
                package_info = MLModelPackage(
                    metadata=metadata,
                    model_file_path=model_file_path,
                    example_implementation_path=example_path,
                )

                # Create output path
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    output_path = Path(f.name)

                try:
                    # Create package archive
                    package_archive_create(package_info, output_path)

                    # Load metadata
                    loaded_metadata = metadata_load(output_path)

                    # Verify metadata
                    assert isinstance(loaded_metadata, MLModelMetadata)
                    assert loaded_metadata.name == "Test Model"
                    assert loaded_metadata.version == "1.0.0"
                    assert loaded_metadata.target == "test-target"
                    assert loaded_metadata.exchange_format == "onnx"
                    assert loaded_metadata.model_file == "model.onnx"
                    assert loaded_metadata.derives == "base-model"
                    assert len(loaded_metadata.authors) == 1
                    assert loaded_metadata.authors[0].name == "John Doe"
                    assert loaded_metadata.authors[0].email == "john@example.com"
                    assert len(loaded_metadata.maintainers) == 1
                    assert loaded_metadata.maintainers[0].name == "Jane Smith"
                    assert loaded_metadata.maintainers[0].email == "jane@example.com"
                finally:
                    output_path.unlink()
            finally:
                model_file_path.unlink()

    def test_model_load(self):
        """Test loading a model package and extracting its model file."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        # Create temporary example implementation directory
        with tempfile.TemporaryDirectory() as example_dir:
            example_path = Path(example_dir)

            # Create a test file in the example directory
            example_file = example_path / "example.py"
            example_file.write_text("# Example implementation")

            try:
                # Create metadata
                metadata = MLModelMetadata(
                    name="Test Model",
                    version="1.0.0",
                    authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                    maintainers=[
                        MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                    ],
                    target="test-target",
                    exchange_format="onnx",
                    model_file="model.onnx",
                )

                # Create package info
                package_info = MLModelPackage(
                    metadata=metadata,
                    model_file_path=model_file_path,
                    example_implementation_path=example_path,
                )

                # Create output path
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    output_path = Path(f.name)

                # Create extraction target directory
                with tempfile.TemporaryDirectory() as target_dir:
                    target_path = Path(target_dir)

                    try:
                        # Create package archive
                        package_archive_create(package_info, output_path)

                        # Load model
                        loaded_metadata = model_load(output_path, target_path)

                        # Verify metadata
                        assert isinstance(loaded_metadata, MLModelMetadata)
                        assert loaded_metadata.name == "Test Model"

                        # Verify model file was extracted
                        extracted_file = target_path / "model.onnx"
                        assert extracted_file.exists()
                        assert extracted_file.read_bytes() == b"fake model content"
                    finally:
                        output_path.unlink()
            finally:
                model_file_path.unlink()

    def test_metadata_load_malformed_package(self):
        """Test that loading metadata fails for a malformed package."""
        # Create a temporary archive without metadata
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)

        try:
            # Create an empty archive
            with tarfile.open(archive_path, "w:gz") as archive:
                pass

            # Try to load metadata - should fail
            with pytest.raises(MalformedModelPackage):
                metadata_load(archive_path)
        finally:
            archive_path.unlink()

    def test_model_load_malformed_package(self):
        """Test that loading model fails for a malformed package."""
        # Create a temporary archive without metadata
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)

        # Create target directory
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = Path(target_dir)

            try:
                # Create an empty archive
                with tarfile.open(archive_path, "w:gz") as archive:
                    pass

                # Try to load model - should fail
                with pytest.raises(MalformedModelPackage):
                    model_load(archive_path, target_path)
            finally:
                archive_path.unlink()

    def test_model_load_missing_model_file(self):
        """Test that loading model fails when model file is missing from archive."""
        # Create a temporary archive with metadata but without the expected model file
        metadata = MLModelMetadata(
            name="Test Model",
            version="1.0.0",
            authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
            maintainers=[
                MLModelMaintainer(name="Jane Smith", email="jane@example.com")
            ],
            target="test-target",
            exchange_format="onnx",
            model_file="missing_model.onnx",  # This file won't be in the archive
        )

        # Create temporary metadata file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_meta:
            meta_path = Path(tmp_meta.name)

        # Write metadata to temporary file
        from overity.exchange.model_package_v1.metadata import to_file

        to_file(metadata, meta_path)

        # Create archive file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)

        # Create target directory
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = Path(target_dir)

            try:
                # Create archive with only metadata
                with tarfile.open(archive_path, "w:gz") as archive:
                    archive.add(meta_path, arcname="model-metadata.json")

                # Try to load model - should fail because model file is missing
                with pytest.raises(MalformedModelPackage):
                    model_load(archive_path, target_path)
            finally:
                archive_path.unlink()
                meta_path.unlink()
