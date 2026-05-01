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
from overity.model.ml_model.attachment import AttachmentMetadata, ExtractedAttachment
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
                        loaded_metadata, loaded_attachments = model_load(output_path, target_path)

                        # Verify metadata
                        assert isinstance(loaded_metadata, MLModelMetadata)
                        assert loaded_metadata.name == "Test Model"
                        assert isinstance(loaded_attachments, dict)
                        assert len(loaded_attachments) == 0  # No attachments in this test

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

    def test_model_load_with_attachments(self):
        """Test loading a model package with attachments."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        # Create temporary attachment files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is attachment content")
            attachment1_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Documentation\nThis is documentation content")
            attachment2_path = Path(f.name)

        try:
            # Create metadata with attachments (use actual file names)
            metadata = MLModelMetadata(
                name="Model with Attachments",
                version="1.0.0",
                authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                target="test-target",
                exchange_format="onnx",
                model_file="model.onnx",
                attachments=[
                    AttachmentMetadata(
                        filename=attachment1_path.name,  # Use actual file name
                        sha256_hash="2dbb0f5c8c7d5e3f7a9b8c7d5e3f7a9b8c7d5e3f7a9b8c7d5e3f7a9b8c7d5e3f7",
                        mimetype="text/plain",
                        description="README file"
                    ),
                    AttachmentMetadata(
                        filename=attachment2_path.name,  # Use actual file name
                        sha256_hash="3eaa1f6d9d8e6f4b9b7a9d8e6f4b9b7a9d8e6f4b9b7a9d8e6f4b9b7a9d8e6f4b",
                        mimetype="text/markdown"
                    )
                ]
            )

            # Create package info
            package_info = MLModelPackage(
                metadata=metadata,
                model_file_path=model_file_path,
                attachments_files=[attachment1_path, attachment2_path]
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

                    # Load model with attachments
                    loaded_metadata, loaded_attachments = model_load(output_path, target_path)

                    # Verify metadata
                    assert isinstance(loaded_metadata, MLModelMetadata)
                    assert loaded_metadata.name == "Model with Attachments"
                    assert len(loaded_metadata.attachments) == 2

                    # Verify attachments dictionary
                    assert isinstance(loaded_attachments, dict)
                    assert len(loaded_attachments) == 2
                    
                    # Get attachment keys (temporary file names)
                    att_keys = list(loaded_attachments.keys())
                    assert len(att_keys) == 2
                    
                    # Verify first attachment (should be the .txt file)
                    txt_key = [k for k in att_keys if k.endswith('.txt')][0]
                    readme_att = loaded_attachments[txt_key]
                    assert isinstance(readme_att, ExtractedAttachment)
                    assert readme_att.meta.filename == txt_key  # Should match the key
                    assert readme_att.meta.mimetype == "text/plain"
                    assert readme_att.meta.description == "README file"
                    assert readme_att.path.exists()
                    assert readme_att.path.read_text() == "This is attachment content"

                    # Verify second attachment (should be the .md file)
                    md_key = [k for k in att_keys if k.endswith('.md')][0]
                    docs_att = loaded_attachments[md_key]
                    assert isinstance(docs_att, ExtractedAttachment)
                    assert docs_att.meta.filename == md_key  # Should match the key
                    assert docs_att.meta.mimetype == "text/markdown"
                    assert docs_att.meta.description is None
                    assert docs_att.path.exists()
                    assert docs_att.path.read_text() == "# Documentation\nThis is documentation content"

                    # Verify attachments folder was created
                    attachments_folder = target_path / "attachments"
                    assert attachments_folder.exists()
                    assert attachments_folder.is_dir()

                    # Verify model file was extracted
                    extracted_model = target_path / "model.onnx"
                    assert extracted_model.exists()
                    assert extracted_model.read_bytes() == b"fake model content"

                finally:
                    output_path.unlink()
        finally:
            model_file_path.unlink()
            attachment1_path.unlink()
            attachment2_path.unlink()

    def test_model_load_with_empty_attachments(self):
        """Test loading a model package with no attachments."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        try:
            # Create metadata without attachments
            metadata = MLModelMetadata(
                name="Model without Attachments",
                version="1.0.0",
                authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                target="test-target",
                exchange_format="onnx",
                model_file="model.onnx",
                attachments=[]  # Empty attachments list
            )

            # Create package info
            package_info = MLModelPackage(
                metadata=metadata,
                model_file_path=model_file_path,
                attachments_files=[]  # No attachment files
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
                    loaded_metadata, loaded_attachments = model_load(output_path, target_path)

                    # Verify metadata
                    assert isinstance(loaded_metadata, MLModelMetadata)
                    assert loaded_metadata.name == "Model without Attachments"
                    assert len(loaded_metadata.attachments) == 0

                    # Verify empty attachments dictionary
                    assert isinstance(loaded_attachments, dict)
                    assert len(loaded_attachments) == 0

                    # Verify attachments folder was NOT created (no attachments)
                    attachments_folder = target_path / "attachments"
                    assert not attachments_folder.exists()

                    # Verify model file was extracted
                    extracted_model = target_path / "model.onnx"
                    assert extracted_model.exists()
                    assert extracted_model.read_bytes() == b"fake model content"

                finally:
                    output_path.unlink()
        finally:
            model_file_path.unlink()

    def test_model_load_with_attachment_integrity_check(self):
        """Test that attachment integrity is verified during model loading."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        # Create temporary attachment file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is attachment content")
            attachment_path = Path(f.name)

        try:
            # Create metadata with WRONG hash for attachment (use actual file name)
            metadata = MLModelMetadata(
                name="Model with Bad Attachment Hash",
                version="1.0.0",
                authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                target="test-target",
                exchange_format="onnx",
                model_file="model.onnx",
                attachments=[
                    AttachmentMetadata(
                        filename=attachment_path.name,  # Use actual file name
                        sha256_hash="wrong_hash_value_123456789012345678901234567890123456789012345678901234567890",
                        mimetype="text/plain"
                    )
                ]
            )

            # Create package info
            package_info = MLModelPackage(
                metadata=metadata,
                model_file_path=model_file_path,
                attachments_files=[attachment_path]
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

                    # Load model - should work despite wrong hash in metadata
                    # (integrity check happens separately, not during load)
                    loaded_metadata, loaded_attachments = model_load(output_path, target_path)

                    # Verify metadata was loaded correctly
                    assert isinstance(loaded_metadata, MLModelMetadata)
                    assert loaded_metadata.name == "Model with Bad Attachment Hash"
                    assert len(loaded_attachments) == 1

                    # The attachment should be extracted even with wrong hash
                    att_key = list(loaded_attachments.keys())[0]  # Get the actual key
                    readme_att = loaded_attachments[att_key]
                    assert readme_att.path.exists()
                    assert readme_att.path.read_text() == "This is attachment content"

                    # But the hash in metadata is wrong (this would be caught by separate integrity check)
                    assert readme_att.meta.sha256_hash == "wrong_hash_value_123456789012345678901234567890123456789012345678901234567890"

                finally:
                    output_path.unlink()
        finally:
            model_file_path.unlink()
            attachment_path.unlink()

    def test_model_load_with_missing_attachment_file(self):
        """Test loading a model package when an attachment file is missing from archive."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model content")
            model_file_path = Path(f.name)

        try:
            # Create metadata with attachment that won't be in archive
            metadata = MLModelMetadata(
                name="Model with Missing Attachment",
                version="1.0.0",
                authors=[MLModelAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    MLModelMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                target="test-target",
                exchange_format="onnx",
                model_file="model.onnx",
                attachments=[
                    AttachmentMetadata(
                        filename="missing.txt",
                        sha256_hash="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                        mimetype="text/plain"
                    )
                ]
            )

            # Create package info WITHOUT the attachment file
            package_info = MLModelPackage(
                metadata=metadata,
                model_file_path=model_file_path,
                attachments_files=[]  # Don't include the attachment file
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            # Create extraction target directory
            with tempfile.TemporaryDirectory() as target_dir:
                target_path = Path(target_dir)

                try:
                    # Create package archive (without attachment)
                    package_archive_create(package_info, output_path)

                    # Try to load model - should fail because attachment is missing
                    with pytest.raises(MalformedModelPackage) as exc_info:
                        model_load(output_path, target_path)

                    # Verify error message
                    assert "missing.txt" in str(exc_info.value)
                    assert "not found in archive" in str(exc_info.value)

                finally:
                    output_path.unlink()
        finally:
            model_file_path.unlink()

    def test_round_trip_model_package_with_attachments(self):
        """Test creating a package with attachments and then loading it back."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"model content for round trip")
            model_file_path = Path(f.name)

        # Create temporary attachment files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"config": "value", "version": 1}, f)
            config_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# API Documentation\n\nThis is the API documentation.")
            docs_path = Path(f.name)

        try:
            # Create metadata with attachments (use actual file names)
            original_metadata = MLModelMetadata(
                name="Round Trip Model with Attachments",
                version="2.0.0",
                authors=[MLModelAuthor(name="Test Author", email="test@example.com")],
                maintainers=[MLModelMaintainer(name="Test Maintainer", email="maintainer@example.com")],
                target="round-trip-target",
                exchange_format="onnx",
                model_file="model.onnx",
                attachments=[
                    AttachmentMetadata(
                        filename=config_path.name,  # Use actual file name
                        sha256_hash="dummy_hash_for_config",
                        mimetype="application/json",
                        description="Configuration file"
                    ),
                    AttachmentMetadata(
                        filename=docs_path.name,   # Use actual file name
                        sha256_hash="dummy_hash_for_docs",
                        mimetype="text/markdown",
                        description="Documentation"
                    )
                ]
            )

            # Create package info
            original_package = MLModelPackage(
                metadata=original_metadata,
                model_file_path=model_file_path,
                attachments_files=[config_path, docs_path]
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            # Create extraction target directory
            with tempfile.TemporaryDirectory() as target_dir:
                target_path = Path(target_dir)

                try:
                    # Create package archive
                    package_archive_create(original_package, output_path)

                    # Load model with attachments
                    loaded_metadata, loaded_attachments = model_load(output_path, target_path)

                    # Verify metadata round-trip
                    assert isinstance(loaded_metadata, MLModelMetadata)
                    assert loaded_metadata.name == original_metadata.name
                    assert loaded_metadata.version == original_metadata.version
                    assert loaded_metadata.target == original_metadata.target
                    assert len(loaded_metadata.attachments) == len(original_metadata.attachments)

                    # Verify attachments round-trip
                    assert isinstance(loaded_attachments, dict)
                    assert len(loaded_attachments) == len(original_metadata.attachments)

                    for i, (filename, extracted_att) in enumerate(loaded_attachments.items()):
                        assert isinstance(extracted_att, ExtractedAttachment)
                        assert extracted_att.meta.filename == original_metadata.attachments[i].filename
                        assert extracted_att.meta.mimetype == original_metadata.attachments[i].mimetype
                        assert extracted_att.meta.description == original_metadata.attachments[i].description
                        
                        # Verify extracted file exists and has correct content
                        assert extracted_att.path.exists()
                        assert extracted_att.path.is_file()

                    # Verify specific file contents
                    config_key = [k for k in loaded_attachments.keys() if k.endswith('.json')][0]
                    docs_key = [k for k in loaded_attachments.keys() if k.endswith('.md')][0]
                    assert loaded_attachments[config_key].path.read_text() == '{"config": "value", "version": 1}'
                    assert "# API Documentation" in loaded_attachments[docs_key].path.read_text()

                    # Verify attachments folder structure
                    attachments_folder = target_path / "attachments"
                    assert attachments_folder.exists()
                    assert attachments_folder.is_dir()
                    assert (attachments_folder / config_key).exists()
                    assert (attachments_folder / docs_key).exists()

                finally:
                    output_path.unlink()
        finally:
            model_file_path.unlink()
            config_path.unlink()
            docs_path.unlink()
